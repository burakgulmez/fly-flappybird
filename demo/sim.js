// Flappy Bird engine + looming-escape rate network: line-by-line port of src/game.py and
// src/brain.py (same float64 operation order, same xorshift32), for the browser and Node.
(function (root) {
  "use strict";
  const P = {
    GRAV: 0.0020, FLAP_V: 0.0200, MAX_FALL: 0.0400, SPEED: 0.0100, SPACING: 0.55, PIPE_W: 0.10,
    GAP: 0.30, BIRD_X: 0.25, R: 0.025, FIRST_X: 1.0, MAX_SCORE: 100,
  };
  P.N_GAPS = P.MAX_SCORE + 4;
  const CAUSES = ["alive", "floor", "ceiling", "pipe", "win"];

  function rngNext(x) {
    x = (x ^ (x << 13)) >>> 0;
    x = (x ^ (x >>> 17)) >>> 0;
    x = (x ^ (x << 5)) >>> 0;
    return x >>> 0;
  }

  class Game {
    constructor(seed) {
      let x = Number((BigInt(seed) * 2654435761n + 12345n) & 0xFFFFFFFFn);
      if (x === 0) x = 1;
      for (let i = 0; i < 4; i++) x = rngNext(x);
      this.gaps = new Float64Array(P.N_GAPS);
      for (let k = 0; k < P.N_GAPS; k++) {
        x = rngNext(x);
        this.gaps[k] = 0.25 + 0.5 * ((x % 10000) / 9999.0);
      }
      this.y = 0.5; this.vy = 0.0; this.frame = 0.0; this.score = 0.0; this.done = false; this.cause = 0;
    }
    pipeX(k) { return P.FIRST_X + k * P.SPACING - this.frame * P.SPEED; }
    nextPipe() {
      let k = this.score;
      while (this.pipeX(k) + P.PIPE_W < P.BIRD_X - P.R) k++;
      return k;
    }
    step(action) {
      if (this.done) return;
      if (action === 1) this.vy = P.FLAP_V;
      else { this.vy = this.vy - P.GRAV; if (this.vy < -P.MAX_FALL) this.vy = -P.MAX_FALL; }
      this.y = this.y + this.vy;
      this.frame = this.frame + 1.0;
      const y = this.y;
      if (y - P.R < 0.0) { this.done = true; this.cause = 1; return; }
      if (y + P.R > 1.0) { this.done = true; this.cause = 2; return; }
      const k = this.score;
      for (let j = k; j < k + 3; j++) {
        const px = this.pipeX(j);
        if (P.BIRD_X + P.R > px && P.BIRD_X - P.R < px + P.PIPE_W) {
          if (y - P.R < this.gaps[j] - P.GAP / 2 || y + P.R > this.gaps[j] + P.GAP / 2) {
            this.done = true; this.cause = 3; return;
          }
        }
      }
      while (this.pipeX(this.score) + P.PIPE_W < P.BIRD_X - P.R) {
        this.score = this.score + 1.0;
        if (this.score >= P.MAX_SCORE) { this.done = true; this.cause = 4; return; }
      }
    }
    features(u) {
      const k = this.nextPipe(), y = this.y, g = this.gaps[k];
      u[0] = 2.0 * y - 1.0;
      u[1] = this.vy / P.MAX_FALL;
      u[2] = (this.pipeX(k) - P.BIRD_X) / P.SPACING;
      u[3] = (g - y) * 4.0;
      u[4] = (g + P.GAP / 2 - y) * 4.0;
      u[5] = (y - (g - P.GAP / 2)) * 4.0;
    }
  }

  class Net {
    // m: {n, indptr, indices, data, in_grp, out_grp, out_size, k_in, k_out, T, theta}
    constructor(m) {
      this.m = m; this.n = m.n;
      this.mask = new Float64Array(m.n).fill(1);
      this.reset();
    }
    reset() {
      const n = this.n;
      this.x = new Float64Array(n); this.phi = new Float64Array(n); this.rec = new Float64Array(n);
      this.drive = new Float64Array(Math.max(this.m.k_in, 1)); this.rvec = new Float64Array(Math.max(this.m.k_out, 1));
      this.z = 0;
    }
    step(u) {
      const m = this.m, th = m.theta, nf = 6, ki = m.k_in, ko = m.k_out, N = this.n;
      let p = ki * nf;
      for (let k = 0; k < ki; k++) {
        let s = th[p + k];
        for (let j = 0; j < nf; j++) s += th[k * nf + j] * u[j];
        this.drive[k] = s;
      }
      p += ki;
      const a = 0.05 + 0.95 / (1.0 + Math.exp(-th[p + ko + 1]));
      const g = 2.0 / (1.0 + Math.exp(-th[p + ko + 2]));
      const x = this.x, phi = this.phi, rec = this.rec, ip = m.indptr, ix = m.indices, wd = m.data;
      for (let t = 0; t < m.T; t++) {
        for (let i = 0; i < N; i++) {
          let s = 0.0;
          for (let q = ip[i]; q < ip[i + 1]; q++) s += wd[q] * phi[ix[q]];
          rec[i] = s;
        }
        for (let i = 0; i < N; i++) {
          let inp = 0.0;
          if (m.in_grp[i] >= 0) inp = this.drive[m.in_grp[i]];
          x[i] += a * (-x[i] + g * rec[i] + inp);
          phi[i] = Math.tanh(x[i]) * this.mask[i];
        }
      }
      this.rvec.fill(0);
      for (let i = 0; i < N; i++) if (m.out_grp[i] >= 0) this.rvec[m.out_grp[i]] += phi[i];
      let z = th[p + ko];
      for (let k = 0; k < ko; k++) z += th[p + k] * this.rvec[k] / m.out_size[k];
      this.z = z;
      return z > 0.0 ? 1 : 0;
    }
  }

  function playGame(net, seed, maxFrames) {
    const g = new Game(seed), u = new Float64Array(6), actions = [];
    net.reset();
    while (!g.done && actions.length < (maxFrames || 1e9)) {
      g.features(u);
      const a = net.step(u);
      actions.push(a);
      g.step(a);
    }
    return { actions, score: g.score, frames: g.frame, cause: g.cause };
  }

  const api = { P, Game, Net, playGame, CAUSES };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.FlyFlappy = api;
})(this);
