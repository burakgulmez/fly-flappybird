import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src import game as F


def test_deterministic_gaps():
    _, g1 = F.new_game(5)
    _, g2 = F.new_game(5)
    _, g3 = F.new_game(6)
    assert np.array_equal(g1, g2) and not np.array_equal(g1, g3)
    assert g1.min() >= 0.25 and g1.max() <= 0.75


def test_never_flap_hits_floor():
    st, g = F.new_game(1)
    while st[F.DONE] == 0:
        F.step(st, g, 0)
    assert st[F.CAUSE] == 1 and st[F.SCORE] == 0


def test_always_flap_hits_ceiling():
    st, g = F.new_game(1)
    while st[F.DONE] == 0:
        F.step(st, g, 1)
    assert st[F.CAUSE] == 2


def test_gravity_and_flap():
    st, g = F.new_game(1)
    F.step(st, g, 0)
    assert st[F.VY] == pytest.approx(-F.GRAV) and st[F.Y] == pytest.approx(0.5 - F.GRAV)
    F.step(st, g, 1)
    assert st[F.VY] == F.FLAP_V


def test_pipe_collision_and_scoring():
    st, g = F.new_game(1)
    g[:] = 0.5  # all gaps centred at the start height
    st[F.Y] = 0.5
    # hold height by teleporting vy each frame; score must increase as pipes pass
    for _ in range(200):
        st[F.VY] = F.GRAV  # cancels gravity next step
        F.step(st, g, 0)
        st[F.Y] = 0.5
    assert st[F.DONE] == 0 and st[F.SCORE] >= 1
    st2, g2 = F.new_game(1)
    g2[:] = 0.2  # gap far below
    for _ in range(200):
        st2[F.VY] = F.GRAV
        F.step(st2, g2, 0)
        st2[F.Y] = 0.5
        if st2[F.DONE]:
            break
    assert st2[F.CAUSE] == 3


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_python_js_identical(tmp_path):
    if not os.path.exists(os.path.join(ROOT, "results", "runs", "seed0", "theta.npy")):
        pytest.skip("no trained run")
    from experiments.export_demo import export
    from experiments.train import make_net, CFG
    m = export(0)
    w = tmp_path / "w.json"
    w.write_text(json.dumps(m))
    seeds = list(range(2_000_000_000, 2_000_000_020))
    js = f"""const S=require({json.dumps(os.path.join(ROOT, 'demo', 'sim.js'))});
const m=JSON.parse(require('fs').readFileSync({json.dumps(str(w))}));
const net=new S.Net(m); const out=[];
for (const s of {json.dumps(seeds)}) {{ const r=S.playGame(net,s,3000); out.push(r.actions); }}
console.log(JSON.stringify(out));"""
    res = subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True)
    out = json.loads(res.stdout)
    net = make_net(CFG)
    theta = np.array(m["theta"])
    for s, ja in zip(seeds, out):
        _, pa, _ = net.trace(theta, s, max_frames=3000)
        assert list(pa) == ja[: len(pa)] and len(pa) == len(ja)
