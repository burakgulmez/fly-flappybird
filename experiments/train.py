"""Train the looming-escape network to play Flappy Bird with CMA-ES.

usage: python experiments/train.py --seed S [--budget FRAMES]
-> results/runs/seed{S}/{train_log.csv, theta.npy, test_games.csv, summary.json}
"""
import argparse, json, os, sys, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
import cma
import numpy as np
import pandas as pd
from src import connectome as C
from src.brain import Net

CFG = dict(rho=0.9, k_in=8, T=4, popsize=16, games_per_eval=8, sigma0=0.5, budget_frames=1_000_000,
           val_every=10, val_games=32, test_games=200, survival_bonus_per_frame=1 / 300)
TRAIN_MAX, VAL_BASE, TEST_BASE = 100_000_000, 500_000_000, 1_000_000_000


def make_net(cfg):
    W, z = C.load(cfg["rho"])
    return Net(W, z["role"], k_in=cfg["k_in"], T=cfg["T"], out_labels=z["types"])


def fitness(res, cfg):
    return (res[:, :, 0] + cfg["survival_bonus_per_frame"] * res[:, :, 1]).mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--budget", type=int, default=CFG["budget_frames"])
    a = ap.parse_args()
    cfg = dict(CFG, budget_frames=a.budget)
    out = os.path.join(ROOT, "results", "runs", f"seed{a.seed}")
    os.makedirs(out, exist_ok=True)
    net = make_net(cfg)
    cfg["n_params"] = net.D
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    rng = np.random.default_rng(a.seed)
    es = cma.CMAEvolutionStrategy(np.zeros(net.D), cfg["sigma0"], {"seed": a.seed + 1, "popsize": cfg["popsize"], "verbose": -9})
    val_seeds = np.arange(VAL_BASE, VAL_BASE + cfg["val_games"])
    best, best_theta, rows, frames, gen, t0 = -1e9, np.zeros(net.D), [], 0, 0, time.time()
    while frames < cfg["budget_frames"]:
        X = np.array(es.ask())
        res = net.evaluate(X, rng.integers(0, TRAIN_MAX, cfg["games_per_eval"]))
        f = fitness(res, cfg)
        es.tell(list(X), list(-f))
        frames += int(res[:, :, 1].sum())
        row = dict(gen=gen, frames=frames, time_s=time.time() - t0, fit_mean=f.mean(), fit_best=f.max(),
                   pipes_mean_pop=res[:, :, 0].mean())
        if gen % cfg["val_every"] == 0 or frames >= cfg["budget_frames"]:
            cand = np.stack([es.mean, X[np.argmax(f)]])
            vr = net.evaluate(cand, val_seeds)
            vf = fitness(vr, cfg)
            j = int(np.argmax(vf))
            row.update(val_pipes=vr[j, :, 0].mean())
            if vf[j] > best:
                best, best_theta = vf[j], cand[j].copy()
        rows.append(row)
        gen += 1
    train_time = time.time() - t0
    pd.DataFrame(rows).to_csv(os.path.join(out, "train_log.csv"), index=False)
    np.save(os.path.join(out, "theta.npy"), best_theta)
    seeds = np.arange(TEST_BASE, TEST_BASE + cfg["test_games"])
    r = net.evaluate(best_theta, seeds)[0]
    pd.DataFrame(dict(game_seed=seeds, pipes=r[:, 0], frames=r[:, 1], cause=r[:, 2])).to_csv(
        os.path.join(out, "test_games.csv"), index=False)
    summ = dict(seed=a.seed, n_params=net.D, generations=gen, train_frames=frames, train_time_s=train_time,
                test_mean_pipes=float(r[:, 0].mean()), test_median_pipes=float(np.median(r[:, 0])),
                test_max_pipes=float(r[:, 0].max()), test_win_rate=float((r[:, 2] == 4).mean()))
    json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
