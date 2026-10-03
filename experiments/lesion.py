"""Silence groups of neurons in trained networks -> results/lesion.csv"""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from experiments.train import make_net, CFG, TEST_BASE
from src import connectome as C

_, z = C.load()
role, types = z["role"], z["types"]
net = make_net(CFG)
seeds = np.arange(TEST_BASE, TEST_BASE + 200)
conds = {"intact": np.zeros(net.n, bool), "all intermediates": role == 1, "LC4 inputs": types == "LC4",
         "LPLC2 inputs": types == "LPLC2"}
for t in sorted(set(types[role == 2])):
    conds[f"{t} output"] = types == t
rows = []
for s in range(5):
    theta = np.load(os.path.join(ROOT, "results", "runs", f"seed{s}", "theta.npy"))
    for name, silenced in conds.items():
        net.mask[:] = 1.0
        net.mask[silenced] = 0.0
        r = net.evaluate(theta, seeds)[0]
        rows.append(dict(seed=s, condition=name, n_silenced=int(silenced.sum()), mean_pipes=r[:, 0].mean(),
                         win_rate=(r[:, 2] == 4).mean()))
    net.mask[:] = 1.0
df = pd.DataFrame(rows)
df.to_csv(os.path.join(ROOT, "results", "lesion.csv"), index=False)
print(df.pivot(index="condition", columns="seed", values="mean_pipes").round(1).to_string())
