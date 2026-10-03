"""Hand-written rule and random flapping on the 200 test games -> results/reference_agents.csv"""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from experiments.train import make_net, CFG, TEST_BASE

net = make_net(CFG)
seeds = np.arange(TEST_BASE, TEST_BASE + CFG["test_games"])
rows = []
for kind, name in [(1, "heuristic"), (2, "random_10pct_flap")]:
    r = net.evaluate(np.zeros(net.D), seeds, kind)[0]
    rows.append(dict(agent=name, mean_pipes=r[:, 0].mean(), median_pipes=float(np.median(r[:, 0])), win_rate=(r[:, 2] == 4).mean()))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(ROOT, "results", "reference_agents.csv"), index=False)
print(df.to_string())
