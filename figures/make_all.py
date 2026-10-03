"""Figures for the Flappy Bird study (PDF + PNG) from results/."""
import glob
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.train import make_net, CFG, TEST_BASE
from src import connectome as C
from src import game as F

R = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                     "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
BLUE, ORANGE, AQUA, RED, GRAY, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#8a8983", "#52514e"


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_circuit():
    W, z = C.load()
    role, types = z["role"], z["types"]
    order = np.lexsort((types, role))
    M = W.toarray()[order][:, order]
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.imshow(np.sign(M), cmap=matplotlib.colors.ListedColormap([RED, "#ffffff", BLUE]), vmin=-1, vmax=1,
              interpolation="nearest")
    for c in np.cumsum([np.sum(role == r) for r in range(3)])[:-1]:
        ax.axhline(c - 0.5, color=INK2, lw=0.6)
        ax.axvline(c - 0.5, color=INK2, lw=0.6)
    ax.set_xlabel("presynaptic (LPLC2/LC4 | intermediate | escape DNs)")
    ax.set_ylabel("postsynaptic")
    ax.set_title(f"Looming-escape circuit: {W.shape[0]} neurons, {W.nnz} connections\nblue = excitatory, red = inhibitory")
    save(fig, "fig1_circuit")


def fig_training():
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    for i, f in enumerate(sorted(glob.glob(os.path.join(R, "runs", "seed*", "train_log.csv")))):
        d = pd.read_csv(f).dropna(subset=["val_pipes"])
        ax.plot(d.frames / 1e6, d.val_pipes, "-o", ms=3, lw=1.5, color=BLUE, alpha=0.35 + 0.13 * i,
                label="training seeds (5)" if i == 0 else None)
    ax.axhline(F.MAX_SCORE, color=GRAY, lw=1, ls="--")
    ax.text(0.01, F.MAX_SCORE - 7, "maximum (100 pipes)", color=INK2, fontsize=8)
    ax.set_xlabel("training frames (millions)")
    ax.set_ylabel("validation pipes passed\n(mean of 32 games)")
    ax.legend(frameon=False, loc="center right")
    save(fig, "fig2_training")


def fig_trajectory():
    net = make_net(CFG)
    theta = np.load(os.path.join(R, "runs", "seed0", "theta.npy"))
    seed = TEST_BASE
    st, gaps = F.new_game(seed)
    net_res, acts, _ = net.trace(theta, seed, max_frames=700)
    ys, fl = [], []
    for a in acts:
        F.step(st, gaps, a)
        ys.append(st[F.Y])
        fl.append(a)
    t = np.arange(1, len(ys) + 1)
    fig, ax = plt.subplots(figsize=(7, 2.6))
    for k in range(0, 20):
        x0 = F.FIRST_X + k * F.SPACING
        f0 = (x0 - F.BIRD_X - F.R) / F.SPEED
        f1 = (x0 + F.PIPE_W - F.BIRD_X + F.R) / F.SPEED
        if f0 > len(ys):
            break
        ax.fill_between([f0, f1], 0, gaps[k] - F.GAP / 2, color=AQUA, alpha=0.35, lw=0)
        ax.fill_between([f0, f1], gaps[k] + F.GAP / 2, 1, color=AQUA, alpha=0.35, lw=0)
    ax.plot(t, ys, color=ORANGE, lw=1.8, label="bird height")
    fi = t[np.array(fl) == 1]
    ax.plot(fi, np.array(ys)[np.array(fl) == 1], "|", color=BLUE, ms=6, label="flap")
    ax.set_xlim(0, len(ys))
    ax.set_ylim(0, 1)
    ax.set_xlabel("frame")
    ax.set_ylabel("height")
    ax.set_title(f"First 700 frames of test game {seed} (seed-0 network); green = pipes")
    ax.legend(frameon=False, loc="lower right", ncol=2)
    save(fig, "fig3_trajectory")


def fig_lesion():
    les = pd.read_csv(os.path.join(R, "lesion.csv"))
    conds = list(dict.fromkeys(les.condition))
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    for i, c in enumerate(conds):
        v = les[les.condition == c].mean_pipes.values
        ax.bar(i, v.mean(), color=BLUE if c == "intact" else GRAY, width=0.6, edgecolor="white")
        ax.scatter(i + np.linspace(-0.15, 0.15, len(v)), v, s=10, color="#0b0b0b", zorder=3, lw=0)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(conds, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("mean pipes (200 test games)")
    ax.set_title("Silencing parts of the circuit (bars = mean over 5 trained networks, dots = networks)")
    save(fig, "fig4_lesion")


if __name__ == "__main__":
    for f in [fig_circuit, fig_training, fig_trajectory, fig_lesion]:
        try:
            f()
            print("ok", f.__name__)
        except FileNotFoundError as e:
            print("skip", f.__name__, e)
