"""Looming-escape subcircuit from FlyWire FAFB v783 (tables in data/raw)."""
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
RAW = os.path.join(REPO, "data", "raw")
# pair-aggregated graph, built from data/raw on first use
GRAPH_CACHE = os.path.join(REPO, "data", "processed", "fafb783_graph.npz")
OUT = os.path.join(HERE, "..", "data", "loom_escape.npz")

INPUT_TYPES = ["LPLC2", "LC4"]
OUTPUT_TYPES = ["DNp01", "DNp02", "DNp04", "DNp06", "DNp11"]
MID_THRESHOLD = 40  # synapses from all inputs AND to all outputs
EDGE_THRESHOLD = 5
SIGN = {"ACH": 1.0, "GABA": -1.0, "GLUT": -1.0, "DA": 1.0, "SER": 1.0, "OCT": 1.0}


def build_graph():
    """Pair-aggregated FlyWire graph (synapse counts summed over neuropils) -> GRAPH_CACHE."""
    con = pd.read_csv(os.path.join(RAW, "connections_princeton.csv.gz"))
    ids = np.sort(pd.read_csv(os.path.join(RAW, "classification.csv.gz")).root_id.values)
    pre_nt = con.groupby("pre_root_id").nt_type.first()  # one predicted transmitter per presynaptic neuron
    pairs = con.groupby(["pre_root_id", "post_root_id"], sort=False).syn_count.sum().reset_index()
    pre = np.searchsorted(ids, pairs.pre_root_id.values)
    post = np.searchsorted(ids, pairs.post_root_id.values)
    nt = pd.Series(ids).map(pre_nt).fillna("").values.astype("U4")
    os.makedirs(os.path.dirname(GRAPH_CACHE), exist_ok=True)
    np.savez_compressed(GRAPH_CACHE, root_ids=ids, pre=pre.astype(np.int64), post=post.astype(np.int64),
                        syn=pairs.syn_count.values.astype(np.int64), nt=nt)


def build():
    if not os.path.exists(GRAPH_CACHE):
        build_graph()
    z = np.load(GRAPH_CACHE)
    ids, pre, post, syn, nt = z["root_ids"], z["pre"], z["post"], z["syn"], z["nt"]
    types = pd.read_csv(os.path.join(RAW, "consolidated_cell_types.csv.gz"))
    pt = pd.Series(ids).map(types.set_index("root_id").primary_type).fillna("").values
    N = len(ids)
    A = sp.csr_matrix((syn, (pre, post)), shape=(N, N))
    I = np.where(np.isin(pt, INPUT_TYPES))[0]
    O = np.where(np.isin(pt, OUTPUT_TYPES))[0]
    from_i = np.asarray(A[I].sum(0)).ravel()
    to_o = np.asarray(A[:, O].sum(1)).ravel()
    M = np.setdiff1d(np.where((from_i >= MID_THRESHOLD) & (to_o >= MID_THRESHOLD))[0], np.union1d(I, O))
    nodes = np.concatenate([I, M, O])  # order: inputs, intermediates, outputs
    pos = -np.ones(N, np.int64)
    pos[nodes] = np.arange(len(nodes))
    m = (syn >= EDGE_THRESHOLD) & (pos[pre] >= 0) & (pos[post] >= 0)
    w = np.array([SIGN[x] for x in nt[pre[m]]]) * np.log1p(syn[m])
    n = len(nodes)
    W = sp.csr_matrix((w, (pos[post[m]], pos[pre[m]])), shape=(n, n))
    W.sum_duplicates()
    W.sort_indices()
    rho = float(np.abs(np.linalg.eigvals(W.toarray())).max())
    role = np.array([0] * len(I) + [1] * len(M) + [2] * len(O))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, indptr=W.indptr, indices=W.indices, data=W.data, n=n, role=role,
                        root_ids=ids[nodes], types=pt[nodes].astype("U32"), rho_raw=rho)
    return dict(neurons=n, inputs=len(I), intermediates=len(M), outputs=len(O), edges=int(W.nnz),
                frac_excitatory_edges=float((W.data > 0).mean()), spectral_radius_raw=rho,
                nt_edges={k: int(v) for k, v in pd.Series(nt[pre[m]]).value_counts().items()},
                input_types={t: int((pt[I] == t).sum()) for t in INPUT_TYPES},
                output_types={t: int((pt[O] == t).sum()) for t in OUTPUT_TYPES},
                intermediate_top_types={k: int(v) for k, v in pd.Series(pt[M]).value_counts().head(10).items()})


def load(rho=0.9):
    z = np.load(OUT)
    n = int(z["n"])
    W = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=(n, n)) * (rho / float(z["rho_raw"]))
    return W.tocsr(), z
