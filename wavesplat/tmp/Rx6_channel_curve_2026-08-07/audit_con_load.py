"""CON audit for Rx-6. Independent re-derivation from results.jsonl.
Writes nothing except stdout. Does not modify any existing file.
"""
import json, math, itertools, statistics as st
from collections import defaultdict

D = "/home/vexit/wavesplat/wavesplat/tmp/Rx6_channel_curve_2026-08-07/"

def load():
    recs = []
    for line in open(D + "results.jsonl"):
        recs.append(json.loads(line))
    return recs

RECS = load()

# index: (cell, ep, seed) -> rec
IDX = {}
for r in RECS:
    IDX[(r["cell"], r["episode"], r["seed"])] = r

CELLS = sorted({r["cell"] for r in RECS})
EPS = sorted({r["episode"] for r in RECS})
SEEDS = sorted({r["seed"] for r in RECS})

NULLS = {n["episode"]: n for n in json.load(open(D + "nulls_rx6.json"))}


def med(xs):
    return st.median(xs)


def block(cell, metric):
    """episode -> median over seeds"""
    out = {}
    for ep in EPS:
        vals = [IDX[(cell, ep, s)][metric] for s in SEEDS]
        out[ep] = med(vals)
    return out


# ---- exact two-sided sign test ----
def C(n, k):
    return math.comb(n, k)


def sign_test(deltas, tol=0.0):
    """deltas: list of paired differences (a - b). Returns (n_eff, wins_pos, p_two_sided).
    'wins_pos' = count of delta > 0."""
    nz = [d for d in deltas if abs(d) > tol]
    n = len(nz)
    pos = sum(1 for d in nz if d > 0)
    if n == 0:
        return 0, 0, 1.0
    k = max(pos, n - pos)
    p = 2.0 * sum(C(n, i) for i in range(k, n + 1)) / (2.0 ** n)
    return n, pos, min(1.0, p)


def paired(cellA, cellB, metric="nRMSE", eps=None):
    eps = eps or EPS
    bA, bB = block(cellA, metric), block(cellB, metric)
    return [bA[e] - bB[e] for e in eps]


def report(cellA, cellB, metric="nRMSE", eps=None, label=None):
    eps = eps or EPS
    d = paired(cellA, cellB, metric, eps)
    n, pos, p = sign_test(d)
    return dict(label=label or f"{cellA} - {cellB}", medD=med(d), n=n,
                pos=pos, neg=n - pos, p=p, meanD=sum(d) / len(d))


def fmt(r):
    return (f"{r['label']:<34} medD={r['medD']:+.5f}  pos={r['pos']:2d}/{r['n']:2d} "
            f"(neg={r['neg']:2d})  p={r['p']:.4g}")
