"""NEUTRAL A1 — re-derive the '3.88x init gap' from raw artifacts.

Aggregation protocol (PREREG sec.1): per-episode median-of-3 seeds, then paired
per-episode deltas, median of those; two-sided sign test, k/n reported.
"""
import json, os, math, itertools, statistics as st
from collections import defaultdict

TMP = "/home/vexit/wavesplat/wavesplat/tmp"
OUT = []
def P(*a):
    s = " ".join(str(x) for x in a)
    print(s); OUT.append(s)

def load(path, arm_key="arm"):
    rows = []
    for line in open(os.path.join(TMP, path)):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows

def signtest(deltas):
    """two-sided sign test; returns (k_neg, n, p) where k_neg = # deltas < 0 (improvement)."""
    d = [x for x in deltas if x == x and x != 0.0]
    n = len(d)
    k = sum(1 for x in d if x < 0)
    if n == 0:
        return 0, 0, float('nan')
    # two-sided exact binomial p at q=0.5
    def C(n, r):
        return math.comb(n, r)
    kk = min(k, n - k)
    p = sum(C(n, i) for i in range(0, kk + 1)) / 2 ** n * 2
    return k, n, min(1.0, p)

# ---------------- load everything with init_nrmse ----------------
s1s2 = load("s1s2_2026-08-05/results.jsonl")
rx1a = load("rx1_tracker_2026-08-05/results.jsonl")
rx1b = load("rx1_tracker_2026-08-05/results2.jsonl")
conp = load("rx1_tracker_2026-08-05/CON_probe_results.jsonl")

for r in rx1a + rx1b:
    r["arm"] = r.get("cell", "?")

ALL = s1s2 + rx1a + rx1b + conp

# median-of-3 over seeds -> per (arm, episode)
def cellmed(rows, key):
    d = defaultdict(list)
    for r in rows:
        v = r.get(key)
        if v is None or v != v:
            continue
        d[(r["arm"], r["episode"])].append(float(v))
    return {k: st.median(v) for k, v in d.items()}, {k: len(v) for k, v in d.items()}

P("=" * 100)
P("A1 — per-arm init_nrmse and nRMSE (median-of-seeds per episode, then across 4 crowded episodes)")
P("=" * 100)

EPS4 = ['episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_uniform.h5',
        'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5']

med_init, cnt = cellmed(ALL, "init_nrmse")
med_nr, _ = cellmed(ALL, "nRMSE")
med_N, _ = cellmed(ALL, "N")

arms = sorted(set(a for a, e in med_init))
P(f"{'arm':24s} {'init_nrmse (4ep med)':>20s} {'nRMSE (4ep med)':>16s} {'N med':>7s} {'nseed':>6s}  per-ep init")
for a in arms:
    vals = [med_init.get((a, e)) for e in EPS4]
    nvals = [med_nr.get((a, e)) for e in EPS4]
    Ns = [med_N.get((a, e)) for e in EPS4]
    if any(v is None for v in vals):
        continue
    ns = cnt.get((a, EPS4[0]), 0)
    P(f"{a:24s} {st.median(vals):20.4f} {st.median([x for x in nvals if x is not None]):16.4f} "
      f"{st.median([x for x in Ns if x is not None]):7.1f} {ns:6d}  " +
      " ".join(f"{v:.4f}" for v in vals))

P("")
P("=" * 100)
P("A1a — the three candidate 'gaps'. All paired per-episode, n=4, floor p=0.125")
P("=" * 100)

def paired(a_treat, a_ctrl, key):
    ds = []
    for e in EPS4:
        t = (med_init if key == "init_nrmse" else med_nr).get((a_treat, e))
        c = (med_init if key == "init_nrmse" else med_nr).get((a_ctrl, e))
        if t is None or c is None:
            return None
        ds.append(t - c)
    return ds

PAIRS = [
    ("S2-T1f", "S1-A0", "PREREG's implied gap: oracle(k frozen, N=N_gt, 1 stage) vs SHIPPED ladder"),
    ("S2-T1",  "S1-A0", "oracle(k trainable) vs SHIPPED ladder"),
    ("S2-T1f", "S2-C1", "*** SINGLE-FACTOR: placement only (same N, same 1-stage 900-iter schedule)"),
    ("S2-T1",  "S2-C1", "placement only, k trainable (same N, same schedule)"),
    ("S2-C1",  "S1-A0", "*** SCHEDULE/N only, no oracle: v2_konly N=N_gt 1-stage vs ladder [3,8,15]"),
]
for t, c, label in PAIRS:
    for key in ("init_nrmse", "nRMSE"):
        ds = paired(t, c, key)
        if ds is None:
            P(f"  {t} - {c} [{key}] : MISSING")
            continue
        k, n, p = signtest(ds)
        tv = st.median([(med_init if key == 'init_nrmse' else med_nr)[(t, e)] for e in EPS4])
        cv = st.median([(med_init if key == 'init_nrmse' else med_nr)[(c, e)] for e in EPS4])
        ratio = cv / tv if tv else float('nan')
        P(f"  {t:8s} - {c:8s} [{key:10s}] med(t)={tv:.4f} med(c)={cv:.4f} "
          f"ratio={ratio:5.2f}x  medDelta={st.median(ds):+.4f}  k/n={k}/{n} p={p:.4f}   "
          + " ".join(f"{d:+.4f}" for d in ds))
    P(f"      ^ {label}")
    P("")

P("=" * 100)
P("A1b — is the gap uniform across episodes?  (per-episode init_nrmse, 15comp vs 30comp)")
P("=" * 100)
P(f"{'episode':40s} {'S2-T1f':>9s} {'S2-C1':>9s} {'S1-A0':>9s} {'T1f/A0 ratio':>13s} {'N(T1f)':>7s} {'N(A0)':>7s}")
for e in EPS4:
    a = med_init.get(("S2-T1f", e)); b = med_init.get(("S2-C1", e)); c = med_init.get(("S1-A0", e))
    na = med_N.get(("S2-T1f", e)); nc = med_N.get(("S1-A0", e))
    P(f"{e:40s} {a:9.4f} {b:9.4f} {c:9.4f} {c/a:13.2f} {na:7.1f} {nc:7.1f}")

P("")
P("=" * 100)
P("A1c — seed-level spread of init_nrmse (does median-of-3 hide bimodality?)")
P("=" * 100)
raw = defaultdict(list)
for r in ALL:
    v = r.get("init_nrmse")
    if v is not None and v == v:
        raw[(r["arm"], r["episode"])].append((r.get("seed"), float(v)))
for a in ["S1-A0", "S2-C1", "S2-T1", "S2-T1f"]:
    for e in EPS4:
        vs = sorted(raw.get((a, e), []))
        if not vs:
            continue
        xs = [v for _, v in vs]
        P(f"  {a:8s} {e[8:28]:22s} n={len(xs)} min={min(xs):.4f} med={st.median(xs):.4f} "
          f"max={max(xs):.4f} range={max(xs)-min(xs):.4f}  " + " ".join(f"{s}:{v:.4f}" for s, v in vs))

open(os.path.join(TMP, "rx2_init_2026-08-05/NEUTRAL_A1_gap.txt"), "w").write("\n".join(OUT) + "\n")
