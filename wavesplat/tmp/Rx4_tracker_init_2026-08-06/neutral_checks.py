#!/usr/bin/env python3
"""NEUTRAL: (a) independently recompute the nulls that drive episode exclusion
straight from the HDF5, (b) sign-test power / MDE, (c) tie-rule sensitivity."""
import json, math, os, sys, statistics as st
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, REPO)
DATA = os.path.join(REPO, 'data_finemesh_2026-07-23')

# ---------- (a) null recompute, my own code path ----------
from wavesplat.data_loader import EpisodeDataLoader  # noqa: E402

CROWDED = [f"episode_{b}_{c}comp_seed{s}_uniform.h5"
           for b in ("BF4", "BF5") for c in (15, 30) for s in range(4)]
FULL = {d["episode"]: d for d in json.load(open(os.path.join(HERE, "nulls_full.json")))}

print("== (a) NEUTRAL recompute of the flat-sea null, all 16 crowded episodes ==")
worst = 0.0
mine = {}
for ep in CROWDED:
    L = EpisodeDataLoader(os.path.join(DATA, ep), device="cpu")
    Hs = float(L.Hs_true)
    tf = L.get_tracking_frames(tp_factor=2.0)
    # flat-sea (z=0) predictor: RMSE of GT itself, normalised by Hs, averaged over frames
    v = float(np.mean([float(np.sqrt(np.mean(f.gt_field.cpu().numpy() ** 2)) / max(Hs, 1e-6))
                       for f in tf]))
    mine[ep] = v
    d = abs(v - FULL[ep]["zero_null_tracking_nRMSE_mean"])
    worst = max(worst, d)
    print(f"  {ep[8:-3]:30s} mine={v:.6f}  file={FULL[ep]['zero_null_tracking_nRMSE_mean']:.6f}"
          f"  |diff|={d:.2e}  n_track_frames={len(tf)}")
print(f"  worst |diff| = {worst:.3e}  -> {'PASS' if worst == 0.0 else 'CHECK'}")

# ---------- (b) power / MDE for the paired sign test ----------
def cdf_ge(k, n):
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2.0 ** n

def two_sided(k, n):
    lo = sum(math.comb(n, i) for i in range(0, k + 1)) / 2.0 ** n
    return min(1.0, 2.0 * min(lo, cdf_ge(k, n)))

def crit(n, a=0.05):
    for k in range(math.ceil(n / 2), n + 1):
        if two_sided(k, n) <= a:
            return k
    return None

def power(n, pwin, a=0.05):
    k = crit(n, a)
    if k is None:
        return 0.0
    return sum(math.comb(n, i) * pwin ** i * (1 - pwin) ** (n - i) for i in range(k, n + 1))

print()
print("== (b) SIGN-TEST POWER (two-sided a=0.05) ==")
for n in (14, 16):
    k = crit(n)
    print(f"  n={n}: critical k={k}/{n}  (p={two_sided(k,n):.4f}); k={k-1} -> p={two_sided(k-1,n):.4f}")
    for pw in (0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
        print(f"     P(win)={pw:.2f} -> power={power(n, pw):.3f}")
    # MDE expressed as the win-probability giving 80% power
    lo, hi = 0.5, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if power(n, mid) < 0.80:
            lo = mid
        else:
            hi = mid
    print(f"     MDE (80% power) = P(win) >= {hi:.3f}")

# observed win rates on block C-kept
RES = [json.loads(l) for l in open(os.path.join(HERE, "results.jsonl"))]
MED = {}
slots = {}
for r in RES:
    slots.setdefault((r["cell"], r["episode"]), []).append(r["nRMSE"])
for k, v in slots.items():
    MED[k] = st.median(v)
EXCL = [e for e in CROWDED if MED[("AUTO", e)] >= FULL[e]["zero_null_tracking_nRMSE_mean"]]
KEEP = [e for e in CROWDED if e not in EXCL]
print()
print("== observed win rate vs the MDE, block C-kept (n=%d) ==" % len(KEEP))
for c in ("AUTO-NOTRACK", "T-PHIONLY", "T-WONLY", "T-EARLY20", "T-K4REF"):
    d = [MED[(c, e)] - MED[("AUTO", e)] for e in KEEP]
    w = sum(1 for x in d if x < 0)
    print(f"  {c:14s} wins={w}/{len(d)} (rate {w/len(d):.3f})  medD={st.median(d):+.5f}  "
          f"power_if_true_rate={power(len(d), max(w/len(d), 1e-9)):.3f}")

# ---------- (c) tie-rule sensitivity ----------
print()
print("== (c) TIE-RULE SENSITIVITY, block C-kept ==")
print("  PREREG rule (ties stay in n, count as losses) vs ties-dropped:")
CELLS = ["AUTO-DUP", "AUTO-NOTRACK", "T-PHIONLY", "T-WONLY", "T-EARLY20", "T-K4REF"]
for ref in ("AUTO", "AUTO-NOTRACK"):
    print(f"  --- REF = {ref} ---")
    for c in ["AUTO"] + CELLS:
        if c == ref:
            continue
        d = [MED[(c, e)] - MED[(ref, e)] for e in KEEP]
        w = sum(1 for x in d if x < 0)
        t = sum(1 for x in d if x == 0.0)
        n = len(d)
        p1 = two_sided(w, n)
        p2 = two_sided(w, n - t) if n - t > 0 else float("nan")
        flag = "  <== TIE-DRIVEN" if t > 0 else ""
        print(f"    {c:14s} ties={t:2d}  prereg {w}/{n} p={p1:.4f} | tiesdropped {w}/{n-t} "
              f"p={p2:.4f}{flag}")

# how small is the smallest non-zero |delta|? (are ties "real" or float noise?)
print()
print("  smallest non-zero |Delta| per cell (vs AUTO), block C-kept:")
for c in CELLS:
    d = [abs(MED[(c, e)] - MED[("AUTO", e)]) for e in KEEP]
    nz = sorted(x for x in d if x > 0)
    print(f"    {c:14s} n_nonzero={len(nz)}/{len(d)}  min|D|={nz[0]:.2e}" if nz
          else f"    {c:14s} ALL ZERO")
