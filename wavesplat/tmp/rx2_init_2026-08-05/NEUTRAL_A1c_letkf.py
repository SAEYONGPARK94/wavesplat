"""NEUTRAL A1c — re-derive the LETKF comparator that motivates the Rx-2 prize, and put it
next to the flat-sea null.  Raw: tmp/rerun_2026-07-25/batchA_baselines.jsonl (post-phase-fix
finemesh baselines), tmp/s1s2_2026-08-05/results.jsonl.
"""
import json, os, math, statistics as st
from collections import defaultdict

TMP = "/home/vexit/wavesplat/wavesplat/tmp"
HERE = os.path.join(TMP, "rx2_init_2026-08-05")
OUT = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); OUT.append(s)

EPS4 = ['episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_uniform.h5',
        'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5']

base = [json.loads(l) for l in open(os.path.join(TMP, "rerun_2026-07-25/batchA_baselines.jsonl")) if l.strip()]
BASE = defaultdict(dict)
nseed = defaultdict(int)
for r in base:
    if r["episode"] in EPS4:
        BASE[r["method"]][r["episode"]] = float(r["nRMSE"])
        nseed[(r["method"], r["episode"])] += 1

s1s2 = [json.loads(l) for l in open(os.path.join(TMP, "s1s2_2026-08-05/results.jsonl")) if l.strip()]
cell = defaultdict(dict)
for r in s1s2:
    cell[(r["arm"], r["episode"])][r["seed"]] = r
nulls = {r["episode"]: r for r in json.load(open(os.path.join(HERE, "NEUTRAL_A3_nulls.json")))}

def sign_p(k, n):
    kk = min(k, n - k)
    return min(1.0, sum(math.comb(n, i) for i in range(kk + 1)) / 2 ** n * 2)

P("A1c — baselines on the four Rx-2 episodes (runs per episode in the baseline file):")
for m in sorted(BASE):
    vals = [BASE[m].get(e) for e in EPS4]
    if any(v is None for v in vals):
        continue
    P(f"  {m:10s} per-ep " + " ".join(f"{v:.4f}" for v in vals)
      + f"   mean={st.mean(vals):.4f} median={st.median(vals):.4f}"
      + f"   runs/episode={nseed[(m, EPS4[0])]}")
P("")
P(f"  flat-sea null   " + " ".join(f"{nulls[e]['zero_null_tracking_nRMSE_mean']:.4f}" for e in EPS4)
  + f"   mean={st.mean([nulls[e]['zero_null_tracking_nRMSE_mean'] for e in EPS4]):.4f}")
P("")
P("  ==> PRO_analyze.py hardcodes LETKF = 0.2394; that is the MEAN of the four LETKF rows")
P(f"      (recomputed mean = {st.mean([BASE['LETKF'][e] for e in EPS4]):.4f}). Median = "
  f"{st.median([BASE['LETKF'][e] for e in EPS4]):.4f}.  The pre-registered statistic is the median.")
P("")
P("A1c2 — paired sign tests vs LETKF and vs the flat-sea null (n=4, floor p=0.125)")
P(f"{'arm':16s} {'vs':10s} {'medDelta':>9s} {'k/n':>5s} {'p':>7s}   per-episode delta")
for arm in ["S1-A0", "S1-A1", "S2-C1", "S2-T1", "S2-T1f", "X-T1f-notrack"]:
    for ref_name, ref in (("LETKF", {e: BASE["LETKF"][e] for e in EPS4}),
                          ("flat-sea", {e: nulls[e]["zero_null_tracking_nRMSE_mean"] for e in EPS4})):
        ds = []
        for e in EPS4:
            v = st.median([float(cell[(arm, e)][s]["nRMSE"]) for s in cell[(arm, e)]])
            ds.append(v - ref[e])
        k = sum(1 for x in ds if x < 0)
        P(f"{arm:16s} {ref_name:10s} {st.median(ds):+9.4f} {k}/4 {sign_p(k,4):7.3f}   "
          + " ".join(f"{x:+.4f}" for x in ds))
P("")
P("  NOTE: LETKF has exactly ONE run per episode in the baseline file (no seed replicates),")
P("        so every 'k/4 vs LETKF' pairs a median-of-3 WS cell against a single LETKF draw.")

open(os.path.join(HERE, "NEUTRAL_A1c_letkf.txt"), "w").write("\n".join(OUT) + "\n")
