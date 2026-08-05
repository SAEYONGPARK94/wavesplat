"""NEUTRAL A1b — provenance hunt: which aggregation produces 0.0629 / 0.2438 / 0.2667 / 0.1671?"""
import json, os, statistics as st
from collections import defaultdict

TMP = "/home/vexit/wavesplat/wavesplat/tmp"
OUT = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); OUT.append(s)

def load(p):
    return [json.loads(l) for l in open(os.path.join(TMP, p)) if l.strip()]

s1s2 = load("s1s2_2026-08-05/results.jsonl")
rx1a = load("rx1_tracker_2026-08-05/results.jsonl")
rx1b = load("rx1_tracker_2026-08-05/results2.jsonl")
conp = load("rx1_tracker_2026-08-05/CON_probe_results.jsonl")
for r in rx1a + rx1b:
    r["arm"] = r.get("cell", "?")
ALL = s1s2 + rx1a + rx1b + conp

EPS4 = ['episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_uniform.h5',
        'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5']

by_arm = defaultdict(list)
for r in ALL:
    by_arm[r["arm"]].append(r)

P("TARGETS: PREREG says oracle init=0.0629, best-deployable init=0.2438, gap=3.88x;")
P("         prize nRMSE 0.2667 -> 0.1671.")
P("")
P(f"{'arm':16s} {'metric':11s} {'mean(all runs)':>15s} {'med(all runs)':>14s} {'med-of-3 then med':>18s} {'nruns':>6s}")
for a in sorted(by_arm):
    for key in ("init_nrmse", "nRMSE"):
        vs = [float(r[key]) for r in by_arm[a] if r.get(key) is not None and r[key] == r[key]]
        if not vs:
            continue
        per = defaultdict(list)
        for r in by_arm[a]:
            if r.get(key) is not None:
                per[r["episode"]].append(float(r[key]))
        m3 = st.median([st.median(v) for v in per.values()])
        P(f"{a:16s} {key:11s} {st.mean(vs):15.4f} {st.median(vs):14.4f} {m3:18.4f} {len(vs):6d}")

P("")
P("MATCHES within 1e-3:")
targets = {"0.0629 (oracle init)": 0.0629, "0.2438 (best-deployable init)": 0.2438,
           "0.2667 (crowded nRMSE now)": 0.2667, "0.1671 (crowded nRMSE after)": 0.1671}
for lbl, tv in targets.items():
    hits = []
    for a in sorted(by_arm):
        for key in ("init_nrmse", "nRMSE"):
            vs = [float(r[key]) for r in by_arm[a] if r.get(key) is not None and r[key] == r[key]]
            if not vs:
                continue
            per = defaultdict(list)
            for r in by_arm[a]:
                if r.get(key) is not None:
                    per[r["episode"]].append(float(r[key]))
            cands = {"mean_all": st.mean(vs), "median_all": st.median(vs),
                     "med3_then_med": st.median([st.median(v) for v in per.values()]),
                     "med3_then_mean": st.mean([st.median(v) for v in per.values()]),
                     "mean_of_epmeans": st.mean([st.mean(v) for v in per.values()])}
            for cn, cv in cands.items():
                if abs(cv - tv) < 1e-3:
                    hits.append(f"{a}/{key}/{cn}={cv:.4f}")
    P(f"  {lbl:32s} -> " + (", ".join(hits) if hits else "*** NO MATCH IN ANY ARM/AGGREGATION ***"))

P("")
P("Implied ratios under each aggregation, oracle S2-T1f vs deployable S1-A0:")
for cn in ("mean_all", "med3_then_med"):
    def g(a, key):
        vs = [float(r[key]) for r in by_arm[a] if r.get(key) is not None]
        per = defaultdict(list)
        for r in by_arm[a]:
            if r.get(key) is not None:
                per[r["episode"]].append(float(r[key]))
        return st.mean(vs) if cn == "mean_all" else st.median([st.median(v) for v in per.values()])
    P(f"  [{cn}] init: {g('S1-A0','init_nrmse'):.4f} / {g('S2-T1f','init_nrmse'):.4f} = "
      f"{g('S1-A0','init_nrmse')/g('S2-T1f','init_nrmse'):.2f}x   "
      f"nRMSE: {g('S1-A0','nRMSE'):.4f} -> {g('S2-T1f','nRMSE'):.4f}")

open(os.path.join(TMP, "rx2_init_2026-08-05/NEUTRAL_A1b_provenance.txt"), "w").write("\n".join(OUT) + "\n")
