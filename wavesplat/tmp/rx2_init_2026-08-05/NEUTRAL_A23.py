"""NEUTRAL A2 (MDE) + A3 (is init_nrmse the right instrument?) — all from raw artifacts."""
import json, os, math, random, statistics as st
from collections import defaultdict
import numpy as np

TMP = "/home/vexit/wavesplat/wavesplat/tmp"
HERE = os.path.join(TMP, "rx2_init_2026-08-05")
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

# operating point from cfg
def oppoint(r):
    im = (r.get("cfg") or {}).get("init_method")
    return "oracle" if im == "oracle" else "deploy" if im == "v2_konly" else str(im)

cell = defaultdict(dict)   # (arm, ep) -> seed -> row
for r in ALL:
    cell[(r["arm"], r["episode"])][r.get("seed")] = r

arms = sorted(set(a for a, e in cell))

def sign_p(k, n):
    kk = min(k, n - k)
    return min(1.0, sum(math.comb(n, i) for i in range(kk + 1)) / 2 ** n * 2)

# ==================================================================== A2 MDE
P("=" * 104)
P("A2 — MDE of the Rx-2 design (n=4 paired episodes, median-of-3 seeds/cell, two-sided sign test)")
P("=" * 104)
P("Power model: the sign test can only reach its floor p=0.125 at 4/4.  For 80% power we need")
P("  P(correct sign per episode) = 0.80^(1/4) = 0.9457  ->  z = 1.6035")
P("  MDE = 1.6035 * SD(paired median-of-3 delta).  SD is bootstrapped over the 3 seeds,")
P("  RESAMPLING THE SAME SEEDS IN BOTH ARMS so the seed pairing (and its correlation) is kept.")
P("")

rng = random.Random(20260806)
def boot_sd(armA, armB, key, nboot=4000):
    """SD of [median_over_seeds(A) - median_over_seeds(B)] per episode, seed-paired bootstrap."""
    sds = []
    for e in EPS4:
        ca, cb = cell.get((armA, e), {}), cell.get((armB, e), {})
        seeds = sorted(set(ca) & set(cb))
        if len(seeds) < 3:
            return None
        a = [float(ca[s][key]) for s in seeds]
        b = [float(cb[s][key]) for s in seeds]
        vals = []
        n = len(seeds)
        for _ in range(nboot):
            idx = [rng.randrange(n) for _ in range(n)]
            vals.append(st.median([a[i] for i in idx]) - st.median([b[i] for i in idx]))
        sds.append(st.pstdev(vals))
    return sds

Z = 1.6035
P(f"{'arm pair':30s} {'metric':11s} {'per-episode SD of paired median':>34s} {'medSD':>8s} {'MDE(80%)':>9s}")
PAIRS = [("S2-T1f", "S2-C1", "oracle"), ("S2-T1", "S2-C1", "oracle"),
         ("X-BASE", "K4", "oracle"), ("S2-T1f", "S1-A0", "cross"),
         ("S1-A0", "S1-A1", "deploy"), ("S1-A0", "S1-A2", "deploy"),
         ("S1-A0", "S2-C1", "deploy"), ("D-BASE", "D-K12", "deploy"),
         ("D-BASE", "D-NOTRACK", "deploy")]
mde_tab = {}
for a, b, op in PAIRS:
    for key in ("init_nrmse", "nRMSE"):
        sds = boot_sd(a, b, key)
        if sds is None:
            P(f"{a+' vs '+b:30s} {key:11s}  (seeds not shared / missing)")
            continue
        ms = st.median(sds)
        P(f"{a+' vs '+b:30s} {key:11s} " + " ".join(f"{s:.4f}" for s in sds)
          + f"  {ms:8.4f} {Z*ms:9.4f}   [{op}]")
        mde_tab[(a, b, key)] = (ms, Z * ms, op)

P("")
for op in ("oracle", "deploy"):
    for key in ("init_nrmse", "nRMSE"):
        vs = [v[1] for k, v in mde_tab.items() if v[2] == op and k[2] == key]
        if vs:
            P(f"  ==> MDE at {op:7s} operating point, {key:11s} = {st.median(vs):.4f}  "
              f"(range {min(vs):.4f}-{max(vs):.4f}, {len(vs)} arm pairs)")

P("")
P("A2b — the prize, and what fraction of it is detectable")
med = lambda a, k: st.median([st.median([float(cell[(a, e)][s][k]) for s in cell[(a, e)]])
                              for e in EPS4])
pairs_for_prize = [("S2-T1f", "S1-A0"), ("S2-T1", "S1-A0"), ("S2-T1f", "S2-C1"), ("S2-T1", "S2-C1")]
for t, c in pairs_for_prize:
    for key in ("init_nrmse", "nRMSE"):
        ds = []
        for e in EPS4:
            ds.append(st.median([float(cell[(t, e)][s][key]) for s in cell[(t, e)]]) -
                      st.median([float(cell[(c, e)][s][key]) for s in cell[(c, e)]]))
        d = abs(st.median(ds))
        op = "oracle" if t.startswith("S2") and c.startswith("S2") else "cross"
        mvals = [v[1] for k, v in mde_tab.items() if k[2] == key and v[2] == "deploy"]
        m = st.median(mvals) if mvals else float("nan")
        P(f"  prize |{t}-{c}| [{key:11s}] = {d:.4f} ; deploy-MDE = {m:.4f} ; "
          f"smallest detectable fraction of prize = {100*m/d:5.1f}%")

# ==================================================================== A3
P("")
P("=" * 104)
P("A3 — is init_nrmse the right instrument?")
P("=" * 104)

nulls = {r["episode"]: r for r in json.load(open(os.path.join(HERE, "NEUTRAL_A3_nulls.json")))}
P("A3a — the FLAT-SEA null (predict z=0 everywhere), measured at the exact frame run.py scores:")
for e in EPS4:
    n = nulls[e]
    P(f"  {e[8:30]:24s} zero-null init_nrmse = {n['zero_null_init_nrmse']:.4f} ; "
      f"zero-null tracking nRMSE = {n['zero_null_tracking_nRMSE_mean']:.4f}")
P("")
P("A3b — which arms BEAT the flat-sea null?  (paired per episode, n=4, floor p=0.125)")
P(f"{'arm':16s} {'metric':11s} {'arm med':>8s} {'null med':>9s} {'medDelta':>9s} {'k/n':>5s} {'p':>7s}  per-episode delta")
for a in arms:
    if any((a, e) not in cell for e in EPS4):
        continue
    for key, nk in (("init_nrmse", "zero_null_init_nrmse"), ("nRMSE", "zero_null_tracking_nRMSE_mean")):
        ds, av, nv = [], [], []
        ok = True
        for e in EPS4:
            try:
                v = st.median([float(cell[(a, e)][s][key]) for s in cell[(a, e)]])
            except Exception:
                ok = False; break
            ds.append(v - nulls[e][nk]); av.append(v); nv.append(nulls[e][nk])
        if not ok:
            continue
        k = sum(1 for x in ds if x < 0)
        P(f"{a:16s} {key:11s} {st.median(av):8.4f} {st.median(nv):9.4f} {st.median(ds):+9.4f} "
          f"{k}/4 {sign_p(k,4):7.3f}  " + " ".join(f"{x:+.4f}" for x in ds))

P("")
P("A3c — SAME init_nrmse, DIFFERENT final nRMSE (does init determine the outcome?)")
grp = defaultdict(list)
for a in arms:
    if any((a, e) not in cell for e in EPS4):
        continue
    try:
        i = med(a, "init_nrmse"); n = med(a, "nRMSE")
    except Exception:
        continue
    grp[round(i, 4)].append((a, n))
for i in sorted(grp):
    g = grp[i]
    if len(g) < 2:
        continue
    ns = [n for _, n in g]
    P(f"  init_nrmse = {i:.4f}  ({len(g)} arms)  final nRMSE {min(ns):.4f} .. {max(ns):.4f}  "
      f"spread {max(ns)-min(ns):.4f} = {max(ns)/min(ns):.2f}x   " +
      ", ".join(f"{a}:{n:.4f}" for a, n in sorted(g, key=lambda x: x[1])))

P("")
P("A3d — empirical coupling r(init_nrmse, final nRMSE), run level, episode-demeaned")
def corr(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = st.mean(xs), st.mean(ys)
    sx = math.sqrt(sum((x-mx)**2 for x in xs)); sy = math.sqrt(sum((y-my)**2 for y in ys))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((x-mx)*(y-my) for x, y in zip(xs, ys))/(sx*sy)

def slope(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    den = sum((x-mx)**2 for x in xs)
    return float("nan") if den == 0 else sum((x-mx)*(y-my) for x, y in zip(xs, ys))/den

for label, filt in (("ALL runs", lambda r: True),
                    ("oracle op point only", lambda r: oppoint(r) == "oracle"),
                    ("deploy op point only", lambda r: oppoint(r) == "deploy")):
    rows = [r for r in ALL if filt(r) and r.get("init_nrmse") is not None
            and r.get("nRMSE") is not None and r["episode"] in EPS4]
    xs = [float(r["init_nrmse"]) for r in rows]; ys = [float(r["nRMSE"]) for r in rows]
    # episode-demeaned
    epm_x = defaultdict(list); epm_y = defaultdict(list)
    for r in rows:
        epm_x[r["episode"]].append(float(r["init_nrmse"])); epm_y[r["episode"]].append(float(r["nRMSE"]))
    mx = {e: st.mean(v) for e, v in epm_x.items()}; my = {e: st.mean(v) for e, v in epm_y.items()}
    dx = [float(r["init_nrmse"]) - mx[r["episode"]] for r in rows]
    dy = [float(r["nRMSE"]) - my[r["episode"]] for r in rows]
    P(f"  {label:24s} n={len(rows):4d}  pooled r={corr(xs,ys):+.3f}  "
      f"episode-demeaned r={corr(dx,dy):+.3f}  R2={corr(dx,dy)**2:5.3f}  "
      f"slope d(nRMSE)/d(init)={slope(dx,dy):+.3f}   init range [{min(xs):.4f},{max(xs):.4f}]")

P("")
P("A3e — DELTA coupling: for every arm paired against its own operating-point base,")
P("      does an init improvement predict an nRMSE improvement?")
BASES = {"oracle": "X-BASE", "deploy": "D-BASE"}
pts = []
for a in arms:
    if any((a, e) not in cell for e in EPS4):
        continue
    rr = next(iter(cell[(a, EPS4[0])].values()))
    op = oppoint(rr)
    base = BASES.get(op)
    if base is None or base == a or any((base, e) not in cell for e in EPS4):
        continue
    di = med(a, "init_nrmse") - med(base, "init_nrmse")
    dn = med(a, "nRMSE") - med(base, "nRMSE")
    pts.append((a, op, di, dn))
for op in ("oracle", "deploy"):
    sub = [(a, di, dn) for a, o, di, dn in pts if o == op]
    if len(sub) < 3:
        P(f"  {op}: only {len(sub)} arms -- not estimable"); continue
    xs = [d for _, d, _ in sub]; ys = [n for _, _, n in sub]
    P(f"  {op:7s} n_arms={len(sub)}  r(Dinit, DnRMSE) = {corr(xs,ys):+.3f}  "
      f"slope = {slope(xs,ys):+.3f}   |Dinit| range [{min(map(abs,xs)):.4f},{max(map(abs,xs)):.4f}]")
    for a, d, n in sorted(sub, key=lambda t: t[1]):
        P(f"      {a:16s} Dinit={d:+.4f}  DnRMSE={n:+.4f}")

open(os.path.join(HERE, "NEUTRAL_A23.txt"), "w").write("\n".join(OUT) + "\n")
