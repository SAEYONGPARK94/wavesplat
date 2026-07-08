"""RMSE gap audit — 2026-07-08 (user directive: no premature conclusions).

Decomposes the real-data encounter-forecast RMSE into amplitude / phase / bias
terms (Murphy decomposition), adds trivial baselines (zero, persistence), and
bootstrap CIs, from the stored pred/gt in falcon_real_operator_*.json and
falcon_hos_probe*.json. No re-estimation — pure re-analysis of stored outputs.

  nRMSE^2 = (b)^2 + (a - rho)^2 + (1 - rho^2)
  b = bias/sigma_g, a = sigma_p/sigma_g, rho = corr
  -> phase floor = sqrt(1 - rho^2) (best achievable by rescaling alone)
  -> MMSE-optimal shrinkage is a* = rho (jointly-Gaussian case)
"""
import json
import numpy as np

RES = "/home/vexit/wavesplat/wavesplat/results/direction3"
RECORDS = ["YS17", "AA15a", "AA14", "AA15b", "LJ18g"]
BANDS = [(0, 10, "0-10s"), (10, 20, "10-20s")]


def stats(p, g):
    m = np.isfinite(p) & np.isfinite(g)
    p, g = p[m], g[m]
    if len(g) < 3:
        return None
    sg, sp = np.std(g), np.std(p)
    rho = float(np.corrcoef(p, g)[0, 1])
    b = float((np.mean(p) - np.mean(g)) / sg)
    a = float(sp / sg)
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    return dict(n=int(len(g)), rmse=rmse, nrmse=rmse / sg, corr=rho, amp=a,
                bias_n=b, sigma_g=float(sg),
                phase_floor=float(np.sqrt(max(0.0, 1 - rho ** 2))),
                shrink_pen=float((a - rho) ** 2))


def band_pool(windows, leads, W, lo, hi, mode="pred"):
    ps, gs = [], []
    for w in windows:
        gt = np.asarray(w["gt"], float)
        if mode == "pred":
            pr = np.asarray(w["by_W"][W]["pred"], float)
        elif mode == "persist":
            pr = np.full_like(gt, gt[0])
        elif mode == "zero":
            pr = np.zeros_like(gt)
        n = min(len(pr), len(gt), len(leads))
        m = (leads[:n] >= lo) & (leads[:n] <= hi)
        ps.append(pr[:n][m]); gs.append(gt[:n][m])
    return np.concatenate(ps), np.concatenate(gs)


def bootstrap_nrmse(windows, leads, W, lo, hi, n_boot=4000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(windows))
    vals = []
    for _ in range(n_boot):
        sel = rng.choice(idx, size=len(idx), replace=True)
        p, g = band_pool([windows[i] for i in sel], leads, W, lo, hi)
        m = np.isfinite(p) & np.isfinite(g)
        if m.sum() < 3 or np.std(g[m]) == 0:
            continue
        vals.append(np.sqrt(np.mean((p[m] - g[m]) ** 2)) / np.std(g[m]))
    return [float(np.percentile(vals, q)) for q in (2.5, 50, 97.5)]


out = {"records": {}, "hos": {}}
for rec in RECORDS:
    try:
        d = json.load(open(f"{RES}/falcon_real_operator_{rec}.json"))
    except FileNotFoundError:
        continue
    leads = np.asarray(d["leads"], float)
    wins = d["windows"]
    rrec = {"Hs": float(np.mean([w["Hs"] for w in wins])), "n_win": len(wins),
            "bands": {}, "per_window": []}
    for lo, hi, bl in BANDS:
        row = {}
        for W in [str(float(w)) for w in d["w_mems"]]:
            p, g = band_pool(wins, leads, W, lo, hi)
            s = stats(p, g)
            if s is None:
                continue
            pp, gg = band_pool(wins, leads, W, lo, hi, "persist")
            sp_ = stats(pp, gg)
            s["persist_nrmse"] = sp_["nrmse"] if sp_ else None
            s["ci"] = bootstrap_nrmse(wins, leads, W, lo, hi)
            row[f"W{W}"] = s
        rrec["bands"][bl] = row
    # per-window at the mid W, 0-10s band
    Wmid = str(float(d["w_mems"][1]))
    for i, w in enumerate(wins):
        p, g = band_pool([w], leads, Wmid, 0, 10)
        s = stats(p, g)
        if s:
            rrec["per_window"].append(
                dict(wi=i, t0=w["t0"], Tp=w["Tp"], spread=w.get("spread"),
                     adv_corr=w.get("adv_corr"), nrmse=s["nrmse"],
                     corr=s["corr"], amp=s["amp"]))
    out["records"][rec] = rrec

# HOS probes: same decomposition
for name, label in [("falcon_hos_probe", "hos_headline"),
                    ("falcon_hos_probe_design22kt", "hos_design22kt")]:
    d = json.load(open(f"{RES}/{name}.json"))
    leads = np.asarray(d["leads"], float)
    ent = {}
    for W, seas in d["raw"].items():
        for lo, hi, bl in ([(0, 2, "0-2s")] if "22kt" in name else []) + BANDS:
            ps, gs, pers = [], [], []
            for pred, gt in seas:
                pred, gt = np.asarray(pred, float), np.asarray(gt, float)
                n = min(len(pred), len(gt), len(leads))
                m = (leads[:n] >= lo) & (leads[:n] <= hi)
                ps.append(pred[:n][m]); gs.append(gt[:n][m])
                pers.append(np.full(m.sum(), gt[0]))
            p, g = np.concatenate(ps), np.concatenate(gs)
            s = stats(p, g)
            if s is None:
                continue
            sp_ = stats(np.concatenate(pers), g)
            s["persist_nrmse"] = sp_["nrmse"] if sp_ else None
            ent.setdefault(bl, {})[f"W{W}"] = s
    out["hos"][label] = ent

with open(f"{RES}/gap_rmse_audit.json", "w") as f:
    json.dump(out, f, indent=1)

# ---- console report ----
def fmt(s):
    return (f"nRMSE={s['nrmse']:.2f} [CI {s['ci'][0]:.2f}-{s['ci'][2]:.2f}] "
            if "ci" in s else f"nRMSE={s['nrmse']:.2f} ") + \
           (f"corr={s['corr']:.2f} amp={s['amp']:.2f} floor={s['phase_floor']:.2f} "
            f"shrinkPen={s['shrink_pen']:.3f} bias={s['bias_n']:.2f} "
            f"persist={s['persist_nrmse']:.2f} n={s['n']}")

for rec, r in out["records"].items():
    print(f"\n### {rec} (Hs {r['Hs']:.2f} m, {r['n_win']} windows)")
    for bl, row in r["bands"].items():
        for Wk, s in row.items():
            print(f"  {bl:7} {Wk:6}: {fmt(s)}")
    print("  per-window (mid W, 0-10s):")
    for w in r["per_window"]:
        print(f"    wi{w['wi']} t0={w['t0']:.0f} Tp={w['Tp']:.1f} "
              f"spread={w['spread']:.0f} adv={w['adv_corr']:.2f} | "
              f"nRMSE={w['nrmse']:.2f} corr={w['corr']:.2f} amp={w['amp']:.2f}")

for label, ent in out["hos"].items():
    print(f"\n### {label}")
    for bl, row in ent.items():
        for Wk, s in row.items():
            print(f"  {bl:7} {Wk:6}: {fmt(s)}")
