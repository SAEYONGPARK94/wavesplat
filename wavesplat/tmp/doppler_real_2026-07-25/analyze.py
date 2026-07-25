"""Compute the pre-registered verdict + secondary analyses from doppler_real.json."""
import json
import numpy as np
from scipy import stats as ss

d = json.load(open('tmp/doppler_real_2026-07-25/doppler_real.json'))
R = sorted(d['results'], key=lambda x: (x['record'], x['win']))
recs = ['YS17', 'AA15a', 'AA14', 'AA15b', 'LJ18g']

normal = [r for r in R if not r['collapse']]
collapse = [r for r in R if r['collapse']]
print(f"# windows: {len(R)} total, {len(normal)} normal, {len(collapse)} collapse\n")

# ---------- paired table
print("=== PAIRED TABLE (per window) ===")
print(f"{'window':11s} {'U_hat (Ux,Uy)':16s} {'|U|':>5s} {'bnd':>3s} "
      f"{'nRMSE_u0':>8s} {'nRMSE_Uhat':>10s} {'d_now':>7s} {'d_lead':>7s} "
      f"{'vr_u0':>6s} {'vr_Uh':>6s}")
for r in R:
    star = '*' if r['collapse'] else ' '
    dl = r['delta_lead']
    print(f"{r['record']:5s} w{r['win']}{star}   "
          f"({r['U_hat'][0]:+.1f},{r['U_hat'][1]:+.1f})    {r['U_mag']:5.2f} "
          f"{'Y' if r['U_at_boundary'] else '.':>3s} "
          f"{r['nrmse_now_u0']:8.3f} {r['nrmse_now_uhat']:10.3f} "
          f"{r['delta_now']:+7.3f} {(f'{dl:+.3f}' if dl is not None else '   --'):>7s} "
          f"{r['vr_u0']:6.3f} {r['vr_uhat']:6.3f}")

def wtest(arr, label):
    a = np.array(arr, float)
    nz = a[a != 0]
    if len(nz) < 1:
        print(f"  {label}: all zero deltas, no test")
        return
    try:
        w, p = ss.wilcoxon(a, alternative='greater')
    except Exception as e:
        w, p = float('nan'), float('nan')
    print(f"  {label}: n={len(a)} (nonzero {len(nz)})  "
          f"mean d={a.mean():+.4f}  median d={np.median(a):+.4f}  "
          f"wins(d>0)={int((a>0).sum())}/{len(a)}  Wilcoxon(greater) W={w:.1f} p={p:.4f}")

print("\n=== PRIMARY: paired d_now, 27 NORMAL windows, one-sided Wilcoxon H1 median>0 ===")
d_now_norm = [r['delta_now'] for r in normal]
wtest(d_now_norm, "PRIMARY d_now (normal-27)")

print("\n=== SECONDARY ===")
wtest([r['delta_now'] for r in R], "ALL-30 d_now")
dl_norm = [r['delta_lead'] for r in normal if r['delta_lead'] is not None]
wtest(dl_norm, "lead-avg d (normal)")

print("\n-- per-record d_now (all 6 windows each) --")
for rec in recs:
    rr = [r for r in R if r['record'] == rec]
    dn = np.array([x['delta_now'] for x in rr])
    print(f"  {rec:6s} mean d_now {dn.mean():+.4f}  median {np.median(dn):+.4f}  "
          f"wins {int((dn>0).sum())}/{len(dn)}  |U| mean {np.mean([x['U_mag'] for x in rr]):.2f}")

print("\n-- U_hat stability & plausibility --")
allmag = np.array([r['U_mag'] for r in R])
print(f"  |U_hat| over 30: min {allmag.min():.2f} med {np.median(allmag):.2f} "
      f"max {allmag.max():.2f}  at-boundary(|U|>=1.5): {sum(r['U_at_boundary'] for r in R)}/30")
print(f"  U_hat==(0,0) exactly: {sum(r['U_mag']==0 for r in R)}/30")
for rec in recs:
    rr = [r for r in R if r['record'] == rec]
    ux = np.array([x['U_hat'][0] for x in rr]); uy = np.array([x['U_hat'][1] for x in rr])
    mags = np.array([x['U_mag'] for x in rr])
    ang = np.degrees(np.arctan2(uy, ux))
    # circular std of direction (weighted trivially)
    cs = np.sqrt(np.mean(np.cos(np.radians(ang)))**2 + np.mean(np.sin(np.radians(ang)))**2)
    circ_std = np.degrees(np.sqrt(-2*np.log(max(cs, 1e-9))))
    print(f"  {rec:6s} Ux {ux.mean():+.2f}+-{ux.std():.2f}  Uy {uy.mean():+.2f}+-{uy.std():.2f}  "
          f"|U| {mags.mean():.2f}+-{mags.std():.2f}  dir_circstd {circ_std:.0f}deg")

print("\n-- collapse windows (does Doppler fix any?) --")
for r in collapse:
    print(f"  {r['record']} w{r['win']}: nRMSE {r['nrmse_now_u0']:.3f}->{r['nrmse_now_uhat']:.3f} "
          f"(d {r['delta_now']:+.3f})  U_hat=({r['U_hat'][0]:+.1f},{r['U_hat'][1]:+.1f}) "
          f"still_collapse={r['nrmse_now_uhat']>=1.0}")

# vr improvement (obs-fit) vs nRMSE improvement (GT) correlation
dvr = np.array([r['vr_u0']-r['vr_uhat'] for r in R])
dnr = np.array([r['delta_now'] for r in R])
print(f"\n-- obs-residual drop vs GT-nRMSE drop: corr over 30 = "
      f"{np.corrcoef(dvr, dnr)[0,1]:+.3f}")
print(f"   mean vr drop {dvr.mean():+.4f} (>0 by construction where U_hat!=0)")
