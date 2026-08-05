"""Independent verifier (fable): re-derive every load-bearing number for
claims A-H of the e29/e30 closure arc, from the raw JSONs + generator code.
CPU-only part."""
import json
import numpy as np
from scipy import stats as st

D = 'results/direction3/'


def arr(rows, k):
    return np.array([r[k] for r in rows], float)


def tstat(d):
    m = d.mean()
    se = d.std(ddof=1) / np.sqrt(len(d))
    t = m / se if se > 0 else 0.0
    p = 2 * st.t.sf(abs(t), len(d) - 1)
    return m, se, t, p, int((d > 0).sum())


print('=' * 78)
print('A. gate-0 ladder (e29_gate0_ladder.json)')
J = json.load(open(D + 'e29_gate0_ladder.json'))
R = J['rows']
cur = arr(R, 'cur')
print(f"  n = {len(R)}")
print(f"  cur mean        = {cur.mean():+.4f}   (claimed +0.4032)")
for k, claimed in [('ens_blind', -0.0118), ('ens_oracle', +0.0411),
                   ('real_free', +0.1824)]:
    m, se, t, p, npos = tstat(arr(R, k) - cur)
    print(f"  {k:10s} mean {arr(R,k).mean():+.4f}  vs cur {m:+.4f} "
          f"(claim {claimed:+.4f})  se {se:.4f} t {t:+.2f} p {p:.4f}  pos {npos}/16")
print(f"  json h_blind/h_oracle/h_real = {J['h_blind']:+.4f}/{J['h_oracle']:+.4f}/{J['h_real']:+.4f}")

print()
print('=' * 78)
print('B. conditionality (e30_conditionality.json)')
J = json.load(open(D + 'e30_conditionality.json'))
R = J['rows']
A = arr(R, 'cur_own'); B = arr(R, 'cur_mean'); C = arr(R, 'cur_noalign')
d = B - A
m, se, t, p, npos = tstat(d)
print(f"  cur_own  mean = {A.mean():+.4f} (claim +0.3923)")
print(f"  cur_mean mean = {B.mean():+.4f} (claim +0.3794)")
print(f"  cur_noalign   = {C.mean():+.4f} (claim +0.1442)")
print(f"  B-A = {m:+.4f} +- {se:.4f}  t={t:+.2f} p={p:.4f}  B>=A {int((d>=0).sum())}/16 "
      f"(claim -0.0129+-0.0242 t=-0.53 p=0.604 6/16)")
print(f"  direction read-off (B - C) = {(B - C).mean():+.4f} (claim +0.2352)")
print(f"  MDE(80% power, alpha=.05) for B-A: {2.8 * se:.4f}")
print(f"  probe2: {json.dumps(J['probe2'], indent=4)}")

print()
print('=' * 78)
print('D/E. speckle control (e30_speckle_control.json)')
J = json.load(open(D + 'e30_speckle_control.json'))
R = J['rows']
cur = arr(R, 'cur')
for k, claim in [('rl_true', 'claim -0.0003 t=-0.01 p=.991 6/16'),
                 ('perm', 'claim -0.0771 t=-2.09 3/16'),
                 ('bf_true', ''), ('rl_obs', ''), ('bf_obs', ''),
                 ('ens_blind', ''), ('real_free', '')]:
    m, se, t, p, npos = tstat(arr(R, k) - cur)
    print(f"  {k:9s} mean {arr(R,k).mean():+.4f}  vs cur {m:+.4f} se {se:.4f} "
          f"t {t:+.2f} p {p:.4f}  pos {npos}/16   {claim}")
perm = arr(R, 'perm'); ens = arr(R, 'ens_blind'); real = arr(R, 'real_free')
print(f"  perm - ens_blind = {(perm-ens).mean():+.4f}  (sharpness alone)")
print(f"  real - perm      = {(real-perm).mean():+.4f}  (speckle value)")
rl = arr(R, 'rl_true'); bf = arr(R, 'bf_true')
print(f"  rl_true - bf_true = {(rl-bf).mean():+.4f}  (deconv recovers?)")

print()
print('=' * 78)
print('F. dof and bank (e30_dof_and_bank.json)')
J = json.load(open(D + 'e30_dof_and_bank.json'))
R = J['rows']
pn = arr(R, 'p_net'); pb = arr(R, 'p_bf')
m, se, t, p, npos = tstat(pn)
lo, hi = m - st.t.ppf(0.975, 15) * se, m + st.t.ppf(0.975, 15) * se
print(f"  p_net = {m:+.4f} t {t:+.2f} p {p:.2e} pos {npos}/16 CI[{lo:+.3f},{hi:+.3f}]"
      f"  (claim +0.1681 t 7.51 p 1.85e-06 15/16 CI[+0.120,+0.216])")
m, se, t, p, npos = tstat(pb)
print(f"  p_bf  = {m:+.4f} t {t:+.2f} p {p:.2e} pos {npos}/16  (claim +0.0688)")
m, se, t, p, npos = tstat(pn - pb)
print(f"  p_net - p_bf = {m:+.4f} t {t:+.2f} p {p:.4f} pos {npos}/16 "
      f"ratio {pn.mean()/pb.mean():.2f}x  (claim +0.0994 t 4.33 p .0006 14/16 = 2.44x)")
print(f"  a_psf mean {arr(R,'a_psf').mean():.1f}  a_gt mean {arr(R,'a_gt').mean():.1f}  "
      f"n_res mean {arr(R,'n_res').mean():.2f}  n_res_band {arr(R,'n_res_band').mean():.1f}")
fine = arr(R, 'fine101'); rf = arr(R, 'real_free')
print(f"  fine101 - real_free(51) = {(fine-rf).mean():+.4f}  (bank cap test)")

print()
print('=' * 78)
print('G. exchange rate (e30_exchange_rate.json)')
J = json.load(open(D + 'e30_exchange_rate.json'))
R = J['rows']; FR = J['fracs']
cur = arr(R, 'cur')
fc = np.array([np.mean([r['f_corr'][f'{f:g}'] for r in R]) for f in FR])
fp = np.array([np.mean([r['f_p'][f'{f:g}'] for r in R]) for f in FR])
for f, c, pv in zip(FR, fc, fp):
    print(f"  f={f:4.2f}  corr {c:+.4f}  vs cur {c-cur.mean():+.4f}  p {pv:+.4f}")
pnet = arr(R, 'p_net').mean()
f_at_net = np.interp(pnet, fp, FR)
f_cross = np.interp(0.0, fc - cur.mean(), FR)
print(f"  p_net(here) = {pnet:+.4f} -> f_equiv = {f_at_net:.3f} ({f_at_net*200:.0f}/200)")
print(f"  f-ladder crosses cur at f = {f_cross:.3f} ({f_cross*200:.0f}/200)")
for tgt in (0.05, 0.10):
    fx = np.interp(tgt, fc - cur.mean(), FR)
    print(f"  +{tgt:.2f} corr needs f = {fx:.3f} ({fx*200:.0f}/200), p = {np.interp(fx, FR, fp):+.4f}")
# value of net's p at f_equiv:
val_at_net = np.interp(f_at_net, FR, fc) - cur.mean()
print(f"  field-corr value at f_equiv({f_at_net:.3f}) = {val_at_net:+.4f}  <- B/F reconciliation")
d_rob = arr(R, 'derr_rob'); d_top = arr(R, 'derr_top')
m, se, t, p, npos = tstat(arr(R, 'ens_rob') - cur)
print(f"  ens_rob vs cur = {m:+.4f} t {t:+.2f} pos {npos}/16 | derr top {d_top.mean():.1f} -> rob {d_rob.mean():.1f}")

print()
print('=' * 78)
print('C3. e32 bf prior arms (all negative?)')
J = json.load(open(D + 'e32_bf_prior.json'))
R = J['rows']
ens = arr(R, 'ens')
neg = 0
for k in [k for k in R[0] if k not in ('si', 'ens', 'real')]:
    m, se, t, p, npos = tstat(arr(R, k) - ens)
    tag = 'NEG' if m < 0 else '*** POSITIVE ***'
    if m < 0:
        neg += 1
    print(f"  {k:10s} vs ens {m:+.4f} se {se:.4f} t {t:+.2f} pos {npos}/16  {tag}")
m, se, t, p, npos = tstat(arr(R, 'real') - ens)
print(f"  real(oracle) vs ens {m:+.4f}  | negative arms: {neg}/10")

print()
print('=' * 78)
print('C1. e35 multilook admix (hand-optimal extraction)')
R = json.load(open(D + 'e35_multilook_admix.json'))
ens = arr(R, 'ens')
best = None
for k in [k for k in R[0] if k not in ('si', 'ens')]:
    m, se, t, p, npos = tstat(arr(R, k) - ens)
    print(f"  {k:10s} vs ens {m:+.4f} se {se:.4f} t {t:+.2f} pos {npos}/16")
    if best is None or m > best[1]:
        best = (k, m, se, t)
print(f"  BEST arm: {best[0]} {best[1]:+.4f} +- {best[2]:.4f} (t={best[3]:.2f})"
      f"   (claim ml2^0.5 +0.0116 +- 0.0116 t=1.00)")

print()
print('=' * 78)
print('C4. e36 bimodal')
print(open(D + 'e36_bimodal.json').read()[:1500])

print()
print('=' * 78)
print('e34 speckle partial spearman (map DOES carry realization info)')
J = json.load(open(D + 'e34_speckle.json'))
R = J['rows']
for L in (1, 2, 4, 8):
    m, se, t, p, npos = tstat(arr(R, f'partial_L{L}'))
    print(f"  L={L}  partial {m:+.4f} se {se:.4f} t {t:+.2f} pos {npos}/16"
          + ("   (claim +0.12 t 3.92)" if L == 1 else ""))

print()
print('=' * 78)
print('A-basis: universality of the radial profile across the 16 eval seas')
import sys, os
sys.path.insert(0, '.')
from wavesim.wave_generator import jonswap_components
sys.path.insert(0, 'scripts')
from d3_sa_achievability_gap import hashabs, DEPTH
KM, EN = [], []
rng_check = []
for si in range(16):
    rng = np.random.default_rng(7000 + 31 * si + hashabs('spread'))
    Hs = rng.uniform(1.5, 3.0)
    mdir = rng.uniform(0, 360)
    c = jonswap_components(Hs=Hs, Tp=7.0, n_components=200, depth=DEPTH,
                           gamma=3.3, direction_deg=mdir,
                           direction_spread_deg=30.0, seed=4000 + si)
    km = np.array([np.hypot(x.kx, x.ky) for x in c])
    A = np.array([x.A for x in c])
    e = A ** 2 / 2
    kp = km[np.argmax(e)]
    KM.append(km / kp)
    EN.append(e / e.sum())
KM = np.array(KM); EN = np.array(EN)
dev_k = np.abs(KM - KM[0]).max()
dev_e = np.abs(EN - EN[0]).max()
print(f"  |k_n|/kp inter-sea max dev  = {dev_k:.3e}   (claim 3.55e-15)")
print(f"  normalized A_n^2/2 max dev  = {dev_e:.3e}   (claim 2.78e-17)")

print()
print('=' * 78)
print('incumbent checkpoint')
import torch
ck = torch.load(D + 'prior_net_l1s2bw.pt', map_location='cpu')
print(f"  keys: {list(ck.keys())}")
for k in ('step', 'val', 'best_val'):
    if k in ck:
        print(f"  {k} = {ck[k]}")
