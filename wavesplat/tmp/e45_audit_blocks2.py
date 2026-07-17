"""e45 audit part 2b — block heterogeneity + blind-route statistics."""
import json
import numpy as np
from scipy import stats

D = 'results/direction3/'


def paired(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    t, p = stats.ttest_1samp(d, 0)
    lo, hi = stats.t.interval(0.95, n - 1, loc=d.mean(), scale=se)
    return dict(mean=d.mean(), se=se, t=float(t), p=float(p), lo=lo, hi=hi,
                win=int((d > 0).sum()), n=n)


def fmt(s):
    return (f"{s['mean']:+.4f} +-{s['se']:.4f} CI[{s['lo']:+.4f},{s['hi']:+.4f}] "
            f"win {s['win']}/{s['n']}")


ests = []
e37 = json.load(open(D + 'e37_crosstp_exploratory.json'))
s = paired([r['ens2_oracle'] - r['cur_grid'] for r in e37['rows']])
print('si=0..31    (e37 ens2_oracle-cur):', fmt(s)); ests.append(s)

e42 = json.load(open(D + 'e42_blind_recovery.json'))
s = paired([r['oracle'] - r['cur_grid'] for r in e42['rows']])
print('si=100..163 (e42 oracle-cur)    :', fmt(s)); ests.append(s)

e43b = json.load(open(D + 'e43b_v3_pipeline.json'))
s = paired([r['oracle'] - r['cur_grid'] for r in e43b['rows']])
print('si=200..263 (e43b oracle-cur)   :', fmt(s)); ests.append(s)

e43d = json.load(open(D + 'e43d_gate.json'))
s = paired([r['oracle_x'] - r['cur_grid'] for r in e43d['rows']])
print('si=300..347 (e43d oracle_x-cur) :', fmt(s)); ests.append(s)

e43e = json.load(open(D + 'e43e_fresh_battery.json'))
s = paired([r['oracle_x'] - r['cur_grid'] for r in e43e['rows']])
print('si=400..447 (e43e oracle_x-cur) :', fmt(s)); ests.append(s)

e44 = json.load(open(D + 'e44_encounter_transfer.json'))
s = paired([r['fld_orc'] - r['fld_cur'] for r in e44['rows']])
print('si=1000..1959 (e44 fld orc-cur) :', fmt(s)); ests.append(s)

m = np.array([e['mean'] for e in ests])
se = np.array([e['se'] for e in ests])
w = 1 / se ** 2
mu_fe = (w * m).sum() / w.sum()
Q = ((m - mu_fe) ** 2 * w).sum()
k = len(m)
pQ = 1 - stats.chi2.cdf(Q, k - 1)
print(f'\nfixed-effect mean {mu_fe:+.4f} +- {1/np.sqrt(w.sum()):.4f}')
print(f'Cochran Q = {Q:.2f} df {k-1}  p = {pQ:.3f}  I2 = {max(0,(Q-(k-1))/Q)*100:.0f}%')
# without e44 (5 blocks the banner cites)
m5, se5 = m[:5], se[:5]
w5 = 1 / se5 ** 2
mu5 = (w5 * m5).sum() / w5.sum()
Q5 = ((m5 - mu5) ** 2 * w5).sum()
print(f'5 blocks only: mean {mu5:+.4f}, unweighted {m5.mean():+.4f}, '
      f'Q {Q5:.2f} df 4 p {1-stats.chi2.cdf(Q5,4):.3f}')
# winner's curse test: discovery block vs the rest
z = (m[0] - mu_fe) / np.sqrt(se[0] ** 2)
print(f"discovery si=0..31 vs FE-mean: diff {m[0]-mu_fe:+.4f}, z≈{(m[0]-mu_fe)/se[0]:+.2f}")
z12 = (0.0982 - 0.0715) / np.sqrt(0.0204**2 + 0.0141**2)
print(f"doc '27% winner's curse' z (e37 vs e42): {z12:+.2f} -> p {2*(1-stats.norm.cdf(abs(z12))):.2f}")

print('\n=========== blind route ===========')
sc = paired([r['FULL'] - r['cur_grid'] for r in e43d['rows']])
print('e43d si=300..347 FULL-cur :', fmt(sc))
se_ = paired([r['FULL'] - r['cur_grid'] for r in e43e['rows']])
print('e43e si=400..447 FULL-cur :', fmt(se_))
sm = paired([r['FULL'] - r['m2'] for r in e43d['rows']])
print('e43d FULL-m2 (GATE)       :', fmt(sm))
sm2 = paired([r['FULL'] - r['m2'] for r in e43e['rows']])
print('e43e FULL-m2 (GATE)       :', fmt(sm2))
# pooled n=96
dp = ([r['FULL'] - r['cur_grid'] for r in e43d['rows']] +
      [r['FULL'] - r['cur_grid'] for r in e43e['rows']])
print('pooled FULL-cur n=96      :', fmt(paired(dp)))
zhet = (sc['mean'] - se_['mean']) / np.sqrt(sc['se']**2 + se_['se']**2)
print(f'heterogeneity z = {zhet:+.2f}  p = {2*(1-stats.norm.cdf(abs(zhet))):.4f}')
# DerSimonian-Laird RE with 2 blocks
mm = np.array([sc['mean'], se_['mean']]); ss = np.array([sc['se'], se_['se']])
ww = 1/ss**2
mu = (ww*mm).sum()/ww.sum()
Q2 = ((mm-mu)**2*ww).sum()
tau2 = max(0, (Q2 - 1) / (ww.sum() - (ww**2).sum()/ww.sum()))
wr = 1/(ss**2 + tau2)
mu_re = (wr*mm).sum()/wr.sum(); se_re = 1/np.sqrt(wr.sum())
print(f'RE(DL): tau2 {tau2:.5f}  mu {mu_re:+.4f} +- {se_re:.4f} '
      f'CI[{mu_re-1.96*se_re:+.4f},{mu_re+1.96*se_re:+.4f}]  (df=1: meaningless)')
# earlier blind attempts (flawed search)
sb = paired([r['fit_bmaF'] - r['cur_grid'] for r in e42['rows']])
print('e42 fit_bmaF-cur (topK6)  :', fmt(sb))
sv = paired([r['v3_point'] - r['cur_grid'] for r in e43b['rows']])
print('e43b v3_point-cur (topK6) :', fmt(sv))
# consistency of 4 blind estimates with common effect
bm = np.array([sb['mean'], sv['mean'], sc['mean'], se_['mean']])
bs = np.array([sb['se'], sv['se'], sc['se'], se_['se']])
wb = 1/bs**2
mub = (wb*bm).sum()/wb.sum()
Qb = ((bm-mub)**2*wb).sum()
print(f'4 blind estimates: {bm.round(4)} FE mean {mub:+.4f} '
      f'Q {Qb:.2f} df 3 p {1-stats.chi2.cdf(Qb,3):.4f}')

print('\n=========== e43e unimodal ungated FULL ===========')
u = e43e['uni']
print('uni keys:', sorted(u[0].keys()), 'n', len(u))
if 'FULL' in u[0]:
    print('uni FULL-cur:', fmt(paired([r['FULL'] - r['cur_grid'] for r in u])))
