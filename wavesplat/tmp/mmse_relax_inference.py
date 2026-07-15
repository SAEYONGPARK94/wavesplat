"""#2 re-adjudication: net-MMSE (SPARSE kp) vs RELAX, cluster-robust over records.

The old "MMSE > RELAX" headline gave the MMSE arms a kp/Hs hint from the DENSE
stereo GT (§6.1 confound). This compares, per record, condition A (fixed) field
skill for three arms:
  * net-MMSE SPARSE kp  (sa_multirecord_mmse_net_sparsekp.json) -- the fair arm
  * net-MMSE DENSE  kp  (sa_multirecord_mmse_net.json)          -- the old arm
  * RELAX               (sa_multirecord.json)  -- never gets kp, sensor-realizable
Paired per record; cluster-robust t(k-1) + Webb wild bootstrap on net(sparse)-RELAX.
"""
import json
import numpy as np
from scipy import stats

RECS8 = ['BS11a', 'AA14', 'AA15a', 'BS11b', 'BS13a', 'BS13b', 'AA15b', 'YS17']
MET = ('common_unobs', 'corr_bl')          # A-condition field skill on held-out


def load(path):
    try:
        return json.load(open(path))
    except FileNotFoundError:
        return None


def rec_A(d, rec, metric=MET):
    """Mean over windows of condition-A metric for one record."""
    if d is None or rec not in d:
        return np.nan
    vals = []
    for w in d[rec]['windows']:
        m = w['conditions']['A']['mean']
        v = m.get(metric[0], {})
        vals.append(v.get(metric[1], v.get('corr', np.nan)))
    return float(np.nanmean(vals)) if vals else np.nan


sparse = load('results/direction3/sa_multirecord_mmse_net_sparsekp.json')
dense = load('results/direction3/sa_multirecord_mmse_net.json')
relax = load('results/direction3/sa_multirecord.json')

rows = []
for rec in RECS8:
    rows.append((rec, rec_A(sparse, rec), rec_A(dense, rec), rec_A(relax, rec)))

print(f"{'='*72}\n#2  net-MMSE(sparse kp) vs RELAX, condition A {MET}\n{'='*72}")
print(f"{'record':>8}{'net-sparse':>12}{'net-dense':>11}{'RELAX':>9}"
      f"{'sp-RELAX':>10}")
d_sp = []
for rec, sp, dn, rx in rows:
    dd = sp - rx
    if np.isfinite(dd):
        d_sp.append(dd)
    print(f"{rec:>8}{sp:>12.3f}{dn:>11.3f}{rx:>9.3f}{dd:>+10.3f}")

d_sp = np.array(d_sp)
k = len(d_sp)
WEBB = np.array([-np.sqrt(1.5), -1., -np.sqrt(.5), np.sqrt(.5), 1., np.sqrt(1.5)])
rng = np.random.default_rng(0)
t_o = d_sp.mean() / (d_sp.std(ddof=1) / np.sqrt(k))
boot = []
for _ in range(20000):
    yb = d_sp * rng.choice(WEBB, k)
    se = yb.std(ddof=1) / np.sqrt(k)
    boot.append(yb.mean() / se if se > 0 else 0.0)
p_wild = float((np.abs(boot) >= abs(t_o)).mean())
t_c, p_c = stats.ttest_1samp(d_sp, 0.0)
n_win = int((d_sp > 0).sum())

sp_mean = np.nanmean([r[1] for r in rows])
dn_mean = np.nanmean([r[2] for r in rows])
rx_mean = np.nanmean([r[3] for r in rows])
print(f"\n  mean over {k} records:  net-sparse {sp_mean:+.3f}  "
      f"net-dense {dn_mean:+.3f}  RELAX {rx_mean:+.3f}")
print(f"  net(sparse) - RELAX:  mean {d_sp.mean():+.4f}  "
      f"cluster t({k-1}) {t_c:+.2f} p {p_c:.4f}  wild p {p_wild:.4f}  "
      f"win {n_win}/{k}")
print(f"  net(sparse) vs net(dense): the confound moved the mean by "
      f"{sp_mean - dn_mean:+.3f} (sparse - dense)")
verdict = bool(d_sp.mean() > 0 and p_c < 0.05 and p_wild < 0.05)
print(f"  *** MMSE(sparse) > RELAX headline: "
      f"{'SURVIVES' if verdict else 'not significant at k=%d' % k}")

json.dump(dict(records=RECS8, net_sparse_mean=float(sp_mean),
               net_dense_mean=float(dn_mean), relax_mean=float(rx_mean),
               delta_sparse_relax=float(d_sp.mean()), k=k,
               cluster_t=float(t_c), cluster_p=float(p_c), wild_p=p_wild,
               win=n_win, survives=verdict,
               per_record={r[0]: dict(net_sparse=r[1], net_dense=r[2],
                                      relax=r[3]) for r in rows}),
          open('results/direction3/mmse_relax_sparsekp_inference.json', 'w'),
          indent=2)
print("\n-> results/direction3/mmse_relax_sparsekp_inference.json")
