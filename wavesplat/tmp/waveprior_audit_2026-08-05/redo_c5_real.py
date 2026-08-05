import json
import numpy as np
from scipy import stats

d = json.load(open('/home/vexit/wavesplat/wavesplat/tmp/wp_real_2026-07-25/wp_real.json'))
R = d['results']
print('n windows', len(R), 'collapse flagged', sum(1 for r in R if r['collapse']))

norm = [r for r in R if not r['collapse']]
def deltas(rows, key_h='hand_now', key_l='learned_now'):
    return np.array([r[key_h] - r[key_l] for r in rows])   # >0 => learned better

for lab, rows in (('normal-27 (PREREG decision set)', norm), ('all-30', R)):
    dd = deltas(rows)
    h = np.array([r['hand_now'] for r in rows]); l = np.array([r['learned_now'] for r in rows])
    W, p = stats.wilcoxon(dd)
    print('\n--- %s ---' % lab)
    print('  hand mean %.4f  learned mean %.4f' % (h.mean(), l.mean()))
    print('  hand median %.4f learned median %.4f' % (np.median(h), np.median(l)))
    print('  mean D(hand-learned) %+.4f   median D %+.4f   [SIGN AGREE? %s]'
          % (dd.mean(), np.median(dd), 'yes' if np.sign(dd.mean()) == np.sign(np.median(dd)) else '**NO**'))
    print('  learned-better wins %d/%d   Wilcoxon W=%.1f p=%.4g' % (int((dd > 0).sum()), len(dd), W, p))
    dnz = dd[dd != 0]; rk = stats.rankdata(np.abs(dnz))
    Wp = rk[dnz > 0].sum(); Wm = rk[dnz < 0].sum()
    print('  rank-biserial %.3f' % ((Wp - Wm) / (Wp + Wm)))
    sd = dd.std(ddof=1)
    print('  sd=%.4f  MDE(2-sided,80%%)=%.4f  = %.0f%% of hand mean' %
          (sd, (stats.norm.ppf(.975)+stats.norm.ppf(.8))*sd/np.sqrt(len(dd)),
           100*(stats.norm.ppf(.975)+stats.norm.ppf(.8))*sd/np.sqrt(len(dd))/h.mean()))

# per-record cluster
print('\n--- per-record (clustered unit = record, n=5) ---')
recs = sorted({r['record'] for r in R})
cl = []
for rec in recs:
    rows = [r for r in norm if r['record'] == rec]
    dd = deltas(rows)
    cl.append(np.median(dd))
    print('  %-6s n=%d  medD=%+.4f  learned wins %d/%d  n_eff mean %.1f' %
          (rec, len(rows), np.median(dd), int((dd > 0).sum()), len(dd),
           np.mean([r['n_eff'] for r in rows])))
cl = np.array(cl)
print('  record-level sign test: learned better on %d/5 -> two-sided floor p=%.4f (n=5 CANNOT reach 0.05)'
      % (int((cl > 0).sum()), 2*0.5**5))

# mechanism: n_eff vs delta
ne = np.array([r['n_eff'] for r in R]); dd = deltas(R)
print('\n--- MECHANISM CHECK: n_eff vs delta ---')
print('  n_eff range %.1f-%.1f mean %.1f median %.1f' % (ne.min(), ne.max(), ne.mean(), np.median(ne)))
print('  Pearson r(n_eff, D)=%.3f p=%.3f | Spearman rho=%.3f p=%.3f'
      % (*stats.pearsonr(ne, dd), *stats.spearmanr(ne, dd)))
ne_n = np.array([r['n_eff'] for r in norm]); dd_n = deltas(norm)
print('  normal-27: Pearson r=%.3f p=%.3f | Spearman rho=%.3f p=%.3f'
      % (*stats.pearsonr(ne_n, dd_n), *stats.spearmanr(ne_n, dd_n)))
print('  learned_now vs n_eff: Spearman rho=%.3f p=%.3f'
      % stats.spearmanr([r['n_eff'] for r in norm], [r['learned_now'] for r in norm]))

# collapse windows
print('\n--- collapse windows (secondary, "robustness" claim) ---')
for r in R:
    if r['collapse']:
        print('  %-6s w%d  hand %.3f -> learned %.3f  (%s)' %
              (r['record'], r['win'], r['hand_now'], r['learned_now'],
               'FIXED (<1.0)' if r['learned_now'] < 1.0 else 'not fixed'))
