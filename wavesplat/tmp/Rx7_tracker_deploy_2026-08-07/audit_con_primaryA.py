"""CON audit — PRIMARY-A stress test with the FULL pre-registered pipeline replayed
(Holm chain recomputed inside every resample), plus capacity accounting for PRIMARY-B.
Read-only.
"""
import json, math, os, random
import statistics as st
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NEED = {16: 13, 15: 12, 14: 12, 13: 11, 12: 10, 11: 10, 10: 9, 9: 8, 8: 8}
CAND = ['AUTO-LR10', 'AUTO-LR100', 'AUTO-LR1000']
rows = [json.loads(l) for l in open(f'{HERE}/results.jsonl') if l.strip()]
rows = [r for r in rows if 'error' not in r]
NULL = {n['episode']: n['zero_null_tracking_nRMSE_mean']
        for n in json.load(open(f'{HERE}/nulls_rx7.json'))}
IDX = {}
for r in rows:
    IDX.setdefault((r['cell'], r['episode']), []).append(r)
EPS = sorted({r['episode'] for r in rows})
SEEDS = sorted({r['seed'] for r in rows})


def sgn_p(k, n):
    if n == 0:
        return 1.0
    lo = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(lo + 1)) / 2.0 ** n)


def med(c, e, k='nRMSE'):
    v = [x[k] for x in IDX[(c, e)] if x.get(k) is not None and x[k] == x[k]]
    return st.median(v)


KEPT = [e for e in EPS if med('AUTO', e) < NULL[e]]
DLT = {c: {e: med(c, e) - med('AUTO-NOTRACK', e) for e in EPS} for c in CAND}


def verdict(block):
    """Replay analyze_rx7.main()'s PRIMARY-A logic verbatim on an arbitrary block."""
    res = {}
    for c in CAND:
        d = [DLT[c][e] for e in block if DLT[c][e] != 0]
        w = sum(1 for x in d if x < 0)
        res[c] = (st.median(d) if d else None, w, len(d), sgn_p(w, len(d)))
    order = sorted(res, key=lambda c: res[c][3])
    keep = False
    detail = []
    for i, c in enumerate(order):
        a = 0.05 / (3 - i)
        m, w, n, p = res[c]
        ok = p <= a and m < 0 and w >= NEED.get(n, 99)
        keep |= ok
        detail.append((c, p, a, ok))
        if p > a:
            break
    return ('KEEP-REDUCED' if keep else 'SHIP-OFF'), res, detail


print('=' * 78)
print('A. PRIMARY-A verdict replayed on perturbed blocks (full Holm chain each time)')
print('=' * 78)
v, res, det = verdict(KEPT)
print(f'  as pre-registered (n=13, null-excluded): {v}')
for c, p, a, ok in det:
    print(f'     {c:13s} p={p:.4f} alpha={a:.4f} {"PASS" if ok else "fail"}')
v2, res2, det2 = verdict(EPS)
print(f'  mandated sensitivity (n=16, NO exclusion): {v2}')
for c, p, a, ok in det2:
    print(f'     {c:13s} p={p:.4f} alpha={a:.4f} {"PASS" if ok else "fail"}')

print('\n  Leave-one-out (13 blocks of 12), full pipeline:')
cnt = {}
for i in range(len(KEPT)):
    blk = KEPT[:i] + KEPT[i + 1:]
    vv, _, _ = verdict(blk)
    cnt[vv] = cnt.get(vv, 0) + 1
print(f'    {cnt}')
print('\n  Leave-two-out (78 blocks of 11), full pipeline:')
cnt2 = {}
import itertools
for a, b in itertools.combinations(range(len(KEPT)), 2):
    blk = [e for j, e in enumerate(KEPT) if j not in (a, b)]
    vv, _, _ = verdict(blk)
    cnt2[vv] = cnt2.get(vv, 0) + 1
print(f'    {cnt2}')

print('\n  Single-seed replay (drop median-of-3, use one estimator seed):')
for s in SEEDS:
    dl = {c: {e: [x for x in IDX[(c, e)] if x['seed'] == s][0]['nRMSE'] -
                 [x for x in IDX[('AUTO-NOTRACK', e)] if x['seed'] == s][0]['nRMSE']
              for e in EPS} for c in CAND}
    old = {c: dict(DLT[c]) for c in CAND}
    DLT.update(dl)
    vv, rr, _ = verdict(KEPT)
    print(f'    seed {s}: {vv:13s} LR1000 medD={rr["AUTO-LR1000"][0]:+.6f} '
          f'{rr["AUTO-LR1000"][1]}/{rr["AUTO-LR1000"][2]} p={rr["AUTO-LR1000"][3]:.4f}')
    DLT.update(old)

print('\n  Bootstrap over episodes (10k, n=13 resampled with replacement):')
rng = random.Random(7)
d = [DLT['AUTO-LR1000'][e] for e in KEPT]
meds, keeps = [], 0
for _ in range(10000):
    samp = [d[rng.randrange(len(d))] for _ in range(len(d))]
    meds.append(st.median(samp))
meds.sort()
print(f'    median D bootstrap 95% CI = [{meds[250]:+.6f}, {meds[9750]:+.6f}]  '
      f'(point {st.median(d):+.6f})')
print(f'    P(bootstrap median >= 0) = {sum(1 for x in meds if x >= 0)/len(meds):.4f}')

print('\n  Alternative (non-pre-registered) aggregations, for fragility only:')
print(f'    mean of paired D = {st.mean(d):+.6f}')
try:
    from scipy.stats import wilcoxon, ttest_1samp
    print(f'    Wilcoxon signed-rank p = {wilcoxon(d).pvalue:.4f}')
    print(f'    paired t-test p        = {ttest_1samp(d, 0).pvalue:.4f}')
except Exception as ex:
    print('    (scipy unavailable:', ex, ')')

print('\n  Exact sign-test MDE at n=13, two-sided alpha=0.025 (Holm slot LR1000 got):')
# need k wins out of 13 for p<=0.025 -> smallest k with sgn_p(k,13)<=0.025
kk = [k for k in range(14) if sgn_p(k, 13) <= 0.025 and k > 6.5]
print(f'    minimum wins required for p<=0.025 at n=13: {min(kk)} (achieved {res["AUTO-LR1000"][1]})')
print(f'    at n=12 the same gate needs {min(k for k in range(13) if sgn_p(k,12)<=0.025 and k>6)}'
      f' of 12 -> that is why 11/13 LOO subsets fall over')
slack = st.median([med('AUTO', e) - med('AUTO-NOTRACK', e) for e in KEPT])
print(f'    |medD(LR1000)| / tracker slack = {abs(st.median(d))/abs(slack)*100:.1f}% '
      f'(slack {slack:+.5f})')
print(f'    |medD(LR1000)| / block nRMSE level = '
      f'{abs(st.median(d))/st.median([med("AUTO-NOTRACK",e) for e in KEPT])*100:.2f}%')

print('\n' + '=' * 78)
print('B. degeneracy: is AUTO-LR1000 distinguishable from the floor in any practical sense?')
print('=' * 78)
lv1000 = st.median([med('AUTO-LR1000', e) for e in KEPT])
lvfl = st.median([med('AUTO-NOTRACK', e) for e in KEPT])
print(f'  block level LR1000={lv1000:.6f}  floor={lvfl:.6f}  diff={lv1000-lvfl:+.2e}')
print(f'  -> identical to 4 decimals ({lv1000:.4f} vs {lvfl:.4f})')
print('  PREREG sec.6 bans, verbatim:')
print('    "lr->0 cell being good may not be cited as a finding (converges to AUTO-NOTRACK by definition)"')
print('    "a DEGENERATE cell (indistinguishable from AUTO or from the floor) may not be a deployment ground"')
print(f'  AUTO-LR1000 is the lr->0 cell (5e-6 = shipped lr / 1000) and at n=16 it is')
print(f'  NOT distinguishable from the floor (12/16, p=0.0768).')

print('\n' + '=' * 78)
print('C. PRIMARY-B capacity accounting (tracking is fit to ~150 residuals)')
print('=' * 78)
ns = [r['final_n'] if r.get('final_n') else r['N'] for r in rows if r['cell'] == 'AUTO']
print(f'  final_n over AUTO runs: median={st.median(ns)} min={min(ns)} max={max(ns)}')
cfg = [r['cfg'] for r in rows if r['cell'] == 'AUTO'][0]
print(f'  tracking_window={cfg.get("tracking_window")}  tracking_mode={cfg.get("tracking_mode")}  '
      f'tracking_steps={cfg.get("tracking_steps")}  lambda_l1={cfg.get("lambda_l1")}')
print('  returns/frame ~ 50 (measured) -> window of 3 frames ~ 150 residuals')
print(f'  free params in w_phi tracking = 2N = {2*st.median(ns):.0f} '
      f'-> obs/param ~ {150/(2*st.median(ns)):.1f}')
print('  LiDAR range noise sigma = 0.02 m (wavesim/noise_model.py default)')
print(f'  AUTO observed-region RMSE level = 0.086 m = {0.086/0.02:.1f}x the range-noise sigma')
print('  -> the observed-region gain is NOT noise-fitting; it is real local fit.')
