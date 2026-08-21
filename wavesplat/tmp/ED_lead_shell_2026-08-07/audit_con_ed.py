"""CON audit of E-D.  Independent re-derivation from results.jsonl. No GPU."""
import json, math, os, statistics as st, itertools

HERE = os.path.dirname(os.path.abspath(__file__))
A, B = 'AUTO', 'AUTO-NOTRACK'
LEADS = [('1', '0.1s'), ('5', '0.5s'), ('10', '1.0s'), ('20', '2.0s')]
SHELLS = ['0_5', '5_10', '10_20', '20_30', '30_50', '50_75', '75_100', '100_150', '150_inf']
NEED = {48: 31, 47: 31, 46: 30, 45: 30, 44: 29, 43: 29, 42: 28, 41: 28, 40: 27, 39: 27, 38: 26}


def sign_p(k, n):
    if n == 0:
        return 1.0
    lo = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(lo + 1)) / 2 ** n)


rows = [json.loads(l) for l in open(os.path.join(HERE, 'results.jsonl')) if l.strip()]
ok = [r for r in rows if 'error' not in r]
IDX = {}
for r in ok:
    IDX.setdefault((r['cell'], r['episode']), []).append(r)
eps = sorted({r['episode'] for r in ok})

EXCL = ['BF4_30comp_seed6_un', 'BF5_30comp_seed4_un']


def is_excl(e):
    return any(x in e for x in EXCL)


KEEP = [e for e in eps if not is_excl(e)]


def vals(c, e, path):
    out = []
    for x in IDX.get((c, e), []):
        v = x
        for k in path:
            v = v.get(k) if isinstance(v, dict) else None
            if v is None:
                break
        if v is not None and v == v:
            out.append((x['seed'], v))
    return dict(out)


def med(c, e, path, agg=st.median, seed=None):
    d = vals(c, e, path)
    if not d:
        return None
    if seed is not None:
        return d.get(seed)
    return agg(list(d.values()))


def report(name, deltas, n_need=None):
    d = [x for x in deltas if x is not None]
    n = len(d)
    k = sum(1 for x in d if x < 0)
    ties = sum(1 for x in d if x == 0)
    need = NEED.get(n - ties, '?') if n_need is None else n_need
    print(f'  {name:34s} med={st.median(d):+.5f}  {k}/{n}  need={need}  p={sign_p(k, n - ties):.5f}')
    return st.median(d), k, n


print('=' * 78)
print('## ATTACK 1 — PRIMARY-1 on a RELATIVE scale (paired, per-episode; no ratio of aggregates)')
print('=' * 78)
print('  (per-episode seed-median first, exactly as PREREG §3; then paired signed delta)')
abs_d, log_d, rel_d = {}, {}, {}
for lk, ln in LEADS:
    da, dl, dr = [], [], []
    for e in KEEP:
        a = med(A, e, ['lead_resid', lk]); b = med(B, e, ['lead_resid', lk])
        if a is None or b is None:
            continue
        da.append(a - b)
        dl.append(math.log(a) - math.log(b))
        dr.append((a - b) / b)
    abs_d[lk], log_d[lk], rel_d[lk] = da, dl, dr
    print(f'\n  lead {ln}:')
    report('  absolute  medDelta [m]', da)
    report('  log-ratio med[ln(A)-ln(B)]', dl)
    report('  relative  med[(A-B)/B]', dr)

print('\n  --- decay ratio A(2.0s)/A(0.1s) under each scale ---')
for nm, D in (('absolute (PREREG)', abs_d), ('log-ratio', log_d), ('relative (A-B)/B', rel_d)):
    r = abs(st.median(D['20'])) / abs(st.median(D['1']))
    verdict = 'STRUCTURE-REAL(>=0.5)' if r >= 0.5 else '**WHITE-FITTING(<0.5)**'
    print(f'    {nm:22s} A(20)/A(1) = {r:.3f}   -> {verdict}')

print('\n  --- per-episode decay ratio (median of per-pair ratios, absolute scale) ---')
pr = []
for e in KEEP:
    a1 = med(A, e, ['lead_resid', '1']); b1 = med(B, e, ['lead_resid', '1'])
    a20 = med(A, e, ['lead_resid', '20']); b20 = med(B, e, ['lead_resid', '20'])
    if None in (a1, b1, a20, b20):
        continue
    pr.append(abs(a20 - b20) / abs(a1 - b1))
print(f'    median per-pair A(20)/A(1) = {st.median(pr):.3f}   '
      f'frac<0.5 = {sum(1 for x in pr if x < 0.5)}/{len(pr)}')

print('\n  --- growth of each arm (per-episode median of the ratio resid(20)/resid(1)) ---')
for c in (A, B):
    g = []
    for e in KEEP:
        a1 = med(c, e, ['lead_resid', '1']); a20 = med(c, e, ['lead_resid', '20'])
        if a1 and a20:
            g.append(a20 / a1)
    print(f'    {c:14s} median resid(2.0s)/resid(0.1s) = {st.median(g):.3f}')

print('\n  --- how far could AUTO degrade before the PREREG rule flips? ---')
mb20 = st.median([med(B, e, ['lead_resid', '20']) for e in KEEP])
ma20 = st.median([med(A, e, ['lead_resid', '20']) for e in KEEP])
half = 0.5 * abs(st.median(abs_d['1']))
print(f'    WHITE-FITTING needs |medDelta(20)| < {half:.4f}; floor(20) med = {mb20:.4f}')
print(f'    => AUTO(20) would have to exceed {mb20-half:.4f} m  (it is {ma20:.4f} m, '
      f'{100*ma20/(mb20-half):.0f}% of the way)')

print('\n' + '=' * 78)
print('## ATTACK 2 — PRIMARY-2: is the 20-30 m crossing robust?')
print('=' * 78)


def shell_deltas(shell, keep, agg=st.median, seed=None):
    d = []
    for e in keep:
        a = med(A, e, ['shell_rmse', shell], agg, seed)
        b = med(B, e, ['shell_rmse', shell], agg, seed)
        if a is not None and b is not None:
            d.append((e, a - b))
    return d


base = {s: shell_deltas(s, KEEP) for s in SHELLS}
print('\n  reproduction (seed-median, n=46):')
for s in SHELLS:
    d = [v for _, v in base[s]]
    k = sum(1 for v in d if v < 0)
    print(f'    {s:10s} med={st.median(d):+.5f}  {k}/{len(d)}  p={sign_p(k,len(d)):.6f}')

print('\n  (a) leave-one-out on shell 20_30 (need 30 of 45):')
d20 = base['20_30']
worst = []
for i in range(len(d20)):
    sub = [v for j, (_, v) in enumerate(d20) if j != i]
    k = sum(1 for v in sub if v < 0)
    worst.append((k, len(sub), d20[i][0]))
kmin = min(worst)
print(f'    worst LOO = {kmin[0]}/{kmin[1]} (dropping {kmin[2]})  need=30  '
      f'-> {"PASS" if kmin[0] >= 30 else "FAIL"}')

print('  (b) drop-k jackknife (adversarial: drop the k episodes with most negative delta):')
srt = sorted(d20, key=lambda t: t[1])
for kdrop in (1, 2, 3, 4):
    sub = [v for _, v in srt[kdrop:]]
    k = sum(1 for v in sub if v < 0)
    n = len(sub)
    print(f'    drop {kdrop}: {k}/{n}  need={NEED[n]}  p={sign_p(k,n):.4f}  '
          f'-> {"PASS" if k >= NEED[n] else "**FAIL**"}')

print('  (c) single-seed instead of median-of-3 (shell 20_30):')
for sd in sorted({r['seed'] for r in ok}):
    d = [v for _, v in shell_deltas('20_30', KEEP, seed=sd)]
    k = sum(1 for v in d if v < 0)
    print(f'    seed {sd}: med={st.median(d):+.5f}  {k}/{len(d)}  need={NEED[len(d)]}  '
          f'p={sign_p(k,len(d)):.4f}  -> {"PASS" if k >= NEED[len(d)] else "**FAIL**"}')

print('  (d) mean-of-3 seeds instead of median-of-3 (shell 20_30 and 30_50):')
for s in ('20_30', '30_50'):
    d = [v for _, v in shell_deltas(s, KEEP, agg=lambda x: sum(x) / len(x))]
    k = sum(1 for v in d if v < 0)
    print(f'    {s}: med={st.median(d):+.5f}  {k}/{len(d)}  p={sign_p(k,len(d)):.4f}')

print('  (e) unexcluded n=48 (need 31):')
for s in ('20_30', '30_50'):
    d = [v for _, v in shell_deltas(s, eps)]
    k = sum(1 for v in d if v < 0)
    print(f'    {s}: med={st.median(d):+.5f}  {k}/{len(d)}  need={NEED[len(d)]}  '
          f'p={sign_p(k,len(d)):.4f}  -> {"PASS" if max(k,len(d)-k) >= NEED[len(d)] else "**FAIL**"}')

print('  (f) multiplicity: 9 shells tested, PREREG has no correction. Holm on the '
      'crossing pair:')
ps = sorted([(sign_p(sum(1 for v in [x for _, x in base[s]] if v < 0), len(base[s])), s)
             for s in SHELLS])
for rank, (p, s) in enumerate(ps):
    m = 9 - rank
    print(f'    {s:10s} p={p:.6f}  Holm alpha=0.05/{m}={0.05/m:.5f}  '
          f'-> {"pass" if p <= 0.05/m else "**FAIL**"}')

print('  (g) MDE per shell (PREREG §3 requires MDE; analyze_ed.py printed it only for leads):')
for s in ('10_20', '20_30', '30_50'):
    d = [v for _, v in base[s]]
    sd = st.stdev(d)
    mde = 2.8 * sd / math.sqrt(len(d))
    print(f'    {s:10s} sd={sd:.5f}  MDE(80%)={mde:.5f}  |obs|={abs(st.median(d)):.5f}  '
          f'obs/MDE={abs(st.median(d))/mde:.2f}x')

print('  (h) cross-seed spread of the AUTO arm itself at 20_30 (determinism floor check):')
sp = []
for e in KEEP:
    v = list(vals(A, e, ['shell_rmse', '20_30']).values())
    if len(v) == 3:
        sp.append(max(v) - min(v))
print(f'    median seed range within AUTO at 20_30 = {st.median(sp):.5f}  '
      f'(vs |medDelta| = {abs(st.median([v for _,v in base["20_30"]])):.5f})')

print('\n' + '=' * 78)
print('## ATTACK 3/4 — reuse and null exclusion (short)')
print('=' * 78)
print('  sign-agreement of the PRIMARY-1 lead-1 delta with the disk oos residual delta '
      '(same quantity, sanity):')
agree = 0
for e in KEEP:
    a = med(A, e, ['obs_resid_rms_oos']) - med(B, e, ['obs_resid_rms_oos'])
    b = med(A, e, ['lead_resid', '1']) - med(B, e, ['lead_resid', '1'])
    agree += (a < 0) == (b < 0)
print(f'    {agree}/{len(KEEP)}')
print('  effective independent units: episodes are 48 but they are the SAME 48 scored by '
      '4 prior batches; the two NEW metrics are new, the EPISODES are not.')
print('  null exclusion: shell results with and without the 2 excluded episodes above (e).')
print('  excluded-episode shell 20_30 deltas:')
for e in eps:
    if is_excl(e):
        a = med(A, e, ['shell_rmse', '20_30']); b = med(B, e, ['shell_rmse', '20_30'])
        print(f'    {e[:34]:36s} delta={a-b:+.5f}')

print('\n' + '=' * 78)
print('## ATTACK 1c — decorrelation arithmetic at 2 s')
print('=' * 78)
for frac in (0.011, 0.055, 0.11, 0.22):
    print(f'    lead = {frac*100:5.1f}% of a period -> cos(2*pi*f) = {math.cos(2*math.pi*frac):+.3f}')
print('    (a 2 s lead at Tp=9 s leaves cos=0.19: a WHITE component is ~81% decorrelated,')
print('     so the test does have teeth in the temporal axis.)')
