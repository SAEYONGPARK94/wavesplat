"""CON re-derivation for Rx-2 attack lines 1 (MDE / excluded middle), 3 (transfer), 5 (regime).

Pure re-analysis of artifacts already on disk. No new runs here (probes live in CON_probe.py).
Aggregation follows the Rx-2 PREREG: per-episode median-of-seeds -> paired delta ->
median of paired deltas; two-sided sign test; k/n always reported with p.
"""
import json, os, itertools
import numpy as np
from math import comb

TMP = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SNAPD = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasnap')
OUT = []


def P(*a):
    s = ' '.join(str(x) for x in a)
    print(s)
    OUT.append(s)


def load(path, cellkey):
    rows = []
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if 'error' in r:
            continue
        r['_cell'] = r[cellkey]
        rows.append(r)
    return rows


def by_cell_ep(rows, metric):
    """{(cell, ep): [values across seeds]}"""
    d = {}
    for r in rows:
        v = r.get(metric)
        if v is None or not np.isfinite(v):
            continue
        d.setdefault((r['_cell'], r['episode']), []).append(float(v))
    return d


def sign_p(k, n):
    """two-sided exact sign test p-value for k successes of n."""
    if n == 0:
        return float('nan')
    tail = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired(rows, cell_a, cell_b, metric, eps=None):
    """median-of-seeds per episode, then paired delta (b - a). Returns (deltas, eps)."""
    d = by_cell_ep(rows, metric)
    e_a = {e for (c, e) in d if c == cell_a}
    e_b = {e for (c, e) in d if c == cell_b}
    E = sorted(e_a & e_b) if eps is None else [e for e in eps if e in e_a & e_b]
    dl = [float(np.median(d[(cell_b, e)]) - np.median(d[(cell_a, e)])) for e in E]
    return dl, E


def boot_primary(rows, cell_a, cell_b, metric, n_boot=20000, seed=0, resample_eps=False):
    """Bootstrap the PRIMARY statistic (median of paired deltas).

    Resamples SEEDS within each (cell, episode) -- i.e. the run-to-run noise that a
    median-of-3 does not remove. With resample_eps also resamples episodes, which adds
    the episode-sampling noise that a 4-episode panel carries.
    """
    d = by_cell_ep(rows, metric)
    E = sorted({e for (c, e) in d if c == cell_a} & {e for (c, e) in d if c == cell_b})
    rng = np.random.default_rng(seed)
    A = [np.asarray(d[(cell_a, e)]) for e in E]
    B = [np.asarray(d[(cell_b, e)]) for e in E]
    out = np.empty(n_boot)
    n = len(E)
    for i in range(n_boot):
        idx = rng.integers(0, n, n) if resample_eps else np.arange(n)
        dl = []
        for j in idx:
            a, b = A[j], B[j]
            dl.append(np.median(rng.choice(b, len(b))) - np.median(rng.choice(a, len(a))))
        out[i] = np.median(dl)
    return out, E


def mde_sign(sd_delta, n, power=0.80):
    """Effect size needed for `power` chance that all n paired deltas share a sign
    (= the only outcome that reaches the n=4 p-floor of 0.125)."""
    from scipy.stats import norm
    return float(norm.ppf(power ** (1.0 / n)) * sd_delta)


# ===========================================================================
P('=' * 100)
P('SECTION 0 — sources')
P('=' * 100)
S1S2 = os.path.join(SNAPD, 's1s2_2026-08-05.results.jsonl')
RX1 = os.path.join(SNAPD, 'rx1_tracker_2026-08-05.results2.jsonl')
E2 = os.path.join(SNAPD, 'e2_grid_sep_2026-08-05.results.jsonl')
E1 = os.path.join(SNAPD, 'e1_grid_2026-08-05.results.jsonl')
s1s2 = load(S1S2, 'arm')
rx1 = load(RX1, 'cell')
e2 = load(E2, 'cell')
e1 = load(E1, 'cell')
for nm, rr in (('s1s2', s1s2), ('rx1', rx1), ('e2', e2), ('e1', e1)):
    P(f'  {nm:6s} rows={len(rr):4d} cells={sorted({r["_cell"] for r in rr})}')

# ===========================================================================
P('')
P('=' * 100)
P('ATTACK 1 — what effect could each "dead" candidate still be hiding?')
P('=' * 100)
P('PRIMARY statistic = median of paired per-episode deltas (Rx-2 PREREG).')
P('CI-seed  = 95% bootstrap over SEEDS only (episodes held fixed).')
P('CI-both  = 95% bootstrap over seeds AND episodes (generalisation to new episodes).')
P('MDE      = |effect| needed for an 80% chance of n/n sign agreement, the ONLY outcome')
P('           that reaches the p-floor at that n.')
P('')
CAND = [
    ('tracker OFF @deployment',      rx1,  'D-BASE',  'D-NOTRACK', 'nRMSE'),
    ('tracker lambda=0 @deployment', rx1,  'D-BASE',  'D-K12',     'nRMSE'),
    ('tracker steps=1 @deployment',  rx1,  'D-BASE',  'D-K6',      'nRMSE'),
    ('budget 15 -> 40 (S1)',         s1s2, 'S1-A0',   'S1-A3',     'nRMSE'),
    ('budget 15 -> 22 (S1)',         s1s2, 'S1-A0',   'S1-A1',     'nRMSE'),
    ('budget 15 -> 30 (S1)',         s1s2, 'S1-A0',   'S1-A2',     'nRMSE'),
    ('prune OFF @oracle',            s1s2, 'S2-T1',   'S2-T2',     'nRMSE'),
    ('adaptive-N OFF @oracle',       s1s2, 'S2-T1f',  'S2-T4',     'nRMSE'),
    ('grid 81->401 (E2 U1)',         e2,   'A0',      'U1',        'nRMSE'),
    ('grid+minsep (E2 U1M2)',        e2,   'A0',      'U1M2',      'nRMSE'),
    ('radius 0.2->0.1 (E2 U2)',      e2,   'A0',      'U2',        'nRMSE'),
]
P(f'{"candidate":32s} {"n":>3s} {"PRIMARY":>9s} {"k/n":>6s} {"p":>7s} '
  f'{"CI-seed":>19s} {"CI-both":>19s} {"sd":>8s} {"MDE80":>8s}')
mde_rows = {}
for name, rows, a, b, metric in CAND:
    dl, E = paired(rows, a, b, metric)
    if not dl:
        P(f'{name:32s}  -- missing')
        continue
    n = len(dl)
    k = sum(1 for v in dl if v < 0)          # nRMSE: negative = improvement
    bs, _ = boot_primary(rows, a, b, metric, n_boot=4000, resample_eps=False)
    bb, _ = boot_primary(rows, a, b, metric, n_boot=4000, resample_eps=True)
    sd = float(np.std(dl, ddof=1)) if n > 1 else float('nan')
    mde = mde_sign(sd, n)
    mde_rows[name] = dict(n=n, primary=float(np.median(dl)), k=k, sd=sd, mde=mde,
                          ci_seed=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                          ci_both=[float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))],
                          deltas=dl, eps=E)
    P(f'{name:32s} {n:3d} {np.median(dl):+9.4f} {k:3d}/{n:<2d} {sign_p(k,n):7.4f} '
      f'[{np.percentile(bs,2.5):+7.4f},{np.percentile(bs,97.5):+7.4f}] '
      f'[{np.percentile(bb,2.5):+7.4f},{np.percentile(bb,97.5):+7.4f}] '
      f'{sd:8.4f} {mde:8.4f}')

P('')
P('The prize being claimed for initialisation, on the same 4 crowded episodes:')
dl_orc, E = paired(s1s2, 'S1-A0', 'S2-T1f', 'nRMSE')
P(f'  oracle-placement medDelta(nRMSE) = {np.median(dl_orc):+.4f}  per-ep '
  f'{[round(v,4) for v in dl_orc]}')
P('  -> compare each MDE above with this. A candidate whose MDE exceeds the prize could')
P('     be hiding an effect the size of the prize and still have come back "NULL".')

# ===========================================================================
P('')
P('=' * 100)
P('ATTACK 3 — does init_nrmse buy nRMSE? (empirical transfer coefficient)')
P('=' * 100)


def cell_levels(rows, metric):
    d = by_cell_ep(rows, metric)
    return {ce: float(np.median(v)) for ce, v in d.items()}


def transfer(rows, tag, cells=None, eps=None):
    """All within-episode arm pairs: regress DeltanRMSE on Deltainit_nrmse."""
    lin = cell_levels(rows, 'init_nrmse')
    lnr = cell_levels(rows, 'nRMSE')
    C = sorted({c for (c, e) in lin} if cells is None else cells)
    E = sorted({e for (c, e) in lin}) if eps is None else eps
    X, Y, LAB = [], [], []
    for e in E:
        for a, b in itertools.combinations(C, 2):
            if (a, e) in lin and (b, e) in lin and (a, e) in lnr and (b, e) in lnr:
                X.append(lin[(b, e)] - lin[(a, e)])
                Y.append(lnr[(b, e)] - lnr[(a, e)])
                LAB.append((e, a, b))
    X, Y = np.asarray(X), np.asarray(Y)
    if len(X) < 3:
        P(f'  {tag}: too few pairs ({len(X)})')
        return None
    slope_o = float((X @ Y) / (X @ X))              # through the origin
    A = np.vstack([X, np.ones_like(X)]).T
    sl, ic = np.linalg.lstsq(A, Y, rcond=None)[0]
    r = float(np.corrcoef(X, Y)[0, 1])
    # bootstrap the origin slope
    rng = np.random.default_rng(1)
    bs = [float((X[i] @ Y[i]) / (X[i] @ X[i]))
          for i in (rng.integers(0, len(X), len(X)) for _ in range(4000))]
    P(f'  {tag:44s} pairs={len(X):4d}  slope0={slope_o:+.3f} '
      f'[{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}]  '
      f'slope={sl:+.3f}  r={r:+.3f}  |dinit| med={np.median(np.abs(X)):.4f}')
    return dict(slope0=slope_o, slope=sl, r=r, n=len(X), X=X.tolist(), Y=Y.tolist(),
                lab=LAB)


P('Transfer = d(final nRMSE) / d(init_nrmse), from within-episode arm pairs.')
P('')
P('[a] DEPLOYABLE arms only (no oracle anywhere) — this is the operating point that matters')
tr_dep = transfer(s1s2, 's1s2 deployable {A0,A1,A2,A3,C1}',
                  cells=['S1-A0', 'S1-A1', 'S1-A2', 'S1-A3', 'S2-C1'])
tr_e2 = transfer(e2, 'E2 grid/radius/minsep cells (12 ep, all deployable)')
tr_e1 = transfer(e1, 'E1 cells (all deployable)')
P('')
P('[b] pairs that SPAN the oracle (one arm is oracle placement)')
tr_span = transfer(s1s2, 's1s2 all arms incl. oracle',
                   cells=['S1-A0', 'S1-A1', 'S1-A2', 'S1-A3', 'S2-C1',
                          'S2-T1', 'S2-T1f'])
P('')
P('[c] the single biggest deployable init improvement measured, and what it bought:')
di, E = paired(s1s2, 'S1-A0', 'S1-A3', 'init_nrmse')
dn, _ = paired(s1s2, 'S1-A0', 'S1-A3', 'nRMSE')
for e, a, b in zip(E, di, dn):
    P(f'    {e:42s} dinit={a:+.4f}  dnRMSE={b:+.4f}  ratio={b/a if a else float("nan"):+.2f}')
P(f'    median dinit={np.median(di):+.4f}  median dnRMSE={np.median(dn):+.4f}  '
  f'-> transfer {np.median(dn)/np.median(di):+.3f}')

# ===========================================================================
P('')
P('=' * 100)
P('ATTACK 4 — is the oracle 0.0629 confounded with init_k_lr_scale=0?')
P('=' * 100)
lvl_i = cell_levels(s1s2, 'init_nrmse')
lvl_n = cell_levels(s1s2, 'nRMSE')
P(f'{"arm":10s} {"init_method":12s} {"k frozen":9s} {"init_nrmse":>11s} {"nRMSE":>8s}')
FRZ = {'S1-A0': 'no', 'S1-A3': 'no', 'S2-C1': 'no', 'S2-T1': 'no',
       'S2-T1f': 'YES', 'S2-T4': 'YES', 'X-T1f-notrack': 'YES'}
IM = {'S1-A0': 'v2_konly', 'S1-A3': 'v2_konly', 'S2-C1': 'v2_konly',
      'S2-T1': 'oracle', 'S2-T1f': 'oracle', 'S2-T4': 'oracle',
      'X-T1f-notrack': 'oracle'}
for arm in ['S1-A0', 'S1-A3', 'S2-C1', 'S2-T1', 'S2-T1f', 'X-T1f-notrack']:
    ii = [v for (c, e), v in lvl_i.items() if c == arm]
    nn = [v for (c, e), v in lvl_n.items() if c == arm]
    P(f'{arm:10s} {IM[arm]:12s} {FRZ[arm]:9s} {np.mean(ii):11.4f} {np.mean(nn):8.4f}')
P('')
gap_quoted = np.mean([v for (c, e), v in lvl_i.items() if c == 'S1-A3']) - \
             np.mean([v for (c, e), v in lvl_i.items() if c == 'S2-T1f'])
gap_honest = np.mean([v for (c, e), v in lvl_i.items() if c == 'S1-A3']) - \
             np.mean([v for (c, e), v in lvl_i.items() if c == 'S2-T1'])
frz_part = np.mean([v for (c, e), v in lvl_i.items() if c == 'S2-T1']) - \
           np.mean([v for (c, e), v in lvl_i.items() if c == 'S2-T1f'])
P(f'  quoted gap  (S1-A3 -> S2-T1f, oracle AND k frozen) = {gap_quoted:.4f}')
P(f'  placement-only gap (S1-A3 -> S2-T1, same machinery) = {gap_honest:.4f}')
P(f'  k-FREEZING component (S2-T1 -> S2-T1f)              = {frz_part:.4f} '
  f'= {100*frz_part/gap_quoted:.1f}% of the quoted gap')
dl, E = paired(s1s2, 'S2-T1', 'S2-T1f', 'init_nrmse')
k = sum(1 for v in dl if v < 0)
P(f'  paired: medDelta init_nrmse (T1f-T1) = {np.median(dl):+.4f} {k}/{len(dl)} '
  f'p={sign_p(k,len(dl)):.4f}   per-ep {[round(v,4) for v in dl]}')
dl, E = paired(s1s2, 'S2-T1', 'S2-T1f', 'nRMSE')
k = sum(1 for v in dl if v < 0)
P(f'  paired: medDelta nRMSE      (T1f-T1) = {np.median(dl):+.4f} {k}/{len(dl)} '
  f'p={sign_p(k,len(dl)):.4f}')

# ===========================================================================
P('')
P('=' * 100)
P('ATTACK 5 — is "initialisation dominates deployment error" crowded-only?')
P('=' * 100)
P('E2 cell A0 == the shipped ladder == s1s2 S1-A0, run on the 12-episode paper panel.')
lin = cell_levels(e2, 'init_nrmse')
lnr = cell_levels(e2, 'nRMSE')
rowsA0 = sorted([(e, lin[('A0', e)], lnr[('A0', e)])
                 for (c, e) in lin if c == 'A0' and ('A0', e) in lnr])
P(f'{"episode":42s} {"init_nrmse":>11s} {"nRMSE":>8s}')
for e, i, n in rowsA0:
    P(f'{e:42s} {i:11.4f} {n:8.4f}')
arr_i = np.array([r[1] for r in rowsA0]); arr_n = np.array([r[2] for r in rowsA0])
P(f'{"MEDIAN (12 ep)":42s} {np.median(arr_i):11.4f} {np.median(arr_n):8.4f}')
crowd = [r for r in rowsA0 if ('15comp' in r[0] or '30comp' in r[0])]
sparse = [r for r in rowsA0 if r not in crowd]
P(f'{"  crowded subset (15/30comp)":42s} '
  f'{np.median([r[1] for r in crowd]):11.4f} {np.median([r[2] for r in crowd]):8.4f}  '
  f'n={len(crowd)}')
P(f'{"  non-crowded subset (1/2/5comp)":42s} '
  f'{np.median([r[1] for r in sparse]):11.4f} {np.median([r[2] for r in sparse]):8.4f}  '
  f'n={len(sparse)}')

json.dump({'mde': {k: {kk: vv for kk, vv in v.items()} for k, v in mde_rows.items()},
           'transfer_deployable_s1s2': tr_dep, 'transfer_e2': tr_e2,
           'transfer_e1': tr_e1, 'transfer_span': tr_span,
           'gap_decomp': {'quoted': gap_quoted, 'placement_only': gap_honest,
                          'freeze': frz_part},
           'e2_A0_levels': [{'ep': e, 'init_nrmse': i, 'nRMSE': n} for e, i, n in rowsA0]},
          open(os.path.join(os.path.dirname(__file__), 'CON_rederive.json'), 'w'),
          indent=1)
open(os.path.join(os.path.dirname(__file__), 'CON_rederive.txt'), 'w').write('\n'.join(OUT) + '\n')
