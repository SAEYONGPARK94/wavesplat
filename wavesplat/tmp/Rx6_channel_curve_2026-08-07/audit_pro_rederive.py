"""PRO independent re-derivation of Rx-6. Written from scratch against PREREG sec.4/5/7.

Rules honored:
  - metric nRMSE: seed-median -> PAIRED signed delta per episode -> median of deltas
  - exact two-sided sign test, ties dropped from n
  - NO ratio-of-aggregates for any judgment quantity
  - NO block-median-difference for any judgment quantity
  - NO absolute-value decision rule
  - nulls reported alongside; AUTO-null-fail episodes excluded from PRIMARY
"""
import json
import math
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results.jsonl')
NULLS = os.path.join(HERE, 'nulls_rx6.json')

CELLS = ['AUTO', 'AUTO-DUP', 'AUTO-NOTRACK',
         'W-S1', 'W-S2', 'W-S3', 'W-S4', 'W-S5',
         'P-S1', 'P-S2', 'P-S3', 'P-S4', 'P-S5']
SLEV = ['S1', 'S2', 'S3', 'S4', 'S5']
S_VALUE = {'S1': 5.000e-3, 'S2': 5.000e-4, 'S3': 2.205e-4, 'S4': 5.000e-5, 'S5': 5.000e-6}
NEEDED = {32: 22, 31: 22, 30: 21, 29: 21, 28: 20, 27: 20, 26: 19}


# ---------------------------------------------------------------- sign test
def sign_p(wins, n):
    """exact two-sided sign test"""
    if n == 0:
        return 1.0
    def C(n, k):
        return math.comb(n, k)
    tail_hi = sum(C(n, k) for k in range(wins, n + 1)) / 2.0 ** n
    tail_lo = sum(C(n, k) for k in range(0, wins + 1)) / 2.0 ** n
    return min(1.0, 2.0 * min(tail_hi, tail_lo))


def med(x):
    return float(np.median(np.asarray(x, dtype=float)))


def paired(dmap, a, b, eps):
    """per-episode signed delta a - b, median, wins(a>b), n(nonzero), p"""
    d = [dmap[a][e] - dmap[b][e] for e in eps]
    nz = [v for v in d if v != 0.0]
    wins = sum(1 for v in nz if v > 0)
    return dict(med=med(d) if d else float('nan'), wins=wins, n=len(nz),
                p=sign_p(wins, len(nz)), deltas=d)


# ---------------------------------------------------------------- load
raw = defaultdict(dict)          # cell -> (ep,seed) -> record
for line in open(RES):
    r = json.loads(line)
    raw[r['cell']][(r['episode'], r['seed'])] = r

eps_all = sorted({k[0] for k in raw['AUTO']})
seeds = sorted({k[1] for k in raw['AUTO']})
print(f'cells={len(raw)}  episodes={len(eps_all)}  seeds={seeds}  runs={sum(len(v) for v in raw.values())}')

# ---------------------------------------------------------------- integrity
print('\n=== INTEGRITY ===')
# 1. no NaN / missing
bad = 0
for c in CELLS:
    for k, r in raw[c].items():
        if r.get('nRMSE') is None or not np.isfinite(r['nRMSE']):
            bad += 1
print(f'1. missing/NaN nRMSE: {bad}  -> {"PASS" if bad == 0 else "FAIL"}')

# 2. AUTO-DUP bit identity
SCAL = ['nRMSE', 'Variance_Ratio', 'Pearson_rho', 'Skill_Score', 'Directional_R2',
        'Hs_error', 'absVR_framewise', 'decorr_mean', 'shrink_mean', 'm_mean',
        'avg_RMSE_unobserved', 'N', 'init_nrmse', 'init_rmse', 'auto_selected',
        'auto_n_selected', 'ensemble_k']
ARR = ['kx', 'ky', 'w', 'phi']
ndup = nfield = 0
for k in raw['AUTO']:
    a, b = raw['AUTO'][k], raw['AUTO-DUP'][k]
    ok = all(a[f] == b[f] for f in SCAL) and all(a[f] == b[f] for f in ARR)
    ndup += ok
    nfield += 1
print(f'2. AUTO-DUP == AUTO bit-identical (scalars+arrays): {ndup}/{nfield} '
      f'-> {"PASS" if ndup == nfield else "FAIL"}')

# 3. Phase-1 invariance across 13 cells
P1 = ['init_nrmse', 'init_rmse', 'auto_selected', 'auto_n_selected', 'N', 'ensemble_k']
inv = 0
tot = 0
for k in raw['AUTO']:
    tot += 1
    ref = tuple(raw['AUTO'][k][f] for f in P1)
    if all(tuple(raw[c][k][f] for f in P1) == ref for c in CELLS):
        inv += 1
print(f'3. Phase-1 invariant across all 13 cells: {inv}/{tot} -> {"PASS" if inv == tot else "FAIL"}')

# 3b. kx/ky bit-identical across cells (atom correspondence for D)
kinv = 0
for k in raw['AUTO']:
    ref = (raw['AUTO'][k]['kx'], raw['AUTO'][k]['ky'])
    if all((raw[c][k]['kx'], raw[c][k]['ky']) == ref for c in CELLS):
        kinv += 1
print(f'3b. kx/ky bit-identical across all 13 cells: {kinv}/{tot} '
      f'-> {"PASS" if kinv == tot else "FAIL"}  (atom index correspondence for D)')

# 3c. channel wiring: W cells must not move phi, P cells must not move w
print('3c. channel wiring (vs AUTO-NOTRACK final params):')
for c in CELLS[3:]:
    nphi = nw = 0
    for k in raw['AUTO']:
        base = raw['AUTO-NOTRACK'][k]
        cur = raw[c][k]
        if cur['phi'] != base['phi']:
            nphi += 1
        if cur['w'] != base['w']:
            nw += 1
    print(f'    {c:6s} phi moved {nphi:3d}/{tot}   w moved {nw:3d}/{tot}')

# ---------------------------------------------------------------- seed medians
level = defaultdict(dict)        # cell -> ep -> seed-median nRMSE
for c in CELLS:
    for e in eps_all:
        level[c][e] = med([raw[c][(e, s)]['nRMSE'] for s in seeds])

# ---------------------------------------------------------------- nulls
nulls = {r['episode']: r['zero_null_tracking_nRMSE_mean'] for r in json.load(open(NULLS))}
missing = [e for e in eps_all if e not in nulls]
print(f'\n=== NULLS ===\nepisodes without a null: {missing}')
auto_fail = [e for e in eps_all if level['AUTO'][e] >= nulls[e]]
print(f'AUTO null-FAIL episodes (excluded from PRIMARY): {len(auto_fail)}')
for e in auto_fail:
    print(f'   {e[8:-3]:34s} AUTO {level["AUTO"][e]:.4f}  null {nulls[e]:.4f}')
EPS = [e for e in eps_all if e not in auto_fail]
n = len(EPS)
print(f'PRIMARY n = {n}   needed wins = {NEEDED[n]}')

print('\ncell-wise null pass (seed-median nRMSE < null) and m_mean>1 counts, over ALL 32 eps:')
print(f'{"cell":14s} {"nullpass/32":>12s} {"m>1 /32":>9s} {"blockmed nRMSE":>15s}')
nullpass = {}
for c in CELLS:
    np_ = sum(1 for e in eps_all if level[c][e] < nulls[e])
    mgt = sum(1 for e in eps_all if med([raw[c][(e, s)]['m_mean'] for s in seeds]) > 1.0)
    nullpass[c] = np_
    print(f'{c:14s} {np_:>9d}/32 {mgt:>7d}/32 {med([level[c][e] for e in eps_all]):>15.4f}')

# ---------------------------------------------------------------- D (realized field displacement)
def wrap(x):
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


Dw = defaultdict(dict)
Dp = defaultdict(dict)
for c in CELLS:
    for e in eps_all:
        dws, dps = [], []
        for s in seeds:
            base = raw['AUTO-NOTRACK'][(e, s)]
            cur = raw[c][(e, s)]
            w0 = np.asarray(base['w'], dtype=float)
            p0 = np.asarray(base['phi'], dtype=float)
            w1 = np.asarray(cur['w'], dtype=float)
            p1 = np.asarray(cur['phi'], dtype=float)
            assert w0.shape == w1.shape
            dws.append(float(np.sqrt(np.mean((w1 - w0) ** 2))))
            dps.append(float(np.sqrt(np.mean((w0 * wrap(p1 - p0)) ** 2))))
        Dw[c][e] = med(dws)
        Dp[c][e] = med(dps)

print('\n=== REALIZED FIELD DISPLACEMENT D [m] (block median of seed medians, ALL 32) ===')
print(f'{"cell":14s} {"D_w":>12s} {"D_phi":>12s} {"D_total":>12s}')
Dtot = {}
for c in CELLS:
    dw = med([Dw[c][e] for e in eps_all])
    dp = med([Dp[c][e] for e in eps_all])
    Dtot[c] = med([math.hypot(Dw[c][e], Dp[c][e]) for e in eps_all])
    print(f'{c:14s} {dw:>12.6f} {dp:>12.6f} {Dtot[c]:>12.6f}')

print('\nmatched-pair D check (PREREG sec.1.1). channel-native D: W->D_w, P->D_phi')
print(f'{"S":4s} {"D(W)":>11s} {"D(P)":>11s} {"blockmed ratio":>15s} {"med per-ep ratio":>17s}')
Dratio = {}
for S in SLEV:
    dwv = med([Dw['W-' + S][e] for e in eps_all])
    dpv = med([Dp['P-' + S][e] for e in eps_all])
    perep = med([Dw['W-' + S][e] / Dp['P-' + S][e] for e in eps_all])
    Dratio[S] = dwv / dpv
    print(f'{S:4s} {dwv:>11.6f} {dpv:>11.6f} {dwv / dpv:>15.4f} {perep:>17.4f}')

# ---------------------------------------------------------------- PRIMARY
print('\n=== PRIMARY: C(S) = median_ep[ (W-S - AUTO) - (P-S - AUTO) ] = median_ep[W-S - P-S] ===')
print('(AUTO cancels exactly; this is a per-episode paired difference, not a difference of medians)')
print(f'{"S":4s} {"C(S)":>11s} {"wins(W worse)":>14s} {"n":>4s} {"p":>10s} {"needed":>7s} {"verdict":>9s}')
Cres = {}
for S in SLEV:
    r = paired(level, 'W-' + S, 'P-' + S, EPS)
    Cres[S] = r
    ok = 'CROSSES' if r['wins'] >= NEEDED[r['n']] or (r['n'] - r['wins']) >= NEEDED[r['n']] else '-'
    print(f'{S:4s} {r["med"]:>+11.5f} {r["wins"]:>14d} {r["n"]:>4d} {r["p"]:>10.4f} '
          f'{NEEDED[r["n"]]:>7d} {ok:>9s}')

crossers = [S for S in SLEV if Cres[S]['wins'] >= NEEDED[Cres[S]['n']]
            or (Cres[S]['n'] - Cres[S]['wins']) >= NEEDED[Cres[S]['n']]]
signs = {S: (1 if Cres[S]['med'] > 0 else -1) for S in crossers}
print(f'\ncrossing S: {crossers}  signs {signs}')
if len(crossers) <= 1:
    verdict = 'NO-DIFFERENCE'
elif len(set(signs.values())) == 1 and len(crossers) >= 3:
    verdict = 'CHANNEL-REAL'
elif len(set(signs.values())) > 1:
    verdict = 'CROSSOVER'
else:
    verdict = 'NO-DIFFERENCE (2 crossers, same sign, <3)'
print(f'PRIMARY VERDICT = {verdict}')

# unexcluded (all 32) sensitivity
print('\nsensitivity: same C(S) computed on ALL 32 episodes (no null exclusion)')
for S in SLEV:
    r = paired(level, 'W-' + S, 'P-' + S, eps_all)
    print(f'  {S}: {r["med"]:+.5f}  {r["wins"]}/{r["n"]}  p={r["p"]:.4f}')

# ---------------------------------------------------------------- key contrasts
print('\n=== KEY PAIRED CONTRASTS (PRIMARY episode set, n=%d) ===' % n)
def show(a, b, label=''):
    r = paired(level, a, b, EPS)
    tag = 'PASS' if r['wins'] >= NEEDED[r['n']] or (r['n'] - r['wins']) >= NEEDED[r['n']] else 'null'
    print(f'{a:14s} - {b:14s} med {r["med"]:>+9.5f}  wins {r["wins"]:>3d}/{r["n"]:<3d} '
          f'p={r["p"]:<9.5f} {tag:5s} {label}')
    return r

hr = show('AUTO-NOTRACK', 'AUTO', '<- tracker off vs shipping')
for S in SLEV:
    show('W-' + S, 'AUTO')
for S in SLEV:
    show('P-' + S, 'AUTO')
print()
for S in SLEV:
    show('W-' + S, 'AUTO-NOTRACK', '<- vs floor (degeneracy)')
for S in SLEV:
    show('P-' + S, 'AUTO-NOTRACK', '<- vs floor (degeneracy)')

# ---------------------------------------------------------------- levels
print('\n=== LEVELS (block median of seed-median nRMSE) ===')
print(f'{"cell":14s} {"PRIMARY n=%d" % n:>14s} {"ALL 32":>10s}')
for c in CELLS:
    print(f'{c:14s} {med([level[c][e] for e in EPS]):>14.4f} {med([level[c][e] for e in eps_all]):>10.4f}')
print(f'{"NULL":14s} {med([nulls[e] for e in EPS]):>14.4f} {med([nulls[e] for e in eps_all]):>10.4f}')

# ---------------------------------------------------------------- damage vs D curve
print('\n=== DAMAGE vs REALIZED DISPLACEMENT D  (PRIMARY set) ===')
print('damage = per-episode paired (cell - AUTO-NOTRACK) median  [nRMSE above the floor]')
print(f'{"cell":14s} {"D_native[m]":>12s} {"D/median|w|":>12s} {"damage":>10s} {"wins/n":>9s} {"p":>9s}')
curve = {}
for c in ['W-S5', 'W-S4', 'W-S3', 'W-S2', 'W-S1']:
    r = paired(level, c, 'AUTO-NOTRACK', EPS)
    d = med([Dw[c][e] for e in EPS])
    curve[c] = (d, r['med'])
    print(f'{c:14s} {d:>12.6f} {d / 0.044095315:>12.4f} {r["med"]:>+10.5f} '
          f'{r["wins"]:>4d}/{r["n"]:<4d} {r["p"]:>9.5f}')
for c in ['P-S5', 'P-S4', 'P-S3', 'P-S2', 'P-S1']:
    r = paired(level, c, 'AUTO-NOTRACK', EPS)
    d = med([Dp[c][e] for e in EPS])
    curve[c] = (d, r['med'])
    print(f'{c:14s} {d:>12.6f} {d / 0.044095315:>12.4f} {r["med"]:>+10.5f} '
          f'{r["wins"]:>4d}/{r["n"]:<4d} {r["p"]:>9.5f}')
r = paired(level, 'AUTO', 'AUTO-NOTRACK', EPS)
print(f'{"AUTO(both)":14s} {med([math.hypot(Dw["AUTO"][e], Dp["AUTO"][e]) for e in EPS]):>12.6f} '
      f'{"":>12s} {r["med"]:>+10.5f} {r["wins"]:>4d}/{r["n"]:<4d} {r["p"]:>9.5f}')

# ---- do the two channels lie on ONE function of D? log-log fit + interleaved test
print('\n--- one-function test: damage vs D, both channels, log-log ---')
xs_w = np.log10([curve['W-' + S][0] for S in SLEV])
ys_w = np.array([curve['W-' + S][1] for S in SLEV])
xs_p = np.log10([curve['P-' + S][0] for S in SLEV])
ys_p = np.array([curve['P-' + S][1] for S in SLEV])
print('W channel (log10 D, damage):', list(zip(np.round(xs_w, 3), np.round(ys_w, 5))))
print('P channel (log10 D, damage):', list(zip(np.round(xs_p, 3), np.round(ys_p, 5))))

# interpolate P curve at each W point's D and test the residual per-episode where D overlaps
print('\n--- interleaving: rank all 10 cells by D and by damage ---')
order_D = sorted(curve, key=lambda c: curve[c][0])
order_dm = sorted(curve, key=lambda c: curve[c][1])
print('by D    :', ' < '.join(f'{c}({curve[c][0]:.4f})' for c in order_D))
print('by damage:', ' < '.join(f'{c}({curve[c][1]:+.4f})' for c in order_dm))
rd = {c: i for i, c in enumerate(order_D)}
rm = {c: i for i, c in enumerate(order_dm)}
dr = np.array([rd[c] for c in curve])
mr = np.array([rm[c] for c in curve])
sp = np.corrcoef(dr, mr)[0, 1]
print(f'Spearman rank corr (D vs damage) over 10 cells = {sp:+.4f}')

# per-episode Spearman: does D predict damage WITHIN episode across cells?
sps = []
for e in EPS:
    dvals, mvals = [], []
    for c in curve:
        dvals.append(Dw[c][e] if c.startswith('W') else Dp[c][e])
        mvals.append(level[c][e] - level['AUTO-NOTRACK'][e])
    a = np.argsort(np.argsort(dvals)).astype(float)
    b = np.argsort(np.argsort(mvals)).astype(float)
    sps.append(float(np.corrcoef(a, b)[0, 1]))
print(f'per-episode Spearman(D, damage) over 10 cells: median {med(sps):+.4f}, '
      f'min {min(sps):+.4f}, max {max(sps):+.4f}, positive in {sum(1 for v in sps if v > 0)}/{len(sps)}')

# ---- channel offset at MATCHED D (the honest one-function test)
print('\n--- matched-D test: at each S the two cells sit at D ratio; compare damage ---')
for S in SLEV:
    rw = paired(level, 'W-' + S, 'AUTO-NOTRACK', EPS)
    rp = paired(level, 'P-' + S, 'AUTO-NOTRACK', EPS)
    print(f'{S}: D_W {med([Dw["W-"+S][e] for e in EPS]):.6f} damage {rw["med"]:+.5f} | '
          f'D_P {med([Dp["P-"+S][e] for e in EPS]):.6f} damage {rp["med"]:+.5f} | '
          f'D ratio {Dratio[S]:.3f}')

# ---------------------------------------------------------------- power / MDE
print('\n=== POWER / MDE (bootstrap, sign test at the frozen threshold) ===')
rng = np.random.default_rng(20260807)


def mde(deltas, need, nboot=4000, target=0.80):
    d = np.asarray(deltas, dtype=float)
    d0 = d - np.median(d)
    nn = len(d0)
    lo, hi = 0.0, max(0.2, 4 * float(np.std(d0)) + 0.05)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        samp = rng.choice(d0, size=(nboot, nn), replace=True) + mid
        wins = (samp > 0).sum(axis=1)
        pw = float((wins >= need).mean())
        if pw < target:
            lo = mid
        else:
            hi = mid
    return hi


def obs_power(deltas, need, nboot=4000):
    d = np.asarray(deltas, dtype=float)
    samp = rng.choice(d, size=(nboot, len(d)), replace=True)
    wins = (samp > 0).sum(axis=1)
    return float((wins >= need).mean())


headroom = abs(hr['med'])
print(f'tracker headroom |AUTO-NOTRACK - AUTO| = {headroom:.5f}')
for S in SLEV:
    r = Cres[S]
    m = mde(r['deltas'], NEEDED[r['n']])
    print(f'C({S}): obs {r["med"]:+.5f}  MDE(80%) {m:.5f}  = {m / headroom * 100:5.1f}% of headroom  '
          f'obs/MDE {abs(r["med"]) / m:5.2f}x')
m = mde(hr['deltas'], NEEDED[hr['n']])
print(f'AUTO-NOTRACK-AUTO: obs {hr["med"]:+.5f} MDE(80%) {m:.5f} = {m / headroom * 100:.1f}% of headroom  '
      f'power(obs)={obs_power(hr["deltas"], NEEDED[hr["n"]]):.3f}')

# ---------------------------------------------------------------- folds
print('\n=== CONFIG FOLDS (n=8 each; NO significance, NO effect size per PREREG sec.4.6) ===')
folds = defaultdict(list)
for e in eps_all:
    bf = 'BF4' if '_BF4_' in e else 'BF5'
    nc = '15comp' if '_15comp_' in e else '30comp'
    folds[(bf, nc)].append(e)
print(f'{"fold":14s} {"C(S1) wins/8":>13s} {"NOTRACK-AUTO wins/8":>21s}')
for k in sorted(folds):
    fe = [e for e in folds[k] if e in EPS]
    r1 = paired(level, 'W-S1', 'P-S1', fe)
    r2 = paired(level, 'AUTO-NOTRACK', 'AUTO', fe)
    print(f'{str(k):14s} {r1["wins"]:>6d}/{r1["n"]:<6d} {r2["wins"]:>14d}/{r2["n"]:<6d}   '
          f'(medΔ {r1["med"]:+.4f} / {r2["med"]:+.4f})')

# ---------------------------------------------------------------- LOO stability
print('\n=== LOO stability of AUTO-NOTRACK - AUTO ===')
d = hr['deltas']
fails = 0
for i in range(len(d)):
    dd = [v for j, v in enumerate(d) if j != i]
    nz = [v for v in dd if v != 0]
    w = sum(1 for v in nz if v > 0)
    if w < NEEDED[len(nz)]:
        fails += 1
print(f'LOO n=31 -> 30 (needed {NEEDED[30]}): threshold failures {fails}/{len(d)}')

# ---------------------------------------------------------------- secondary metrics
print('\n=== SECONDARY METRICS on the NOTRACK-AUTO contrast (block medians, descriptive) ===')
for mkey, worse in [('Pearson_rho', 'smaller'), ('absVR_framewise', 'larger'),
                    ('m_mean', 'larger'), ('decorr_mean', 'larger'), ('shrink_mean', 'larger')]:
    lv = {}
    for c in ['AUTO', 'AUTO-NOTRACK', 'W-S1', 'P-S1', 'W-S4', 'P-S5']:
        lv[c] = med([med([raw[c][(e, s)][mkey] for s in seeds]) for e in EPS])
    dm = {}
    for c in ['AUTO', 'AUTO-NOTRACK', 'W-S1', 'P-S1']:
        dm[c] = lv[c]
    # paired for the headline contrast
    per = {c: {e: med([raw[c][(e, s)][mkey] for s in seeds]) for e in EPS}
           for c in ['AUTO', 'AUTO-NOTRACK']}
    rr = paired(per, 'AUTO-NOTRACK', 'AUTO', EPS)
    print(f'{mkey:18s} ({worse}=worse)  AUTO {lv["AUTO"]:.4f}  NOTRACK {lv["AUTO-NOTRACK"]:.4f}  '
          f'paired medΔ {rr["med"]:+.4f} {rr["wins"]}/{rr["n"]} p={rr["p"]:.4f}')

json.dump({'C': {S: {k: v for k, v in Cres[S].items() if k != 'deltas'} for S in SLEV},
           'verdict': verdict,
           'notrack_auto': {k: v for k, v in hr.items() if k != 'deltas'},
           'levels_primary': {c: med([level[c][e] for e in EPS]) for c in CELLS},
           'levels_all32': {c: med([level[c][e] for e in eps_all]) for c in CELLS},
           'curve': {c: list(v) for c, v in curve.items()},
           'Dratio': Dratio, 'nullpass': nullpass, 'n_primary': n},
          open(os.path.join(HERE, 'audit_pro_out.json'), 'w'), indent=1)
print('\nwrote audit_pro_out.json')
