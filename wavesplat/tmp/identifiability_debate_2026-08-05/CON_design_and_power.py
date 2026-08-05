"""CON: (a) realized-total-matched within ladder, (b) 2x2 rank check,
(c) power for today's NULL, (d) B-vs-C attribution test, (e) arm-C hidden confounds."""
import json, os, sys, itertools
import numpy as np
from scipy import stats, optimize
ROOT = '/home/vexit/wavesplat/wavesplat'
sys.path.insert(0, ROOT)
from wavesim.wave_generator import jonswap_components, check_q_constraint

FMIN, FMAX = 0.04, 0.50
Hs, Tp, DEPTH = 1.2, 7.0, 50.0
OUT = {}


def s1deg(r1):
    r1 = min(max(float(r1), 0.0), 1.0)
    return float(np.degrees(np.sqrt(max(2.0*(1.0-r1), 0.0))))


def decomp(A, om, th):
    fr = om/(2*np.pi); E = 0.5*A**2
    m = (fr >= FMIN) & (fr <= FMAX)
    A, om, th, E = A[m], om[m], th[m], E[m]
    w = E/E.sum()
    r1t = float(np.hypot((w*np.cos(th)).sum(), (w*np.sin(th)).sum()))
    r1w, Ew = 0.0, 0.0
    for o in np.unique(om):
        b = om == o; wb = E[b]/E[b].sum()
        r1w += E[b].sum()*float(np.hypot((wb*np.cos(th[b])).sum(), (wb*np.sin(th[b])).sum())); Ew += E[b].sum()
    r1w /= Ew
    return s1deg(r1w), s1deg(r1t/r1w if r1w > 0 else np.nan), s1deg(r1t)


def draws(N, M, seed):
    c = jonswap_components(Hs=Hs, Tp=Tp, n_components=N, depth=DEPTH,
                           direction_spread_deg=30.0, seed=seed, dirs_per_freq=M)
    A = np.array([x.A for x in c]); om = np.array([x.omega for x in c])
    n_freq = N//M
    rng = np.random.default_rng(10_000+seed)
    return A, om, rng.standard_normal(n_freq), rng.standard_normal(N), np.repeat(np.arange(n_freq), M)


# ---------- (a) per-EPISODE total-sigma1-matched ladder ----------------------
TARGET = 24.18       # arm A's 5-seed median realized total sigma1 (deg)
SEEDS = list(range(12))
print('=== (a) ladder at N=90 n_freq=30 M=3, per-episode total sigma1 forced to '
      f'{TARGET} deg by solving for sigma_across ===')
lad = []
for sw in [0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 22.0]:
    ws, acs, tos, sas, ok = [], [], [], [], 0
    for sd in SEEDS:
        A, om, a_std, w_std, bidx = draws(90, 3, sd)

        def f(sa):
            th = np.deg2rad(sa)*a_std[bidx] + np.deg2rad(sw)*w_std
            return decomp(A, om, th)[2] - TARGET
        try:
            if f(0.0) > 0:            # even sigma_a=0 overshoots -> infeasible
                continue
            sa = optimize.brentq(f, 0.0, 90.0, xtol=1e-3)
        except Exception:
            continue
        th = np.deg2rad(sa)*a_std[bidx] + np.deg2rad(sw)*w_std
        w, ac, to = decomp(A, om, th)
        ws.append(w); acs.append(ac); tos.append(to); sas.append(sa); ok += 1
    if ok == 0:
        print(f'  sig_w={sw:5.1f}  INFEASIBLE (total floor > target in all seeds)'); continue
    row = dict(sig_w=sw, n_feasible=ok, sigma_across_solved_med=float(np.median(sas)),
               within=float(np.median(ws)), across=float(np.median(acs)),
               total_med=float(np.median(tos)), total_max_abs_err=float(np.max(np.abs(np.array(tos)-TARGET))))
    lad.append(row)
    print(f"  sig_w={sw:5.1f}  feasible {ok}/{len(SEEDS)}  sigma_a*={row['sigma_across_solved_med']:5.2f}  "
          f"-> within={row['within']:6.2f}  across={row['across']:6.2f}  "
          f"total={row['total_med']:6.2f} (max|err| {row['total_max_abs_err']:.2e})")
OUT['ladder_total_matched'] = lad

# ---------- (b) is a 2x2 in (N,M) rank-deficient? ---------------------------
print('\n=== (b) 2x2 in (N,M): can beta_N, beta_nfreq, beta_within be separated? ===')
L3 = np.log(3.0)
cells = {'(30,1)': (30, 1), '(30,3)': (30, 3), '(90,1)': (90, 1), '(90,3)': (90, 3)}
X, names = [], []
for lab, (N, M) in cells.items():
    X.append([1.0, np.log(N/30.0)/L3, np.log((N//M)/30.0)/L3, 1.0 if M > 1 else 0.0]); names.append(lab)
X = np.array(X)
print('  design matrix [1, logN, log n_freq, within-present]:')
for lab, r in zip(names, X):
    print(f'   {lab:8s} {r}')
print('  rank =', int(np.linalg.matrix_rank(X)), 'of', X.shape[1], 'columns, from', X.shape[0], 'cells')
ns = np.linalg.svd(X)[2][-1]
print('  null-space direction (aliased contrast):', np.round(ns/np.abs(ns).max(), 6))
OUT['two_by_two_rank'] = dict(rank=int(np.linalg.matrix_rank(X)), ncol=int(X.shape[1]),
                              null_dir=[float(v) for v in ns/np.abs(ns).max()])

# same check for the ladder-at-fixed-n_freq
Xl = np.array([[1.0, np.log(N/30.0), 0.0, w] for N, w in
               [(30, 0.0), (60, 18.60), (90, 22.96), (300, 27.39)]])
print('  fixed-n_freq ladder, corr(logN, within) =',
      round(float(np.corrcoef(np.log([30, 60, 90, 300]), [0, 18.6, 22.96, 27.39])[0, 1]), 4))

# ---------- (c) power of today's NULL --------------------------------------
print('\n=== (c) power for today PRIMARY (WS nRMSE, B-A) ===')
d = np.array([-0.04707, 0.00646, 0.01646, -0.03345, -0.02995])
sd = float(np.std(d, ddof=1)); md = float(np.median(d)); mn = float(np.mean(d))
print(f'  n=5  mean={mn:+.5f}  median={md:+.5f}  sd={sd:.5f}  '
      f'signs: {int((d<0).sum())} neg / {int((d>0).sum())} pos')
t = stats.ttest_1samp(d, 0.0); wsr = stats.wilcoxon(d)
print(f'  paired t two-sided p={t.pvalue:.4f} (t={t.statistic:+.3f})   Wilcoxon two-sided p={wsr.pvalue:.4f}')


def sign_power(n, delta, sigma, alpha=0.05):
    p = stats.norm.cdf(delta/sigma)
    k = min([kk for kk in range(n+1) if stats.binom.sf(kk-1, n, 0.5) <= alpha], default=None)
    return (None, None) if k is None else (k, float(stats.binom.sf(k-1, n, p)))


def t_power(n, delta, sigma, alpha=0.05):
    ncp = delta/sigma*np.sqrt(n)
    return float(stats.nct.sf(stats.t.ppf(1-alpha, n-1), n-1, ncp))


for lab, delta in [('observed median 0.0300', 0.0300), ('observed mean 0.0175', 0.0175),
                   ('prereg MDE 0.0470', 0.0470)]:
    k5, p5 = sign_power(5, delta, sd)
    print(f'  delta={lab:24s} d/sd={delta/sd:5.2f}  sign-test power @n=5 (bar {k5}/5) = {p5:.3f}   '
          f't-test power @n=5 = {t_power(5, delta, sd):.3f}')
need = {}
for lab, delta in [('0.0300', 0.0300), ('0.0175', 0.0175)]:
    for test, fn in [('sign', lambda n: sign_power(n, delta, sd)[1]), ('t', lambda n: t_power(n, delta, sd))]:
        n = next((n for n in range(3, 400) if (fn(n) or 0) >= 0.80), None)
        need[f'{test}_delta{lab}'] = n
        print(f'  n needed for 80% power, delta={lab}, {test}-test: n = {n}')
OUT['power'] = dict(sd=sd, median=md, mean=mn, t_p=float(t.pvalue), wilcoxon_p=float(wsr.pvalue), n_needed=need)

# ---------- (d) attribution: B-A vs C-A for LETKF --------------------------
print('\n=== (d) "attribution failure" or "attribution untested"?  LETKF nRMSE ===')
BA = np.array([0.05289945046691552, 0.02276959281629118, 0.00968116229320612,
               0.008078161936471068, 0.008118673170552176])
J = json.load(open('tmp/s4_within_spread_2026-08-05/analysis.json'))['paired']
CA = np.array(J['LETKF|C_minus_A']['nRMSE']['diffs'])
BC = BA - CA
print('  B-A:', np.round(BA, 5), ' median', round(float(np.median(BA)), 5))
print('  C-A:', np.round(CA, 5), ' median', round(float(np.median(CA)), 5))
print('  B-C:', np.round(BC, 5), ' median', round(float(np.median(BC)), 5), ' mean', round(float(BC.mean()), 5),
      ' sd', round(float(np.std(BC, ddof=1)), 5))
print('  signs B-C: %d/5 positive; one-sided sign p = %.4f' %
      (int((BC > 0).sum()), float(stats.binom.sf(int((BC > 0).sum())-1, 5, 0.5))))
tt = stats.ttest_1samp(BC, 0.0)
ci = stats.t.interval(0.90, 4, loc=BC.mean(), scale=stats.sem(BC))
print(f'  paired t p={tt.pvalue:.4f}; 90% CI for mean(B-C) = [{ci[0]:+.5f}, {ci[1]:+.5f}]')
print(f'  achieved MDE(80%,sign,5/5) for B-C = {1.712*np.std(BC, ddof=1):.5f}  '
      f'vs the whole B-A effect being explained ({np.median(BA):.5f})')
print('  => can n=5 separate +0.0097 from +0.0073 (gap 0.0024)?  '
      f'gap/sd = {0.0024/np.std(BC, ddof=1):.3f};  '
      f't-power @n=5 for delta=0.0024: {t_power(5, 0.0024, float(np.std(BC, ddof=1))):.3f}')
OUT['attribution'] = dict(BA=BA.tolist(), CA=CA.tolist(), BC=BC.tolist(),
                          BC_median=float(np.median(BC)), BC_mean=float(BC.mean()),
                          BC_sd=float(np.std(BC, ddof=1)), BC_t_p=float(tt.pvalue),
                          BC_ci90=[float(ci[0]), float(ci[1])])

# ---------- (e) undeclared confounds in arm C ------------------------------
print('\n=== (e) arm C: confounds beyond the two the prereg declared ===')
for lab, N, M in [('A N30 M1', 30, 1), ('B N30 M3', 30, 3), ('C N10 M1', 10, 1), ('E N90 M3', 90, 3)]:
    qs, amax = [], []
    for sd_ in range(5):
        c = jonswap_components(Hs=Hs, Tp=Tp, n_components=N, depth=DEPTH,
                               direction_spread_deg=30.0, seed=sd_, dirs_per_freq=M)
        qs.append(check_q_constraint(c)[1]); amax.append(max(x.A for x in c))
    print(f'  {lab}:  sum(Q*w*A) = {np.median(qs):.4f}   A_max = {np.median(amax):.4f}   '
          f'clipped_by_global_Q_cap = {np.median(qs) > 0.949}')
OUT['armC_steepness'] = 'see stdout'

json.dump(OUT, open(os.path.join(ROOT, 'tmp/identifiability_debate_2026-08-05/CON_design_and_power.json'), 'w'), indent=1)
