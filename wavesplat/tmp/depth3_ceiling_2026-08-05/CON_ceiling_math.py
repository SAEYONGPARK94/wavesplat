"""CON (depth 3) — is the "one direction per frequency bin" structure a CEILING?

Everything here is arithmetic on already-measured quantities.  NO wave runs.
Inputs:
  tmp/cdip_support_2026-08-03/cdip_support_raw.json   (25,426 real records, paired
                                                       spread_band_deg / spread_total_deg)
  tmp/audit_2026-08-04/cdip_discretize.json           (measurement A, station 028p1)
  wavesim/wave_generator.py                           (the generator itself, imported)

Estimator identity (both arms, byte-for-byte the CDIP formula):
  r1     = |a1 + i b1|
  sigma1 = sqrt(2 (1 - r1))            [rad]
"""
import json
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)
from wavesim.wave_generator import jonswap_components  # noqa: E402

OUT = {}


# ---------------------------------------------------------------- estimator
def sig_from_r1(r1):
    return np.degrees(np.sqrt(np.clip(2.0 * (1.0 - np.clip(r1, 0, 1)), 0, None)))


def r1_from_sig(sig_deg):
    return 1.0 - np.radians(np.asarray(sig_deg, float)) ** 2 / 2.0


def sig1_of_set(E, th):
    """sigma_1 of a discrete component set, exactly as cdip_support.py:synth_records."""
    e = E.sum()
    a1 = (E * np.cos(th)).sum() / e
    b1 = (E * np.sin(th)).sum() / e
    return sig_from_r1(np.hypot(a1, b1))


# ============================================================ 1. POPULATION LIMIT
# Directions are rng.normal(mu, s) -> wrapped normal -> E[e^{i theta}] = exp(-s^2/2)
# EXACTLY, independent of N and independent of the amplitude weights.
def sigma1_pop(s_deg):
    return sig_from_r1(np.exp(-np.radians(s_deg) ** 2 / 2.0))


def s_for_target(sig_target_deg):
    """Invert: which direction_spread_deg makes the POPULATION sigma_1 equal target?"""
    r1 = r1_from_sig(sig_target_deg)
    return np.degrees(np.sqrt(-2.0 * np.log(r1)))


OUT['population_limit'] = {
    'formula': 'sigma1_inf(s) = sqrt(2(1-exp(-s^2/2))), N-INDEPENDENT',
    'sigma1_inf_deg': {f'{s:g}': float(sigma1_pop(s)) for s in
                       (10, 20, 25, 30, 35, 37.2, 40, 43.5, 45, 50, 60, 80, 120)},
    's_needed_for': {
        'CDIP_total_median_35.35': float(s_for_target(35.349227870107086)),
        'CDIP_band_median_29.28': float(s_for_target(29.28)),
        '028p1_full_median_40.55': float(s_for_target(40.550073591600686)),
        '028p1_deltaized_allbins_25.17': float(s_for_target(25.172872325811397)),
    },
}


# ============================================================ 2. OUR GENERATOR'S CURVE
# The real thing: import jonswap_components and measure sigma_1 of what it returns.
def gen_sigma1(N, s_deg, seed, Hs=1.2, Tp=7.0, depth=50.0):
    cs = jonswap_components(Hs=Hs, Tp=Tp, n_components=N, depth=depth,
                            direction_deg=0.0, direction_spread_deg=s_deg, seed=seed)
    A = np.array([c.A for c in cs])
    th = np.array([np.arctan2(c.ky, c.kx) for c in cs])
    return float(sig1_of_set(0.5 * A ** 2, th))


def neff(N, Hs=1.2, Tp=7.0, gamma=3.3):
    """Energy-effective component count = 1/sum(w_i^2), w = E_i/sum(E)."""
    op = 2 * np.pi / Tp
    edges = np.linspace(0.5 * op, 3.0 * op, N + 1)
    oc, do = 0.5 * (edges[:-1] + edges[1:]), np.diff(edges)
    sg = np.where(oc <= op, 0.07, 0.09)
    S = (0.3125 * Hs ** 2 * op ** 4 / oc ** 5 * np.exp(-1.25 * (op / oc) ** 4)
         * gamma ** np.exp(-0.5 * ((oc / op - 1) / sg) ** 2))
    E = S * do                       # variance per bin ~ A_i^2/2
    w = E / E.sum()
    return float(1.0 / (w ** 2).sum())


NS = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500, 1000]
REPS = 400
curve = {}
for s in (30.0, 37.2, 43.5, 60.0):
    rows = {}
    for N in NS:
        v = np.array([gen_sigma1(N, s, seed=1000 + r) for r in range(REPS)])
        rows[N] = {'mean': float(v.mean()), 'median': float(np.median(v)),
                   'p05': float(np.percentile(v, 5)), 'p95': float(np.percentile(v, 95))}
    curve[f's={s:g}'] = {'pop_limit_deg': float(sigma1_pop(s)), 'by_N': rows}
OUT['generator_curve'] = curve
OUT['N_eff'] = {'gamma3.3': {N: neff(N) for N in NS},
                'gamma1.0_PM': {N: neff(N, gamma=1.0) for N in NS},
                'note': 'N_eff grows LINEARLY in N; slope = spectral flatness of JONSWAP'}

# where does the curve reach the real targets?
reach = {}
for tgt_name, tgt in (('CDIP_total_median_35.35', 35.349227870107086),
                      ('028p1_full_median_40.55', 40.550073591600686)):
    hits = {}
    for s in (30.0, 37.2, 43.5, 60.0):
        by = curve[f's={s:g}']['by_N']
        ok = [N for N in NS if by[N]['median'] >= tgt]
        hits[f's={s:g}'] = (min(ok) if ok else None)
    reach[tgt_name] = hits
OUT['smallest_N_reaching_target'] = reach


# ============================================================ 3. REAL WITHIN / ACROSS
# Multiplicative decomposition (exact when r1(f) is f-constant):
#   r1_total = r1_within * r1_across
# r1_within from spread_band_deg (energy-weighted per-band sigma_1)
# r1_total   from spread_total_deg
raw = json.load(open(os.path.join(ROOT, 'tmp/cdip_support_2026-08-03/cdip_support_raw.json')))
per_station = {}
allb, allt, alla = [], [], []
for st in raw['real']:
    b = np.array(st['spread_band_deg'], float)
    t = np.array(st['spread_total_deg'], float)
    m = np.isfinite(b) & np.isfinite(t)
    b, t = b[m], t[m]
    r1b, r1t = r1_from_sig(b), r1_from_sig(t)
    ok = (r1b > 1e-6) & (r1t > 0) & (r1t <= r1b + 1e-9)
    a = sig_from_r1(np.clip(r1t[ok] / r1b[ok], 0, 1))
    per_station[st['station']] = {
        'n': int(m.sum()), 'n_decomposable': int(ok.sum()),
        'within_med': float(np.median(b)), 'total_med': float(np.median(t)),
        'across_med': float(np.median(a))}
    allb += b.tolist(); allt += t.tolist(); alla += a.tolist()
allb, allt, alla = map(lambda x: np.array(x), (allb, allt, alla))
OUT['real_decomposition'] = {
    'n_records': int(allb.size), 'n_decomposable': int(alla.size),
    'pooled_median': {'within_band_deg': float(np.median(allb)),
                      'across_freq_deg': float(np.median(alla)),
                      'total_deg': float(np.median(allt))},
    'pooled_p05_p95': {
        'within': [float(np.percentile(allb, 5)), float(np.percentile(allb, 95))],
        'across': [float(np.percentile(alla, 5)), float(np.percentile(alla, 95))],
        'total': [float(np.percentile(allt, 5)), float(np.percentile(allt, 95))]},
    'frac_across_gt_within': float((alla > np.median(allb)).mean()),
    'per_station': per_station,
}

# cross-validate against measurement A on the SAME station 028p1
st28 = [s for s in raw['real'] if s['station'] == '028p1'][0]
b28 = np.array(st28['spread_band_deg'], float)
t28 = np.array(st28['spread_total_deg'], float)
r1b, r1t = r1_from_sig(b28), r1_from_sig(t28)
ok = (r1b > 1e-6) & (r1t > 0) & (r1t <= r1b + 1e-9)
a28 = sig_from_r1(np.clip(r1t[ok] / r1b[ok], 0, 1))
OUT['crossvalidate_measurement_A'] = {
    'station': '028p1', 'n_raw_this_file': int(t28.size),
    'A_median_full_deg': 40.550073591600686,
    'my_total_median_deg': float(np.median(t28)),
    'A_median_deltaized_allbins_deg': 25.172872325811397,
    'my_across_median_deg': float(np.median(a28)),
    'my_within_median_deg': float(np.median(b28)),
    'note': ('A delta-izes every frequency bin at its own mean direction; that is '
             'EXACTLY the across-frequency factor r1_across.  If the two agree, A is '
             'not a generator ceiling -- it is a measurement of the OCEAN.'),
}


# ============================================================ 4. WHAT PROCEDURE IS "B"?
# B reported: N=5 -> 29.17, N=30 -> 34.41 (= 97.3% of 35.35).
# Candidate procedures for "discretise a real record into N components keeping its
# own spread".  Only one of them is our generator.
sig_tot_all = allt.copy()
s_rec = np.degrees(np.sqrt(np.clip(-2.0 * np.log(np.clip(r1_from_sig(sig_tot_all), 1e-9, 1)), 0, None)))
rng = np.random.default_rng(20260805)
SUB = rng.choice(sig_tot_all.size, size=4000, replace=False)   # 4k of 25,426, cheap + unbiased


def w_jonswap(N, Tp=7.0, gamma=3.3):
    op = 2 * np.pi / Tp
    edges = np.linspace(0.5 * op, 3.0 * op, N + 1)
    oc, do = 0.5 * (edges[:-1] + edges[1:]), np.diff(edges)
    sg = np.where(oc <= op, 0.07, 0.09)
    S = (0.3125 * op ** 4 / oc ** 5 * np.exp(-1.25 * (op / oc) ** 4)
         * gamma ** np.exp(-0.5 * ((oc / op - 1) / sg) ** 2))
    E = S * do
    return E / E.sum()


procs = {}
for N in (1, 2, 5, 15, 30, 100):
    wj = w_jonswap(N)
    we = np.full(N, 1.0 / N)
    res = {'iid_JONSWAP_weights(OUR GENERATOR)': [], 'iid_equal_weights': [],
           'quantile_stratified_equal_weights': []}
    q = (np.arange(N) + 0.5) / N                      # deterministic stratification
    from scipy.stats import norm
    zq = norm.ppf(q)
    for i in SUB:
        s = np.radians(s_rec[i])
        th = rng.normal(0.0, s, N)
        res['iid_JONSWAP_weights(OUR GENERATOR)'].append(sig1_of_set(wj, th))
        res['iid_equal_weights'].append(sig1_of_set(we, th))
        res['quantile_stratified_equal_weights'].append(sig1_of_set(we, zq * s))
    procs[N] = {k: float(np.median(v)) for k, v in res.items()}
procs['B_reported'] = {'5': 29.17, '30': 34.41, 'target_median': 35.35}
procs['note'] = ('All three keep each record\'s own spread.  They differ ONLY in the '
                 'energy weights / draw rule.  Our generator is the first row.')
OUT['what_procedure_is_B'] = procs

json.dump(OUT, open(os.path.join(HERE, 'CON_ceiling_math.json'), 'w'), indent=1)
print(json.dumps({k: v for k, v in OUT.items()
                  if k in ('population_limit', 'crossvalidate_measurement_A',
                           'smallest_N_reaching_target')}, indent=1))
print('\n--- real decomposition ---')
print(json.dumps(OUT['real_decomposition']['pooled_median'], indent=1))
print(json.dumps(OUT['real_decomposition']['pooled_p05_p95'], indent=1))
print('\n--- N_eff ---'); print(OUT['N_eff']['gamma3.3'])
print('\n--- what is B ---'); print(json.dumps(OUT['what_procedure_is_B'], indent=1))
print('\n--- generator curve (median) ---')
for k, v in OUT['generator_curve'].items():
    print(k, 'pop', round(v['pop_limit_deg'], 3),
          {N: round(v['by_N'][N]['median'], 2) for N in NS})
