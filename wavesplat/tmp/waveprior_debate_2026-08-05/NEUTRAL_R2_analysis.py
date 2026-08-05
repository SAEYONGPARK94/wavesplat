"""NEUTRAL round-2 reanalysis. NO new runs. Existing JSONL/JSON/HDF5-metadata only.

Targets the two gaps the round-1 NEUTRAL verdict left open (its own section 8, items 4 and
the unasked question 4 of the caller's brief):
  Q1  exact overlap SCOPE of the inherited bit-identity fact (independent re-derivation)
  Q2  does the LEARNED arm change the component COUNT, or only the placement?
  Q3  restratify the WavePrior "crowded niche" by per-episode EFFECTIVE order N_eff
  Q4  three-arm "placement" identity: what does each arm actually set?
  Q5  S2-C1 vs S1-A0 confound check (count vs optimizer schedule)
"""
import json, re, math, collections, statistics as st
import numpy as np

ROOT = '/home/vexit/wavesplat/wavesplat/'
OUT = {}


def load(p):
    return [json.loads(l) for l in open(ROOT + p) if l.strip()]


S1S2 = load('tmp/s1s2_2026-08-05/results.jsonl')
WPI = load('tmp/wp_integration_2026-07-25/arm_runs.jsonl')
WSC = load('tmp/wp_crowded_2026-07-25/ws_runs.jsonl')
CDC = load('tmp/wp_crowded_2026-07-25/cond_runs.jsonl')


def ncomp(ep):
    return int(re.search(r'_(\d+)comp', ep).group(1))


# ---------------------------------------------------------------- Q1 overlap scope
eps_s = set(r['episode'] for r in S1S2)
eps_w = set(r['episode'] for r in WPI)
overlap = sorted(eps_s & eps_w)
lanes_w = sorted(set((r['est_seed'], r['diff_seed']) for r in WPI))
seeds_s = sorted(set(r['seed'] for r in S1S2))

# bit identity, independently re-derived, on the ONLY shared lane (est_seed 42 <-> seed 42)
METRIC_MAP = {  # S1S2 key -> WPI key
    'nRMSE': 'avg_nRMSE', 'Variance_Ratio': 'avg_Variance_Ratio',
    'Pearson_rho': 'avg_Pearson_rho', 'Skill_Score': 'avg_Skill_Score',
    'Directional_R2': 'avg_Directional_R2', 'Hs_error': 'avg_Hs_error',
    'init_nrmse': 'init_nrmse', 'N': 'final_n'}
idx_s = {(r['episode'], r['seed']): r for r in S1S2 if r['arm'] == 'S1-A0'}
idx_w = {(r['episode'], r['est_seed']): r for r in WPI if r['arm'] == 'v2_konly'}
bit = {}
for ep in overlap:
    a, b = idx_s.get((ep, 42)), idx_w.get((ep, 42))
    bit[ep] = {k: (repr(a[k]) == repr(b[v])) for k, v in METRIC_MAP.items()} if (a and b) else None
n_exact = sum(sum(v.values()) for v in bit.values() if v)
n_tot = sum(len(v) for v in bit.values() if v)

OUT['Q1_overlap_scope'] = {
    'n_episodes_S1S2': len(eps_s), 'n_episodes_wp_integration': len(eps_w),
    'n_episodes_wp_crowded': len(set(r['episode'] for r in WSC)),
    'overlap_episodes': overlap, 'n_overlap_episodes': len(overlap),
    'S1S2_seeds': seeds_s, 'wp_integration_lanes_(est,diff)': [list(x) for x in lanes_w],
    'shared_lanes': [42], 'n_shared_lanes': 1,
    'n_bit_identical_runs': len(overlap) * 1,
    'bit_identity_recheck': bit, 'exact_cells': f'{n_exact}/{n_tot}',
    'SCOPE_STATEMENT': ('bit-identity is established on 4 episodes x 1 lane (est_seed=42, '
                        'diff_seed=0) x 1 arm (v2_konly/S1-A0) = 4 runs. The wp_integration '
                        'universe is 52 episodes x 5 lanes; S1S2 is 4 episodes x 3 seeds. '
                        'Seeds 123/456 (S1S2) and lanes 1-4 (wp_integration) share NOTHING.'),
    'unshared_S1S2_seeds': [s for s in seeds_s if s != 42],
    'unshared_wp_lanes': [l[0] for l in lanes_w if l[0] != 42],
    'frac_of_S1S2_runs_covered_by_overlap': round(4 / len(S1S2), 4),
    'frac_of_WPI_runs_covered_by_overlap': round(4 / len(WPI), 4)}

# ---------------------------------------------------------------- Q2 count channel
# final_n per arm, crowded (15/30) vs sparse (<15)
cnt = collections.defaultdict(list)
for r in WPI:
    if r.get('err') or r.get('final_n') is None:
        continue
    strat = 'crowded' if ncomp(r['episode']) in (15, 30) else 'sparse'
    cnt[(r['arm'], strat)].append(r['final_n'])
    cnt[(r['arm'], f'N{ncomp(r["episode"])}')].append(r['final_n'])
OUT['Q2a_final_n_by_arm'] = {
    f'{k[0]}|{k[1]}': {'n': len(v), 'median': st.median(v), 'mean': round(st.mean(v), 2),
                       'min': min(v), 'max': max(v)} for k, v in sorted(cnt.items())}

# paired per (episode, lane): does learned change N relative to v2_konly?
key = lambda r: (r['episode'], r['est_seed'])
V = {key(r): r for r in WPI if r['arm'] == 'v2_konly' and not r.get('err')}
L = {key(r): r for r in WPI if r['arm'] == 'learned' and not r.get('err')}
LK = {key(r): r for r in WPI if r['arm'] == 'learned_konly' and not r.get('err')}

pairs = []
for k in sorted(set(V) & set(L)):
    v, l = V[k], L[k]
    if None in (v.get('final_n'), l.get('final_n')):
        continue
    pairs.append({'episode': k[0], 'lane': k[1], 'ncomp': ncomp(k[0]),
                  'n_v2': v['final_n'], 'n_learned': l['final_n'],
                  'dn': l['final_n'] - v['final_n'],
                  'nrmse_v2': v['avg_nRMSE'], 'nrmse_learned': l['avg_nRMSE'],
                  'gain': v['avg_nRMSE'] - l['avg_nRMSE']})
crow = [p for p in pairs if p['ncomp'] in (15, 30)]
spar = [p for p in pairs if p['ncomp'] not in (15, 30)]


def summ(ps, field):
    v = [p[field] for p in ps]
    return {'n': len(v), 'median': round(st.median(v), 4),
            'mean': round(st.mean(v), 4),
            'q1': round(float(np.percentile(v, 25)), 4),
            'q3': round(float(np.percentile(v, 75)), 4)}


def corr(ps, a, b):
    x = np.array([p[a] for p in ps], float); y = np.array([p[b] for p in ps], float)
    if x.std() == 0 or y.std() == 0:
        return None
    r = float(np.corrcoef(x, y)[0, 1])
    from scipy import stats as sps
    sr = float(sps.spearmanr(x, y).statistic)
    return {'pearson': round(r, 4), 'spearman': round(sr, 4), 'n': len(x)}


OUT['Q2b_count_change_paired'] = {
    'crowded': {'dn': summ(crow, 'dn'), 'gain': summ(crow, 'gain'),
                'n_v2': summ(crow, 'n_v2'), 'n_learned': summ(crow, 'n_learned'),
                'corr_dn_vs_gain': corr(crow, 'dn', 'gain'),
                'frac_dn_positive': round(sum(p['dn'] > 0 for p in crow) / len(crow), 3),
                'frac_dn_zero': round(sum(p['dn'] == 0 for p in crow) / len(crow), 3)},
    'sparse': {'dn': summ(spar, 'dn'), 'gain': summ(spar, 'gain'),
               'n_v2': summ(spar, 'n_v2'), 'n_learned': summ(spar, 'n_learned'),
               'corr_dn_vs_gain': corr(spar, 'dn', 'gain'),
               'frac_dn_positive': round(sum(p['dn'] > 0 for p in spar) / len(spar), 3)}}
# split crowded by nominal N
for N in (15, 30):
    sub = [p for p in crow if p['ncomp'] == N]
    OUT['Q2b_count_change_paired'][f'crowded_N{N}'] = {
        'dn': summ(sub, 'dn'), 'gain': summ(sub, 'gain'),
        'n_v2': summ(sub, 'n_v2'), 'n_learned': summ(sub, 'n_learned'),
        'corr_dn_vs_gain': corr(sub, 'dn', 'gain')}

# does learned_konly carry the SAME count as learned?  (tests whether the 07-25 ablation
# isolates placement, or placement+count)
same, diff, both = 0, 0, []
for k in sorted(set(L) & set(LK)):
    a, b = L[k]['final_n'], LK[k]['final_n']
    if None in (a, b):
        continue
    both.append((a, b))
    same += (a == b); diff += (a != b)
OUT['Q2c_learned_konly_carries_learned_count'] = {
    'n_pairs': len(both), 'final_n_equal': same, 'final_n_differ': diff,
    'median_final_n_learned': st.median([x[0] for x in both]),
    'median_final_n_learned_konly': st.median([x[1] for x in both]),
    'median_abs_diff': st.median([abs(a - b) for a, b in both]),
    'NOTE': ('learned_konly replaces ONLY w/phi (w=0.01, random phi) per '
             'learned_init.py L381-386; kx, ky AND n=len(kx) all come from the learned '
             'proposal. final_n can still drift via adaptive-N during tracking.')}

# ---------------------------------------------------------------- Q3 N_eff restratification
import h5py, glob, os
DATA = ROOT + 'data_finemesh_2026-07-23/'
neff_ep = {}
for p in sorted(glob.glob(DATA + '*.h5')):
    try:
        with h5py.File(p, 'r') as f:
            comp = f['config/wave_components'][:]
        A = np.asarray(comp[:, 0], float)
        e = A ** 2
        w = e / e.sum()
        neff_ep[os.path.basename(p)] = {'n_nominal': int(len(A)),
                                        'N_eff': round(float(1.0 / np.sum(w ** 2)), 3)}
    except Exception as ex:
        neff_ep[os.path.basename(p)] = {'err': str(ex)}

by_nom = collections.defaultdict(list)
for ep, d in neff_ep.items():
    if 'N_eff' in d:
        by_nom[d['n_nominal']].append(d['N_eff'])
OUT['Q3a_Neff_per_nominal_N'] = {
    str(k): {'n_eps': len(v), 'min': round(min(v), 3), 'median': round(st.median(v), 3),
             'max': round(max(v), 3)} for k, v in sorted(by_nom.items())}

# overlap of N_eff ranges between the WavePrior "sparse" and "crowded" strata
crow_neff = [d['N_eff'] for ep, d in neff_ep.items()
             if 'N_eff' in d and d['n_nominal'] in (15, 30)]
spar_neff = [d['N_eff'] for ep, d in neff_ep.items()
             if 'N_eff' in d and d['n_nominal'] not in (15, 30)]
OUT['Q3b_stratum_separability_on_Neff'] = {
    'crowded_Neff': {'n': len(crow_neff), 'min': round(min(crow_neff), 3),
                     'median': round(st.median(crow_neff), 3), 'max': round(max(crow_neff), 3)},
    'sparse_Neff': {'n': len(spar_neff), 'min': round(min(spar_neff), 3),
                    'median': round(st.median(spar_neff), 3), 'max': round(max(spar_neff), 3)},
    'ranges_overlap': bool(min(crow_neff) <= max(spar_neff)),
    'gap': round(min(crow_neff) - max(spar_neff), 3)}
# best single threshold on N_eff reproducing the n_comp>=15 cut
allv = sorted([(d['N_eff'], d['n_nominal'] in (15, 30)) for d in neff_ep.values() if 'N_eff' in d])
best = None
for i in range(len(allv) + 1):
    thr = allv[i][0] if i < len(allv) else allv[-1][0] + 1
    acc = sum((v >= thr) == lab for v, lab in allv) / len(allv)
    if best is None or acc > best[1]:
        best = (thr, acc)
OUT['Q3b_stratum_separability_on_Neff']['best_single_threshold'] = {
    'thr_Neff': round(best[0], 3), 'accuracy': round(best[1], 4), 'n': len(allv)}

# WavePrior gain restratified by N_eff (per-episode, median over 5 lanes)
gain_ep = collections.defaultdict(list)
for p in pairs:
    gain_ep[p['episode']].append(p['gain'])
rows = []
for ep, g in gain_ep.items():
    d = neff_ep.get(ep)
    if not d or 'N_eff' not in d:
        continue
    rows.append({'episode': ep, 'n_nominal': d['n_nominal'], 'N_eff': d['N_eff'],
                 'gain_median': round(st.median(g), 4), 'n_lanes': len(g)})
rows.sort(key=lambda r: r['N_eff'])
x = np.array([r['N_eff'] for r in rows]); y = np.array([r['gain_median'] for r in rows])
xn = np.array([r['n_nominal'] for r in rows], float)
from scipy import stats as sps
OUT['Q3c_gain_vs_axis'] = {
    'n_episodes': len(rows),
    'corr_gain_vs_N_eff': {'pearson': round(float(np.corrcoef(x, y)[0, 1]), 4),
                           'spearman': round(float(sps.spearmanr(x, y).statistic), 4)},
    'corr_gain_vs_n_nominal': {'pearson': round(float(np.corrcoef(xn, y)[0, 1]), 4),
                               'spearman': round(float(sps.spearmanr(xn, y).statistic), 4)},
    'per_episode_sorted_by_Neff': rows}

# median gain in tertiles of N_eff vs tertiles of n_nominal
def strat_summary(vals, labels):
    d = collections.defaultdict(list)
    for v, l in zip(vals, labels):
        d[l].append(v)
    return {str(k): {'n': len(v), 'median_gain': round(st.median(v), 4),
                     'wins': f'{sum(1 for z in v if z > 0)}/{len(v)}'}
            for k, v in sorted(d.items())}


t1, t2 = np.percentile(x, [33.333, 66.667])
OUT['Q3d_gain_by_stratum'] = {
    'by_n_nominal': strat_summary(y, [int(v) for v in xn]),
    'by_Neff_tertile': strat_summary(y, ['T1_low' if v <= t1 else ('T2_mid' if v <= t2 else 'T3_high')
                                         for v in x]),
    'Neff_tertile_bounds': [round(float(t1), 3), round(float(t2), 3)]}

# ---------------------------------------------------------------- Q4 three-arm anatomy
OUT['Q4_three_arm_anatomy'] = {
    'v2_konly_beamform': {'kx_ky': 'estimated from observation (beamform peaks)',
                          'count_N': 'stage ladder [3,8,15] + adaptive-N prune/grow',
                          'w_phi': 'w=0.01, random phi', 'energy': 'not emitted',
                          'uncertainty': 'not emitted', 'sees_GT': False},
    'learned_set_prior': {'kx_ky': 'diffusion proposal, 4 posterior samples pooled',
                          'count_N': 'EMERGENT: n = len(pool_and_select(...)) with '
                                     'energy_thresh=0.03, dedup_dk=0.03, max_n=30, then adaptive-N',
                          'w_phi': 'LS fit (learned) / w=0.01+random phi (learned_konly)',
                          'energy': 'EMITTED and USED (thresholding + strongest-first ordering)',
                          'uncertainty': 'implicit in the 4-sample spread; collapsed by pooling',
                          'sees_GT': False},
    'oracle_S2_T1': {'kx_ky': 'GT planted verbatim', 'count_N': 'FORCED n_gaussians=[n_gt]',
                     'w_phi': 'optimizer (900 iters)', 'energy': 'not emitted',
                     'uncertainty': 'not emitted', 'sees_GT': True,
                     'NOTE': 'placement and count are set JOINTLY -- not a single factor '
                             'relative to A0; single-factor only against S2-C1'},
    'VERDICT': ('The three arms are NOT on one axis. v2_konly and oracle set (placement, count); '
                'learned sets (placement, count, energy-ordering) and its count is emergent '
                'rather than specified. Only the oracle sees GT.')}

# ---------------------------------------------------------------- Q5 S2-C1 confound
s_idx = collections.defaultdict(dict)
for r in S1S2:
    s_idx[r['arm']][(r['episode'], r['seed'])] = r


def med4(arm, field='nRMSE'):
    per = {}
    for (ep, sd), r in s_idx[arm].items():
        per.setdefault(ep, []).append(r[field])
    return ({ep: round(st.median(v), 4) for ep, v in sorted(per.items())},
            round(st.mean([st.median(v) for v in per.values()]), 4))


arms_of_interest = ['S1-A0', 'S2-C1', 'S2-T1', 'S2-T1f', 'X-T1f-notrack', 'X-A0-notrack']
OUT['Q5_S2C1_confound'] = {
    'levels': {a: {'per_ep_median': med4(a)[0], 'mean4': med4(a)[1],
                   'mean4_N': med4(a, 'N')[1]} for a in arms_of_interest},
    'config_deltas': {
        'S1-A0': 'n_gaussians=[3,8,15], loss_thresholds=[0.1,0.05,None], iters=[200,200,500]',
        'S2-C1': 'n_gaussians=[n_gt], loss_thresholds=[None], iters=[900]',
        'S2-T1': 'n_gaussians=[n_gt], loss_thresholds=[None], iters=[900], init=oracle'},
    'FINDING': ('S2-C1 vs S1-A0 differs in TWO factors: component count (n_gt vs ladder) AND '
                'optimizer schedule (single 900-iter stage vs 3-stage 200/200/500 ladder). '
                'The round-1 reading "giving the count alone is counterproductive" is therefore '
                'CONFOUNDED. S2-C1 vs S2-T1 is the clean single-factor placement contrast, and '
                'it holds count AND schedule fixed.')}

with open(ROOT + 'tmp/waveprior_debate_2026-08-05/NEUTRAL_R2_analysis.json', 'w') as f:
    json.dump(OUT, f, indent=1, default=str)
print(json.dumps({k: v for k, v in OUT.items() if k != 'Q3c_gain_vs_axis'}, indent=1,
                 default=str)[:7000])
