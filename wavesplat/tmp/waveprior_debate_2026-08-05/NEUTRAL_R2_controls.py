"""NEUTRAL R2 controls: do NOT promote corr(dn, gain) without controlling the shared term.

gain = nrmse_v2 - nrmse_learned ;  dn = n_learned - n_v2.
Both contain the v2 run. A bad v2 run (over-pruned -> low n_v2, high nrmse_v2) inflates BOTH
-> spurious positive correlation. This is the exact error class the programme has repeated.
"""
import json, re, collections, statistics as st
import numpy as np
from scipy import stats as sps

ROOT = '/home/vexit/wavesplat/wavesplat/'
load = lambda p: [json.loads(l) for l in open(ROOT + p) if l.strip()]
WPI = load('tmp/wp_integration_2026-07-25/arm_runs.jsonl')
ncomp = lambda ep: int(re.search(r'_(\d+)comp', ep).group(1))
key = lambda r: (r['episode'], r['est_seed'])
V = {key(r): r for r in WPI if r['arm'] == 'v2_konly' and not r.get('err')}
L = {key(r): r for r in WPI if r['arm'] == 'learned' and not r.get('err')}
LK = {key(r): r for r in WPI if r['arm'] == 'learned_konly' and not r.get('err')}
OUT = {}

rows = []
for k in sorted(set(V) & set(L)):
    v, l = V[k], L[k]
    if None in (v.get('final_n'), l.get('final_n')):
        continue
    n_gt = ncomp(k[0])
    rows.append(dict(episode=k[0], lane=k[1], n_gt=n_gt,
                     n_v2=v['final_n'], n_l=l['final_n'], dn=l['final_n'] - v['final_n'],
                     e_v2=v['avg_nRMSE'], e_l=l['avg_nRMSE'],
                     gain=v['avg_nRMSE'] - l['avg_nRMSE'],
                     err_n_v2=abs(v['final_n'] - n_gt), err_n_l=abs(l['final_n'] - n_gt)))


def sp(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return None
    return {'pearson': round(float(np.corrcoef(a, b)[0, 1]), 4),
            'spearman': round(float(sps.spearmanr(a, b).statistic), 4),
            'p_spearman': round(float(sps.spearmanr(a, b).pvalue), 5), 'n': len(a)}


# ---- C1: split the shared-term correlation into its two halves, WITHIN nominal-N stratum
OUT['C1_within_arm_count_vs_error'] = {}
for N in (15, 30, 'sparse'):
    sub = [r for r in rows if (r['n_gt'] == N if N != 'sparse' else r['n_gt'] not in (15, 30))]
    OUT['C1_within_arm_count_vs_error'][str(N)] = {
        'n': len(sub),
        'v2_arm: n_v2 vs nrmse_v2': sp([r['n_v2'] for r in sub], [r['e_v2'] for r in sub]),
        'learned_arm: n_l vs nrmse_l': sp([r['n_l'] for r in sub], [r['e_l'] for r in sub]),
        'dn vs gain (SHARED TERM - suspect)': sp([r['dn'] for r in sub], [r['gain'] for r in sub]),
        'n_l vs nrmse_l is the CLEAN within-arm test': True}

# ---- C2: does the learned gain survive when the count is (near) matched?
OUT['C2_gain_at_matched_count'] = {}
for N in (15, 30):
    sub = [r for r in rows if r['n_gt'] == N]
    for tag, sel in (('|dn|<=1', [r for r in sub if abs(r['dn']) <= 1]),
                     ('|dn|<=2', [r for r in sub if abs(r['dn']) <= 2]),
                     ('dn<=0 (learned used FEWER or equal)',
                      [r for r in sub if r['dn'] <= 0]),
                     ('all', sub)):
        if not sel:
            continue
        g = [r['gain'] for r in sel]
        OUT['C2_gain_at_matched_count'][f'N{N} | {tag}'] = {
            'n': len(g), 'median_gain': round(st.median(g), 4),
            'wins': f'{sum(1 for x in g if x > 0)}/{len(g)}',
            'sign_test_p_twosided': round(float(sps.binomtest(
                sum(1 for x in g if x > 0), len(g), 0.5).pvalue), 5)}

# ---- C3: count ACCURACY (|n - n_gt|) per arm, and whether it explains the stratum flip
OUT['C3_count_accuracy'] = {}
for N in (1, 2, 5, 15, 30):
    sub = [r for r in rows if r['n_gt'] == N]
    if not sub:
        continue
    OUT['C3_count_accuracy'][f'N{N}'] = {
        'n': len(sub), 'n_gt': N,
        'median_n_v2': st.median([r['n_v2'] for r in sub]),
        'median_n_learned': st.median([r['n_l'] for r in sub]),
        'median_abs_count_err_v2': st.median([r['err_n_v2'] for r in sub]),
        'median_abs_count_err_learned': st.median([r['err_n_l'] for r in sub]),
        'learned_count_closer_frac': round(
            sum(r['err_n_l'] < r['err_n_v2'] for r in sub) / len(sub), 3),
        'median_gain': round(st.median([r['gain'] for r in sub]), 4)}

# does "whose count is closer to truth" predict who wins, across ALL 260 lanes?
closer = [(r['err_n_l'] < r['err_n_v2'], r['gain'] > 0) for r in rows]
tab = collections.Counter(closer)
OUT['C3_count_accuracy']['contingency_countCloser_x_learnedWins'] = {
    f'countCloser={a}, learnedWins={b}': c for (a, b), c in sorted(tab.items())}
a = np.array([[tab[(True, True)], tab[(True, False)]],
              [tab[(False, True)], tab[(False, False)]]])
OUT['C3_count_accuracy']['fisher_p'] = round(float(sps.fisher_exact(a).pvalue), 6)
OUT['C3_count_accuracy']['odds_ratio'] = round(float(sps.fisher_exact(a).statistic), 4)

# ---- C4: does the ORACLE arm's count survive pruning while beamform's does not?
S = load('tmp/s1s2_2026-08-05/results.jsonl')
NGT = {'episode_BF4_15comp_seed0_uniform.h5': 15, 'episode_BF4_30comp_seed0_uniform.h5': 30,
       'episode_BF5_15comp_seed0_uniform.h5': 15, 'episode_BF5_30comp_seed0_uniform.h5': 30}
byarm = collections.defaultdict(lambda: collections.defaultdict(list))
for r in S:
    byarm[r['arm']][r['episode']].append(r['N'])
OUT['C4_pruner_spares_correct_placements'] = {
    arm: {ep.replace('episode_', '').replace('_seed0_uniform.h5', ''):
          {'n_gt': NGT[ep], 'median_final_N': st.median(v),
           'kept_frac_of_requested': round(st.median(v) / NGT[ep], 3)}
          for ep, v in sorted(d.items())}
    for arm, d in sorted(byarm.items()) if arm in ('S2-C1', 'S2-T1', 'S2-T1f', 'S1-A0')}
OUT['C4_pruner_spares_correct_placements']['NOTE'] = (
    'S2-C1 and S2-T1 request the SAME count n_gt under the SAME schedule; they differ only in '
    'placement. If S2-C1 ends below n_gt while S2-T1 holds n_gt, the adaptive-N pruner is '
    'deleting BADLY-PLACED components -> count is not an independent knob from placement.')

# ---- C5: learned vs learned_konly, paired, on crowded (does w/phi matter at all?)
d5 = []
for k in sorted(set(L) & set(LK)):
    if ncomp(k[0]) in (15, 30):
        d5.append(L[k]['avg_nRMSE'] - LK[k]['avg_nRMSE'])
OUT['C5_learned_vs_learned_konly_crowded'] = {
    'n': len(d5), 'median_delta_(learned - konly)': round(st.median(d5), 4),
    'learned_better_frac': round(sum(x < 0 for x in d5) / len(d5), 3),
    'wilcoxon_p': round(float(sps.wilcoxon(d5).pvalue), 5),
    'INTERPRETATION': ('near-zero => the LS w/phi fit adds little => the 07-25 ablation excludes '
                       'w/phi ONLY. It cannot separate placement from count, because BOTH arms '
                       'carry the learned proposal count n=len(pool_and_select(...)).')}

with open(ROOT + 'tmp/waveprior_debate_2026-08-05/NEUTRAL_R2_controls.json', 'w') as f:
    json.dump(OUT, f, indent=1, default=str)
print(json.dumps(OUT, indent=1, default=str))
