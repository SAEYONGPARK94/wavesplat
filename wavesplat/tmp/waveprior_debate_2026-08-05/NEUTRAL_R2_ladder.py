"""NEUTRAL R2: factor ladder. How many factors does the headline 0.2667 -> 0.1671 change?"""
import json, collections, statistics as st
import numpy as np
from scipy import stats as sps

ROOT = '/home/vexit/wavesplat/wavesplat/'
S = [json.loads(l) for l in open(ROOT + 'tmp/s1s2_2026-08-05/results.jsonl') if l.strip()]
NGT = {'episode_BF4_15comp_seed0_uniform.h5': 15, 'episode_BF4_30comp_seed0_uniform.h5': 30,
       'episode_BF5_15comp_seed0_uniform.h5': 15, 'episode_BF5_30comp_seed0_uniform.h5': 30}
by = collections.defaultdict(dict)
for r in S:
    by[r['arm']].setdefault(r['episode'], []).append(r['nRMSE'])
med = {a: {e: st.median(v) for e, v in d.items()} for a, d in by.items()}
mean4 = {a: round(st.mean(list(d.values())), 4) for a, d in med.items()}
OUT = {'mean4_by_arm': mean4}

# arm anatomy: which factors each arm changes relative to S1-A0
FACT = {
    'S1-A0':          dict(placement='beamform', count='ladder[3,8,15]+adaptiveN',
                           schedule='3-stage 200/200/500', k='trainable', tracker='on'),
    'S2-C1':          dict(placement='beamform', count='forced n_gt',
                           schedule='1-stage 900', k='trainable', tracker='on'),
    'S2-T1':          dict(placement='ORACLE',   count='forced n_gt',
                           schedule='1-stage 900', k='trainable', tracker='on'),
    'S2-T1f':         dict(placement='ORACLE',   count='forced n_gt',
                           schedule='1-stage 900', k='FROZEN',   tracker='on'),
    'X-T1f-notrack':  dict(placement='ORACLE',   count='forced n_gt',
                           schedule='1-stage 900', k='FROZEN',   tracker='OFF'),
    'X-A0-notrack':   dict(placement='beamform', count='ladder[3,8,15]+adaptiveN',
                           schedule='3-stage 200/200/500', k='trainable', tracker='OFF')}
OUT['arm_anatomy'] = FACT
OUT['n_factors_changed_A0_to_T1f'] = {
    'factors': [k for k in FACT['S1-A0'] if FACT['S1-A0'][k] != FACT['S2-T1f'][k]],
    'count': sum(FACT['S1-A0'][k] != FACT['S2-T1f'][k] for k in FACT['S1-A0'])}

# single-factor steps that ARE clean
def step(a, b, label, factor):
    da = [med[b][e] - med[a][e] for e in sorted(med[a])]
    return {'label': label, 'single_factor': factor, 'from': mean4[a], 'to': mean4[b],
            'delta_mean4': round(mean4[b] - mean4[a], 4),
            'per_ep_delta': {e.replace('episode_', '').replace('_seed0_uniform.h5', ''):
                             round(med[b][e] - med[a][e], 4) for e in sorted(med[a])},
            'n_eps_improved': sum(1 for d in da if d < 0)}


OUT['ladder'] = [
    step('S1-A0', 'S2-C1', 'A0 -> C1', 'count AND schedule (TWO factors, confounded)'),
    step('S2-C1', 'S2-T1', 'C1 -> T1', 'PLACEMENT ONLY (clean single factor)'),
    step('S2-T1', 'S2-T1f', 'T1 -> T1f', 'k-freeze ONLY (clean single factor, GT-free-executable)'),
    step('S2-T1f', 'X-T1f-notrack', 'T1f -> notrack', 'tracker OFF (clean single factor)'),
    step('S1-A0', 'X-A0-notrack', 'A0 -> A0-notrack', 'tracker OFF at BEAMFORM placement')]

span = mean4['S1-A0'] - mean4['X-T1f-notrack']
OUT['share_of_total_span'] = {
    'total_span_A0_to_XT1fnotrack': round(span, 4),
    'placement_only_C1_to_T1': round((mean4['S2-C1'] - mean4['S2-T1']) / span, 4),
    'k_freeze_T1_to_T1f': round((mean4['S2-T1'] - mean4['S2-T1f']) / span, 4),
    'tracker_off_T1f_to_notrack': round((mean4['S2-T1f'] - mean4['X-T1f-notrack']) / span, 4),
    'count+schedule_A0_to_C1': round((mean4['S1-A0'] - mean4['S2-C1']) / span, 4),
    'NOTE': 'shares are of the A0->X-T1f-notrack span and do not sum to 1 because the '
            'A0->C1 step is NEGATIVE (C1 is worse than A0) and the path is not additive.'}

# placement x tracker interaction
OUT['placement_x_tracker_interaction'] = {
    'at_beamform_placement (A0)': {'tracker_on': mean4['S1-A0'],
                                   'tracker_off': mean4['X-A0-notrack'],
                                   'tracker_effect': round(mean4['X-A0-notrack'] - mean4['S1-A0'], 4)},
    'at_oracle_placement (T1f)': {'tracker_on': mean4['S2-T1f'],
                                  'tracker_off': mean4['X-T1f-notrack'],
                                  'tracker_effect': round(mean4['X-T1f-notrack'] - mean4['S2-T1f'], 4)},
    'interaction': round((mean4['X-T1f-notrack'] - mean4['S2-T1f'])
                         - (mean4['X-A0-notrack'] - mean4['S1-A0']), 4),
    'READ': ('the tracker HELPS when placement is wrong and HURTS when placement is right. '
             'Tracker quality is therefore not a property of the tracker alone.')}

# lens audit: is "73% of the gap is placement quality" reproducible from this file?
OUT['lens_audit'] = {
    'lens_claim': 'oracle (kx,ky) => crowded 0.2667 -> 0.1671; 73% of the gap is placement quality',
    'A0_to_T1f_delta': round(mean4['S2-T1f'] - mean4['S1-A0'], 4),
    'factors_in_that_delta': OUT['n_factors_changed_A0_to_T1f'],
    'clean_placement_only_delta': round(mean4['S2-T1'] - mean4['S2-C1'], 4),
    'placement_share_of_A0_to_notrack_span': OUT['share_of_total_span']['placement_only_C1_to_T1'],
    'VERDICT': ('0.2667 -> 0.1671 changes FOUR factors (placement, count, schedule, k-freeze), '
                'not one. The clean single-factor placement effect is C1->T1 = -0.0706. '
                'As a share of the A0->tracker-off span (0.2024) placement is 34.9%, while the '
                'tracker-off step alone is 50.8%. No 73% for placement is reproducible here.')}

with open(ROOT + 'tmp/waveprior_debate_2026-08-05/NEUTRAL_R2_ladder.json', 'w') as f:
    json.dump(OUT, f, indent=1, default=str)
print(json.dumps(OUT, indent=1, default=str))
