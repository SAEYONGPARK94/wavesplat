"""Merge the window-selection evidence into tmp/confirm2_2026-07-27/real_gif_info.json."""
import os, sys, json
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import numpy as np

INFO = 'tmp/confirm2_2026-07-27/real_gif_info.json'
SCAN = 'tmp/real_gif_2026-07-27/scan.jsonl'
WPREAL = 'tmp/wp_real_2026-07-25/wp_real.json'

info = json.load(open(INFO))
scan = [json.loads(l) for l in open(SCAN)]
wp = {(r['record'], r['win']): r for r in json.load(open(WPREAL))['results']}

cand = []
for r in sorted(scan, key=lambda r: -r['nowcast_corr']):
    h = wp[(r['record'], r['win'])]
    cand.append(dict(record=r['record'], win=r['win'],
                     deployed_nowcast_nrmse=round(r['nowcast_nrmse'], 4),
                     deployed_nowcast_corr=round(r['nowcast_corr'], 4),
                     deployed_lead_mean_nrmse=round(r['lead_nrmse'], 4),
                     deployed_lead_mean_corr=round(r['lead_corr'], 4),
                     n_gaussians=r['n_gauss'],
                     frozen_bankmmse_hand_nowcast=round(h['hand_now'], 4),
                     frozen_learned_nowcast=round(h['learned_now'], 4)))

from d3_falcon_real_operator import RECORDS
info['record_path'] = RECORDS[info['record']]['path']
info['record_U_mps'] = RECORDS[info['record']]['U']
info['record_r_fan_m'] = list(RECORDS[info['record']]['r_fan'])

info['window_selection'] = dict(
    criterion=('among 10 candidate windows (the 8 best-conditioned windows of the frozen '
               'bank-MMSE protocol + the two YS17 windows the task suggested), the one '
               'with the highest field correlation under the DEPLOYED path, both at '
               'nowcast and averaged over the forecast leads'),
    chosen=f"{info['record']}_w{info['window']}",
    why=('AA15a w5 wins on the deployed path (corr 0.636 nowcast / 0.581 lead-mean, the '
         'lowest lead-mean nRMSE 0.806) AND is independently the best of all 30 windows '
         'in the frozen bank-MMSE real protocol (hand_now 0.461, tmp/wp_real_2026-07-25/'
         'wp_real.json) -- two independent criteria agree, so it is not a cherry-pick of '
         'the display metric alone.'),
    note_YS17=('the task suggested YS17; measured, YS17 has NO field skill under the '
               'deployed path (w1 corr -0.03, w0 corr +0.01). The YS17 record in memory '
               'is good for the ENCOUNTER-point forecast (rho ~0.7, d3_falcon_real_'
               'operator) and for the Doppler probe, not for full-field nRMSE.'),
    candidates=cand,
    scan_raw='tmp/real_gif_2026-07-27/scan.jsonl')

info['caveat_two_analytic_arms'] = (
    'The number shown in the GIF (nowcast nRMSE 0.858) is the OPERATIONAL wavesplat/ '
    'estimator (SequentialUpdater, v2_konly, N=30). It is NOT the number published for '
    'real data: every real-sea result in the program (wp_real.json, the real figures, the '
    'blind-auto record) scores the analytic arm with scripts/d3_sa_nomp.mmse_bank_estimate, '
    'which gets 0.461 on this same window. The two estimators have never been compared '
    'head-to-head on the stereo records; this run is the first such measurement and the '
    'gap (0.86 vs 0.46 on the best window) is an open item, not a claim.')

info['scripts'] = dict(
    gif='tmp/real_gif_2026-07-27/make_real_estimation_gif.py',
    scan='tmp/real_gif_2026-07-27/scan_windows.py',
    reused_from=['tmp/figs_opus_real_2026-07-26/make_real_gif.py (frame/quantise recipe)',
                 'tmp/blind_auto_real_2026-07-26/run_blind_auto.py (scene->FrameData adapter)',
                 'tmp/figs_program_2026-07-26/make_gif.py (panel styling)'])

json.dump(info, open(INFO, 'w'), indent=1)
print('updated', INFO)
for c in cand:
    print(f"  {c['record']:6s} w{c['win']} deployed now {c['deployed_nowcast_nrmse']:.3f}/"
          f"{c['deployed_nowcast_corr']:+.3f} lead {c['deployed_lead_mean_nrmse']:.3f}/"
          f"{c['deployed_lead_mean_corr']:+.3f} | frozen bank-MMSE "
          f"{c['frozen_bankmmse_hand_nowcast']:.3f}")
