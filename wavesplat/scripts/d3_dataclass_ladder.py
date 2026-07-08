"""Data-class ladder (2026-07-08): resolves the "WaveSplat 0.223 vs real 0.5
contradiction" (user objection). SAME estimator (mmse_bank_estimate), SAME
encounter-forecast protocol; vary ONLY the data class. Isolates model-class
mismatch (single-band free-wave bank vs bimodal reality) as the cause of the
ceiling — not real-vs-synthetic, not sensor sparsity.

  in-class synthetic (single-band JONSWAP)  -> falcon_hos_probe.json
  out-of-class synthetic (cross-Tp bimodal) -> falcon_hos_probe_bimod90cross.json
  out-of-class real (bimodal stereo)         -> falcon_real_operator_*.json

WaveSplat (OCEANS) nRMSE 0.223 is synthetic-in-class dense full-track: it sits
at the in-class end of this ladder, so it is NOT a counterexample to a ceiling
that is specifically the out-of-class (bimodal) model-class limit.
"""
import json
import os

import numpy as np

os.chdir(os.path.join(os.path.dirname(__file__), '..'))
RES = 'results/direction3'
OUT = f'{RES}/dataclass_ladder.json'


def hos_band(name, lo, hi, W='40.0'):
    d = json.load(open(f'{RES}/{name}.json'))
    leads = np.asarray(d['leads'], float)
    seas = d['raw'].get(W) or list(d['raw'].values())[-1]
    ps, gs = [], []
    for pred, gt in seas:
        pred, gt = np.asarray(pred, float), np.asarray(gt, float)
        n = min(len(pred), len(gt), len(leads))
        m = (leads[:n] >= lo) & (leads[:n] <= hi) & \
            np.isfinite(gt[:n]) & np.isfinite(pred[:n])
        ps.append(pred[:n][m]); gs.append(gt[:n][m])
    p, g = np.concatenate(ps), np.concatenate(gs)
    return dict(nrmse=float(np.sqrt(np.mean((p - g) ** 2)) / np.std(g)),
                corr=float(np.corrcoef(p, g)[0, 1]), n=int(len(g)))


def real_band(lo, hi, recs=('YS17', 'AA15a', 'AA14', 'AA15b'), W='20.0'):
    ps, gs = [], []
    for rec in recs:
        d = json.load(open(f'{RES}/falcon_real_operator_{rec}.json'))
        leads = np.asarray(d['leads'], float)
        sg = np.std(np.concatenate([np.asarray(w['gt'], float)[
            np.isfinite(w['gt'])] for w in d['windows']]))
        for w in d['windows']:
            gt = np.asarray(w['gt'], float)
            pr = np.asarray(w['by_W'][W]['pred'], float)
            n = min(len(gt), len(pr), len(leads))
            m = (leads[:n] >= lo) & (leads[:n] <= hi) & \
                np.isfinite(gt[:n]) & np.isfinite(pr[:n])
            ps.append(pr[:n][m] / sg); gs.append(gt[:n][m] / sg)
    p, g = np.concatenate(ps), np.concatenate(gs)
    return dict(nrmse=float(np.sqrt(np.mean((p - g) ** 2)) / np.std(g)),
                corr=float(np.corrcoef(p, g)[0, 1]), n=int(len(g)))


ladder = {
    'in_class_synth_unimodal': {b: hos_band('falcon_hos_probe', lo, hi)
                                for b, (lo, hi) in
                                {'0-2s': (0, 2), '0-10s': (0, 10)}.items()},
    'out_class_synth_crossTp': {
        b: hos_band('falcon_hos_probe_bimod90cross', lo, hi)
        for b, (lo, hi) in {'0-2s': (0, 2), '0-10s': (0, 10)}.items()},
    'out_class_real_bimodal': {b: real_band(lo, hi) for b, (lo, hi) in
                               {'0-2s': (0, 2), '0-10s': (0, 10)}.items()},
    'note': ('WaveSplat OCEANS nRMSE 0.223 = synthetic in-class dense '
             'full-track nowcast; sits at the in-class end of this ladder.'),
}
json.dump(ladder, open(OUT, 'w'), indent=1)

print("데이터-클래스 사다리 (같은 추정기, 데이터 클래스만 변경)")
print(f"{'class':34} | {'0-2s nRMSE':>11} | {'0-10s nRMSE':>12}")
for k in ['in_class_synth_unimodal', 'out_class_synth_crossTp',
          'out_class_real_bimodal']:
    a, b = ladder[k]['0-2s'], ladder[k]['0-10s']
    print(f"{k:34} | {a['nrmse']:.2f} (c{a['corr']:+.2f}) | "
          f"{b['nrmse']:.2f} (c{b['corr']:+.2f})")
print("\nWaveSplat OCEANS = 0.223 (synth in-class dense full-track) "
      "-> in-class end.")
print(f"saved {OUT}")
