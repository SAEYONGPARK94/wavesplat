"""E-12d — does the real-data learning gain survive WITHOUT the dense-GT hint?

Pre-registered in docs/E12_KP_FLOOR_AUDIT_prereg_2026-07-14.md section 7.

Why E-12b did not answer this (see section 6):
  The frozen protocol hands both MMSE arms kp from characterize(Zmem[-1]) --
  the spatial FFT of the DENSE stereo field the estimator is supposed to
  reconstruct from 600 sparse points. That is not a handicap, it is a
  GT-derived SIDE CHANNEL. And the net was trained on kp from a two-stage
  beamform zoom over the SPARSE observations (d3_prior_net_data.py:132,
  "exactly as at inference time"), so the dense hint is also out of its
  training distribution.

  E-12b swapped one dense-derived hint (spatial) for another (causal). The arm
  that is BOTH training-consistent AND sensor-realizable was missing.

Design: identical 60 windows, identical observations (P, z), identical seeds.
The only thing that changes is the kp handed to the two estimators:

    spatial : directional_spread(Zmem[-1])   -> dense GT FFT   [frozen baseline]
    sparse  : two-stage beamform_top(P, z)   -> 600 sparse obs [PRIMARY]

The sparse kp is computed ONCE per window and given to BOTH arms, so the pairing
is preserved. It is the same deterministic recipe net_gamma_mmse runs internally
when kp_hint=None, and the same one the net was trained with.

GATES (fixed before running):
  PASS      delta_sparse >= +0.020 AND cluster-robust 95% CI excludes 0
  DEMOTED   significant but < +0.017
  FAIL      CI includes 0 -> withdraw the real-data learning-gain claim
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from d3_falcon_real_operator import RECORDS      # noqa: E402
from d3_sa_multirecord import MRecord            # noqa: E402
from d3_prior_net_eval import get_net, DEVICE    # noqa: E402
from d3_l1_s2head_train import SmultHead         # noqa: E402
from d3_external_figs import Window              # noqa: E402
from d3_cluster_se import cluster_se             # noqa: E402
from d18_relax_estimator import beamform_top     # noqa: E402

OUT = 'results/direction3/e12d_sparse_hint.json'
RECS = ['YS17', 'AA15a', 'AA14', 'AA15b', 'LJ18g']
LEADS = [0.0, 2.0, 5.0, 8.0]
CKPT = 'results/direction3/prior_net_l1s2bw.pt'


def kp_sparse(P, z, depth):
    """The estimator's own kp: two-stage beamform zoom on the SPARSE obs.

    Byte-for-byte the recipe in d3_prior_net_data.py:132 (training) and the
    kp_hint=None branch of d3_prior_net_eval.py:71 (inference). Deterministic.
    """
    Pf, zf = P.astype(np.float32), z.astype(np.float32)
    bx, by = beamform_top(Pf, zf, depth, klim=2.0, n=120)
    kp = max(float(np.hypot(bx, by)), 1e-3)
    bx, by = beamform_top(Pf, zf, depth, klim=3.5 * kp, n=120)
    return max(float(np.hypot(bx, by)), 1e-3)


def score(w, leads):
    gx, gy, tr, fh, fn = w.fields(leads, stride=2)
    trs, fhs, fns = np.array(tr), np.array(fh), np.array(fn)
    m = np.isfinite(trs)
    if m.sum() < 500:
        return None
    return (float(np.corrcoef(fhs[m], trs[m])[0, 1]),
            float(np.corrcoef(fns[m], trs[m])[0, 1]), int(m.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_win', type=int, default=12)
    ap.add_argument('--records', nargs='+', default=RECS)
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()

    net = get_net(CKPT)
    head = SmultHead().to(DEVICE)
    head.load_state_dict(torch.load(CKPT, map_location=DEVICE)['head'])
    head.eval()

    rows = []
    for name in a.records:
        cfg = RECORDS[name]
        rec = MRecord(cfg['path'], stride=cfg['stride'])
        for wi in range(a.n_win):
            t0 = time.time()
            w = Window(name, wi, rec=rec, net=net, head=head)
            leads = [l_ for l_ in LEADS if l_ <= min(10.0, w.leads.max())]

            kp_sp = float(w.kp)                                # dense GT FFT
            kp_bf = kp_sparse(w.P, w.z, cfg['depth'])          # 600 sparse obs

            w.kp = kp_sp
            s_sp = score(w, leads)
            w.kp = kp_bf
            s_bf = score(w, leads)
            if s_sp is None or s_bf is None:
                print(f'[{name} w{wi}] too few finite px — skipped')
                continue

            row = dict(record=name, wi=wi, n_px=s_sp[2],
                       kp_spatial=kp_sp, kp_sparse=kp_bf,
                       mismatch=kp_sp / kp_bf,
                       hand_spatial=s_sp[0], learned_spatial=s_sp[1],
                       hand_sparse=s_bf[0], learned_sparse=s_bf[1])
            row['d_spatial'] = row['learned_spatial'] - row['hand_spatial']
            row['d_sparse'] = row['learned_sparse'] - row['hand_sparse']
            rows.append(row)
            print(f"[{name} w{wi}] kp dense {kp_sp:.3f} / sparse {kp_bf:.3f} "
                  f"(x{kp_sp/kp_bf:.2f}) | SPATIAL d {row['d_spatial']:+.3f} "
                  f"(h {s_sp[0]:+.3f} n {s_sp[1]:+.3f}) | SPARSE d "
                  f"{row['d_sparse']:+.3f} (h {s_bf[0]:+.3f} n {s_bf[1]:+.3f})"
                  f" [{time.time()-t0:.0f}s]", flush=True)
        rec.f.close()

    g = [r['record'] for r in rows]
    print(f"\n{'='*78}\nE-12d  (n={len(rows)} windows, {len(set(g))} record clusters)\n{'='*78}")
    print(f"{'arm':<10}{'hand':>9}{'net':>9}{'delta':>10}{'CR-SE':>9}"
          f"{'t_cl':>7}{'95% CI':>22}{'net wins':>10}")
    res = {}
    for mode in ('spatial', 'sparse'):
        h = np.array([r[f'hand_{mode}'] for r in rows])
        n_ = np.array([r[f'learned_{mode}'] for r in rows])
        d = n_ - h
        se, G = cluster_se(d, g)
        lo, hi = d.mean() - 1.96 * se, d.mean() + 1.96 * se
        res[mode] = dict(hand=float(h.mean()), net=float(n_.mean()),
                         delta=float(d.mean()), se_cluster=float(se),
                         n_clusters=G, ci=[float(lo), float(hi)],
                         t_cluster=float(d.mean() / se),
                         wins=int((d > 0).sum()), n=len(d))
        print(f"{mode:<10}{h.mean():+9.3f}{n_.mean():+9.3f}{d.mean():+10.4f}"
              f"{se:9.4f}{d.mean()/se:7.2f}"
              f"    [{lo:+.4f}, {hi:+.4f}]{f'{(d>0).sum()}/{len(d)}':>10}")

    # per-record deltas on the primary arm
    print(f"\nper-record delta (sparse arm):")
    for nm in RECS:
        dd = [r['d_sparse'] for r in rows if r['record'] == nm]
        ds = [r['d_spatial'] for r in rows if r['record'] == nm]
        if dd:
            print(f"   {nm:<7} sparse {np.mean(dd):+.4f}   "
                  f"(spatial {np.mean(ds):+.4f})   n={len(dd)}")

    mm = np.array([r['mismatch'] for r in rows])
    print(f"\ndiagnostic — dense/sparse kp mismatch: median {np.median(mm):.2f}x, "
          f"range [{mm.min():.2f}, {mm.max():.2f}]")

    ds_ = res['sparse']
    verdict = ('FAIL — CI includes 0; withdraw the real-data learning-gain claim'
               if ds_['ci'][0] <= 0 <= ds_['ci'][1] else
               'PASS — gain survives with a sensor-realizable, training-consistent hint'
               if ds_['delta'] >= 0.020 else
               'DEMOTED — significant but < half the frozen point estimate')
    print(f"\n*** E-12d VERDICT: {verdict}")
    print(f"    sparse delta {ds_['delta']:+.4f}, CR-95% CI "
          f"[{ds_['ci'][0]:+.4f}, {ds_['ci'][1]:+.4f}], t {ds_['t_cluster']:+.2f}")
    res['verdict'] = verdict
    json.dump({'rows': rows, 'summary': res}, open(a.out, 'w'), indent=1)
    print(f"\n-> {a.out}")


if __name__ == '__main__':
    sys.exit(main())
