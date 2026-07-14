"""E-11: can the estimator get its own hints from the 600 points it is allowed?

Prereg: docs/E10_PATTERN_prereg_2026-07-13.md §5 (registered same round).

The real-data headline (+0.033, 60 windows) is measured at 600 observations per
window -- already the sparse budget.  But the estimator does not only receive
those 600 points.  It also receives kp, the peak direction, and Hs, and those are
extracted from the DENSE stereo field:

    directional_spread(np.nan_to_num(Zmem[-1]), ...)   # 2-D FFT of a full frame
    resolve_direction(Zmem, ...)                       # low-pass of full frames
    Hs = 4 * np.nanstd(Zmem)                           # full memory volume

A ship carrying a LiDAR has 600 points, not a dense stereo field.  So the E-9
rule ("validation runs at the originally assumed data") was applied to the
observations and silently violated for the hints.  Both arms get the same hints,
so the learned-vs-hand delta is probably fair -- but the ABSOLUTE skill, and the
claim that this pipeline is deployable, rest on information the sensor cannot
supply.  Nobody had registered this.

Here the hints are re-derived from the 600 points alone:
  Hs   : 4 * std(z)                        -- trivial
  kp   : peak of a dispersion-folded beamform power map over the scattered points
  pdir : the direction of that peak (the space-time phase fixes the 180-flip,
         so no separate resolve_direction step is needed)

PASS  -> the pipeline is fully sensor-realizable; the paper gains a sentence.
FAIL  -> kp/pdir must be declared an EXTERNAL input (ship motion, radar, forecast)
         and written into the requirements.  Either way the paper gets more honest.
"""
import json
import os
import sys
import time

import numpy as np
from scipy import stats
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from d3_cluster_se import cluster_se
from d3_e10_pattern_real import window_data                          # noqa: E402
from d3_falcon_real_operator import RECORDS                          # noqa: E402
from d3_gate_promotion_eval import RECS                              # noqa: E402
from d3_gpu_mmse import mmse_gpu                                     # noqa: E402
from d3_l1_s2head_train import SmultHead                             # noqa: E402
from d3_prior_net_eval import DEVICE, get_net, net_gamma_mmse        # noqa: E402
from d3_sa_multirecord import MRecord                                # noqa: E402

G = 9.81
OUT = 'results/direction3/e11_sparse_characterize.json'


def sparse_hints(P, z, depth, k_lo=0.01, k_hi=0.40, nk=48, nth=72):
    """kp and peak direction from scattered (x, y, t, z) alone.

    Dispersion-folded beamforming: for each candidate (kx, ky) the frequency is
    fixed by omega = sqrt(g k tanh(k d)), so a plane wave contributes coherently
    across the whole space-time window and the 180-degree ambiguity resolves
    itself -- a wave going +x and one going -x differ in the sign of omega*t.
    """
    ks = np.linspace(k_lo, k_hi, nk)
    th = np.linspace(-np.pi, np.pi, nth, endpoint=False)
    KK, TT = np.meshgrid(ks, th, indexing='ij')
    kx, ky = (KK * np.cos(TT)).ravel(), (KK * np.sin(TT)).ravel()
    k = np.hypot(kx, ky)
    om = np.sqrt(G * k * np.tanh(k * depth))

    x = torch.tensor(P[:, 0], device=DEVICE, dtype=torch.float64)
    y = torch.tensor(P[:, 1], device=DEVICE, dtype=torch.float64)
    t = torch.tensor(P[:, 2] - P[:, 2].mean(), device=DEVICE,
                     dtype=torch.float64)
    zz = torch.tensor(z - z.mean(), device=DEVICE, dtype=torch.float64)

    KX = torch.tensor(kx, device=DEVICE, dtype=torch.float64)[:, None]
    KY = torch.tensor(ky, device=DEVICE, dtype=torch.float64)[:, None]
    OM = torch.tensor(om, device=DEVICE, dtype=torch.float64)[:, None]

    ph = KX * x[None, :] + KY * y[None, :] - OM * t[None, :]
    c = (torch.cos(ph) * zz[None, :]).sum(1)
    s = (torch.sin(ph) * zz[None, :]).sum(1)
    pw = (c ** 2 + s ** 2).cpu().numpy()

    i = int(np.argmax(pw))
    kp = float(k[i])
    pdir = float(np.degrees(np.arctan2(ky[i], kx[i])))
    return kp, pdir, float(pw[i] / pw.mean())


def main():
    net = get_net('results/direction3/prior_net_l1s2bw.pt')
    head = SmultHead().to(DEVICE)
    head.load_state_dict(torch.load('results/direction3/prior_net_l1s2bw.pt',
                                    map_location=DEVICE)['head'])
    head.eval()

    rows = []
    for name in RECS:
        cfg = RECORDS[name]
        rec = MRecord(cfg['path'], stride=cfg['stride'])
        nfw = int(round(40.0 / rec.dt))
        n_lead = int(round(cfg['lead_max'] / rec.dt))
        i0s = np.linspace(nfw + 1, rec.T_all - n_lead - 2, 12).astype(int)
        for wi, i0 in enumerate(i0s):
            t0 = time.time()
            # the uniform 600-obs window, identical to the headline arm
            P, z, Pq, gt_q, leads, Hs_dense, kp_dense = window_data(
                rec, cfg, int(i0), nfw, n_lead, wi, 'S0_uniform')
            v = np.isfinite(gt_q) & (leads <= 10.0)
            if v.sum() < 5:
                continue

            Hs_sp = 4.0 * float(np.std(z))
            kp_sp, pdir_sp, snr = sparse_hints(P, z, cfg['depth'])

            row = dict(record=name, wi=wi, n_obs=int(len(z)),
                       kp_dense=float(kp_dense), kp_sparse=kp_sp,
                       Hs_dense=float(Hs_dense), Hs_sparse=Hs_sp,
                       beamform_snr=snr)
            for tag, kp_, Hs_ in (('dense_hint', kp_dense, Hs_dense),
                                  ('sparse_hint', kp_sp, Hs_sp)):
                sig0 = 0.05 * Hs_
                with torch.no_grad():
                    ph = mmse_gpu(
                        torch.tensor(P, device=DEVICE, dtype=torch.float32),
                        torch.tensor(z, device=DEVICE, dtype=torch.float32),
                        cfg['depth'],
                        torch.tensor(Pq, device=DEVICE, dtype=torch.float32),
                        sig0, kp_hint=kp_).cpu().numpy()
                    pn = np.asarray(net_gamma_mmse(
                        net, P, z, cfg['depth'], Pq, sig0, seed=int(i0),
                        kp_hint=kp_, use_em=False, smult_head=head))
                hh = float(np.corrcoef(ph[v], gt_q[v])[0, 1])
                nn = float(np.corrcoef(pn[v], gt_q[v])[0, 1])
                row[tag] = dict(hand=hh, learned=nn, delta=nn - hh)
            rows.append(row)
            print(f'[{name} w{wi}] kp {kp_dense:.4f}->{kp_sp:.4f}  '
                  f'dense d={row["dense_hint"]["delta"]:+.3f} '
                  f'(skill {row["dense_hint"]["hand"]:+.2f})  '
                  f'sparse d={row["sparse_hint"]["delta"]:+.3f} '
                  f'(skill {row["sparse_hint"]["hand"]:+.2f})  '
                  f'[{time.time() - t0:.0f}s]', flush=True)

    # E-12e (2026-07-14): this used se = std/sqrt(n) over 60 WINDOWS and a 1.96
    # critical value. The windows are clustered in 5 records (they share the sea,
    # the reconstruction and the ship track), so the iid SE is not valid, and on
    # G clusters the reference distribution is t(G-1), not the normal.
    # Re-judged: sparse_hint delta +0.0958, CR-SE 0.0135, t 7.10, p 0.0021 --
    # E-11 SURVIVES (the cluster SE is *smaller* than the iid one here).
    g = [r['record'] for r in rows]
    summ = {}
    for tag in ('dense_hint', 'sparse_hint'):
        d = np.array([r[tag]['delta'] for r in rows])
        se_iid = float(d.std(ddof=1) / np.sqrt(len(d)))
        se, n_clu = cluster_se(d, g)
        tcrit = float(stats.t.ppf(0.975, n_clu - 1))
        summ[tag] = dict(
            n=len(d), n_clusters=n_clu, delta=float(d.mean()),
            se=se, se_iid=se_iid, t=float(d.mean() / se),
            p=float(2 * stats.t.sf(abs(d.mean() / se), n_clu - 1)),
            ci_lo=float(d.mean() - tcrit * se),
            ci_hi=float(d.mean() + tcrit * se),
            hand=float(np.mean([r[tag]['hand'] for r in rows])),
            learned=float(np.mean([r[tag]['learned'] for r in rows])))
    kpe = np.array([abs(r['kp_sparse'] - r['kp_dense']) / r['kp_dense']
                    for r in rows])

    res = dict(date='2026-07-13',
               prereg='docs/E10_PATTERN_prereg_2026-07-13.md',
               question='are the estimator hints (kp, pdir, Hs) recoverable from '
                        'the 600 points the sensor actually delivers?',
               summary=summ,
               kp_rel_err=dict(mean=float(kpe.mean()), median=float(np.median(kpe)),
                               p90=float(np.quantile(kpe, 0.9))),
               windows=rows)
    json.dump(res, open(OUT, 'w'), indent=1)

    print(f'\n{"=" * 74}\n[E-11] hints from the dense stereo field vs from the '
          f'600 points\n')
    print(f'{"hints from":14s} {"hand":>7} {"learned":>8} {"delta":>9} '
          f'{"CI lo":>8} {"t":>6}')
    for tag in ('dense_hint', 'sparse_hint'):
        s = summ[tag]
        print(f'{tag:14s} {s["hand"]:7.3f} {s["learned"]:8.3f} '
              f'{s["delta"]:+9.4f} {s["ci_lo"]:+8.4f} {s["t"]:6.2f}')
    print(f'\nkp recovery error |dk|/k: mean {kpe.mean():.3f}, '
          f'median {np.median(kpe):.3f}, p90 {np.quantile(kpe, 0.9):.3f}')
    print(f'-> {OUT}')


if __name__ == '__main__':
    main()
