"""Critic v2 label factory -- REAL labels on the dense sliding windows.

v1 failed for two identified reasons: synthetic labels were benign (no
failure-window signal) and 10 scalars were thin. v2 fixes both: labels are
computed on the ~1100 REAL dense windows (tmp/m2m7 extraction grid, the
frozen 60 judgment windows excluded there), and features keep the X
beamform maps (6x51x51) alongside the scalars for a CNN critic.

Per (window x 12 candidates): task label = encounter-track corr, features =
X maps (shared per window) + candidate scalars (band, smult, gamma,
amp_ratio, val_resid). Output: per-record npz in tmp/critic_v2/.
Training/judgment: LORO by record, judged zero-shot on the frozen 60.
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from d3_falcon_real_operator import RECORDS
from d3_sa_multirecord import MRecord
from d3_prior_net_eval import (get_net, net_gamma_mmse, DEVICE, split_masks,
                               beamform_power, KREL, NGRID)
from d3_gate_promotion_eval import window_data, RECS
from d3_gpu_mmse import mmse_gpu

OUTDIR = 'tmp/critic_v2'
BANDS = [(0.35, 2.5), (0.15, 2.5)]
SMULTS = [1.0, 3.0, 10.0]
STRIDE_S = 10.0                       # coarser than m2m7 (12 cands each)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    net = get_net('results/direction3/prior_net_l1s2bw.pt')
    for name in RECS:
        out_path = f"{OUTDIR}/{name}.npz"
        if os.path.exists(out_path):
            print(f"skip {name}")
            continue
        cfg = RECORDS[name]
        rec = MRecord(cfg['path'], stride=cfg['stride'])
        nfw = int(round(40.0 / rec.dt))
        n_lead = int(round(cfg['lead_max'] / rec.dt))
        judge = set(np.linspace(nfw + 1, rec.T_all - n_lead - 2,
                                12).astype(int))
        step = max(1, int(round(STRIDE_S / rec.dt)))
        i0s = [i for i in range(nfw + 1, rec.T_all - n_lead - 2, step)
               if i not in judge]
        Xs, feats, labels, winid = [], [], [], []
        t0c = time.time()
        for k, i0 in enumerate(i0s):
            try:
                P, z, Pq, gt_q, leads, Hs, kp, _ = window_data(
                    rec, cfg, int(i0), nfw, n_lead, k + 500)
            except Exception:
                continue
            v = np.isfinite(gt_q) & (leads <= 10.0)
            if v.sum() < 8:
                continue
            sig0 = 0.05 * Hs
            zsd = float(np.std(z))
            # shared X maps (identical recipe to net_gamma_mmse)
            m1, m2 = split_masks(P, mode='legs')
            _, _, p1 = beamform_power(P[m1], z[m1], cfg['depth'],
                                      KREL * kp, NGRID)
            _, _, p2 = beamform_power(P[m2], z[m2], cfg['depth'],
                                      KREL * kp, NGRID)
            p1 = (p1 / max(p1.max(), 1e-12)).reshape(NGRID, NGRID)
            p2 = (p2 / max(p2.max(), 1e-12)).reshape(NGRID, NGRID)
            pm = np.minimum(p1, p2)
            X = np.stack([p1, p2, pm,
                          np.log10(p1 + 1e-6) / 6 + 1,
                          np.log10(p2 + 1e-6) / 6 + 1,
                          np.log10(pm + 1e-6) / 6 + 1]).astype(np.float32)
            vi = np.arange(len(z)) % 4 == 0
            tvi = torch.tensor(vi, device=DEVICE)
            t32 = torch.tensor(P, device=DEVICE, dtype=torch.float32)
            z32 = torch.tensor(z, device=DEVICE, dtype=torch.float32)
            q32 = torch.tensor(Pq, device=DEVICE, dtype=torch.float32)
            for band in BANDS:
                for sm in SMULTS:
                    for g in ('hand', 'net'):
                        if g == 'hand':
                            with torch.no_grad():
                                pr = mmse_gpu(t32, z32, cfg['depth'], q32,
                                              sig0, kp_hint=kp, band=band,
                                              no_em=True, pexp=1.0,
                                              smult=sm).cpu().numpy()
                                pv = mmse_gpu(t32[~tvi], z32[~tvi],
                                              cfg['depth'], t32[tvi], sig0,
                                              kp_hint=kp, band=band,
                                              no_em=True, pexp=1.0,
                                              smult=sm).cpu().numpy()
                        else:
                            pr = np.asarray(net_gamma_mmse(
                                net, P, z, cfg['depth'], Pq, sig0,
                                kp_hint=kp, use_em=False, smult_fixed=sm))
                            pv = np.asarray(net_gamma_mmse(
                                net, P[~vi], z[~vi], cfg['depth'], P[vi],
                                sig0, kp_hint=kp, use_em=False,
                                smult_fixed=sm))
                        vr = float(np.sqrt(np.mean((pv - z[vi]) ** 2))
                                   / (zsd + 1e-9))
                        cc = float(np.corrcoef(pr[v], gt_q[v])[0, 1]) \
                            if np.std(pr[v]) > 1e-12 else 0.0
                        feats.append([band[0], np.log10(sm),
                                      1.0 if g == 'net' else 0.0,
                                      min(float(np.std(pr) / (zsd + 1e-9)),
                                          5.0), min(vr, 3.0), kp * 10,
                                      zsd, Hs])
                        labels.append(cc)
                        winid.append(k)
            Xs.append(X)
            if (k + 1) % 20 == 0:
                print(f"[{name}] {k+1}/{len(i0s)} [{time.time()-t0c:.0f}s]",
                      flush=True)
        np.savez_compressed(out_path, X=np.array(Xs, np.float32),
                            feat=np.array(feats, np.float32),
                            label=np.array(labels, np.float32),
                            winid=np.array(winid, np.int32))
        print(f"[{name}] saved {len(Xs)} windows x 12 -> {out_path} "
              f"[{time.time()-t0c:.0f}s]", flush=True)
    print("CRITIC_V2_LABELS_DONE")
    return 0


if __name__ == '__main__':
    sys.exit(main())
