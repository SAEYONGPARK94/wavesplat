"""Soft-blend cell + track-RMSE audit on the frozen 60 real windows.

Pre-registration: docs/SOFTBLEND_prereg_2026-07-12.md (frozen before run).
Rebuilds the gate_promotion_eval windows verbatim, computes hand and the
incumbent learned arm (bw_s0 + SmultHead) at the protocol query points,
then judges the frozen gates for blend50 and runs the RMSE audit.
All blend/oracle arms are deterministic functions of the two stored
prediction vectors -- no learning, no selection leakage.
"""
import json
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
from d3_prior_net_eval import get_net, net_gamma_mmse, DEVICE
from d3_gate_promotion_eval import window_data
from d3_l1_s2head_train import SmultHead
from d3_gpu_mmse import mmse_gpu

OUT = 'results/direction3/softblend_rmse_eval.json'
RECS = ['YS17', 'AA15a', 'AA14', 'AA15b', 'LJ18g']
REF = {(r['record'], r['wi']): r for r in
       json.load(open('results/direction3/gate_promotion_eval.json'))}


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def nrmse(p, t):
    return float(np.sqrt(np.mean((p - t) ** 2) / np.mean(t ** 2)))


def main():
    net = get_net('results/direction3/prior_net_l1s2bw.pt')
    head = SmultHead().to(DEVICE)
    head.load_state_dict(torch.load(
        'results/direction3/prior_net_l1s2bw.pt', map_location=DEVICE)['head'])
    head.eval()

    rows = []
    for name in RECS:
        cfg = RECORDS[name]
        rec = MRecord(cfg['path'], stride=cfg['stride'])
        nfw = int(round(40.0 / rec.dt))
        n_lead = int(round(cfg['lead_max'] / rec.dt))
        i0s = np.linspace(nfw + 1, rec.T_all - n_lead - 2, 12).astype(int)
        for wi, i0 in enumerate(i0s):
            t_st = time.time()
            P, z, Pq, gt_q, leads, Hs, kp, _ = window_data(
                rec, cfg, int(i0), nfw, n_lead, wi)
            v = np.isfinite(gt_q) & (leads <= 10.0)
            if v.sum() < 5:
                continue
            sig0 = 0.05 * Hs
            with torch.no_grad():
                ph = mmse_gpu(
                    torch.tensor(P, device=DEVICE, dtype=torch.float32),
                    torch.tensor(z, device=DEVICE, dtype=torch.float32),
                    cfg['depth'],
                    torch.tensor(Pq, device=DEVICE, dtype=torch.float32),
                    sig0, kp_hint=kp).cpu().numpy()
            pn = np.asarray(net_gamma_mmse(net, P, z, cfg['depth'], Pq, sig0,
                                           seed=int(i0), kp_hint=kp,
                                           use_em=False, smult_head=head))
            t = gt_q[v]
            h, n_ = ph[v], pn[v]
            row = {'record': name, 'wi': wi,
                   'gt': t.tolist(), 'pred_hand': h.tolist(),
                   'pred_learned': n_.tolist()}
            # sanity vs frozen reference
            ref = REF.get((name, wi))
            row['corr_hand'] = corr(h, t)
            row['corr_learned'] = corr(n_, t)
            if ref is not None:
                assert abs(row['corr_hand'] - ref['hand']) < 0.02, \
                    (name, wi, row['corr_hand'], ref['hand'])
                assert abs(row['corr_learned'] - ref['bw_s0_head']) < 0.02
            for w_ in (0.25, 0.5, 0.75):
                b = w_ * h + (1 - w_) * n_
                row[f'corr_b{int(w_*100)}'] = corr(b, t)
                row[f'nrmse_b{int(w_*100)}'] = nrmse(b, t)
            row['nrmse_hand'] = nrmse(h, t)
            row['nrmse_learned'] = nrmse(n_, t)
            # amplitude ratios (a in nRMSE^2 = 1 + a^2 - 2 a rho)
            row['a_hand'] = float(np.sqrt(np.mean(h**2) / np.mean(t**2)))
            row['a_learned'] = float(np.sqrt(np.mean(n_**2) / np.mean(t**2)))
            # accounting oracles (diagnostic only)
            row['corr_oracle_select'] = max(row['corr_hand'],
                                            row['corr_learned'])
            ws = np.linspace(0, 1, 51)
            cb = [corr(w_ * h + (1 - w_) * n_, t) for w_ in ws]
            row['corr_oracle_blend'] = float(np.max(cb))
            row['w_oracle'] = float(ws[int(np.argmax(cb))])
            for arm, p in (('hand', h), ('learned', n_)):
                a_star = float(np.dot(p, t) / max(np.dot(p, p), 1e-12))
                row[f'nrmse_ampopt_{arm}'] = nrmse(a_star * p, t)
                row[f'ampopt_{arm}'] = a_star
            rows.append(row)
            print(f"[{name} w{wi}] h {row['corr_hand']:+.2f} "
                  f"l {row['corr_learned']:+.2f} b50 {row['corr_b50']:+.2f} | "
                  f"nrmse h {row['nrmse_hand']:.2f} l {row['nrmse_learned']:.2f} "
                  f"b50 {row['nrmse_b50']:.2f} [{time.time()-t_st:.0f}s]",
                  flush=True)

    json.dump(rows, open(OUT, 'w'), indent=1)
    print(f"\nwrote {OUT}  (n={len(rows)})")

    # ---- frozen gates ----
    ch = np.array([r['corr_hand'] for r in rows])
    cl = np.array([r['corr_learned'] for r in rows])
    cb50 = np.array([r['corr_b50'] for r in rows])
    nh = np.array([r['nrmse_hand'] for r in rows])
    nl = np.array([r['nrmse_learned'] for r in rows])
    nb50 = np.array([r['nrmse_b50'] for r in rows])
    good = ch >= 0.5
    d = cb50 - cl
    se = d.std(ddof=1) / np.sqrt(len(d))
    t_ = d.mean() / se
    recs_arr = np.array([r['record'] for r in rows])
    rec_worst = min(d[recs_arr == nm].mean() for nm in RECS)
    dn = nb50 - nl

    g1 = d.mean() >= 0.03 and t_ >= 2.5
    g2 = d[good].mean() >= -0.01 and rec_worst >= -0.05
    g3 = dn.mean() <= 0.005
    print(f"\nG-SB1 VALUE  {'PASS' if g1 else 'FAIL'}  "
          f"dcorr(b50-learned) {d.mean():+.3f} t {t_:+.2f}")
    print(f"G-SB2 HARM   {'PASS' if g2 else 'FAIL'}  "
          f"good-win {d[good].mean():+.3f}, worst-record {rec_worst:+.3f}")
    print(f"G-SB3 RMSE   {'PASS' if g3 else 'FAIL'}  "
          f"dnrmse(b50-learned) {dn.mean():+.4f}")

    # ---- RMSE audit (descriptive, questions frozen in prereg) ----
    bad = ~good
    print(f"\n--- RMSE audit ---")
    print(f"track nRMSE mean: hand {nh.mean():.3f}  learned {nl.mean():.3f} "
          f"(paired d {(nl-nh).mean():+.3f}, learned wins "
          f"{(nl<nh).sum()}/{len(nh)})")
    print(f"  good windows: hand {nh[good].mean():.3f} learned "
          f"{nl[good].mean():.3f} (learned wins {(nl<nh)[good].sum()}"
          f"/{good.sum()})")
    print(f"  bad windows : hand {nh[bad].mean():.3f} learned "
          f"{nl[bad].mean():.3f} (learned wins {(nl<nh)[bad].sum()}"
          f"/{bad.sum()})")
    cw = (cl > ch) & (nl > nh)
    print(f"corr-win but RMSE-loss windows (learned): {cw.sum()}/{len(rows)}")
    ah = np.array([r['a_hand'] for r in rows])
    al = np.array([r['a_learned'] for r in rows])
    print(f"amplitude ratio a: hand {ah.mean():.2f}  learned {al.mean():.2f}")
    noh = np.array([r['nrmse_ampopt_hand'] for r in rows])
    nol = np.array([r['nrmse_ampopt_learned'] for r in rows])
    print(f"amp-oracle nRMSE:  hand {noh.mean():.3f}  learned {nol.mean():.3f}"
          f"  (recoverable: hand {nh.mean()-noh.mean():+.3f}, "
          f"learned {nl.mean()-nol.mean():+.3f})")
    osel = np.array([r['corr_oracle_select'] for r in rows])
    obl = np.array([r['corr_oracle_blend'] for r in rows])
    print(f"corr accounting: learned {cl.mean():+.3f}  b50 {cb50.mean():+.3f}"
          f"  oracle-select {osel.mean():+.3f}  oracle-blend {obl.mean():+.3f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
