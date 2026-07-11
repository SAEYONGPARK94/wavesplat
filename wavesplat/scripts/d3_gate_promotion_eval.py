"""Promotion gates for the G-REAL claim -- 5 records x 12 windows, 7 arms.

The 2026-07-11 night result (l1s2bw beats hand on hand-bad real windows,
+0.192 t 3.76 on n=7) is promoted to a claim ONLY through these gates.

================================ PRE-REGISTRATION ============================
Frozen before the first run of this script. 60 windows = 5 records
(YS17, AA15a, AA14, AA15b, LJ18g) x 12, the d3_netfan_eval window protocol
verbatim. All arms computed fresh and identically (the 'hand' arm here is
the production mmse_gpu grid+EM path -- disclosed: this differs from July's
val_gate_probe hand; the two are cross-checked on the original 24 windows).
Bad windows := hand < 0.5 (the pre-existing definition).

  G-P1 VALUE.        pooled bad-window paired d(l1s2bw_s0 - hand):
                     mean >= +0.10 AND t >= 2.5.
  G-P2 SEED.         bad-window mean d > 0 for ALL of seeds {0,1,2}.
  G-P3 ABLATION.     cleanmatch net (same n=8000, same fan mix 0.57, zero
                     bw-injection, same training) bad-window gain
                     < 50% of the bw-trained mean -- else the
                     "misspecification injection" mechanism is REJECTED
                     (volume/mix was the cause) and the claim is re-worded.
  G-P4 HEAD.         l1s2bw_s0 + SmultHead >= (l1s2bw_s0 val-select) - 0.01
                     on the pooled mean (do-no-harm; superiority not
                     required at one training run).
DECISION: G-P1..P4 pass -> claim: "misspecification-trained amortized priors
rescue the hand estimator's failure windows on real seas." Any FAIL is
reported as the result for that gate; the claim shrinks to what passed.
=============================================================================
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

from realdata_analysis import directional_spread
from d3_falcon_real_operator import RECORDS, bilinear, resolve_direction
from d3_sa_multirecord import MRecord
from d3_prior_net_eval import get_net, net_gamma_mmse, DEVICE
from d3_l1_s2head_train import SmultHead
from d3_gpu_mmse import mmse_gpu

G = 9.81
W = 20.0
OUT = 'results/direction3/gate_promotion_eval.json'
RECS = ['YS17', 'AA15a', 'AA14', 'AA15b', 'LJ18g']


def window_data(rec, cfg, i0, nfw, n_lead, wi):
    """verbatim d3_netfan_eval observation/query construction."""
    rng = np.random.default_rng(700 + wi)
    Zmem = rec.window(i0 - nfw + 1, nfw)
    Zfut = rec.window(i0, n_lead + 1)
    t0 = float(rec.t[i0])
    Hs = 4.0 * float(np.nanstd(Zmem))
    spread, lam_p, pdir_ax, kp = directional_spread(
        np.nan_to_num(Zmem[-1]), rec.dx, rec.dy)
    om_p = np.sqrt(G * kp * np.tanh(kp * cfg['depth']))
    pdir, _ = resolve_direction(Zmem, rec, kp, pdir_ax, om_p / kp)
    h = -np.array([np.cos(np.radians(pdir)), np.sin(np.radians(pdir))])
    cx = 0.5 * (rec.gx.min() + rec.gx.max())
    cy = 0.5 * (rec.gy.min() + rec.gy.max())
    Xg, Yg = np.meshgrid(rec.gx, rec.gy)
    W_max = 40.0
    T_tr = W_max + cfg['lead_max']
    U = cfg['U']

    def ship_pos(t):
        return np.array([cx, cy]) + U * (t - (t0 - W_max) - T_tr / 2) * h

    r0, r1 = cfg['r_fan']
    P_all, z_all = [], []
    for j in range(nfw):
        t_j = float(rec.t[i0 - nfw + 1 + j])
        sp = ship_pos(t_j)
        RX, RY = Xg - sp[0], Yg - sp[1]
        R = np.hypot(RX, RY)
        ca = (RX * h[0] + RY * h[1]) / np.maximum(R, 1e-9)
        m = ((R >= r0) & (R <= r1) & (ca >= 0.5) & np.isfinite(Zmem[j]))
        idx = np.flatnonzero(m)
        if len(idx) == 0:
            continue
        pick = rng.choice(idx, min(12, len(idx)), replace=False)
        iy, ix = np.unravel_index(pick, Zmem[j].shape)
        P_all.append(np.column_stack([rec.gx[ix], rec.gy[iy],
                                      np.full(len(pick), t_j)]))
        z_all.append(Zmem[j][iy, ix] + rng.normal(0, 0.02, len(pick)))
    P_all, z_all = np.vstack(P_all), np.concatenate(z_all)
    m = P_all[:, 2] >= t0 - W
    P, z = P_all[m], z_all[m]
    if len(z) > 600:
        ii = np.random.default_rng(0).choice(len(z), 600, False)
        P, z = P[ii], z[ii]
    leads = np.arange(0.0, cfg['lead_max'] + 1e-9, 1.0)
    Pq = np.array([[*ship_pos(t0 + l_), t0 + l_] for l_ in leads])
    gt_q = np.array([bilinear(Zfut[int(round(l_ / rec.dt))],
                              rec.gx, rec.gy, px, py)
                     for (px, py, _t), l_ in zip(Pq, leads)])
    return P, z, Pq, gt_q, leads, Hs, kp, i0


def main():
    nets = {'v3': get_net('results/direction3/prior_net_v3.pt'),
            'bw_s0': get_net('results/direction3/prior_net_l1s2bw.pt'),
            'bw_s1': get_net('results/direction3/prior_net_l1s2bw_s1.pt'),
            'bw_s2': get_net('results/direction3/prior_net_l1s2bw_s2.pt'),
            'clean': get_net('results/direction3/prior_net_l1s2clean.pt')}
    head = SmultHead().to(DEVICE)
    hck = torch.load('results/direction3/prior_net_l1s2bw.pt',
                     map_location=DEVICE)
    head.load_state_dict(hck['head'])
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
            P, z, Pq, gt_q, leads, Hs, kp, _ = window_data(
                rec, cfg, int(i0), nfw, n_lead, wi)
            v = np.isfinite(gt_q) & (leads <= 10.0)
            if v.sum() < 5:
                continue
            sig0 = 0.05 * Hs
            out = {'record': name, 'wi': wi}
            with torch.no_grad():
                ph = mmse_gpu(torch.tensor(P, device=DEVICE,
                                           dtype=torch.float32),
                              torch.tensor(z, device=DEVICE,
                                           dtype=torch.float32),
                              cfg['depth'],
                              torch.tensor(Pq, device=DEVICE,
                                           dtype=torch.float32),
                              sig0, kp_hint=kp).cpu().numpy()
            out['hand'] = float(np.corrcoef(ph[v], gt_q[v])[0, 1])
            for nm, net in nets.items():
                fq = net_gamma_mmse(net, P, z, cfg['depth'], Pq, sig0,
                                    seed=int(i0), kp_hint=kp, use_em=False)
                out[nm] = float(np.corrcoef(fq[v], gt_q[v])[0, 1])
            fq = net_gamma_mmse(nets['bw_s0'], P, z, cfg['depth'], Pq, sig0,
                                seed=int(i0), kp_hint=kp, use_em=False,
                                smult_head=head)
            out['bw_s0_head'] = float(np.corrcoef(fq[v], gt_q[v])[0, 1])
            rows.append(out)
            print(f"[{name} w{wi}] hand {out['hand']:+.2f} v3 {out['v3']:+.2f}"
                  f" s0 {out['bw_s0']:+.2f} s1 {out['bw_s1']:+.2f}"
                  f" s2 {out['bw_s2']:+.2f} head {out['bw_s0_head']:+.2f}"
                  f" clean {out['clean']:+.2f} [{time.time()-t0:.0f}s]",
                  flush=True)

    json.dump(rows, open(OUT, 'w'), indent=1)
    h = np.array([r['hand'] for r in rows])
    bad = h < 0.5
    print(f"\nn windows {len(rows)}, bad (hand<0.5) {bad.sum()}")

    def paired(key):
        d = np.array([r[key] for r in rows]) - h
        db = d[bad]
        se = db.std(ddof=1) / np.sqrt(len(db))
        return (float(d.mean()), float(db.mean()), float(se),
                float(db.mean() / se) if se > 0 else 0.0)

    stats = {k: paired(k) for k in
             ('v3', 'bw_s0', 'bw_s1', 'bw_s2', 'bw_s0_head', 'clean')}
    for k, (mall, mbad, se, t_) in stats.items():
        print(f"{k:11s} d(all) {mall:+.3f}  d(bad) {mbad:+.3f}+-{se:.3f} "
              f"t {t_:+.2f}")

    gp1 = stats['bw_s0'][1] >= 0.10 and stats['bw_s0'][3] >= 2.5
    gp2 = all(stats[k][1] > 0 for k in ('bw_s0', 'bw_s1', 'bw_s2'))
    bw_mean = np.mean([stats[k][1] for k in ('bw_s0', 'bw_s1', 'bw_s2')])
    gp3 = stats['clean'][1] < 0.5 * bw_mean
    heads = np.array([r['bw_s0_head'] for r in rows])
    base = np.array([r['bw_s0'] for r in rows])
    gp4 = float((heads - base).mean()) >= -0.01
    print(f"\nG-P1 VALUE    {'PASS' if gp1 else 'FAIL'} "
          f"(bad d {stats['bw_s0'][1]:+.3f}, t {stats['bw_s0'][3]:+.2f})")
    print(f"G-P2 SEED     {'PASS' if gp2 else 'FAIL'} "
          f"({[round(stats[k][1], 3) for k in ('bw_s0', 'bw_s1', 'bw_s2')]})")
    print(f"G-P3 ABLATION {'PASS' if gp3 else 'FAIL'} "
          f"(clean {stats['clean'][1]:+.3f} vs bw mean {bw_mean:+.3f})")
    print(f"G-P4 HEAD     {'PASS' if gp4 else 'FAIL'} "
          f"(head-base mean {float((heads-base).mean()):+.4f})")
    print(f"wrote {OUT}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
