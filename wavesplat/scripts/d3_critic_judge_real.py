"""Critic v1 REAL judgment -- the frozen gates (CRITIC_SELECTOR_DESIGN §3).

60 real windows (5 records x 12, d3_netfan_eval protocol). Per window the 12
certified candidates are run on the encounter track; the strategies compared:

  hand      mmse_gpu production default (reference)
  headfix   l1s2bw net + s2-head (yesterday's promoted arm, context)
  fix3      hand gamma, band .35, smult 3 (the distilled constant law)
  valres    argmin held-out obs residual over the 12 candidates (the
            incumbent selector, computed per candidate: refit on 75% obs,
            residual on 25%)
  critic    argmax critic prediction over the 12 candidates
  oracle    max task corr over the 12 (upper bound, GT-only)

Frozen gates: G-C1 critic >= valres + 0.05 (pooled, window-clustered t>=2.5)
G-C2 failure AUC >= 0.8 (critic's predicted corr as score for corr<0.5)
G-C3 harmless: critic never < hand - 0.01 on any record mean
G-C4 zero-consistency: on windows where candidate spread < 0.05, selection
     entropy is reported (no memorized favorite).
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
from d3_l1_s2head_train import SmultHead
from d3_gate_promotion_eval import window_data, RECS
from d3_gpu_mmse import mmse_gpu, build_bank, beamform_prior
from d3_critic_train import Critic, featvec

OUT = 'results/direction3/critic_judge_real.json'
BANDS = [(0.35, 2.5), (0.15, 2.5)]
SMULTS = [1.0, 3.0, 10.0]


def main():
    net = get_net('results/direction3/prior_net_l1s2bw.pt')
    head = SmultHead().to(DEVICE)
    head.load_state_dict(torch.load('results/direction3/prior_net_l1s2bw.pt',
                                    map_location=DEVICE)['head'])
    head.eval()
    critic = Critic().to(DEVICE)
    critic.load_state_dict(torch.load('results/direction3/critic_v1.pt',
                                      map_location=DEVICE)['model'])
    critic.eval()

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
            zsd = float(np.std(z))
            t32 = torch.tensor(P, device=DEVICE, dtype=torch.float32)
            z32 = torch.tensor(z, device=DEVICE, dtype=torch.float32)
            q32 = torch.tensor(Pq, device=DEVICE, dtype=torch.float32)
            # shared features
            half = np.arange(len(P)) % 2 == 0
            kxb, kyb, om = build_bank(kp, (0.35, 2.5), kp / 10.0,
                                      cfg['depth'], DEVICE, torch.float32)
            pw1 = beamform_prior(t32[half], z32[half], kxb, kyb,
                                 om).cpu().numpy()
            pw2 = beamform_prior(t32[~half], z32[~half], kxb, kyb,
                                 om).cpu().numpy()
            p1 = pw1 / max(pw1.max(), 1e-12)
            p2 = pw2 / max(pw2.max(), 1e-12)
            bf_div = float(np.abs(p1 - p2).mean() / (p1.mean() + 1e-12))
            vi = np.arange(len(z)) % 4 == 0
            tvi = torch.tensor(vi, device=DEVICE)

            cands = []
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
                        cands.append({
                            'band': list(band), 'smult': sm, 'gamma': g,
                            'label_corr': cc,
                            'feat': {'amp_ratio': float(np.std(pr)
                                                        / (zsd + 1e-9)),
                                     'val_resid': vr, 'bf_div': bf_div,
                                     'kp': kp, 'zsd': zsd,
                                     'n_obs': int(len(z))}})
            with torch.no_grad():
                Xc = torch.tensor(np.array([featvec(c) for c in cands],
                                           np.float32), device=DEVICE)
                scores = critic(Xc).cpu().numpy()
                ph = mmse_gpu(t32, z32, cfg['depth'], q32, sig0,
                              kp_hint=kp).cpu().numpy()
            hand_c = float(np.corrcoef(ph[v], gt_q[v])[0, 1])
            fq = net_gamma_mmse(net, P, z, cfg['depth'], Pq, sig0,
                                kp_hint=kp, use_em=False, smult_head=head)
            head_c = float(np.corrcoef(np.asarray(fq)[v], gt_q[v])[0, 1])
            ys = np.array([c['label_corr'] for c in cands])
            vrs = np.array([c['feat']['val_resid'] for c in cands])
            row = {'record': name, 'wi': wi, 'hand': hand_c,
                   'headfix': head_c,
                   'fix3': float(next(c['label_corr'] for c in cands
                                      if c['smult'] == 3.0
                                      and c['gamma'] == 'hand'
                                      and c['band'][0] == 0.35)),
                   'valres': float(ys[np.argmin(vrs)]),
                   'critic': float(ys[np.argmax(scores)]),
                   'critic_pick': int(np.argmax(scores)),
                   'critic_score_max': float(scores.max()),
                   'oracle': float(ys.max()),
                   'spread': float(ys.max() - ys.min())}
            rows.append(row)
            print(f"[{name} w{wi}] hand {hand_c:+.2f} valres "
                  f"{row['valres']:+.2f} fix3 {row['fix3']:+.2f} "
                  f"critic {row['critic']:+.2f} oracle {row['oracle']:+.2f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    json.dump(rows, open(OUT, 'w'), indent=1)
    h = np.array([r['hand'] for r in rows])
    bad = h < 0.5
    print(f"\nn {len(rows)} bad {bad.sum()}")
    for k in ('hand', 'headfix', 'fix3', 'valres', 'critic', 'oracle'):
        x = np.array([r[k] for r in rows])
        print(f"{k:8s} mean {x.mean():+.3f}  bad {x[bad].mean():+.3f}  "
              f"good {x[~bad].mean():+.3f}")
    d = np.array([r['critic'] - r['valres'] for r in rows])
    se = d.std(ddof=1) / np.sqrt(len(d))
    gc1 = d.mean() >= 0.05 and d.mean() / se >= 2.5
    print(f"\nG-C1 critic-valres {d.mean():+.4f}±{se:.4f} t "
          f"{d.mean()/se:+.2f}  {'PASS' if gc1 else 'FAIL'}")
    # G-C2: failure AUC using critic max score
    fail = (np.array([r['critic'] for r in rows]) < 0.5).astype(float)
    sc = np.array([r['critic_score_max'] for r in rows])
    pos, neg = sc[fail > 0], sc[fail == 0]
    auc = float(np.mean([(p < n) for p in pos for n in neg])) \
        if len(pos) and len(neg) else float('nan')
    print(f"G-C2 failure AUC {auc:.3f}  "
          f"{'PASS' if auc >= 0.8 else 'FAIL'}")
    worst = min(np.mean([r['critic'] - r['hand'] for r in rows
                         if r['record'] == n]) for n in RECS)
    print(f"G-C3 worst record d(critic-hand) {worst:+.3f}  "
          f"{'PASS' if worst >= -0.01 else 'FAIL'}")
    tie = [r for r in rows if r['spread'] < 0.05]
    picks = [r['critic_pick'] for r in tie]
    print(f"G-C4 tie windows {len(tie)}, distinct picks "
          f"{len(set(picks))}/12")
    print(f"wrote {OUT}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
