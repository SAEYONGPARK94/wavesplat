"""B-2 DEEP judgment -- objective head-to-head on the frozen 60 windows.

4 models (T-s0, T-s1, L-s0, L-s1), each = its own net + its own SmultHead,
scored via net_gamma_mmse(use_em=False, smult_head=head, kp_hint) exactly like
d3_gate_promotion_eval. The 'hand' column is REUSED from
gate_promotion_eval.json (frozen; not recomputed). Gates:

  G-B2D-1  failure-window (hand<0.5) mean d(T-L) >= +0.05 AND both seeds agree.
  G-B2D-2  T-12k vs incumbent bw_s0_head (from gate_promotion_eval.json),
           all-window mean >= -0.01.

Also reports per-window learned smult (head behavior) and a prior-peakiness
probe on bwtask val scenes for T vs L.
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
from d3_prior_net_train import PriorNet, band_mask, NGRID, KREL
from d3_l1_s2head_train import SmultHead
from d3_gate_promotion_eval import window_data, RECS
from d3_prior_net_taskft import load_all

G = 9.81
OUT = 'results/direction3/b2deep_eval.json'
REF = 'results/direction3/gate_promotion_eval.json'
CKPT = {'T_s0': 'results/direction3/b2deep_T_s0.pt',
        'L_s0': 'results/direction3/b2deep_L_s0.pt',
        'T_s1': 'results/direction3/b2deep_T_s1.pt',
        'L_s1': 'results/direction3/b2deep_L_s1.pt'}


def load_model(ckpt):
    ck = torch.load(ckpt, map_location=DEVICE)
    net = PriorNet().to(DEVICE)
    net.load_state_dict(ck['model'])
    net.eval()
    head = SmultHead().to(DEVICE)
    head.load_state_dict(ck['head'])
    head.eval()
    return net, head, ck.get('step'), ck.get('val')


def prior_peakiness(net, D, idx):
    """participation ratio PR = (sum g)^2 / sum g^2 over the band; low = peaky."""
    mask = band_mask()
    prs = []
    with torch.no_grad():
        for i in idx:
            X = torch.tensor(D['X'][i][None], device=DEVICE)
            Ylog = net(X)[0].cpu().numpy()
            g = np.maximum(10.0 ** Ylog[mask], 1e-8)
            g = g / g.sum()
            prs.append(float((g.sum() ** 2) / (g ** 2).sum()))
    return float(np.mean(prs)), float(np.std(prs))


def main():
    ref = {(r['record'], r['wi']): r for r in json.load(open(REF))}
    models = {k: load_model(v) for k, v in CKPT.items()}
    for k, (_, _, st, vl) in models.items():
        print(f"loaded {k}: best step {st} val {vl:.4f}", flush=True)

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
            r = ref[(name, wi)]
            out = {'record': name, 'wi': wi, 'hand': r['hand'],
                   'bw_s0_head': r['bw_s0_head']}
            for mk, (net, head, _, _) in models.items():
                diag = {}
                fq = net_gamma_mmse(net, P, z, cfg['depth'], Pq, sig0,
                                    seed=int(i0), kp_hint=kp, use_em=False,
                                    smult_head=head, diag_out=diag)
                out[mk] = float(np.corrcoef(fq[v], gt_q[v])[0, 1])
                out[f'{mk}_smult'] = float(diag.get('smult', float('nan')))
            rows.append(out)
            print(f"[{name} w{wi}] hand {out['hand']:+.2f} "
                  f"Ts0 {out['T_s0']:+.2f} Ls0 {out['L_s0']:+.2f} "
                  f"Ts1 {out['T_s1']:+.2f} Ls1 {out['L_s1']:+.2f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    json.dump(rows, open(OUT, 'w'), indent=1)
    h = np.array([r['hand'] for r in rows])
    bad = h < 0.5
    n = len(rows)
    print(f"\nn windows {n}, bad (hand<0.5) {int(bad.sum())}", flush=True)

    def arr(k):
        return np.array([r[k] for r in rows])

    # ---- G-B2D-1: paired d(T - L) on failure windows, per seed ----
    d_s0 = arr('T_s0') - arr('L_s0')
    d_s1 = arr('T_s1') - arr('L_s1')
    db0, db1 = d_s0[bad], d_s1[bad]
    pooled = np.concatenate([db0, db1])
    m0, m1, mp = float(db0.mean()), float(db1.mean()), float(pooled.mean())
    same_dir = (m0 > 0) == (m1 > 0)
    gate1 = (mp >= 0.05) and same_dir and (m0 > 0) and (m1 > 0)
    se = pooled.std(ddof=1) / np.sqrt(len(pooled))

    # ---- G-B2D-2: T-12k vs incumbent bw_s0_head, all windows ----
    inc = arr('bw_s0_head')
    g2_s0 = float((arr('T_s0') - inc).mean())
    g2_s1 = float((arr('T_s1') - inc).mean())
    gate2 = g2_s0 >= -0.01

    # ---- head/prior behavior ----
    def sm(k):
        s = arr(f'{k}_smult')
        return float(s.mean()), float(s.std())
    D = load_all('tmp/prior_train/bwtask')
    vidx = np.arange(0, 300)  # val region scenes
    pk = {k: prior_peakiness(models[k][0], D, vidx) for k in CKPT}

    summary = {
        'n': n, 'n_bad': int(bad.sum()),
        'G_B2D_1': {'pass': bool(gate1), 'pooled_mean_d': mp, 'se': float(se),
                    'seed0_mean_d': m0, 'seed1_mean_d': m1,
                    'same_direction': bool(same_dir)},
        'G_B2D_2': {'pass': bool(gate2), 'Ts0_vs_incumbent': g2_s0,
                    'Ts1_vs_incumbent': g2_s1},
        'means_all': {k: float(arr(k).mean()) for k in
                      ('hand', 'bw_s0_head', 'T_s0', 'L_s0', 'T_s1', 'L_s1')},
        'means_bad': {k: float(arr(k)[bad].mean()) for k in
                      ('hand', 'bw_s0_head', 'T_s0', 'L_s0', 'T_s1', 'L_s1')},
        'smult': {k: sm(k) for k in CKPT},
        'prior_PR': {k: pk[k] for k in CKPT},
        'ckpt_step_val': {k: [models[k][2], models[k][3]] for k in CKPT},
    }
    json.dump({'rows': rows, 'summary': summary}, open(OUT, 'w'), indent=1)

    print("\n===== means (all 60 / bad) =====")
    for k in ('hand', 'bw_s0_head', 'T_s0', 'L_s0', 'T_s1', 'L_s1'):
        print(f"  {k:12s} all {arr(k).mean():+.3f}  bad {arr(k)[bad].mean():+.3f}")
    print("\n===== learned smult (mean+-std) / prior PR =====")
    for k in CKPT:
        print(f"  {k}: smult {sm(k)[0]:.2f}+-{sm(k)[1]:.2f}  "
              f"PR {pk[k][0]:.1f}+-{pk[k][1]:.1f}")
    print(f"\nG-B2D-1  {'PASS' if gate1 else 'FAIL'}  "
          f"pooled d(T-L) bad {mp:+.3f}+-{se:.3f}  "
          f"(s0 {m0:+.3f}, s1 {m1:+.3f}, same_dir {same_dir})")
    print(f"G-B2D-2  {'PASS' if gate2 else 'FAIL'}  "
          f"T_s0 - incumbent {g2_s0:+.4f}  (T_s1 {g2_s1:+.4f})")
    print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
