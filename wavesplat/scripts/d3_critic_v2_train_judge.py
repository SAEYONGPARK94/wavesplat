"""Critic v2 -- train (LORO, real labels, CNN features) and judge on the
frozen 60 windows. THE deciding run for the selection cell (user: 결판).

v2 fixes v1's two identified failure causes: labels are REAL (dense
sliding windows, judgment windows excluded at extraction) and features
keep the beamform X maps (6x51x51, CNN) beside candidate scalars.

Protocol: 5 LORO folds (train on 4 records' dense labels, ~3 min each).
Judgment: each record's 12 frozen windows are scored with ITS held-out
critic (zero-shot per record). Gates identical to v1 (frozen):
  G-C1 critic >= valres + 0.05 pooled, t >= 2.5
  G-C2 failure AUC >= 0.8      G-C3 no record < hand - 0.01
  plus context arms: fix3, headfix, oracle.
Either verdict closes the selection cell: PASS -> 학습 4승째; FAIL ->
+0.14 선택 상금은 이 특징·데이터로도 미회수(사망 확인).
"""
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from d3_falcon_real_operator import RECORDS
from d3_sa_multirecord import MRecord
from d3_prior_net_eval import (get_net, net_gamma_mmse, DEVICE, split_masks,
                               beamform_power, KREL, NGRID)
from d3_l1_s2head_train import SmultHead
from d3_gate_promotion_eval import window_data, RECS
from d3_gpu_mmse import mmse_gpu

OUT = 'results/direction3/critic_v2_judge.json'
BANDS = [(0.35, 2.5), (0.15, 2.5)]
SMULTS = [1.0, 3.0, 10.0]


class CriticV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(6, 16, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(4), nn.Flatten(),
            nn.Linear(32 * 16, 64), nn.SiLU())
        self.sc = nn.Sequential(nn.Linear(8, 32), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(96, 64), nn.SiLU(),
                                  nn.Linear(64, 1))

    def forward(self, X, f):
        return self.head(torch.cat([self.cnn(X), self.sc(f)], 1)).squeeze(-1)


def load_labels():
    data = {}
    for p in sorted(glob.glob('tmp/critic_v2/*.npz')):
        d = np.load(p)
        data[os.path.basename(p)[:-4]] = dict(
            X=d['X'], feat=d['feat'], label=d['label'], winid=d['winid'])
    return data


def train_fold(data, hold, steps=4000, seed=0):
    torch.manual_seed(seed)
    Xs, Fs, Ys = [], [], []
    for r, d in data.items():
        if r == hold:
            continue
        Xs.append(d['X'][d['winid']])        # per-row window map
        Fs.append(d['feat'])
        Ys.append(d['label'])
    X = torch.tensor(np.concatenate(Xs), device=DEVICE)
    F = torch.tensor(np.concatenate(Fs), device=DEVICE)
    Y = torch.tensor(np.concatenate(Ys), device=DEVICE)
    net = CriticV2().to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    for step in range(steps):
        ii = rng.choice(len(Y), 128)
        loss = ((net(X[ii], F[ii]) - Y[ii]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    return net


def main():
    data = load_labels()
    print({r: len(d['label']) for r, d in data.items()})
    critics = {}
    for hold in RECS:
        t0 = time.time()
        critics[hold] = train_fold(data, hold)
        print(f"fold -{hold} trained [{time.time()-t0:.0f}s]", flush=True)

    gnet = get_net('results/direction3/prior_net_l1s2bw.pt')
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
        crit = critics[name]
        for wi, i0 in enumerate(i0s):
            P, z, Pq, gt_q, leads, Hs, kp, _ = window_data(
                rec, cfg, int(i0), nfw, n_lead, wi)
            v = np.isfinite(gt_q) & (leads <= 10.0)
            if v.sum() < 5:
                continue
            sig0 = 0.05 * Hs
            zsd = float(np.std(z))
            m1, m2 = split_masks(P, mode='legs')
            _, _, p1 = beamform_power(P[m1], z[m1], cfg['depth'],
                                      KREL * kp, NGRID)
            _, _, p2 = beamform_power(P[m2], z[m2], cfg['depth'],
                                      KREL * kp, NGRID)
            p1 = (p1 / max(p1.max(), 1e-12)).reshape(NGRID, NGRID)
            p2 = (p2 / max(p2.max(), 1e-12)).reshape(NGRID, NGRID)
            pm = np.minimum(p1, p2)
            Xmap = np.stack([p1, p2, pm, np.log10(p1 + 1e-6) / 6 + 1,
                             np.log10(p2 + 1e-6) / 6 + 1,
                             np.log10(pm + 1e-6) / 6 + 1]).astype(np.float32)
            vi = np.arange(len(z)) % 4 == 0
            tvi = torch.tensor(vi, device=DEVICE)
            t32 = torch.tensor(P, device=DEVICE, dtype=torch.float32)
            z32 = torch.tensor(z, device=DEVICE, dtype=torch.float32)
            q32 = torch.tensor(Pq, device=DEVICE, dtype=torch.float32)
            ys, feats = [], []
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
                                gnet, P, z, cfg['depth'], Pq, sig0,
                                kp_hint=kp, use_em=False, smult_fixed=sm))
                            pv = np.asarray(net_gamma_mmse(
                                gnet, P[~vi], z[~vi], cfg['depth'], P[vi],
                                sig0, kp_hint=kp, use_em=False,
                                smult_fixed=sm))
                        vr = float(np.sqrt(np.mean((pv - z[vi]) ** 2))
                                   / (zsd + 1e-9))
                        cc = float(np.corrcoef(pr[v], gt_q[v])[0, 1]) \
                            if np.std(pr[v]) > 1e-12 else 0.0
                        ys.append(cc)
                        feats.append([band[0], np.log10(sm),
                                      1.0 if g == 'net' else 0.0,
                                      min(float(np.std(pr) / (zsd + 1e-9)),
                                          5.0), min(vr, 3.0), kp * 10,
                                      zsd, Hs])
            ys = np.array(ys)
            with torch.no_grad():
                sc = crit(torch.tensor(np.repeat(Xmap[None], 12, 0),
                                       device=DEVICE),
                          torch.tensor(np.array(feats, np.float32),
                                       device=DEVICE)).cpu().numpy()
                ph = mmse_gpu(t32, z32, cfg['depth'], q32, sig0,
                              kp_hint=kp).cpu().numpy()
            fq = net_gamma_mmse(gnet, P, z, cfg['depth'], Pq, sig0,
                                kp_hint=kp, use_em=False, smult_head=head)
            vrs = np.array([f[4] for f in feats])
            rows.append({'record': name, 'wi': wi,
                         'hand': float(np.corrcoef(ph[v], gt_q[v])[0, 1]),
                         'headfix': float(np.corrcoef(
                             np.asarray(fq)[v], gt_q[v])[0, 1]),
                         'fix3': float(ys[2]),
                         'valres': float(ys[np.argmin(vrs)]),
                         'critic': float(ys[np.argmax(sc)]),
                         'critic_score_max': float(sc.max()),
                         'oracle': float(ys.max()),
                         'spread': float(ys.max() - ys.min())})
            r = rows[-1]
            print(f"[{name} w{wi}] hand {r['hand']:+.2f} valres "
                  f"{r['valres']:+.2f} critic {r['critic']:+.2f} "
                  f"oracle {r['oracle']:+.2f}", flush=True)

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
    print(f"G-C1 critic-valres {d.mean():+.4f}±{se:.4f} t {d.mean()/se:+.2f}"
          f"  {'PASS' if d.mean() >= 0.05 and d.mean()/se >= 2.5 else 'FAIL'}")
    fail = (np.array([r['critic'] for r in rows]) < 0.5).astype(float)
    sc_ = np.array([r['critic_score_max'] for r in rows])
    pos, neg = sc_[fail > 0], sc_[fail == 0]
    auc = float(np.mean([(p < n) for p in pos for n in neg])) \
        if len(pos) and len(neg) else float('nan')
    print(f"G-C2 failure AUC {auc:.3f}  {'PASS' if auc >= 0.8 else 'FAIL'}")
    worst = min(np.mean([r['critic'] - r['hand'] for r in rows
                         if r['record'] == n]) for n in RECS)
    print(f"G-C3 worst record {worst:+.3f} "
          f"{'PASS' if worst >= -0.01 else 'FAIL'}")
    dh = np.array([r['critic'] - r['headfix'] for r in rows])
    print(f"context: critic-headfix {dh.mean():+.4f} "
          f"(bad {dh[bad].mean():+.4f})")
    print(f"wrote {OUT}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
