"""Critic v1 training (CRITIC_SELECTOR_DESIGN step 2).

Input rows from the label factory (v1 dir0 + v1h head-sea). Features are
deployment-observable scalars; target = task corr of the candidate.
Split BY SCENARIO FAMILY (sea state + comp count) so validation scenes are
unseen families. Tiny MLP -- the information is in ~9 scalars (N5 lesson).

Prints: val regret of critic-argmax vs oracle / vs fix3-hand / vs
val-resid-argmin (the broken incumbent) on held-out synthetic families.
The REAL judgment (60 windows, frozen gates) is a separate script.
"""
import argparse
import json
import re

import numpy as np
import torch
import torch.nn as nn

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT = 'results/direction3/critic_v1.pt'


def featvec(r):
    f = r['feat']
    vr = f.get('val_resid', float('nan'))
    vr_ok = 0.0 if (vr is None or not np.isfinite(vr)) else 1.0
    vr_v = 0.0 if vr_ok == 0.0 else min(float(vr), 3.0)
    return [min(f['amp_ratio'], 5.0), f['bf_div'], f['kp'] * 10.0,
            f['zsd'], np.log10(max(f['n_obs'], 1)) - 3.0,
            r['band'][0], np.log10(r['smult']),
            1.0 if r['gamma'] == 'net' else 0.0, vr_v, vr_ok]


def family(fname):
    m = re.match(r'episode_(BF\d+)_(\d+)comp', fname)
    return f"{m.group(1)}_{m.group(2)}" if m else fname


class Critic(nn.Module):
    def __init__(self, d=10, h=64):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, h), nn.SiLU(),
                               nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 1))

    def forward(self, x):
        return self.f(x).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--seed', type=int, default=0)
    a_ = ap.parse_args()
    torch.manual_seed(a_.seed)

    rows = []
    for p in ('results/direction3/critic_labels_v1.json',
              'results/direction3/critic_labels_v1h.json'):
        try:
            rows += json.load(open(p))
        except FileNotFoundError:
            print(f"(no {p})")
    print(f"{len(rows)} rows")
    fams = sorted({family(r['file']) for r in rows})
    val_fams = set(fams[::4])                    # every 4th family held out
    print(f"families {len(fams)}, val {sorted(val_fams)}")

    X = np.array([featvec(r) for r in rows], np.float32)
    y = np.array([r['label_corr'] for r in rows], np.float32)
    isv = np.array([family(r['file']) in val_fams for r in rows])
    Xt = torch.tensor(X[~isv], device=DEV)
    yt = torch.tensor(y[~isv], device=DEV)
    Xv = torch.tensor(X[isv], device=DEV)
    yv = torch.tensor(y[isv], device=DEV)

    net = Critic().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(a_.seed)
    best = (np.inf, None)
    for step in range(1, a_.steps + 1):
        ii = rng.choice(len(Xt), 128)
        loss = ((net(Xt[ii]) - yt[ii]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0:
            with torch.no_grad():
                vl = float(((net(Xv) - yv) ** 2).mean())
            if vl < best[0]:
                best = (vl, step)
                torch.save({'model': net.state_dict()}, CKPT)
            print(f"step {step:5d} train {float(loss):.4f} val {vl:.4f}",
                  flush=True)

    # selection regret on held-out families
    net.load_state_dict(torch.load(CKPT, map_location=DEV)['model'])
    net.eval()
    byep = {}
    for i, r in enumerate(rows):
        if isv[i]:
            byep.setdefault(r['file'], []).append(i)
    reg_or, sel_c, sel_f3, sel_vr = [], [], [], []
    with torch.no_grad():
        for f, idx in byep.items():
            ys = y[idx]
            preds = net(torch.tensor(X[idx], device=DEV)).cpu().numpy()
            sel_c.append(ys[np.argmax(preds)])
            reg_or.append(ys.max())
            f3 = [j for j, i in enumerate(idx)
                  if rows[i]['smult'] == 3.0 and rows[i]['gamma'] == 'hand'
                  and rows[i]['band'][0] == 0.35]
            sel_f3.append(ys[f3[0]] if f3 else np.nan)
            vrs = np.array([X[i][8] if X[i][9] > 0 else np.inf for i in idx])
            sel_vr.append(ys[np.argmin(vrs)])
    print(f"\nheld-out families ({len(byep)} episodes):")
    print(f"oracle      {np.mean(reg_or):+.3f}")
    print(f"critic      {np.mean(sel_c):+.3f}   regret {np.mean(reg_or)-np.mean(sel_c):.3f}")
    print(f"fix3-hand   {np.nanmean(sel_f3):+.3f}")
    print(f"val-resid   {np.mean(sel_vr):+.3f}   <- 고장난 현직")
    print(f"saved {CKPT} (best val {best[0]:.4f} @ {best[1]})")
    return 0


if __name__ == '__main__':
    sys.exit(main())
