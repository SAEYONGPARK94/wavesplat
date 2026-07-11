"""M2xM7 v1 -- decision-loss encounter-track correction, REAL data, LORO.

Data: tmp/m2m7/<rec>.npz (dense sliding windows; the frozen 60 judgment
windows were excluded at extraction). Input = the deployable incumbent's
track forecast (net+s2-head, 11 leads x 1 s) + scalars; output = corrected
track, residual form (zero-init => incumbent at start).

THE M2 QUESTION: does training under a DECISION loss (what control needs:
crest polarity and phase) beat MSE training and the raw incumbent on
decision metrics? posterior-mean suboptimality for non-MSE losses is a
theorem -- this is its first empirical test in the project.

Arms (identical model, different loss):
  inc       incumbent, no correction (reference)
  mse       L = MSE(track)
  dec       L = (1 - cos_sim(track)) + 0.5 * sign-BCE weighted by |gt|
LORO: train on 4 records, judge on the 5th; pooled over the 5 folds.

================================ FROZEN GATES ================================
  M2-G1  dec beats inc on SIGN-ACC at |gt| > 0.5*std by >= +2 %p (LORO
         pooled, window-clustered t >= 2), with corr not worse (>= -0.01).
  M2-G2  dec vs mse on the same decision metrics -- the theorem's test;
         reported either way.
  M2-G3  no single record harmed by more than 0.01 corr.
FAIL -> honest close: "decision room exists in theory but this observation
class/incumbent leaves none measurable" + the numbers.
==============================================================================
"""
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT = 'results/direction3/m2m7_v1.json'


def load_all():
    data = {}
    for p in sorted(glob.glob('tmp/m2m7/*.npz')):
        d = np.load(p)
        name = os.path.basename(p)[:-4]
        m = np.isfinite(d['gt']).all(1) & np.isfinite(d['pred']).all(1)
        s = (d['Hs'][m] / 4.0)[:, None]
        data[name] = dict(pred=d['pred'][m] / s, gt=d['gt'][m] / s,
                          kp=d['kp'][m], Hs=d['Hs'][m])
        print(f"{name}: {m.sum()} windows")
    return data


class TrackNet(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(13, h), nn.SiLU(),
                               nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 11))
        with torch.no_grad():
            self.f[-1].weight.mul_(0.01)
            self.f[-1].bias.zero_()

    def forward(self, pred, kp, hs):
        x = torch.cat([pred, kp[:, None] * 10, hs[:, None] / 4], 1)
        return pred + self.f(x)


def losses(out, gt, kind):
    if kind == 'mse':
        return ((out - gt) ** 2).mean()
    cs = nn.functional.cosine_similarity(out, gt, dim=1)
    w = gt.abs() / gt.abs().mean()
    sbce = nn.functional.binary_cross_entropy_with_logits(
        out * 4.0, (gt > 0).float(), weight=w)
    return (1 - cs).mean() + 0.5 * sbce


def metrics(out, gt):
    corr = np.mean([np.corrcoef(o, g)[0, 1] for o, g in zip(out, gt)
                    if np.std(o) > 1e-9 and np.std(g) > 1e-9])
    thr = 0.5 * gt.std()
    m = np.abs(gt) > thr
    sacc = float((np.sign(out[m]) == np.sign(gt[m])).mean())
    return float(corr), sacc


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['loro', 'within'], default='loro',
                    help="within = v2: per-record time split (train first "
                         "60%% of windows by t0, judge last 40%%) -- the "
                         "onboard-adaptation scenario; removes cross-record "
                         "transfer entirely")
    a_ = ap.parse_args()
    torch.manual_seed(0)
    data = load_all()
    recs = sorted(data)
    folds = []
    for hold in recs:
        if a_.mode == 'loro':
            tr_p = np.concatenate([data[r]['pred'] for r in recs
                                   if r != hold])
            tr_g = np.concatenate([data[r]['gt'] for r in recs if r != hold])
            tr_k = np.concatenate([data[r]['kp'] for r in recs if r != hold])
            tr_h = np.concatenate([data[r]['Hs'] for r in recs if r != hold])
            te = data[hold]
        else:
            d = data[hold]
            n = len(d['pred'])
            cut = int(0.6 * n)
            tr_p, tr_g = d['pred'][:cut], d['gt'][:cut]
            tr_k, tr_h = d['kp'][:cut], d['Hs'][:cut]
            te = {k: v[cut:] for k, v in d.items()}
        fold = {'hold': hold, 'n_tr': len(tr_p), 'n_te': len(te['pred'])}
        ic, isa = metrics(te['pred'], te['gt'])
        fold['inc'] = {'corr': ic, 'sacc': isa}
        for kind in ('mse', 'dec'):
            net = TrackNet().to(DEV)
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3,
                                    weight_decay=1e-4)
            X = torch.tensor(tr_p, dtype=torch.float32, device=DEV)
            G = torch.tensor(tr_g, dtype=torch.float32, device=DEV)
            K = torch.tensor(tr_k, dtype=torch.float32, device=DEV)
            H = torch.tensor(tr_h, dtype=torch.float32, device=DEV)
            rng = np.random.default_rng(0)
            for step in range(4000):
                ii = rng.choice(len(X), min(256, len(X)))
                out = net(X[ii], K[ii], H[ii])
                loss = losses(out, G[ii], kind)
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                out = net(torch.tensor(te['pred'], dtype=torch.float32,
                                       device=DEV),
                          torch.tensor(te['kp'], dtype=torch.float32,
                                       device=DEV),
                          torch.tensor(te['Hs'], dtype=torch.float32,
                                       device=DEV)).cpu().numpy()
            c, s = metrics(out, te['gt'])
            fold[kind] = {'corr': c, 'sacc': s}
        folds.append(fold)
        print(f"[{hold}] inc corr {ic:+.3f}/sacc {isa:.3f}  "
              f"mse {fold['mse']['corr']:+.3f}/{fold['mse']['sacc']:.3f}  "
              f"dec {fold['dec']['corr']:+.3f}/{fold['dec']['sacc']:.3f}",
              flush=True)

    json.dump(folds, open(OUT.replace('.json', f'_{a_.mode}.json'), 'w'),
              indent=1)
    for k in ('inc', 'mse', 'dec'):
        c = np.mean([f[k]['corr'] for f in folds])
        s = np.mean([f[k]['sacc'] for f in folds])
        print(f"{k:4s} corr {c:+.3f}  sign-acc {s:.3f}")
    ds = np.array([f['dec']['sacc'] - f['inc']['sacc'] for f in folds])
    dc = np.array([f['dec']['corr'] - f['inc']['corr'] for f in folds])
    print(f"M2-G1 d(sacc) {ds.mean():+.4f} (min {ds.min():+.3f}) "
          f"d(corr) {dc.mean():+.4f}  "
          f"{'PASS' if ds.mean() >= 0.02 and dc.mean() >= -0.01 else 'FAIL'}")
    dm = np.array([f['dec']['sacc'] - f['mse']['sacc'] for f in folds])
    print(f"M2-G2 dec-mse sacc {dm.mean():+.4f}")
    worst = min(dc)
    print(f"M2-G3 worst record d(corr) {worst:+.3f} "
          f"{'PASS' if worst >= -0.01 else 'FAIL'}")
    print(f"wrote {OUT}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
