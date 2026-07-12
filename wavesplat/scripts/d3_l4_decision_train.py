"""L4-v1: decision-loss readout head, real-data LORO, frozen-gate judgment.

Pre-registration: docs/L4_DECISION_prereg_2026-07-12.md (incl. amendment
#60: train on m2m7 real windows, record-LORO; v1 input = incumbent track).

Task: from the incumbent's 11-lead track forecast + scalars, predict
  T1 crest-arrival bin (leads 1..10 s, 10-class CE) and
  T2 per-lead elevation sign (10x BCE).
Judgment: frozen 60 windows rebuilt verbatim (Window class), each record
judged zero-shot by the fold that excluded it, against the direct readout
(argmax / sign) of the SAME incumbent track. Gates G-L4-1/2/3 + G-L4-4
(2 seeds + per-fold direction).
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
sys.path.insert(0, 'scripts')

from d3_falcon_real_operator import RECORDS
from d3_sa_multirecord import MRecord
from d3_prior_net_eval import get_net, DEVICE
from d3_l1_s2head_train import SmultHead
from d3_external_figs import Window

G = 9.81
OUT = 'results/direction3/l4_decision_v1.json'
RECS = ['YS17', 'AA15a', 'AA14', 'AA15b', 'LJ18g']
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def features(pred, Hs, kp, depth):
    """(n,11) track + scalars -> (n,14) normalized features."""
    om = np.sqrt(G * kp * np.tanh(kp * depth))
    Tp = 2 * np.pi / om
    a = pred / (Hs[:, None] / 4.0 + 1e-9)
    return np.column_stack([a, np.log(Hs), kp, Tp / 10.0]).astype(np.float32)


def labels(gt, Hs):
    """truth track (n,11) -> crest bin (argmax leads 1..10) + signs+mask."""
    tc = np.argmax(gt[:, 1:], axis=1)
    sg = (gt[:, 1:] > 0).astype(np.float32)
    w = (np.abs(gt[:, 1:]) >= 0.05 * Hs[:, None]).astype(np.float32)
    return tc.astype(np.int64), sg, w


class Head(nn.Module):
    def __init__(self, nin=14, nh=64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(nin, nh), nn.ReLU(),
                                  nn.Linear(nh, nh), nn.ReLU())
        self.crest = nn.Linear(nh, 10)
        self.sign = nn.Linear(nh, 10)

    def forward(self, x):
        h = self.body(x)
        return self.crest(h), self.sign(h)


def train_fold(Xtr, tc, sg, wt, seed):
    torch.manual_seed(seed)
    net = Head(Xtr.shape[1]).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4, weight_decay=1e-4)
    X = torch.tensor(Xtr, device=DEV)
    y1 = torch.tensor(tc, device=DEV)
    y2 = torch.tensor(sg, device=DEV)
    w = torch.tensor(wt, device=DEV)
    ce = nn.CrossEntropyLoss()
    n = len(X)
    for ep in range(400):
        idx = torch.randperm(n, device=DEV)
        for b in range(0, n, 128):
            ii = idx[b:b + 128]
            lc, ls = net(X[ii])
            bce = nn.functional.binary_cross_entropy_with_logits(
                ls, y2[ii], weight=w[ii], reduction='sum') / \
                (w[ii].sum() + 1e-9)
            loss = ce(lc, y1[ii]) + bce
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    return net


def build_judgment_cache():
    """Rebuild the frozen windows once; cache incumbent track + truth."""
    netp = get_net('results/direction3/prior_net_l1s2bw.pt')
    head = SmultHead().to(DEVICE)
    head.load_state_dict(torch.load(
        'results/direction3/prior_net_l1s2bw.pt', map_location=DEVICE)['head'])
    head.eval()
    cache = []
    for name in RECS:
        cfg = RECORDS[name]
        rec = MRecord(cfg['path'], stride=cfg['stride'])
        for wi in range(12):
            w = Window(name, wi, rec=rec, net=netp, head=head)
            m10 = w.leads <= 10.0
            leads = w.leads[m10]
            gt = w.gt_q[m10]
            if len(leads) < 11:
                continue
            fin = np.isfinite(gt[1:11])
            if fin.sum() < 6:
                print(f"[{name} w{wi}] skipped (finite 1-10s leads "
                      f"{int(fin.sum())})")
                continue
            _, pn = w.predict(w.Pq[m10])
            cache.append({'record': name, 'wi': wi, 'gt': gt[:11],
                          'pred': pn[:11], 'fin': fin, 'Hs': w.Hs,
                          'kp': w.kp, 'depth': cfg['depth']})
            print(f"[{name} w{wi}] cached (finite {int(fin.sum())}/10)",
                  flush=True)
    return cache


def main():
    data = {}
    for f in sorted(glob.glob('tmp/m2m7/*.npz')):
        d = np.load(f)
        data[os.path.basename(f)[:-4]] = {k: d[k] for k in d.files}

    cache = build_judgment_cache()
    print(f"judgment windows cached: {len(cache)}")

    results = []
    for seed in (0, 1):
        for hold in RECS:
            Xtr, tctr, sgtr, wtr = [], [], [], []
            for nm in RECS:
                if nm == hold:
                    continue
                d = data[nm]
                gt, pred = d['gt'], d['pred']
                ok = np.all(np.isfinite(gt), axis=1) & \
                    np.all(np.isfinite(pred), axis=1)
                Xtr.append(features(pred[ok], d['Hs'][ok], d['kp'][ok],
                                    RECORDS[nm]['depth']))
                tc, sg, w_ = labels(gt[ok], d['Hs'][ok])
                tctr.append(tc)
                sgtr.append(sg)
                wtr.append(w_)
            net = train_fold(np.vstack(Xtr), np.concatenate(tctr),
                             np.vstack(sgtr), np.vstack(wtr), seed)
            for c in cache:
                if c['record'] != hold:
                    continue
                gt, pn, fin = c['gt'], c['pred'], c['fin']
                X = features(pn[None, :], np.array([c['Hs']]),
                             np.array([c['kp']]), c['depth'])
                with torch.no_grad():
                    lc, ls = net(torch.tensor(X, device=DEV))
                logits = lc[0].cpu().numpy()
                logits[~fin] = -1e9            # restrict to finite bins
                tc_net = 1.0 + float(np.argmax(logits))
                sg_net = ls[0].cpu().numpy() > 0
                g1 = np.where(fin, gt[1:11], -np.inf)
                p1 = np.where(fin, pn[1:11], -np.inf)
                tc_true = 1.0 + float(np.argmax(g1))
                tc_inc = 1.0 + float(np.argmax(p1))
                msk = fin & (np.abs(np.nan_to_num(gt[1:11]))
                             >= 0.05 * c['Hs'])
                res = {'seed': seed, 'record': hold, 'wi': c['wi'],
                       'tc_err_net': abs(tc_net - tc_true),
                       'tc_err_inc': abs(tc_inc - tc_true)}
                if msk.sum() > 0:
                    tr_sign = gt[1:11][msk] > 0
                    res['sacc_net'] = float(np.mean(sg_net[msk] == tr_sign))
                    res['sacc_inc'] = float(np.mean(
                        (pn[1:11][msk] > 0) == tr_sign))
                results.append(res)
        print(f"seed {seed} folds done", flush=True)

    json.dump(results, open(OUT, 'w'), indent=1)
    print(f"wrote {OUT}")

    for seed in (0, 1):
        rs = [r for r in results if r['seed'] == seed]
        d1 = np.array([r['tc_err_inc'] - r['tc_err_net'] for r in rs])
        sa = [(r['sacc_net'], r['sacc_inc']) for r in rs if 'sacc_net' in r]
        san = np.array([a for a, b in sa])
        sai = np.array([b for a, b in sa])
        se = d1.std(ddof=1) / np.sqrt(len(d1))
        print(f"\nseed {seed}: n={len(rs)}")
        print(f"  T1 crest MAE: net "
              f"{np.mean([r['tc_err_net'] for r in rs]):.2f}s  inc-readout "
              f"{np.mean([r['tc_err_inc'] for r in rs]):.2f}s  (paired d "
              f"{d1.mean():+.2f}s, t {d1.mean()/se:+.2f})")
        print(f"  T2 sign acc:  net {san.mean():.1%}  inc {sai.mean():.1%}"
              f"  (d {(san-sai).mean():+.1%})")
        dirs = []
        for nm in RECS:
            dd = [r['tc_err_inc'] - r['tc_err_net'] for r in rs
                  if r['record'] == nm]
            dirs.append(round(float(np.mean(dd)), 2) if dd else None)
        print(f"  per-record T1 gain: {dirs}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
