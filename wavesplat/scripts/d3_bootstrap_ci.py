"""
d3_bootstrap_ci.py — bootstrap CIs for the headline numbers (R2's
approval condition #1). 2000 resamples each.

  FF-band 0.91   design point (n=12 seas): resample seas, pooled corr
                 over leads 0-2 s, W=10.
  real 0.70      YS17+AA15a W=20 pooled 0-10 s: resample the 12 windows
                 (6+6, preds/gts stored in the operator JSONs).
  gate #1 split  48 confirmatory cells: resample cells, split delta.

Usage: python3 scripts/d3_bootstrap_ci.py
"""
import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

OUT = 'results/direction3/bootstrap_ci.json'
NB = 2000


def ci(vals, lo=2.5, hi=97.5):
    return [float(np.percentile(vals, lo)), float(np.percentile(vals, hi))]


def main():
    rng = np.random.default_rng(0)
    res = {}

    # ---- FF band, design point (12 seas)
    d = json.load(open(
        'results/direction3/falcon_hos_probe_design22kt_n12.json'))
    leads = np.array(d['leads'])
    m = leads <= 2.0
    pairs = d['raw']['10.0']
    P = np.array([p for p, _ in pairs])[:, m]
    Gt = np.array([g for _, g in pairs])[:, m]
    n = len(P)
    boots = []
    for _ in range(NB):
        ii = rng.integers(0, n, n)
        boots.append(np.corrcoef(P[ii].ravel(), Gt[ii].ravel())[0, 1])
    est = float(np.corrcoef(P.ravel(), Gt.ravel())[0, 1])
    res['ff_band_design'] = dict(est=est, ci95=ci(boots), n_seas=n)
    print(f"FF-band design: {est:+.3f}  95% CI {ci(boots)}")

    # ---- real pooled 0-10 s (W=20, YS17+AA15a windows)
    Pw, Gw = [], []
    for recn in ('YS17', 'AA15a'):
        dd = json.load(open(
            f'results/direction3/falcon_real_operator_{recn}.json'))
        ll = np.array(dd['leads'])
        mm = ll <= 10.0
        for w in dd['windows']:
            p_ = np.array(w['by_W']['20.0']['pred'])[mm]
            g_ = np.array(w['gt'])[mm]
            v = np.isfinite(g_)
            Pw.append(p_[v])
            Gw.append(g_[v])
    nw = len(Pw)
    boots = []
    for _ in range(NB):
        ii = rng.integers(0, nw, nw)
        boots.append(np.corrcoef(np.concatenate([Pw[i] for i in ii]),
                                 np.concatenate([Gw[i] for i in ii]))[0, 1])
    est = float(np.corrcoef(np.concatenate(Pw), np.concatenate(Gw))[0, 1])
    res['real_pooled_W20'] = dict(est=est, ci95=ci(boots), n_windows=nw)
    print(f"real pooled (W=20, 0-10s): {est:+.3f}  95% CI {ci(boots)}")

    # ---- confirmatory #1 split delta (48 cells)
    dd = json.load(open('results/direction3/gate_confirmatory.json'))
    cells = dd['cells']
    recs = sorted(set(c['record'] for c in cells))
    def split_delta(cs):
        out = []
        for name in recs:
            g = [c for c in cs if c['record'] == name]
            if len(g) < 4:
                continue
            vv = np.array([c['vr_rel'] for c in g])
            med = np.median(vv)
            mad = np.median(np.abs(vv - med)) + 1e-9
            for c, vn in zip(g, (vv - med) / mad):
                out.append((vn, c['realized']))
        x = np.array([a for a, _ in out])
        r = np.array([b for _, b in out])
        md = np.median(x)
        return float(r[x <= md].mean() - r[x > md].mean())
    boots = []
    for _ in range(NB):
        ii = rng.integers(0, len(cells), len(cells))
        boots.append(split_delta([cells[i] for i in ii]))
    res['gate_conf1_split'] = dict(est=dd['split_delta'], ci95=ci(boots),
                                   n_cells=len(cells))
    print(f"gate #1 split delta: {dd['split_delta']:+.3f}  "
          f"95% CI {ci(boots)}")

    json.dump(res, open(OUT, 'w'), indent=1)
    print(f"saved {OUT}")


if __name__ == '__main__':
    main()
