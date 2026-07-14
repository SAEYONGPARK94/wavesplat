"""E-12g — close the LAST oracle input. A fully sensor-realizable arm.

E-12d removed the dense-GT kp hint. But E-12e found a second oracle input that
survived: sigma0 = 0.05 * Hs, with Hs = 4*std(Zmem) -- the DENSE stereo buffer
(d3_gate_promotion_eval.py:61). E-11's own stored data shows the gap between the
oracle Hs and the Hs a sensor computes from its own 600 points is not negligible:
median 1.04x, only 22% of windows within 5%, tail out to 1.75x. And this project
already knows the noise floor is a sensitive knob (2026-07-11: "do not pick the
noise floor from in-window residuals").

So: an arm in which the estimator sees NOTHING but its own 600 sparse points.

    oracle  : kp = directional_spread(Zmem[-1])   [dense GT FFT]     (frozen)
              Hs = 4*std(Zmem)                    [dense GT buffer]
    sensor  : kp = two-stage beamform_top(P, z)   [600 sparse obs]   PRIMARY
              Hs = 4*std(z)                       [600 sparse obs]

Identical windows, identical observations (P, z), identical seeds. Both arms give
hand and net the SAME (kp, sigma0), so the paired Delta stays clean.

GATE (fixed before running, same as E-12d):
  PASS      delta_sensor >= +0.020 AND cluster-robust CI (t(G-1)!) excludes 0
  DEMOTED   significant but < +0.017
  FAIL      CI includes 0 -> the gain depended on oracle inputs; withdraw.

Note the CI uses t(G-1), not 1.96 (E-12e defect D1).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from d3_falcon_real_operator import RECORDS      # noqa: E402
from d3_sa_multirecord import MRecord            # noqa: E402
from d3_prior_net_eval import get_net, DEVICE    # noqa: E402
from d3_l1_s2head_train import SmultHead         # noqa: E402
from d3_external_figs import Window              # noqa: E402
from d3_cluster_se import cluster_se             # noqa: E402
from d18_relax_estimator import beamform_top     # noqa: E402

OUT = 'results/direction3/e12g_full_sensor.json'
RECS = ['YS17', 'AA15a', 'AA14', 'AA15b', 'LJ18g']
LEADS = [0.0, 2.0, 5.0, 8.0]
CKPT = 'results/direction3/prior_net_l1s2bw.pt'


def kp_sparse(P, z, depth):
    """The estimator's own kp: two-stage beamform zoom on the SPARSE obs.
    Byte-for-byte d3_prior_net_data.py:132 (training) and the kp_hint=None
    branch of d3_prior_net_eval.py:71 (inference). Deterministic."""
    Pf, zf = P.astype(np.float32), z.astype(np.float32)
    bx, by = beamform_top(Pf, zf, depth, klim=2.0, n=120)
    kp = max(float(np.hypot(bx, by)), 1e-3)
    bx, by = beamform_top(Pf, zf, depth, klim=3.5 * kp, n=120)
    return max(float(np.hypot(bx, by)), 1e-3)


def score(w, leads):
    gx, gy, tr, fh, fn = w.fields(leads, stride=2)
    trs, fhs, fns = np.array(tr), np.array(fh), np.array(fn)
    m = np.isfinite(trs)
    if m.sum() < 500:
        return None
    return (float(np.corrcoef(fhs[m], trs[m])[0, 1]),
            float(np.corrcoef(fns[m], trs[m])[0, 1]), int(m.sum()))


def judge(name, d, g):
    d = np.asarray(d, float)
    mu = d.mean()
    se, G = cluster_se(d, g)
    tcrit = float(stats.t.ppf(0.975, G - 1))
    p = float(2 * stats.t.sf(abs(mu / se), G - 1))
    lo, hi = mu - tcrit * se, mu + tcrit * se
    return dict(delta=float(mu), se_cluster=float(se), n_clusters=int(G),
                t_cluster=float(mu / se), p_cluster=p, t_crit=tcrit,
                ci=[float(lo), float(hi)], excludes_zero=bool(lo > 0 or hi < 0),
                wins=int((d > 0).sum()), n=int(len(d)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_win', type=int, default=12)
    ap.add_argument('--records', nargs='+', default=RECS)
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()

    net = get_net(CKPT)
    head = SmultHead().to(DEVICE)
    head.load_state_dict(torch.load(CKPT, map_location=DEVICE)['head'])
    head.eval()

    rows = []
    for name in a.records:
        cfg = RECORDS[name]
        rec = MRecord(cfg['path'], stride=cfg['stride'])
        for wi in range(a.n_win):
            t0 = time.time()
            w = Window(name, wi, rec=rec, net=net, head=head)
            leads = [l_ for l_ in LEADS if l_ <= min(10.0, w.leads.max())]

            kp_o, sig_o = float(w.kp), float(w.sig0)          # dense GT
            Hs_o = w.Hs
            kp_s = kp_sparse(w.P, w.z, cfg['depth'])          # 600 sparse obs
            Hs_s = 4.0 * float(np.std(w.z))
            sig_s = 0.05 * Hs_s

            w.kp, w.sig0 = kp_o, sig_o
            s_o = score(w, leads)
            w.kp, w.sig0 = kp_s, sig_s
            s_s = score(w, leads)
            if s_o is None or s_s is None:
                print(f'[{name} w{wi}] too few finite px — skipped')
                continue

            row = dict(record=name, wi=wi, n_px=s_o[2],
                       kp_oracle=kp_o, kp_sensor=kp_s,
                       Hs_oracle=float(Hs_o), Hs_sensor=Hs_s,
                       kp_ratio=kp_o / kp_s, Hs_ratio=float(Hs_o) / Hs_s,
                       hand_oracle=s_o[0], learned_oracle=s_o[1],
                       hand_sensor=s_s[0], learned_sensor=s_s[1])
            row['d_oracle'] = row['learned_oracle'] - row['hand_oracle']
            row['d_sensor'] = row['learned_sensor'] - row['hand_sensor']
            rows.append(row)
            print(f"[{name} w{wi}] kp x{row['kp_ratio']:.2f} Hs x{row['Hs_ratio']:.2f}"
                  f" | ORACLE d {row['d_oracle']:+.3f} "
                  f"(h {s_o[0]:+.3f} n {s_o[1]:+.3f}) | SENSOR d "
                  f"{row['d_sensor']:+.3f} (h {s_s[0]:+.3f} n {s_s[1]:+.3f})"
                  f" [{time.time()-t0:.0f}s]", flush=True)
        rec.f.close()

    g = [r['record'] for r in rows]
    res = {}
    print(f"\n{'='*88}\nE-12g  (n={len(rows)} windows, {len(set(g))} record clusters)"
          f"\n{'='*88}")
    print(f"{'arm':<10}{'hand':>9}{'net':>9}{'delta':>10}{'CR-SE':>9}{'t':>7}"
          f"{'p':>9}{'95% CI (t(G-1))':>22}{'wins':>8}")
    for mode in ('oracle', 'sensor'):
        h = np.array([r[f'hand_{mode}'] for r in rows])
        n_ = np.array([r[f'learned_{mode}'] for r in rows])
        j = judge(mode, n_ - h, g)
        j['hand'], j['net'] = float(h.mean()), float(n_.mean())
        res[mode] = j
        print(f"{mode:<10}{h.mean():+9.3f}{n_.mean():+9.3f}{j['delta']:+10.4f}"
              f"{j['se_cluster']:9.4f}{j['t_cluster']:7.2f}{j['p_cluster']:9.4f}"
              f"   [{j['ci'][0]:+.4f}, {j['ci'][1]:+.4f}]"
              f"{f'{j['wins']}/{j['n']}':>8}")

    print('\nper-record delta (sensor arm):')
    for nm in RECS:
        ds = [r['d_sensor'] for r in rows if r['record'] == nm]
        do = [r['d_oracle'] for r in rows if r['record'] == nm]
        if ds:
            print(f"   {nm:<7} sensor {np.mean(ds):+.4f}   "
                  f"(oracle {np.mean(do):+.4f})   n={len(ds)}")

    # leave-one-record-out on the primary arm
    print('\nLORO (sensor arm):')
    for drop in RECS:
        m = np.array([r['record'] != drop for r in rows])
        if m.sum() == 0:
            continue
        j = judge('', np.array([r['d_sensor'] for r in rows])[m],
                  np.array(g)[m])
        print(f"   drop {drop:<7} delta {j['delta']:+.4f}  p {j['p_cluster']:.4f}"
              f"  {'ok' if j['excludes_zero'] else 'NOT SIG'}")

    kr = np.array([r['kp_ratio'] for r in rows])
    hr = np.array([r['Hs_ratio'] for r in rows])
    print(f"\noracle/sensor input gaps: kp median {np.median(kr):.3f}x "
          f"[{kr.min():.2f},{kr.max():.2f}]  |  Hs median {np.median(hr):.3f}x "
          f"[{hr.min():.2f},{hr.max():.2f}]")

    s = res['sensor']
    verdict = ('FAIL — CI includes 0; the gain depended on oracle inputs'
               if not s['excludes_zero'] else
               'PASS — gain survives with NO oracle input at all'
               if s['delta'] >= 0.020 else
               'DEMOTED — significant but < half the frozen estimate')
    print(f"\n*** E-12g VERDICT: {verdict}")
    print(f"    sensor delta {s['delta']:+.4f}, CR-95% CI "
          f"[{s['ci'][0]:+.4f}, {s['ci'][1]:+.4f}] (t({s['n_clusters']-1})), "
          f"p {s['p_cluster']:.4f}")
    res['verdict'] = verdict
    json.dump({'rows': rows, 'summary': res}, open(a.out, 'w'), indent=1)
    print(f"\n-> {a.out}")


if __name__ == '__main__':
    sys.exit(main())
