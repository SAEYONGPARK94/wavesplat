"""d3_static_limit.py — C1': is the out-of-fan wall physics or algorithm?

The stationary-field run showed the estimator recovers the field inside the
observed fan (corr 0.8-0.9) and fails beyond the sensor's range gate
(nRMSE 1.1-2.2). Two very different explanations:

  physics    — the water past the range gate was never seen, and it never
               will be: it flows toward the sensor from even farther out.
               No estimator can recover it, so the fix is a sensor, not a prior.
  algorithm  — the information is in the observations (a wave crest seen in
               the fan constrains its own continuation) and our estimator
               fails to use it. Then a better prior is exactly the fix.

Peel the handicaps off one at a time and see which region moves:

  1 fan_noisy   observations in the fan, sigma = 0.02 m   (today's setting)
  2 fan_clean   same points, zero noise
  3 fan_dense   zero noise, 4x the points, still fan-only
  4 all_dense   zero noise, points over the WHOLE domain  (oracle, cheating)

If 2 and 3 leave the beyond-range error where it is and only 4 moves it, the
wall is information, not algorithm. The near/lateral blind wedge is the
control: it is also unobserved, but the water there DID cross the fan, so a
good estimator should recover it — and if it does, the experiment is sound.

Writes results/direction3/static_limit.json + static_limit_<case>.png
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from d3_sa_nomp import mmse_bank_estimate
from d3_prior_net_eval import get_net, net_gamma_mmse
import d3_static_field as SF

OUT_JSON = 'results/direction3/static_limit.json'
OUT_PNG = 'results/direction3/static_limit_{}.png'
NET_CKPT = 'results/direction3/prior_net_v3.pt'

LEADS = np.array([0.0, 1.0, 2.0])
N_DENSE = 2400          # solve cost is O(n^3) in the training split; 2400 is
                        # 4x the baseline budget and still a few seconds
CONFIGS = ['fan_noisy', 'fan_clean', 'fan_dense', 'all_dense']


def resample(sc, config, rng):
    """Re-draw the observation set under one of the four handicap settings.

    The scene's own eta evaluator is reused, so the sea is identical across
    configs — only what we are allowed to measure of it changes.
    """
    cx, cy = sc['center']
    hx, hy = sc['hvec']
    eta = sc['make_eta']()
    t0 = sc['t0']
    # The domain is whatever the scene's query grid spans — the canonical
    # rectangle for the sims, the stereo patch for the real records. Taking
    # DOM_X/DOM_Y offsets from the sensor would put the fan outside the patch
    # for any record whose wave direction is not +x.
    dom_x = (sc['gx'].min(), sc['gx'].max())
    dom_y = (sc['gy'].min(), sc['gy'].max())
    scan_ts = np.arange(t0 - SF.W_OBS, t0 + 1e-9, SF.SCAN_DT)
    noise = 0.0 if config != 'fan_noisy' else SF.NOISE
    n_target = N_DENSE if config in ('fan_dense', 'all_dense') else SF.M_CAP
    per_scan = int(np.ceil(n_target / len(scan_ts)))

    P, Z = [], []
    for t_s in scan_ts:                       # ascending: HOS-safe
        if config == 'all_dense':
            xs = rng.uniform(*dom_x, per_scan)
            ys = rng.uniform(*dom_y, per_scan)
        else:
            rr = rng.uniform(*SF.R_FAN, per_scan)
            aa = rng.uniform(-SF.HALF_FOV, SF.HALF_FOV, per_scan)
            ca, sa = np.cos(aa), np.sin(aa)
            xs = cx + rr * (hx * ca - hy * sa)      # fan about the heading
            ys = cy + rr * (hx * sa + hy * ca)
            keep = ((xs >= dom_x[0]) & (xs <= dom_x[1]) &
                    (ys >= dom_y[0]) & (ys <= dom_y[1]))
            xs, ys = xs[keep], ys[keep]
            if len(xs) == 0:
                continue
        zz = eta(xs, ys, t_s)
        ok = np.isfinite(zz)
        P.append(np.column_stack([xs[ok], ys[ok], np.full(ok.sum(), t_s)]))
        Z.append(zz[ok] + (rng.normal(0, noise, ok.sum()) if noise else 0.0))
    P, Z = np.vstack(P), np.concatenate(Z)
    if len(Z) > n_target:
        ii = rng.choice(len(Z), n_target, replace=False)
        P, Z = P[ii], Z[ii]
    return P, Z, noise


def score(sc, Xq, Yq, gt, pred):
    cx, cy = sc['center']
    inside, R = SF.fan_mask(Xq, Yq, cx, cy, sc['hvec'])
    beyond = (R > SF.R_FAN[1]) & ~inside
    blind = ~inside & ~beyond
    sg = float(np.nanstd(gt))

    def nr(m, i):
        mm = m & np.isfinite(pred[i]) & np.isfinite(gt[i])
        if mm.sum() < 16:
            return None
        return float(np.sqrt(np.mean((pred[i][mm] - gt[i][mm]) ** 2)) / sg)

    return {'in_fan': [nr(inside, i) for i in range(len(LEADS))],
            'beyond_range': [nr(beyond, i) for i in range(len(LEADS))],
            'blind_near': [nr(blind, i) for i in range(len(LEADS))]}


def run_case(case):
    sc = SF.scene_sim(case) if case in ('linear', 'hos') \
        else SF.scene_real(case.split('_')[1])
    Xq, Yq = np.meshgrid(sc['gx'], sc['gy'])
    npx = Xq.size
    Pq = np.vstack([np.column_stack([Xq.ravel(), Yq.ravel(),
                                     np.full(npx, sc['t0'] + l_)])
                    for l_ in LEADS])
    eta = sc['make_eta']()
    gt = np.array([eta(Xq.ravel(), Yq.ravel(), sc['t0'] + l_).reshape(Xq.shape)
                   for l_ in LEADS])

    net = get_net(NET_CKPT)
    s0 = 0.05 * sc['Hs']
    res = {'label': sc['label'], 'leads': LEADS.tolist(), 'configs': {}}
    for cfg in CONFIGS:
        rng = np.random.default_rng(11)
        P, Z, noise = resample(sc, cfg, rng)
        # sigma0 is the estimator's ASSUMED noise, i.e. its regularization.
        # Hold it fixed across configs. Lowering it along with the measurement
        # noise changes the solver at the same time as the data, and then a
        # blown-up extrapolation cannot be attributed to either: an earlier
        # version did that and produced beyond-range nRMSE of 5.7 (the
        # unobserved region is unconstrained, so weak regularization lets the
        # solution run away there).
        sig = s0
        print(f"  [{cfg}] {len(Z)} obs, noise={noise}", flush=True)
        fq = mmse_bank_estimate(P, Z, sc['depth'], Pq, sigma0=sig,
                                kp_hint=sc['kp'], seed=sc['seed'])
        ent = {'n_obs': int(len(Z)), 'noise': noise,
               'WaveSplat': score(sc, Xq, Yq, gt,
                                  fq.reshape(len(LEADS), *Xq.shape))}
        if cfg in ('fan_noisy', 'fan_dense'):
            fq2 = net_gamma_mmse(net, P, Z, sc['depth'], Pq, sigma0=sig,
                                 seed=sc['seed'], kp_hint=sc['kp'])
            ent['WavePrior'] = score(sc, Xq, Yq, gt,
                                     fq2.reshape(len(LEADS), *Xq.shape))
        res['configs'][cfg] = ent
        w = ent['WaveSplat']
        print(f"      nowcast  in-fan {w['in_fan'][0]:.3f}  "
              f"beyond {w['beyond_range'][0]:.3f}  "
              f"blind {w['blind_near'][0]:.3f}", flush=True)
    return res


def figure(res, path):
    regions = ['in_fan', 'blind_near', 'beyond_range']
    titles = ['inside fan (observed)', 'near/lateral blind\n(water crossed fan)',
              'beyond range gate\n(never seen, never will be)']
    fig, ax = plt.subplots(figsize=(8.4, 4.2), dpi=120)
    x = np.arange(len(regions))
    cols = ['#B0B0B0', '#8FA9D8', '#5B7EC9', '#2E4B7A']
    for i, cfg in enumerate(CONFIGS):
        v = [res['configs'][cfg]['WaveSplat'][r][0] for r in regions]
        ax.bar(x + (i - 1.5) * 0.2, v, width=0.2, color=cols[i], label=cfg)
    ax.axhline(1.0, color='r', lw=1, ls='--')
    ax.text(2.35, 1.02, 'zero-predictor', fontsize=7, color='r')
    ax.set_xticks(x)
    ax.set_xticklabels(titles, fontsize=8)
    ax.set_ylabel('nowcast field nRMSE', fontsize=9)
    ax.set_title(f"{res['label']} — peeling the estimator's handicaps "
                 f"(U=0)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis='y')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', default='all',
                    choices=['linear', 'hos', 'real_YS17', 'real_LJ18g', 'all'])
    a = ap.parse_args()
    cases = ['linear', 'real_YS17', 'real_LJ18g'] if a.case == 'all' \
        else [a.case]
    allm = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
    for case in cases:
        print(f"[{case}]", flush=True)
        allm[case] = run_case(case)
        figure(allm[case], OUT_PNG.format(case))
        with open(OUT_JSON, 'w') as f:
            json.dump(allm, f, indent=1)
    print(f"\n-> {OUT_JSON}")


if __name__ == '__main__':
    sys.exit(main())
