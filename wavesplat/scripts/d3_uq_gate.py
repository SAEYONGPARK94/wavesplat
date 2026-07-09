"""d3_uq_gate.py — C: make the estimator say "I don't know".

Today's sharpest negative result: beyond the sensor's range gate the estimator
returns values WORSE than silence. nRMSE there is 1.0 with sparse observations
and 1.9 once the fan is densely sampled (STATIC_LIMIT_information_vs_model) —
above 1.0 means the field would be better predicted by outputting zero. The
posterior mean shrinks toward zero when it is uncertain, and that shrinkage was
protecting us; feeding the fan more data removed the protection and the
estimator extrapolated a confidently wrong sea.

An estimator that cannot be trusted everywhere must know where. The MMSE
posterior already carries that knowledge: its predictive variance

    var(x) = phi(x)^T Gamma phi(x) - phi(x)^T Gamma Phi^T K^-1 Phi Gamma phi(x)

is large exactly where the observations do not constrain the bank. This script
asks three questions of it:

  1. calibration  is the predictive variance honest? (z-score std, 95% coverage)
  2. ranking      does it rank errors? (correlation of predicted sd with |error|,
                  AUC of "is this pixel worse than the zero-predictor")
  3. gating       if we refuse to answer where sd exceeds a threshold, what do
                  we get? Report nRMSE on the retained region and the fraction
                  of the domain retained, sweeping the threshold.

The point of the gate is not accuracy. It is that "no answer" beats an answer
worse than zero, and the analytic pipeline has no way to make that call.

Usage: python3 scripts/d3_uq_gate.py [--case linear|storm|bimodal|real_YS17]
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
import d3_static_field as SF
import d3_storm_wall as SW

OUT = 'results/direction3/uq_gate.json'
PNG = 'results/direction3/uq_gate_{}.png'
BAND = (0.15, 5.0)      # the widened bank, per today's ablation
KEEP = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]


def build(case, seed=0):
    if case.startswith('real_'):
        sc = SF.scene_real(case.split('_', 1)[1])
        Xq, Yq = np.meshgrid(sc['gx'], sc['gy'])
        eta = sc['make_eta']()
        gt = eta(Xq.ravel(), Yq.ravel(), sc['t0']).reshape(Xq.shape)
        return (sc['P'], sc['z'], sc['depth'], sc['Hs'], sc['kp'], sc['t0'],
                Xq, Yq, gt, sc['center'], sc['hvec'])
    spec = SW.SEAS[case]
    eta, kp, Hs = SW.make_sea(spec, 1000 + 37 * seed)
    gx = np.arange(SF.DOM_X[0], SF.DOM_X[1] + 0.1, SF.DX)
    gy = np.arange(SF.DOM_Y[0], SF.DOM_Y[1] + 0.1, SF.DY)
    Xq, Yq = np.meshgrid(gx, gy)
    t0 = SF.W_OBS
    gt = eta(Xq.ravel(), Yq.ravel(), t0).reshape(Xq.shape)
    rng = np.random.default_rng(11 + seed)
    P, Z = SW.observations(eta, 'fan_noisy', rng, t0)
    return (P, Z, spec['depth'], Hs, kp, t0, Xq, Yq, gt, (0.0, 0.0),
            np.array([1.0, 0.0]))


def auc(score, label):
    """P(score of a positive > score of a negative). No sklearn dependency."""
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return None
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', default='storm',
                    choices=['wind', 'storm', 'bimodal', 'real_YS17',
                             'real_LJ18g'])
    a = ap.parse_args()

    P, Z, depth, Hs, kp, t0, Xq, Yq, gt, ctr, hvec = build(a.case)
    Pq = np.column_stack([Xq.ravel(), Yq.ravel(), np.full(Xq.size, t0)])
    print(f"[{a.case}] {len(Z)} obs, {Xq.size} queries", flush=True)

    fq, vq = mmse_bank_estimate(P, Z, depth, Pq, sigma0=0.05 * Hs,
                                kp_hint=kp, seed=0, band=BAND,
                                return_var=True)
    pr = fq.reshape(Xq.shape)
    sd = np.sqrt(np.maximum(vq, 0)).reshape(Xq.shape)

    fin = np.isfinite(gt) & np.isfinite(pr)
    sg = float(np.nanstd(gt[fin]))
    err = np.abs(pr - gt)
    inside, R = SF.fan_mask(Xq, Yq, ctr[0], ctr[1], hvec)
    beyond = (R > SF.R_FAN[1]) & ~inside

    # 1. calibration
    z = (pr - gt)[fin] / np.maximum(sd[fin], 1e-9)
    cal = {'z_std': float(np.std(z)),
           'cov68': float(np.mean(np.abs(z) < 1.0)),
           'cov95': float(np.mean(np.abs(z) < 1.96))}

    # 2. ranking: does sd know where the error is?
    worse_than_zero = (err > np.abs(gt)) & fin
    rank = {'corr_sd_err': float(np.corrcoef(sd[fin], err[fin])[0, 1]),
            'auc_worse_than_zero': auc(sd[fin], worse_than_zero[fin]),
            'frac_worse_than_zero': float(worse_than_zero[fin].mean()),
            'sd_in_fan': float(sd[inside & fin].mean()),
            'sd_beyond': float(sd[beyond & fin].mean())}

    # 3. gating: keep the most confident fraction, report nRMSE there
    order = np.argsort(sd[fin])
    e2 = ((pr - gt)[fin] ** 2)[order]
    keep_rows = []
    for k in KEEP:
        m = int(len(e2) * k)
        nr = float(np.sqrt(e2[:m].mean()) / sg)
        # what fraction of the retained pixels sit beyond the range gate?
        idx = np.where(fin.ravel())[0][order[:m]]
        frac_beyond = float(beyond.ravel()[idx].mean())
        keep_rows.append({'keep': k, 'nrmse': nr, 'frac_beyond': frac_beyond})

    res = {'case': a.case, 'n_obs': int(len(Z)), 'gt_std': sg,
           'nrmse_all': float(np.sqrt(np.mean((pr - gt)[fin] ** 2)) / sg),
           'calibration': cal, 'ranking': rank, 'gate': keep_rows}

    print(f"\n1. calibration   z-std {cal['z_std']:.3f} "
          f"(1.0 = honest)   cov68 {cal['cov68']:.3f}   cov95 {cal['cov95']:.3f}")
    print(f"2. ranking       corr(sd, |err|) {rank['corr_sd_err']:+.3f}   "
          f"AUC(worse-than-zero) {rank['auc_worse_than_zero']:.3f}")
    print(f"                 mean sd: in-fan {rank['sd_in_fan']:.3f}  "
          f"beyond {rank['sd_beyond']:.3f}  "
          f"({rank['sd_beyond']/rank['sd_in_fan']:.1f}x)")
    print(f"3. gate          nRMSE over the whole domain: {res['nrmse_all']:.3f}")
    print(f"{'keep':>6} {'nRMSE':>8} {'% of kept that is beyond-range':>32}")
    for r in keep_rows:
        print(f"{r['keep']:>6.0%} {r['nrmse']:>8.3f} {r['frac_beyond']:>31.1%}")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), dpi=110)
    ext = [Xq.min(), Xq.max(), Yq.min(), Yq.max()]
    v = 2 * np.nanstd(gt)
    axes[0].imshow(np.abs(pr - gt), origin='lower', extent=ext, cmap='magma',
                   vmin=0, vmax=v)
    axes[0].set_title('|error|', fontsize=10)
    axes[1].imshow(sd, origin='lower', extent=ext, cmap='magma', vmin=0, vmax=v)
    axes[1].set_title('predicted sd (the gate reads this)', fontsize=10)
    for ax in axes[:2]:
        ax.contour(Xq, Yq, inside.astype(float), levels=[0.5], colors='c',
                   linewidths=0.9, linestyles='--')
        ax.set_xlabel('x [m]', fontsize=8); ax.tick_params(labelsize=7)
    k = [r['keep'] for r in keep_rows]
    axes[2].plot(k, [r['nrmse'] for r in keep_rows], 'o-', color='#5B7EC9')
    axes[2].axhline(1.0, color='r', ls='--', lw=1)
    axes[2].text(0.55, 1.02, 'zero-predictor', color='r', fontsize=7)
    axes[2].set_xlabel('fraction of domain answered', fontsize=9)
    axes[2].set_ylabel('nRMSE on answered pixels', fontsize=9)
    axes[2].invert_xaxis(); axes[2].grid(alpha=0.25)
    axes[2].set_title('refusing to answer buys accuracy', fontsize=10)
    fig.suptitle(f"{a.case}: UQ gate (dashed = observed fan)", fontsize=11)
    fig.tight_layout()
    fig.savefig(PNG.format(a.case))
    print(f"\n-> {PNG.format(a.case)}")

    allr = json.load(open(OUT)) if os.path.exists(OUT) else {}
    allr[a.case] = res
    json.dump(allr, open(OUT, 'w'), indent=1)
    print(f"-> {OUT}")


if __name__ == '__main__':
    sys.exit(main())
