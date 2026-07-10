"""E2: the criterion, not the network. Evidence and SURE against the residual.

Six attempts at learning failed today, and the last two named the reason. The
gate ranks twenty certified estimators by their residual on held-out
observations, and O1 showed exactly what that criterion cannot see:

    a smooth range-dependent sensor bias is representable by the bank's longest
    atoms. A wide band, or a prior tilted toward low k, ABSORBS the bias. The
    held-out residual goes DOWN. The field gets a spurious long wave.
    The residual rewards what the field punishes.

That is not a data problem, and no critic trained where the criterion works can
learn to distrust it. It is a criterion problem, and statistics has known the
answer since 1985: a fit that buys residual with freedom must be charged for the
freedom.

Our estimator is a linear smoother in z -- p_hat = Phi_q Gam Phi^T K^-1 z -- so
its degrees of freedom are exactly tr(H) with H = Phi Gam Phi^T K^-1, and three
classical criteria become available in closed form, none of them learned:

  evidence   log N(z | 0, K),  K = Phi Gam Phi^T + s2 I.  The Occam factor lives
             in log|K|: a model with large prior variance pays for the volume of
             data it could have explained and did not.
  SURE       ||z - Hz||^2 / n - s2 + 2 s2 tr(H) / n.  An unbiased estimate of the
             prediction risk at the observed points, with the freedom penalty
             explicit.
  GCV        ||z - Hz||^2 / n / (1 - tr(H)/n)^2.  SURE's cousin, no s2 needed.

PRE-REGISTERED, and the interesting cell is the second one:
  1. on CLEAN synthetic seas, no criterion may be much worse than the residual
     (regret 0.0102). A criterion that fixes the biased case by being generally
     conservative has fixed nothing.
  2. on BIAS-injected seas, where the residual's regret explodes to 0.1217,
     at least one criterion must stay below 0.03.
  3. on the 30 real record-window cells, the winner must beat the residual gate's
     +0.0101 gap to the oracle.

If (2) passes and (3) fails, the real leak is not the bias, which O1 already
suspected, and we will have said so twice.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from d3_gpu_mmse import mmse_gpu
from d3_l1_learned_prior import RadialGain, CANDS, S0
from d3_o1_defect_ablation import synth_sea, observe, gains_for
from d3_g3_adversarial_sea import geometry, DEV, DEPTH, KP
from d3_a2_real_revalidation import draw, HELD
import d3_static_field as SF

OUT = 'results/direction3/e2_evidence_sure.json'
CRITERIA = ('residual', 'evidence', 'sure', 'gcv')
REAL_RECS = ['AA14', 'AA15a', 'AA15b', 'LJ18g', 'YS17']
WINDOWS = (0, 1, 2, 3, 4, 5)
RSEEDS = (11, 12, 13)


def criteria(P, z, depth, kp, band, gain, sig0, tr, he):
    """All four scores for one candidate. Lower is better for every one."""
    with torch.no_grad():
        pv = mmse_gpu(P[tr], z[tr], depth, P[he], sigma0=sig0, kp_hint=kp,
                      band=band, prior_gain=gain)
        r_res = float(torch.sqrt(((pv - z[he]) ** 2).mean()))

        _, mo = mmse_gpu(P, z, depth, P[:1], sigma0=sig0, kp_hint=kp,
                         band=band, prior_gain=gain, return_model=True)
        K, Phi, gam, s2 = mo['K'], mo['Phi'], mo['gam'], mo['s2']
        n = len(z)
        Ki = torch.linalg.inv(K)
        a = Ki @ z
        # evidence: -2 log N(z | 0, K), so that lower is better
        sign, logdet = torch.linalg.slogdet(K)
        ev = float(z @ a + logdet)

        # hat matrix and its trace: H = Phi Gam Phi^T Ki = (K - s2 I) Ki
        H = (K - s2 * torch.eye(n, device=DEV, dtype=K.dtype)) @ Ki
        dof = float(torch.diagonal(H).sum())
        rss = float(((z - H @ z) ** 2).sum())
        sure = rss / n - float(s2) + 2 * float(s2) * dof / n
        gcv = (rss / n) / max((1 - dof / n) ** 2, 1e-6)
    return dict(residual=r_res, evidence=ev, sure=sure, gcv=gcv, dof=dof)


def score_sea(P, z, Pq, gt, kp, depth, gains, sig0, gen):
    n = len(z)
    idx = torch.randperm(n, device=DEV, generator=gen)
    tr, he = idx[:int((1 - HELD) * n)], idx[int((1 - HELD) * n):]
    zero = float(gt.std())
    rows, rats, dofs = [], [], []
    with torch.no_grad():
        for (band, al), g in gains.items():
            c = criteria(P, z, depth, kp, band, g, sig0, tr, he)
            rows.append([c[k] for k in CRITERIA])
            dofs.append(c['dof'])
            pr = mmse_gpu(P, z, depth, Pq, sigma0=sig0, kp_hint=kp, band=band,
                          prior_gain=g)
            pr = pr - pr.mean()
            rats.append(float(torch.sqrt(((pr - gt) ** 2).mean()) / zero))
    C, r = np.array(rows), np.array(rats)
    best = r.min()
    return {k: r[int(np.argmin(C[:, i]))] - best for i, k in enumerate(CRITERIA)}, \
        best, float(np.mean(dofs))


def synthetic(net, cfg, n_sea, gen):
    P0, Pq = geometry()
    out = {k: [] for k in CRITERIA}
    for i in range(n_sea):
        rng = np.random.default_rng(5000 + i)
        s = synth_sea(rng)
        P, z, gt = observe(P0, Pq, s, rng, gen, cfg)
        if len(z) < 150:
            continue
        gains = gains_for(net, KP, DEPTH)
        reg, _, _ = score_sea(P, z, Pq, gt, KP, DEPTH, gains, S0, gen)
        for k in CRITERIA:
            out[k].append(reg[k])
    return {k: (np.mean(v), np.std(v, ddof=1) / np.sqrt(len(v)))
            for k, v in out.items()}


def real(net, gen):
    t = lambda x: torch.tensor(x, device=DEV, dtype=torch.float32)
    out = {k: [] for k in CRITERIA}
    for rec in REAL_RECS:
        for w in WINDOWS:
            try:
                sc = SF.scene_real(rec, win=w)
            except Exception:
                continue
            Xq, Yq = np.meshgrid(sc['gx'], sc['gy'])
            eta = sc['make_eta']()
            g_all = eta(Xq.ravel(), Yq.ravel(), sc['t0'])
            cx, cy = sc['center']
            inside, _ = SF.fan_mask(Xq, Yq, cx, cy, sc['hvec'])
            m = inside.ravel() & np.isfinite(g_all)
            if m.sum() < 150:
                continue
            Pq = t(np.column_stack([Xq.ravel()[m], Yq.ravel()[m],
                                    np.full(m.sum(), sc['t0'])]))
            gtq = t(g_all[m])
            gtq = gtq - gtq.mean()
            if float(gtq.std()) < 1e-6:
                continue
            gains = gains_for(net, sc['kp'], sc['depth'])
            for ds in RSEEDS:
                rng = np.random.default_rng(1000 + ds)
                P, Z = draw(sc, rng)
                if len(Z) < 200:
                    continue
                reg, _, _ = score_sea(t(P), t(Z), Pq, gtq, sc['kp'],
                                      sc['depth'], gains, 0.05 * sc['Hs'], gen)
                for k in CRITERIA:
                    out[k].append(reg[k])
    return {k: (np.mean(v), np.std(v, ddof=1) / np.sqrt(len(v)))
            for k, v in out.items()}, len(out['residual'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seas', type=int, default=30)
    ap.add_argument('--skip-real', action='store_true')
    a_ = ap.parse_args()

    net = RadialGain().to(DEV)
    net.load_state_dict(torch.load('results/direction3/l1_prior_gain.pt'))
    net.eval()
    gen = torch.Generator(device=DEV).manual_seed(0)
    res = {}

    print("regret of each selection criterion (lower is better)\n")
    print(f"{'seas':<14}" + "".join(f"{c:>18}" for c in CRITERIA))
    for cfg in ('none', 'bias'):
        r = synthetic(net, cfg, a_.seas, gen)
        res[f'synth_{cfg}'] = {k: list(v) for k, v in r.items()}
        print(f"{'synth ' + cfg:<14}" +
              "".join(f"{r[c][0]:>12.4f}±{r[c][1]:.4f}" for c in CRITERIA),
              flush=True)

    if not a_.skip_real:
        r, n = real(net, gen)
        res['real'] = {k: list(v) for k, v in r.items()}
        print(f"{'real (n=' + str(n) + ')':<14}" +
              "".join(f"{r[c][0]:>12.4f}±{r[c][1]:.4f}" for c in CRITERIA),
              flush=True)

    print("\nregret = field nRMSE of the chosen candidate minus the sea's best "
          "of twenty.\n")
    b = res['synth_bias']
    g1 = all(res['synth_none'][c][0] <= res['synth_none']['residual'][0] + 0.01
             for c in ('evidence', 'sure', 'gcv'))
    g2 = min(b[c][0] for c in ('evidence', 'sure', 'gcv')) < 0.03
    print(f"=== E2 gates (pre-registered) ===")
    print(f"  1) no criterion much worse on clean seas : "
          f"{'PASS' if g1 else 'FAIL'}")
    print(f"  2) some criterion survives the bias      : "
          f"{'PASS' if g2 else 'FAIL'}"
          f"  (best {min(b[c][0] for c in ('evidence','sure','gcv')):.4f}, "
          f"residual {b['residual'][0]:.4f})")
    if 'real' in res:
        rr = res['real']
        best = min(('evidence', 'sure', 'gcv'), key=lambda c: rr[c][0])
        g3 = rr[best][0] < rr['residual'][0]
        print(f"  3) beats the residual gate on real       : "
              f"{'PASS' if g3 else 'FAIL'}"
              f"  ({best} {rr[best][0]:.4f} vs residual {rr['residual'][0]:.4f})")

    os.makedirs('results/direction3', exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(res, f, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
