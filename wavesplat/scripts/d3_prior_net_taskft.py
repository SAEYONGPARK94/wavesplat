"""
d3_prior_net_taskft.py — v3: task-loss fine-tuning of the amortized prior
net. The v2 label-regression net only reprocesses the information the hand
prior already uses (G1: parity on moving, -0.05 on fixed). Here the net's
gamma(k) is trained THROUGH the differentiable MMSE (Woodbury solve in
torch) to minimize FIELD error at held-out query pixels — the actual task
— which also sidesteps the stochastic-label noise floor (gamma* is one
200-mode draw). A global log-noise-multiplier is learned jointly.

Init from the v2 checkpoint. Data: shards from d3_prior_net_data.py with
Q/gt_q (task0 batch).

Usage: python3 scripts/d3_prior_net_taskft.py [--steps 4000]
       [--data tmp/prior_train/task0] [--init results/direction3/prior_net_v2.pt]
       [--ckpt results/direction3/prior_net_v3.pt]
"""
import os
import sys
import glob
import time
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from d3_prior_net_train import PriorNet, band_mask, NGRID, KREL

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
G = 9.81
DEPTH = 50.0


def load_all(stem):
    ks = ('X', 'P', 'z', 'Q', 'gt_q', 'meta')
    acc = {k: [] for k in ks}
    for p in sorted(glob.glob(f"{stem}_s*.npz")):
        d = np.load(p)
        for k in ks:
            acc[k].append(d[k])
    return {k: np.concatenate(v) for k, v in acc.items()}


def _build_bank(net_out, z, kp_e, mask_t, kxn, kyn, bound_wave, bw_scale):
    """Shared dispersion-constrained bank + variance-constrained prior.

    Factored out of task_loss_one (2026-07-19, E-B2K) so that the task arm and
    the NLL arm are guaranteed to train through bit-identical bank machinery --
    the loss is then the ONLY thing that differs between arms, which is what the
    loss-swap claim requires. Verified bit-identical to the pre-factor path by
    tmp/e60k_verify_refactor.py."""
    g = torch.pow(10.0, net_out[mask_t])
    kxb = kxn * kp_e
    kyb = kyn * kp_e
    k = torch.sqrt(kxb ** 2 + kyb ** 2 + 1e-12)
    om = torch.sqrt(G * k * torch.tanh(k * DEPTH) + 1e-12)
    if bound_wave:
        kxb = torch.cat([kxb, 2.0 * kxb])
        kyb = torch.cat([kyb, 2.0 * kyb])
        om = torch.cat([om, 2.0 * om])
        g = torch.cat([g, bw_scale * g])
    g = g / g.sum()
    gam = torch.cat([g, g]) * torch.var(z)

    def design(Qp):
        psi = kxb[None] * Qp[:, 0:1] + kyb[None] * Qp[:, 1:2] \
            - om[None] * Qp[:, 2:3]
        return torch.cat([torch.cos(psi), torch.sin(psi)], dim=1)

    return gam, design


def task_loss_one(net_out, P, z, Q, gt_q, kp_e, sigma0, log_smult, mask_t,
                  kxn, kyn, bound_wave=False, bw_scale=0.15):
    """Differentiable MMSE through Woodbury; returns normalized field MSE.
    bound_wave: pair the learned free-wave prior with (2k,2omega) off-shell
    bound-wave candidates (#2+#1 retrain, 2026-07-08), prior tied to the parent
    free-wave energy and downscaled — matches inference net_gamma_mmse."""
    gam, design = _build_bank(net_out, z, kp_e, mask_t, kxn, kyn,
                              bound_wave, bw_scale)
    Phi = design(P)
    s2 = (torch.exp(log_smult) * sigma0) ** 2
    n = len(z)
    K = (Phi * gam[None]) @ Phi.T + s2 * torch.eye(n, device=DEVICE)
    alpha = torch.linalg.solve(K, z)
    pred = (design(Q) * gam[None]) @ (Phi.T @ alpha)
    return ((pred - gt_q) ** 2).mean()


def nll_loss_one(net_out, P, z, kp_e, sigma0, log_smult, mask_t, kxn, kyn,
                 bound_wave=False, bw_scale=0.15):
    """Arm NLL (E-B2K, 2026-07-19): TRUE marginal likelihood (ML-II).

    Under the same model that task_loss_one inverts -- z = Phi a + eps,
    a ~ N(0, Gamma), eps ~ N(0, sigma^2 I) -- the marginal is z ~ N(0, K) with
    K = Phi Gamma Phi^T + sigma^2 I, so

        -log p(z) = 0.5 * [ z^T K^-1 z + logdet K + n log 2pi ].

    Returned PER OBSERVATION (divided by n) so the loss magnitude is O(1) and
    comparable to the other arms under the shared learning rate -- the arms must
    differ in objective, not in effective step size.

    Why this arm exists: the shipped arm L is a held-out z-residual, i.e. a
    CROSS-VALIDATION estimator, not ML-II. Bachoc (CSDA 2013) established at
    theorem level that ML mis-selects under misspecification while CV is robust,
    so an L-only comparison cannot license the word "likelihood" in the method
    claim. This arm closes that measurement gap and doubles as the audit against
    arm L being a straw man (it uses ALL observations, L discards 20%).
    """
    gam, design = _build_bank(net_out, z, kp_e, mask_t, kxn, kyn,
                              bound_wave, bw_scale)
    Phi = design(P)
    s2 = (torch.exp(log_smult) * sigma0) ** 2
    n = len(z)
    K = (Phi * gam[None]) @ Phi.T + s2 * torch.eye(n, device=DEVICE)
    # Cholesky: stable logdet on the same K the other arms invert.
    Lc = torch.linalg.cholesky(K)
    alpha = torch.cholesky_solve(z[:, None], Lc)[:, 0]
    quad = (z * alpha).sum()
    logdet = 2.0 * torch.log(torch.diagonal(Lc)).sum()
    return 0.5 * (quad + logdet + n * float(np.log(2 * np.pi))) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--data', default='tmp/prior_train/task0')
    ap.add_argument('--init', default='results/direction3/prior_net_v2.pt')
    ap.add_argument('--ckpt', default='results/direction3/prior_net_v3.pt')
    ap.add_argument('--bound_wave', action='store_true',
                    help='train the prior WITH the (2k,2omega) bound-wave bank '
                         '(#2+#1 retrain, 2026-07-08)')
    a = ap.parse_args()

    D = load_all(a.data)
    n_all = len(D['X'])
    n_val = max(200, n_all // 12)
    print(f"data {a.data}: {n_all} samples ({n_val} val)")
    net = PriorNet().to(DEVICE)
    ck = torch.load(a.init, map_location=DEVICE)
    net.load_state_dict(ck['model'])
    log_smult = torch.tensor(float(np.log(3.0)), device=DEVICE,
                             requires_grad=True)
    opt = torch.optim.AdamW([{'params': net.parameters(), 'lr': a.lr},
                             {'params': [log_smult], 'lr': 3e-3}],
                            weight_decay=1e-5)
    mask_t = torch.tensor(band_mask(), device=DEVICE)
    axn = torch.linspace(-KREL, KREL, NGRID, device=DEVICE)
    KXn, KYn = torch.meshgrid(axn, axn, indexing='xy')
    kxn, kyn = KXn[mask_t], KYn[mask_t]

    def sample_loss(i, out):
        P = torch.tensor(D['P'][i], device=DEVICE)
        z = torch.tensor(D['z'][i], device=DEVICE)
        Q = torch.tensor(D['Q'][i], device=DEVICE)
        gt = torch.tensor(D['gt_q'][i], device=DEVICE)
        kp_e, Hs, _, _, nfrac = D['meta'][i]
        sig0 = float(nfrac * Hs)
        return task_loss_one(out, P, z, Q, gt, float(kp_e), sig0,
                             log_smult, mask_t, kxn, kyn,
                             bound_wave=a.bound_wave) / (Hs / 4) ** 2

    rng = np.random.default_rng(0)
    vi = np.arange(n_val)
    ti = np.arange(n_val, n_all)
    best = (np.inf, 0)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        ii = rng.choice(ti, a.bs, replace=False)
        X = torch.tensor(D['X'][ii], device=DEVICE)
        outs = net(X)
        loss = sum(sample_loss(int(i), outs[j])
                   for j, i in enumerate(ii)) / a.bs
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % 200 == 0 or step == a.steps:
            net.eval()
            with torch.no_grad():
                vsub = vi[:160]
                vls = []
                for i0 in range(0, len(vsub), a.bs):
                    jj = vsub[i0:i0 + a.bs]
                    Xv = torch.tensor(D['X'][jj], device=DEVICE)
                    ov = net(Xv)
                    vls += [float(sample_loss(int(i), ov[j]))
                            for j, i in enumerate(jj)]
                vl = float(np.mean(vls))
            net.train()
            tag = ''
            if vl < best[0]:
                best = (vl, step)
                torch.save(dict(model=net.state_dict(), step=step, val=vl,
                                log_smult=float(log_smult),
                                ngrid=NGRID, krel=KREL), a.ckpt)
                tag = ' *saved'
            print(f"step {step} train {float(loss):.4f} val {vl:.4f} "
                  f"smult {float(torch.exp(log_smult)):.2f} "
                  f"({time.time()-t0:.0f}s){tag}", flush=True)
    print(f"best val {best[0]:.4f} @ {best[1]} -> {a.ckpt}")


if __name__ == '__main__':
    main()
