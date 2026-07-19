"""E-B2K -- prove the _build_bank refactor did not perturb the frozen T/L path.

task_loss_one is the training path of the SHIPPED arms T and L. Factoring the
bank build out of it is only admissible if it is bit-identical. This reimplements
the pre-factor body inline and compares, on random inputs, both the value and the
gradient w.r.t. the two learned hyper-paths (net_out, log_smult).
Also sanity-checks the new NLL arm against a naive slogdet reference.
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from d3_prior_net_train import band_mask, NGRID, KREL
from d3_prior_net_taskft import task_loss_one, nll_loss_one, DEVICE, G, DEPTH


def task_loss_one_PREFACTOR(net_out, P, z, Q, gt_q, kp_e, sigma0, log_smult,
                            mask_t, kxn, kyn, bound_wave=False, bw_scale=0.15):
    """Verbatim copy of the body as it stood before 2026-07-19."""
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

    Phi = design(P)
    s2 = (torch.exp(log_smult) * sigma0) ** 2
    n = len(z)
    K = (Phi * gam[None]) @ Phi.T + s2 * torch.eye(n, device=DEVICE)
    alpha = torch.linalg.solve(K, z)
    pred = (design(Q) * gam[None]) @ (Phi.T @ alpha)
    return ((pred - gt_q) ** 2).mean()


def main():
    torch.manual_seed(0)
    mask_t = torch.tensor(band_mask(), device=DEVICE)
    axn = torch.linspace(-KREL, KREL, NGRID, device=DEVICE)
    KXn, KYn = torch.meshgrid(axn, axn, indexing='xy')
    kxn, kyn = KXn[mask_t], KYn[mask_t]

    print(f'device {DEVICE}   bank atoms {int(mask_t.sum())}')
    worst_v = worst_gn = worst_gs = 0.0
    for trial in range(6):
        n, nq = 120, 40
        P = torch.rand(n, 3, device=DEVICE) * torch.tensor(
            [80., 80., 6.], device=DEVICE)
        Q = torch.rand(nq, 3, device=DEVICE) * torch.tensor(
            [80., 80., 6.], device=DEVICE)
        z = torch.randn(n, device=DEVICE) * 0.4
        gt = torch.randn(nq, device=DEVICE) * 0.4
        kp_e, sigma0 = 0.11, 0.03
        bw = bool(trial % 2)

        outs, vals, grads = [], [], []
        for fn in (task_loss_one, task_loss_one_PREFACTOR):
            no = torch.randn(NGRID, NGRID, device=DEVICE, requires_grad=True)
            ls = torch.tensor(np.log(3.0), device=DEVICE, requires_grad=True)
            torch.manual_seed(100 + trial)          # identical no/ls per fn
            with torch.no_grad():
                no.copy_(torch.randn(NGRID, NGRID, device=DEVICE))
            v = fn(no, P, z, Q, gt, kp_e, sigma0, ls, mask_t, kxn, kyn,
                   bound_wave=bw)
            v.backward()
            vals.append(float(v))
            grads.append((no.grad.clone(), ls.grad.clone()))
            outs.append(no)

        dv = abs(vals[0] - vals[1])
        dgn = float((grads[0][0] - grads[1][0]).abs().max())
        dgs = float((grads[0][1] - grads[1][1]).abs().max())
        worst_v = max(worst_v, dv)
        worst_gn = max(worst_gn, dgn)
        worst_gs = max(worst_gs, dgs)
        print(f'  trial {trial} bw={int(bw)}  loss {vals[0]:.10f} vs '
              f'{vals[1]:.10f}   d_val {dv:.2e}  d_grad(net) {dgn:.2e}  '
              f'd_grad(smult) {dgs:.2e}')

    ok = worst_v == 0.0 and worst_gn == 0.0 and worst_gs == 0.0
    print(f'\nrefactor bit-identical: {ok}  '
          f'(worst d_val {worst_v:.2e}, d_grad {max(worst_gn, worst_gs):.2e})')

    # --- NLL arm sanity vs naive reference -------------------------------
    print('\nNLL arm vs naive slogdet reference:')
    no = torch.randn(NGRID, NGRID, device=DEVICE)
    ls = torch.tensor(np.log(3.0), device=DEVICE)
    n = 120
    P = torch.rand(n, 3, device=DEVICE) * torch.tensor([80., 80., 6.],
                                                       device=DEVICE)
    z = torch.randn(n, device=DEVICE) * 0.4
    v_new = float(nll_loss_one(no, P, z, 0.11, 0.03, ls, mask_t, kxn, kyn))

    from d3_prior_net_taskft import _build_bank
    gam, design = _build_bank(no, z, 0.11, mask_t, kxn, kyn, False, 0.15)
    Phi = design(P)
    K = (Phi * gam[None]) @ Phi.T + (torch.exp(ls) * 0.03) ** 2 * torch.eye(
        n, device=DEVICE)
    sign, ld = torch.linalg.slogdet(K)
    quad = z @ torch.linalg.solve(K, z)
    v_ref = float(0.5 * (quad + ld + n * np.log(2 * np.pi)) / n)
    print(f'  cholesky {v_new:.8f}   slogdet {v_ref:.8f}   '
          f'rel {abs(v_new - v_ref) / abs(v_ref):.2e}   logdet sign {int(sign)}')

    # gradient must flow to BOTH hyper-paths, else the arm is degenerate
    no2 = no.clone().requires_grad_(True)
    ls2 = ls.clone().requires_grad_(True)
    nll_loss_one(no2, P, z, 0.11, 0.03, ls2, mask_t, kxn, kyn).backward()
    print(f'  grad flows: net |g|max {float(no2.grad.abs().max()):.3e}  '
          f'smult g {float(ls2.grad):.3e}')
    print('\nVERDICT:', 'PASS' if ok else 'FAIL -- refactor perturbed T/L path')


if __name__ == '__main__':
    main()
