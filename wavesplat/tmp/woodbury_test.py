"""Verify the new rank-2M Woodbury loglik_atoms == dense O(m^3) reference (value AND
gradient), then benchmark wall-time scaling in m. Run: python3 tmp/woodbury_test.py
"""
import os, sys, time, math, numpy as np, torch
sys.path.insert(0, 'scripts')
import d3_setdiff_posterior as DP
from d3_setdiff_posterior import loglik_atoms  # the new Woodbury version

DEV = DP.DEVICE
G = DP.G
SIGMA = DP.SIGMA
S2 = SIGMA**2 + 1e-4


def dense_ref(kx, ky, a, P, z, depth):
    """Old dense O(m^3) form (fixed jitter = S2, matches pre-Woodbury code)."""
    k = torch.sqrt(kx**2 + ky**2 + 1e-12)
    om = torch.sqrt(G * k * torch.tanh(k * depth) + 1e-12)
    dx = P[:, 0:1] - P[None, :, 0]; dy = P[:, 1:2] - P[None, :, 1]; dt = P[:, 2:3] - P[None, :, 2]
    phase = (kx[:, None, None]*dx[None] + ky[:, None, None]*dy[None] - om[:, None, None]*dt[None])
    K = torch.einsum('m,mij->ij', a, torch.cos(phase))
    eye = torch.eye(len(z), device=P.device, dtype=P.dtype)
    Lc = torch.linalg.cholesky(K + S2 * eye)
    alpha = torch.cholesky_solve(z[:, None], Lc)[:, 0]
    return -0.5 * (z @ alpha) - torch.log(torch.diag(Lc)).sum()


def make(m, M, seed=0):
    g = torch.Generator(device='cpu').manual_seed(seed)
    kx = (torch.rand(M, generator=g) - 0.5) * 0.6
    ky = (torch.rand(M, generator=g) - 0.5) * 0.6
    a = torch.rand(M, generator=g) * 0.05
    P = torch.zeros(m, 3)
    P[:, 0] = (torch.rand(m, generator=g) - 0.5) * 400
    P[:, 1] = (torch.rand(m, generator=g) - 0.5) * 300
    P[:, 2] = torch.rand(m, generator=g) * 20
    z = (torch.rand(m, generator=g) - 0.5) * 2
    return [t.to(DEV).double() for t in (kx, ky, a, P, z)]


print("=== EQUIVALENCE (double precision) ===")
for m, M in [(64, 8), (200, 32), (500, 32), (800, 16)]:
    kx, ky, a, P, z = make(m, M, seed=m)
    depth = torch.tensor(50.0, device=DEV, dtype=torch.float64)
    # values
    v_new = loglik_atoms(kx, ky, a, P, z, depth)
    v_ref = dense_ref(kx, ky, a, P, z, depth)
    # gradients wrt (kx,ky,a)
    def gnorm(fn):
        kk = kx.clone().requires_grad_(True); ll = ky.clone().requires_grad_(True); aa = a.clone().requires_grad_(True)
        out = fn(kk, ll, aa, P, z, depth); gk, gl, ga = torch.autograd.grad(out, (kk, ll, aa))
        return torch.cat([gk, gl, ga])
    g_new = gnorm(loglik_atoms); g_ref = gnorm(dense_ref)
    val_err = abs((v_new - v_ref).item()) / (abs(v_ref.item()) + 1e-9)
    grad_err = (g_new - g_ref).norm().item() / (g_ref.norm().item() + 1e-9)
    print(f"  m={m:4d} M={M:3d}  value rel-err={val_err:.2e}  grad rel-err={grad_err:.2e}  "
          f"(new={v_new.item():.4f} ref={v_ref.item():.4f})")

print("\n=== SPEED (float32, GPU, forward+backward, 20 reps) ===")
depth = torch.tensor(50.0, device=DEV, dtype=torch.float32)
for m in [200, 500, 1000, 2000, 3000]:
    kx, ky, a, P, z = [t.float() for t in make(m, 32, seed=1)]
    def bench(fn):
        kk = kx.clone().requires_grad_(True); ll = ky.clone().requires_grad_(True); aa = a.clone().requires_grad_(True)
        if DEV.type == 'cuda': torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(20):
            out = fn(kk, ll, aa, P, z, depth); torch.autograd.grad(out, (kk, ll, aa))
        if DEV.type == 'cuda': torch.cuda.synchronize()
        return (time.time() - t0) / 20 * 1e3   # ms/call
    t_new = bench(loglik_atoms)
    try:
        t_ref = bench(dense_ref) if m <= 2000 else float('nan')
    except RuntimeError:
        t_ref = float('nan')
    print(f"  m={m:4d}  Woodbury {t_new:7.2f} ms   dense {t_ref:8.2f} ms   speedup "
          f"{(t_ref/t_new if t_ref==t_ref else float('nan')):.1f}x")
print("\nT=500 diffusion steps × nsamp × frames × episodes — per-step cost dominates.")
