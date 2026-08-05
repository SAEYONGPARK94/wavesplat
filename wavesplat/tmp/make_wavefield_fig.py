"""Static wave-field estimation figure — the thesis headline panel:
HOS-like sea truth | net-MMSE reconstruction | |error| | sigma_CRB operability envelope.
Reconstruct one episode on a regular grid. Out: tmp/fig_wavefield.png
"""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'scripts'); sys.path.insert(0, 'tmp')
from scripts.d1_crb import DT, DEVICE, load_ep, collect_obs, predict, jacobian, fisher
from d3_prior_net_eval import get_net, net_gamma_mmse
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

SIGMA = 0.02
def dom_kp(th):
    w = th[:, 0].abs(); i = int(torch.argmax(w)); return float(torch.hypot(th[i, 1], th[i, 2]))
def kgrid(kp, nk=25):
    ax = np.linspace(-1.75*kp, 1.75*kp, nk); KX, KY = np.meshgrid(ax, ax)
    kx, ky = KX.ravel(), KY.ravel(); km = np.hypot(kx, ky)/kp; keep = (km >= 0.35) & (km <= 1.75)
    return kx[keep], ky[keep]
def design_np(kx, ky, P):
    om = np.sqrt(9.81*np.hypot(kx, ky)*np.tanh(np.hypot(kx, ky)*50.)+1e-12)
    psi = kx[None]*P[:, 0:1] + ky[None]*P[:, 1:2] - om[None]*P[:, 2:3]
    return np.concatenate([np.cos(psi), np.sin(psi)], 1)

ep = load_ep('data/episode_BF4_15comp_seed0_uniform.h5')
depth = ep['depth']; th0 = ep['theta']; N = th0.shape[0]; kp = dom_kp(th0)
t_now = float(ep['ts'][len(ep['ts'])//2])
x, y, t = collect_obs(ep, t_now)
g = torch.Generator(device=DEVICE).manual_seed(0)
nkeep = min(600, x.numel()); idx = torch.randperm(x.numel(), device=DEVICE, generator=g)[:nkeep]; x, y, t = x[idx], y[idx], t[idx]
thr = th0.clone(); thr[:, 3] = torch.rand(N, dtype=DT, device=DEVICE, generator=g)*2*np.pi
# regular grid
gxn = np.linspace(float(ep['gx'][0]), float(ep['gx'][-1]), 90)
gyn = np.linspace(float(ep['gy'][0]), float(ep['gy'][-1]), 90)
GX, GY = np.meshgrid(gxn, gyn)
qx = torch.tensor(GX.ravel(), dtype=DT, device=DEVICE); qy = torch.tensor(GY.ravel(), dtype=DT, device=DEVICE)
qt = torch.full_like(qx, float(t_now))
truth = predict(thr, qx, qy, qt, depth).cpu().numpy().reshape(GX.shape)
s = predict(thr, x, y, t, depth); z = s + SIGMA*torch.randn(s.shape, dtype=DT, device=DEVICE, generator=g)
P = torch.stack([x, y, t], 1).cpu().numpy(); Q = torch.stack([qx, qy, qt], 1).cpu().numpy()
recon = np.asarray(net_gamma_mmse(get_net(), P, z.cpu().numpy(), depth, Q, SIGMA, seed=0, kp_hint=None, smult_fixed=3.25)).reshape(GX.shape)
err = np.abs(recon - truth)
# sigma_CRB envelope at fitted theta_hat (RELAX)
kx, ky = kgrid(kp); C = design_np(kx, ky, P); Nc = len(kx); r = z.cpu().numpy().copy(); sel = []
cos_c, sin_c = C[:, :Nc], C[:, Nc:]
for _ in range(15):
    pw = (cos_c.T@r)**2 + (sin_c.T@r)**2
    if sel: pw[sel] = -1
    j = int(np.argmax(pw)); sel.append(j)
    Ps = np.concatenate([cos_c[:, sel], sin_c[:, sel]], 1); coef = np.linalg.lstsq(Ps, z.cpu().numpy(), rcond=None)[0]; r = z.cpu().numpy() - Ps@coef
kxs, kys = kx[sel], ky[sel]; m = len(sel); ac, as_ = coef[:m], coef[m:]
theta_h = np.stack([np.sqrt(ac**2+as_**2), kxs, kys, np.arctan2(-as_, ac)], 1)
th_t = torch.tensor(theta_h, dtype=DT, device=DEVICE)
F = fisher(th_t, x, y, t, depth, SIGMA); nF = F.shape[0]
Finv = torch.linalg.inv(F + 1e-8*torch.trace(F)/nF*torch.eye(nF, dtype=DT, device=DEVICE))
Jq = jacobian(th_t, qx, qy, qt, depth)
sig_crb = (torch.sqrt(torch.clamp(torch.einsum('qa,ab,qb->q', Jq, Finv, Jq), min=0))/(4*np.std(z.cpu().numpy()))).cpu().numpy().reshape(GX.shape)

# --- plot ---
fig, ax = plt.subplots(1, 4, figsize=(18, 4.6))
vmax = np.nanpercentile(np.abs(truth), 99)
ims = []
ims.append(ax[0].pcolormesh(GX, GY, truth, cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')); ax[0].set_title('Sea surface TRUTH (m)', fontsize=12)
ims.append(ax[1].pcolormesh(GX, GY, recon, cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')); ax[1].set_title('Estimated (learned net-MMSE)', fontsize=12)
ims.append(ax[2].pcolormesh(GX, GY, err, cmap='magma', shading='auto')); ax[2].set_title('|error| (m)', fontsize=12)
ims.append(ax[3].pcolormesh(GX, GY, sig_crb, cmap='viridis', shading='auto')); ax[3].set_title('σ_CRB operability envelope\n(design-time trust map, no estimator run)', fontsize=11)
for i, a in enumerate(ax):
    px, py = P[:, 0], P[:, 1]
    a.scatter(px, py, s=1.5, c='k', alpha=0.35 if i < 3 else 0.2)   # obs points
    a.set_aspect('equal'); a.set_xlabel('x (m)'); a.set_xlim(gxn[0], gxn[-1]); a.set_ylim(gyn[0], gyn[-1])
    fig.colorbar(ims[i], ax=a, fraction=0.046, pad=0.04)
ax[0].set_ylabel('y (m)')
c = float(np.corrcoef(recon.ravel(), truth.ravel())[0, 1])
fig.suptitle(f'Phase-resolved wave estimation from sparse ({len(P)}-pt) ship-borne observations  —  reconstruction corr {c:.2f}  |  black dots = LiDAR obs', fontsize=13, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig('tmp/fig_wavefield.png', dpi=130, bbox_inches='tight')
print(f'saved tmp/fig_wavefield.png  (recon corr {c:.3f})')
