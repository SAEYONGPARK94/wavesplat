"""STAGE-2 e2e-style fine-tune of the beamform-conditioned set-diffusion proposal net.

ADDITIVE. Imports scripts/d3_beamform_cond_diffusion (never modifies it). Reuses the SAME
synthetic scene generator distribution the current checkpoint was trained with (make_batch's
32-comp JONSWAP, Hs~U[0.8,3.2], Tp~U[5,11], dir~U[-50,50], spread~U[12,40], gamma~U[1.5,4.0],
depth=50, M_OBS=400 obs over the LiDAR footprint pool, noise 0.02, linear forward). Training
scenes are freshly sampled synthetic (held-out from the fixed finemesh eval episodes).

ROUTE THROUGH THE DDPM-SAMPLER BLOCKER: x-hat0 (x0-prediction) FIELD-CONSISTENCY fine-tune.
The eval PRIMARY runs the *identical frozen* ancestral DDPM sampler + polish + pool_and_select on
the fine-tuned weights, so fine-tuning must improve the eps-predictor's implicit x0 decode -- which
every one of the 500 ancestral steps computes. We therefore supervise x0-hat directly (blocker #1's
"supervise the x0-prediction directly" option), the minimal-risk route: NO change to the eval
sampler, NO distillation / new head (no architecture risk, no second model, checkpoint format is an
identical CondSetDenoiser state_dict), and NO backprop through 500 stochastic steps.

LOSS = anchor eps-MSE (all samples, all t -> preserves the generative model + the energy channel
used by pool_and_select) + lambda_field * field_MSE-on-x0-hat (a low-t subset only).
The field term: x0-hat -> denorm_atoms (kx,ky,energy) -> differentiable torch-LS solves w/phi on
the obs -> render at a held-out spatial grid at a nowcast time -> (RMSE/Hs)^2 vs the true field.
Because w/phi are an LS function of kx/ky and the amplitude is re-solved, the field gradient lands
almost entirely on the kx/ky component PLACEMENT -- the exact quantity Stage-1 showed carries the
whole integrated advantage (learned_konly ~= learned).

Usage:
  python3 tmp/wp_e2e_2026-07-25/e2e_finetune.py --config A --steps 2500 --out results/direction3/prior_net_e2eft_A.pt
  python3 tmp/wp_e2e_2026-07-25/e2e_finetune.py --smoke
"""
import os, sys, json, time, argparse, math
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, 'scripts')
import d3_beamform_cond_diffusion as cd            # os.chdir's to repo root on import
torch.set_num_threads(4)
from d3_setdiff_prior import denorm_atoms
from wavesplat.model.tnse import render_wave_field_torch, differentiable_ls_fit

DEVICE = cd.DEVICE
M = cd.M                 # 32 atoms
T = cd.T                 # 500 diffusion steps
DEPTH = cd.DEPTH         # 50.0
M_OBS = cd.M_OBS         # 400
K_TP = cd.K_TP           # 2.2
LAMBDA_REG = 0.5         # LS ridge, matches fit_amplitudes_phases / atoms_to_field
NGRID = 20               # held-out render grid is NGRID x NGRID over the obs footprint


# ── scene generation: mirrors cd.make_batch EXACTLY, but also returns obs + true comps ──
def make_scene_batch(Bsz, pool, stats, grid_bounds, rng_np):
    """Returns everything make_batch produces (normalized atoms x0, beamform map) PLUS the
    obs (P,z), true components (kx,ky,amp,phi), Hs, and the true field on a held-out grid at a
    nowcast time t_target = Tobs/2. Distribution identical to cd.make_batch (linear forward)."""
    K_LIM = stats['K_LIM']; lo_e, hi_e = stats['lo_e'], stats['hi_e']
    atoms = np.zeros((Bsz, M, 3), np.float32)
    P = np.zeros((Bsz, M_OBS, 3), np.float32); Z = np.zeros((Bsz, M_OBS), np.float32)
    kx_t = np.zeros((Bsz, M), np.float32); ky_t = np.zeros((Bsz, M), np.float32)
    amp_t = np.zeros((Bsz, M), np.float32); phi_t = np.zeros((Bsz, M), np.float32)
    Hs_b = np.zeros(Bsz, np.float32); ttar = np.zeros(Bsz, np.float32)
    for b in range(Bsz):
        Hs = rng_np.uniform(0.8, 3.2); Tp = rng_np.uniform(5.0, 11.0)
        dird = rng_np.uniform(-50, 50); spread = rng_np.uniform(12, 40)
        gamma = rng_np.uniform(1.5, 4.0)
        c = cd.jonswap_components(Hs, Tp, n_components=M, depth=DEPTH, gamma=gamma,
                                  direction_deg=dird, direction_spread_deg=spread,
                                  seed=int(rng_np.integers(1e9)))
        kx = np.array([x.kx for x in c], np.float32); ky = np.array([x.ky for x in c], np.float32)
        amp = np.array([x.A for x in c], np.float32); om = np.array([x.omega for x in c])
        ph = np.array([x.phi for x in c], np.float32)
        atoms[b, :, 0] = kx / K_LIM; atoms[b, :, 1] = ky / K_LIM
        atoms[b, :, 2] = 2 * (np.log(amp ** 2 + 1e-12) - lo_e) / (hi_e - lo_e) - 1
        Tobs = K_TP * Tp
        idx = rng_np.integers(0, len(pool), M_OBS)
        ox, oy = pool[idx, 0], pool[idx, 1]; ot = rng_np.uniform(0, Tobs, M_OBS)
        Z[b] = cd.elevation(kx, ky, amp, om, ph, ox, oy, ot, DEPTH, nonlinear=False) \
            + rng_np.normal(0, 0.02, M_OBS)
        P[b] = np.column_stack([ox, oy, ot])
        kx_t[b] = kx; ky_t[b] = ky; amp_t[b] = amp; phi_t[b] = ph
        Hs_b[b] = Hs; ttar[b] = 0.5 * Tobs
    Pt = torch.tensor(P, device=DEVICE); Zt = torch.tensor(Z, device=DEVICE)
    Bmap = cd.beamform(Pt, Zt)
    # held-out render grid over the pooled footprint (same for all samples, per-sample t_target)
    gx = np.linspace(grid_bounds[0], grid_bounds[1], NGRID)
    gy = np.linspace(grid_bounds[2], grid_bounds[3], NGRID)
    GX, GY = np.meshgrid(gx, gy)
    gxf = torch.tensor(GX.reshape(-1), device=DEVICE, dtype=torch.float32)   # (NG2,)
    gyf = torch.tensor(GY.reshape(-1), device=DEVICE, dtype=torch.float32)
    NG2 = gxf.shape[0]
    gxb = gxf[None].expand(Bsz, -1).contiguous()                            # (B,NG2)
    gyb = gyf[None].expand(Bsz, -1).contiguous()
    ttb = torch.tensor(ttar, device=DEVICE)[:, None].expand(-1, NG2).contiguous()
    kxT = torch.tensor(kx_t, device=DEVICE); kyT = torch.tensor(ky_t, device=DEVICE)
    ampT = torch.tensor(amp_t, device=DEVICE); phiT = torch.tensor(phi_t, device=DEVICE)
    with torch.no_grad():
        gt_field = render_wave_field_torch(kxT, kyT, gxb, gyb, ttb, DEPTH,
                                           A=ampT, phi=phiT)                 # (B,NG2)
    x0 = torch.tensor(atoms, device=DEVICE)
    HsT = torch.tensor(Hs_b, device=DEVICE)
    return dict(x0=x0, Bmap=Bmap, Pt=Pt, Zt=Zt, gxb=gxb, gyb=gyb, ttb=ttb,
                gt_field=gt_field, Hs=HsT)


# ── differentiable field loss on the x0-hat decode ───────────────────────────
def field_loss_from_x0hat(x0hat, batch, stats):
    """x0hat (b,M,3) normalized atoms -> LS w/phi on obs -> render on held-out grid ->
    mean over samples of (RMSE/Hs)^2 vs true field. Differentiable in kx/ky (and energy)."""
    kx, ky, _a = denorm_atoms(x0hat, stats)                 # (b,M) each
    Pt = batch['Pt']; Zt = batch['Zt']
    ox = Pt[..., 0]; oy = Pt[..., 1]; ot = Pt[..., 2]       # (b,M_OBS)
    # design matrix at obs -> differentiable ridge LS -> coef=[a_1..a_N,b_1..b_N]
    Phi_obs = render_wave_field_torch(kx, ky, ox, oy, ot, DEPTH)   # (b,M_OBS,2N)
    coef = differentiable_ls_fit(Phi_obs, Zt, lambda_reg=LAMBDA_REG)  # (b,2N)
    # render prediction on the held-out grid with the SAME (kx,ky) -> same dispersion as GT
    Phi_g = render_wave_field_torch(kx, ky, batch['gxb'], batch['gyb'], batch['ttb'], DEPTH)
    pred = (Phi_g @ coef.unsqueeze(2)).squeeze(2)          # (b,NG2)
    gt = batch['gt_field']                                  # (b,NG2)
    mse = ((pred - gt) ** 2).mean(dim=1)                    # (b,)
    hs2 = torch.clamp(batch['Hs'], min=1e-6) ** 2
    return (mse / hs2).mean()


def finetune(cfg, log=print):
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed'])
    rng_np = np.random.default_rng(cfg['seed'])
    stats = torch.load(cd.PRIOR_CKPT, map_location=DEVICE)['stats']
    betas = cd.cosine_betas(T).to(DEVICE)
    abar = torch.cumprod(1 - betas, 0)
    pool = cd.pool_xy().astype(np.float32)
    gb = (float(pool[:, 0].min()), float(pool[:, 0].max()),
          float(pool[:, 1].min()), float(pool[:, 1].max()))
    log(f"  footprint grid bounds x[{gb[0]:.1f},{gb[1]:.1f}] y[{gb[2]:.1f},{gb[3]:.1f}], "
        f"{NGRID}x{NGRID} render grid")

    model = cd.CondSetDenoiser().to(DEVICE)
    model.load_state_dict(torch.load(cfg['init_ckpt'], map_location=DEVICE))
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg['steps'])

    Bsz = cfg['bsz']; nf = Bsz // 2            # field-half (low-t) size
    lam = cfg['lambda_field']; t_lo = cfg['t_lo']
    curve = []
    t0 = time.time()
    for s in range(cfg['steps']):
        batch = make_scene_batch(Bsz, pool, stats, gb, rng_np)
        x0 = batch['x0']
        # per-sample timesteps: anchor half over [0,T), field half over [0,t_lo)
        t_anchor = torch.randint(0, T, (Bsz - nf,), device=DEVICE)
        t_field = torch.randint(0, t_lo, (nf,), device=DEVICE)
        t = torch.cat([t_anchor, t_field], 0)
        ab = abar[t][:, None, None]
        eps = torch.randn_like(x0)
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        eps_pred = model(xt, t, batch['Bmap'])
        anchor = F.mse_loss(eps_pred, eps)
        # x0-hat decode on the field half (bound-clamp like the samplers)
        xf = xt[Bsz - nf:]; epf = eps_pred[Bsz - nf:]; abf = ab[Bsz - nf:]
        x0hat = ((xf - (1 - abf).sqrt() * epf) / abf.sqrt()).clamp(-1, 1)
        fbatch = {k: (v[Bsz - nf:] if torch.is_tensor(v) and v.shape[0] == Bsz else v)
                  for k, v in batch.items()}
        fld = field_loss_from_x0hat(x0hat, fbatch, stats)
        loss = anchor + lam * fld
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sched.step()
        if s % 100 == 0 or s == cfg['steps'] - 1:
            curve.append(dict(step=s, anchor=float(anchor), field=float(fld),
                              total=float(loss), sec=time.time() - t0))
            log(f"  step {s:5d}  anchor {anchor.item():.5f}  field {fld.item():.5f}  "
                f"total {loss.item():.5f}  ({time.time()-t0:.0f}s)")
    torch.save(model.state_dict(), cfg['out'])
    log(f"  saved {cfg['out']}  ({time.time()-t0:.0f}s)")
    return curve


CONFIGS = {
    # both pre-registered (max 2); selection by held-out synthetic validation field-nRMSE
    'A': dict(lambda_field=1.0, t_lo=T // 4, lr=5e-5, steps=4000, bsz=32, seed=0),
    'B': dict(lambda_field=4.0, t_lo=T // 4, lr=5e-5, steps=4000, bsz=32, seed=0),
}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', choices=['A', 'B'], default='A')
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--init_ckpt', default='results/direction3/beamform_cond_diffusion.pt')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()

    if a.smoke:
        stats = torch.load(cd.PRIOR_CKPT, map_location=DEVICE)['stats']
        betas = cd.cosine_betas(T).to(DEVICE); abar = torch.cumprod(1 - betas, 0)
        pool = cd.pool_xy().astype(np.float32)
        gb = (float(pool[:, 0].min()), float(pool[:, 0].max()),
              float(pool[:, 1].min()), float(pool[:, 1].max()))
        print("footprint bounds", gb)
        # dispersion parity check: jonswap omega vs sqrt(g k tanh kd)
        c = cd.jonswap_components(2.0, 8.0, n_components=8, depth=DEPTH, seed=1)
        k = np.array([math.hypot(x.kx, x.ky) for x in c])
        om_true = np.array([x.omega for x in c])
        om_disp = np.sqrt(9.81 * k * np.tanh(k * DEPTH))
        print("dispersion parity max|domega|:", float(np.max(np.abs(om_true - om_disp))))
        model = cd.CondSetDenoiser().to(DEVICE)
        model.load_state_dict(torch.load(a.init_ckpt, map_location=DEVICE)); model.train()
        rng_np = np.random.default_rng(0)
        t0 = time.time()
        batch = make_scene_batch(32, pool, stats, gb, rng_np)
        print("scene shapes:", {k: tuple(v.shape) for k, v in batch.items() if torch.is_tensor(v)})
        # one training step timing
        opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
        for it in range(3):
            tb = time.time()
            batch = make_scene_batch(32, pool, stats, gb, rng_np)
            x0 = batch['x0']; nf = 16
            t = torch.cat([torch.randint(0, T, (16,), device=DEVICE),
                           torch.randint(0, T // 4, (16,), device=DEVICE)], 0)
            ab = abar[t][:, None, None]; eps = torch.randn_like(x0)
            xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
            epsp = model(xt, t, batch['Bmap']); anchor = F.mse_loss(epsp, eps)
            xf = xt[16:]; epf = epsp[16:]; abf = ab[16:]
            x0hat = ((xf - (1 - abf).sqrt() * epf) / abf.sqrt()).clamp(-1, 1)
            fb = {k: (v[16:] if torch.is_tensor(v) and v.shape[0] == 32 else v)
                  for k, v in batch.items()}
            fld = field_loss_from_x0hat(x0hat, fb, stats)
            loss = anchor + 1.0 * fld
            opt.zero_grad(); loss.backward(); opt.step()
            print(f"  smoke it{it}: anchor {anchor.item():.5f} field {fld.item():.5f} "
                  f"grad_ok={all(p.grad is not None for p in model.parameters())} "
                  f"({time.time()-tb:.2f}s)")
        print(f"smoke OK, ~{(time.time()-t0)/3:.2f}s/step est")
        sys.exit(0)

    cfg = dict(CONFIGS[a.config])
    cfg['init_ckpt'] = a.init_ckpt
    if a.steps is not None:
        cfg['steps'] = a.steps
    cfg['out'] = a.out or f'results/direction3/prior_net_e2eft_{a.config}.pt'
    assert not os.path.exists(cfg['out']), f"refuse to overwrite {cfg['out']}"
    print(f"=== e2e fine-tune config {a.config}: {cfg} ===", flush=True)
    curve = finetune(cfg, log=lambda m: print(m, flush=True))
    json.dump(dict(config=a.config, cfg={k: v for k, v in cfg.items()}, curve=curve),
              open(f'tmp/wp_e2e_2026-07-25/train_curve_{a.config}.json', 'w'), indent=2)
    print("done", flush=True)
