#!/usr/bin/env python3
"""
pino_stage4b_mf_t5.py

WavePINO Stage 4b: Multi-frame T=5, gap=10 (window=4.0s).

Stage 4 (T=3, gap=5, window=1.0s) gave 0.2201 vs PINO v1 0.2220.
This run tests a longer temporal window (4.0s) where waves move 16-56m,
providing much stronger phase velocity information for component disambiguation.

Usage:
    cd wavesplat && python3 -u scripts/pino_stage4b_mf_t5.py
"""

import os, sys, time, json, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'wavesplat'))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.model.wave_pino import FNOBlock, spectral_physics_loss
from wavesplat.evaluation.metrics import nrmse

# ---------------------------------------------------------------------------
# Stage 4b hyperparameters — only T and gap differ from Stage 4
# ---------------------------------------------------------------------------
N_FRAMES    = 5
FRAME_GAP   = 10   # Δt=1.0s/step, total window = (5-1)*10*0.1 = 4.0s

GRID_NX, GRID_NY = 64, 48
FNO_WIDTH, FNO_LAYERS = 16, 4
FNO_MODES_H, FNO_MODES_W = 16, 12

EPOCHS, STEPS_PER_EPOCH = 100, 100
BATCH_SIZE = 8
LR         = 1e-3
WARMUP_EPOCHS = 5
LAMBDA_CORR, LAMBDA_SPEC = 1.0, 0.5
DEPTH, T_MIN, T_MAX = 50.0, 4.0, 25.0

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRAIN_EPISODES = [
    'episode_BF3_1comp_seed0_uniform.h5', 'episode_BF3_1comp_seed1_uniform.h5',
    'episode_BF3_2comp_seed0_uniform.h5', 'episode_BF3_2comp_seed1_uniform.h5',
    'episode_BF3_5comp_seed0_uniform.h5', 'episode_BF3_5comp_seed1_uniform.h5',
    'episode_BF3_1comp_seed0_slope.h5',  'episode_BF3_1comp_seed1_slope.h5',
    'episode_BF3_2comp_seed0_slope.h5',  'episode_BF3_2comp_seed1_slope.h5',
    'episode_BF3_5comp_seed0_slope.h5',  'episode_BF3_5comp_seed1_slope.h5',
    'episode_BF4_1comp_seed0_uniform.h5','episode_BF4_1comp_seed1_uniform.h5',
    'episode_BF4_5comp_seed0_uniform.h5','episode_BF4_5comp_seed1_uniform.h5',
    'episode_BF4_15comp_seed0_uniform.h5','episode_BF4_15comp_seed1_uniform.h5',
    'episode_BF4_30comp_seed0_uniform.h5',
    'episode_BF4_1comp_seed0_slope.h5',  'episode_BF4_1comp_seed1_slope.h5',
    'episode_BF4_5comp_seed0_slope.h5',  'episode_BF4_5comp_seed1_slope.h5',
    'episode_BF4_15comp_seed0_slope.h5', 'episode_BF4_15comp_seed1_slope.h5',
    'episode_BF5_15comp_seed0_uniform.h5','episode_BF5_15comp_seed1_uniform.h5',
    'episode_BF5_30comp_seed0_uniform.h5','episode_BF5_30comp_seed1_uniform.h5',
    'episode_BF5_15comp_seed0_slope.h5', 'episode_BF5_15comp_seed1_slope.h5',
    'episode_BF5_30comp_seed0_slope.h5', 'episode_BF5_30comp_seed1_slope.h5',
]
VAL_EPISODES = [
    'episode_BF3_5comp_seed2_uniform.h5', 'episode_BF4_5comp_seed2_uniform.h5',
    'episode_BF4_15comp_seed2_uniform.h5', 'episode_BF5_15comp_seed2_uniform.h5',
]
TEST_EPISODES = [
    'episode_BF3_5comp_seed0_slope.h5', 'episode_BF4_5comp_seed0_uniform.h5',
    'episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_uniform.h5',
    'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5',
]


# ---------------------------------------------------------------------------
# Model (identical to Stage 4 WavePINOMultiFrame)
# ---------------------------------------------------------------------------

class WavePINOMultiFrame(nn.Module):
    def __init__(self, n_frames=5, grid_nx=64, grid_ny=48,
                 x_range=(10.,250.), y_range=(-100.,100.),
                 fno_width=16, fno_layers=4, fno_modes_h=16, fno_modes_w=12,
                 init_lengthscale=10.):
        super().__init__()
        self.n_frames = n_frames
        self.grid_nx, self.grid_ny = grid_nx, grid_ny
        self.x_range, self.y_range = x_range, y_range
        self.fno_width = fno_width
        gx = torch.linspace(x_range[0], x_range[1], grid_nx)
        gy = torch.linspace(y_range[0], y_range[1], grid_ny)
        xx, yy = torch.meshgrid(gx, gy, indexing='ij')
        self.register_buffer('grid_xy',
                             torch.stack([xx.flatten(), yy.flatten()], dim=-1))
        self.log_lengthscale = nn.Parameter(torch.tensor(math.log(init_lengthscale)))
        self.lifting = nn.Sequential(nn.Conv2d(2*n_frames, fno_width, 1), nn.GELU())
        self.fno_blocks = nn.ModuleList([
            FNOBlock(fno_width, fno_modes_h, fno_modes_w) for _ in range(fno_layers)])
        self.output_head = nn.Sequential(
            nn.Conv2d(fno_width, fno_width*2, 3, padding=1), nn.GELU(),
            nn.Conv2d(fno_width*2, 1, 1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None: nn.init.zeros_(m.bias)

    def _setconv(self, obs_xy, obs_z):
        ls = self.log_lengthscale.exp()
        diff = self.grid_xy.unsqueeze(0).unsqueeze(2) - obs_xy.unsqueeze(1)
        w = torch.exp(-0.5 * (diff**2).sum(-1) / ls**2)
        density = w.sum(-1, keepdim=True)
        signal  = (w.unsqueeze(-1) * obs_z.unsqueeze(1)).sum(2) / (density + 1e-8)
        return torch.cat([signal, density], dim=-1).permute(0,2,1).reshape(
            obs_xy.shape[0], 2, self.grid_nx, self.grid_ny)

    def _grid_predict(self, xc_list, yc_list):
        z = torch.cat([self._setconv(xc, yc) for xc, yc in zip(xc_list, yc_list)], 1)
        z = self.lifting(z)
        for b in self.fno_blocks: z = b(z)
        return self.output_head(z)

    def forward(self, xc_list, yc_list, xt):
        g = self._grid_predict(xc_list, yc_list)
        xn = 2.*(xt[...,0]-self.x_range[0])/(self.x_range[1]-self.x_range[0])-1.
        yn = 2.*(xt[...,1]-self.y_range[0])/(self.y_range[1]-self.y_range[0])-1.
        sg = torch.stack([yn,xn],-1).unsqueeze(2)
        s = F.grid_sample(g, sg, mode='bilinear', align_corners=True, padding_mode='border')
        return s.squeeze(1).squeeze(-1).unsqueeze(-1)

    def predict_grid(self, xc_list, yc_list):
        return self._grid_predict(xc_list, yc_list)

    @property
    def n_params(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_episodes(names, data_dir):
    return [EpisodeDataLoader(os.path.join(data_dir, n))
            for n in names if os.path.exists(os.path.join(data_dir, n))]


def correlation_loss(pred, target):
    pc = pred - pred.mean(1, keepdim=True)
    tc = target - target.mean(1, keepdim=True)
    corr = (pc*tc).sum(1) / (pc.norm(1).clamp(1e-8)*tc.norm(1).clamp(1e-8))
    return 1. - corr.mean()


def sample_batch(episodes, batch_size, n_frames, frame_gap,
                 mgx, mgy, ggx, ggy, device):
    min_fi = (n_frames-1)*frame_gap
    xcl = [[] for _ in range(n_frames)]
    ycl = [[] for _ in range(n_frames)]
    gtl = []
    for _ in range(batch_size):
        ep = episodes[np.random.randint(len(episodes))]
        fi = np.random.randint(min_fi, len(ep))
        Hs = max(ep.Hs_true, .1)
        flip = np.random.random() < .5
        for t in range(n_frames):
            fr = ep[fi-(n_frames-1-t)*frame_gap]
            xc = torch.stack([fr.x_obs, fr.y_obs], -1)
            yc = fr.z_obs.unsqueeze(-1) / Hs
            if flip: xc = xc.clone(); xc[:,1] = -xc[:,1]
            xcl[t].append(xc); ycl[t].append(yc)
        fr_cur = ep[fi]
        gt = fr_cur.gt_field / Hs
        dx0,dx1 = float(ggx[0]),float(ggx[-1])
        dy0,dy1 = float(ggy[0]),float(ggy[-1])
        xn = 2.*(mgx-dx0)/(dx1-dx0)-1.; yn = 2.*(mgy-dy0)/(dy1-dy0)-1.
        yy,xx = torch.meshgrid(yn,xn,indexing='ij')
        sg = torch.stack([xx,yy],-1).unsqueeze(0)
        gtr = F.grid_sample(gt.unsqueeze(0).unsqueeze(0).float(), sg,
                            mode='bilinear', align_corners=True,
                            padding_mode='border')[0,0].T
        if flip: gtr = gtr.flip(1)
        gtl.append(gtr)
    xco, yco = [], []
    for t in range(n_frames):
        mn = max(x.shape[0] for x in xcl[t])
        xb = torch.zeros(batch_size, mn, 2)
        yb = torch.zeros(batch_size, mn, 1)
        for i,(xc,yc) in enumerate(zip(xcl[t],ycl[t])):
            xb[i,:xc.shape[0]] = xc; yb[i,:yc.shape[0]] = yc
        xco.append(xb.to(device)); yco.append(yb.to(device))
    return xco, yco, torch.stack(gtl).to(device)


def evaluate(model, episodes, ggx, ggy, device, n_frames, frame_gap, n_eval=10):
    model.eval()
    xx, yy = torch.meshgrid(ggx, ggy, indexing='ij')
    xt = torch.stack([xx.flatten(),yy.flatten()],-1).unsqueeze(0).to(device)
    min_fi = (n_frames-1)*frame_gap
    vals = []
    for ep in episodes:
        Hs = max(ep.Hs_true,.1)
        for fi in np.linspace(min_fi, len(ep)-1, n_eval, dtype=int):
            xc_list, yc_list = [], []
            for t in range(n_frames):
                fr = ep[fi-(n_frames-1-t)*frame_gap]
                xc_list.append(torch.stack([fr.x_obs,fr.y_obs],-1).unsqueeze(0).to(device))
                yc_list.append(fr.z_obs.unsqueeze(-1).unsqueeze(0).to(device)/Hs)
            with torch.no_grad():
                pred = model(xc_list, yc_list, xt)
            pf = (pred[0,:,0]*Hs).cpu().numpy().reshape(len(ggx),len(ggy)).T
            v = nrmse(pf, ep[fi].gt_field.numpy(), ep.Hs_true)
            if not np.isnan(v): vals.append(v)
    model.train()
    return float(np.mean(vals)) if vals else 1.


def test_eval(model, test_names, test_eps, ggx, ggy, device, n_frames, frame_gap):
    model.eval()
    xx, yy = torch.meshgrid(ggx, ggy, indexing='ij')
    xt = torch.stack([xx.flatten(),yy.flatten()],-1).unsqueeze(0).to(device)
    min_fi = (n_frames-1)*frame_gap
    results = {}
    for name, ep in zip(test_names, test_eps):
        Hs = max(ep.Hs_true,.1)
        ep_v = []
        for fi in np.linspace(min_fi, len(ep)-1, 20, dtype=int):
            xc_list, yc_list = [], []
            for t in range(n_frames):
                fr = ep[fi-(n_frames-1-t)*frame_gap]
                xc_list.append(torch.stack([fr.x_obs,fr.y_obs],-1).unsqueeze(0).to(device))
                yc_list.append(fr.z_obs.unsqueeze(-1).unsqueeze(0).to(device)/Hs)
            with torch.no_grad():
                pred = model(xc_list, yc_list, xt)
            pf = (pred[0,:,0]*Hs).cpu().numpy().reshape(len(ggx),len(ggy)).T
            v = nrmse(pf, ep[fi].gt_field.numpy(), ep.Hs_true)
            if not np.isnan(v): ep_v.append(v)
        key = name.replace('episode_','').replace('.h5','')
        avg = float(np.mean(ep_v)) if ep_v else 1.
        results[key] = avg
        print(f"  {key}: {avg:.4f}")
    return results


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model, train_eps, val_eps, mgx, mgy, ggx, ggy, x_range, y_range, ckpt):
    xe, ye = x_range[1]-x_range[0], y_range[1]-y_range[0]
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    warm = optim.lr_scheduler.LinearLR(opt, .1, 1., WARMUP_EPOCHS)
    cos  = optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS-WARMUP_EPOCHS, LR*.01)
    sched = optim.lr_scheduler.SequentialLR(opt, [warm,cos], [WARMUP_EPOCHS])
    best_val, best_state, t0 = 1., None, time.time()
    for ep in range(1, EPOCHS+1):
        model.train(); eloss = 0.
        for _ in range(STEPS_PER_EPOCH):
            xco,yco,gt_b = sample_batch(train_eps,BATCH_SIZE,N_FRAMES,FRAME_GAP,
                                        mgx,mgy,ggx,ggy,DEVICE)
            gp = model.predict_grid(xco, yco)
            gg = gt_b.unsqueeze(1)
            l = (F.mse_loss(gp,gg)
                 + LAMBDA_CORR * correlation_loss(gp.squeeze(1).flatten(1), gg.squeeze(1).flatten(1))
                 + LAMBDA_SPEC * spectral_physics_loss(gp, xe, ye, DEPTH, T_MIN, T_MAX))
            opt.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step(); eloss += l.item()
        sched.step()
        if ep % 10 == 0:
            vn = evaluate(model, val_eps, ggx, ggy, DEVICE, N_FRAMES, FRAME_GAP)
            if vn < best_val:
                best_val = vn
                best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
            print(f"  Epoch {ep:3d}/{EPOCHS}  loss={eloss/STEPS_PER_EPOCH:.4f}"
                  f"  val={vn:.4f}  best={best_val:.4f}  [{time.time()-t0:.0f}s]")
    if best_state:
        model.load_state_dict({k:v.to(DEVICE) for k,v in best_state.items()})
        torch.save(best_state, ckpt)
    return best_val


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"WavePINO Stage 4b — T={N_FRAMES} gap={FRAME_GAP} "
          f"window={(N_FRAMES-1)*FRAME_GAP*0.1:.1f}s")
    print(f"Device: {DEVICE}")
    print(f"Wave shift: {(N_FRAMES-1)*FRAME_GAP*0.1*4:.0f}–"
          f"{(N_FRAMES-1)*FRAME_GAP*0.1*14:.0f}m  (domain=240m)")

    data_dir = os.path.join(REPO_ROOT, 'wavesplat', 'data')
    ckpt_dir = os.path.join(REPO_ROOT, 'wavesplat', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    print("\nLoading episodes ...")
    train_eps = load_episodes(TRAIN_EPISODES, data_dir)
    val_eps   = load_episodes(VAL_EPISODES, data_dir)
    test_eps  = load_episodes(TEST_EPISODES, data_dir)
    print(f"  Train:{len(train_eps)}  Val:{len(val_eps)}  Test:{len(test_eps)}")

    ep0 = train_eps[0]
    ggx = ep0.gt_grid_x.cpu(); ggy = ep0.gt_grid_y.cpu()
    x_range = (float(ggx[0]), float(ggx[-1]))
    y_range = (float(ggy[0]), float(ggy[-1]))
    mgx = torch.linspace(x_range[0], x_range[1], GRID_NX)
    mgy = torch.linspace(y_range[0], y_range[1], GRID_NY)

    model = WavePINOMultiFrame(
        n_frames=N_FRAMES, grid_nx=GRID_NX, grid_ny=GRID_NY,
        x_range=x_range, y_range=y_range,
        fno_width=FNO_WIDTH, fno_layers=FNO_LAYERS,
        fno_modes_h=FNO_MODES_H, fno_modes_w=FNO_MODES_W,
    ).to(DEVICE)
    print(f"\n  Model: {model.n_params:,} params  (2×{N_FRAMES}={2*N_FRAMES} ch input)")

    ckpt = os.path.join(ckpt_dir, f'wavepino_mf_T{N_FRAMES}_g{FRAME_GAP}.pt')
    t_start = time.time()

    print(f"\n{'='*60}")
    print(f"  Training  T={N_FRAMES}  gap={FRAME_GAP}  λ_spec={LAMBDA_SPEC}")
    print(f"{'='*60}")
    best_val = train(model, train_eps, val_eps, mgx, mgy, ggx, ggy,
                     x_range, y_range, ckpt)

    print("\nTest evaluation ...")
    test_res = test_eval(model, TEST_EPISODES, test_eps, ggx, ggy,
                         DEVICE, N_FRAMES, FRAME_GAP)

    test_avg = float(np.mean(list(test_res.values())))
    total_s  = time.time() - t_start

    result = {
        "method": f"WavePINO-MF T={N_FRAMES} gap={FRAME_GAP} window=4.0s",
        "n_frames": N_FRAMES, "frame_gap": FRAME_GAP,
        "temporal_window_s": (N_FRAMES-1)*FRAME_GAP*0.1,
        "best_val_nrmse": best_val,
        "test_avg_nrmse": test_avg,
        "test_per_episode": test_res,
        "references": {"WaveSplat":0.223,"PINO_v1_T1":0.2220,"PINO_MF_T3_g5":0.2201},
        "delta_vs_pino1": test_avg-0.2220,
        "delta_vs_mf_t3": test_avg-0.2201,
        "total_time_s": total_s,
    }
    out = os.path.join(REPO_ROOT, 'wavesplat', 'results', 'pino_stage4b_mf_t5.json')
    with open(out, 'w') as f: json.dump(result, f, indent=2)

    print(f"\n{'='*60}")
    d1 = test_avg-0.2220; d2 = test_avg-0.2201
    print(f"  WavePINO-MF T=5 g10: {test_avg:.4f}")
    print(f"  WavePINO-MF T=3 g5:  0.2201  (delta={d2*100:+.1f}%)")
    print(f"  PINO v1 T=1:         0.2220  (delta={d1*100:+.1f}%)")
    print(f"  WaveSplat:           0.2230")
    print(f"  Time: {total_s:.0f}s  |  {out}")


if __name__ == '__main__':
    main()
