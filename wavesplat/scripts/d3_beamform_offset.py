"""A1 — beamform-net + offset-regression head (off-grid de-smear).

The DETECTION CNN (d3_beamform_net) localizes components as heatmap PEAKS on a NK×NK
k-grid, so recovered (kx,ky) are quantized to grid centers → "smearing" (뭉개짐). Here we
add a small REGRESSION head that, at each cell, predicts a continuous sub-grid offset
(Δkx,Δky) in cell units. Detected peak cell + its offset → OFF-GRID (kx,ky). Offset
regression relative to a detected anchor is well-posed (no mean-collapse). The offset is
trained in a 3×3 neighbourhood around each true component so a ±1-cell peak-localisation
error still reads a valid offset.

ISOLATION EXPERIMENT: eval reports GRID (offset off) vs REFINED (offset on) nRMSE on the
SAME peaks. If refined ≪ grid → smearing was a grid artifact (fixable). If refined ≈ grid
→ aperture/CRB-limited (information wall; net cannot help).

Run: python3 scripts/d3_beamform_offset.py [--steps 8000] [--eval_only ckpt.pt]
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesim.wave_generator import jonswap_components
from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.model.tnse import fit_amplitudes_phases, render_wave_field
from d6_integrated_eval import tag_to_file
from d15_sequential_fusion import gather_frames
from d23_main_results import K_TP, panel
from d40_learned_acquisition import pool_xy
from d3_beamform_net import beamform, target_heatmap, KLIM, NK, M_OBS, DEPTH
from nonlinear_forward import elevation

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OFF_BOUND = 0.5      # max |offset| in cell units — a true comp lies within ±0.5 cell of its
                     # nearest grid cell, so single-cell supervision keeps the offset field
                     # smooth (a 3×3 nbhd forces ~1.0 jumps between cells → conv net can't fit).
DK = 2 * KLIM / (NK - 1)


class DetectOffsetCNN(nn.Module):
    """Shared conv body → heatmap (sigmoid) + (offx, offy) regression, all NK×NK."""
    def __init__(self, c=48):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.GELU())
        self.body = nn.Sequential(blk(1, c), blk(c, c), blk(c, c), blk(c, c), blk(c, c))
        self.head = nn.Conv2d(c, 3, 1)

    def forward(self, B):
        h = self.head(self.body(B[:, None]))
        H = torch.sigmoid(h[:, 0])
        offx = OFF_BOUND * torch.tanh(h[:, 1])
        offy = OFF_BOUND * torch.tanh(h[:, 2])
        return H, offx, offy


def build_targets(comps, P, z):
    """comps: list of (nc,3)=(kx,ky,A) arrays; P (B,m,3); z (B,m) → Bmap,H,offx,offy,mask (torch).
    Shared by on-the-fly make_batch and the fixed-dataset path so targets are identical."""
    B = len(comps)
    Bmap = beamform(torch.tensor(P, device=DEVICE), torch.tensor(z, device=DEVICE))
    H = torch.stack([target_heatmap(torch.tensor(c[:, 0], device=DEVICE),
                                    torch.tensor(c[:, 1], device=DEVICE),
                                    torch.tensor(c[:, 2] ** 2, device=DEVICE)) for c in comps])
    offx_t = torch.zeros(B, NK, NK, device=DEVICE); offy_t = torch.zeros(B, NK, NK, device=DEVICE)
    mask = torch.zeros(B, NK, NK, device=DEVICE)
    for b, c in enumerate(comps):
        ii = (c[:, 0] + KLIM) / (2 * KLIM) * (NK - 1)   # fractional col (kx)
        jj = (c[:, 1] + KLIM) / (2 * KLIM) * (NK - 1)   # fractional row (ky)
        for f_i, f_j in zip(ii, jj):
            # out-of-grid guard: skip |k|>KLIM (unrepresentable; clipped offset would blow up loss)
            if not (-0.5 <= f_i <= NK - 0.5 and -0.5 <= f_j <= NK - 0.5):
                continue
            ci = min(max(int(round(f_i)), 0), NK - 1)   # nearest cell → offset ∈ [-0.5,0.5]
            cj = min(max(int(round(f_j)), 0), NK - 1)
            offx_t[b, cj, ci] = float(f_i - ci); offy_t[b, cj, ci] = float(f_j - cj)
            mask[b, cj, ci] = 1.0
    return Bmap, H, offx_t, offy_t, mask


def make_batch(Bsz, pool, hos_frac=0.0):
    """On-the-fly synthetic JONSWAP obs → targets. hos_frac: fraction using 2nd-order Stokes."""
    P = np.zeros((Bsz, M_OBS, 3), np.float32); Z = np.zeros((Bsz, M_OBS), np.float32)
    comps = []
    for b in range(Bsz):
        Hs = np.random.uniform(0.5, 3.5); Tp = np.random.uniform(5.0, 12.0)
        nc = int(np.random.choice([5, 15, 30]))
        c = jonswap_components(Hs=Hs, Tp=Tp, n_components=nc, depth=DEPTH, seed=int(np.random.randint(1e9)))
        A = np.array([x.A for x in c]); om = np.array([x.omega for x in c])
        kx = np.array([x.kx for x in c]); ky = np.array([x.ky for x in c]); ph = np.array([x.phi for x in c])
        Tobs = K_TP * Tp
        idx = np.random.choice(len(pool), M_OBS, replace=len(pool) < M_OBS)
        ox, oy = pool[idx, 0], pool[idx, 1]; ot = np.random.uniform(0, Tobs, M_OBS)
        nl = np.random.rand() < hos_frac
        Z[b] = elevation(kx, ky, A, om, ph, ox, oy, ot, DEPTH, nonlinear=nl) \
            + np.random.normal(0, 0.02, M_OBS)
        P[b] = np.column_stack([ox, oy, ot])
        comps.append(np.stack([kx, ky, A], 1).astype(np.float32))
    return build_targets(comps, P, Z)


def train(steps, Bsz=32, lr=3e-4, hos_frac=0.0, dataset=None, linear_only=False, log=print):
    pool = pool_xy().astype(np.float32)
    sampler = None
    if dataset:
        from fixed_dataset import FixedSampler
        sampler = FixedSampler(dataset, aug_noise=0.015, sub_obs=320,
                               nl_filter=(0 if linear_only else None))
        log(f"  training from fixed dataset {dataset} (N={sampler.N}, linear_only={linear_only})")
    model = DetectOffsetCNN().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t0 = time.time()
    for s in range(steps):
        if sampler is not None:
            comps, P, z = sampler.batch(Bsz)
            Bmap, H, oxt, oyt, m = build_targets(comps, P, z)
        else:
            Bmap, H, oxt, oyt, m = make_batch(Bsz, pool, hos_frac=hos_frac)
        Hp, oxp, oyp = model(Bmap)
        loss_h = (((Hp - H) ** 2) * (1 + 20 * H)).mean()
        denom = m.sum() + 1e-6
        loss_off = ((m * ((oxp - oxt) ** 2 + (oyp - oyt) ** 2)).sum()) / denom
        loss = loss_h + 10.0 * loss_off
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if s % 200 == 0 or s == steps - 1:
            with torch.no_grad():
                # RMS offset error at true cells, in rad/m
                off_rmse = torch.sqrt((m * ((oxp - oxt) ** 2 + (oyp - oyt) ** 2)).sum() / denom) * DK
            log(f"  step {s:5d}  loss {loss.item():.5f}  h {loss_h.item():.5f}  "
                f"off_rmse {off_rmse.item():.4f} rad/m  ({time.time()-t0:.0f}s)")
    return model


def extract(H, offx, offy, thresh=0.15, refine=True):
    """local maxima → (kx,ky). refine=True adds the predicted sub-grid offset."""
    from scipy.ndimage import maximum_filter
    Hn = H.cpu().numpy(); ox = offx.cpu().numpy(); oy = offy.cpu().numpy()
    mx = maximum_filter(Hn, size=5)
    pk = np.argwhere((Hn == mx) & (Hn > thresh * Hn.max()))
    if len(pk) == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    rows, cols = pk[:, 0], pk[:, 1]
    ci = cols.astype(np.float32); cj = rows.astype(np.float32)
    if refine:
        ci = ci + ox[rows, cols]; cj = cj + oy[rows, cols]
    kx = -KLIM + ci / (NK - 1) * 2 * KLIM
    ky = -KLIM + cj / (NK - 1) * 2 * KLIM
    return kx.astype(np.float32), ky.astype(np.float32)


@torch.no_grad()
def evaluate(model, episodes, log=print):
    model.eval(); out = {}
    for tag in episodes:
        L = EpisodeDataLoader('data/' + tag_to_file(tag)); depth = L.depth; Hs = L.Hs_true
        gx = L.gt_grid_x.numpy(); gy = L.gt_grid_y.numpy(); nb = int(round(K_TP * L.Tp_true / 0.1))
        i0 = nb + 3; i1 = len(L) - 2; sel = np.linspace(i0, i1, 3).astype(int)
        nr_grid, nr_ref = [], []
        for i_star in sel:
            fstar = L[int(i_star)]; P, z = gather_frames(L, int(i_star), nb)
            if len(z) > M_OBS:
                idx = np.random.default_rng(0).choice(len(z), M_OBS, replace=False); P, z = P[idx], z[idx]
            t0 = P[:, 2].min(); Ps = P.copy(); Ps[:, 2] -= t0
            Bmap = beamform(torch.tensor(Ps[None], device=DEVICE, dtype=torch.float32),
                            torch.tensor(z[None], device=DEVICE, dtype=torch.float32))
            Hp, oxp, oyp = model(Bmap)
            obs = np.column_stack([Ps[:, 0], Ps[:, 1], Ps[:, 2], z]).astype(np.float32)
            for refine, bucket in ((False, nr_grid), (True, nr_ref)):
                kx, ky = extract(Hp[0], oxp[0], oyp[0], refine=refine)
                if len(kx) == 0:
                    bucket.append(1.0); continue
                A, phi = fit_amplitudes_phases(kx, ky, obs, depth, lambda_reg=0.5)
                fld = render_wave_field(kx, ky, A, phi, depth, gx, gy, float(fstar.t) - t0)
                bucket.append(panel(fld, fstar.gt_field.numpy(), Hs)['nrmse'])
        g, r = float(np.mean(nr_grid)), float(np.mean(nr_ref))
        out[tag] = dict(grid=g, refined=r, delta=g - r)
        log(f"  {tag:12s} grid {g:.3f}  refined {r:.3f}  Δ {g-r:+.3f}")
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=8000)
    ap.add_argument('--ckpt', default='results/direction3/beamform_offset.pt')
    ap.add_argument('--eval_only', default=None)
    ap.add_argument('--hos_frac', type=float, default=0.0)
    ap.add_argument('--dataset', default=None)
    ap.add_argument('--linear_only', action='store_true')
    a = ap.parse_args()
    eps = ['BF3_5c_s0', 'BF4_5c_s0', 'BF4_15c_s0', 'BF4_30c_s0', 'BF5_15c_s0', 'BF5_30c_s1']
    if a.eval_only:
        model = DetectOffsetCNN().to(DEVICE); model.load_state_dict(torch.load(a.eval_only, map_location=DEVICE))
    else:
        print(f"CONFIG OFF_BOUND={OFF_BOUND} DK={DK:.5f} weight=10.0 hos_frac={a.hos_frac} "
              f"dataset={a.dataset}", flush=True)
        print(f"training {a.steps} steps on {DEVICE} (beamform detect+offset head)...", flush=True)
        model = train(a.steps, hos_frac=a.hos_frac, dataset=a.dataset,
                      linear_only=a.linear_only, log=lambda m: print(m, flush=True))
        torch.save(model.state_dict(), a.ckpt); print(f"saved {a.ckpt}", flush=True)
    print("\n=== A1 isolation: GRID (offset off) vs REFINED (offset on) ===", flush=True)
    res = evaluate(model, eps, log=lambda m: print(m, flush=True))
    mg = float(np.mean([v['grid'] for v in res.values()]))
    mr = float(np.mean([v['refined'] for v in res.values()]))
    print(f"\nMEAN  grid {mg:.3f}  refined {mr:.3f}  Δ {mg-mr:+.3f}", flush=True)
    print("verdict:", "grid-artifact (refine helps → fixable)" if mg - mr > 0.005
          else "aperture-limited (refine ~ no help → information wall)", flush=True)
    json.dump(dict(per_ep=res, mean_grid=mg, mean_refined=mr),
              open('results/direction3/beamform_offset_eval.json', 'w'), indent=2)
