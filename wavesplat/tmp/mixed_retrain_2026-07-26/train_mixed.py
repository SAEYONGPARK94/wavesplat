"""MIXED-CURRICULUM retrain of the beamform-conditioned set-diffusion proposal net.

ADDITIVE.  Imports `scripts/d3_beamform_cond_diffusion` and never modifies it.  The MODEL, the
DIFFUSION SCHEDULE, the LOSS and the OPTIMISER are byte-for-byte the shipped recipe
(`cd.CondSetDenoiser`, `cd.cosine_betas(T=500)`, eps-MSE, AdamW lr 2e-4 / wd 1e-4, cosine
anneal, batch 32).  The ONLY change is where the training batch comes from:
`cd.make_batch` -> `gen_mixed.sample_scene_ok` + `gen_mixed.batch_from_scenes`.

Writes results/direction3/beamform_cond_diffusion_mixed_v1.pt.  Refuses to touch the shipped
checkpoint results/direction3/beamform_cond_diffusion.pt.

  python3 tmp/mixed_retrain_2026-07-26/train_mixed.py --steps 30000
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, 'scripts')
sys.path.insert(0, 'tmp/mixed_retrain_2026-07-26')
import d3_beamform_cond_diffusion as cd          # os.chdir()s to the repo root on import
import gen_mixed as gm

DEVICE = cd.DEVICE
SHIPPED = 'results/direction3/beamform_cond_diffusion.pt'
OUT = 'results/direction3/beamform_cond_diffusion_mixed_v1.pt'
CURVE = 'tmp/mixed_retrain_2026-07-26/train_curve.json'   # overridden per round
VAL_SEED = 987654321          # held-out validation scenes, disjoint from the training stream


def make_val(bank, n=256, seed=VAL_SEED):
    rng = np.random.default_rng(seed)
    return [gm.sample_scene_ok(rng, bank) for _ in range(n)]


@torch.no_grad()
def val_loss(model, val_scenes, stats, abar, bsz=32, seed=0):
    """Frozen-noise eps-MSE on held-out scenes (reporting only -- never selects a checkpoint)."""
    model.eval()
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    tot, nb = 0.0, 0
    for i in range(0, len(val_scenes), bsz):
        chunk = val_scenes[i:i + bsz]
        x0, Bmap = gm.batch_from_scenes(chunk, stats, DEVICE)
        b = x0.shape[0]
        t = torch.randint(0, cd.T, (b,), device=DEVICE, generator=g)
        ab = abar[t][:, None, None]
        eps = torch.randn(x0.shape, device=DEVICE, generator=g)
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        tot += float(F.mse_loss(model(xt, t, Bmap), eps)); nb += 1
    model.train()
    return tot / max(nb, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--bsz', type=int, default=32)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--seed', type=int, default=20260726)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--val_every', type=int, default=1000)
    ap.add_argument('--curve', default=CURVE)
    ap.add_argument('--smoke', type=int, default=0)
    ap.add_argument('--target_mode', default='replicate', choices=['replicate', 'graded', 'graded2'])
    a = ap.parse_args()

    assert os.path.abspath(a.out) != os.path.abspath(SHIPPED), 'refuse to overwrite the shipped ckpt'
    assert not os.path.exists(a.out), f'refuse to overwrite an existing checkpoint: {a.out}'
    gm.TARGET_MODE = a.target_mode
    print(f'  TARGET_MODE = {gm.TARGET_MODE}', flush=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    stats = torch.load(cd.PRIOR_CKPT, map_location=DEVICE)['stats']
    betas = cd.cosine_betas(cd.T).to(DEVICE)
    abar = torch.cumprod(1 - betas, 0)

    print(f"=== mixed retrain: {a.steps} steps, bsz {a.bsz}, lr {a.lr}, device {DEVICE} ===",
          flush=True)
    bank = gm.GeometryBank(verbose=False)
    print(f"  geometry bank: {len(bank.donors)} donor episodes", flush=True)
    print(f"  mixture: {dict(zip(gm.STRATA, gm.WEIGHTS))}", flush=True)
    val_scenes = make_val(bank)
    print(f"  held-out val: {len(val_scenes)} scenes (seed {VAL_SEED})", flush=True)

    model = cd.CondSetDenoiser().to(DEVICE)          # from scratch: identical architecture
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)

    if a.smoke:
        t0 = time.time()
        for s in range(a.smoke):
            sc = [gm.sample_scene_ok(rng, bank) for _ in range(a.bsz)]
            x0, Bmap = gm.batch_from_scenes(sc, stats, DEVICE)
            t = torch.randint(0, cd.T, (a.bsz,), device=DEVICE)
            ab = abar[t][:, None, None]; eps = torch.randn_like(x0)
            xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
            loss = F.mse_loss(model(xt, t, Bmap), eps)
            opt.zero_grad(); loss.backward(); opt.step()
        dt = (time.time() - t0) / a.smoke
        print(f"smoke: {dt*1000:.0f} ms/step -> {a.steps} steps = {dt*a.steps/3600:.2f} h",
              flush=True)
        v = val_loss(model, val_scenes, stats, abar)
        print(f"smoke val loss {v:.5f}", flush=True)
        return

    curve = []
    strata_seen = {k: 0 for k in gm.STRATA}
    t0 = time.time()
    for s in range(a.steps):
        sc = [gm.sample_scene_ok(rng, bank) for _ in range(a.bsz)]
        for x in sc:
            strata_seen[x['stratum']] += 1
        x0, Bmap = gm.batch_from_scenes(sc, stats, DEVICE)
        t = torch.randint(0, cd.T, (a.bsz,), device=DEVICE)
        ab = abar[t][:, None, None]
        eps = torch.randn_like(x0)
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        loss = F.mse_loss(model(xt, t, Bmap), eps)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if s % a.val_every == 0 or s == a.steps - 1:
            v = val_loss(model, val_scenes, stats, abar)
            curve.append(dict(step=s, train=float(loss), val=float(v),
                              sec=time.time() - t0))
            print(f"  step {s:6d}  train {loss.item():.5f}  val {v:.5f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
            json.dump(dict(curve=curve, strata_seen=strata_seen, args=vars(a)),
                      open(a.curve, 'w'), indent=1)
        elif s % 250 == 0:
            print(f"  step {s:6d}  train {loss.item():.5f}  ({time.time()-t0:.0f}s)", flush=True)

    torch.save(model.state_dict(), a.out)
    print(f"saved {a.out}  ({time.time()-t0:.0f}s total)", flush=True)
    json.dump(dict(curve=curve, strata_seen=strata_seen, args=vars(a),
                   wall_sec=time.time() - t0), open(a.curve, 'w'), indent=1)
    print("strata seen:", {k: round(v / (a.steps * a.bsz), 4)
                           for k, v in strata_seen.items()}, flush=True)


if __name__ == '__main__':
    main()
