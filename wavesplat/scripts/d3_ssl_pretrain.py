"""d3_ssl_pretrain.py — W7: self-supervised pretraining on the 42 GB of real
stereo surfaces, which this project has barely touched.

Every learned prior we have (prior_net_v3, the field nets, wiener2) saw only
synthetic seas: linear superposition JONSWAP, Gaussian by the central limit
theorem. Yesterday's realisation was that a Gaussian sea makes the physics
estimator optimal BY THEOREM, so those benchmarks could not have been won.
The one place learning can win is non-Gaussian structure — and the only
non-Gaussian data we own is the 13 real records (212,991 frames).

So: mask 75% of a 64x64 patch of measured sea surface and reconstruct it.
No labels. The encoder has to learn what real seas look like.

Tonight's checkable deliverable is NOT the reconstruction score — it is a
LINEAR PROBE on the frozen features for the patch's excess kurtosis. If a
linear read-out of the embedding predicts non-Gaussianity, then

  - CL1 (learning the non-Gaussian posterior correction) has a conditioning
    signal, and
  - CL7 (telling wave from stereo-reconstruction error) has a starting point,

because both need a representation that knows how far from Gaussian the local
sea is. If the probe fails, the embedding is a texture detector and W7 is dead.

One record (YS_2017) is held out entirely.

Usage: python3 scripts/d3_ssl_pretrain.py --steps 40000
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from d3_sa_multirecord import MRecord, _unpack

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT = 'results/direction3/ssl_mae.pt'
OUT = 'results/direction3/ssl_pretrain.json'
PATCH = 64
TOK = 8                      # 8x8 token grid of 8x8 pixels
HELD_OUT = 'YS_2017'         # never trained on
MIN_VALID = 0.98


class Patches:
    """Stream 64x64 patches of measured surface. Reads are 1-3 ms per frame,
    so no cache is needed — the disk keeps up with the GPU.

    `MRecord.window` subtracts the per-pixel temporal mean over the window it
    returns — a sensor calibration that removes the smooth stereo
    reconstruction bias (LJ Grid2 carries one with spatial std ~4 m). Asking
    it for a single frame therefore returns exactly zero, and the first
    version of this loader spun forever rejecting blank patches.

    Here the same calibration is applied as a STATIC per-pixel bias, estimated
    once per record from 128 spread-out frames. Subtracting a fixed field
    keeps the sea's temporal statistics — and this pretraining exists to learn
    non-Gaussian structure, which a temporal high-pass would distort.
    """

    def __init__(self, held_out=None, exclude=True):
        self.recs, self.names = [], []
        for f in sorted(glob.glob('data/real_bergamasco/*/*.nc')):
            site = os.path.basename(os.path.dirname(f))
            if exclude and held_out and site == held_out:
                continue
            if not exclude and site != held_out:
                continue
            try:
                # robust=True probes 160 frames per record and takes minutes
                # on the 10.8 GB 601x501 grid; the smoke test timed out there.
                # Patch-level filters (validity >= 0.98, std > 0) do the same
                # job for self-supervision.
                rec = MRecord(f, stride=1, robust=False)
                idx = np.linspace(0, rec.T_all - 1, 128).astype(int)
                fr = np.stack([_unpack(rec.zd, np.s_[int(i)]) for i in idx])
                fr[:, ~rec.valid] = np.nan
                rec.bias = np.nanmean(fr, axis=0)
                self.recs.append(rec)
                self.names.append(os.path.basename(f))
            except Exception as e:
                print(f"  skip {os.path.basename(f)}: {e}", flush=True)
        self.w = np.array([r.T_all for r in self.recs], float)
        self.w /= self.w.sum()
        print(f"  {len(self.recs)} records, "
              f"{sum(r.T_all for r in self.recs)} frames", flush=True)

    def batch(self, rng, bs, want_rec=False):
        out, src = [], []
        while len(out) < bs:
            i = rng.choice(len(self.recs), p=self.w)
            rec = self.recs[i]
            t = int(rng.integers(0, rec.T_all))
            Z = _unpack(rec.zd, np.s_[t]).astype(np.float64)
            Z[~rec.valid] = np.nan
            Z -= rec.bias                    # static stereo-bias calibration
            ny, nx = Z.shape
            if ny < PATCH + 1 or nx < PATCH + 1:
                continue
            for _ in range(8):
                y = int(rng.integers(0, ny - PATCH))
                x = int(rng.integers(0, nx - PATCH))
                p = Z[y:y + PATCH, x:x + PATCH]
                fin = np.isfinite(p)
                if fin.mean() < MIN_VALID:
                    continue
                p = np.where(fin, p, 0.0)
                s = p.std()
                if s < 1e-6:
                    continue
                out.append(((p - p.mean()) / s).astype(np.float32))
                src.append(i)
                if len(out) >= bs:
                    break
        X = torch.tensor(np.stack(out))
        return (X, np.array(src)) if want_rec else X


def excess_kurtosis(x):
    """per-patch, on the standardized patch: E[x^4] - 3."""
    x = x.flatten(1)
    x = x - x.mean(1, keepdim=True)
    v = x.var(1, unbiased=False) + 1e-8
    return ((x ** 4).mean(1) / v ** 2) - 3.0


class MAE(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        def blk(i, o, s=1):
            return nn.Sequential(nn.Conv2d(i, o, 3, s, 1),
                                 nn.GroupNorm(8, o), nn.SiLU())
        # 2 channels: masked surface, mask indicator
        self.enc = nn.Sequential(blk(2, 32), blk(32, 64, 2), blk(64, 64),
                                 blk(64, 128, 2), blk(128, 128),
                                 blk(128, dim, 2))          # 64 -> 8
        self.dec = nn.Sequential(
            blk(dim, 128), nn.Upsample(scale_factor=2), blk(128, 64),
            nn.Upsample(scale_factor=2), blk(64, 32),
            nn.Upsample(scale_factor=2), nn.Conv2d(32, 1, 3, 1, 1))
        self.dim = dim

    def features(self, x):
        return self.enc(x).mean((2, 3))          # (B, dim) pooled

    def forward(self, x):
        h = self.enc(x)
        return self.dec(h).squeeze(1), h.mean((2, 3))


def make_mask(bs, rng, ratio, dev):
    m = torch.rand(bs, TOK, TOK, device=dev, generator=rng) < ratio
    return m.repeat_interleave(PATCH // TOK, 1).repeat_interleave(
        PATCH // TOK, 2)


def _lin_probe(F, y):
    ntr = int(0.7 * len(F))
    A = np.concatenate([F, np.ones((len(F), 1))], 1)
    W, *_ = np.linalg.lstsq(A[:ntr], y[:ntr], rcond=None)
    pred = A[ntr:] @ W
    ss = ((y[ntr:] - pred) ** 2).sum()
    st = ((y[ntr:] - y[ntr:].mean()) ** 2).sum()
    return float(1 - ss / max(st, 1e-12))


def probe(model, ds, rng, n=64, bs=64, ref=None):
    """Frozen-feature linear probes.

    TWO targets, because the obvious one is a trap:

      patch_kurt  the patch's OWN excess kurtosis. A randomly initialised conv
                  encoder already scores R^2 = 0.86 on this — pooled features
                  of random filters are a random-feature basis, and a 4th
                  moment of the visible pixels is trivially in it. Reporting
                  this as evidence of a learned representation would be the
                  same "starved baseline" error made eight times yesterday.

      rec_kurt    the excess kurtosis of the RECORD the patch came from,
                  estimated over many other patches. A single 64x64 window
                  cannot compute it; the encoder must recognise the sea state.

    Both are reported against a random-init control (`ref`), and only the
    IMPROVEMENT over that control counts.
    """
    model.eval()
    F, K, R = [], [], []
    with torch.no_grad():
        for _ in range(n):
            xb, ridx = ds.batch(rng, bs, want_rec=True)
            xb = xb.to(DEV)
            inp = torch.stack([xb, torch.zeros_like(xb)], 1)  # no masking
            F.append(model.features(inp).cpu())
            K.append(excess_kurtosis(xb).cpu())
            R.append(torch.tensor(ridx))
    F = torch.cat(F).double().numpy()
    K = torch.cat(K).double().numpy()
    R = torch.cat(R).numpy()
    # record-level kurtosis = mean patch kurtosis of that record in this sample
    rk = {r: K[R == r].mean() for r in np.unique(R)}
    Y = np.array([rk[r] for r in R])

    out = {'patch_kurt_R2': _lin_probe(F, K),
           'rec_kurt_R2': _lin_probe(F, Y) if len(rk) > 1 else float('nan'),
           'kurt_mean': float(K.mean()), 'kurt_std': float(K.std()),
           'n_records_in_sample': int(len(rk))}
    if ref is not None:
        out['patch_kurt_gain'] = out['patch_kurt_R2'] - ref['patch_kurt_R2']
        out['rec_kurt_gain'] = out['rec_kurt_R2'] - ref['rec_kurt_R2']
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=40000)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--mask', type=float, default=0.75)
    a = ap.parse_args()

    print("[train set]", flush=True)
    tr = Patches(HELD_OUT, exclude=True)
    print("[held-out set]", flush=True)
    va = Patches(HELD_OUT, exclude=False)

    model = MAE().to(DEV)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"MAE {n_par/1e6:.2f} M params on {DEV}", flush=True)

    # random-init control BEFORE any training: the probe must beat this
    ref = probe(model, tr, np.random.default_rng(7))
    print(f"  [random-init control] patch_kurt R2 {ref['patch_kurt_R2']:+.3f}"
          f"   rec_kurt R2 {ref['rec_kurt_R2']:+.3f}"
          f"   ({ref['n_records_in_sample']} records)", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    rng = np.random.default_rng(0)
    vrng = np.random.default_rng(1)
    grng = torch.Generator(device=DEV); grng.manual_seed(0)

    hist, best, t0 = [], np.inf, time.time()
    for step in range(1, a.steps + 1):
        xb = tr.batch(rng, a.bs).to(DEV)
        m = make_mask(len(xb), grng, a.mask, DEV)
        inp = torch.stack([xb * (~m), m.float()], 1)
        pred, _ = model(inp)
        loss = ((pred - xb) ** 2)[m].mean()          # masked pixels only
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        if step % 2000 == 0 or step == a.steps:
            model.eval()
            with torch.no_grad():
                xv = va.batch(vrng, 256).to(DEV)
                mv = make_mask(len(xv), grng, a.mask, DEV)
                iv = torch.stack([xv * (~mv), mv.float()], 1)
                pv, _ = model(iv)
                vl = float(((pv - xv) ** 2)[mv].mean())
            model.train()
            hist.append({'step': step, 'train': float(loss), 'val': vl})
            el = (time.time() - t0) / 60
            print(f"  step {step:6d}  train {float(loss):.4f}  "
                  f"held-out {vl:.4f}  ({el:.1f} min)", flush=True)
            if vl < best:
                best = vl
                torch.save({'model': model.state_dict(), 'step': step,
                            'val': vl, 'dim': model.dim}, CKPT)

        if step % 10000 == 0 or step == a.steps:
            pr = probe(model, tr, np.random.default_rng(7), ref=ref)
            print(f"    [probe @ {step}] patch_kurt R2 {pr['patch_kurt_R2']:+.3f}"
                  f" (gain {pr['patch_kurt_gain']:+.3f})   "
                  f"rec_kurt R2 {pr['rec_kurt_R2']:+.3f}"
                  f" (gain {pr['rec_kurt_gain']:+.3f})", flush=True)
            json.dump({'history': hist, 'probe': pr, 'random_init': ref,
                       'best_val': best, 'held_out': HELD_OUT},
                      open(OUT, 'w'), indent=1)

    print(f"\nbest held-out masked MSE {best:.4f}  -> {CKPT}")


if __name__ == '__main__':
    sys.exit(main())
