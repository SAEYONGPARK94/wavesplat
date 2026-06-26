"""
Direction II — Step 2a: generate a directional-spectrum dataset for the prior.

Samples JONSWAP+spreading parameters (matching the simulator's wave model),
rasterizes each component set to a 64x64 S(kx,ky) energy image. These are the
training samples for the diffusion prior p_theta(S).
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesim.wave_generator import jonswap_components
from wavesplat.model.spectral_field import rasterize_components, N_GRID

N_SAMPLES = 8000
OUT = 'results/direction2/spectra_dataset.npz'


def sample_params(rng):
    Hs    = rng.uniform(0.8, 3.2)
    Tp    = rng.uniform(5.0, 11.0)
    nc    = int(rng.integers(5, 31))
    dird  = rng.uniform(-50, 50)         # ship-forward biased
    spread= rng.uniform(12, 40)
    gamma = rng.uniform(1.5, 4.0)
    return Hs, Tp, nc, dird, spread, gamma


def main():
    rng = np.random.default_rng(0)
    S = np.zeros((N_SAMPLES, N_GRID, N_GRID), dtype=np.float32)
    meta = np.zeros((N_SAMPLES, 6), dtype=np.float32)
    kept = 0
    for i in range(N_SAMPLES):
        Hs, Tp, nc, dird, spread, gamma = sample_params(rng)
        comps = jonswap_components(Hs, Tp, n_components=nc, depth=50.0,
                                   gamma=gamma, direction_deg=dird,
                                   direction_spread_deg=spread, seed=int(rng.integers(1e9)))
        arr = np.array([[c.A, c.omega, c.kx, c.ky] for c in comps], dtype=np.float32)
        s = rasterize_components(arr)
        if s.max() <= 0:
            continue
        S[kept] = s
        meta[kept] = [Hs, Tp, nc, dird, spread, gamma]
        kept += 1
        if (i+1) % 1000 == 0:
            print(f'  {i+1}/{N_SAMPLES}  (kept {kept})')
    S = S[:kept]; meta = meta[:kept]
    np.savez_compressed(OUT, S=S, meta=meta)
    # stats
    energy = S.reshape(kept, -1).sum(1)
    occ = (S > 1e-6).reshape(kept, -1).mean(1)
    print(f'\nSaved {kept} spectra → {OUT}')
    print(f'  energy: mean={energy.mean():.4f} range=[{energy.min():.4f},{energy.max():.4f}]')
    print(f'  occupancy (nonzero frac): mean={occ.mean():.3f}')
    print(f'  shape: {S.shape}')


if __name__ == '__main__':
    main()
