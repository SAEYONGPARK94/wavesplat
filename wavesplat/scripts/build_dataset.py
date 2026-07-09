"""Build ONE fixed training dataset (linear + real-HOS nonlinear) for fair method comparison.

Per user: stop regenerating data every step — freeze a single dataset that all methods (A1, B1,
A2) train on, mixing LINEAR and NONLINEAR (real HOS simulator) realizations. Every realization is
standardized to M=32 free-wave components so A1/A2 (detection targets) and B1 (atom set) share the
SAME labels.

Realization = {atoms (M,3)=(kx,ky,A), obs P (M_OBS,3)=(x,y,t), z (M_OBS,), nl flag, Hs, Tp}.
  - linear : jonswap_components(32) + analytic linear elevation.
  - HOS    : HOSSimulator init→top-32 free modes (kx>0 half-plane) as labels; obs sampled from the
             nonlinearly-evolved surface across the time window (bound harmonics present in obs,
             labels stay the free modes → nonlinearity-robust supervision).

Parallelized with multiprocessing. Saved to data/fixed_train.npz.

Run: python3 scripts/build_dataset.py --n 5000 --frac_hos 0.35 --workers 8
"""
import os, sys, time, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from multiprocessing import Pool
from wavesim.wave_generator import jonswap_components
from wavesim.hos_simulator import HOSSimulator, HOSConfig

M = 32; M_OBS = 400; DEPTH = 50.0; K_TP = 2.2; DT = 0.1
_POOL = None


def _init():
    global _POOL
    sys.path.insert(0, 'scripts')
    from d40_learned_acquisition import pool_xy
    _POOL = pool_xy().astype(np.float32)


def _sea(rng):
    return (rng.uniform(0.8, 3.2), rng.uniform(5.0, 11.0), rng.uniform(-50, 50),
            rng.uniform(12, 40), rng.uniform(1.5, 4.0))


def gen_linear(seed):
    rng = np.random.default_rng(seed)
    Hs, Tp, dird, spread, gamma = _sea(rng)
    c = jonswap_components(Hs, Tp, n_components=M, depth=DEPTH, gamma=gamma,
                           direction_deg=dird, direction_spread_deg=spread, seed=int(rng.integers(1e9)))
    kx = np.array([x.kx for x in c], np.float32); ky = np.array([x.ky for x in c], np.float32)
    A = np.array([x.A for x in c], np.float32); om = np.array([x.omega for x in c])
    ph = np.array([x.phi for x in c])
    Tobs = K_TP * Tp
    idx = rng.choice(len(_POOL), M_OBS, replace=len(_POOL) < M_OBS)
    ox, oy = _POOL[idx, 0], _POOL[idx, 1]; ot = rng.uniform(0, Tobs, M_OBS)
    psi = kx[None] * ox[:, None] + ky[None] * oy[:, None] - om[None] * ot[:, None] + ph[None]
    z = (A[None] * np.cos(psi)).sum(1) + rng.normal(0, 0.02, M_OBS)
    atoms = np.stack([kx, ky, A], 1).astype(np.float32)
    P = np.column_stack([ox, oy, ot]).astype(np.float32)
    return atoms, P, z.astype(np.float32), np.float32(Hs), np.float32(Tp), np.int8(0)


def gen_hos(seed):
    rng = np.random.default_rng(seed)
    Hs, Tp, dird, spread, gamma = _sea(rng)
    Hs = min(Hs, 2.5)               # cap steepness — HOS (no dealiasing) blows up for steep seas
    cfg = HOSConfig(Lx=500.0, Ly=400.0, Nx=64, Ny=64, depth=DEPTH, M=3)
    sim = HOSSimulator(cfg)
    sim.initialize_jonswap(Hs=Hs, Tp=Tp, gamma=gamma, direction_deg=dird,
                           direction_spread_deg=spread, seed=int(rng.integers(1e9)))
    # top-M free modes in the kx>0 half-plane (one per ±k conjugate pair)
    eh = sim.eta_hat; mag = np.where(sim.KX > 1e-9, np.abs(eh), 0.0)
    flat = np.argsort(mag.ravel())[::-1][:M]
    kx = sim.KX.ravel()[flat].astype(np.float32); ky = sim.KY.ravel()[flat].astype(np.float32)
    A = (2.0 * np.abs(eh).ravel()[flat]).astype(np.float32)     # cosine amplitude (±k paired)
    ph = np.angle(eh).ravel()[flat].astype(np.float32); om = sim.omega.ravel()[flat]
    atoms = np.stack([kx, ky, A], 1).astype(np.float32)
    # windowed obs from the nonlinearly-evolving surface
    Tobs = K_TP * Tp; n_times = 40; ppt = M_OBS // n_times
    step_between = max(1, int(round((Tobs / n_times) / DT)))
    Plist, zlist, t, blew = [], [], 0.0, False
    with np.errstate(over='ignore', invalid='ignore'):
        for it in range(n_times):
            if it > 0:
                for _ in range(step_between):
                    sim.step(DT); t += DT
            idx = rng.choice(len(_POOL), ppt, replace=False)
            ox, oy = _POOL[idx, 0], _POOL[idx, 1]
            zo = sim.evaluate_at(ox.reshape(1, -1), oy.reshape(1, -1))[0].ravel()
            if not np.isfinite(zo).all() or np.abs(zo).max() > 8.0 * max(Hs, 0.5):
                blew = True; break                       # HOS numerical blow-up
            zlist.append(zo + rng.normal(0, 0.02, ppt))
            Plist.append(np.column_stack([ox, oy, np.full(ppt, t)]))
    if blew:
        # fallback: LINEAR obs from the same free modes (labels intact, obs just linear)
        idx = rng.choice(len(_POOL), M_OBS, replace=len(_POOL) < M_OBS)
        ox, oy = _POOL[idx, 0], _POOL[idx, 1]; ot = rng.uniform(0, Tobs, M_OBS)
        psi = kx[None] * ox[:, None] + ky[None] * oy[:, None] - om[None] * ot[:, None] + ph[None]
        z = (A[None] * np.cos(psi)).sum(1) + rng.normal(0, 0.02, M_OBS)
        P = np.column_stack([ox, oy, ot]).astype(np.float32)
        return atoms, P.astype(np.float32), z.astype(np.float32), np.float32(Hs), np.float32(Tp), np.int8(2)
    P = np.vstack(Plist).astype(np.float32); z = np.concatenate(zlist).astype(np.float32)
    return atoms, P, z, np.float32(Hs), np.float32(Tp), np.int8(1)


def _gen(args):
    kind, seed = args
    return gen_hos(seed) if kind == 1 else gen_linear(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=5000)
    ap.add_argument('--frac_hos', type=float, default=0.35)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out', default='data/fixed_train.npz')
    ap.add_argument('--seed0', type=int, default=10000)
    a = ap.parse_args()
    n_hos = int(round(a.n * a.frac_hos))
    kinds = np.array([1] * n_hos + [0] * (a.n - n_hos))
    rng = np.random.default_rng(0); rng.shuffle(kinds)
    tasks = [(int(k), a.seed0 + i) for i, k in enumerate(kinds)]
    print(f"generating {a.n} realizations ({n_hos} HOS + {a.n-n_hos} linear) on {a.workers} workers...", flush=True)

    t0 = time.time()
    atoms = np.zeros((a.n, M, 3), np.float32); P = np.zeros((a.n, M_OBS, 3), np.float32)
    Z = np.zeros((a.n, M_OBS), np.float32); NL = np.zeros(a.n, np.int8)
    HS = np.zeros(a.n, np.float32); TP = np.zeros(a.n, np.float32)
    with Pool(a.workers, initializer=_init) as pool:
        for i, r in enumerate(pool.imap(_gen, tasks, chunksize=4)):
            atoms[i], P[i], Z[i], HS[i], TP[i], NL[i] = r
            if (i + 1) % 250 == 0:
                print(f"  {i+1}/{a.n}  ({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(a.out, atoms=atoms, P=P, z=Z, nl=NL, Hs=HS, Tp=TP)
    print(f"saved {a.out}  ({time.time()-t0:.0f}s)  atoms{atoms.shape} P{P.shape}  "
          f"lin={int((NL==0).sum())} HOS={int((NL==1).sum())} HOS-blewup→linfallback={int((NL==2).sum())}",
          flush=True)
    # sanity
    for tag, m in (('lin', NL == 0), ('hos', NL == 1), ('fb', NL == 2)):
        if m.sum():
            zi = Z[m]; ai = atoms[m]
            print(f"  [{tag}] z.std {zi.std():.3f}  A.mean {ai[:,:,2].mean():.3f}  "
                  f"|k| range [{np.sqrt(ai[:,:,0]**2+ai[:,:,1]**2).min():.3f},"
                  f"{np.sqrt(ai[:,:,0]**2+ai[:,:,1]**2).max():.3f}]", flush=True)


if __name__ == '__main__':
    main()
