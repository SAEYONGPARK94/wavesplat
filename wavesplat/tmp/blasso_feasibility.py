"""
blasso_feasibility.py  — Round-3 execution: feasibility gate for the
dispersion-manifold gridless atomic-norm (BLASSO / sliding-Frank-Wolfe) estimator.

Question (from the 4-agent novelty re-review, 2026-07-23):
  Does a gridless atomic-norm estimator on the dispersion manifold
  (a) recover off-grid components that the greedy CLEAN heuristic misses, and
  (b) reduce error / approach a bound as aperture grows,
  relative to the existing greedy (beamform + subtract) initializer pattern?

Controlled comparison: GREEDY (peak -> LS amp -> subtract, never revisit)
vs SFW (peak -> add -> JOINTLY re-optimize all atom locations+amplitudes with an
L1/atomic penalty). Same matched-filter atom search for both, so the ONLY
difference is greedy-subtract vs joint-BLASSO-slide.

Self-contained: inlines dispersion; no dependence on the wavesplat package.
Forward model matches spectral_gaussian.py:  z = sum_i w_i cos(kx_i x + ky_i y - w_i t + phi_i),
with omega = sqrt(g |k| tanh(|k| d)).
"""
import numpy as np, h5py, sys, time
from scipy.optimize import minimize

G = 9.81
RNG = np.random.default_rng(0)


def omega_of_k(kx, ky, depth):
    km = np.hypot(kx, ky)
    return np.sqrt(G * km * np.tanh(km * depth) + 1e-12)


def group_speed(kx, ky, depth):
    km = np.hypot(kx, ky) + 1e-12
    kd = km * depth
    th = np.tanh(kd)
    om = np.sqrt(G * km * th + 1e-12)
    return G * (th + kd * (1 - th ** 2)) / (2 * om + 1e-12)


def load_obs(fn, n_frames=None, stride=1):
    with h5py.File(fn, 'r') as f:
        lp = f['frames/lidar_points'][:]
        vm = f['frames/lidar_valid_mask'][:]
        ts = f['frames/timestamps'][:]
        depth = float(f['config'].attrs['depth'])
        Hs = float(f['config'].attrs['Hs'])
        wc = f['config/wave_components'][:]           # [A, omega, kx, ky, phi, Q]
        gx = f['frames/gt_grid_x'][:]
        gy = f['frames/gt_grid_y'][:]
        gtf = f['frames/gt_wave_field'][:]
    nfr = lp.shape[0] if n_frames is None else min(n_frames, lp.shape[0])
    xs, ys, tl, zs = [], [], [], []
    for fi in range(0, nfr, stride):
        v = vm[fi]; p = lp[fi][v]
        xs.append(p[:, 0]); ys.append(p[:, 1]); zs.append(p[:, 2])
        tl.append(np.full(int(v.sum()), ts[fi]))
    return dict(x=np.concatenate(xs), y=np.concatenate(ys),
                t=np.concatenate(tl), z=np.concatenate(zs),
                depth=depth, Hs=Hs, wc=wc, gx=gx, gy=gy, gtf=gtf, ts=ts, nfr=nfr)


def corr_at(kxc, kyc, x, y, t, r, depth, chunk=1500):
    """|C(k)| = |sum_m r_m exp(-i theta_m(k))| for candidate k's."""
    om = omega_of_k(kxc, kyc, depth)
    out = np.empty(len(kxc))
    for i in range(0, len(kxc), chunk):
        sl = slice(i, i + chunk)
        theta = (np.outer(kxc[sl], x) + np.outer(kyc[sl], y) - np.outer(om[sl], t))
        out[sl] = np.abs(np.exp(-1j * theta) @ r)
    return out


def fit_amp(kx, ky, x, y, t, r, depth):
    om = omega_of_k(kx, ky, depth)
    th = kx * x + ky * y - om * t
    u = np.cos(th); v = -np.sin(th)
    A = np.array([[u @ u, u @ v], [u @ v, v @ v]])
    b = np.array([u @ r, v @ r])
    a, bb = np.linalg.solve(A + 1e-9 * np.eye(2), b)
    return a, bb


def model_eval(atoms, x, y, t, depth):
    z = np.zeros_like(x)
    for (kx, ky, a, b) in atoms:
        om = omega_of_k(kx, ky, depth)
        th = kx * x + ky * y - om * t
        z += a * np.cos(th) - b * np.sin(th)
    return z


def refine_single(kx0, ky0, x, y, t, r, depth):
    """Local ascent of |C(k)| around a grid peak (off-grid refinement)."""
    def neg(p):
        return -corr_at(np.array([p[0]]), np.array([p[1]]), x, y, t, r, depth)[0]
    res = minimize(neg, [kx0, ky0], method='Nelder-Mead',
                   options=dict(xatol=1e-4, fatol=1e-6, maxiter=200))
    return res.x[0], res.x[1]


def build_grid(kmax=1.3, step=0.02):
    g = np.arange(-kmax, kmax + 1e-9, step)
    KX, KY = np.meshgrid(g, g, indexing='xy')
    return KX.ravel(), KY.ravel()


def greedy(obs, n_atoms, gridk, sub_idx):
    x, y, t, z, depth = obs['x'], obs['y'], obs['t'], obs['z'], obs['depth']
    KXf, KYf = gridk
    xs, ys, ts = x[sub_idx], y[sub_idx], t[sub_idx]
    r = z.copy()
    atoms = []
    for _ in range(n_atoms):
        rs = r[sub_idx]
        cm = corr_at(KXf, KYf, xs, ys, ts, rs, depth)
        j = cm.argmax()
        kx, ky = refine_single(KXf[j], KYf[j], x, y, t, r, depth)
        a, b = fit_amp(kx, ky, x, y, t, r, depth)
        atoms.append((kx, ky, a, b))
        r = r - model_eval([(kx, ky, a, b)], x, y, t, depth)
    return atoms


def sfw(obs, n_atoms, gridk, sub_idx, lam):
    x, y, t, z, depth = obs['x'], obs['y'], obs['t'], obs['z'], obs['depth']
    KXf, KYf = gridk
    # joint optimization runs on a subsample for speed
    xs, ys, ts, zs = x[sub_idx], y[sub_idx], t[sub_idx], z[sub_idx]
    atoms = []

    def unpack(p):
        J = len(p) // 4
        return p[:J], p[J:2*J], p[2*J:3*J], p[3*J:4*J]

    def obj(p):
        kx, ky, a, b = unpack(p)
        om = omega_of_k(kx, ky, depth)
        th = (np.outer(xs, kx) + np.outer(ys, ky) - np.outer(ts, om))  # (M,J)
        model = (np.cos(th) * a - np.sin(th) * b).sum(1)
        res = zs - model
        return 0.5 * res @ res + lam * np.sqrt(a**2 + b**2 + 1e-9).sum()

    for _ in range(n_atoms):
        rfull = z - model_eval(atoms, x, y, t, depth)
        cm = corr_at(KXf, KYf, x[sub_idx], y[sub_idx], t[sub_idx], rfull[sub_idx], depth)
        j = cm.argmax()
        peak = cm[j] / len(sub_idx)
        if atoms and peak < lam / len(sub_idx):
            break
        kx0, ky0 = refine_single(KXf[j], KYf[j], x, y, t, rfull, depth)
        a0, b0 = fit_amp(kx0, ky0, x, y, t, rfull, depth)
        cur = atoms + [(kx0, ky0, a0, b0)]
        J = len(cur)
        p0 = np.concatenate([[c[0] for c in cur], [c[1] for c in cur],
                             [c[2] for c in cur], [c[3] for c in cur]])
        res = minimize(obj, p0, method='L-BFGS-B',
                       options=dict(maxiter=300, ftol=1e-10))
        kx, ky, a, b = unpack(res.x)
        atoms = [(kx[i], ky[i], a[i], b[i]) for i in range(J)]
    return atoms


def joint_refine(atoms, obs, sub_idx, lam):
    """One joint L-BFGS refine of ALL atom locations+amplitudes — mimics the
    existing WaveSplat init->coarse_to_fine (Adam) joint polish with L1 amps."""
    if not atoms:
        return atoms
    x, y, t, z, depth = obs['x'], obs['y'], obs['t'], obs['z'], obs['depth']
    xs, ys, ts, zs = x[sub_idx], y[sub_idx], t[sub_idx], z[sub_idx]
    J = len(atoms)

    def unpack(p):
        return p[:J], p[J:2*J], p[2*J:3*J], p[3*J:4*J]

    def obj(p):
        kx, ky, a, b = unpack(p)
        om = omega_of_k(kx, ky, depth)
        th = np.outer(xs, kx) + np.outer(ys, ky) - np.outer(ts, om)
        model = (np.cos(th) * a - np.sin(th) * b).sum(1)
        res = zs - model
        return 0.5 * res @ res + lam * np.sqrt(a**2 + b**2 + 1e-9).sum()

    p0 = np.concatenate([[c[0] for c in atoms], [c[1] for c in atoms],
                         [c[2] for c in atoms], [c[3] for c in atoms]])
    res = minimize(obj, p0, method='L-BFGS-B', options=dict(maxiter=400, ftol=1e-10))
    kx, ky, a, b = unpack(res.x)
    return [(kx[i], ky[i], a[i], b[i]) for i in range(J)]


def match_to_gt(atoms, wc):
    """Match recovered atoms to GT comps by nearest |k|+dir; report per GT."""
    gk = np.stack([wc[:, 2], wc[:, 3]], 1)           # kx,ky
    gA = wc[:, 0]
    rk = np.array([[a[0], a[1]] for a in atoms]) if atoms else np.zeros((0, 2))
    rA = np.array([np.hypot(a[2], a[3]) for a in atoms]) if atoms else np.zeros(0)
    rows = []
    for i in range(len(gk)):
        km_g = np.hypot(*gk[i]); dir_g = np.degrees(np.arctan2(gk[i, 1], gk[i, 0]))
        if len(rk):
            d = np.hypot(rk[:, 0] - gk[i, 0], rk[:, 1] - gk[i, 1])
            j = d.argmin()
            km_r = np.hypot(*rk[j]); dir_r = np.degrees(np.arctan2(rk[j, 1], rk[j, 0]))
            dkm = km_r - km_g
            ddir = (dir_r - dir_g + 180) % 360 - 180
            hit = (abs(dkm) < 0.06) and (abs(ddir) < 15)
            rows.append((km_g, dir_g, gA[i], dkm, ddir, rA[j] / (gA[i] + 1e-9), hit))
        else:
            rows.append((km_g, dir_g, gA[i], np.nan, np.nan, 0, False))
    return rows


def field_nrmse(atoms, obs, frame):
    gx, gy, gtf, Hs = obs['gx'], obs['gy'], obs['gtf'], obs['Hs']
    t = obs['ts'][frame]
    X, Y = np.meshgrid(gx, gy, indexing='xy')
    pred = model_eval(atoms, X.ravel(), Y.ravel(), np.full(X.size, t), obs['depth']).reshape(X.shape)
    truth = gtf[frame]
    return np.sqrt(np.mean((pred - truth) ** 2)) / Hs


def run_episode(fn, n_true, extra=1):
    print("\n" + "=" * 78)
    print("EPISODE:", fn.split('/')[-1])
    obs = load_obs(fn)
    print(f"  obs points={len(obs['x'])}  frames={obs['nfr']}  Hs={obs['Hs']}  depth={obs['depth']}")
    gridk = build_grid()
    M = len(obs['x'])
    sub = RNG.choice(M, size=min(2500, M), replace=False)
    n_atoms = n_true + extra
    lam = 0.03 * np.abs(corr_at(*gridk, obs['x'][sub], obs['y'][sub], obs['t'][sub],
                                obs['z'][sub], obs['depth']).max())
    t0 = time.time()
    g_atoms = greedy(obs, n_atoms, gridk, sub)
    gj_atoms = joint_refine(g_atoms, obs, sub, lam)   # greedy + joint polish = current pipeline analog
    s_atoms = sfw(obs, n_atoms, gridk, sub, lam)
    # oracle: fit amplitudes at TRUE k (upper bound on field fit for linear model)
    o_atoms = []
    for c in obs['wc']:
        a, b = fit_amp(c[2], c[3], obs['x'], obs['y'], obs['t'], obs['z'], obs['depth'])
        o_atoms.append((c[2], c[3], a, b))
    fr = obs['nfr'] // 2
    print(f"  [{time.time()-t0:.1f}s] field nRMSE @frame{fr}:  "
          f"GREEDY={field_nrmse(g_atoms,obs,fr):.4f}  "
          f"GREEDY+JOINT={field_nrmse(gj_atoms,obs,fr):.4f}  "
          f"SFW={field_nrmse(s_atoms,obs,fr):.4f}  "
          f"oracle-k={field_nrmse(o_atoms,obs,fr):.4f}")
    for label, atoms in [("GREEDY", g_atoms), ("SFW", s_atoms)]:
        rows = match_to_gt(atoms, obs['wc'])
        nhit = sum(r[6] for r in rows)
        print(f"  {label}: recovered {nhit}/{n_true} true comps (|Δk|<.06 & |Δdir|<15°)")
        for r in rows:
            km_g, dir_g, A, dkm, ddir, ar, hit = r
            print(f"      GT |k|={km_g:.3f} dir={dir_g:+.0f}° A={A:.3f} | "
                  f"Δ|k|={dkm:+.3f} Δdir={ddir:+.0f}° amp_ratio={ar:.2f}  {'HIT' if hit else 'MISS'}")
    return obs, gridk, sub, n_atoms, lam


def aperture_sweep(fn, n_true, extra=1):
    print("\n  --- aperture sweep (weak/short-λ comp |k| error vs obs window) ---")
    gridk = build_grid()
    for nfr in [30, 100, 301]:
        obs = load_obs(fn, n_frames=nfr)
        M = len(obs['x']); sub = RNG.choice(M, size=min(2500, M), replace=False)
        lam = 0.03 * np.abs(corr_at(*gridk, obs['x'][sub], obs['y'][sub], obs['t'][sub],
                                    obs['z'][sub], obs['depth']).max())
        g_atoms = greedy(obs, n_true + extra, gridk, sub)
        s_atoms = sfw(obs, n_true + extra, gridk, sub, lam)
        # weakest true comp = smallest amplitude
        wi = obs['wc'][:, 0].argmin()
        gr = match_to_gt(g_atoms, obs['wc'])[wi]
        sr = match_to_gt(s_atoms, obs['wc'])[wi]
        print(f"    window={nfr*0.1:4.1f}s ({M:5d} pts): weak comp |k|={np.hypot(obs['wc'][wi,2],obs['wc'][wi,3]):.3f}  "
              f"Δ|k| GREEDY={gr[3]:+.3f}({'hit' if gr[6] else 'MISS'})  SFW={sr[3]:+.3f}({'hit' if sr[6] else 'MISS'})")


if __name__ == "__main__":
    import os
    base = "data/"
    eps = [("episode_BF3_5comp_seed0_uniform.h5", 5, 2),
           ("episode_BF3_5comp_seed1_uniform.h5", 5, 2),
           ("episode_BF4_15comp_seed0_uniform.h5", 15, 3)]
    if not os.environ.get("QUICK"):
        run_episode(base + "episode_BF3_2comp_seed1_slope.h5", n_true=2)
        aperture_sweep(base + "episode_BF3_2comp_seed1_slope.h5", n_true=2)
    for fn, nt, ex in eps:
        try:
            run_episode(base + fn, n_true=nt, extra=ex)
        except Exception as e:
            print("  ERR", fn, e)
