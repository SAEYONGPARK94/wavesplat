"""Step 1b — DISAMBIGUATION of Step 1's net-MMSE washout (fable red-team ③⑤).
net-MMSE changed TWO variables (oracle->estimated wavenumbers AND net-OOD on the
moving-trajectory geometry). This isolates the wavenumber variable cleanly: SAME
estimator family as d74's oracle achievability (RELAX/LS + render), but with
BEAMFORM-ESTIMATED wavenumbers (relax_step) instead of oracle sea['kx'],sea['ky'].
Also emits per-policy obs-geometry stats (OOD diagnostic). Linear seas.
Usage: python3 tmp/traj_step1b_relax.py [--episodes N]
Out: results/direction1/traj_step1b_relax.json
"""
import os, sys, json, math, time, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts'); sys.path.insert(0, 'tmp')

from d74_closed_loop_trajectory import (make_sea, dom_dir, fly_path, T_OBS, DEPTH,
                                        SPEEDS_FULL, headings_straight, headings_greedy,
                                        headings_cem, headings_learned, train_policy, K)
from wavesplat.model.tnse import fit_amplitudes_phases, render_wave_field
from d18_relax_estimator import relax_step

GXN = np.linspace(-30, 120, 40)
GYN = np.linspace(-80, 80, 40)
POLICIES = ['stationary', 'straight-cross', 'greedy', 'CEM', 'learned']


def gt_field(sea):
    Xg, Yg = np.meshgrid(GXN, GYN)
    psi = (sea['kx'][None, None] * Xg[..., None] + sea['ky'][None, None] * Yg[..., None]
           - sea['om'][None, None] * T_OBS + sea['phi'][None, None])
    return (sea['A'][None, None] * np.cos(psi)).sum(-1)


def achieved(sea, speed, headings, oracle, seed=0):
    """oracle=True: d74 original (true wavenumbers). oracle=False: relax_step estimates them."""
    x, y, t, z = fly_path(sea, speed, headings, gen_z=True, seed=seed)
    obs = np.column_stack([x, y, t, z])
    if len(obs) > 1500:
        obs = obs[np.random.default_rng(0).choice(len(obs), 1500, replace=False)]
    if oracle:
        kx, ky = sea['kx'], sea['ky']
    else:
        kx, ky = relax_step([], [], obs[:, :3], obs[:, 3], DEPTH,
                            klim=3.5 * math.hypot(sea['kx'].mean(), sea['ky'].mean()) or 0.6)
        if len(kx) == 0:
            return np.nan
    A, phi = fit_amplitudes_phases(kx, ky, obs, DEPTH, lambda_reg=0.5)
    fld = render_wave_field(kx, ky, A, phi, DEPTH, GXN, GYN, T_OBS)
    gt = gt_field(sea)
    Hs = 4 * math.sqrt((sea['A'] ** 2 / 2).sum())
    return float(np.sqrt(np.mean((fld - gt) ** 2)) / Hs)


def geom_stats(sea, speed, headings, seed=0):
    x, y, t, _ = fly_path(sea, speed, headings, gen_z=False, seed=seed)
    return dict(x_extent=float(x.max() - x.min()), y_extent=float(y.max() - y.min()),
                n_obs=int(len(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=24)
    ap.add_argument('--train_eps', type=int, default=60)
    a = ap.parse_args()
    speed = SPEEDS_FULL[-1]
    print(f"training learned policy @ {speed} m/s...", flush=True)
    pol = train_policy(speed, a.train_eps)

    def headings_for(p, sea):
        return {'stationary': np.full(K, dom_dir(sea)),
                'straight-cross': headings_straight(sea, 90),
                'greedy': headings_greedy(sea, speed),
                'CEM': headings_cem(sea, speed, seed=0),
                'learned': headings_learned(sea, speed, pol)}[p]

    rows, geoms = [], {p: [] for p in POLICIES}
    t0 = time.time()
    for e in range(a.episodes):
        sea = make_sea(seed=1000 + e)
        row = {'ep': e}
        for p in POLICIES:
            spd = 0.0 if p == 'stationary' else speed
            h = headings_for(p, sea)
            row[f'{p}_oracle'] = achieved(sea, spd, h, oracle=True, seed=e)
            row[f'{p}_relax'] = achieved(sea, spd, h, oracle=False, seed=e)
            geoms[p].append(geom_stats(sea, spd, h, seed=e))
        rows.append(row)
        if e % 4 == 0:
            print(f"  ep {e} [oracle|relax]: " +
                  " ".join(f"{p[:4]}={row[f'{p}_oracle']:.2f}|{row[f'{p}_relax']:.2f}" for p in POLICIES)
                  + f"  [{time.time()-t0:.0f}s]", flush=True)
        json.dump(dict(rows=rows), open('results/direction1/traj_step1b_relax.json', 'w'), indent=1, default=float)
    analyze(rows, geoms)


def analyze(rows, geoms):
    arr = lambda k: np.array([r[k] for r in rows])
    print(f"\n=== Step 1b: achieved nRMSE — ORACLE vs RELAX(estimated) wavenumbers (n={len(rows)}) ===")
    print(f"  {'policy':16s} | {'oracle':>7s} {'Δvs stat':>9s} | {'relax':>7s} {'Δvs stat':>9s} {'sign':>6s}")
    out = {}
    for tag in ['oracle', 'relax']:
        stat = arr(f'stationary_{tag}')
        for p in POLICIES:
            v = arr(f'{p}_{tag}'); m = np.isfinite(v) & np.isfinite(stat)
            d = (stat - v)[m]
            rng = np.random.default_rng(0)
            bs = [np.mean(rng.choice(d, len(d))) for _ in range(4000)] if len(d) else [0]
            out[f'{p}_{tag}'] = dict(nrmse=float(np.nanmean(v)), delta=float(np.mean(d)),
                                     ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                     nbetter=int((d > 0).sum()), n=int(m.sum()))
    for p in POLICIES:
        o, r = out[f'{p}_oracle'], out[f'{p}_relax']
        print(f"  {p:16s} | {o['nrmse']:7.3f} {o['delta']:+9.3f} | {r['nrmse']:7.3f} {r['delta']:+9.3f} "
              f"{r['nbetter']:>3d}/{r['n']}")
    print("\n  geometry (mean obs extent, m) — OOD diagnostic vs net training (~static 400pt patch):")
    for p in POLICIES:
        g = geoms[p]
        print(f"    {p:16s} x_extent {np.mean([q['x_extent'] for q in g]):6.1f}  "
              f"y_extent {np.mean([q['y_extent'] for q in g]):6.1f}")
    # verdict: does the ORACLE advantage survive under RELAX(estimated)?
    def best(tag): return min(['straight-cross', 'CEM', 'learned'], key=lambda p: out[f'{p}_{tag}']['nrmse'])
    bo, br = best('oracle'), best('relax')
    orac_adv = out[f'{bo}_oracle']['delta'] >= 0.02 and out[f'{bo}_oracle']['ci'][0] > 0
    relax_adv = out[f'{br}_relax']['delta'] >= 0.02 and out[f'{br}_relax']['ci'][0] > 0
    if orac_adv and relax_adv:
        v = f"PASS: advantage survives ESTIMATED wavenumbers (not oracle artifact) [oracle {bo} +{out[f'{bo}_oracle']['delta']:.3f}, relax {br} +{out[f'{br}_relax']['delta']:.3f}]"
    elif orac_adv and not relax_adv:
        v = f"FAIL(clean): oracle advantage ({bo} +{out[f'{bo}_oracle']['delta']:.3f}) DISAPPEARS under estimated wavenumbers = oracle-wavenumber artifact"
    elif not orac_adv:
        v = f"INCONCLUSIVE: even oracle shows no advantage here ({bo} +{out[f'{bo}_oracle']['delta']:.3f}) — check achievability scale"
    else:
        v = "MIXED"
    print(f"\n=== VERDICT === {v}")
    out['verdict'] = v
    j = json.load(open('results/direction1/traj_step1b_relax.json')); j['summary'] = out
    j['geom'] = {p: dict(x=float(np.mean([q['x_extent'] for q in geoms[p]])),
                         y=float(np.mean([q['y_extent'] for q in geoms[p]]))) for p in POLICIES}
    json.dump(j, open('results/direction1/traj_step1b_relax.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
