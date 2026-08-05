"""Independent verifier (fable) — GPU part.
(1) replicate e30 arms A/B bitwise-ish;
(2) interpolation (rotation-blur) confound control for B;
(3) leave-one-out mean map (in-sample leakage control) for B;
(4) smult=3 confound: per-arm smult sweep with GT-peek best, for
    own / mean / ens_blind / ens_oracle / real_free;
(5) spot-check rung1 'cur' (full incumbent path) for 3 seas.
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from e29_gennet_C import (build_sea_c, deposit_grid, bank_atoms, design_t,
                          corr, DEVICE, DEPTH, _MASK, REGIMES, load_incumbent,
                          maps6, NGRID, KREL, incumbent_field)
from e29_gate0_ladder import universal_radial, ensemble_grid
from e30_conditionality_probe import rotate_map, net_gamma_map
from d18_relax_estimator import beamform_top
from d3_sa_achievability_gap import hashabs

OUT = 'results/direction3/verify_fable_gpu.json'
SMULTS = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]


def solve_smult(pre, g_free_grid, smult):
    """Identical math to e29_gate0_ladder.solve_with_grid (free arm only),
    with Phi hoisted and smult a parameter."""
    Phi, Phi_q, zt, gt, s0, varz = pre
    gb = np.zeros_like(g_free_grid)
    g = np.concatenate([g_free_grid[_MASK], gb[_MASK]])
    if g.sum() <= 0:
        return float('nan')
    g = g / g.sum()
    gam = torch.tensor(np.concatenate([g, g]) * varz, dtype=torch.float32,
                       device=DEVICE)
    K = (Phi * gam[None]) @ Phi.T + (smult * s0) ** 2 * torch.eye(
        len(zt), device=DEVICE)
    w = Phi.T @ torch.linalg.solve(K, zt)
    return float(corr(((Phi_q * gam[None]) @ w).cpu().numpy(), gt))


def main(n_seas=16):
    kmag_u, en_u = universal_radial()
    ref30 = json.load(open('results/direction3/e30_conditionality.json'))['rows']
    ref29 = json.load(open('results/direction3/e29_gate0_ceiling.json'))['rows']
    R = REGIMES['spread']
    seas, maps, mus, mus_true, pres = [], [], [], [], []
    for si in range(n_seas):
        S = build_sea_c('spread', si)
        Y = net_gamma_map(S)
        bx, by = beamform_top(S['P'].astype(np.float32),
                              S['z'].astype(np.float32),
                              DEPTH, klim=3.5 * S['kp'], n=120)
        mu = np.arctan2(by, bx)
        rng = np.random.default_rng(7000 + 31 * si + hashabs('spread'))
        _ = rng.uniform(*R['hs'])
        mu_t = np.deg2rad(rng.uniform(0, 360))
        kxb, kyb, om = bank_atoms(S['kp'])
        t = lambda a: torch.tensor(a, dtype=torch.float32, device=DEVICE)
        Phi = design_t(t(kxb), t(kyb), t(om), t(S['P']))
        Phi_q = design_t(t(kxb), t(kyb), t(om), t(S['Pq']))
        pre = (Phi, Phi_q, t(S['z']), S['gt_u'], S['sigma0'],
               float(np.var(S['z'])))
        seas.append(S); maps.append(Y); mus.append(mu); mus_true.append(mu_t)
        pres.append(pre)
        print(f'built sea{si}', flush=True)

    aligned = np.stack([rotate_map(maps[i], -mus[i]) for i in range(n_seas)])
    mean_aligned = aligned.mean(0)

    rows = []
    for i in range(n_seas):
        S, Y, mu = seas[i], maps[i], mus[i]
        pre = pres[i]
        g_own = np.maximum(10.0 ** Y, 1e-8)
        # interpolation control: identity rotate-unrotate of the OWN map
        Y_rr = rotate_map(rotate_map(Y, -mu), mu)
        g_own_rr = np.maximum(10.0 ** Y_rr, 1e-8)
        g_mean = np.maximum(10.0 ** rotate_map(mean_aligned, mu), 1e-8)
        loo = (aligned.sum(0) - aligned[i]) / (n_seas - 1)
        g_loo = np.maximum(10.0 ** rotate_map(loo, mu), 1e-8)
        g_ensb = ensemble_grid(kmag_u, en_u, mu, S['kp'])
        g_enso = ensemble_grid(kmag_u, en_u, mus_true[i], S['kp'])
        g_real = deposit_grid(S['m_kx'], S['m_ky'], S['m_A'] ** 2 / 2, S['kp'])
        r = dict(si=i)
        for tag, g in (('own', g_own), ('own_rr', g_own_rr), ('mean', g_mean),
                       ('loo', g_loo), ('ensb', g_ensb), ('enso', g_enso),
                       ('real', g_real)):
            for sm in SMULTS:
                r[f'{tag}@{sm:g}'] = solve_smult(pre, g, sm)
        r['e30_own'] = ref30[i]['cur_own']
        r['e30_mean'] = ref30[i]['cur_mean']
        rows.append(r)
        print(f"sea{i:2d} own@3 {r['own@3']:+.4f} (e30 {r['e30_own']:+.4f}) "
              f"own_rr@3 {r['own_rr@3']:+.4f} mean@3 {r['mean@3']:+.4f} "
              f"(e30 {r['e30_mean']:+.4f}) loo@3 {r['loo@3']:+.4f}", flush=True)

    a = lambda k: np.array([r[k] for r in rows], float)

    def tst(d):
        m = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d))
        return m, se, (m / se if se > 0 else 0), int((d > 0).sum())

    print('\n=== replication of e30 (should be ~0) ===')
    print(f"  own@3 - e30_cur_own : max abs dev {np.abs(a('own@3')-a('e30_own')).max():.2e}")
    print(f"  mean@3 - e30_cur_mean: max abs dev {np.abs(a('mean@3')-a('e30_mean')).max():.2e}")

    print('\n=== (2) interpolation confound: own_rr@3 - own@3 ===')
    m, se, tt, npos = tst(a('own_rr@3') - a('own@3'))
    print(f"  {m:+.4f} +- {se:.4f} t {tt:+.2f} pos {npos}/16")
    print(f"  => B-A corrected for interpolation blur (B - own_rr): ", end='')
    m2, se2, t2, np2 = tst(a('mean@3') - a('own_rr@3'))
    print(f"{m2:+.4f} +- {se2:.4f} t {t2:+.2f} pos {np2}/16")

    print('\n=== (3) leakage control: loo@3 vs mean@3, loo@3 - own@3 ===')
    m, se, tt, npos = tst(a('loo@3') - a('mean@3'))
    print(f"  loo - mean: {m:+.4f} +- {se:.4f}")
    m, se, tt, npos = tst(a('loo@3') - a('own@3'))
    print(f"  loo - own : {m:+.4f} +- {se:.4f} t {tt:+.2f} pos {npos}/16")

    print('\n=== (4) smult sweep (per-smult means) ===')
    print(f"{'arm':>8}" + ''.join(f"{f'@{s:g}':>10}" for s in SMULTS) + f"{'best(GT)':>10}")
    best = {}
    for tag in ('own', 'own_rr', 'mean', 'loo', 'ensb', 'enso', 'real'):
        vals = {sm: a(f'{tag}@{sm:g}') for sm in SMULTS}
        b = np.max(np.stack([vals[sm] for sm in SMULTS]), axis=0)
        best[tag] = b
        print(f"{tag:>8}" + ''.join(f"{vals[sm].mean():+10.4f}" for sm in SMULTS)
              + f"{b.mean():+10.4f}")
    print('\n  B-A at each smult and at per-sea best:')
    for sm in SMULTS:
        m, se, tt, npos = tst(a(f'mean@{sm:g}') - a(f'own@{sm:g}'))
        print(f"    smult {sm:4g}: mean-own {m:+.4f} +- {se:.4f} t {tt:+.2f} pos {npos}/16")
    m, se, tt, npos = tst(best['mean'] - best['own'])
    print(f"    BEST     : mean-own {m:+.4f} +- {se:.4f} t {tt:+.2f} pos {npos}/16")
    m, se, tt, npos = tst(best['ensb'] - best['own'])
    print(f"    BEST     : ensb-own {m:+.4f} +- {se:.4f} t {tt:+.2f} pos {npos}/16")
    m, se, tt, npos = tst(best['real'] - best['own'])
    print(f"    BEST     : real-own {m:+.4f} +- {se:.4f} t {tt:+.2f} pos {npos}/16")

    print('\n=== (5) rung1 cur spot-check (full incumbent path) ===')
    for i in (0, 1, 6):
        S = seas[i]
        f = incumbent_field(S['P'], S['z'], S['Pq'], S['kp'], S['sigma0'])
        c = corr(f, S['gt_u'])
        print(f"  sea{i}: incumbent_field corr {c:+.4f}  vs ceiling-json cur "
              f"{ref29[i]['cur']:+.4f}")

    json.dump(rows, open(OUT, 'w'), indent=1, default=float)
    print(f'saved -> {OUT}')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 16)
