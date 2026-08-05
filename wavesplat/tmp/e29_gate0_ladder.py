"""gate-0 LADDER — decompose C's +0.198 class headroom into
"regime specialization (free, algorithm-side)" vs "realization-specific (information-side)".

Motivated by the fable discussant's decisive point (Opus verified):
  build_sea_c hardcodes Tp=7.0, gamma=3.3, spread=30, n_components=200, so the
  RADIAL profile of gamma_true is a UNIVERSAL CONSTANT across seas:
      |k_n|/kp   inter-sea max deviation = 3.55e-15
      normalized A_n^2/2  inter-sea max deviation = 2.78e-17
  The generator's rng touches ONLY phases and directions. Hs cancels (gamma is
  normalized then multiplied by var(z)). So gate-0's oracle knows:
      (a) a universal radial profile   -> knowable A PRIORI (not sensor info)
      (b) the mean direction, 1 scalar -> beamform_top ALREADY estimates it
      (c) the 200 individual direction draws -> genuine realization info
  If (a)+(b) buy most of the +0.198, the headroom is an ALGORITHM/specialization
  gap, not an information wall, and BOTH the section-19 "information wall" reading
  and the gate-0 "inference wall" reading die.

RUNGS (identical arena, identical solver, fixed smult=3 / blur=0 -> no GT peek):
  1 cur        : the incumbent (deployed config)                    [baseline]
  2 ens_blind  : universal radial x N(mu_hat_beamform, 30 deg)      [ZERO oracle info]
  3 ens_oracle : universal radial x N(mu_true, 30 deg)              [+ true mean dir]
  4 real_free  : the actual 200 draws, free arm only                [+ realization]
  4b real_full : the actual draws + true bound-wave arm             [= C's gate-0]
All of 2/3/4 use the FREE arm only, so the ladder is apples-to-apples; 4b is shown
only to connect back to C's published +0.198.
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from e29_gennet_C import (build_sea_c, deposit_grid, bank_atoms, design_t,
                          corr, DEVICE, DEPTH, _MASK, REGIMES)
from wavesim.wave_generator import jonswap_components
from d18_relax_estimator import beamform_top
from d3_sa_achievability_gap import hashabs

OUT = 'results/direction3/e29_gate0_ladder.json'
SMULT = 3.0          # fixed, no per-sea GT selection
NTH = 181            # directional quadrature nodes
SPREAD_DEG = 30.0


def universal_radial():
    """(|k_n|, normalized energy) of the hardcoded eval regime. Sea-independent."""
    c = jonswap_components(Hs=2.0, Tp=7.0, n_components=200, depth=DEPTH,
                           gamma=3.3, direction_deg=0.0,
                           direction_spread_deg=1e-6, seed=1234)
    kmag = np.array([np.hypot(x.kx, x.ky) for x in c])
    A = np.array([x.A for x in c])
    e = A ** 2 / 2
    return kmag, e / e.sum()


def ensemble_grid(kmag, en, mu_rad, kp):
    """Analytic ensemble spectrum: radial profile smeared over N(mu, 30 deg)."""
    s = np.deg2rad(SPREAD_DEG)
    th = np.linspace(mu_rad - 4 * s, mu_rad + 4 * s, NTH)
    w = np.exp(-0.5 * ((th - mu_rad) / s) ** 2)
    w /= w.sum()
    kxs = np.concatenate([k * np.cos(th) for k in kmag])
    kys = np.concatenate([k * np.sin(th) for k in kmag])
    ens = np.concatenate([e * w for e in en])
    return deposit_grid(kxs, kys, ens, kp)


def solve_with_grid(S, g_free_grid, g_bw_grid=None):
    """Same solver path as C's oracle_field, with a caller-supplied gamma."""
    P, z, Pq, gt = S['P'], S['z'], S['Pq'], S['gt_u']
    kp, s0 = S['kp'], S['sigma0']
    gb = np.zeros_like(g_free_grid) if g_bw_grid is None else g_bw_grid
    g = np.concatenate([g_free_grid[_MASK], gb[_MASK]])
    if g.sum() <= 0:
        return float('nan')
    g = g / g.sum()
    kxb, kyb, om = bank_atoms(kp)
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=DEVICE)
    Phi = design_t(t(kxb), t(kyb), t(om), t(P))
    Phi_q = design_t(t(kxb), t(kyb), t(om), t(Pq))
    gam = t(np.concatenate([g, g]) * float(np.var(z)))
    K = (Phi * gam[None]) @ Phi.T + (SMULT * s0) ** 2 * torch.eye(len(z), device=DEVICE)
    w = Phi.T @ torch.linalg.solve(K, t(z))
    return float(corr(((Phi_q * gam[None]) @ w).cpu().numpy(), gt))


def main(n_seas=16):
    kmag_u, en_u = universal_radial()
    ref = json.load(open('results/direction3/e29_gate0_ceiling.json'))['rows']
    R = REGIMES['spread']
    rows = []
    for si in range(n_seas):
        S = build_sea_c('spread', si)
        cur = ref[si]['cur']
        # replicate build_sea_c's rng to recover the TRUE mean direction
        rng = np.random.default_rng(7000 + 31 * si + hashabs('spread'))
        _ = rng.uniform(*R['hs'])
        mu_true = np.deg2rad(rng.uniform(0, 360))
        # blind mean direction: beamforming on the data (no GT)
        bx, by = beamform_top(S['P'].astype(np.float32), S['z'].astype(np.float32),
                             DEPTH, klim=3.5 * S['kp'], n=120)
        mu_hat = np.arctan2(by, bx)
        r2 = solve_with_grid(S, ensemble_grid(kmag_u, en_u, mu_hat, S['kp']))
        r3 = solve_with_grid(S, ensemble_grid(kmag_u, en_u, mu_true, S['kp']))
        gf = deposit_grid(S['m_kx'], S['m_ky'], S['m_A'] ** 2 / 2, S['kp'])
        r4 = solve_with_grid(S, gf)
        gbw = None
        if S['bw'] is not None:
            b = S['bw']
            gbw = deposit_grid(b['kx'] / 2, b['ky'] / 2,
                               (S['bw_sc'] * b['A']) ** 2 / 2, S['kp'])
        r4b = solve_with_grid(S, gf, gbw)
        derr = np.rad2deg(np.abs(np.angle(np.exp(1j * (mu_hat - mu_true)))))
        rows.append(dict(si=si, cur=cur, ens_blind=r2, ens_oracle=r3,
                         real_free=r4, real_full=r4b, dir_err_deg=float(derr)))
        r = rows[-1]
        print(f"sea{si:2d} cur {cur:+.3f} | ens_blind {r2:+.3f}({r2-cur:+.3f}) "
              f"| ens_oracle {r3:+.3f}({r3-cur:+.3f}) | real_free {r4:+.3f}({r4-cur:+.3f}) "
              f"| real_full {r4b:+.3f}({r4b-cur:+.3f}) | dir_err {derr:5.1f}deg", flush=True)

    a = lambda k: np.array([r[k] for r in rows], float)
    cur = a('cur')
    print(f"\n=== gate-0 LADDER (n={len(rows)}, smult={SMULT} fixed, no GT peek) ===")
    for k, lab in [('ens_blind', '2 ens_blind  (오라클 정보 0)'),
                   ('ens_oracle', '3 ens_oracle (+참 평균방향 1스칼라)'),
                   ('real_free', '4 real_free  (+200 draw, free만)'),
                   ('real_full', '4b real_full (+참 구속파) = C의 gate-0')]:
        d = a(k) - cur
        print(f"  {lab:38s}: {a(k).mean():+.4f}  vs cur {d.mean():+.4f}  "
              f"(median {np.median(d):+.4f}, 양수 {int((d>0).sum())}/{len(d)})")
    hb, ho, hr = (a('ens_blind')-cur).mean(), (a('ens_oracle')-cur).mean(), (a('real_free')-cur).mean()
    print(f"\n  ★분해 (free 축):")
    print(f"     regime 특화 (공짜, 오라클 0)      : {hb:+.4f}")
    print(f"     + 참 평균방향이 더하는 것          : {ho-hb:+.4f}")
    print(f"     + 200 realization draw이 더하는 것 : {hr-ho:+.4f}")
    print(f"     ---- free 총 headroom             : {hr:+.4f}")
    frac = hb / hr if abs(hr) > 1e-9 else float('nan')
    print(f"\n  ★판정: 오라클-0 사다리가 총 headroom의 {frac:.1%}를 먹음")
    print(f"     -> {'알고리즘/특화 벽 (정보 벽 아님)' if frac > 0.5 else '정보 벽 (realization 고유)'}")
    print(f"  평균 방향오차(빔포밍 vs 참): {a('dir_err_deg').mean():.1f} deg")
    json.dump(dict(rows=rows, h_blind=float(hb), h_oracle=float(ho),
                   h_real=float(hr), frac_free=float(frac)), open(OUT, 'w'), indent=1)
    print(f"saved -> {OUT}")


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 16)
