"""E-30c: (A) resolution DOF count  (B) gamma-space recovery  (C) the 180 flip
         (D) candidate-4: is the +0.198 ceiling BANK-capped?

(A) how many independent k-resolution cells does a 400-pt / 70-s aperture buy,
    and how many components must share them?
(B) does ANY data-derived gamma recover speckle? corr(gamma_hat, gamma_true)
    with the ensemble component partialled out. Includes the incumbent's OWN
    gamma -- if the net already matches the best deconvolution, C's
    "unexploited sharpening mechanism" is in fact already exploited.
(C) sea6: genuine +-k ambiguity (data prefers the wrong side => not learnable)
    or an estimator slip (=> learnable)?
(D) deposit the true modes on a 2x finer bank (101x101, cell 0.05 kp) -> does
    the class ceiling move? If not, the bank is not the cap.
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from e29_gennet_C import (build_sea_c, deposit_grid, bank_power_np, corr,
                          DEVICE, DEPTH, _MASK, REGIMES, load_incumbent,
                          maps6, two_stage_kp)
from e29_gate0_ladder import universal_radial, ensemble_grid, solve_with_grid
from e30_speckle_control import bf_grid
from d18_relax_estimator import beamform_top, omega
from d3_sa_achievability_gap import hashabs
from d3_prior_net_train import NGRID, KREL

OUT = 'results/direction3/e30_dof_and_bank.json'


def part_area(g):
    """participation ratio = effective number of cells carrying the mass."""
    g = np.maximum(np.asarray(g, float), 0).ravel()
    s = g.sum()
    return float(s ** 2 / max((g ** 2).sum(), 1e-30)) if s > 0 else 0.0


def pcorr(a, b, c):
    """partial corr(a,b | c) on the bank."""
    A = np.column_stack([np.ones_like(c), c])
    ra = a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
    rb = b - A @ np.linalg.lstsq(A, b, rcond=None)[0]
    return float(np.corrcoef(ra, rb)[0, 1])


def net_gamma_grid(S):
    """the incumbent's own predicted gamma on the free bank -> 51x51 grid."""
    net, _ = load_incumbent()
    X = maps6(S['P'], S['z'], S['kp'])
    with torch.no_grad():
        Ylog = net(torch.tensor(X[None], device=DEVICE))[0].cpu().numpy()
    g = np.zeros((NGRID, NGRID))
    g[_MASK] = np.maximum(10.0 ** Ylog[_MASK], 1e-8)
    return g


def fine_bank_solve(S, nfine=101, krel=KREL, smult=3.0):
    """class ceiling on a 2x finer bank (cell 0.05 kp)."""
    ax = np.linspace(-krel, krel, nfine)
    KXn, KYn = np.meshgrid(ax, ax)
    kk = np.hypot(KXn, KYn)
    m = (kk > 0.35) & (kk < krel)
    kp, s0 = S['kp'], S['sigma0']
    # deposit true modes onto the fine grid
    grid = np.zeros((nfine, nfine))
    sc = (nfine - 1) / (2 * krel)
    for kx, ky, e in zip(S['m_kx'], S['m_ky'], S['m_A'] ** 2 / 2):
        u = (kx / kp + krel) * sc
        v = (ky / kp + krel) * sc
        if not (0 <= u <= nfine - 1 and 0 <= v <= nfine - 1):
            continue
        i0, j0 = int(v), int(u)
        fv, fu = v - i0, u - j0
        for di, wv in ((0, 1 - fv), (1, fv)):
            for dj, wu in ((0, 1 - fu), (1, fu)):
                if i0 + di < nfine and j0 + dj < nfine:
                    grid[i0 + di, j0 + dj] += e * wv * wu
    kxb = np.concatenate([KXn[m] * kp, 2 * KXn[m] * kp])
    kyb = np.concatenate([KYn[m] * kp, 2 * KYn[m] * kp])
    omb = omega(np.hypot(KXn[m] * kp, KYn[m] * kp) + 1e-12, DEPTH)
    omb = np.concatenate([omb, 2 * omb])
    g = np.concatenate([grid[m], np.zeros(int(m.sum()))])
    if g.sum() <= 0:
        return float('nan')
    g = g / g.sum()
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=DEVICE)

    def design(Q):
        psi = t(kxb)[None] * t(Q)[:, 0:1] + t(kyb)[None] * t(Q)[:, 1:2] \
            - t(omb)[None] * t(Q)[:, 2:3]
        return torch.cat([torch.cos(psi), torch.sin(psi)], 1)

    Phi, Phi_q = design(S['P']), design(S['Pq'])
    gam = t(np.concatenate([g, g]) * float(np.var(S['z'])))
    K = (Phi * gam[None]) @ Phi.T + (smult * s0) ** 2 * torch.eye(len(S['z']), device=DEVICE)
    w = Phi.T @ torch.linalg.solve(K, t(S['z']))
    out = float(corr(((Phi_q * gam[None]) @ w).cpu().numpy(), S['gt_u']))
    del Phi, Phi_q, K, w
    torch.cuda.empty_cache()
    return out


def main(n_seas=16):
    kmag_u, en_u = universal_radial()
    ref = json.load(open('results/direction3/e29_gate0_ceiling.json'))['rows']
    lad = {r['si']: r for r in
           json.load(open('results/direction3/e29_gate0_ladder.json'))['rows']}
    R = REGIMES['spread']
    rows = []
    for si in range(n_seas):
        S = build_sea_c('spread', si)
        cur, kp = ref[si]['cur'], S['kp']
        rng = np.random.default_rng(7000 + 31 * si + hashabs('spread'))
        _ = rng.uniform(*R['hs'])
        mu_true = np.deg2rad(rng.uniform(0, 360))
        # --- grids on the free bank
        gt_g = deposit_grid(S['m_kx'], S['m_ky'], S['m_A'] ** 2 / 2, kp)
        bx, by = beamform_top(S['P'].astype(np.float32), S['z'].astype(np.float32),
                              DEPTH, klim=3.5 * kp, n=120)
        mu_hat = np.arctan2(by, bx)
        ens_g = ensemble_grid(kmag_u, en_u, mu_hat, kp)
        bf_g = bf_grid(S['P'], S['z'], kp)
        net_g = net_gamma_grid(S)
        # --- PSF (unit mode at true mean dir)
        om0 = float(omega(np.array([kp]), DEPTH)[0])
        rr = np.cos(kp * np.cos(mu_true) * S['P'][:, 0]
                    + kp * np.sin(mu_true) * S['P'][:, 1] - om0 * S['P'][:, 2])
        psf_g = bf_grid(S['P'], rr, kp)
        a_psf, a_gt, a_band = part_area(psf_g), part_area(gt_g), int(_MASK.sum())
        n_res = a_gt / max(a_psf, 1e-9)
        # --- (B) speckle recovery in gamma space
        v = lambda g: np.asarray(g)[_MASK].astype(float)
        gt_v, ens_v = v(gt_g), v(ens_g)
        cc = {}
        for tag, g in (('bf', bf_g), ('net', net_g), ('ens', ens_g)):
            cc[f'r_{tag}'] = float(np.corrcoef(v(g), gt_v)[0, 1])
            if tag != 'ens':
                cc[f'p_{tag}'] = pcorr(v(g), gt_v, ens_v)
        # --- (C) +-k ambiguity: beamform power at true vs flipped mean dir
        def pw_at(mu):
            kx, ky = kp * np.cos(mu), kp * np.sin(mu)
            o = float(omega(np.array([kp]), DEPTH)[0])
            ps = kx * S['P'][:, 0] + ky * S['P'][:, 1] - o * S['P'][:, 2]
            return float((S['z'] * np.cos(ps)).sum() ** 2 + (S['z'] * np.sin(ps)).sum() ** 2)
        p_true, p_flip = pw_at(mu_true), pw_at(mu_true + np.pi)
        # --- (D) fine bank ceiling
        fine = fine_bank_solve(S)
        rows.append(dict(si=si, cur=cur, dir_err=lad[si]['dir_err_deg'],
                         a_psf=a_psf, a_gt=a_gt, a_band=a_band, n_res=n_res,
                         n_res_band=a_band / max(a_psf, 1e-9),
                         pw_true=p_true, pw_flip=p_flip,
                         flip_ratio=p_flip / max(p_true, 1e-12),
                         fine101=fine, real_free=lad[si]['real_free'], **cc))
        r = rows[-1]
        print(f"sea{si:2d} psf_area {a_psf:5.1f} gt_area {a_gt:6.1f} n_res {n_res:5.2f} "
              f"| r(bf,gt) {r['r_bf']:+.3f} r(net,gt) {r['r_net']:+.3f} r(ens,gt) {r['r_ens']:+.3f} "
              f"| p(net) {r['p_net']:+.3f} p(bf) {r['p_bf']:+.3f} "
              f"| flip {r['flip_ratio']:5.2f} | fine101 {fine:+.3f} (51: {r['real_free']:+.3f})",
              flush=True)

    a = lambda k: np.array([r[k] for r in rows], float)
    cur = a('cur')
    print(f"\n=== E-30c (n={len(rows)}) ===")
    print(f"(A) PSF 등가면적 {a('a_psf').mean():.1f} cells; 밴드 원자 {rows[0]['a_band']}")
    print(f"    -> 밴드 전체가 지지하는 독립 해상 셀 = {a('n_res_band').mean():.1f}")
    print(f"    스펙트럼 지지 등가면적 {a('a_gt').mean():.1f} cells "
          f"-> 스펙트럼 안 독립 셀 = {a('n_res').mean():.2f}  (성분 200개)")
    print(f"    => 초분해 요구 배율 = 200 / {a('n_res').mean():.2f} = {200/a('n_res').mean():.0f}x")
    print(f"\n(B) gamma 공간 회수  corr(., gamma_true):")
    for k in ('r_ens', 'r_bf', 'r_net'):
        print(f"    {k:8s} {a(k).mean():+.4f}")
    print(f"    부분상관 (ens 제거 = speckle만):")
    for k in ('p_bf', 'p_net'):
        print(f"    {k:8s} {a(k).mean():+.4f}  (양수 {int((a(k)>0).sum())}/{len(rows)})")
    print(f"\n(C) +-k: flip_ratio = P(mu+180)/P(mu_true), 전체 평균 {a('flip_ratio').mean():.2f}")
    for r in rows:
        if r['dir_err'] > 90:
            print(f"    sea{r['si']}: dir_err {r['dir_err']:.1f}deg  flip_ratio "
                  f"{r['flip_ratio']:.2f} -> {'데이터가 틀린 쪽을 선호(비식별)' if r['flip_ratio']>1 else '추정기 슬립(학습가능)'}")
    print(f"\n(D) 뱅크 상한: 101x101 fine {a('fine101').mean():+.4f} vs 51x51 "
          f"{a('real_free').mean():+.4f}  (차 {(a('fine101')-a('real_free')).mean():+.4f})")
    print(f"    fine101 vs cur {(a('fine101')-cur).mean():+.4f}")
    json.dump(dict(rows=rows), open(OUT, 'w'), indent=1)
    print(f"saved -> {OUT}")


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 16)
