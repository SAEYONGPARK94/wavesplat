"""e38b — BANK-RESOLUTION CONTROL for the e38 T4n4 headroom collapse.

e38 found H(real-ens): base +0.216 -> T4n4 (T=280s, n=1600) +0.049 (paired
t=-2.90). CONFOUND: the 51x51 bank (cell 0.1 kp) has omega-quantization
Delta_omega*T ~ 1.6 rad at T=70 but ~6.3 rad at T=280 -> the bank may fail to
REPRESENT the realization over long windows, faking the collapse.

CONTROL: rerun {base, T4n4} with banks 51 / 101 / 201 (cell 0.1 / 0.05 /
0.025 kp). Constant-phase-error scaling says T=280 needs 201 to match
base@51.

PRE-REGISTERED READING:
  - H restored at finer bank (H(T4n4,201) ~= H(base,51))  => e38 collapse was
    a bank artifact; the realization prize does NOT close with observation
    (supports Agent 2's persistence claim) -> then look at X_eb(fine): if the
    data prior cashes it at sub-mode resolution, resolution was the binder
    after all (Agent 1); if still <=0, speckle/extraction wall (Agent 2).
  - H stays collapsed at 201 => more observation genuinely closes the
    realization gap (Agent 1).
Arms per (sea, bank): ens, real, bf1, eb05 — same recipes as e38, smult=3.
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from e29_gennet_C import (REGIMES, corr, DEVICE, DEPTH, KREL)
from e30_conditionality_probe import radial_of
from e38_aperture_sweep import build_sea_ap
from d3_sa_nomp import beamform_power
from d18_relax_estimator import beamform_top, omega

OUT = 'results/direction3/e38b_bank_control.json'
SMULT = 3.0
CONDS = [('base', 70.0, 400, (51, 101)), ('T4n4', 280.0, 1600, (51, 101, 201))]


def fine_mask(nfine):
    ax = np.linspace(-KREL, KREL, nfine)
    KX, KY = np.meshgrid(ax, ax)
    kk = np.hypot(KX, KY)
    return KX, KY, (kk > 0.35) & (kk < KREL)


def deposit_fine(kxs, kys, en, kp, nfine):
    grid = np.zeros((nfine, nfine))
    sc = (nfine - 1) / (2 * KREL)
    for kx, ky, e in zip(kxs, kys, en):
        u = (kx / kp + KREL) * sc
        v = (ky / kp + KREL) * sc
        if not (0 <= u <= nfine - 1 and 0 <= v <= nfine - 1):
            continue
        i0, j0 = int(v), int(u)
        fv, fu = v - i0, u - j0
        for di, wv in ((0, 1 - fv), (1, fv)):
            for dj, wu in ((0, 1 - fu), (1, fu)):
                if i0 + di < nfine and j0 + dj < nfine:
                    grid[i0 + di, j0 + dj] += e * wv * wu
    return grid


def ens_fine(kn, en, mu, kp, spread_deg, nfine):
    s = np.deg2rad(spread_deg)
    th = np.linspace(mu - 4 * s, mu + 4 * s, 181)
    w = np.exp(-0.5 * ((th - mu) / s) ** 2); w /= w.sum()
    kxs = np.concatenate([k * kp * np.cos(th) for k in kn])
    kys = np.concatenate([k * kp * np.sin(th) for k in kn])
    ee = np.concatenate([e * w for e in en])
    return deposit_fine(kxs, kys, ee, kp, nfine)


def solve_fine(S, g_fine, KX, KY, m, kp):
    """MMSE with a free-arm-only fine bank (bw atoms would carry gamma=0)."""
    g = np.maximum(g_fine[m], 0)
    if g.sum() <= 0:
        return float('nan')
    g = g / g.sum()
    kxb = (KX[m] * kp).astype(np.float32)
    kyb = (KY[m] * kp).astype(np.float32)
    omb = omega(np.hypot(kxb, kyb) + 1e-12, DEPTH).astype(np.float32)
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=DEVICE)
    kxt, kyt, omt = t(kxb), t(kyb), t(omb)

    def design(Q):
        Qt = t(Q.astype(np.float32))
        psi = kxt[None] * Qt[:, 0:1] + kyt[None] * Qt[:, 1:2] \
            - omt[None] * Qt[:, 2:3]
        return torch.cat([torch.cos(psi), torch.sin(psi)], 1)

    gam = t(np.concatenate([g, g]).astype(np.float32) * float(np.var(S['z'])))
    Phi = design(S['P'])
    K = (Phi * gam[None]) @ Phi.T \
        + (SMULT * S['sigma0']) ** 2 * torch.eye(len(S['z']), device=DEVICE)
    w = Phi.T @ torch.linalg.solve(K, t(S['z']))
    del K, Phi
    Phi_q = design(S['Pq'])
    out = float(corr(((Phi_q * gam[None]) @ w).cpu().numpy(), S['gt_u']))
    del Phi_q, w
    torch.cuda.empty_cache()
    return out


def main(n_seas=16):
    kn, en = radial_of(7.0, 3.3)
    out = {}
    for tag, T_s, n_obs, banks in CONDS:
        rows = []
        print(f'\n===== {tag}: T={T_s:g}s n={n_obs}, banks {banks} =====',
              flush=True)
        for si in range(n_seas):
            S = build_sea_ap(si, T_s, n_obs)
            kp = S['kp']
            bx, by = beamform_top(S['P'].astype(np.float32),
                                  S['z'].astype(np.float32),
                                  DEPTH, klim=3.5 * kp, n=120)
            mu_hat = np.arctan2(by, bx)
            r = dict(si=si)
            for nf in banks:
                KX, KY, m = fine_mask(nf)
                g_ens = ens_fine(kn, en, mu_hat, kp, 30.0, nf)
                g_real = deposit_fine(S['m_kx'], S['m_ky'],
                                      S['m_A'] ** 2 / 2, kp, nf)
                _, _, pf = beamform_power(S['P'], S['z'], DEPTH,
                                          KREL * kp, nf)
                pf = (pf / max(pf.max(), 1e-12)).reshape(nf, nf)
                bfp = np.maximum(pf, 1e-8)
                r[f'ens{nf}'] = solve_fine(S, g_ens, KX, KY, m, kp)
                r[f'real{nf}'] = solve_fine(S, g_real, KX, KY, m, kp)
                r[f'bf{nf}'] = solve_fine(S, bfp, KX, KY, m, kp)
                r[f'eb{nf}'] = solve_fine(S, g_ens * bfp ** 0.5, KX, KY, m, kp)
            rows.append(r)
            msg = ' | '.join(
                f"b{nf}: ens {r[f'ens{nf}']:+.3f} real {r[f'real{nf}']:+.3f} "
                f"eb {r[f'eb{nf}']:+.3f}" for nf in banks)
            print(f'  sea{si:2d} {msg}', flush=True)
        out[tag] = dict(rows=rows, banks=list(banks))
        json.dump(out, open(OUT, 'w'), indent=1)
        for nf in banks:
            a = lambda k: np.array([x[k] for x in rows], float)
            ens, real = a(f'ens{nf}'), a(f'real{nf}')
            ok = np.isfinite(ens) & np.isfinite(real)
            H = real[ok] - ens[ok]
            Xeb = (a(f'eb{nf}') - ens)[ok]
            Xbf = (a(f'bf{nf}') - ens)[ok]
            se = lambda d: d.std(ddof=1) / np.sqrt(len(d))
            print(f'  --> {tag} bank{nf}: ens {ens[ok].mean():+.4f} '
                  f'real {real[ok].mean():+.4f} H {H.mean():+.4f}±{se(H):.4f} '
                  f'| X_eb {Xeb.mean():+.4f}±{se(Xeb):.4f} '
                  f'(pos {(Xeb>0).sum()}/{ok.sum()}) '
                  f'| X_bf {Xbf.mean():+.4f}±{se(Xbf):.4f}', flush=True)
    print(f'saved -> {OUT}')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 16)
