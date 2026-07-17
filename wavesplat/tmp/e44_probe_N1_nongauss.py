"""e44 probe N1-nongauss — HOW MANY NATS does the non-Gaussian structure carry?

Fact 11 (orchestrator): the generator is a random-phase model with deterministic
amplitudes on a fixed frequency lattice; the ensemble-Gaussian evidence provably
cannot see that structure. This probe QUANTIFIES the ceiling of that channel:

  1. analytic phase-channel excess kurtosis  kappa4 = -1.5 * sum(A^4)/(sum(A^2))^2
     (exact for fixed directions), M_eff, noise dilution.
  2. empirical: R random-phase realizations of a fixed theta vs R Gaussian
     controls with the SAME covariance (estimated empirically), moments compared
     in the empirical-covariance eigenbasis -> per-DOF standardized cumulants.
  3. frequency-lattice resolvability: component spacing d_omega vs 2*pi/T.

Bound: per-window non-Gaussian information ~ n_dof * (g3^2/12 + g4^2/48) nats
(leading Edgeworth). Compare with the ~3 nats separating argmax from true node.
Output: tmp/e44_probe_N1_nongauss.json
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from e29_gennet_C import DEPTH
from e37_bimodal_ladder import build_sea_crosstp, SPREAD_DEG
from wavesim.wave_generator import jonswap_components

OUT = 'tmp/e44_probe_N1_nongauss.json'
R = 3000          # realizations for covariance; moments use same set
T_WIN = 70.0


def render(P, kx, ky, om, A, ph):
    psi = (kx[None] * P[:, 0:1] + ky[None] * P[:, 1:2]
           - om[None] * P[:, 2:3] + ph[None])
    return (A[None] * np.cos(psi)).sum(1)


def omega_of(kx, ky, depth):
    k = np.hypot(kx, ky)
    return np.sqrt(9.81 * k * np.tanh(k * depth))


def main():
    res = {}
    for si in [305, 341]:                     # two seas, one report each
        S = build_sea_crosstp(si)
        A = S['m_A']
        m_eff = (A ** 2).sum() ** 2 / (A ** 4).sum()
        k4_phase = -1.5 / m_eff
        # noise dilution: z = wave + noise, kurtosis scales by (var_w/var_z)^2
        var_w = (A ** 2 / 2).sum()
        var_z = var_w + S['sigma0'] ** 2
        dil = (var_w / var_z) ** 2

        # frequency lattice spacing (per lobe) vs window resolution
        om = omega_of(S['m_kx'], S['m_ky'], DEPTH)
        om1 = np.sort(om[:100]); om2 = np.sort(om[100:])
        sp1 = np.median(np.diff(om1)); sp2 = np.median(np.diff(om2))
        res_om = 2 * np.pi / T_WIN

        # ---- empirical: fresh realizations of the SAME theta on S['P']
        rng = np.random.default_rng(123456 + si)
        P = S['P']
        n = len(P)
        Hs, r2, Tp2 = S['Hs'], S['r2'], S['Tp2']
        d1 = np.rad2deg(S['d1']); d2 = np.rad2deg(S['d2'])
        Z = np.empty((R, n))
        for r in range(R):
            c = list(jonswap_components(Hs=Hs * np.sqrt(1 - r2), Tp=7.0,
                                        n_components=100, depth=DEPTH,
                                        gamma=3.3, direction_deg=d1,
                                        direction_spread_deg=SPREAD_DEG,
                                        seed=int(rng.integers(1 << 30))))
            c += list(jonswap_components(Hs=Hs * np.sqrt(r2), Tp=Tp2,
                                         n_components=100, depth=DEPTH,
                                         gamma=3.3, direction_deg=d2,
                                         direction_spread_deg=SPREAD_DEG,
                                         seed=int(rng.integers(1 << 30))))
            kx = np.array([x.kx for x in c]); ky = np.array([x.ky for x in c])
            Am = np.array([x.A for x in c]); ph = np.array([x.phi for x in c])
            omc = omega_of(kx, ky, DEPTH)
            Z[r] = render(P, kx, ky, omc, Am, ph)
        Z += rng.normal(0, S['sigma0'], Z.shape)

        C = (Z.T @ Z) / R
        ev, V = np.linalg.eigh(C)
        keep = ev > ev.max() * 1e-6
        U = (Z @ V[:, keep]) / np.sqrt(ev[keep])[None]   # whitened coords
        # Gaussian control with identical second moments = N(0, I) exactly
        # (whitening used the empirical covariance) -> compare U's cumulants to 0
        g3 = float(np.mean(U ** 3))                       # pooled skewness
        g4 = float(np.mean(U ** 4) - 3.0)                 # pooled excess kurt
        # per-DOF spread (sanity: sampling noise at R=3000 is ~ sqrt(24/R)=0.09)
        k4_dof = np.mean(U ** 4, axis=0) - 3.0
        n_dof = int(keep.sum())
        # leading-order per-window non-Gaussian KL (order of magnitude)
        kl_win = n_dof * (g3 ** 2 / 12.0 + g4 ** 2 / 48.0)
        kl_win_analytic = n_dof * (k4_phase * dil) ** 2 / 48.0
        res[si] = dict(m_eff=float(m_eff), k4_phase=float(k4_phase),
                       noise_dilution=float(dil),
                       freq_spacing_lobe1=float(sp1), freq_spacing_lobe2=float(sp2),
                       window_resolution=float(res_om),
                       n_dof=n_dof, g3_pooled=g3, g4_pooled=g4,
                       k4_dof_mean=float(k4_dof.mean()),
                       k4_dof_se=float(k4_dof.std() / np.sqrt(n_dof)),
                       kl_per_window_empirical=float(kl_win),
                       kl_per_window_analytic=float(kl_win_analytic))
        print(f"si{si}: M_eff {m_eff:.0f} -> kappa4(phase) {k4_phase:+.4f} "
              f"(dilution {dil:.2f})")
        print(f"       freq spacing {sp1:.4f}/{sp2:.4f} rad/s vs window res "
              f"{res_om:.4f} rad/s -> lattice "
              f"{'UNRESOLVABLE' if max(sp1, sp2) < res_om else 'partly resolvable'}")
        print(f"       empirical pooled g3 {g3:+.4f} g4 {g4:+.4f} "
              f"(per-DOF k4 mean {k4_dof.mean():+.4f} +- {k4_dof.std()/np.sqrt(n_dof):.4f})")
        print(f"       leading-order non-Gaussian KL per window: "
              f"empirical ~{kl_win:.4f} nats | analytic ~{kl_win_analytic:.4f} nats "
              f"(needed to flip argmax->true: ~3 nats)", flush=True)
    json.dump(res, open(OUT, 'w'), indent=1)
    print(f"saved -> {OUT}")


if __name__ == '__main__':
    main()
