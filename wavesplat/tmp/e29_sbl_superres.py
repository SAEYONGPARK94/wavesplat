"""Does z DETERMINE a sharp gamma? — the test my PSF-blur argument could not answer.

My earlier closure argument blurred the TRUE spectrum to the beamforming PSF width
and found only 17.9% of the realization gain survives => "super-resolution needed
=> closed". That argument is WRONG as stated: the beamform power map |Phi^H z|^2 is a
QUADRATIC, phase-discarding statistic, whereas the measurement z = Phi a + n is a
LINEAR map that retains phase. The Rayleigh limit binds the PERIODOGRAM, not the
INFORMATION. MUSIC/ESPRIT/RELAX/SBL/compressed-sensing all live in that gap, and this
project's own strongest classical estimator (mmse_bank_estimate) already runs
"val-gated SBL-EM refinement of the prior".

So the honest question is: starting from data ONLY (no oracle), how sharp a gamma can
sparse Bayesian learning recover from z, and what does that gamma score?

RUNGS (same arena / same solver / fixed smult=3 as the ladder):
  cur        incumbent (learned CNN)                     +0.4032   [ladder]
  ens_blind  hand ensemble prior, zero oracle            +0.3914   [ladder]
  bf         raw beamform-power prior (classical)        ~+0.25    [e28 'cl']
  sbl_bf     SBL-EM started from the beamform prior      <- THIS
  sbl_ens    SBL-EM started from the ensemble prior      <- THIS
  real_free  TRUE realization spectrum (sharp, oracle)   +0.5856   [ladder]

If sbl_* climbs toward real_free, z DOES determine a sharp gamma => the net path is
OPEN and the incumbent is simply not doing the inference. If sbl_* stalls at ~cur,
z genuinely does not determine it => closer to closed (and then the PSF story is
about information, not just about the periodogram).

SBL-EM (Tipping), dual form since n=400 << M=7616:
  K       = Phi Gam Phi^T + s2 I                (n x n)
  mu      = Gam Phi^T K^-1 z
  Sig_ii  = gam_i - gam_i^2 * phi_i^T K^-1 phi_i
  gam_i  <- mu_i^2 + Sig_ii          (cos/sin of an atom share gam -> average)
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from e29_gennet_C import (build_sea_c, deposit_grid, bank_atoms, design_t,
                          corr, DEVICE, DEPTH, _MASK, NGRID, KREL)
from e29_gate0_ladder import universal_radial, ensemble_grid, SMULT
from d3_sa_nomp import beamform_power
from d18_relax_estimator import beamform_top
from d3_sa_achievability_gap import hashabs
from e29_gennet_C import REGIMES

OUT = 'results/direction3/e29_sbl_superres.json'
N_EM = 12


def _mats(S):
    kxb, kyb, om = bank_atoms(S['kp'])
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=DEVICE)
    Phi = design_t(t(kxb), t(kyb), t(om), t(S['P']))
    Phi_q = design_t(t(kxb), t(kyb), t(om), t(S['Pq']))
    return Phi, Phi_q, t(S['z'])


def score(S, Phi, Phi_q, zt, g_atoms):
    """g_atoms: (3808,) per-atom prior weight -> corr at held-out pixels."""
    g = g_atoms / max(float(g_atoms.sum()), 1e-30)
    gam = torch.cat([g, g]) * float(np.var(S['z']))
    s2 = (SMULT * S['sigma0']) ** 2
    K = (Phi * gam[None]) @ Phi.T + s2 * torch.eye(len(S['z']), device=DEVICE)
    w = Phi.T @ torch.linalg.solve(K, zt)
    return float(corr(((Phi_q * gam[None]) @ w).cpu().numpy(), S['gt_u']))


def sbl_em(S, Phi, zt, g0, n_em=N_EM):
    """SBL-EM on the shared-per-atom gamma. Returns gamma trace + sharpness."""
    n = len(S['z'])
    M = Phi.shape[1] // 2
    s2 = (SMULT * S['sigma0']) ** 2
    vz = float(np.var(S['z']))
    g = (g0 / max(float(g0.sum()), 1e-30) * vz).clone()      # per-atom, absolute
    eye = torch.eye(n, device=DEVICE)
    trace = []
    for it in range(n_em):
        gam = torch.cat([g, g])
        K = (Phi * gam[None]) @ Phi.T + s2 * eye
        Kinv = torch.linalg.inv(K)
        mu = gam * (Phi.T @ (Kinv @ zt))                      # (2M,)
        # phi_i^T Kinv phi_i for every column
        q = ((Phi.T @ Kinv) * Phi.T).sum(1)                   # (2M,)
        Sig = gam - gam ** 2 * q
        newg = mu ** 2 + Sig.clamp_min(0)
        g = 0.5 * (newg[:M] + newg[M:])                       # share cos/sin
        g = g.clamp_min(1e-12)
        trace.append(float((g / g.sum()).max()))              # sharpness proxy
    return g, trace


def main(n_seas=16):
    lad = json.load(open('results/direction3/e29_gate0_ladder.json'))['rows']
    kmag_u, en_u = universal_radial()
    R = REGIMES['spread']
    rows = []
    for si in range(n_seas):
        S = build_sea_c('spread', si)
        Phi, Phi_q, zt = _mats(S)
        nA = Phi.shape[1] // 2                                 # 3808 atoms
        t = lambda a: torch.tensor(a, dtype=torch.float32, device=DEVICE)

        # --- data-only starting priors -------------------------------------
        _, _, pw = beamform_power(S['P'].astype(np.float32),
                                  S['z'].astype(np.float32), DEPTH,
                                  KREL * S['kp'], NGRID)
        pw = np.asarray(pw.cpu() if hasattr(pw, 'cpu') else pw).reshape(NGRID, NGRID)
        g_bf = np.concatenate([pw[_MASK], np.zeros(int(np.asarray(_MASK).sum()))])
        g_bf = np.maximum(g_bf, 1e-12)
        rng = np.random.default_rng(7000 + 31 * si + hashabs('spread'))
        _ = rng.uniform(*R['hs'])
        mu_true = np.deg2rad(rng.uniform(0, 360))
        bx, by = beamform_top(S['P'].astype(np.float32), S['z'].astype(np.float32),
                              DEPTH, klim=3.5 * S['kp'], n=120)
        ens = ensemble_grid(kmag_u, en_u, np.arctan2(by, bx), S['kp'])
        g_ens = np.concatenate([ens[_MASK], np.zeros(int(np.asarray(_MASK).sum()))])
        g_ens = np.maximum(g_ens, 1e-12)
        # --- oracle reference ----------------------------------------------
        gf = deposit_grid(S['m_kx'], S['m_ky'], S['m_A'] ** 2 / 2, S['kp'])
        g_real = np.concatenate([gf[_MASK], np.zeros(int(np.asarray(_MASK).sum()))])

        s_bf = score(S, Phi, Phi_q, zt, t(g_bf))
        s_ens = score(S, Phi, Phi_q, zt, t(g_ens))
        s_real = score(S, Phi, Phi_q, zt, t(np.maximum(g_real, 1e-12)))
        gs_bf, tr_bf = sbl_em(S, Phi, zt, t(g_bf))
        gs_ens, tr_ens = sbl_em(S, Phi, zt, t(g_ens))
        s_sbl_bf = score(S, Phi, Phi_q, zt, gs_bf)
        s_sbl_ens = score(S, Phi, Phi_q, zt, gs_ens)
        # sharpness: participation ratio (lower = sharper)
        pr = lambda g: float((g.sum() ** 2 / (g ** 2).sum()).item())
        r = dict(si=si, cur=lad[si]['cur'], ens=s_ens, bf=s_bf,
                 sbl_bf=s_sbl_bf, sbl_ens=s_sbl_ens, real=s_real,
                 pr_bf=pr(t(g_bf)), pr_sbl=pr(gs_bf),
                 pr_real=pr(t(np.maximum(g_real, 1e-12))))
        rows.append(r)
        print(f"sea{si:2d} cur {r['cur']:+.3f} | bf {s_bf:+.3f} ens {s_ens:+.3f} "
              f"| SBL(bf) {s_sbl_bf:+.3f} SBL(ens) {s_sbl_ens:+.3f} "
              f"| real {s_real:+.3f} | PR bf {r['pr_bf']:.0f}->sbl {r['pr_sbl']:.0f} "
              f"(real {r['pr_real']:.0f})", flush=True)

    a = lambda k: np.array([r[k] for r in rows], float)
    cur = a('cur')
    print(f"\n=== SBL super-resolution from z (n={len(rows)}, smult={SMULT}) ===")
    for k, lab in [('bf', 'beamform prior (고전)'), ('ens', 'ensemble prior (손)'),
                   ('cur', 'incumbent (학습 CNN)'),
                   ('sbl_bf', 'SBL-EM from beamform  <- 데이터만'),
                   ('sbl_ens', 'SBL-EM from ensemble  <- 데이터만'),
                   ('real', 'TRUE realization (오라클)')]:
        v = a(k)
        print(f"  {lab:36s}: {v.mean():+.4f}   vs cur {(v-cur).mean():+.4f}  "
              f"(승 {int((v>cur).sum())}/{len(v)})")
    print(f"\n  sharpness (participation ratio, 낮을수록 뾰족):")
    print(f"    beamform prior {a('pr_bf').mean():8.0f} -> SBL {a('pr_sbl').mean():8.0f} "
          f"| TRUE {a('pr_real').mean():8.0f}")
    gap = (a('real') - cur).mean()
    got = (a('sbl_bf') - cur).mean()
    print(f"\n  ★SBL이 realization 간격의 {got/gap:.1%}를 회수 "
          f"(간격 {gap:+.4f}, SBL {got:+.4f})")
    json.dump(dict(rows=rows), open(OUT, 'w'), indent=1)
    print(f"saved -> {OUT}")


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 16)
