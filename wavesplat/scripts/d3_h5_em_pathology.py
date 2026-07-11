"""H5 -- the n-poison's fingerprint: WHICH part of the model pipeline kills
temporal extrapolation as observations grow?

H4-SEL measured, monotone in n: encounter rho rand@2k 0.65 -> @20k 0.16
(worst -0.56) while the spatial annulus stays ~0.93. Candidate mechanisms,
given the mmse_gpu code path (grid-select (pexp, smult) on held-out residual,
then gam-EM with val early stop):

  M1  grid selection drifts with n (small s2 / peaked prior at large n)
  M2  gam-EM inflates high-k atoms: out-of-band energy (56% of slope
      variance) aliases onto in-band atoms on dense near-crowded points;
      the EM reads that in-window texture as signal; omega phases of those
      atoms are wrong -> the future explodes (07-08 "long-lead amplitude
      blowup" family)
  M3  neither: the kernel solve itself (misspecification needs a noise
      floor that grows with model error, not with 1/n)

ARMS, factorial over n in {2000, 20000} x seas {0,1,2,4} (2 bad, 2 good):
  em       production path (reproduces H4)
  no_em    grid-selected (pexp, smult), NO gam-EM      -> isolates M2
  floor    no_em + fixed pexp=1, smult=10 (strong s2)  -> isolates M1/M3
Each cell records (pexp, smult, val_resid, gam high-k mass fraction) and
enc/ann rho. Random subsets, same scene + speckle across cells (paired).

READING: no_em restores @20k -> M2 (EM is the poison; production fix =
val-DOF-charged or capped EM). floor restores but no_em does not -> M1/M3
(fix = noise-floor scaling with n or misspecification-aware s2). Neither
restores -> deeper (bank/solve); escalate.
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

import d3_h3_fixedT_operating_domain as h3
from d3_h3_fixedT_operating_domain import gen_hits, ENC_T
from d3_h1_mask_destination import sea_dir, eta_xt
from d3_gpu_mmse import mmse_gpu, build_bank, design
from d3_door1b_field_bias import (R_MIN, R_MAX, NOISE, DEPTH, FOV_H)
from d3_g0b_selection_aware import DEV, DT, BAND

OUT = 'results/direction3/h5_em_pathology.json'
T_SEL = 0.0173
SEAS = [0, 1, 2, 4]
NS = [2000, 20000]


def main():
    h3.sea = sea_dir(180.0)
    print(f"=== H5 EM pathology  seas={SEAS} ns={NS} ===")
    rng = np.random.default_rng(47)   # H4's stream root for scene similarity
    rows = []
    for seed in SEAS:
        t0 = time.time()
        S, kp, scans = gen_hits(seed, 3.33, rng)
        P, zv = [], []
        for t, hp, r, pw in scans:
            m = pw > T_SEL
            P.append(np.column_stack([hp[m, :2], np.full(int(m.sum()), t)]))
            zv.append(hp[m, 2])
        P, zv = np.vstack(P), np.concatenate(zv)
        zv = zv + rng.normal(0, NOISE, len(zv))

        Qe = np.column_stack([np.zeros(len(ENC_T)), np.zeros(len(ENC_T)),
                              ENC_T])
        gt_e = eta_xt(Qe[:, 0], Qe[:, 1], Qe[:, 2], S)
        na = 400
        ra = np.sqrt(rng.uniform((R_MIN / 73.0) ** 2, 1.0, na)) * 73.0
        aa = rng.uniform(-np.deg2rad(FOV_H), np.deg2rad(FOV_H), na)
        Qa = np.column_stack([ra * np.cos(aa), ra * np.sin(aa), np.zeros(na)])
        gt_a = eta_xt(Qa[:, 0], Qa[:, 1], Qa[:, 2], S)

        kxb, kyb, om = build_bank(kp, BAND, kp / 10.0, DEPTH, DEV,
                                  torch.float32)
        kxb64, kyb64, om64 = kxb.to(DT), kyb.to(DT), om.to(DT)
        Ce = design(torch.tensor(Qe, device=DEV, dtype=DT), kxb64, kyb64, om64)
        Ca = design(torch.tensor(Qa, device=DEV, dtype=DT), kxb64, kyb64, om64)
        kmag = torch.hypot(kxb64, kyb64)
        hik = (torch.cat([kmag, kmag]) > 1.5 * kp)

        for n in NS:
            idx = rng.choice(len(P), min(n, len(P)), replace=False)
            Ps, zs = P[idx], zv[idx]
            t32 = torch.tensor(Ps, device=DEV, dtype=torch.float32)
            z32 = torch.tensor(zs, device=DEV, dtype=torch.float32)
            for arm, kw in (('em', {}), ('no_em', {'no_em': True}),
                            ('floor', {'no_em': True, 'pexp': 1.0,
                                       'smult': 10.0})):
                _, mdl = mmse_gpu(t32, z32, DEPTH, t32[:64], NOISE, kp,
                                  seed=seed, return_model=True, **kw)
                gam, s2 = mdl['gam'].to(DT), float(mdl['s2'])
                Phi = design(torch.tensor(Ps, device=DEV, dtype=DT),
                             kxb64, kyb64, om64)
                A0 = Phi.T @ Phi / s2 + torch.diag(1.0 / gam)
                a = torch.linalg.solve(A0, Phi.T @ torch.tensor(
                    zs, device=DEV, dtype=DT) / s2)

                def rho(C, gt):
                    f = (C @ a).cpu().numpy()
                    return float(np.corrcoef(f, gt)[0, 1]) \
                        if np.std(f) > 1e-12 else 0.0

                hkf = float(gam[hik].sum() / gam.sum())
                rows.append({'seed': seed, 'n': int(n), 'arm': arm,
                             'pexp': mdl['pexp'], 'smult': mdl['smult'],
                             's2': s2, 'hik_frac': hkf,
                             'enc': rho(Ce, gt_e), 'ann': rho(Ca, gt_a)})
                r = rows[-1]
                print(f"seed {seed} n {n:6d} {arm:6s} pexp {r['pexp']:.1f} "
                      f"smult {r['smult']:4.1f} hik {hkf:.3f} | "
                      f"enc {r['enc']:+.2f} ann {r['ann']:+.2f}", flush=True)
        print(f"  sea {seed} done [{time.time()-t0:.0f}s]")

    print("\n" + "=" * 68)
    for arm in ('em', 'no_em', 'floor'):
        for n in NS:
            sel = [r for r in rows if r['arm'] == arm and r['n'] == n]
            e = np.mean([r['enc'] for r in sel])
            emin = min(r['enc'] for r in sel)
            hk = np.mean([r['hik_frac'] for r in sel])
            print(f"{arm:6s} n {n:6d}: enc mean {e:+.3f} min {emin:+.2f}  "
                  f"hik_frac {hk:.3f}")
    json.dump(rows, open(OUT, 'w'), indent=2)
    print(f"wrote {OUT}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
