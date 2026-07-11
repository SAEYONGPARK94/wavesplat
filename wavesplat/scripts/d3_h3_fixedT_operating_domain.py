"""H3-OD -- the operating domain under the sensor's ACTUAL law: fixed-T sweep.

Today's mcar contrast (H1) showed the task family is killed not by the
600-point budget but by WHERE threshold top-k puts it (uniform spray: enc rho
0.86, slope 0.84; top-k: 0.10, 0.52). But top-k@50 is the EPISODE CONVENTION
(an AGC/CFAR reading of the receiver, raydrop_threshold.py), while F2's
verdict is that the Falcon is a FIXED-threshold detector (manual p.13 "250 m
@ 10% reflectivity", p.105 peak-over-threshold language). G0b flagged the
contradiction; this probe measures it.

Design. H1 scene machinery (head seas, sigma_eta ~ 1 m, occluded first hits,
speckle X ~ Exp(1), radiometry A(s,r) = Fresnel(theta)cos(theta)/(r/125)^2).
Selection = A*X > T at FIXED T, swept over a log grid that brackets:
  T_k50   the per-scan top-50-equivalent threshold (median of G0b-style Ts),
  T_m50   manual-calibrated: 50% detection of the 10%-Lambertian target at
          250 m under this A normalisation: A_m = 0.10/(250/125)^2 = 0.025,
          T = A_m ln 2 = 0.0173,
  T_m90   90% detection: T = A_m * ln(1/0.9) = 0.00263.
No count cap: n(T) is a MEASUREMENT. For solver tractability obs are
subsampled to <= OBS_CAP uniformly at random (random-within-selection keeps
the spatial distribution -- the lever the mcar contrast identified; both the
raw and used counts are reported).

Scored (naive estimator only -- this is an operating-domain probe, no EP):
full-fan elevation (1200 q, 15-200 m), annulus elevation, in-band slope,
encounter 0-3 / 3-10 s; nRMSE (B1-capped per functional, #51) and rho.

READING, fixed in advance: a functional is LIVE at rho >= 0.7. If fixed-T at
manual calibration still concentrates returns so the fan/enc stay dead, the
20 m information radius is the SENSOR's own geometry and the operating-domain
conclusion survives sensor-faithful selection. If they revive, the killer was
the 50-point convention, and every closure measured at that operating point
is conditional on a protocol the sensor does not have. Either way the number
goes in the map; no verdict about learning is drawn here.
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

import d3_g0b_selection_aware as g0b
from d3_g0b_selection_aware import A_of, DEV, DT, BAND
from d3_gpu_mmse import mmse_gpu, build_bank, design
from d3_s2_mask_information import grad_design
from d3_door1b_field_bias import (sea, eta_dr, ray_grid, occluded_hit,
                                  R_MIN, R_MAX, SCAN_T, NOISE, DEPTH, FOV_H)
from d3_h1_mask_destination import sea_dir, eta_xt

OUT = 'results/direction3/h3_fixedT_operating_domain.json'
A_M = 0.10 / (250.0 / 125.0) ** 2
T_GRID = [0.3, 0.1, 0.03, 0.0173, 0.01, 0.003, 0.00263, 0.001]
T_MARK = {0.0173: 'T_m50', 0.00263: 'T_m90'}
OBS_CAP = 5000
N_SEAS = 8
ENC_T = np.arange(0.25, 10.001, 0.25)


def gen_hits(seed, amp_scale, rng):
    """all first hits + slopes + speckle, selection applied later per T."""
    S = sea(seed, amp_scale)
    kp = float(np.hypot(S[0], S[1])[np.argmax(S[3])])
    o, d, _ = ray_grid()
    per_scan = []
    for t in SCAN_T:
        hm, hp = occluded_hit(o, d, t, S)
        idx = np.where(hm)[0]
        r = np.hypot(hp[idx, 0], hp[idx, 1])
        s = eta_dr(hp[idx, 0], hp[idx, 1], t, S)
        A = A_of(s, r)
        X = rng.exponential(1.0, len(idx))
        per_scan.append((t, hp[idx], r, A * X))
    return S, kp, per_scan


def run_sea(seed, rng):
    S, kp, scans = gen_hits(seed, 3.33, rng)

    nq = 1200
    rq = np.sqrt(rng.uniform((R_MIN / R_MAX) ** 2, 1.0, nq)) * R_MAX
    aq = rng.uniform(-np.deg2rad(FOV_H), np.deg2rad(FOV_H), nq)
    Qf = np.column_stack([rq * np.cos(aq), rq * np.sin(aq), np.zeros(nq)])
    gt_f = eta_xt(Qf[:, 0], Qf[:, 1], Qf[:, 2], S)
    sig_f = float(np.std(gt_f))
    kt = np.hypot(S[0], S[1])
    ib = (kt >= BAND[0] * kp) & (kt <= BAND[1] * kp)
    S_ib = tuple(c[ib] for c in S)
    Qe = np.column_stack([np.zeros(len(ENC_T)), np.zeros(len(ENC_T)), ENC_T])
    gt_e = eta_xt(Qe[:, 0], Qe[:, 1], Qe[:, 2], S)

    kxb, kyb, om = build_bank(kp, BAND, kp / 10.0, DEPTH, DEV, torch.float32)
    kxb, kyb, om = kxb.to(DT), kyb.to(DT), om.to(DT)
    Cf = design(torch.tensor(Qf, device=DEV, dtype=DT), kxb, kyb, om)
    Ce = design(torch.tensor(Qe, device=DEV, dtype=DT), kxb, kyb, om)

    out = []
    for T in T_GRID:
        P, zv, rr = [], [], []
        n_raw = 0
        for t, hp, r, pw in scans:
            m = pw > T
            n_raw += int(m.sum())
            P.append(np.column_stack([hp[m, :2], np.full(m.sum(), t)]))
            zv.append(hp[m, 2])
            rr.append(r[m])
        P, zv, rr = np.vstack(P), np.concatenate(zv), np.concatenate(rr)
        if len(P) < 60:
            out.append({'T': T, 'n_raw': n_raw, 'n_used': len(P),
                        'dead': 'too_few'})
            continue
        if len(P) > OBS_CAP:
            k = rng.choice(len(P), OBS_CAP, replace=False)
            P, zv, rr = P[k], zv[k], rr[k]
        zv = zv + rng.normal(0, NOISE, len(zv))
        r90 = float(np.quantile(rr, 0.90))

        # annulus queries for this T's informed region
        na = 400
        ra = np.sqrt(rng.uniform((R_MIN / max(r90, R_MIN + 1)) ** 2, 1.0, na)
                     ) * max(r90, R_MIN + 1)
        aa = rng.uniform(-np.deg2rad(FOV_H), np.deg2rad(FOV_H), na)
        Qa = np.column_stack([ra * np.cos(aa), ra * np.sin(aa), np.zeros(na)])
        gt_a = eta_xt(Qa[:, 0], Qa[:, 1], Qa[:, 2], S)
        gs_a = eta_dr(Qa[:, 0], Qa[:, 1], 0.0, S_ib)

        tP32 = torch.tensor(P, device=DEV, dtype=torch.float32)
        tz32 = torch.tensor(zv, device=DEV, dtype=torch.float32)
        _, mdl = mmse_gpu(tP32, tz32, DEPTH, tP32[:64], NOISE, kp, seed=seed,
                          return_model=True)
        gam, s2 = mdl['gam'].to(DT), float(mdl['s2'])
        Phi = design(torch.tensor(P, device=DEV, dtype=DT), kxb, kyb, om)
        A0 = Phi.T @ Phi / s2 + torch.diag(1.0 / gam)
        a = torch.linalg.solve(A0, Phi.T @ torch.tensor(
            zv, device=DEV, dtype=DT) / s2)
        Ca = design(torch.tensor(Qa, device=DEV, dtype=DT), kxb, kyb, om)
        Cs = grad_design(torch.tensor(Qa, device=DEV, dtype=DT), kxb, kyb, om)

        zsd = float(np.std(zv))

        def sc(C, gt, sig):
            f = (C @ a).cpu().numpy()
            beta = min(1.0, zsd / max(np.std(f), 1e-12))
            e = float(np.sqrt(np.mean((beta * f - gt) ** 2))) / max(sig, 1e-9)
            rho = float(np.corrcoef(f, gt)[0, 1]) if np.std(f) > 1e-12 else 0.0
            return e, rho

        ef, rf = sc(Cf, gt_f, sig_f)
        ea, ra_ = sc(Ca, gt_a, float(np.std(gt_a)))
        es, rs = sc(Cs, gs_a, float(np.std(gs_a)))
        fe = (Ce @ a).cpu().numpy()
        beta_e = min(1.0, zsd / max(np.std(fe), 1e-12))
        m03 = ENC_T <= 3.0
        e0 = float(np.sqrt(np.mean((beta_e * fe[m03] - gt_e[m03]) ** 2))) / sig_f
        e1 = float(np.sqrt(np.mean((beta_e * fe[~m03] - gt_e[~m03]) ** 2))) / sig_f
        re = float(np.corrcoef(fe, gt_e)[0, 1])
        out.append({'T': T, 'mark': T_MARK.get(T, ''), 'n_raw': n_raw,
                    'n_used': len(P), 'r90': r90,
                    'fan': [ef, rf], 'ann': [ea, ra_], 'slp': [es, rs],
                    'enc03': e0, 'enc310': e1, 'rho_enc': re})
    return {'seed': seed, 'kp': kp, 'rows': out}


def main():
    g0b_sea_patch = sea_dir(180.0)
    import d3_door1b_field_bias as db
    global sea
    sea = g0b_sea_patch                       # head seas, H1 continuity

    print(f"=== H3-OD fixed-T operating domain  seas={N_SEAS} ===")
    print(f"A_manual = {A_M:.4f}  T_m50 = {A_M*np.log(2):.4f}  "
          f"T_m90 = {A_M*np.log(1/0.9):.5f}")
    rng = np.random.default_rng(31)
    seas = []
    for s in range(N_SEAS):
        t0 = time.time()
        r = run_sea(s, rng)
        seas.append(r)
        print(f"sea {s} [{time.time()-t0:5.1f}s]")
        for w in r['rows']:
            if 'dead' in w:
                print(f"   T {w['T']:8.4f} n_raw {w['n_raw']:6d} DEAD")
            else:
                print(f"   T {w['T']:8.4f}{w['mark']:>6s} n_raw {w['n_raw']:6d}"
                      f" used {w['n_used']:5d} r90 {w['r90']:6.1f} | "
                      f"fan {w['fan'][0]:.3f}/ρ{w['fan'][1]:+.2f} "
                      f"ann {w['ann'][0]:.3f}/ρ{w['ann'][1]:+.2f} "
                      f"slp {w['slp'][0]:.3f}/ρ{w['slp'][1]:+.2f} "
                      f"encρ {w['rho_enc']:+.2f}", flush=True)

    print("\n" + "=" * 76)
    agg = {}
    for T in T_GRID:
        rows = [w for r in seas for w in r['rows']
                if w['T'] == T and 'dead' not in w]
        if not rows:
            agg[str(T)] = 'dead'
            print(f"T {T:8.4f} : dead in all seas")
            continue
        f = np.mean([w['fan'][1] for w in rows])
        a_ = np.mean([w['ann'][1] for w in rows])
        s_ = np.mean([w['slp'][1] for w in rows])
        e = np.mean([w['rho_enc'] for w in rows])
        n = np.mean([w['n_raw'] for w in rows])
        r9 = np.mean([w['r90'] for w in rows])
        agg[str(T)] = {'rho_fan': f, 'rho_ann': a_, 'rho_slp': s_,
                       'rho_enc': e, 'n_raw': n, 'r90': r9, 'n_sea': len(rows)}
        print(f"T {T:8.4f}{T_MARK.get(T,''):>6s} n_raw {n:8.0f} r90 {r9:6.1f}"
              f" | rho fan {f:+.3f} ann {a_:+.3f} slp {s_:+.3f} enc {e:+.3f}")
    os.makedirs('results/direction3', exist_ok=True)
    json.dump({'seas': seas, 'agg': agg, 'T_grid': T_GRID,
               'A_manual': A_M, 'obs_cap': OBS_CAP},
              open(OUT, 'w'), indent=2)
    print(f"wrote {OUT}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
