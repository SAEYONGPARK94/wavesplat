"""Experiment B2 (deeper, tenacious re-test of Option ①).

Two weaknesses in the B (nowcast) kill:
 (1) it was NOWCAST only; B0's headroom was LEAD-AVERAGED (forecast), and the net's
     advantage was a forecast effect. → score the FORECAST band (lead 10-20s).
 (2) the γ-splat oracle may be a weak oracle. → add a STRONG method-agnostic ceiling:
     true (kx,ky) → optimal LS-fit of A,φ (tnse.fit_amplitudes_phases) → render.

Arms per HOS scene (known components), scored /gt_std at nowcast (lead 0) AND far (10-20s):
  hand      mmse_bank beamform prior
  net       mmse_bank learned prior
  oracle_g  mmse_bank true-spectrum γ prior (the B0 oracle)
  oracle_ls true-(kx,ky) LS A,φ reconstruction  ← STRONG ceiling
  cond      cond-diffusion set-predictor (competitive method)

Resurrects ① if, at FORECAST, cond does NOT reach oracle_ls AND net captures a meaningful
fraction of the (hand→oracle_ls) gap. Kills it if cond ≈ oracle_ls at forecast too.
"""
import argparse
import json
import time

import numpy as np
import torch

import d3_compare_gifs as cg
import d3_beamform_cond_diffusion as cd
from d3_sa_nomp import mmse_bank_estimate
from d3_prior_net_eval import get_net, net_gamma_mmse
from wavesplat.model.tnse import fit_amplitudes_phases, render_wave_field
from run_mechanism import hos_scene_known

NET_CKPT = cg.NET_CKPT


def per_lead_from_fq(fq, sc):
    npx, nl = sc['npx'], len(sc['leads'])
    maps = fq[:npx * nl].reshape(nl, *sc['Xq'].shape)
    gt = sc['maps_gt']
    sg = float(np.std(gt[np.isfinite(gt)]))
    out = []
    for li in range(nl):
        m = np.isfinite(gt[li]) & np.isfinite(maps[li])
        out.append(np.sqrt(np.mean((maps[li][m] - gt[li][m]) ** 2)) / sg if m.sum() >= 8 else np.nan)
    return np.array(out)


def per_lead_oracle_ls(sc, lam=0.5):
    """Strong ceiling: true (kx,ky) → LS-fit A,φ from obs → render forward per lead."""
    P, z = sc['P'], sc['z']
    kx, ky = sc['Ctrue'][:, 0], sc['Ctrue'][:, 1]
    obs = np.column_stack([P[:, 0], P[:, 1], P[:, 2], z])
    A, phi = fit_amplitudes_phases(kx, ky, obs, sc['depth'], lambda_reg=lam)
    t0 = float(P[:, 2].max())
    gt = sc['maps_gt']; sg = float(np.std(gt[np.isfinite(gt)]))
    out = []
    for li, L in enumerate(sc['leads']):
        eta = render_wave_field(kx, ky, A, phi, sc['depth'], sc['gxq'], sc['gyq'], t0 + float(L))
        m = np.isfinite(gt[li]) & np.isfinite(eta)
        out.append(np.sqrt(np.mean((eta[m] - gt[li][m]) ** 2)) / sg if m.sum() >= 8 else np.nan)
    return np.array(out)


def per_lead_cond(sc, model, stats, abar, betas, nsamp=4):
    P, z = sc['P'], sc['z']
    tq = float(P[:, 2].max())
    Ps = P.copy(); Ps[:, 2] -= tq
    Bmap = cd._bmap_from_obs(Ps, z)
    X = cd.sample(model, Bmap, abar, betas, n=nsamp)
    Pt = torch.tensor(Ps, device=cd.DEVICE, dtype=torch.float32)
    zt = torch.tensor(z, device=cd.DEVICE, dtype=torch.float32)
    polished = [cd.polish_atoms(X[b], stats, Pt, zt, sc['depth'], iters=80) for b in range(nsamp)]
    gt = sc['maps_gt']; sg = float(np.std(gt[np.isfinite(gt)]))
    out = []
    for li, L in enumerate(sc['leads']):
        target = {'gx': sc['gxq'], 'gy': sc['gyq'], 't': float(L), 'gt': gt[li], 'Hs': sc['Hs']}
        fs = []
        for xb in polished:
            f, _ = cd.atoms_to_field(xb, stats, Ps, z, sc['depth'], target)
            if f is not None:
                fs.append(f)
        if fs:
            bma = np.mean(fs, axis=0)
            m = np.isfinite(gt[li]) & np.isfinite(bma)
            out.append(np.sqrt(np.mean((bma[m] - gt[li][m]) ** 2)) / sg if m.sum() >= 8 else np.nan)
        else:
            out.append(np.nan)
    return np.array(out)


def band(pl, leads, lo, hi):
    m = (leads >= lo) & (leads <= hi)
    return float(np.nanmean(pl[m]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=6)
    ap.add_argument('--Hs', type=float, default=2.0)
    ap.add_argument('--Tp', type=float, default=8.0)
    ap.add_argument('--nc', type=int, default=60)
    a = ap.parse_args()

    net = get_net(NET_CKPT)
    stats = torch.load(cd.PRIOR_CKPT, map_location=cd.DEVICE)['stats']
    betas = cd.cosine_betas(cd.T).to(cd.DEVICE)
    abar = torch.cumprod(1 - betas, 0)
    cmodel = cd.CondSetDenoiser().to(cd.DEVICE)
    cmodel.load_state_dict(torch.load('results/direction3/beamform_cond_diffusion.pt',
                                      map_location=cd.DEVICE))
    cmodel.eval()

    ARMS = ('hand', 'net', 'oracle_g', 'oracle_ls', 'cond')
    rows = {b: {k: [] for k in ARMS} for b in ('now', 'far')}
    for r in range(a.reps):
        t0 = time.time()
        sc = hos_scene_known(a.Hs, a.Tp, r, nc=a.nc)
        s0 = 0.05 * sc['Hs']; P, z, Pq = sc['P'], sc['z'], sc['Pq']
        leads = sc['leads']
        pl = {
            'hand': per_lead_from_fq(mmse_bank_estimate(P, z, sc['depth'], Pq, sigma0=s0, kp_hint=sc['kp'], seed=r), sc),
            'net': per_lead_from_fq(net_gamma_mmse(net, P, z, sc['depth'], Pq, sigma0=s0, seed=r, kp_hint=sc['kp']), sc),
            'oracle_g': per_lead_from_fq(net_gamma_mmse(net, P, z, sc['depth'], Pq, sigma0=s0, seed=r, kp_hint=sc['kp'], oracle_components=sc['Ctrue']), sc),
            'oracle_ls': per_lead_oracle_ls(sc),
            'cond': per_lead_cond(sc, cmodel, stats, abar, betas),
        }
        for k in ARMS:
            rows['now'][k].append(band(pl[k], leads, 0, 0))
            rows['far'][k].append(band(pl[k], leads, 10, 20))
        print(f"[{r}/{a.reps}] now: " + " ".join(f"{k[:4]} {rows['now'][k][-1]:.2f}" for k in ARMS)
              + " | far: " + " ".join(f"{k[:4]} {rows['far'][k][-1]:.2f}" for k in ARMS)
              + f" | {time.time()-t0:.0f}s", flush=True)

    out = {b: {k: float(np.nanmean(rows[b][k])) for k in ARMS} for b in ('now', 'far')}
    json.dump(out, open('tmp/G1_waveprior_power/expB2.json', 'w'), indent=1)
    print("\n=== EXP B2 (nRMSE/gt_std) ===")
    for b in ('now', 'far'):
        print(f"  [{b}] " + "  ".join(f"{k} {out[b][k]:.3f}" for k in ARMS))
    # verdict at forecast
    f = out['far']
    gap_ceiling = f['hand'] - f['oracle_ls']
    net_capture = (f['hand'] - f['net']) / gap_ceiling if gap_ceiling > 1e-6 else float('nan')
    cond_gap = f['cond'] - f['oracle_ls']
    print(f"\n  FORECAST: hand→oracle_ls gap {gap_ceiling:+.3f}; net captures {net_capture*100:.0f}%; "
          f"cond−oracle_ls {cond_gap:+.3f}")
    if cond_gap > 0.03:
        print("  ⟹ cond does NOT reach the strong ceiling at forecast — real headroom survives ⟹ ① MAY LIVE (forecast framing)")
    else:
        print("  ⟹ cond reaches the strong ceiling at forecast too ⟹ ① stays dead")


if __name__ == '__main__':
    main()
