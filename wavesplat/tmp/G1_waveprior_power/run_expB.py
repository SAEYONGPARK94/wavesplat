"""Experiment B (Option ① make-or-break) — is the 67% oracle headroom a UNIVERSAL
information gap, or just mmse_bank's weakness?

On the SAME HOS scenes as B0 (known components), score four arms at nowcast (lead 0),
same /gt_std normalizer:
  hand    = mmse_bank, beamform-power prior
  net     = mmse_bank, learned prior (prior_net_v3)
  oracle  = mmse_bank, TRUE-spectrum prior
  cond    = a COMPETITIVE learned set-predictor (beamform_cond_diffusion + polish),
            a *different, stronger* solver

Test: does cond reach the mmse-oracle level?
  cond ≲ oracle  ⟹ a stronger method closes the gap ⟹ 67% is mmse_bank-specific ⟹ ① weakens
  cond ≈ net (≫ oracle) ⟹ even a strong learned method leaves the headroom ⟹ universal gap ⟹ ① lives
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
from run_mechanism import hos_scene_known  # reuses the known-component HOS scene

NET_CKPT = cg.NET_CKPT


def nowcast_nrmse(fq, sc):
    """nRMSE/gt_std at lead 0 (nowcast)."""
    npx, nl = sc['npx'], len(sc['leads'])
    maps = fq[:npx * nl].reshape(nl, *sc['Xq'].shape)
    gt = sc['maps_gt']
    sg = float(np.std(gt[np.isfinite(gt)]))
    m = np.isfinite(gt[0]) & np.isfinite(maps[0])
    return float(np.sqrt(np.mean((maps[0][m] - gt[0][m]) ** 2)) / sg)


def cond_nowcast(sc, model, stats, abar, betas, nsamp=4):
    """cond-diffusion+polish reconstruction at nowcast (lead 0), /gt_std."""
    P, z = sc['P'], sc['z']
    # obs times relative to the nowcast instant (latest scan == query time = lead 0)
    tq = P[:, 2].max()
    Ps = P.copy(); Ps[:, 2] -= tq
    Bmap = cd._bmap_from_obs(Ps, z)
    X = cd.sample(model, Bmap, abar, betas, n=nsamp)
    gt0 = sc['maps_gt'][0]
    sg = float(np.std(sc['maps_gt'][np.isfinite(sc['maps_gt'])]))
    target = {'gx': sc['gxq'], 'gy': sc['gyq'], 't': 0.0, 'gt': gt0, 'Hs': sc['Hs']}
    Pt = torch.tensor(Ps, device=cd.DEVICE, dtype=torch.float32)
    zt = torch.tensor(z, device=cd.DEVICE, dtype=torch.float32)
    fs = []
    for b in range(nsamp):
        xb = cd.polish_atoms(X[b], stats, Pt, zt, sc['depth'], iters=80)
        f, _ = cd.atoms_to_field(xb, stats, Ps, z, sc['depth'], target)
        if f is not None:
            fs.append(f)
    if not fs:
        return 1.0
    bma = np.mean(fs, axis=0)
    return float(np.sqrt(np.mean((bma - gt0) ** 2)) / sg)


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

    rows = []
    for r in range(a.reps):
        t0 = time.time()
        sc = hos_scene_known(a.Hs, a.Tp, r, nc=a.nc)  # returns gxq, gyq for cond render
        s0 = 0.05 * sc['Hs']
        P, z, Pq = sc['P'], sc['z'], sc['Pq']
        fq_hand = mmse_bank_estimate(P, z, sc['depth'], Pq, sigma0=s0, kp_hint=sc['kp'], seed=r)
        fq_net = net_gamma_mmse(net, P, z, sc['depth'], Pq, sigma0=s0, seed=r, kp_hint=sc['kp'])
        fq_orac = net_gamma_mmse(net, P, z, sc['depth'], Pq, sigma0=s0, seed=r,
                                 kp_hint=sc['kp'], oracle_components=sc['Ctrue'])
        res = {'hand': nowcast_nrmse(fq_hand, sc), 'net': nowcast_nrmse(fq_net, sc),
               'oracle': nowcast_nrmse(fq_orac, sc)}
        try:
            res['cond'] = cond_nowcast(sc, cmodel, stats, abar, betas)
        except Exception as e:
            res['cond'] = None
            print(f"  cond arm failed seed {r}: {type(e).__name__}: {e}", flush=True)
        rows.append(res)
        print(f"[{r}/{a.reps}] hand {res['hand']:.3f} net {res['net']:.3f} "
              f"oracle {res['oracle']:.3f} cond {res['cond'] if res['cond'] is None else round(res['cond'],3)}"
              f" | {time.time()-t0:.0f}s", flush=True)

    def arr(k): return np.array([x[k] for x in rows if x[k] is not None])
    out = {k: float(arr(k).mean()) for k in ('hand', 'net', 'oracle', 'cond') if len(arr(k))}
    json.dump({'means': out, 'rows': rows}, open('tmp/G1_waveprior_power/expB.json', 'w'), indent=1)
    print("\n=== EXP B (nowcast /gt_std) ===")
    for k in ('hand', 'net', 'oracle', 'cond'):
        if k in out:
            print(f"  {k:7s} {out[k]:.3f}")
    if 'cond' in out and 'oracle' in out:
        if out['cond'] <= out['oracle'] + 0.02:
            print("  ⟹ cond reaches the mmse-oracle ⟹ 67% is mmse_bank-specific ⟹ ① WEAKENS")
        else:
            print("  ⟹ cond leaves oracle headroom too ⟹ UNIVERSAL gap ⟹ ① lives")


if __name__ == '__main__':
    main()
