"""HOS nonlinear evaluation for A1/A2/B1/B2 (standing rule: headline must reproduce on HOS).

All four methods are trained on LINEAR synthetic spectra (they need discrete component labels,
which HOS surfaces don't provide). This harness tests transfer to weakly-NONLINEAR HOS waves
(d44 build_window, M=3) — the honest robustness check. RELAX (linear-physics, no learning) is
the reference. Any checkpoint that doesn't exist yet is skipped, so this can run now (B2+RELAX)
and be re-run once A1/A2/B1 finish.

Run: python3 scripts/d3_hos_eval.py [--nsamp 4] [--polish_iters 60]
"""
import os, sys, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from d23_main_results import panel
from d26_relax_iterative import iterative_relax
from wavesplat.model.tnse import fit_amplitudes_phases, render_wave_field
import d44_hos_nonlinear as H
import d3_setdiff_posterior as DP
from d3_setdiff_prior import SetDenoiser, cosine_betas, T
from d3_setdiff_posterior_b2 import polish_atoms
from d3_beamform_net import beamform

DEVICE = DP.DEVICE
SEAS = [('BF3', 0.6, 5.5), ('BF4', 1.2, 7.0), ('BF5', 2.0, 9.0)]
SEEDS = [0, 1, 2, 3, 4]      # 15 windows — 3 seeds too few for a sign test (Fable)
GX = np.linspace(10, 250, 80); GY = np.linspace(-100, 100, 60)
PRIOR = 'results/direction3/setdiff_prior.pt'
CK = dict(a2='results/direction3/beamform_detr.pt',
          # ds = fixed dataset (linear+HOS mix); dslin = linear-only control (isolates the
          # HOS-augmentation effect from dataset scale/standardization — Fable's key control)
          a1_ds='results/direction3/beamform_offset_ds.pt',
          a1_dslin='results/direction3/beamform_offset_dslin.pt',
          b1_ds='results/direction3/beamform_cond_diffusion_ds.pt',
          b1_dslin='results/direction3/beamform_cond_diffusion_dslin.pt')


def _field_from_kxky(kx, ky, Ps, zs, depth, target):
    if len(kx) == 0:
        return None
    obs = np.column_stack([Ps[:, 0], Ps[:, 1], Ps[:, 2], zs]).astype(np.float32)
    A, phi = fit_amplitudes_phases(kx, ky, obs, depth, lambda_reg=0.5)
    return render_wave_field(kx, ky, A, phi, depth, target['gx'], target['gy'], target['t'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nsamp', type=int, default=4)
    ap.add_argument('--polish_iters', type=int, default=60)
    a = ap.parse_args()

    # ---- load whatever is ready ----
    ck = torch.load(PRIOR, map_location=DEVICE); stats = ck['stats']
    prior = SetDenoiser(**ck.get('arch', {})).to(DEVICE); prior.load_state_dict(ck['model']); prior.eval()
    abar = ck['abar'].to(DEVICE); betas = ck['betas'].to(DEVICE)

    a1_extract = a2_extract = b1_sample = None
    a1_models = {}; b1_models = {}; a2 = None
    for key, path in (('a1_ds', CK['a1_ds']), ('a1_dslin', CK['a1_dslin'])):   # mix vs linear-only control
        if os.path.exists(path):
            from d3_beamform_offset import DetectOffsetCNN, extract as a1_extract
            m_ = DetectOffsetCNN().to(DEVICE); m_.load_state_dict(torch.load(path, map_location=DEVICE)); m_.eval()
            a1_models[key] = m_
    if os.path.exists(CK['a2']):
        from d3_beamform_detr import DETRSet, extract as a2_extract
        a2 = DETRSet().to(DEVICE); a2.load_state_dict(torch.load(CK['a2'], map_location=DEVICE)); a2.eval()
    for key, path in (('b1_ds', CK['b1_ds']), ('b1_dslin', CK['b1_dslin'])):
        if os.path.exists(path):
            from d3_beamform_cond_diffusion import CondSetDenoiser, sample as b1_sample
            m_ = CondSetDenoiser().to(DEVICE); m_.load_state_dict(torch.load(path, map_location=DEVICE)); m_.eval()
            b1_models[key] = m_
    b1_betas = cosine_betas(T).to(DEVICE); b1_abar = torch.cumprod(1 - b1_betas, 0)

    methods = (['relax', 'b2'] + list(a1_models) + (['a2'] if a2 else []) + list(b1_models))
    print(f"methods: {methods}  nsamp={a.nsamp} polish={a.polish_iters}", flush=True)
    acc = {m: [] for m in methods}
    win_tags = []
    t0 = time.time()

    for tag, Hs, Tp in SEAS:
        n_back = int(round(2.2 * Tp / 0.1))
        for seed in SEEDS:
            win_tags.append(tag)
            fr = H.build_window(Hs, Tp, seed, n_back, GX, GY, M=3)
            i_star = len(fr) - 2
            target = {'gx': GX, 'gy': GY, 't': fr[i_star]['t'], 'gt': fr[i_star]['gt'],
                      'Hs': float(fr[i_star]['gt'].std() * 4)}
            depth = H.DEPTH
            P = np.vstack([f['P'] for f in fr[:i_star + 1]]); z = np.concatenate([f['z'] for f in fr[:i_star + 1]])
            idx = np.random.default_rng(seed).choice(len(z), min(400, len(z)), replace=False)
            Ps, zs = P[idx].astype(np.float32), z[idx].astype(np.float32)

            # RELAX reference
            fld_r, _ = iterative_relax(Ps, zs, depth, target, 3)
            acc['relax'].append(panel(fld_r, target['gt'], target['Hs'])['nrmse'])

            # B2: frozen-prior DPS + Gauss-Newton polish, BMA
            Xd = DP.dps_sample(prior, stats, abar, betas, Ps, zs, depth, guidance=0.1, n=a.nsamp, seed=0)
            Pt = torch.tensor(Ps, device=DEVICE); zt = torch.tensor(zs, device=DEVICE)
            flds = []
            for b in range(a.nsamp):
                xb = polish_atoms(Xd[b], stats, Pt, zt, depth, iters=a.polish_iters)
                f, _ = DP.atoms_to_field(xb, stats, Ps, zs, depth, target)
                if f is not None:
                    flds.append(f)
            bma = np.mean(flds, axis=0) if flds else np.zeros_like(target['gt'])
            acc['b2'].append(panel(bma, target['gt'], target['Hs'])['nrmse'])

            # beamform map (invariant to global t-shift) for A1/A2/B1
            if a1_models or a2 or b1_models:
                Bmap = beamform(torch.tensor(Ps[None], device=DEVICE), torch.tensor(zs[None], device=DEVICE))
            for key, m_ in a1_models.items():
                with torch.no_grad():
                    Hp, oxp, oyp = m_(Bmap)
                kx, ky = a1_extract(Hp[0], oxp[0], oyp[0], refine=True)
                f = _field_from_kxky(kx, ky, Ps, zs, depth, target)
                acc[key].append(panel(f, target['gt'], target['Hs'])['nrmse'] if f is not None else 1.0)
            if a2:
                kx, ky = a2_extract(a2, Bmap)
                f = _field_from_kxky(kx, ky, Ps, zs, depth, target)
                acc['a2'].append(panel(f, target['gt'], target['Hs'])['nrmse'] if f is not None else 1.0)
            for key, m_ in b1_models.items():
                X = b1_sample(m_, Bmap, b1_abar, b1_betas, n=a.nsamp)
                flds = []
                for b in range(a.nsamp):
                    xb = polish_atoms(X[b], stats, Pt, zt, depth, iters=a.polish_iters)
                    f, _ = DP.atoms_to_field(xb, stats, Ps, zs, depth, target)
                    if f is not None:
                        flds.append(f)
                bmab = np.mean(flds, axis=0) if flds else np.zeros_like(target['gt'])
                acc[key].append(panel(bmab, target['gt'], target['Hs'])['nrmse'])

            msg = "  ".join(f"{m} {acc[m][-1]:.3f}" for m in methods)
            print(f"  [{tag} s{seed}] {msg}  ({time.time()-t0:.0f}s)", flush=True)

    wt = np.array(win_tags)
    seas = [s[0] for s in SEAS]
    print(f"\n=== HOS nonlinear (M=3) nRMSE — {len(wt)} windows, per-sea + pooled ===", flush=True)
    summary = {}
    for m in methods:
        arr = np.array(acc[m])
        per_sea = {s: float(arr[wt == s].mean()) for s in seas}
        summary[m] = dict(mean=float(arr.mean()), std=float(arr.std()), per_sea=per_sea)
        ps = "  ".join(f"{s} {per_sea[s]:.3f}" for s in seas)
        print(f"  {m:10s} pooled {arr.mean():.3f}±{arr.std():.3f}  |  {ps}", flush=True)
    # paired control(dslin) − mix(ds): does HOS augmentation help, per-window sign test
    print("\n  paired Δ = linear-only(control) − mix   (Δ>0 ⇒ HOS augmentation helps):", flush=True)
    for a_, b_ in (('a1_dslin', 'a1_ds'), ('b1_dslin', 'b1_ds')):
        if a_ in acc and b_ in acc:
            d = np.array(acc[a_]) - np.array(acc[b_])
            wins = int((d > 0).sum())
            summary[f'paired_{b_}'] = dict(mean_delta=float(d.mean()), wins=wins, n=len(d))
            print(f"    {b_:8s}: mean Δ {d.mean():+.4f}  improved {wins}/{len(d)} windows", flush=True)
    json.dump(dict(methods=summary, win_tags=list(wt)),
              open('results/direction3/hos_eval_new.json', 'w'), indent=2)
    print("saved results/direction3/hos_eval_new.json", flush=True)


if __name__ == '__main__':
    main()
