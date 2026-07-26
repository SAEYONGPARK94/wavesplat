"""MIXED-RETRAIN GATE G4 (accuracy). Verbatim copy of
tmp/wp_real_2026-07-25/run_wp_real.py; the ONLY edits are the --ckpt / --out
arguments. Original untouched.

ORIGINAL DOCSTRING FOLLOWS
STAGE-3b — WavePrior learned set-proposal vs analytic mmse_bank, REAL stereo.
Frozen protocol: tmp/wp_real_2026-07-25/PREREG.md. Additive only; checkpoints read-only;
data/ untouched. Writes tmp/wp_real_2026-07-25/wp_real.json.

Phase A (CPU pool): hand arm (mmse_bank) nowcast+lead per window, 30 windows.
Phase B (GPU main): learned arm (beamform-cond-diffusion + polish + LS field), median-of-5
                    diffusion seeds; records n_eff, beamform peak, per-seed nowcast/lead;
                    for YS17 also learned nowcast at U=0 and U=U_med (helper) for the 2x2.
Phase C (CPU main): hand arm at U=U_med for YS17 (2x2 Doppler on the gated record).
"""
import os, sys, json, time, argparse
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import warnings; warnings.filterwarnings('ignore')
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))

RECORDS_ORDER = ['YS17', 'AA15a', 'AA14', 'AA15b', 'LJ18g']
WINS = list(range(6))
SEEDS = (0, 1, 2, 3, 4)
COLLAPSE = {('AA14', 0), ('AA15a', 3), ('LJ18g', 3)}
YS17_U_MED = (-0.95, 0.60)                      # doppler_real.json record-median
OUT = 'tmp/mixed_retrain_2026-07-26/wp_real_mixed_v1.json'
CKPT = 'results/direction3/beamform_cond_diffusion_mixed_v1.pt'
G = 9.81


# ── shared metric (exact d3_compare_gifs normalization) ──────────────────────
def score_maps(sc, maps):
    """maps: (nl,ny,nx). Returns (nowcast_nrmse, lead_avg_nrmse, per_lead list)."""
    gt = sc['maps_gt']; sg = float(np.std(gt[np.isfinite(gt)]))
    per = []
    for li in range(len(sc['leads'])):
        m = np.isfinite(gt[li]) & np.isfinite(maps[li])
        if m.sum() < 8:
            per.append(None); continue
        per.append(float(np.sqrt(np.mean((maps[li][m] - gt[li][m]) ** 2)) / sg))
    ok = [v for v in per if v is not None]
    return per[0], (float(np.mean(ok)) if ok else None), per


# ── hand arm (CPU worker) ────────────────────────────────────────────────────
def hand_window(args):
    rec, win = args
    import d3_compare_gifs as cg
    from d3_sa_nomp import mmse_bank_estimate
    sc = cg.scene_real(rec, win=win)
    s0 = 0.05 * sc['Hs']; npx = sc['npx']; nl = len(sc['leads'])
    fq = mmse_bank_estimate(sc['P'], sc['z'], sc['depth'], sc['Pq'], sigma0=s0,
                            kp_hint=sc['kp'], seed=sc['seed'])
    now, lead, _ = score_maps(sc, fq[:npx * nl].reshape(nl, *sc['Xq'].shape))
    out = dict(record=rec, win=win, Hs=float(sc['Hs']), kp=float(sc['kp']),
               n_obs=int(len(sc['z'])), collapse=(rec, win) in COLLAPSE,
               hand_now=now, hand_lead=lead)
    if rec == 'YS17':                              # 2x2: hand at U_med
        fqd = mmse_bank_estimate(sc['P'], sc['z'], sc['depth'], sc['Pq'], sigma0=s0,
                                 kp_hint=sc['kp'], seed=sc['seed'], Uc=YS17_U_MED)
        nowd, _, _ = score_maps(sc, fqd[:npx * nl].reshape(nl, *sc['Xq'].shape))
        out['hand_now_umed'] = nowd
    return out


# ── learned-arm LS-fit + render (dispersion + optional Doppler) ───────────────
def fit_render_leads(kx, ky, Ps, z, depth, gx, gy, lead_ts, Uc=(0.0, 0.0),
                     lambda_reg=0.5):
    """LS-fit A,phi on obs then render (nl,ny,nx) at lead_ts. omega=disp(k)+k.U.
    At Uc=(0,0) reproduces fit_amplitudes_phases + render_wave_field (verified ~1e-5)."""
    kx = np.asarray(kx, float); ky = np.asarray(ky, float); N = len(kx)
    k = np.sqrt(kx ** 2 + ky ** 2 + 1e-12)
    om = np.sqrt(G * k * np.tanh(k * depth)) + kx * Uc[0] + ky * Uc[1]
    x, y, t = Ps[:, 0], Ps[:, 1], Ps[:, 2]
    psi = kx[None] * x[:, None] + ky[None] * y[:, None] - om[None] * t[:, None]
    Phi = np.concatenate([np.cos(psi), np.sin(psi)], axis=1)
    coef = np.linalg.solve(Phi.T @ Phi + lambda_reg * np.eye(2 * N), Phi.T @ z)
    a, b = coef[:N], coef[N:]
    A = np.sqrt(a ** 2 + b ** 2); phi = np.arctan2(-b, a)
    Xg, Yg = np.meshgrid(gx, gy)
    out = np.empty((len(lead_ts), *Xg.shape))
    base = kx[:, None, None] * Xg[None] + ky[:, None, None] * Yg[None] + phi[:, None, None]
    for li, tt in enumerate(lead_ts):
        out[li] = (A[:, None, None] * np.cos(base - om[:, None, None] * tt)).sum(0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=CKPT)
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    globals()['OUT'] = a.out
    print(f'# checkpoint {a.ckpt}', flush=True)
    t_all = time.time()
    # ---- Phase A: hand arm (CPU pool) ----
    jobs = [(r, w) for r in RECORDS_ORDER for w in WINS]
    print(f"# Phase A: hand arm, {len(jobs)} windows, 4 workers", flush=True)
    hand = {}
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(hand_window, j): j for j in jobs}
        for fut in as_completed(futs):
            o = fut.result(); hand[(o['record'], o['win'])] = o
            print(f"  [hand {o['record']} w{o['win']}] now {o['hand_now']:.3f} "
                  f"lead {o['hand_lead']:.3f}"
                  + (f" | umed {o['hand_now_umed']:.3f}" if 'hand_now_umed' in o else ''),
                  flush=True)
    print(f"# Phase A done ({time.time()-t_all:.0f}s)\n", flush=True)

    # ---- Phase B/C: learned arm (GPU main) ----
    import torch
    import d3_compare_gifs as cg
    import d3_beamform_cond_diffusion as cd
    from d3_setdiff_prior import denorm_atoms
    from d3_beamform_net import KLIM, NK
    stats = torch.load(cd.PRIOR_CKPT, map_location=cd.DEVICE)['stats']
    betas = cd.cosine_betas(cd.T).to(cd.DEVICE); abar = torch.cumprod(1 - betas, 0)
    model = cd.CondSetDenoiser().to(cd.DEVICE)
    model.load_state_dict(torch.load(a.ckpt,
                                     map_location=cd.DEVICE))
    model.eval()
    kax = np.linspace(-KLIM, KLIM, NK)
    print(f"# Phase B: learned arm on {cd.DEVICE}, median-of-{len(SEEDS)} seeds", flush=True)

    results = []
    for rec in RECORDS_ORDER:
        for win in WINS:
            tw = time.time()
            sc = cg.scene_real(rec, win=win)
            depth = sc['depth']
            P = sc['P'].astype(np.float32); z = sc['z'].astype(np.float32)
            if len(z) > cd.M_OBS:
                idx = np.random.default_rng(0).choice(len(z), cd.M_OBS, replace=False)
                P, z = P[idx], z[idx]
            t0 = P[:, 2].min(); Ps = P.copy(); Ps[:, 2] -= t0
            Bmap = cd._bmap_from_obs(Ps, z)
            bm = Bmap[0].cpu().numpy(); ij = np.unravel_index(np.argmax(bm), bm.shape)
            peak = [float(kax[ij[1]]), float(kax[ij[0]])]
            peak_edge = bool(ij[0] in (0, NK - 1) or ij[1] in (0, NK - 1))
            tabs0 = float(sc['Pq'][0, 2])
            lead_ts = [tabs0 + float(L) - t0 for L in sc['leads']]
            target = {'gx': sc['gxq'], 'gy': sc['gyq'], 't': lead_ts[0],
                      'gt': sc['maps_gt'][0], 'Hs': sc['Hs']}
            Pt = torch.tensor(Ps, device=cd.DEVICE, dtype=torch.float32)
            zt = torch.tensor(z, device=cd.DEVICE, dtype=torch.float32)

            seed_now, seed_lead, seed_neff = [], [], []
            seed_now_u0h, seed_now_umed = [], []      # YS17 2x2 (helper)
            for sd in SEEDS:
                X = cd.sample(model, Bmap, abar, betas, n=4, seed=int(sd))
                preds_a2f, leadfields, leadfields_dop, neffs = [], [], [], []
                for b in range(4):
                    xb = cd.polish_atoms(X[b], stats, Pt, zt, depth, iters=80)
                    pa, _ = cd.atoms_to_field(xb, stats, Ps, z, depth, target)
                    if pa is not None:
                        preds_a2f.append(pa)
                    kxi, kyi, ai = denorm_atoms(xb, stats)
                    kxi = kxi.cpu().numpy(); kyi = kyi.cpu().numpy(); ai = ai.cpu().numpy()
                    if ai.max() <= 0:
                        continue
                    keep = ai > 0.03 * ai.max()
                    kxi, kyi, aii = kxi[keep], kyi[keep], ai[keep]
                    if len(kxi) > 30:
                        o = np.argsort(-aii)[:30]; kxi, kyi = kxi[o], kyi[o]
                    neffs.append(len(kxi))
                    leadfields.append(fit_render_leads(kxi, kyi, Ps, z, depth,
                                      sc['gxq'], sc['gyq'], lead_ts, Uc=(0.0, 0.0)))
                    if rec == 'YS17':
                        leadfields_dop.append(fit_render_leads(kxi, kyi, Ps, z, depth,
                                              sc['gxq'], sc['gyq'], lead_ts, Uc=YS17_U_MED))
                # primary nowcast = atoms_to_field BMA, scored lead-0 with pooled-GT sg
                bma_now = np.mean(preds_a2f, axis=0)
                gt = sc['maps_gt']; sg = float(np.std(gt[np.isfinite(gt)]))
                m0 = np.isfinite(gt[0]) & np.isfinite(bma_now)
                now_a = float(np.sqrt(np.mean((bma_now[m0] - gt[0][m0]) ** 2)) / sg)
                # lead-averaged from helper BMA
                bl = np.mean(leadfields, axis=0)
                _, lead_a, _ = score_maps(sc, bl)
                seed_now.append(now_a); seed_lead.append(lead_a)
                seed_neff.append(float(np.mean(neffs)))
                if rec == 'YS17':
                    m0h = np.isfinite(gt[0]) & np.isfinite(bl[0])
                    seed_now_u0h.append(float(np.sqrt(np.mean((bl[0][m0h] - gt[0][m0h]) ** 2)) / sg))
                    bld = np.mean(leadfields_dop, axis=0)
                    md = np.isfinite(gt[0]) & np.isfinite(bld[0])
                    seed_now_umed.append(float(np.sqrt(np.mean((bld[0][md] - gt[0][md]) ** 2)) / sg))

            h = hand[(rec, win)]
            r = dict(record=rec, win=win, collapse=(rec, win) in COLLAPSE,
                     Hs=h['Hs'], kp=h['kp'], n_obs=h['n_obs'],
                     hand_now=h['hand_now'], hand_lead=h['hand_lead'],
                     learned_now=float(np.median(seed_now)),
                     learned_now_seeds=[float(v) for v in seed_now],
                     learned_now_sd=float(np.std(seed_now, ddof=1) if len(seed_now) > 1 else 0),
                     learned_lead=float(np.median([v for v in seed_lead if v is not None])),
                     n_eff=float(np.mean(seed_neff)),
                     bf_peak=peak, bf_peak_edge=peak_edge,
                     delta_now=h['hand_now'] - float(np.median(seed_now)),
                     sec=time.time() - tw)
            if rec == 'YS17':
                r['hand_now_umed'] = h['hand_now_umed']
                r['learned_now_u0_helper'] = float(np.median(seed_now_u0h))
                r['learned_now_umed'] = float(np.median(seed_now_umed))
            results.append(r)
            print(f"  [{rec} w{win}] hand {r['hand_now']:.3f} / learned {r['learned_now']:.3f}"
                  f" (sd {r['learned_now_sd']:.3f}) d {r['delta_now']:+.3f} | n_eff {r['n_eff']:.1f}"
                  f" | bfpeak ({peak[0]:+.2f},{peak[1]:+.2f}){' EDGE' if peak_edge else ''}"
                  f" ({r['sec']:.0f}s)"
                  + (f" | 2x2 h[{r['hand_now']:.3f}->{r['hand_now_umed']:.3f}]"
                     f" l[{r['learned_now_u0_helper']:.3f}->{r['learned_now_umed']:.3f}]"
                     if rec == 'YS17' else ''), flush=True)

    json.dump(dict(results=results, prereg='PREREG.md', seeds=list(SEEDS),
                   ys17_u_med=list(YS17_U_MED), collapse=sorted(list(map(list, COLLAPSE)))),
              open(OUT, 'w'), indent=1)
    print(f"\nsaved {OUT}  ({len(results)} windows, {time.time()-t_all:.0f}s total)", flush=True)


if __name__ == '__main__':
    main()
