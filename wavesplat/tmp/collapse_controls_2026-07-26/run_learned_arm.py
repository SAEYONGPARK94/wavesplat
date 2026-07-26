"""W8 GPU half — LEARNED ARM on the 18 collapse windows (15 held-out + 3 design).

Protocol frozen in tmp/collapse_controls_2026-07-26/PREREG_GPU_ADDENDUM.md; the learned-arm
body is copied verbatim from tmp/wp_real_2026-07-25/run_wp_real.py Phase B (5 diffusion
seeds x n=4 posterior samples, 80-iter polish, atoms_to_field BMA, lead-0 pooled-GT nRMSE,
median-of-5 cited).

GPU (proposal + polish). Baselines are NOT recomputed (escapes.json already passed the
bit-identity control). Design-3 must reproduce wp_real.json to 1e-9.

  python3 tmp/collapse_controls_2026-07-26/run_learned_arm.py
"""
import os, sys, json, time, hashlib
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import warnings; warnings.filterwarnings('ignore')
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.join(ROOT, 'tmp', 'heldout_real_2026-07-26'))

OUT = os.path.join(ROOT, 'tmp/collapse_controls_2026-07-26/learned_arm.json')
PREREG = os.path.join(ROOT, 'tmp/collapse_controls_2026-07-26/PREREG_GPU_ADDENDUM.md')

HELDOUT = [('BS11b', 4), ('BS11c', 3), ('BS11c', 4),
           ('BS13a', 0), ('BS13a', 1), ('BS13a', 2), ('BS13a', 4), ('BS13a', 5),
           ('BS13c', 2), ('BS13c', 3),
           ('LJ18n', 0), ('LJ18n', 1), ('LJ18n', 3), ('LJ18n', 4), ('LJ18n', 5)]
DESIGN = [('AA14', 0), ('AA15a', 3), ('LJ18g', 3)]
TARGETS = HELDOUT + DESIGN
SEEDS = (0, 1, 2, 3, 4)          # frozen, verbatim from run_wp_real.py
G = 9.81


def main():
    with open(PREREG, 'rb') as f:
        ph = hashlib.sha256(f.read()).hexdigest()
    print(f"# PREREG_GPU_ADDENDUM sha256 {ph[:16]}  ({len(TARGETS)} windows)", flush=True)

    import heldout_cfg
    heldout_cfg.install()
    import torch
    import d3_compare_gifs as cg
    import d3_beamform_cond_diffusion as cd
    from d3_setdiff_prior import denorm_atoms
    from d3_beamform_net import KLIM, NK

    stats = torch.load(cd.PRIOR_CKPT, map_location=cd.DEVICE)['stats']
    betas = cd.cosine_betas(cd.T).to(cd.DEVICE); abar = torch.cumprod(1 - betas, 0)
    model = cd.CondSetDenoiser().to(cd.DEVICE)
    model.load_state_dict(torch.load('results/direction3/beamform_cond_diffusion.pt',
                                     map_location=cd.DEVICE))
    model.eval()
    kax = np.linspace(-KLIM, KLIM, NK)
    print(f"# learned arm on {cd.DEVICE}, median-of-{len(SEEDS)} diffusion seeds", flush=True)

    t_all = time.time(); rows = []
    for rec, win in TARGETS:
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
        gt = sc['maps_gt']; sg = float(np.std(gt[np.isfinite(gt)]))

        seed_now, seed_neff, seed_npx = [], [], []
        for sd in SEEDS:
            X = cd.sample(model, Bmap, abar, betas, n=4, seed=int(sd))
            preds_a2f, neffs = [], []
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
            if not preds_a2f:
                print(f"  [{rec} w{win}] seed {sd}: no field (all samples None)", flush=True)
                continue
            bma_now = np.mean(preds_a2f, axis=0)
            m0 = np.isfinite(gt[0]) & np.isfinite(bma_now)
            if m0.sum() < 8:
                print(f"  [{rec} w{win}] seed {sd}: mask {m0.sum()} < 8", flush=True)
                continue
            seed_now.append(float(np.sqrt(np.mean((bma_now[m0] - gt[0][m0]) ** 2)) / sg))
            seed_neff.append(float(np.mean(neffs)) if neffs else float('nan'))
            seed_npx.append(int(m0.sum()))

        r = dict(record=rec, win=int(win),
                 arena=('design' if (rec, win) in DESIGN else 'heldout'),
                 Hs=float(sc['Hs']), kp=float(sc['kp']), depth=float(depth),
                 n_obs=int(len(sc['z'])), npx_scored=seed_npx[0] if seed_npx else None,
                 learned_now=float(np.median(seed_now)) if seed_now else None,
                 learned_now_seeds=[float(v) for v in seed_now],
                 learned_now_sd=float(np.std(seed_now, ddof=1)) if len(seed_now) > 1 else 0.0,
                 n_eff=float(np.mean(seed_neff)) if seed_neff else None,
                 bf_peak=peak, bf_peak_edge=peak_edge, sec=time.time() - tw)
        rows.append(r)
        print(f"  [{rec} w{win}] learned {r['learned_now']:.4f} (sd {r['learned_now_sd']:.3f})"
              f" | n_eff {r['n_eff']:.1f} | npx {r['npx_scored']} ({r['sec']:.0f}s)", flush=True)

    json.dump(dict(rows=rows, prereg=PREREG, prereg_sha256=ph, seeds=list(SEEDS),
                   targets=[list(t) for t in TARGETS], elapsed_s=time.time() - t_all),
              open(OUT, 'w'), indent=1)
    print(f"\nwrote {OUT}: {len(rows)} rows, {time.time()-t_all:.0f}s", flush=True)


if __name__ == '__main__':
    main()
