"""A2 reproduction spot-check of the LEARNED arm (6 windows).

Re-runs the *identical* learned-arm code path of tmp/wp_real_2026-07-25/run_wp_real.py
(imported read-only, not copied) on 6 windows -- one per record plus one collapse window --
and compares to wp_real.json:learned_now. Frozen ckpt, no training, no GT in the loop.
Writes tmp/A2_realproto_2026-08-06/spot_learned.json.
"""
import os, sys, json, time
import warnings; warnings.filterwarnings('ignore')
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.join(ROOT, 'tmp', 'wp_real_2026-07-25'))
os.chdir(ROOT)

OUT = 'tmp/A2_realproto_2026-08-06/spot_learned.json'
SPOT = [('YS17', 0), ('AA15a', 5), ('AA14', 2), ('AA15b', 3), ('LJ18g', 1), ('LJ18g', 3)]
SEEDS = (0, 1, 2, 3, 4)


def main():
    import torch
    import d3_compare_gifs as cg
    import d3_beamform_cond_diffusion as cd
    from d3_setdiff_prior import denorm_atoms
    from d3_beamform_net import KLIM, NK
    from run_wp_real import fit_render_leads, score_maps      # read-only import

    stats = torch.load(cd.PRIOR_CKPT, map_location=cd.DEVICE)['stats']
    betas = cd.cosine_betas(cd.T).to(cd.DEVICE); abar = torch.cumprod(1 - betas, 0)
    model = cd.CondSetDenoiser().to(cd.DEVICE)
    model.load_state_dict(torch.load('results/direction3/beamform_cond_diffusion.pt',
                                     map_location=cd.DEVICE))
    model.eval()

    ref = {(r['record'], r['win']): r for r in
           json.load(open('tmp/wp_real_2026-07-25/wp_real.json'))['results']}
    out = []
    for rec, win in SPOT:
        tw = time.time()
        sc = cg.scene_real(rec, win=win)
        depth = sc['depth']
        P = sc['P'].astype(np.float32); z = sc['z'].astype(np.float32)
        if len(z) > cd.M_OBS:
            idx = np.random.default_rng(0).choice(len(z), cd.M_OBS, replace=False)
            P, z = P[idx], z[idx]
        t0 = P[:, 2].min(); Ps = P.copy(); Ps[:, 2] -= t0
        Bmap = cd._bmap_from_obs(Ps, z)
        tabs0 = float(sc['Pq'][0, 2])
        lead_ts = [tabs0 + float(L) - t0 for L in sc['leads']]
        target = {'gx': sc['gxq'], 'gy': sc['gyq'], 't': lead_ts[0],
                  'gt': sc['maps_gt'][0], 'Hs': sc['Hs']}
        Pt = torch.tensor(Ps, device=cd.DEVICE, dtype=torch.float32)
        zt = torch.tensor(z, device=cd.DEVICE, dtype=torch.float32)
        gt = sc['maps_gt']; sg = float(np.std(gt[np.isfinite(gt)]))

        seed_now = []
        for sd in SEEDS:
            X = cd.sample(model, Bmap, abar, betas, n=4, seed=int(sd))
            preds = []
            for b in range(4):
                xb = cd.polish_atoms(X[b], stats, Pt, zt, depth, iters=80)
                pa, _ = cd.atoms_to_field(xb, stats, Ps, z, depth, target)
                if pa is not None:
                    preds.append(pa)
            bma = np.mean(preds, axis=0)
            m0 = np.isfinite(gt[0]) & np.isfinite(bma)
            seed_now.append(float(np.sqrt(np.mean((bma[m0] - gt[0][m0]) ** 2)) / sg))
        med = float(np.median(seed_now))
        r0 = ref[(rec, win)]['learned_now']
        out.append(dict(record=rec, win=win, learned_now_repro=med,
                        learned_now_wp_real=r0, delta=med - r0,
                        per_seed=seed_now, sec=round(time.time() - tw, 1)))
        print(f"  [{rec} w{win}] repro {med:.6f} vs wp_real {r0:.6f}  d {med-r0:+.2e} "
              f"({out[-1]['sec']}s)", flush=True)

    json.dump(dict(spot=SPOT and [list(s) for s in SPOT], seeds=list(SEEDS),
                   results=out), open(OUT, 'w'), indent=1)
    print('saved', OUT, flush=True)


if __name__ == '__main__':
    main()
