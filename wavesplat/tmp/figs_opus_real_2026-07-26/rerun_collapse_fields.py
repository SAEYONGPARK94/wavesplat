"""Re-run the two arms on the analytic COLLAPSE windows and dump the actual fields.

Protocol copied verbatim from tmp/wp_real_2026-07-25/run_wp_real.py (frozen PREREG):
  hand    = mmse_bank_estimate(P, z, depth, Pq, sigma0=0.05*Hs, kp_hint, seed=i0)
  learned = beamform-cond set-diffusion (5 seeds x n=4 BMA) + 80-iter polish + LS field
            primary nowcast metric = atoms_to_field BMA ; lead fields = fit_render_leads BMA
            (the two agree to ~1e-7 at lead 0, as verified in WP_REAL_2026-07-25.md)

Cited value = median over the 5 diffusion seeds (frozen). The RENDERED field is the
BMA field of the MEDIAN seed, so the picture and the number are the same object.

Writes tmp/figs_opus_real_2026-07-26/collapse_fields.npz  (+ .json with all numbers)
Run from wavesplat/ package root.
"""
import os, sys, json, time
os.environ.setdefault('OMP_NUM_THREADS', '4')
import warnings; warnings.filterwarnings('ignore')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

WINDOWS = [('LJ18g', 3), ('AA14', 0)]
SEEDS = (0, 1, 2, 3, 4)
G = 9.81
OUT_NPZ = os.path.join(HERE, 'collapse_fields.npz')
OUT_JSON = os.path.join(HERE, 'collapse_fields.json')


def score_maps(sc, maps):
    gt = sc['maps_gt']; sg = float(np.std(gt[np.isfinite(gt)]))
    per = []
    for li in range(len(sc['leads'])):
        m = np.isfinite(gt[li]) & np.isfinite(maps[li])
        if m.sum() < 8:
            per.append(None); continue
        per.append(float(np.sqrt(np.mean((maps[li][m] - gt[li][m]) ** 2)) / sg))
    ok = [v for v in per if v is not None]
    return per[0], (float(np.mean(ok)) if ok else None), per


def fit_render_leads(kx, ky, Ps, z, depth, gx, gy, lead_ts, Uc=(0.0, 0.0),
                     lambda_reg=0.5):
    """verbatim from run_wp_real.py"""
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
    import torch
    import d3_compare_gifs as cg
    import d3_beamform_cond_diffusion as cd
    from d3_setdiff_prior import denorm_atoms
    from d3_beamform_net import KLIM, NK
    from d3_sa_nomp import mmse_bank_estimate

    stats = torch.load(cd.PRIOR_CKPT, map_location=cd.DEVICE)['stats']
    betas = cd.cosine_betas(cd.T).to(cd.DEVICE); abar = torch.cumprod(1 - betas, 0)
    model = cd.CondSetDenoiser().to(cd.DEVICE)
    model.load_state_dict(torch.load('results/direction3/beamform_cond_diffusion.pt',
                                     map_location=cd.DEVICE))
    model.eval()
    kax = np.linspace(-KLIM, KLIM, NK)
    print(f'# device {cd.DEVICE}, windows {WINDOWS}', flush=True)

    store, meta = {}, {}
    for rec, win in WINDOWS:
        tw = time.time()
        key = f'{rec}_w{win}'
        sc = cg.scene_real(rec, win=win)
        depth = sc['depth']; npx = sc['npx']; nl = len(sc['leads'])
        s0 = 0.05 * sc['Hs']

        # ---- hand arm (identical call) ----
        fq = mmse_bank_estimate(sc['P'], sc['z'], depth, sc['Pq'], sigma0=s0,
                                kp_hint=sc['kp'], seed=sc['seed'])
        hand_maps = fq[:npx * nl].reshape(nl, *sc['Xq'].shape)
        h_now, h_lead, h_per = score_maps(sc, hand_maps)

        # ---- learned arm ----
        P = sc['P'].astype(np.float32); z = sc['z'].astype(np.float32)
        if len(z) > cd.M_OBS:
            idx = np.random.default_rng(0).choice(len(z), cd.M_OBS, replace=False)
            P, z = P[idx], z[idx]
        t0 = P[:, 2].min(); Ps = P.copy(); Ps[:, 2] -= t0
        Bmap = cd._bmap_from_obs(Ps, z)
        bm = Bmap[0].cpu().numpy(); ij = np.unravel_index(np.argmax(bm), bm.shape)
        peak = [float(kax[ij[1]]), float(kax[ij[0]])]
        tabs0 = float(sc['Pq'][0, 2])
        lead_ts = [tabs0 + float(L) - t0 for L in sc['leads']]
        target = {'gx': sc['gxq'], 'gy': sc['gyq'], 't': lead_ts[0],
                  'gt': sc['maps_gt'][0], 'Hs': sc['Hs']}
        Pt = torch.tensor(Ps, device=cd.DEVICE, dtype=torch.float32)
        zt = torch.tensor(z, device=cd.DEVICE, dtype=torch.float32)

        gt = sc['maps_gt']; sg = float(np.std(gt[np.isfinite(gt)]))
        seed_now, seed_neff, seed_fields = [], [], []
        for sd in SEEDS:
            X = cd.sample(model, Bmap, abar, betas, n=4, seed=int(sd))
            preds_a2f, leadfields, neffs = [], [], []
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
                                  sc['gxq'], sc['gyq'], lead_ts))
            bma_now = np.mean(preds_a2f, axis=0)
            m0 = np.isfinite(gt[0]) & np.isfinite(bma_now)
            now_a = float(np.sqrt(np.mean((bma_now[m0] - gt[0][m0]) ** 2)) / sg)
            seed_now.append(now_a)
            seed_neff.append(float(np.mean(neffs)))
            seed_fields.append(np.mean(leadfields, axis=0))
            print(f'  [{key} seed {sd}] learned nowcast {now_a:.4f} '
                  f'n_eff {np.mean(neffs):.1f}', flush=True)

        l_now = float(np.median(seed_now))
        i_med = int(np.argsort(seed_now)[len(seed_now) // 2])   # median seed
        learn_maps = seed_fields[i_med]
        _, l_lead, l_per = score_maps(sc, learn_maps)
        # helper-vs-a2f control at lead 0 for the displayed seed
        m0 = np.isfinite(gt[0]) & np.isfinite(learn_maps[0])
        l_now_helper = float(np.sqrt(np.mean((learn_maps[0][m0] - gt[0][m0]) ** 2)) / sg)

        store[f'{key}_gt'] = gt
        store[f'{key}_hand'] = hand_maps
        store[f'{key}_learned'] = learn_maps
        store[f'{key}_leads'] = sc['leads']
        store[f'{key}_gxq'] = sc['gxq']
        store[f'{key}_gyq'] = sc['gyq']
        store[f'{key}_obsP'] = sc['P']
        meta[key] = dict(record=rec, win=win, Hs=float(sc['Hs']), kp=float(sc['kp']),
                         depth=float(depth), n_obs=int(len(sc['z'])), gt_std=sg,
                         hand_now=h_now, hand_lead=h_lead, hand_per_lead=h_per,
                         learned_now_median=l_now, learned_now_seeds=seed_now,
                         median_seed_index=i_med, median_seed=int(SEEDS[i_med]),
                         learned_now_displayed_helper=l_now_helper,
                         learned_lead_displayed=l_lead, learned_per_lead=l_per,
                         n_eff=float(np.mean(seed_neff)), bf_peak=peak,
                         sec=time.time() - tw)
        print(f'  [{key}] hand {h_now:.4f} | learned(median-of-5) {l_now:.4f} '
              f'| displayed seed {SEEDS[i_med]} helper {l_now_helper:.4f} '
              f'({time.time()-tw:.0f}s)', flush=True)

    np.savez_compressed(OUT_NPZ, **store)
    json.dump(meta, open(OUT_JSON, 'w'), indent=1)

    # ---- verification against the frozen wp_real.json ----
    wp = {(r['record'], r['win']): r for r in
          json.load(open('tmp/wp_real_2026-07-25/wp_real.json'))['results']}
    print('\n# VERIFICATION vs tmp/wp_real_2026-07-25/wp_real.json')
    for rec, win in WINDOWS:
        k = f'{rec}_w{win}'; ref = wp[(rec, win)]; got = meta[k]
        print(f'  {k}: hand    ref {ref["hand_now"]:.6f}  got {got["hand_now"]:.6f}'
              f'  |diff| {abs(ref["hand_now"]-got["hand_now"]):.2e}')
        print(f'  {k}: learned ref {ref["learned_now"]:.6f}  got '
              f'{got["learned_now_median"]:.6f}'
              f'  |diff| {abs(ref["learned_now"]-got["learned_now_median"]):.2e}')
        print(f'  {k}: learned per-seed ref {[round(v,6) for v in ref["learned_now_seeds"]]}')
        print(f'  {k}: learned per-seed got {[round(v,6) for v in got["learned_now_seeds"]]}')
    print(f'\nwrote {OUT_NPZ}, {OUT_JSON}')


if __name__ == '__main__':
    main()
