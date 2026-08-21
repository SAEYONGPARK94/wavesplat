"""PRO (6) — the separation experiment E1 flagged as UNRUN (tmp/e1_grid_2026-08-05/RESULT.md
section 6-1: "grid step and effective suppression radius were never separated").

PLACEMENT ONLY. No optimizer, no adaptive-N, no tracker, no GPU. Deterministic
(seed fixed; the only RNG is the pad branch). Production files are NOT modified --
grid params are passed as call arguments and MIN_DK is restored in a finally block.

LADDER (each step changes exactly ONE derived quantity):
  A  box +-4.0  step 0.100  supp +-0.200   <- SHIPPED
  G  box +-4.0  step 0.020  supp +-0.200   <- A + finer STEP only        (never run)
  F  box +-4.0  step 0.020  supp +-0.160   <- G + tighter SUPPRESSION    (never run)
  B  box +-0.8  step 0.020  supp +-0.160   <- F + narrower BOX  == G1/E1 cell B
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, REPO)
DATA = os.path.join(REPO, 'data_finemesh_2026-07-23')
SEED = 42
BUDGET = 15

# cell -> (kx_range, n_coarse, MIN_DK)   MIN_DK chosen so supp = int(MIN_DK/step)+1 cells
CELLS = {
    'A_shipped':  ((-4.0, 4.0),  81, 0.15),   # int(1.5)+1=2  -> +-0.200
    'G_step':     ((-4.0, 4.0), 401, 0.19),   # int(9.5)+1=10 -> +-0.200
    'F_step_sup': ((-4.0, 4.0), 401, 0.15),   # int(7.5)+1=8  -> +-0.160
    'B_narrow':   ((-0.8, 0.8),  81, 0.15),   # int(7.5)+1=8  -> +-0.160
}
EPISODES = ['episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_uniform.h5',
            'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5',
            'episode_BF3_2comp_seed0_uniform.h5', 'episode_BF3_5comp_seed0_uniform.h5',
            'episode_BF2_1comp_seed0_uniform.h5']


def match(ekx, eky, gkx, gky):
    from scipy.optimize import linear_sum_assignment
    if not len(ekx) or not len(gkx):
        return []
    ks = max(np.hypot(gkx, gky).max(), 0.01)
    C = np.hypot(ekx[:, None] - gkx[None, :], eky[:, None] - gky[None, :]) / ks
    r, c = linear_sum_assignment(C)
    return [(i, j, C[i, j]) for i, j in zip(r, c) if C[i, j] < 0.5]


def main():
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    import h5py, torch
    from wavesplat.data_loader import EpisodeDataLoader
    import wavesplat.optimizer.initializer as I

    out = []
    for ep in EPISODES:
        p = os.path.join(DATA, ep)
        with h5py.File(p, 'r') as f:
            gt = np.asarray(f['config/wave_components'])
            depth = float(f['config'].attrs['depth'])
        frames = EpisodeDataLoader(p).get_init_window(tp_factor=2.0)
        gkx, gky, gA = gt[:, 2], gt[:, 3], gt[:, 0]
        for cell, (rng, nc, mdk) in CELLS.items():
            saved = I.MIN_DK
            t0 = time.time()
            try:
                I.MIN_DK = mdk
                torch.manual_seed(SEED)
                kx, ky, w, phi = I.initialize_from_frames_v2(
                    frames, BUDGET, depth, kx_range=rng, ky_range=rng,
                    n_coarse=nc, seed=SEED)
            finally:
                I.MIN_DK = saved
            kx = kx.numpy().astype(float); ky = ky.numpy().astype(float)
            m = match(kx, ky, gkx, gky)
            idx = [j for _, j, _ in m]
            rec = dict(episode=ep, cell=cell, n_gt=len(gA), n_matched=len(m),
                       det=len(m) / len(gA),
                       Edet=float(np.sum(gA[idx] ** 2) / np.sum(gA ** 2)) if idx else 0.0,
                       n_oob=int((np.hypot(kx, ky) > 0.8).sum()),
                       kmax=float(np.hypot(kx, ky).max()),
                       secs=round(time.time() - t0, 1))
            if m:
                d = []
                for i, j, _ in m:
                    a = np.degrees(np.arctan2(ky[i], kx[i])) - np.degrees(np.arctan2(gky[j], gkx[j]))
                    a = abs(a); d.append(360 - a if a > 180 else a)
                rec['dir_deg'] = float(np.mean(d))
                rec['kmag_rel'] = float(np.mean([abs(np.hypot(kx[i], ky[i]) - np.hypot(gkx[j], gky[j]))
                                                 / max(np.hypot(gkx[j], gky[j]), 1e-6) for i, j, _ in m]))
            else:
                rec['dir_deg'] = float('nan'); rec['kmag_rel'] = float('nan')
            out.append(rec)
            print(f"{ep[8:-3]:28s} {cell:12s} det={rec['det']:.3f} Edet={rec['Edet']:.3f} "
                  f"dir={rec['dir_deg']:7.2f} |k|rel={rec['kmag_rel']:.4f} "
                  f"oob={rec['n_oob']:2d} kmax={rec['kmax']:.2f} ({rec['secs']}s)", flush=True)
    json.dump(out, open(os.path.join(HERE, 'pro_separate_grid.json'), 'w'), indent=1)

    print('\n' + '=' * 96)
    print('LADDER — median over the 4 CROWDED episodes (each step = ONE derived quantity)')
    print('=' * 96)
    cr = [e for e in EPISODES if 'comp' in e and int(e.split('_')[2][:-4]) >= 15]
    sp = [e for e in EPISODES if e not in cr]
    for nm, eps in (('CROWDED4', cr), ('SPARSE3', sp)):
        print(f'\n{nm}')
        print(f'  {"cell":12s}{"det":>7s}{"Edet":>7s}{"dir_deg":>9s}{"|k|rel":>9s}{"n_oob":>7s}')
        for cell in CELLS:
            v = [r for r in out if r['cell'] == cell and r['episode'] in eps]
            print(f'  {cell:12s}{np.median([r["det"] for r in v]):7.3f}'
                  f'{np.median([r["Edet"] for r in v]):7.3f}'
                  f'{np.nanmedian([r["dir_deg"] for r in v]):9.2f}'
                  f'{np.nanmedian([r["kmag_rel"] for r in v]):9.4f}'
                  f'{np.median([r["n_oob"] for r in v]):7.1f}')
        print(f'  --- attributable deltas (paired median over episodes) ---')
        for a, b, lab in (('A_shipped', 'G_step', 'STEP 0.100->0.020'),
                          ('G_step', 'F_step_sup', 'SUPP 0.200->0.160'),
                          ('F_step_sup', 'B_narrow', 'BOX 4.0->0.8')):
            da = np.median([[r for r in out if r['cell'] == b and r['episode'] == e][0]['det'] -
                            [r for r in out if r['cell'] == a and r['episode'] == e][0]['det']
                            for e in eps])
            dE = np.median([[r for r in out if r['cell'] == b and r['episode'] == e][0]['Edet'] -
                            [r for r in out if r['cell'] == a and r['episode'] == e][0]['Edet']
                            for e in eps])
            print(f'    {lab:22s} d_det {da:+.3f}   d_Edet {dE:+.3f}')


if __name__ == '__main__':
    main()
