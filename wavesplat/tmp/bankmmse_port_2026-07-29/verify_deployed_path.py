"""End-to-end verification of the bank-MMSE port THROUGH THE DEPLOYED PATH.

Runs wavesplat.optimizer.sequential_update.make_updater on the shipped
DEFAULT_CONFIG (init_method='auto'), feeds it the same frozen scene_real windows
as the 2026-07-27 GIF ledger via the same FrameData adapter, lets the
REAL-DOMAIN GATE act, and scores the protocol 1 s leads with the ledger's
normalization.  Also measures init and per-frame predict latency.

Expected: gate fires -> bank_mmse route, median-of-3, scores ~= the blind m3
measurement (not bit-identical: the updater sees time-sorted float32 obs, which
changes the val-split draw -- an ordinary seed-level difference).
"""
import os, sys, json, time
os.environ.setdefault('OMP_NUM_THREADS', '4')
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
os.chdir(ROOT)

sys.path.insert(0, 'tmp/bankmmse_port_2026-07-29')
from measure_blind_vs_frozen import dense_gt_std, score, WINDOWS

OUT = 'tmp/bankmmse_port_2026-07-29/deployed_path_verify.json'


def frames_from_scene(sc, t_shift):
    """Verbatim from tmp/real_gif_2026-07-27/make_real_estimation_gif.py."""
    from wavesplat.data_loader import FrameData
    P = np.asarray(sc['P'], float); z = np.asarray(sc['z'], float)
    out = []
    for tt in np.unique(P[:, 2]):
        m = P[:, 2] == tt
        n = int(m.sum())
        out.append(FrameData(t=float(tt - t_shift),
                             x_obs=torch.tensor(P[m, 0], dtype=torch.float32),
                             y_obs=torch.tensor(P[m, 1], dtype=torch.float32),
                             z_obs=torch.tensor(z[m], dtype=torch.float32),
                             confidence=torch.ones(n, dtype=torch.float32),
                             incidence_angles=torch.zeros(n, dtype=torch.float32)))
    return out


def main():
    import d3_compare_gifs as cg
    from wavesplat.optimizer.sequential_update import make_updater
    from wavesplat.run import DEFAULT_CONFIG, set_seed

    results = {}
    for rec, win in WINDOWS:
        key = f'{rec}_w{win}'
        print(f'== {key}', flush=True)
        sc = cg.scene_real(rec, win)
        sg = dense_gt_std(rec, win)
        t_shift = float(np.asarray(sc['P'])[:, 2].min())
        t0 = float(sc['Pq'][0, 2]) - t_shift
        frames = frames_from_scene(sc, t_shift)

        cfg = dict(DEFAULT_CONFIG)
        cfg.update(depth=float(sc['depth']), seed=42, device='cpu')
        set_seed(42)
        upd = make_updater(cfg)
        t_init = time.time()
        upd.initialize(frames,
                       x_range=(float(sc['gxq'].min()), float(sc['gxq'].max())),
                       y_range=(float(sc['gyq'].min()), float(sc['gyq'].max())),
                       verbose=False)
        init_s = time.time() - t_init
        prim = upd._primary
        assert prim._auto_real_gate is True and prim.init_method == 'bank_mmse'

        gxt = torch.tensor(sc['gxq'], dtype=torch.float32)
        gyt = torch.tensor(sc['gyq'], dtype=torch.float32)
        maps, pred_ms = [], []
        for L in sc['leads']:
            tp = time.time()
            maps.append(upd.predict_field(gxt, gyt, t0 + float(L)).cpu().numpy())
            pred_ms.append((time.time() - tp) * 1000.0)
        r = score(np.array(maps), sc, sg)
        entry = dict(
            route=prim.init_method,
            real_gate=bool(prim._auto_real_gate),
            bank_members=len(prim._bank),
            bank_seeds=list(prim._bank_seeds),
            bank_kp=[f.kp for f in prim._bank],
            bank_val_misfit=[round(f.vr, 4) for f in prim._bank],
            init_seconds=round(init_s, 1),
            predict_ms_per_frame=round(float(np.mean(pred_ms)), 1),
            scores_protocol_1s_leads=r)
        results[key] = entry
        print(f"  route={entry['route']} members={entry['bank_members']} "
              f"init {entry['init_seconds']}s predict {entry['predict_ms_per_frame']}ms/frame",
              flush=True)
        print(f"  now {r['nowcast_nrmse']:.4f} mean {r['mean_nrmse']:.4f} "
              f"corr(now) {r['nowcast_corr']:+.3f}", flush=True)

    json.dump(results, open(OUT, 'w'), indent=1)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
