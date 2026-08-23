"""B-6 러너 — 실시간 예산. 판정은 `gates_b6.py` 에 결과 전 동결."""
import json, os, sys, time, warnings
warnings.filterwarnings('ignore')
for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[v] = '1'
import numpy as np, torch
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for p in (HERE, ROOT, os.path.join(ROOT, 'scripts'),
          os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23'),
          os.path.join(ROOT, 'tmp', 'bankmmse_port_2026-07-29')):
    sys.path.insert(0, p)
os.chdir(ROOT)
import gates_b6 as G
from scene_station import scene_station
import make_gif_bank_deployed as M

OUT = os.path.join(HERE, 'results_b6.jsonl')
pc = time.perf_counter


def run(record):
    from wavesplat.data_loader import FrameData
    from wavesplat.optimizer.sequential_update import make_updater
    import wavesplat.optimizer.sequential_update as SU
    from wavesplat.run import DEFAULT_CONFIG, set_seed
    sc = scene_station(record, win=G.WIN, noise=G.NOISE, with_future_obs=True)
    t_shift = float(np.asarray(sc['P'])[:, 2].min()); t0 = float(sc['Pq'][0, 2]) - t_shift
    frames = M.frames_from_scene(sc, t_shift, dtype=torch.float32)
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(depth=float(sc['depth']), seed=G.SEED, device='cpu',
               bank_refit_every_s=G.TAU)

    refit_t, upd_t, pred_t, init_t = [], [], [], []
    orig = SU.SequentialUpdater._bank_fit_window
    def timed(self, *a, **kw):
        s = pc(); r = orig(self, *a, **kw); refit_t.append(pc() - s); return r
    SU.SequentialUpdater._bank_fit_window = timed
    try:
        for rep in range(G.N_REPEAT):
            set_seed(G.SEED); upd = make_updater(cfg)
            n0 = len(refit_t)
            s = pc(); upd.initialize(frames,
                                     x_range=(float(sc['gxq'].min()), float(sc['gxq'].max())),
                                     y_range=(float(sc['gyq'].min()), float(sc['gyq'].max())),
                                     verbose=False); init_t.append(pc() - s)
            n_init_refit = len(refit_t) - n0
            gxq, gyq = np.asarray(sc['gxq']), np.asarray(sc['gyq'])
            tx = torch.tensor(gxq, dtype=torch.float32); ty = torch.tensor(gyq, dtype=torch.float32)
            s = pc(); upd.predict_field(tx, ty, t0); pred_t.append(pc() - s)
            Pf = np.asarray(sc['P_fut'], float); zf = np.asarray(sc['z_fut'], float)
            for l_ in sc['leads'][1:]:
                tt = t0 + float(l_) + t_shift
                m = np.abs(Pf[:, 2] - tt) < 1e-6
                if m.sum() == 0: continue
                nq = int(m.sum())
                fr = FrameData(t=float(l_) + t0,
                               x_obs=torch.tensor(Pf[m, 0], dtype=torch.float32),
                               y_obs=torch.tensor(Pf[m, 1], dtype=torch.float32),
                               z_obs=torch.tensor(zf[m], dtype=torch.float32),
                               confidence=torch.ones(nq, dtype=torch.float32),
                               incidence_angles=torch.zeros(nq, dtype=torch.float32))
                before = len(refit_t)
                s = pc(); upd.update(fr); dt_ = pc() - s
                if len(refit_t) == before:          # 재적합이 없던 프레임만 갱신 단가로 센다
                    upd_t.append(dt_)
    finally:
        SU.SequentialUpdater._bank_fit_window = orig
    med = lambda v: float(np.median(v)) if v else float('nan')
    P = np.asarray(sc['P']); ts = np.unique(P[:, 2]); dt_frame = float(np.median(np.diff(ts)))
    return dict(record=record, t_init=med(init_t), t_refit=med(refit_t[n_init_refit:] or refit_t),
                t_upd=med(upd_t), t_pred=med(pred_t), n_refit=len(refit_t),
                refit_samples=refit_t[:12], dt_frame=dt_frame, npx=int(sc['npx']),
                n_obs=int(len(sc['z'])), threads=torch.get_num_threads(), device='cpu',
                ratio=med(refit_t[n_init_refit:] or refit_t) / G.TAU,
                frame_cost=med(upd_t) + med(pred_t))


def main():
    with open(OUT, 'w') as fh:
        for rec in G.RECORDS:
            try: out = run(rec)
            except Exception as e:
                import traceback
                out = dict(record=rec, error=repr(e), tb=traceback.format_exc()[-500:])
            fh.write(json.dumps(out) + '\n'); fh.flush()
            print(f"{rec:8s} " + (f"init={out['t_init']:.3f}s refit={out['t_refit']:.4f}s "
                  f"upd={out['t_upd']:.5f}s pred={out['t_pred']:.4f}s ratio={out['ratio']:.3f} "
                  f"dt={out['dt_frame']:.3f}" if 'error' not in out else str(out['error'])[:70]), flush=True)


if __name__ == '__main__':
    main()
