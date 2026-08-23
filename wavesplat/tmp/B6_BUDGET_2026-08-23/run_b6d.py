"""B-6c 러너 — 제어 경로 질의(8점)로 프레임 예산 재측정. 판정은 `gates_b6c.py` 에 동결."""
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
import gates_b6d as G
from scene_station import scene_station
import make_gif_bank_deployed as M
pc = time.perf_counter
OUT = os.path.join(HERE, 'results_b6d.jsonl')


def run(record):
    from wavesplat.data_loader import FrameData
    from wavesplat.optimizer.sequential_update import make_updater
    from wavesplat.run import DEFAULT_CONFIG, set_seed
    sc = scene_station(record, win=G.WIN, noise=G.NOISE, with_future_obs=True)
    t_shift = float(np.asarray(sc['P'])[:, 2].min()); t0 = float(sc['Pq'][0, 2]) - t_shift
    frames = M.frames_from_scene(sc, t_shift, dtype=torch.float32)
    cfg = dict(DEFAULT_CONFIG); cfg.update(depth=float(sc['depth']), seed=G.SEED, device='cpu')
    cfg.update(G.CANDIDATE)              # ★배포 후보 구성 (재적합 5 s + 비동기)
    set_seed(G.SEED); upd = make_updater(cfg)
    s = pc(); upd.initialize(frames, x_range=(float(sc['gxq'].min()), float(sc['gxq'].max())),
                             y_range=(float(sc['gyq'].min()), float(sc['gyq'].max())),
                             verbose=False); t_init = pc() - s
    gx, gy = np.asarray(sc['gxq']), np.asarray(sc['gyq'])
    # ★제어 경로 질의: 8점 (2 foils x 4 stencil). 축을 2x4 로 잡아 정확히 8점을 만든다.
    cx = torch.tensor(np.linspace(gx.min(), gx.max(), 4), dtype=torch.float32)
    cy = torch.tensor(np.linspace(gy.min(), gy.max(), 2), dtype=torch.float32)
    gxt = torch.tensor(gx, dtype=torch.float32); gyt = torch.tensor(gy, dtype=torch.float32)
    upd.predict_field(cx, cy, t0); upd.predict_field(gxt, gyt, t0)     # 워밍업
    sc_, sg_ = [], []
    for _ in range(G.N_REPEAT):
        a = pc(); Zc = upd.predict_field(cx, cy, t0); sc_.append(pc() - a)
        a = pc(); Zg = upd.predict_field(gxt, gyt, t0); sg_.append(pc() - a)
    # 추적 갱신 단가
    Pf = np.asarray(sc['P_fut'], float); zf = np.asarray(sc['z_fut'], float)
    upd_t = []
    _t0run = time.perf_counter()
    for l_ in sc['leads'][1:]:
        if getattr(G, 'REALTIME_PACING', False):
            _slack = (_t0run + float(l_)) - time.perf_counter()
            if _slack > 0:
                time.sleep(_slack)
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
        a = pc(); upd.update(fr); upd_t.append(pc() - a)
    upd.bank_refit_wait(timeout=180)
    _h = upd.bank_refit_health()
    P = np.asarray(sc['P']); ts = np.unique(P[:, 2]); dt_frame = float(np.median(np.diff(ts)))
    med = lambda v: float(np.median(v))
    t_ctrl, t_grid, t_upd = med(sc_), med(sg_), med(upd_t)
    # 14 s 창 점유율 (τ=10 ⟹ 재적합 1회; 재적합 단가는 B-6 verdict 에서 읽는다)
    vb6 = json.load(open(os.path.join(HERE, 'verdict_b6.json')))
    t_refit = vb6['per_record'][record]['t_refit']
    n_frames = int(round(14.0 / dt_frame))
    duty = (t_init + 1 * t_refit + n_frames * (t_upd + t_ctrl)) / 14.0
    return dict(record=record, t_init=t_init, t_ctrl=t_ctrl, t_grid=t_grid, t_upd=t_upd,
                t_refit=t_refit, n_ctrl=int(Zc.numel()), n_grid=int(Zg.numel()),
                dt_frame=dt_frame, frame_cost_ctrl=t_upd + t_ctrl,
                fits_frame=bool(t_upd + t_ctrl < dt_frame),
                fits_100hz=bool(t_upd + t_ctrl < G.FAST_BUDGET),
                duty_14s=duty, n_frames=n_frames,
                samples_ctrl=sc_, samples_grid=sg_,
                threads=torch.get_num_threads(), device='cpu',
                wall_s=time.perf_counter() - _t0run,
                async_kicks=_h['kicks'], async_applied=_h['applied'],
                async_skipped=_h['skipped'], async_effective=_h['effective'])


def main():
    with open(OUT, 'w') as fh:
        for rec in G.RECORDS:
            try: out = run(rec)
            except Exception as e:
                import traceback
                out = dict(record=rec, error=repr(e), tb=traceback.format_exc()[-500:])
            fh.write(json.dumps(out) + '\n'); fh.flush()
            print(f"{rec:8s} " + (f"kick={out['async_kicks']} appl={out['async_applied']} "
                  f"ctrl8={out['t_ctrl']*1e3:7.3f}ms grid={out['t_grid']*1e3:8.2f}ms "
                  f"upd={out['t_upd']*1e3:6.3f}ms frame={out['frame_cost_ctrl']*1e3:6.3f}ms "
                  f"dt={out['dt_frame']*1e3:.0f}ms {'OK' if out['fits_frame'] else '⛔'} "
                  f"100Hz={'OK' if out['fits_100hz'] else 'X'} duty={out['duty_14s']:.3f}"
                  if 'error' not in out else str(out['error'])[:70]), flush=True)


if __name__ == '__main__':
    main()
