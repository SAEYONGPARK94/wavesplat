"""B-6b 러너 — 렌더 점당 비용. 판정은 `gates_b6b.py` 에 결과 전 동결."""
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
import gates_b6b as G
from scene_station import scene_station
import make_gif_bank_deployed as M
pc = time.perf_counter
OUT = os.path.join(HERE, 'results_b6b.jsonl')


def main():
    from wavesplat.optimizer.sequential_update import make_updater
    from wavesplat.run import DEFAULT_CONFIG, set_seed
    with open(OUT, 'w') as fh:
        print(f"{'레코드':8s} {'N':>7s} {'t(ms)':>9s} {'µs/점':>8s}")
        for rec in G.RECORDS:
            sc = scene_station(rec, win=G.WIN, noise=G.NOISE)
            t_shift = float(np.asarray(sc['P'])[:, 2].min()); t0 = float(sc['Pq'][0, 2]) - t_shift
            frames = M.frames_from_scene(sc, t_shift, dtype=torch.float32)
            cfg = dict(DEFAULT_CONFIG); cfg.update(depth=float(sc['depth']), seed=G.SEED, device='cpu')
            set_seed(G.SEED); upd = make_updater(cfg)
            upd.initialize(frames, x_range=(float(sc['gxq'].min()), float(sc['gxq'].max())),
                           y_range=(float(sc['gyq'].min()), float(sc['gyq'].max())), verbose=False)
            gx, gy = np.asarray(sc['gxq']), np.asarray(sc['gyq'])
            for n in G.NS:
                side = int(round(n ** 0.5))
                ax = torch.tensor(np.linspace(gx.min(), gx.max(), side), dtype=torch.float32)
                ay = torch.tensor(np.linspace(gy.min(), gy.max(), side), dtype=torch.float32)
                upd.predict_field(ax, ay, t0)                     # 워밍업 1회
                s = []
                for _ in range(G.N_REPEAT):
                    a = pc(); Z = upd.predict_field(ax, ay, t0); s.append(pc() - a)
                t = float(np.median(s))
                row = dict(record=rec, n_req=side * side, n_pts=int(Z.numel()), t=t,
                           samples=s, threads=torch.get_num_threads(), device='cpu')
                fh.write(json.dumps(row) + '\n'); fh.flush()
                print(f"{rec:8s} {row['n_pts']:7d} {t*1e3:9.3f} {t/row['n_pts']*1e6:8.2f}", flush=True)


if __name__ == '__main__':
    main()
