"""W12 러너 — 실측까지 포함한 통일의 비용. 규칙은 gates_w12 단일출처.

`tmp/P1_real_path_2026-08-11/final_compare.py` 의 하네스를 그대로 쓰되,
팔마다 config 를 덮어쓴다(그 파일의 `estimate()` 에는 override 훅이 없어 여기서 재구현).
채점 규약도 그 파일과 동일: **σ 정규화 RMSE, 제로필드 = 1.0**.
"""
import json, os, sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for p in (HERE, ROOT, os.path.join(ROOT, 'scripts'),
          os.path.join(ROOT, 'tmp', 'bankmmse_port_2026-07-29')):
    sys.path.insert(0, p)
os.chdir(ROOT)
import gates_w12 as G  # noqa: E402

import d3_compare_gifs as cg          # noqa: E402
import run_rolling_refit as RR        # noqa: E402
import make_gif_bank_deployed as M    # noqa: E402
OUT = os.path.join(HERE, 'results.jsonl')


def estimate_with(sc, extra, seed=G.SEED, sc_obs=None):
    """`M.estimate` 와 동일하되 config 를 덮어쓴다."""
    from wavesplat.optimizer.sequential_update import make_updater
    from wavesplat.run import DEFAULT_CONFIG, set_seed
    sc_obs = sc if sc_obs is None else sc_obs
    t_shift = float(np.asarray(sc_obs['P'])[:, 2].min())
    t0 = float(sc['Pq'][0, 2]) - t_shift
    frames = M.frames_from_scene(sc_obs, t_shift, dtype=torch.float32)
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(depth=float(sc['depth']), seed=seed, device='cpu')
    cfg.update(extra)
    set_seed(seed)
    upd = make_updater(cfg)
    t = time.time()
    upd.initialize(frames, x_range=(float(sc['gxq'].min()), float(sc['gxq'].max())),
                   y_range=(float(sc['gyq'].min()), float(sc['gyq'].max())), verbose=False)
    return upd, t0, time.time() - t


def main():
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            try: r = json.loads(l)
            except json.JSONDecodeError: continue
            if 'error' not in r: done.add((r['cell'], r['record']))
    with open(OUT, 'a') as fh:
        for rec in G.RECORDS:
            sc = cg.scene_real(rec, win=G.WIN)
            st = RR.build_stream(rec, G.WIN)
            gxq, gyq = np.asarray(st['gxq']), np.asarray(st['gyq'])
            Xq, Yq = np.meshgrid(gxq, gyq)
            taus = np.arange(0.0, st['lead_max'] + 1e-9, 1.0)
            for arm in G.ARM_ORDER:
                if (arm, rec) in done:
                    continue
                t0w = time.time()
                try:
                    upd, t0, el = estimate_with(sc, G.ARMS[arm])
                    curve = []
                    for tau in taus:
                        g = np.asarray(st['Zq'][int(round(tau / st['dt']))], float)
                        mm = np.isfinite(g)
                        Z = upd.predict_field(torch.tensor(gxq, dtype=torch.float32),
                                              torch.tensor(gyq, dtype=torch.float32),
                                              t0 + tau).cpu().numpy().reshape(len(gyq), len(gxq))
                        curve.append(float(np.sqrt(np.mean((Z[mm] - g[mm]) ** 2)) / st['sg']))
                    rec_out = dict(cell=arm, record=rec, win=G.WIN,
                                   route=getattr(upd, '_auto_selected', None)
                                         or ('bank_mmse' if getattr(upd, '_bank', None) else None),
                                   real_gate=getattr(upd, '_auto_real_gate', None),
                                   varpro_fired=bool(getattr(upd, '_last_varpro', None)) or None,
                                   n_atoms=getattr(getattr(upd, 'model', None), 'n_gaussians', None),
                                   nowcast=curve[0], leadmean=float(np.mean(curve)),
                                   curve=curve, init_s=el, wall_s=time.time() - t0w)
                except Exception as e:
                    import traceback
                    rec_out = dict(cell=arm, record=rec, error=repr(e),
                                   tb=traceback.format_exc()[-1200:], wall_s=time.time() - t0w)
                fh.write(json.dumps(rec_out) + '\n'); fh.flush()
                tag = (f"nowcast={rec_out['nowcast']:.4f} lead={rec_out['leadmean']:.4f} "
                       f"route={rec_out['route']} N={rec_out['n_atoms']}"
                       if 'error' not in rec_out else str(rec_out['error'])[:70])
                print(f'{arm:14s} {rec:7s} {tag}', flush=True)
    rows = [json.loads(l) for l in open(OUT)]
    print('\n=== 게이트 ===')
    for gid, ok, msg in G.gates(rows):
        print(f'  [{"PASS" if ok else "FAIL"}] {gid}: {msg[:110]}')


if __name__ == '__main__':
    main()
