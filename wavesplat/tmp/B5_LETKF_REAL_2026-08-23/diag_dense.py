"""B-5 진단 — ⚠**비교 팔이 아니다.** *"경쟁자가 나쁜가, 내가 하네스를 망가뜨렸는가"* 를 가른다.

`G-8` 이 **0/12** 로 떨어졌다. 저장소 규율(`B2_WANGFIX` 선례)은 여기서 **진단**을 요구한다.
결정적 시험: **관측을 조밀하게 주면 LETKF 가 작동하는가?**
- 작동하면 → 실패 원인은 **관측 밀도/기하**이고 **구현은 멀쩡**하다.
- 작동 안 하면 → **우리 LETKF 구현이 이 기하에서 못 쓰는 것**이다.

⛔**이 결과는 우열 비교에 쓰지 않는다** — 우리 팔은 조밀 관측을 받지 않았다.
프레임당 12점(배포 조건) → **200점**, 창 상한 600 → **6000**.
"""
import json, os, sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for p in (HERE, ROOT, os.path.join(ROOT, 'scripts'), os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23')):
    sys.path.insert(0, p)
os.chdir(ROOT)
import gates_b5 as G
from scene_station import scene_station
from zero_ref import zero_ref_nowcast
from wavesplat.baselines.letkf_grid import LETKFGrid
from scipy.interpolate import RegularGridInterpolator

OUT = os.path.join(HERE, 'diag_dense.jsonl')


def one(record, m_scan, m_cap, scale):
    sc = scene_station(record, win=G.WINDOWS[1], noise=G.NOISE, with_future_obs=False)
    # ★조밀 재추출: 같은 창·같은 잡음 규약, 프레임당 점 수만 올린다
    sc2 = scene_station(record, win=G.WINDOWS[1], noise=G.NOISE)
    from scene_station import RECORDS_12
    import importlib, scene_station as SS
    sc2 = SS.scene_station.__wrapped__(record, win=G.WINDOWS[1], noise=G.NOISE) \
        if hasattr(SS.scene_station, '__wrapped__') else None
    return sc


def run(record, m_scan, m_cap, scale):
    import scene_station as SS
    sc = SS.scene_station(record, win=G.WINDOWS[1], noise=G.NOISE,
                          m_scan=m_scan, m_cap=m_cap)
    gx, gy = np.asarray(sc['gxq']), np.asarray(sc['gyq'])
    def axis(a):
        n = min(G.GRID_MAX, max(8, int(round((a.max() - a.min()) / G.GRID_DX)) + 1))
        return np.linspace(a.min(), a.max(), n)
    lx, ly = axis(gx), axis(gy)
    P = np.asarray(sc['P'], float); z = np.asarray(sc['z'], float)
    ts = np.unique(P[:, 2]); dt = float(np.median(np.diff(ts)))
    np.random.seed(G.SEED)
    f = LETKFGrid(lx, ly, depth=float(sc['depth']), n_ensemble=G.N_ENS, dt=dt,
                  obs_noise=G.OBS_NOISE, loc_radius=50.0)
    if scale:
        amp = max(1e-6, float(np.sqrt(np.mean(z ** 2))))
        f.ensemble = f.ensemble / 0.1 * amp
        f.add_inflation = f.add_inflation / 0.1 * amp
    t0 = time.time()
    for tt in ts:
        m = P[:, 2] == tt
        f.update(P[m, 0], P[m, 1], z[m])
    Zl = f.predict_grid()
    Xq, Yq = np.meshgrid(gx, gy)
    it = RegularGridInterpolator((ly, lx), Zl, bounds_error=False, fill_value=None)
    Z = it(np.column_stack([Yq.ravel(), Xq.ravel()])).reshape(len(gy), len(gx))
    g = np.asarray(sc['maps_gt'][0], float); mm = np.isfinite(g)
    now = float(np.sqrt(np.mean((Z[mm] - g[mm]) ** 2)) / sc['sg'])
    return dict(record=record, m_scan=m_scan, m_cap=m_cap, scale=scale,
                n_obs=int(len(z)), nowcast=now, zero=zero_ref_nowcast(record, G.WINDOWS[1]),
                pred_std=float(np.std(Z[mm])), gt_std=float(np.std(g[mm])),
                wall_s=time.time() - t0)


if __name__ == '__main__':
    with open(OUT, 'w') as fh:
        print(f"{'레코드':8s} {'설정':16s} {'관측':>6s} {'제로':>7s} {'LETKF':>7s} {'예측σ/참σ':>9s}")
        for rec in G.RECORDS:
            for tag, ms, mc, sc_ in (('배포 12pt', 12, 600, False),
                                     ('조밀 200pt', 200, 6000, False),
                                     ('조밀+스케일', 200, 6000, True)):
                try:
                    r = run(rec, ms, mc, sc_); r['tag'] = tag
                except Exception as e:
                    r = dict(record=rec, tag=tag, error=repr(e)[:80])
                fh.write(json.dumps(r) + '\n'); fh.flush()
                print(f"{rec:8s} {tag:16s} " + (
                    f"{r['n_obs']:6d} {r['zero']:7.3f} {r['nowcast']:7.3f} "
                    f"{r['pred_std']/r['gt_std']:9.3f}" if 'error' not in r else r['error']), flush=True)
