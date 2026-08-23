"""B-5 러너 — LETKF 를 실측에 건다. 판정은 `gates_b5.py` 에 결과 전 동결.

**프로토콜은 `B-3`/`B-4` 와 글자 그대로 같다**:
기억창으로 스핀업 → lead ℓ 마다 **① 먼저 예측·채점 ② 그 다음 그 시각 관측 투입**.
`LETKF-FC50` 은 ②를 **하지 않고 예보만 전진**한다(우리 `FROZEN` 과 같은 과제).
채점식도 같다: `sqrt(mean((Z-gt)^2)) / sg`, 채점 격자는 `gxq × gyq`.
"""
import json, os, sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
B2 = os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23')
for p in (HERE, ROOT, os.path.join(ROOT, 'scripts'), B2):
    sys.path.insert(0, p)
os.chdir(ROOT)
import gates_b5 as G                       # noqa: E402
from scene_station import scene_station     # noqa: E402
from wavesplat.baselines.letkf_grid import LETKFGrid   # noqa: E402
from scipy.interpolate import RegularGridInterpolator  # noqa: E402

OUT = os.path.join(HERE, 'results_b5.jsonl')


def _axis(a):
    n = min(G.GRID_MAX, max(8, int(round((a.max() - a.min()) / G.GRID_DX)) + 1))
    return np.linspace(a.min(), a.max(), n)


def run(record, win, arm):
    spec = G.ARMS[arm]
    t0w = time.time()
    sc = scene_station(record, win=win, noise=G.NOISE, with_future_obs=True)
    gx, gy = np.asarray(sc['gxq']), np.asarray(sc['gyq'])
    lx, ly = _axis(gx), _axis(gy)
    box = float(np.hypot(gx.max() - gx.min(), gy.max() - gy.min()))
    loc = 50.0 if spec['loc'] == 'fixed50' else max(8.0, box / 2.0)
    P = np.asarray(sc['P'], float); z = np.asarray(sc['z'], float)
    ts = np.unique(P[:, 2]); dt = float(np.median(np.diff(ts)))
    np.random.seed(G.SEED)
    f = LETKFGrid(lx, ly, depth=float(sc['depth']), n_ensemble=G.N_ENS, dt=dt,
                  obs_noise=G.OBS_NOISE, loc_radius=loc)
    for tt in ts:                                    # 스핀업 (기억창)
        m = P[:, 2] == tt
        f.update(P[m, 0], P[m, 1], z[m])
    Xq, Yq = np.meshgrid(gx, gy)
    q = np.column_stack([Yq.ravel(), Xq.ravel()])    # (y, x) 순서
    Pf = np.asarray(sc['P_fut'], float); zf = np.asarray(sc['z_fut'], float)
    n_steps = max(1, int(round(1.0 / dt)))           # lead 격자 1 s
    per_lead, fut_cyc, finite = {}, 0, True
    for li, l_ in enumerate(sc['leads']):
        # ① 먼저 예측·채점
        g = np.asarray(sc['maps_gt'][li], float); mm = np.isfinite(g)
        if mm.sum() > 0:
            Zl = f.predict_grid()
            finite &= bool(np.isfinite(Zl).all())
            it = RegularGridInterpolator((ly, lx), Zl, bounds_error=False, fill_value=None)
            Z = it(q).reshape(len(gy), len(gx))
            per_lead[float(l_)] = float(np.sqrt(np.mean((Z[mm] - g[mm]) ** 2)) / sc['sg'])
        if li == len(sc['leads']) - 1:
            break
        # ② 1 s 전진 — 동화 팔은 그 시각 관측을 먹고, 예보 팔은 안 먹는다
        tt = sc['t0'] + float(sc['leads'][li + 1])
        m = np.abs(Pf[:, 2] - tt) < 1e-6
        for s in range(n_steps):
            last = (s == n_steps - 1)
            if spec['da'] and last and m.sum() > 0:
                f.update(Pf[m, 0], Pf[m, 1], zf[m]); fut_cyc += 1
            else:
                f.ensemble = np.stack([f._forecast_member(x) for x in f.ensemble], axis=0)
    fc = [v for k, v in per_lead.items() if k >= 2.0]
    return dict(cell=arm, record=record, win=win, lead0=per_lead.get(0.0),
                fc_mean=float(np.mean(fc)) if fc else None,
                per_lead={str(k): v for k, v in sorted(per_lead.items())},
                loc_radius=loc, box_diag=box, n_ens=G.N_ENS, obs_noise=G.OBS_NOISE,
                letkf_grid=[len(ly), len(lx)], dt=dt, future_cycles=fut_cyc,
                finite=finite, sigma=float(sc['sg']), wall_s=time.time() - t0w)


def main():
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            try: r = json.loads(l)
            except json.JSONDecodeError: continue
            if 'error' not in r: done.add((r['cell'], r['record'], r['win']))
    with open(OUT, 'a') as fh:
        for rec in G.RECORDS:
            for win in G.WINDOWS:
                for arm in G.ARM_ORDER:
                    if (arm, rec, win) in done: continue
                    try: out = run(rec, win, arm)
                    except Exception as e:
                        import traceback
                        out = dict(cell=arm, record=rec, win=win, error=repr(e),
                                   tb=traceback.format_exc()[-600:])
                    fh.write(json.dumps(out) + '\n'); fh.flush()
                    print(f"{arm:12s} {rec:7s} w{win} " + (
                        f"lead0={out['lead0']:.4f} fc={out['fc_mean']:.4f} loc={out['loc_radius']:.0f} "
                        f"fut={out['future_cycles']} {out['wall_s']:.1f}s"
                        if 'error' not in out else str(out['error'])[:70]), flush=True)


if __name__ == '__main__':
    main()
