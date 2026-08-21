"""BS-3 부속 프로브 — **배로부터의 거리**로 오차와 처치효과를 재층화한다.

## 왜 필요한가 (BS-3 가 코드에서 직접 확인한 사실)
`wavesplat/run.py:355-366` 의 `shell_rmse` 는 **최근접 LiDAR 반환점까지의 거리**로 층화한다.
`bs3_recompute.py shells` 로 재계산한 결과 `0_5` 셸의 **배-거리 중앙값은 28.3 m**
(p10 15.6 / p90 50.2 / 최대 174.3 m) 이다. ⟹ 셸은 배-거리의 대용이 **아니다**.
"배 근처가 손해/이득" 류의 문장은 이 자료로 지지되지 않는다.

## 무엇을 하는가
`wavesplat.run` 이 프레임마다 호출하는 `rmse_observed` 를 **monkeypatch** 해서, 같은 프레임의
(pred, gt) 로 **배(원점) 중심 동심 고리** RMSE 를 추가 계산해 모듈 전역에 쌓는다.
원 함수에 그대로 위임하므로 기존 지표·판정은 **바뀌지 않는다**(순수 가산).
⚠ 저장소 코드는 **수정하지 않는다**.

배는 정지해 있다 (`wavesim/config.py:76 ship_speed_kts` 는 어디서도 읽히지 않는 dead config,
`wavesim/pipeline.py` 가 ship_x=ship_y=0 고정). 따라서 배 = 격자 원점, 고리는 시간불변.

사용:
    python3 tmp/BLANKSLATE_2026-08-12/bs3_ship_rings.py run      # 런
    python3 tmp/BLANKSLATE_2026-08-12/bs3_ship_rings.py analyze  # 집계
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
RX2 = os.path.join(REPO, 'tmp', 'rx2_init_2026-08-05')
for p in (REPO, RX2):
    sys.path.insert(0, p)

OUT = os.path.join(HERE, 'ship_rings.jsonl')

# ── 고리 정의 ────────────────────────────────────────────────────────────────
# 격자는 x[10,250] y[-100,100] 이므로 배-거리 최소는 10 m, 최대 269 m.
SHIP_EDGES = [10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0, np.inf]
SHIP_KEYS = [f'{int(a)}_{"inf" if b == np.inf else int(b)}'
             for a, b in zip(SHIP_EDGES[:-1], SHIP_EDGES[1:])]
# 비교용: 같은 프레임에서 관측거리 고리도 자체 계산 (run.py 산식과 일치 확인용)
OBS_EDGES = [0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, np.inf]
OBS_KEYS = [f'{int(a)}_{"inf" if b == np.inf else int(b)}'
            for a, b in zip(OBS_EDGES[:-1], OBS_EDGES[1:])]

ACC = []            # 프레임별 dict


def _rings(pred, gt, d, edges, keys):
    out = {}
    for k, (a, b) in zip(keys, zip(edges[:-1], edges[1:])):
        m = (d >= a) & (d < b)
        n = int(m.sum())
        out[k] = (float(np.mean((pred[m] - gt[m]) ** 2)) if n else float('nan'), n)
    return out


def install_patch():
    """`wavesplat.run.rmse_observed` 를 감싼다 — 원 동작 보존 + 고리 누적."""
    import wavesplat.run as WR
    orig = WR.rmse_observed
    if getattr(orig, '_bs3', False):
        return

    def patched(pred, gt, gx, gy, ox, oy, min_dist=10.0):
        if min_dist == 5.0:                      # 프레임당 첫 호출에서만 계산
            Xg, Yg = np.meshgrid(np.asarray(gx).ravel(), np.asarray(gy).ravel())
            dship = np.hypot(Xg, Yg)
            dobs = np.full(Xg.shape, np.inf)
            ox = np.asarray(ox).ravel(); oy = np.asarray(oy).ravel()
            for j in range(len(ox)):
                dobs = np.minimum(dobs, np.hypot(Xg - ox[j], Yg - oy[j]))
            ACC.append(dict(
                ship=_rings(pred, gt, dship, SHIP_EDGES, SHIP_KEYS),
                obs=_rings(pred, gt, dobs, OBS_EDGES, OBS_KEYS),
                mse_all=float(np.mean((pred - gt) ** 2)),
                var_gt=float(np.var(gt)),
                # 셸<->배거리 교차: 0_5 셸 격자점의 배-거리 중앙값
                shell05_ship_med=(float(np.median(dship[dobs < 5.0]))
                                  if np.any(dobs < 5.0) else float('nan')),
            ))
        return orig(pred, gt, gx, gy, ox, oy, min_dist)

    patched._bs3 = True
    WR.rmse_observed = patched


# ── 셀 ──────────────────────────────────────────────────────────────────────
CELLS = {
    # 현행 출하 구성 (estimation.yaml:12 lambda_l1=1e-4, run.py:32 tracking_lr=0.005)
    'SHIP':  {'lambda_l1': 1e-4, 'tracking_lr': 0.005},
    # 추적 축: 추적 끄기
    'NT':    {'lambda_l1': 1e-4, 'tracking_lr': 0.0},
    # λ 축: 배포 이전 값 (1e-3)
    'L1HI':  {'lambda_l1': 1e-3, 'tracking_lr': 0.005},
}

SPARSE = [f'data_s9_sparse_2026-08-11/episode_{bf}_{nc}comp_seed{s}_uniform_ra_rf_ds.h5'
          for bf in ('BF4', 'BF5') for nc in (2, 5) for s in (56, 57, 58)]
CROWDED = [f'data_s9_crowded_2026-08-11/episode_{bf}_{nc}comp_seed{s}_uniform_ra_rf_ds.h5'
           for bf in ('BF4', 'BF5') for nc in (15, 30) for s in (52, 53, 54)]
EPISODES = SPARSE + CROWDED
SEED = 1901                       # 어떤 배치에도 쓰인 적 없는 신선 분석 시드


def regime(ep):
    return 'sparse' if 'sparse' in ep else 'crowded'


def run_one(cell, ep, seed):
    import run_rx2b as R
    R.DATA = REPO
    install_patch()
    from wavesplat.run import run_estimation

    ACC.clear()
    cfg = R.cell_config('AUTO', ep)
    cfg.update(CELLS[cell])
    cfg['seed'] = seed
    t0 = time.time()
    r = run_estimation(os.path.join(REPO, ep), cfg)
    fr = r['frame_results']

    # 고리 집계: 프레임 평균 MSE -> RMSE (면적가중은 고리 안에서 이미 균등)
    def agg(kind, keys):
        out = {}
        for k in keys:
            v = [f[kind][k][0] for f in ACC if np.isfinite(f[kind][k][0])]
            n = [f[kind][k][1] for f in ACC]
            out[k] = dict(rmse=float(np.sqrt(np.mean(v))) if v else float('nan'),
                          frac=float(np.mean(n) / max(np.mean([
                              sum(f[kind][kk][1] for kk in keys) for f in ACC]), 1)))
        return out

    return dict(
        cell=cell, episode=ep, seed=seed, regime=regime(ep),
        nRMSE=float(np.nanmean([f['nRMSE'] for f in fr if 'nRMSE' in f])),
        all_resid=float(np.nanmean([f['obs_resid_rms'] for f in fr
                                    if 'obs_resid_rms' in f])),
        ship=agg('ship', SHIP_KEYS), obs=agg('obs', OBS_KEYS),
        shell05_ship_med=float(np.nanmean([f['shell05_ship_med'] for f in ACC])),
        n_frames=len(ACC), wall_s=time.time() - t0,
        cfg={k: cfg.get(k) for k in ('lambda_l1', 'tracking_lr', 'init_method')},
    )


def main_run():
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((r['cell'], r['episode'], r['seed']))
    jobs = [(c, e, SEED) for e in EPISODES for c in CELLS]
    todo = [j for j in jobs if j not in done]
    print(f'# 총 {len(jobs)} 잡, 남은 {len(todo)}', flush=True)
    with open(OUT, 'a') as fh:
        for i, (c, e, s) in enumerate(todo, 1):
            try:
                r = run_one(c, e, s)
                fh.write(json.dumps(r) + '\n'); fh.flush()
                print(f'[{i}/{len(todo)}] {c:5s} {os.path.basename(e):48s} '
                      f'nRMSE={r["nRMSE"]:.4f} {r["wall_s"]:.0f}s', flush=True)
            except Exception as ex:
                import traceback
                print(f'[{i}/{len(todo)}] {c} {e} ERROR {ex}', flush=True)
                traceback.print_exc()


def main_analyze():
    from scipy import stats
    rows = [json.loads(l) for l in open(OUT)]
    print(f'# n_rows={len(rows)}  프레임/런 중앙 {np.median([r["n_frames"] for r in rows]):.0f}')
    print(f'# ★셸 0_5 의 배-거리 중앙값 (런 평균): '
          f'{np.nanmean([r["shell05_ship_med"] for r in rows]):.1f} m  '
          f'⟹ "0_5 셸 = 배 근처" 는 거짓')

    def key(r):
        return (r['episode'], r['seed'])

    for rg in ('sparse', 'crowded'):
        sub = [r for r in rows if r['regime'] == rg]
        print(f'\n{"="*74}\n[{rg}]  n_ep={len(set(r["episode"] for r in sub))}')
        base = {key(r): r for r in sub if r['cell'] == 'SHIP'}
        # 절대 수준: 출하 팔의 배-거리 프로파일
        print('\n  출하 팔(SHIP) 배-거리 고리별 RMSE [m] / 격자면적비')
        for k in SHIP_KEYS:
            v = [r['ship'][k]['rmse'] for r in base.values()
                 if np.isfinite(r['ship'][k]['rmse'])]
            f = [r['ship'][k]['frac'] for r in base.values()]
            if v:
                print(f'    배-거리 {k:>8}  RMSE {np.mean(v):.4f}  면적 {100*np.mean(f):5.1f}%')
        for treat in ('NT', 'L1HI'):
            T = {key(r): r for r in sub if r['cell'] == treat}
            ks = sorted(set(T) & set(base))
            if not ks:
                continue
            a = np.array([T[k]['nRMSE'] for k in ks])
            b = np.array([base[k]['nRMSE'] for k in ks])
            w = int(np.sum(a < b))
            p = stats.binomtest(w, len(ks), 0.5, alternative='greater').pvalue
            print(f'\n  ★{treat} - SHIP  (n={len(ks)})')
            print(f'    전역 nRMSE {100*(a.mean()-b.mean())/b.mean():+.1f}%  승 {w}/{len(ks)}  p={p:.3g}')
            ar = np.array([T[k]['all_resid'] for k in ks])
            br = np.array([base[k]['all_resid'] for k in ks])
            wr = int(np.sum(ar < br))
            print(f'    GT-free 센서잔차 {100*(ar.mean()-br.mean())/br.mean():+.1f}%  승 {wr}/{len(ks)}')
            print(f'    {"배-거리":>10} {"면적%":>7} {"Δ%":>9} {"승/n":>8}   |'
                  f' {"관측거리":>9} {"면적%":>7} {"Δ%":>9} {"승/n":>8}')
            for i in range(max(len(SHIP_KEYS), len(OBS_KEYS))):
                cols = []
                for kind, keys in (('ship', SHIP_KEYS), ('obs', OBS_KEYS)):
                    if i >= len(keys):
                        cols.append(' ' * 36); continue
                    kk = keys[i]
                    va = np.array([T[k][kind][kk]['rmse'] for k in ks])
                    vb = np.array([base[k][kind][kk]['rmse'] for k in ks])
                    m = np.isfinite(va) & np.isfinite(vb)
                    if m.sum() < 3:
                        cols.append(' ' * 36); continue
                    fr = np.mean([base[k][kind][kk]['frac'] for k in ks])
                    d = 100 * (va[m].mean() - vb[m].mean()) / vb[m].mean()
                    ww = int(np.sum(va[m] < vb[m]))
                    cols.append(f'{kk:>10} {100*fr:>6.1f}% {d:>+8.1f}% {ww:>4}/{m.sum()}')
                print('    ' + cols[0] + '   | ' + cols[1])


if __name__ == '__main__':
    (main_run if (sys.argv[1:] or ['run'])[0] == 'run' else main_analyze)()
