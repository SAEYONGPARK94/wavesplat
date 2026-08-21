"""R2-P1 프로브 — `SUPPORT-TRADEOFF` 라는 이름이 오칭인지 판정한다.

BS-3 는 두 자(배-거리 / 최근접-관측거리)를 **각각 1D 로만** 쟀다. 두 자는 이 기하에서
강하게 공선(共線)이므로(P1 R1 §3: 배-고리 10_20 의 98.7% 가 관측셸 0_5 안), 1D 프로파일
두 개로는 "손상의 원인이 지지(support) 결손인가, 센서로부터의 거리(lever arm)인가"를
가를 수 없다.

이 프로브는 **같은 프레임에서 네 개의 자를 동시에** 누적해 공선을 깬다.

  R1  배-거리 고리 (8)                      — BS-3 재현용
  R2  최근접 관측거리 셸 (9)                — BS-3 재현용 (= support 자)
  R3  ★2D 교차표: 배-거리 × 관측거리        — 부분 기울기 분리 (결정적)
  R4  ★FOV 콘 자연실험: 같은 배-거리에서 콘 안(관측됨) vs 콘 밖(영원히 미관측)
  R5  국소 관측밀도 rho_R = #returns within R m  (R=10)
  R6  이방성: 진행방향(x) 대 횡방향(|y|) — 파수오차 지레팔 가설의 서명
  ★D  방향 자 (P1 vs P2 판정): s = x·cosθ + y·sinθ (평균 파향 축 = **경과시간** 자) ·
      q = 직교축 · 같은 배-거리 고리 안에서 s 상위/하위 · dobs 상위/하위.
      θ 는 에피소드마다 −41°~+44° 로 회전하므로 s-등고선은 돌고 r-등고선은 안 돈다
      ⟹ "경과시간(P2)" 과 "지지/지레팔(P1)" 이 **처음으로 갈린다**.
      ⛔R1 §7-3 의 "평균 파향 +4.9°, 범위 [+3.1°,+9.6°]" 은 `wave_components` 열 순서
      오독 산물(실제 열 = [A, omega, kx, ky, phi, Q]) — R2 에서 철회.

`wavesplat.run.rmse_observed` 를 monkeypatch 하는 방식은 BS-3 와 동일(순수 가산, 저장소 무수정).

사용:
    python3 tmp/DEBATE_blankslate_2026-08-12/r2_p1_support_probe.py run
    python3 tmp/DEBATE_blankslate_2026-08-12/r2_p1_support_probe.py analyze
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

OUT = os.path.join(HERE, 'r2_support.jsonl')

SHIP_EDGES = [10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0, np.inf]
SHIP_KEYS = [f'{int(a)}_{"inf" if b == np.inf else int(b)}'
             for a, b in zip(SHIP_EDGES[:-1], SHIP_EDGES[1:])]
OBS_EDGES = [0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, np.inf]
OBS_KEYS = [f'{int(a)}_{"inf" if b == np.inf else int(b)}'
            for a, b in zip(OBS_EDGES[:-1], OBS_EDGES[1:])]
# 2D 교차표용 거친 관측거리 구간
OBSC_EDGES = [0.0, 5.0, 15.0, 40.0, np.inf]
OBSC_KEYS = ['o0_5', 'o5_15', 'o15_40', 'o40_inf']
RHO_R = 10.0
RHO_EDGES = [0, 1, 2, 4, 8, np.inf]          # 0 / 1 / 2-3 / 4-7 / 8+
RHO_KEYS = ['rho0', 'rho1', 'rho2_3', 'rho4_7', 'rho8+']
XB_EDGES = [10.0, 30.0, 60.0, 100.0, np.inf]
XB_KEYS = ['x10_30', 'x30_60', 'x60_100', 'x100+']
YB_EDGES = [0.0, 20.0, 50.0, np.inf]
YB_KEYS = ['y0_20', 'y20_50', 'y50+']
S_EDGES = [-np.inf, 20.0, 40.0, 70.0, 110.0, 160.0, np.inf]
S_KEYS = ['s<20', 's20_40', 's40_70', 's70_110', 's110_160', 's160+']
Q_EDGES = [0.0, 20.0, 50.0, 100.0, np.inf]
Q_KEYS = ['q0_20', 'q20_50', 'q50_100', 'q100+']
YS_EDGES = [-np.inf, -50.0, -20.0, 0.0, 20.0, 50.0, np.inf]
YS_KEYS = ['y<-50', 'y-50_-20', 'y-20_0', 'y0_20', 'y20_50', 'y50+']

ACC = []
THETA = {'deg': 0.0}


def wave_theta(path):
    """에너지가중 평균 전파방향 [deg].  열 = [A, omega, kx, ky, phi, Q]."""
    import h5py
    with h5py.File(path, 'r') as h:
        wc = h['config/wave_components'][:]
    A, kx, ky = wc[:, 0], wc[:, 2], wc[:, 3]
    E = A ** 2
    km = np.hypot(kx, ky)
    return float(np.degrees(np.arctan2(np.sum(E * ky / km), np.sum(E * kx / km))))


def _acc(store, key, err2, mask):
    n = int(mask.sum())
    if n:
        store[key] = (float(np.sum(err2[mask])), n)


def install_patch():
    import wavesplat.run as WR
    from scipy.spatial import cKDTree
    orig = WR.rmse_observed
    if getattr(orig, '_r2p1', False):
        return

    def patched(pred, gt, gx, gy, ox, oy, min_dist=10.0):
        if min_dist == 5.0:
            gx = np.asarray(gx).ravel(); gy = np.asarray(gy).ravel()
            Xg, Yg = np.meshgrid(gx, gy)
            ox = np.asarray(ox).ravel(); oy = np.asarray(oy).ravel()
            dship = np.hypot(Xg, Yg)
            # 반환점이 프레임당 ~50 개뿐이므로 (4800 x n_obs) 완전행렬이 KD-tree 보다 빠르다
            D = np.hypot(Xg.ravel()[:, None] - ox[None, :],
                         Yg.ravel()[:, None] - oy[None, :])
            dobs = D.min(axis=1).reshape(Xg.shape)
            rho = (D < RHO_R).sum(axis=1).reshape(Xg.shape)
            # FOV 콘: 반환점 방위각의 [p2.5, p97.5] 를 콘으로 본다
            az_o = np.degrees(np.arctan2(oy, ox))
            lo, hi = np.percentile(az_o, [2.5, 97.5])
            az_g = np.degrees(np.arctan2(Yg, Xg))
            incone = (az_g >= lo) & (az_g <= hi)
            # 반환점 반경 상한 (콘 안이라도 이보다 멀면 미관측)
            r_o = np.hypot(ox, oy)
            r_hi = float(np.percentile(r_o, 97.5))

            e2 = (pred - gt) ** 2
            f = {}
            ship_b = [(dship >= a) & (dship < b)
                      for a, b in zip(SHIP_EDGES[:-1], SHIP_EDGES[1:])]
            obs_b = [(dobs >= a) & (dobs < b)
                     for a, b in zip(OBS_EDGES[:-1], OBS_EDGES[1:])]
            obsc_b = [(dobs >= a) & (dobs < b)
                      for a, b in zip(OBSC_EDGES[:-1], OBSC_EDGES[1:])]
            rho_b = [(rho >= a) & (rho < b)
                     for a, b in zip(RHO_EDGES[:-1], RHO_EDGES[1:])]
            xb_b = [(Xg >= a) & (Xg < b) for a, b in zip(XB_EDGES[:-1], XB_EDGES[1:])]
            yb_b = [(np.abs(Yg) >= a) & (np.abs(Yg) < b)
                    for a, b in zip(YB_EDGES[:-1], YB_EDGES[1:])]

            for k, m in zip(SHIP_KEYS, ship_b):
                _acc(f, 'S:' + k, e2, m)
            for k, m in zip(OBS_KEYS, obs_b):
                _acc(f, 'O:' + k, e2, m)
            for k, m in zip(RHO_KEYS, rho_b):
                _acc(f, 'R:' + k, e2, m)
            for ks, ms in zip(SHIP_KEYS, ship_b):
                for ko, mo in zip(OBSC_KEYS, obsc_b):
                    _acc(f, f'X:{ks}|{ko}', e2, ms & mo)
                _acc(f, f'C:{ks}|in', e2, ms & incone)
                _acc(f, f'C:{ks}|out', e2, ms & (~incone))
                _acc(f, f'C:{ks}|in_near', e2, ms & incone & (dship <= r_hi))
                _acc(f, f'C:{ks}|in_far', e2, ms & incone & (dship > r_hi))
            for kx, mx in zip(XB_KEYS, xb_b):
                for ky, my in zip(YB_KEYS, yb_b):
                    _acc(f, f'A:{kx}|{ky}', e2, mx & my)

            # ── 방향 자 (P1 vs P2) ────────────────────────────────────────
            th = np.radians(THETA['deg'])
            Sax = Xg * np.cos(th) + Yg * np.sin(th)
            Qax = -Xg * np.sin(th) + Yg * np.cos(th)
            s_b = [(Sax >= a) & (Sax < b) for a, b in zip(S_EDGES[:-1], S_EDGES[1:])]
            q_b = [(np.abs(Qax) >= a) & (np.abs(Qax) < b)
                   for a, b in zip(Q_EDGES[:-1], Q_EDGES[1:])]
            ys_b = [(Yg >= a) & (Yg < b) for a, b in zip(YS_EDGES[:-1], YS_EDGES[1:])]
            for k, m in zip(S_KEYS, s_b):
                _acc(f, 'D:' + k, e2, m)
            for k, m in zip(Q_KEYS, q_b):
                _acc(f, 'Q:' + k, e2, m)
            for k1, m1 in zip(S_KEYS, s_b):
                for k2, m2 in zip(Q_KEYS, q_b):
                    _acc(f, f'SQ:{k1}|{k2}', e2, m1 & m2)
            for kr, mr in zip(SHIP_KEYS, ship_b):
                if not mr.any():
                    continue
                smed = np.median(Sax[mr]); omed = np.median(dobs[mr])
                _acc(f, f'RS:{kr}|s_lo', e2, mr & (Sax <= smed))
                _acc(f, f'RS:{kr}|s_hi', e2, mr & (Sax > smed))
                _acc(f, f'RD:{kr}|d_lo', e2, mr & (dobs <= omed))
                _acc(f, f'RD:{kr}|d_hi', e2, mr & (dobs > omed))
            for kx, mx in zip(XB_KEYS, xb_b):
                for ky, my in zip(YS_KEYS, ys_b):
                    _acc(f, f'XY:{kx}|{ky}', e2, mx & my)

            f['_meta'] = dict(
                n_obs=int(len(ox)), az_lo=float(lo), az_hi=float(hi), r_hi=r_hi,
                theta=THETA['deg'],
                shell05_ship_med=(float(np.median(dship[dobs < 5.0]))
                                  if np.any(dobs < 5.0) else float('nan')),
                # 관측 중심(centroid)까지의 거리로 잰 손상 자 (R6 보조)
                cx=float(ox.mean()), cy=float(oy.mean()),
                mse_all=float(np.mean(e2)),
            )
            ACC.append(f)
        return orig(pred, gt, gx, gy, ox, oy, min_dist)

    patched._r2p1 = True
    WR.rmse_observed = patched


CELLS = {
    'SHIP': {'lambda_l1': 1e-4, 'tracking_lr': 0.005},
    'NT':   {'lambda_l1': 1e-4, 'tracking_lr': 0.0},
}

SPARSE = [f'data_s9_sparse_2026-08-11/episode_{bf}_{nc}comp_seed{s}_uniform_ra_rf_ds.h5'
          for bf in ('BF4', 'BF5') for nc in (2, 5) for s in (56, 57, 58)]
CROWDED = [f'data_s9_crowded_2026-08-11/episode_{bf}_{nc}comp_seed{s}_uniform_ra_rf_ds.h5'
           for bf in ('BF4', 'BF5') for nc in (15, 30) for s in (52, 53, 54)]
EPISODES = CROWDED + SPARSE      # 혼잡이 부호 12/12 로 깨끗하므로 먼저 돌린다
SEED = 1901                      # BS-3 와 동일 분석 시드 (짝 비교를 유지하기 위해)


def regime(ep):
    return 'sparse' if 'sparse' in ep else 'crowded'


def ncomp(ep):
    return int(ep.split('comp')[0].split('_')[-1])


def run_one(cell, ep, seed):
    import run_rx2b as R
    R.DATA = REPO
    install_patch()
    from wavesplat.run import run_estimation

    ACC.clear()
    THETA['deg'] = wave_theta(os.path.join(REPO, ep))
    cfg = R.cell_config('AUTO', ep)
    cfg.update(CELLS[cell])
    cfg['seed'] = seed
    t0 = time.time()
    r = run_estimation(os.path.join(REPO, ep), cfg)
    fr = r['frame_results']

    keys = set()
    for f in ACC:
        keys |= set(k for k in f if k != '_meta')
    bins = {}
    for k in sorted(keys):
        sse = sum(f[k][0] for f in ACC if k in f)
        n = sum(f[k][1] for f in ACC if k in f)
        bins[k] = [sse, n, sum(1 for f in ACC if k in f)]

    meta = {kk: float(np.nanmean([f['_meta'][kk] for f in ACC]))
            for kk in ('n_obs', 'az_lo', 'az_hi', 'r_hi', 'shell05_ship_med',
                       'cx', 'cy', 'mse_all', 'theta')}
    return dict(
        cell=cell, episode=ep, seed=seed, regime=regime(ep), ncomp=ncomp(ep),
        nRMSE=float(np.nanmean([f['nRMSE'] for f in fr if 'nRMSE' in f])),
        all_resid=float(np.nanmean([f['obs_resid_rms'] for f in fr
                                    if 'obs_resid_rms' in f])),
        bins=bins, meta=meta, n_frames=len(ACC), wall_s=time.time() - t0,
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
                print(f'[{i}/{len(todo)}] {c:5s} {os.path.basename(e):46s} '
                      f'nRMSE={r["nRMSE"]:.4f} {r["wall_s"]:.0f}s', flush=True)
            except Exception as ex:
                import traceback
                print(f'[{i}/{len(todo)}] {c} {e} ERROR {ex}', flush=True)
                traceback.print_exc()


# ── 집계 ────────────────────────────────────────────────────────────────────
def _rmse(b):
    return np.sqrt(b[0] / b[1]) if b and b[1] else np.nan


def main_analyze():
    from scipy import stats
    rows = [json.loads(l) for l in open(OUT)]
    print(f'# n_rows={len(rows)}')
    m = rows[0]['meta']
    print(f"# 프레임 평균 반환점 {np.mean([r['meta']['n_obs'] for r in rows]):.1f} 개 · "
          f"콘 [{np.mean([r['meta']['az_lo'] for r in rows]):.1f}, "
          f"{np.mean([r['meta']['az_hi'] for r in rows]):.1f}]° · "
          f"r_hi {np.mean([r['meta']['r_hi'] for r in rows]):.1f} m · "
          f"관측중심 ({np.mean([r['meta']['cx'] for r in rows]):.1f}, "
          f"{np.mean([r['meta']['cy'] for r in rows]):.1f})")

    def key(r):
        return (r['episode'], r['seed'])

    for rg in ('sparse', 'crowded'):
        sub = [r for r in rows if r['regime'] == rg]
        base = {key(r): r for r in sub if r['cell'] == 'SHIP'}
        T = {key(r): r for r in sub if r['cell'] == 'NT'}
        ks = sorted(set(T) & set(base))
        if not ks:
            continue
        print(f'\n{"="*96}\n[{rg}]  NT − SHIP   n_ep={len(ks)}')

        def line(label, bk, area_den=4800.0):
            a = np.array([_rmse(T[k]['bins'].get(bk)) for k in ks])
            b = np.array([_rmse(base[k]['bins'].get(bk)) for k in ks])
            nn = np.array([base[k]['bins'].get(bk, [0, 0, 0])[1]
                           / max(base[k]['n_frames'], 1) for k in ks])
            msk = np.isfinite(a) & np.isfinite(b)
            if msk.sum() < 3:
                print(f'  {label:>26}  (n<3)')
                return None
            rel = 100 * (a[msk].mean() - b[msk].mean()) / b[msk].mean()
            medrel = 100 * np.median((a[msk] - b[msk]) / b[msk])
            w = int(np.sum(a[msk] > b[msk]))       # NT 더 나쁨 = 손상
            p = stats.binomtest(w, int(msk.sum()), 0.5).pvalue
            print(f'  {label:>26} area{100*nn.mean()/area_den:>5.1f}% '
                  f'S={b[msk].mean():.4f} T={a[msk].mean():.4f} '
                  f'dRMSE={a[msk].mean()-b[msk].mean():+.5f} '
                  f'rel={rel:+8.1f}% med={medrel:+8.1f}% 손상{w:>3}/{msk.sum()} p={p:.3g}')
            return rel

        print('\n R1 배-거리 고리')
        for k in SHIP_KEYS:
            line(k, 'S:' + k)
        print('\n R2 최근접-관측거리 셸 (= support 자)')
        for k in OBS_KEYS:
            line(k, 'O:' + k)
        print('\n R5 국소 관측밀도 rho_10')
        for k in RHO_KEYS:
            line(k, 'R:' + k)
        print('\n ★R3 2D 교차표  배-거리 × 관측거리')
        for ksh in SHIP_KEYS:
            for ko in OBSC_KEYS:
                line(f'{ksh} × {ko}', f'X:{ksh}|{ko}')
        print('\n ★R4 FOV 콘 자연실험 (같은 배-거리, 콘 안 vs 콘 밖)')
        for ksh in SHIP_KEYS:
            for side in ('in', 'out'):
                line(f'{ksh} × cone-{side}', f'C:{ksh}|{side}')
        print('\n R6 이방성  x밴드 × |y|밴드')
        for kx in XB_KEYS:
            for ky in YB_KEYS:
                line(f'{kx} × {ky}', f'A:{kx}|{ky}')
        print('\n ★D1 s 밴드 (평균 파향 축 = 경과시간 자)')
        for k in S_KEYS:
            line(k, 'D:' + k)
        print('\n ★D2 |q| 밴드 (파향 직교축)')
        for k in Q_KEYS:
            line(k, 'Q:' + k)
        print('\n ★D3 2D  s × |q|   (경과시간 고정하고 횡거리만 바꾼다)')
        for a in S_KEYS:
            for b_ in Q_KEYS:
                line(f'{a} × {b_}', f'SQ:{a}|{b_}')
        print('\n ★D4 같은 배-거리 고리 안에서  s 하위 vs 상위 (경과시간 대비)')
        for k in SHIP_KEYS:
            for h in ('s_lo', 's_hi'):
                line(f'{k} × {h}', f'RS:{k}|{h}')
        print('\n ★D5 같은 배-거리 고리 안에서  dobs 하위 vs 상위 (지지 대비)')
        for k in SHIP_KEYS:
            for h in ('d_lo', 'd_hi'):
                line(f'{k} × {h}', f'RD:{k}|{h}')
        print('\n D6 x밴드 × 부호 있는 y밴드')
        for kx in XB_KEYS:
            for ky in YS_KEYS:
                line(f'{kx} × {ky}', f'XY:{kx}|{ky}')


if __name__ == '__main__':
    (main_run if (sys.argv[1:] or ['run'])[0] == 'run' else main_analyze)()
