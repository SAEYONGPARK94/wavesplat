"""BS-3 백지 재도출 — 기제와 경계. 원자료에서 직접 재계산.

⛔ 이 스크립트는 어떤 RESULT.md / 메모리 / .tex 의 수치도 읽지 않는다.
   읽는 것은 (a) `tmp/*/results.jsonl` 원자료, (b) `data_*/**.h5` 에피소드,
   (c) 코드(`wavesplat/`) 뿐이다.

사용: python3 tmp/BLANKSLATE_2026-08-12/bs3_recompute.py <section>
  geom     — 관측 개구 vs 평가 격자 기하 (h5 직접)
  regime   — 레짐 경계: 성분수 용량-반응, 배치 간 독립 재현
  shells   — 셸 이름 검증: 셸 -> 배로부터 거리 교차표 + 거리 재층화
  optim    — 옵티마이저 실패 (S4/S5 원자료)
  crb      — 정보 바닥 vs 실제
  aperture — B3 개구 곡선 원자료 재확인
  all
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))   # /home/vexit/wavesplat/wavesplat
os.chdir(REPO)


# ---------------------------------------------------------------- utilities
def load(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def sign_test(diffs, alt='less'):
    """부호검정. diffs = treat - control. alt='less' -> treat 가 더 작다(좋다)."""
    from scipy import stats
    d = np.asarray([x for x in diffs if np.isfinite(x) and x != 0.0])
    n = len(d)
    if n == 0:
        return 0, 0, 1.0
    k = int(np.sum(d < 0))          # treat 가 이긴 횟수
    if alt == 'less':
        p = stats.binomtest(k, n, 0.5, alternative='greater').pvalue
    else:
        p = stats.binomtest(k, n, 0.5, alternative='two-sided').pvalue
    return k, n, float(p)


def ep_key(e):
    """에피소드 파일명에서 (bf, ncomp, seed) 추출."""
    b = os.path.basename(e)
    parts = b.split('_')
    bf = parts[1]
    nc = int(parts[2].replace('comp', ''))
    sd = int(parts[3].replace('seed', ''))
    return bf, nc, sd


def paired(rows, cell_a, cell_b, key='nRMSE'):
    """같은 (episode, seed) 에서 cell_a - cell_b 짝짓기."""
    A = {(r['episode'], r['seed']): r for r in rows if r['cell'] == cell_a}
    B = {(r['episode'], r['seed']): r for r in rows if r['cell'] == cell_b}
    ks = sorted(set(A) & set(B))
    return ks, np.array([A[k][key] for k in ks]), np.array([B[k][key] for k in ks])


BATCHES = {
    'S1': 'tmp/S1_indep_sparse_2026-08-08/results.jsonl',
    'S2': 'tmp/S2_control_metric_2026-08-08/results.jsonl',
    'S3': 'tmp/S3_place_x_track_2026-08-08/results.jsonl',
    'S4': 'tmp/S4_phase_oracle_2026-08-08/results.jsonl',
    'S5': 'tmp/S5_window_loss_2026-08-08/results.jsonl',
    'S6': 'tmp/S6_l1_sweep_2026-08-08/results.jsonl',
    'S7': 'tmp/S7_confirm_n32_2026-08-10/results.jsonl',
    'S8': 'tmp/S8_deploy_confirm_2026-08-10/results.jsonl',
    'S9': 'tmp/S9_lr_x_l1_2026-08-11/results.jsonl',
    'X2': 'tmp/X2_no_learned_2026-08-11/results.jsonl',
}


# ---------------------------------------------------------------- A. geometry
def sec_geom():
    import h5py
    print('=' * 78)
    print('A. 관측 개구 vs 평가 격자 — h5 원자료 (LiDAR 실호출 파이프라인)')
    print('=' * 78)
    eps = [
        'data_s9_sparse_2026-08-11/episode_BF4_2comp_seed48_uniform_ra_rf_ds.h5',
        'data_s9_sparse_2026-08-11/episode_BF5_5comp_seed48_uniform_ra_rf_ds.h5',
        'data_s9_crowded_2026-08-11/episode_BF4_15comp_seed48_uniform_ra_rf_ds.h5',
        'data_s9_crowded_2026-08-11/episode_BF5_30comp_seed48_uniform_ra_rf_ds.h5',
        'data/episode_BF4_5comp_seed0_uniform.h5',
    ]
    for e in eps:
        if not os.path.exists(e):
            print(f'  MISS {e}')
            continue
        with h5py.File(e, 'r') as f:
            gx = f['frames/gt_grid_x'][:].ravel()
            gy = f['frames/gt_grid_y'][:].ravel()
            lp = f['frames/lidar_points']
            vm = f['frames/lidar_valid_mask'][:].astype(bool)
            nfr = lp.shape[0]
            npv = vm.sum(axis=1)
            # 전 프레임 유효점 반경 (배=원점 기준)
            rr, xx = [], []
            for i in range(0, nfr, 10):
                p = lp[i][vm[i]]
                if len(p):
                    rr.append(np.hypot(p[:, 0], p[:, 1]))
                    xx.append(p[:, 0])
            rr = np.concatenate(rr) if rr else np.array([np.nan])
            xx = np.concatenate(xx) if xx else np.array([np.nan])
            area = (gx.max() - gx.min()) * (gy.max() - gy.min())
            print(f'\n  {os.path.basename(e)}')
            print(f'    격자: x[{gx.min():.0f},{gx.max():.0f}] y[{gy.min():.0f},{gy.max():.0f}] '
                  f'= {gx.size}x{gy.size}, 면적 {area:,.0f} m^2')
            print(f'    프레임 {nfr}, 유효 LiDAR 점/프레임: '
                  f'median {np.median(npv):.0f}, min {npv.min()}, max {npv.max()} '
                  f'(raw slots {lp.shape[1]})')
            print(f'    점의 배-거리 r: p05 {np.percentile(rr,5):.1f}  med {np.median(rr):.1f}  '
                  f'p95 {np.percentile(rr,95):.1f}  max {rr.max():.1f} m')
            print(f'    점의 x    : p05 {np.percentile(xx,5):.1f}  med {np.median(xx):.1f}  '
                  f'p95 {np.percentile(xx,95):.1f}  max {xx.max():.1f} m')
            # 격자점 중 관측 30m/10m 내 비율 (프레임 0)
            Xg, Yg = np.meshgrid(gx, gy)
            p0 = lp[0][vm[0]]
            md = np.full(Xg.shape, np.inf)
            for j in range(len(p0)):
                md = np.minimum(md, np.hypot(Xg - p0[j, 0], Yg - p0[j, 1]))
            for R in (5, 10, 30, 50, 100):
                print(f'    격자면적 중 최근접관측 <{R:3d} m: {100*np.mean(md < R):5.1f}%')


# ---------------------------------------------------------------- B. regime
def sec_regime():
    print('=' * 78)
    print('B. 레짐 경계 — 성분수 용량-반응 + 배치 간 독립 재현')
    print('=' * 78)

    print('\nB-0. nRMSE 분모(Hs)가 성분수와 교락돼 있는가? (h5 attrs)')
    try:
        import h5py
        for d in ('data_s9_sparse_2026-08-11', 'data_s9_crowded_2026-08-11'):
            for e in sorted(glob.glob(f'{d}/episode_*seed48_*.h5')):
                with h5py.File(e, 'r') as f:
                    Hs = float(f['config'].attrs['Hs'])
                    gt = f['frames/gt_wave_field'][:20]
                    s = float(np.std(gt))
                bf, nc, sd = ep_key(e)
                print(f'    {bf} {nc:>2}comp : Hs(attr)={Hs:.2f}  4*std(GT)={4*s:.3f}  '
                      f'ratio={4*s/Hs:.3f}')
    except Exception as ex:
        print('   h5 읽기 실패:', ex)

    print('\nB-1. 출하 팔(AUTO / init_method=auto, adaptive_n) nRMSE — 성분수별')
    print('     ⚠ 배치마다 에피소드 집합이 다르다(신선 표본). 배치 안에서만 비교.')
    hdr = f'{"배치":<4} {"셀":<10} {"BF":<4} {"ncomp":>5} {"n":>4} {"med nRMSE":>10} {"IQR":>16}'
    print(hdr)
    print('-' * len(hdr))
    dose = defaultdict(list)
    for bn, path in BATCHES.items():
        if not os.path.exists(path):
            continue
        rows = load(path)
        cells = set(r['cell'] for r in rows)
        cell = 'AUTO' if 'AUTO' in cells else ('AUTO-NT' if 'AUTO-NT' in cells else sorted(cells)[0])
        g = defaultdict(list)
        for r in rows:
            if r['cell'] != cell or not np.isfinite(r.get('nRMSE', np.nan)):
                continue
            bf, nc, sd = ep_key(r['episode'])
            g[(bf, nc)].append(r['nRMSE'])
        for (bf, nc) in sorted(g, key=lambda t: (t[0], t[1])):
            v = np.array(g[(bf, nc)])
            print(f'{bn:<4} {cell:<10} {bf:<4} {nc:>5} {len(v):>4} {np.median(v):>10.4f} '
                  f'[{np.percentile(v,25):.4f},{np.percentile(v,75):.4f}]')
            dose[(bn, cell, bf)].append((nc, np.median(v), len(v)))
        print()

    print('\nB-2. 배치 안 성분수 단조성 (Spearman on episode-level, BF 고정)')
    print(f'{"배치":<4} {"셀":<10} {"BF":<4} {"ncomp 수준":<16} {"n_ep":>5} {"rho":>7} {"p":>10}')
    from scipy import stats
    mono = []
    for bn, path in BATCHES.items():
        if not os.path.exists(path):
            continue
        rows = load(path)
        cells = set(r['cell'] for r in rows)
        cell = 'AUTO' if 'AUTO' in cells else ('AUTO-NT' if 'AUTO-NT' in cells else sorted(cells)[0])
        bybf = defaultdict(lambda: ([], []))
        for r in rows:
            if r['cell'] != cell or not np.isfinite(r.get('nRMSE', np.nan)):
                continue
            bf, nc, sd = ep_key(r['episode'])
            bybf[bf][0].append(nc)
            bybf[bf][1].append(r['nRMSE'])
        for bf in sorted(bybf):
            nc, y = np.array(bybf[bf][0]), np.array(bybf[bf][1])
            lv = sorted(set(nc.tolist()))
            if len(lv) < 2:
                continue
            rho, p = stats.spearmanr(nc, y)
            print(f'{bn:<4} {cell:<10} {bf:<4} {str(lv):<16} {len(y):>5} {rho:>7.3f} {p:>10.2e}')
            mono.append((bn, bf, rho, p, len(y)))
    pos = [m for m in mono if m[2] > 0 and m[3] < 0.05]
    print(f'\n  → 성분수 ↑ = nRMSE ↑ 가 유의한 (배치,BF) 조합: {len(pos)}/{len(mono)}')

    print('\nB-3. 레짐 경계는 어디인가 — 2/5 vs 15/30 안에서도 갈리는가?')
    print('     (같은 배치·같은 BF 안에서 인접 수준 쌍의 중앙값 비)')
    for bn, path in BATCHES.items():
        if not os.path.exists(path):
            continue
        rows = load(path)
        cells = set(r['cell'] for r in rows)
        cell = 'AUTO' if 'AUTO' in cells else ('AUTO-NT' if 'AUTO-NT' in cells else sorted(cells)[0])
        g = defaultdict(list)
        for r in rows:
            if r['cell'] != cell or not np.isfinite(r.get('nRMSE', np.nan)):
                continue
            bf, nc, sd = ep_key(r['episode'])
            g[(bf, nc)].append(r['nRMSE'])
        bfs = sorted(set(b for b, _ in g))
        for bf in bfs:
            lv = sorted(nc for b, nc in g if b == bf)
            if len(lv) < 2:
                continue
            s = ' '.join(f'{a}->{b}: x{np.median(g[(bf,b)])/np.median(g[(bf,a)]):.2f}'
                         for a, b in zip(lv[:-1], lv[1:]))
            print(f'  {bn} {bf}: {s}')


# -------------------------------------------------------- B'. skill vs trivial
def sec_skill():
    """nRMSE = RMSE/Hs 이고 해면은 Hs = 4*sigma 이므로 **0 예측기의 nRMSE = 0.25**.
    (B-0 에서 4*std(GT)/Hs = 0.96-1.03 로 측정 확인됨.)
    ⟹ 어떤 nRMSE 든 자명 기준선 대비 설명분산으로 바꿀 수 있다:
         skill = 1 - (nRMSE / nRMSE_0)^2 ,  nRMSE_0 = std(GT)/Hs
    """
    import h5py
    print('=' * 78)
    print("B'. 자명 기준선(0 예측기) 대비 — 혼잡에서 추정기는 실제로 무엇을 버는가")
    print('=' * 78)

    print('\n  B\'-0. 에피소드별 자명 바닥 nRMSE_0 = std(GT)/Hs 실측')
    z = []
    for d in ('data_s9_sparse_2026-08-11', 'data_s9_crowded_2026-08-11'):
        for e in sorted(glob.glob(f'{d}/*.h5'))[:200]:
            with h5py.File(e, 'r') as f:
                Hs = float(f['config'].attrs['Hs'])
                g = f['frames/gt_wave_field']
                idx = np.linspace(0, g.shape[0] - 1, 12).astype(int)
                s = float(np.std(np.asarray(g[idx], float)))
            bf, nc, sd = ep_key(e)
            z.append((bf, nc, s / Hs))
    for nc in sorted(set(t[1] for t in z)):
        v = np.array([t[2] for t in z if t[1] == nc])
        print(f'    {nc:>2}comp: nRMSE_0 = {v.mean():.4f} +- {v.std():.4f}  (n_ep={len(v)})')
    NR0 = {nc: float(np.mean([t[2] for t in z if t[1] == nc]))
           for nc in sorted(set(t[1] for t in z))}

    print('\n  B\'-1. 출하 팔 skill = 1 - (nRMSE/nRMSE_0)^2  [1=완벽, 0=자명, <0=자명보다 나쁨]')
    print(f'  {"배치":<4} {"BF":<4} {"ncomp":>5} {"n":>4} {"med nRMSE":>10} '
          f'{"nRMSE_0":>8} {"skill":>7} {"자명대비 설명분산":>10}')
    for bn in ('S1', 'S3', 'S6', 'S7', 'S8', 'S9'):
        path = BATCHES[bn]
        if not os.path.exists(path):
            continue
        rows = load(path)
        g = defaultdict(list)
        for r in rows:
            if r['cell'] != 'AUTO' or not np.isfinite(r.get('nRMSE', np.nan)):
                continue
            bf, nc, sd = ep_key(r['episode'])
            g[(bf, nc)].append(r['nRMSE'])
        for (bf, nc) in sorted(g, key=lambda t: (t[0], t[1])):
            v = np.array(g[(bf, nc)])
            n0 = NR0.get(nc, 0.25)
            sk = 1 - (np.median(v) / n0) ** 2
            print(f'  {bn:<4} {bf:<4} {nc:>5} {len(v):>4} {np.median(v):>10.4f} '
                  f'{n0:>8.4f} {sk:>7.3f} {100*sk:>9.1f}%')
        print()

    print("  B'-2. 직접 측정된 재구성 지표 (S4 원자료, AUTO-NT 셀)")
    rows = load(BATCHES['S4'])
    print(f'  {"BF":<4} {"ncomp":>5} {"n":>3} {"Pearson_rho":>12} {"Var_Ratio":>10} '
          f'{"Skill_Score":>12} {"nRMSE":>8}')
    g = defaultdict(list)
    for r in rows:
        if r['cell'] != 'AUTO-NT':
            continue
        bf, nc, sd = ep_key(r['episode'])
        g[(bf, nc)].append(r)
    for k in sorted(g, key=lambda t: (t[0], t[1])):
        v = g[k]
        f = lambda key: np.median([x[key] for x in v if np.isfinite(x.get(key, np.nan))])
        print(f'  {k[0]:<4} {k[1]:>5} {len(v):>3} {f("Pearson_rho"):>12.4f} '
              f'{f("Variance_Ratio"):>10.4f} {f("Skill_Score"):>12.4f} {f("nRMSE"):>8.4f}')

    print('\n  B\'-3. 오차 분해: decorrelation vs 진폭 (S4, ρ 와 VR 에서)')
    print('       MSE/var_gt = 1 - 2*rho*sqrt(VR) + VR  (rho=상관, VR=분산비)')
    print('       decorr 몫 = 2*sqrt(VR)*(1-rho) ,  진폭 몫 = (1-sqrt(VR))^2')
    for k in sorted(g, key=lambda t: (t[0], t[1])):
        v = g[k]
        rho = np.median([x['Pearson_rho'] for x in v])
        vr = np.median([x['Variance_Ratio'] for x in v])
        dec = 2 * np.sqrt(vr) * (1 - rho)
        amp = (1 - np.sqrt(vr)) ** 2
        tot = dec + amp
        print(f'  {k[0]} {k[1]:>2}comp: rho={rho:.4f} VR={vr:.4f} | '
              f'MSE/var={tot:.4f}  decorr {100*dec/max(tot,1e-9):5.1f}%  '
              f'진폭 {100*amp/max(tot,1e-9):5.1f}%')


# ---------------------------------------------------------------- C. shells
def sec_shells():
    import h5py
    print('=' * 78)
    print('C. 셸 이름 검증 — "0_5" 는 최근접 *관측*까지 5 m 이지 배까지 5 m 가 아니다')
    print('   정의: wavesplat/run.py:355-366 (min_dist to nearest LiDAR return)')
    print('=' * 78)
    edges = [0, 5, 10, 20, 30, 50, 75, 100, 150, np.inf]
    keys = [f'{int(a)}_{"inf" if b==np.inf else int(b)}' for a, b in zip(edges[:-1], edges[1:])]

    eps = sorted(glob.glob('data_s9_crowded_2026-08-11/episode_BF4_15comp_seed4*.h5'))[:4] + \
        sorted(glob.glob('data_s9_sparse_2026-08-11/episode_BF4_2comp_seed4*.h5'))[:4]
    acc = {k: [] for k in keys}     # 셸별 배-거리 표본
    frac = {k: [] for k in keys}
    for e in eps:
        with h5py.File(e, 'r') as f:
            gx = f['frames/gt_grid_x'][:].ravel()
            gy = f['frames/gt_grid_y'][:].ravel()
            lp = f['frames/lidar_points']
            vm = f['frames/lidar_valid_mask'][:].astype(bool)
            Xg, Yg = np.meshgrid(gx, gy)
            Rship = np.hypot(Xg, Yg)        # 배 = 원점
            for i in range(0, lp.shape[0], 25):
                p = lp[i][vm[i]]
                if len(p) == 0:
                    continue
                md = np.full(Xg.shape, np.inf)
                for j in range(len(p)):
                    md = np.minimum(md, np.hypot(Xg - p[j, 0], Yg - p[j, 1]))
                for k, (a, b) in zip(keys, zip(edges[:-1], edges[1:])):
                    m = (md >= a) & (md < b)
                    frac[k].append(float(np.mean(m)))
                    if np.any(m):
                        acc[k].append(Rship[m])
    print(f'\n  에피소드 {len(eps)}개 x 매 25프레임 표본')
    print(f'{"셸(관측거리)":<14} {"격자면적%":>9} {"배-거리 p10":>12} {"중앙값":>9} {"p90":>8} {"최대":>8}')
    print('-' * 66)
    for k in keys:
        if not acc[k]:
            print(f'{k:<14} {100*np.mean(frac[k]):>8.2f}%      (비어있음)')
            continue
        v = np.concatenate(acc[k])
        print(f'{k:<14} {100*np.mean(frac[k]):>8.2f}% {np.percentile(v,10):>12.1f} '
              f'{np.median(v):>9.1f} {np.percentile(v,90):>8.1f} {v.max():>8.1f}')
    print('\n  → 셸 이름은 배로부터의 거리가 아니다. 배-거리로 재층화가 필요하다.')

    print('\nC-2. 역방향 — 배-거리 고리별 "최근접 관측까지 거리" 분포')
    ship_edges = [0, 20, 30, 50, 75, 100, 150, np.inf]
    for e in eps[:2]:
        with h5py.File(e, 'r') as f:
            gx = f['frames/gt_grid_x'][:].ravel()
            gy = f['frames/gt_grid_y'][:].ravel()
            lp = f['frames/lidar_points']
            vm = f['frames/lidar_valid_mask'][:].astype(bool)
            Xg, Yg = np.meshgrid(gx, gy)
            Rship = np.hypot(Xg, Yg)
            p = lp[0][vm[0]]
            md = np.full(Xg.shape, np.inf)
            for j in range(len(p)):
                md = np.minimum(md, np.hypot(Xg - p[j, 0], Yg - p[j, 1]))
        print(f'  {os.path.basename(e)}')
        for a, b in zip(ship_edges[:-1], ship_edges[1:]):
            m = (Rship >= a) & (Rship < b)
            if not np.any(m):
                continue
            print(f'    배-거리 [{a:>3},{"inf" if b==np.inf else int(b):>4}) '
                  f'면적{100*np.mean(m):5.1f}%  최근접관측거리 med {np.median(md[m]):6.1f} m')


def sec_shell_effect():
    """셸 RMSE 원자료로 SUPPORT-TRADEOFF 를 재계산 + 셸->배거리 지도로 재해석."""
    print('=' * 78)
    print('C-3. SUPPORT-TRADEOFF 재계산 — 처치가 셸별로 어떻게 갈리는가')
    print('=' * 78)
    keys = ['0_5', '5_10', '10_20', '20_30', '30_50', '50_75', '75_100', '100_150', '150_inf']
    tests = [
        ('S7', 'AUTO-NT-L1-1e4', 'AUTO-NT'),
        ('S8', 'AUTO-L1-1e4', 'AUTO'),
        ('S8', 'AUTO-NT', 'AUTO'),
        ('S9', 'AUTO-L1', 'AUTO'),
        ('S9', 'AUTO-NT', 'AUTO'),
    ]
    for bn, ta, tb in tests:
        path = BATCHES[bn]
        if not os.path.exists(path):
            continue
        rows = load(path)
        for regime in ('sparse', 'crowded'):
            rr = [r for r in rows if r.get('regime') == regime]
            ks, a, b = paired(rr, ta, tb, 'nRMSE')
            if len(ks) == 0:
                continue
            k, n, p = sign_test(a - b)
            print(f'\n  [{bn}] {ta} - {tb}  ({regime}, n={n})')
            print(f'    전역 nRMSE : {100*(np.mean(a)-np.mean(b))/np.mean(b):+.1f}%  '
                  f'승 {k}/{n}  p={p:.2e}')
            print(f'    {"셸":<10} {"면적%":>7} {"Δ%":>8} {"승/n":>8}')
            A = {(r['episode'], r['seed']): r for r in rr if r['cell'] == ta}
            B = {(r['episode'], r['seed']): r for r in rr if r['cell'] == tb}
            for sk in keys:
                va, vb, fa = [], [], []
                for key in ks:
                    x = A[key].get('shell_rmse') or {}
                    y = B[key].get('shell_rmse') or {}
                    fr = B[key].get('shell_frac') or {}
                    if sk in x and sk in y and np.isfinite(x[sk]) and np.isfinite(y[sk]):
                        va.append(x[sk])
                        vb.append(y[sk])
                        fa.append(fr.get(sk, np.nan))
                if len(va) < 5:
                    continue
                va, vb = np.array(va), np.array(vb)
                kk, nn, _ = sign_test(va - vb)
                print(f'    {sk:<10} {100*np.nanmean(fa):>6.2f}% '
                      f'{100*(np.mean(va)-np.mean(vb))/np.mean(vb):>+7.1f}% {kk:>4}/{nn}')
            # GT-free 잔차도
            ksr, ar, br = paired(rr, ta, tb, 'all_resid')
            if len(ksr):
                kr, nr, pr = sign_test(ar - br)
                print(f'    all_resid(GT-free): {100*(np.mean(ar)-np.mean(br))/np.mean(br):+.1f}% '
                      f'승 {kr}/{nr} p={pr:.2e}')


# ---------------------------------------------------------------- D. optimizer
def sec_optim():
    print('=' * 78)
    print('D. 옵티마이저 실패 — 진실에서 출발시켜도 나빠지는가 (S4/S5 원자료)')
    print('=' * 78)
    for bn in ('S4', 'S5'):
        path = BATCHES[bn]
        if not os.path.exists(path):
            continue
        rows = load(path)
        cells = sorted(set(r['cell'] for r in rows))
        print(f'\n  [{bn}] 셀: {cells}')
        for regime in ('sparse', 'crowded'):
            rr = [r for r in rows if r.get('regime') == regime]
            print(f'  -- {regime} --')
            for key in ('nRMSE', 'avg_obs_loss', 'window_resid', 'all_resid'):
                if not any(key in r for r in rr):
                    continue
                line = []
                for c in cells:
                    v = [r[key] for r in rr if r['cell'] == c and key in r
                         and np.isfinite(r[key])]
                    if v:
                        line.append(f'{c}={np.median(v):.5g}')
                print(f'    {key:<14} ' + '  '.join(line))
            # 핵심 대조: 진실에서 0 step vs 진실에서 출발 후 최적화
            for a, b in (('ORA-WPHI-NT', 'ORA-WPHI-0STEP'),
                         ('ORA-WPHI-NOREG', 'ORA-WPHI-0STEP'),
                         ('ORA-WPHI-NT', 'ORA-WPHI-NOREG'),
                         ('AUTO-NT-NOL1', 'AUTO-NT'),
                         ('AUTO-NT-NOREG', 'AUTO-NT')):
                ks, va, vb = paired(rr, a, b, 'avg_obs_loss')
                if len(ks) == 0:
                    continue
                k, n, p = sign_test(va - vb)
                ratio = np.median(va) / max(np.median(vb), 1e-12)
                ks2, na, nb = paired(rr, a, b, 'nRMSE')
                k2, n2, p2 = sign_test(na - nb)
                print(f'    {a} vs {b}: obs_loss x{ratio:.2f} '
                      f'(a<b {k}/{n}, p={p:.2g}) | nRMSE Δ={np.mean(na)-np.mean(nb):+.5f} '
                      f'({k2}/{n2}, p={p2:.2g})')


# ---------------------------------------------------------------- E. crb
def sec_crb():
    print('=' * 78)
    print('E. 정보 바닥(CRB) vs 실제 — ⚠기하 확인 필수')
    print('=' * 78)
    cands = ['results/direction1/crb_results.json',
             'tmp/rx2_init_2026-08-05/CON_crb.log',
             'scripts/d1_crb.py']
    for c in cands:
        print(f'  {c}: {"있음" if os.path.exists(c) else "없음"}')
    p = 'results/direction1/crb_results.json'
    if os.path.exists(p):
        d = json.load(open(p))
        print('\n  crb_results.json 최상위 키:', list(d)[:20])
        print(json.dumps(d, indent=2)[:3000])


# ---------------------------------------------------------------- F. aperture
def sec_aperture():
    print('=' * 78)
    print('F. 개구 곡선 원자료 재확인 — tmp/B3_regime_position_2026-08-12/')
    print('=' * 78)
    p = 'tmp/B3_regime_position_2026-08-12/aperture_curve.json'
    if not os.path.exists(p):
        print('  없음')
        return
    d = json.load(open(p))
    print('  최상위 키:', list(d) if isinstance(d, dict) else type(d))
    print(json.dumps(d, indent=2)[:1500])


SECTIONS = dict(geom=sec_geom, regime=sec_regime, skill=sec_skill, shells=sec_shells,
                shelleff=sec_shell_effect, optim=sec_optim, crb=sec_crb,
                aperture=sec_aperture)

if __name__ == '__main__':
    want = sys.argv[1:] or ['all']
    if want == ['all']:
        want = list(SECTIONS)
    for w in want:
        SECTIONS[w]()
        print()
