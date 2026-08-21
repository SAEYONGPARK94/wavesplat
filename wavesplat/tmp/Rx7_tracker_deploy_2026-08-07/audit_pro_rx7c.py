"""PRO 3차 — 관측영역의 공간적 정체 / 위상 구동 / 불일치원리(discrepancy) 정지규칙 브래킷."""
import json, math, os, glob, statistics as st
import numpy as np, h5py

H = os.path.dirname(os.path.abspath(__file__))
R = [json.loads(l) for l in open(os.path.join(H, 'results.jsonl')) if l.strip()]
NULLS = {d['episode']: d for d in json.load(open(os.path.join(H, 'nulls_rx7.json')))}
LAD = ['AUTO', 'AUTO-LR10', 'AUTO-LR100', 'AUTO-LR1000', 'AUTO-NOTRACK']
by = {(r['cell'], r['episode'], r['seed']): r for r in R}
EPS = sorted({r['episode'] for r in R}); SEEDS = sorted({r['seed'] for r in R})
med = lambda v: float(np.median(v))
def M(c, e, k='nRMSE'): return med([by[(c, e, s)][k] for s in SEEDS])
def sign_p(k, n):
    lo = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(lo + 1)) / 2 ** n)
kept = [e for e in EPS if M('AUTO', e) < NULLS[e]['zero_null_tracking_nRMSE_mean']]
n = len(kept)

# ---------------- 1. 관측영역이 공간적으로 어디인가 (전 16 에피소드 직접 계산)
print('=== 1. 관측영역의 공간 정체 (min_dist=10 m, 원자료 직접) ===')
fr_all, xs_all = [], []
for f in sorted(glob.glob('/home/vexit/wavesplat/wavesplat/data_indep4_2026-08-07/*.h5')):
    h = h5py.File(f, 'r')
    gx = h['frames/gt_grid_x'][:]; gy = h['frames/gt_grid_y'][:]
    P = h['frames/lidar_points']; V = h['frames/lidar_valid_mask']
    Xg, Yg = np.meshgrid(gx, gy)
    fr, xs = [], []
    for t in range(0, P.shape[0], 40):
        m = V[t][:]; ox = P[t][m][:, 0]; oy = P[t][m][:, 1]
        if len(ox) == 0: continue
        d = np.min(np.sqrt((Xg[..., None] - ox) ** 2 + (Yg[..., None] - oy) ** 2), axis=-1)
        obs = d < 10.0
        fr.append(obs.mean()); xs.append(Xg[obs])
    fr_all.append(np.median(fr)); xs_all.append(np.concatenate(xs))
    h.close()
X = np.concatenate(xs_all)
print(f'  관측 격자비: 에피소드중앙의 중앙 = {np.median(fr_all):.4f}  범위 [{min(fr_all):.4f},{max(fr_all):.4f}]')
print(f'  격자 전체 x = [10, 250] m,  관측셀 x 분위 '
      f'5%={np.percentile(X,5):.0f} 50%={np.percentile(X,50):.0f} 95%={np.percentile(X,95):.0f} max={X.max():.0f} m')
for v in (11.3, 2.0):
    print(f'   ⟹ 선속 {v} m/s 기준 조우 리드타임 = '
          f'{np.percentile(X,5)/v:.1f}–{np.percentile(X,95)/v:.1f} s (중앙 {np.percentile(X,50)/v:.1f} s)')
print('  ⚠단 채점량은 각 프레임의 **나우캐스트**이지 리드타임 예보가 아니다.')

# ---------------- 2. 위상/진폭 실현 구동 사다리
print('\n=== 2. 실현 구동 (GT 없이, AUTO-NOTRACK 동결 기준) ===')
print(f"  {'cell':13s} {'|Δw|/|w|':>10s} {'rms|Δφ| [rad]':>15s}")
for c in LAD[:4]:
    dw, dp = [], []
    for e in EPS:
        for s in SEEDS:
            a = by[(c, e, s)]; b = by[('AUTO-NOTRACK', e, s)]
            w0 = np.array(b['w']); w1 = np.array(a['w'])
            p0 = np.array(b['phi']); p1 = np.array(a['phi'])
            dw.append(np.linalg.norm(w1 - w0) / max(np.linalg.norm(w0), 1e-12))
            dphi = np.angle(np.exp(1j * (p1 - p0)))
            dp.append(float(np.sqrt(np.mean(dphi ** 2))))
    print(f'  {c:13s} {med(dw)*100:9.3f}% {med(dp):15.4f}')
print('  ⚠lr 10배 축소(AUTO->LR10)에서 진폭 구동이 오히려 커진다 = Adam 포화(Rx-5 6c 와 같은 종류).')

# ---------------- 3. 불일치원리 정지규칙 — 관측잔차 문턱 브래킷
print('\n=== 3. ★GT-free 정지규칙: "관측영역 오차를 Phase-1 수준의 몇 % 아래로 내리면 손해인가" ===')
print(f"  {'cell':13s} {'obs [m]':>9s} {'obs/obs_base 짝중앙':>20s} {'ΔnRMSE':>10s} {'승/13':>7s} {'p':>8s}")
for c in LAD:
    rr = [M(c, e, 'avg_RMSE_observed') / M('AUTO-NOTRACK', e, 'avg_RMSE_observed') for e in kept]
    d = [M(c, e) - M('AUTO-NOTRACK', e) for e in kept]
    w = sum(1 for x in d if x < 0)
    print(f'  {c:13s} {med([M(c,e,"avg_RMSE_observed") for e in kept]):9.4f} {med(rr):20.3f}'
          f' {med(d):+10.5f} {w:4d}/13 {sign_p(w,13):8.4f}')

print('\n  에피소드별 영교차: ΔnRMSE(obs비) 가 0 을 지나는 obs/obs_base 를 선형보간')
cross = []
for e in kept:
    pts = sorted([(M(c, e, 'avg_RMSE_observed') / M('AUTO-NOTRACK', e, 'avg_RMSE_observed'),
                   M(c, e) - M('AUTO-NOTRACK', e)) for c in LAD])
    z = None
    for (r1, d1), (r2, d2) in zip(pts, pts[1:]):
        if d1 * d2 < 0: z = r1 + (r2 - r1) * (-d1) / (d2 - d1)
    if z is not None: cross.append(z)
print(f'  영교차 존재 {len(cross)}/13  중앙 obs/obs_base = {med(cross):.3f}'
      f'  IQR [{np.percentile(cross,25):.3f}, {np.percentile(cross,75):.3f}]')
print('  ⟹ 배포 규칙 후보: **추적기가 관측영역 잔차를 Phase-1 잔차의 '
      f'{med(cross)*100:.0f}% 아래로 내리면 정지** (전부 GT-free 로 측정 가능).')

# 그 규칙을 사다리 위에서 실제로 굴려본다
print('\n  규칙 굴리기: "obs(c)/obs(base) ≥ ρ 인 가장 큰 lr"')
LRV = {'AUTO': 5e-3, 'AUTO-LR10': 5e-4, 'AUTO-LR100': 5e-5, 'AUTO-LR1000': 5e-6, 'AUTO-NOTRACK': 0.0}
for rho in (0.99, 0.97, 0.95, 0.90, 0.85, 0.80, 0.70):
    pick = []
    for e in kept:
        ok = [c for c in LAD if M(c, e, 'avg_RMSE_observed')
              / M('AUTO-NOTRACK', e, 'avg_RMSE_observed') >= rho]
        pick.append(max(ok, key=lambda c: LRV[c]) if ok else 'AUTO-NOTRACK')
    d = [M(pick[i], e) - M('AUTO-NOTRACK', e) for i, e in enumerate(kept)]
    o = [M(pick[i], e, 'avg_RMSE_observed') - M('AUTO-NOTRACK', e, 'avg_RMSE_observed')
         for i, e in enumerate(kept)]
    w = sum(1 for x in d if x < 0)
    print(f'   ρ={rho:.2f}: nRMSE medΔ={med(d):+.6f} 개선 {w:2d}/13 p={sign_p(w,13):.4f} | '
          f'관측 medΔ={med(o):+.5f} 개선 {sum(1 for x in o if x<0)}/13 | '
          f'{ {c: pick.count(c) for c in LAD if pick.count(c)} }')

# ---------------- 4. 추적기가 이기는 에피소드가 하나라도 있는가 (on/off 게이트의 천장)
print('\n=== 4. on/off 게이트의 천장 ===')
wins = [e for e in kept if M('AUTO', e) < M('AUTO-NOTRACK', e)]
print(f'  AUTO 가 바닥을 이기는 에피소드: {len(wins)}/13  ⟹ 에피소드별 on/off 게이팅의 이득 천장 = 0')
w2 = [e for e in kept if M('AUTO', e, 'avg_RMSE_unobserved') < M('AUTO-NOTRACK', e, 'avg_RMSE_unobserved')]
print(f'  미관측에서 AUTO 가 이기는 에피소드: {len(w2)}/13')

# ---------------- 5. 관측/미관측 트레이드오프의 기울기
print('\n=== 5. 트레이드오프 기울기 (에피소드별, 관측이득 1 당 미관측손실) ===')
for c in LAD[:4]:
    sl = [(M(c, e, 'avg_RMSE_unobserved') - M('AUTO-NOTRACK', e, 'avg_RMSE_unobserved'))
          / -(M(c, e, 'avg_RMSE_observed') - M('AUTO-NOTRACK', e, 'avg_RMSE_observed')) for e in kept]
    print(f'  {c:13s} 기울기 중앙 = {med(sl):+.4f}  (>0 = 트레이드오프)  양수 {sum(1 for x in sl if x>0)}/13')
