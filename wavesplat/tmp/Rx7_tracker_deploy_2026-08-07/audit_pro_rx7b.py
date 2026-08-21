"""PRO 2차 — 기제 강도 / GT-free 게이트 / 작동점 / 검정력."""
import json, math, os, random, statistics as st
from collections import defaultdict

H = os.path.dirname(os.path.abspath(__file__))
R = [json.loads(l) for l in open(os.path.join(H, 'results.jsonl')) if l.strip()]
NULLS = {d['episode']: d for d in json.load(open(os.path.join(H, 'nulls_rx7.json')))}
LAD = ['AUTO', 'AUTO-LR10', 'AUTO-LR100', 'AUTO-LR1000', 'AUTO-NOTRACK']
LRV = {'AUTO': 5e-3, 'AUTO-LR10': 5e-4, 'AUTO-LR100': 5e-5, 'AUTO-LR1000': 5e-6,
       'AUTO-NOTRACK': 0.0}
by = {(r['cell'], r['episode'], r['seed']): r for r in R}
EPS = sorted({r['episode'] for r in R}); SEEDS = sorted({r['seed'] for r in R})


def median(v):
    v = sorted(v); n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def sign_p(k, n):
    if n == 0: return 1.0
    lo = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(lo + 1)) / 2 ** n)


def M(c, e, k='nRMSE'):
    return median([by[(c, e, s)][k] for s in SEEDS])


kept = [e for e in EPS if M('AUTO', e) < NULLS[e]['zero_null_tracking_nRMSE_mean']]
n = len(kept); NEED = 11
print(f'n={n} need={NEED}\n')

# ---------------- A. 분할 무결성: 관측 격자비 f 를 셀 무관하게 복원하는가
print('=== A. 여집합 분할 무결성 — RMSE_tot² = f·obs² + (1−f)·unobs² 로 f 역산 ===')
print('  (nRMSE = RMSE/Hs, Hs = init_rmse/init_nrmse 로 확인)')
hs_ok = all(abs(by[(c, e, s)]['init_rmse'] / by[(c, e, s)]['init_nrmse']
                - NULLS[e]['Hs_true']) < 1e-3 for c in LAD for e in EPS for s in SEEDS)
print(f'  Hs 정규화 확인: {hs_ok}')
for e in kept[:6] + kept[-2:]:
    hs = NULLS[e]['Hs_true']
    fs = []
    for c in LAD:
        tot = M(c, e) * hs; o = M(c, e, 'avg_RMSE_observed'); u = M(c, e, 'avg_RMSE_unobserved')
        den = u ** 2 - o ** 2
        fs.append((u ** 2 - tot ** 2) / den if den else float('nan'))
    print(f'  {e[8:-20]:22s} f_hat(5셀) = ' + ' '.join(f'{x:.3f}' for x in fs)
          + f'   폭={max(fs)-min(fs):.4f}')
allf = []
for e in kept:
    hs = NULLS[e]['Hs_true']
    for c in LAD:
        tot = M(c, e) * hs; o = M(c, e, 'avg_RMSE_observed'); u = M(c, e, 'avg_RMSE_unobserved')
        allf.append((u ** 2 - tot ** 2) / (u ** 2 - o ** 2))
print(f'  전체 65 f_hat: 중앙 {median(allf):.3f}  범위 [{min(allf):.3f},{max(allf):.3f}]')
print('  ⟹ 셀에 무관한 상수(관측 기하는 궤적으로 고정) 여야 함. 편차는 프레임평균 vs 제곱합의 근사오차.')

# ---------------- B. 센서 노이즈 바닥 대비
print('\n=== B. 관측영역 오차가 센서 노이즈 바닥 아래로 갔는가 (σ_range=0.02 m) ===')
for c in LAD:
    lv = median([M(c, e, 'avg_RMSE_observed') for e in kept])
    print(f'  {c:13s} obsRMSE 블록중앙 = {lv:.4f} m  = 노이즈바닥의 {lv/0.02:.1f}배')

# ---------------- C. 짝별 비 (Rx-6 의 2.12x 와 종류 대응)
print('\n=== C. 짝별 비의 중앙 (⚠집계의 비 아님) ===')
for c in LAD[:4]:
    ro = [abs(M(c, e, 'avg_RMSE_observed') - M('AUTO-NOTRACK', e, 'avg_RMSE_observed'))
          / abs(M(c, e) - M('AUTO-NOTRACK', e)) for e in kept if M(c, e) != M('AUTO-NOTRACK', e)]
    ru = [abs(M(c, e, 'avg_RMSE_unobserved') - M('AUTO-NOTRACK', e, 'avg_RMSE_unobserved'))
          / abs(M(c, e) - M('AUTO-NOTRACK', e)) for e in kept if M(c, e) != M('AUTO-NOTRACK', e)]
    print(f'  {c:13s} |Δobs|/|ΔnRMSE| 중앙={median(ro):5.2f}x   |Δunobs|/|ΔnRMSE| 중앙={median(ru):4.2f}x')

# ---------------- D. 에피소드 내 사다리 순위상관
print('\n=== D. 에피소드 내 5-rung 순위 (관측 vs nRMSE, 관측 vs 미관측) ===')
def sp(a, b):
    def rk(x):
        o = sorted(range(len(x)), key=lambda i: x[i]); r = [0.0] * len(x)
        for j, i in enumerate(o): r[i] = j + 1
        return r
    ra, rb = rk(a), rk(b); m = len(a); ma = mb = (m + 1) / 2
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den
s1 = [sp([M(c, e, 'avg_RMSE_observed') for c in LAD], [M(c, e) for c in LAD]) for e in kept]
s2 = [sp([M(c, e, 'avg_RMSE_observed') for c in LAD],
         [M(c, e, 'avg_RMSE_unobserved') for c in LAD]) for e in kept]
print(f'  Spearman(obs, nRMSE)  중앙={median(s1):+.2f}  음수 {sum(1 for x in s1 if x<0)}/{n}'
      f'  = −1.0 인 에피소드 {sum(1 for x in s1 if x<-0.99)}')
print(f'  Spearman(obs, unobs)  중앙={median(s2):+.2f}  음수 {sum(1 for x in s2 if x<0)}/{n}'
      f'  = −1.0 인 에피소드 {sum(1 for x in s2 if x<-0.99)}')

# ---------------- E. GT-free 선택기 후보
print('\n=== E. GT-free 선택기 ===')
base = {e: M('AUTO-NOTRACK', e) for e in kept}
oracle = [min(LAD, key=lambda c: M(c, e)) for e in kept]
og = [M(oracle[i], e) - base[e] for i, e in enumerate(kept)]
print(f'  (i) 오라클 셀선택(천장): medΔ={median(og):+.5f} 개선 {sum(1 for x in og if x<0)}/{n}'
      f'  선택분포={{c: oracle.count(c) for c in LAD if oracle.count(c)}}'.replace('{c: oracle.count(c) for c in LAD if oracle.count(c)}', str({c: oracle.count(c) for c in LAD if oracle.count(c)})))
print(f'      ⟹ 이 5셀 격자 전체가 바닥 대비 벌 수 있는 최대 = {median(og):+.5f} nRMSE')

print('  (ii) 규칙 "|obs(c) − obs(바닥)| ≤ τ 인 가장 큰 lr" (τ 는 GT-free 로 잴 수 있음)')
for tau in (0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.15, 0.20):
    pick = []
    for e in kept:
        ok = [c for c in LAD if abs(M(c, e, 'avg_RMSE_observed')
                                    - M('AUTO-NOTRACK', e, 'avg_RMSE_observed')) <= tau]
        pick.append(max(ok, key=lambda c: LRV[c]) if ok else 'AUTO-NOTRACK')
    d = [M(pick[i], e) - base[e] for i, e in enumerate(kept)]
    w = sum(1 for x in d if x < 0)
    dist = {c: pick.count(c) for c in LAD if pick.count(c)}
    print(f'   τ={tau:.3f}: medΔ={median(d):+.6f} 개선 {w:2d}/{n} p={sign_p(w,n):.4f}  {dist}')

print('  (iii) 규칙 "obs(c) ≥ obs(바닥) − τ" 대신 nRMSE 최소 셀을 obs 로 맞히는가')
hit = sum(1 for i, e in enumerate(kept)
          if oracle[i] == max(LAD, key=lambda c: M(c, e, 'avg_RMSE_observed')))
print(f'   argmax_obs 가 오라클 셀과 일치: {hit}/{n}')

# ---------------- F. 검정력: 부호검정 MDE (시뮬)
print('\n=== F. 부호검정 MDE (관측 Δ 분포 재표집, 11/13 문턱, 80% 검정력) ===')
random.seed(0)
for c in ('AUTO-LR100', 'AUTO-LR1000'):
    d = [M(c, e) - base[e] for e in kept]
    dc = [x - median(d) for x in d]
    lo, hi = 0.0, 0.05
    for _ in range(40):
        mid = (lo + hi) / 2; hits = 0
        for _ in range(4000):
            samp = [random.choice(dc) - mid for _ in range(n)]
            if sum(1 for x in samp if x < 0) >= NEED: hits += 1
        if hits / 4000 < 0.80: lo = mid
        else: hi = mid
    gap = median([M('AUTO', e) - base[e] for e in kept])
    print(f'  {c:13s} sd={st.stdev(d):.5f}  MDE≈{hi:.5f} nRMSE = 추적기 여유({gap:.5f})의 {hi/gap*100:.1f}%'
          f'   관측치 {median(d):+.5f}')

# ---------------- G. LR1000 퇴화 여부
print('\n=== G. AUTO-LR1000 이 바닥과 퇴화했는가 (파라미터 수준) ===')
bit = sum(1 for e in EPS for s in SEEDS
          if by[('AUTO-LR1000', e, s)]['w'] == by[('AUTO-NOTRACK', e, s)]['w']
          and by[('AUTO-LR1000', e, s)]['phi'] == by[('AUTO-NOTRACK', e, s)]['phi'])
print(f'  w·phi 비트일치 (ep,seed): {bit}/48')
drv = []
for e in EPS:
    for s in SEEDS:
        a = by[('AUTO-LR1000', e, s)]; b = by[('AUTO-NOTRACK', e, s)]
        num = math.sqrt(sum((x - y) ** 2 for x, y in zip(a['w'], b['w'])))
        den = math.sqrt(sum(y ** 2 for y in b['w']))
        drv.append(num / den if den else 0)
print(f'  실현 진폭 구동 |Δw|/|w| 중앙 = {median(drv)*100:.3f}%  (Rx-5 표: AUTO급 96%)')
for c in LAD[:4]:
    dd = []
    for e in EPS:
        for s in SEEDS:
            a = by[(c, e, s)]; b = by[('AUTO-NOTRACK', e, s)]
            num = math.sqrt(sum((x - y) ** 2 for x, y in zip(a['w'], b['w'])))
            den = math.sqrt(sum(y ** 2 for y in b['w']))
            dd.append(num / den if den else 0)
    print(f'   {c:13s} 실현 진폭 구동 중앙 = {median(dd)*100:7.3f}%')

# ---------------- H. 정확한 자릿수 (Rx-6 대조용)
print('\n=== H. 정밀 수치 ===')
for k, lab in [('nRMSE', 'nRMSE'), ('avg_RMSE_observed', '관측'), ('avg_RMSE_unobserved', '미관측')]:
    d = [M('AUTO', e, k) - base[e] if k == 'nRMSE'
         else M('AUTO', e, k) - M('AUTO-NOTRACK', e, k) for e in kept]
    print(f'  AUTO−바닥 {lab:6s}: med={median(d):+.8f}  (Rx-6 미관측 보고치 ±0.04004)')
print(f'  정확 p(13/13) = {sign_p(13,13):.3e}   p(11/13) = {sign_p(11,13):.4f}')
