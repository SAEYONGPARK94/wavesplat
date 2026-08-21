"""PRO 독립 재도출 — Rx-7. analyze_rx7.py 를 읽지 않고 PREREG §3/§4/§5 만 보고 새로 작성.

규약(§3): 시드중앙 -> 짝별 부호 있는 Δ -> Δ 의 중앙, 정확 양측 부호검정, 동점은 n 에서 제외.
⛔블록중앙의 차 금지 · ⛔절댓값 판정식 금지 · ⛔집계의 비 금지(짝별 비의 중앙만, 명시 표기).
"""
import json, math, os, itertools
from collections import defaultdict

H = os.path.dirname(os.path.abspath(__file__))
R = [json.loads(l) for l in open(os.path.join(H, 'results.jsonl')) if l.strip()]
NULLS = {d['episode']: d for d in json.load(open(os.path.join(H, 'nulls_rx7.json')))}
CELLS = ['AUTO', 'AUTO-DUP', 'AUTO-LR10', 'AUTO-LR100', 'AUTO-LR1000', 'AUTO-NOTRACK']
LR = {'AUTO': 5e-3, 'AUTO-DUP': 5e-3, 'AUTO-LR10': 5e-4, 'AUTO-LR100': 5e-5,
      'AUTO-LR1000': 5e-6, 'AUTO-NOTRACK': 0.0}
NEED = {16: 13, 15: 12, 14: 12, 13: 11, 12: 10}


def median(v):
    v = sorted(v); n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def sign_p(k, n):          # exact two-sided sign test
    if n == 0: return 1.0
    lo = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(lo + 1)) / 2 ** n)


def spearman(a, b):
    def rk(x):
        o = sorted(range(len(x)), key=lambda i: x[i]); r = [0.0] * len(x); i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and x[o[j + 1]] == x[o[i]]: j += 1
            for k in range(i, j + 1): r[o[k]] = (i + j) / 2.0 + 1
            i = j + 1
        return r
    ra, rb = rk(a), rk(b); n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else float('nan')


# ---------------------------------------------------------------- integrity
err = [r for r in R if 'error' in r]
print(f'rows={len(R)}  errors={len(err)}')
by = defaultdict(list)
for r in R: by[(r['cell'], r['episode'], r['seed'])].append(r)
assert all(len(v) == 1 for v in by.values()), 'duplicate (cell,ep,seed)'
EPS = sorted({r['episode'] for r in R}); SEEDS = sorted({r['seed'] for r in R})
print(f'cells={len(CELLS)} eps={len(EPS)} seeds={SEEDS}  -> expect {6*16*3}')

SCAL = [k for k, v in R[0].items() if isinstance(v, (int, float)) or v is None
        or isinstance(v, str)]
mism = [k for k in SCAL if any(
    by[('AUTO', e, s)][0].get(k) != by[('AUTO-DUP', e, s)][0].get(k)
    for e in EPS for s in SEEDS if k not in ('job_id', 'cell', 'elapsed_ms', 'wall_s'))]
print(f'AUTO-DUP 비트일치: 검사필드 {len(SCAL)-4}  불일치필드 {mism}')
arrmis = sum(1 for e in EPS for s in SEEDS for k in ('kx', 'ky', 'w', 'phi')
             if by[('AUTO', e, s)][0][k] != by[('AUTO-DUP', e, s)][0][k])
print(f'AUTO-DUP 배열(kx,ky,w,phi) 불일치: {arrmis}')

p1bad = sum(1 for e in EPS for s in SEEDS if len({
    (round(by[(c, e, s)][0]['init_nrmse'], 12), by[(c, e, s)][0]['auto_selected'],
     by[(c, e, s)][0]['N'], by[(c, e, s)][0]['ensemble_k']) for c in CELLS}) > 1)
print(f'Phase-1 불변 위반 (ep,seed): {p1bad}/48')
# NOTRACK 은 lr=0 ⟹ w 가 Phase-1 결과여야 함. 그 셀만 원자 위치 정렬 확인
kmis = sum(1 for e in EPS for s in SEEDS for c in CELLS
           if by[(c, e, s)][0]['kx'] != by[('AUTO-NOTRACK', e, s)][0]['kx'])
print(f'원자 (kx) 셀간 불일치: {kmis}/{6*48}   (w_phi 는 kx 를 안 건드림)')

# ---------------------------------------------------------------- null rule
def M(c, e, k='nRMSE'):
    return median([by[(c, e, s)][0][k] for s in SEEDS])

kept = [e for e in EPS if M('AUTO', e) < NULLS[e]['zero_null_tracking_nRMSE_mean']]
excl = [e for e in EPS if e not in kept]
n = len(kept); need = NEED[n]
print(f'\n널 배제 {len(excl)}: {[e[8:-20] for e in excl]}  -> n={n}, 필요 {need}')

print('\n=== 절대 수준 (블록중앙 = 시드중앙의 중앙, 서술 전용) ===')
print(f"{'cell':14s} {'lr':>8s} {'nRMSE':>8s} {'obsRMSE':>9s} {'unobsRMSE':>10s} "
      f"{'널통과':>7s} {'m>1':>5s}")
for c in CELLS:
    npass = sum(1 for e in kept if M(c, e) < NULLS[e]['zero_null_tracking_nRMSE_mean'])
    m1 = sum(1 for e in kept if M(c, e, 'm_mean') > 1.0)
    print(f'{c:14s} {LR[c]:8.0e} {median([M(c,e) for e in kept]):8.4f} '
          f"{median([M(c,e,'avg_RMSE_observed') for e in kept]):9.4f} "
          f"{median([M(c,e,'avg_RMSE_unobserved') for e in kept]):10.4f} "
          f'{npass:4d}/{n:<3d} {m1:4d}')

# ---------------------------------------------------------------- paired test
GOODNEG = defaultdict(lambda: True); GOODNEG['Pearson_rho'] = False


def pair(a, b, k='nRMSE', blk=None):
    blk = blk or kept
    d = [(e, M(a, e, k) - M(b, e, k)) for e in blk]
    ties = sum(1 for _, x in d if x == 0)
    d = [(e, x) for e, x in d if x != 0]
    if not d: return dict(med=None, w=0, n=0, p=1.0, ties=ties, per={})
    w = sum(1 for _, x in d if (x < 0) == GOODNEG[k])
    return dict(med=median([x for _, x in d]), w=w, n=len(d),
                p=sign_p(w, len(d)), ties=ties, per=dict(d))


print('\n=== PRIMARY-A: 후보 − AUTO-NOTRACK (nRMSE) ===')
res = {}
for c in ['AUTO-LR10', 'AUTO-LR100', 'AUTO-LR1000']:
    r = pair(c, 'AUTO-NOTRACK'); res[c] = r
    print(f"  {c:13s} medΔ={r['med']:+.5f}  개선 {r['w']:2d}/{r['n']}  p={r['p']:.4f}  ties={r['ties']}")
order = sorted(res, key=lambda c: res[c]['p'])
keep = False
for i, c in enumerate(order):
    a = 0.05 / (3 - i); r = res[c]
    ok = r['p'] <= a and r['med'] < 0 and r['w'] >= NEED[r['n']]
    keep |= ok
    print(f"  Holm {c:13s} p={r['p']:.4f} vs α={a:.4f} -> {'PASS' if ok else 'fail'}")
    if r['p'] > a: break
print(f"  ⟹ {'KEEP-REDUCED' if keep else 'SHIP-OFF'}")

print('\n=== PRIMARY-B: AUTO − AUTO-NOTRACK, 관측/미관측 ===')
o = pair('AUTO', 'AUTO-NOTRACK', 'avg_RMSE_observed')
u = pair('AUTO', 'AUTO-NOTRACK', 'avg_RMSE_unobserved')
print(f"  관측   medΔ={o['med']:+.5f}  개선 {o['w']}/{o['n']} p={o['p']:.4f}")
print(f"  미관측 medΔ={u['med']:+.5f}  개선 {u['w']}/{u['n']} p={u['p']:.4f}")
ob = o['med'] < 0 and o['w'] >= NEED[o['n']]
ow = o['med'] > 0 and (o['n'] - o['w']) >= NEED[o['n']]
uw = u['med'] > 0 and (u['n'] - u['w']) >= NEED[u['n']]
print(f"  ⟹ {'OVERFIT' if ob and uw else 'MISSPECIFIED' if ow else 'INCONCLUSIVE-B'}")

# ---------------------------------------------------------------- ladder
print('\n=== 용량-반응 사다리 (각 셀 − 바닥) ===')
print(f"{'cell':13s} {'nRMSE Δ':>10s} {'승':>6s} {'p':>7s} | {'obs Δ':>10s} {'승':>6s} {'p':>7s}"
      f" | {'unobs Δ':>10s} {'승':>6s} {'p':>7s}")
for c in ['AUTO', 'AUTO-LR10', 'AUTO-LR100', 'AUTO-LR1000']:
    a = pair(c, 'AUTO-NOTRACK')
    b = pair(c, 'AUTO-NOTRACK', 'avg_RMSE_observed')
    d = pair(c, 'AUTO-NOTRACK', 'avg_RMSE_unobserved')
    print(f"{c:13s} {a['med']:+10.5f} {a['w']:3d}/{a['n']:<2d} {a['p']:7.4f} |"
          f" {b['med']:+10.5f} {b['w']:3d}/{b['n']:<2d} {b['p']:7.4f} |"
          f" {d['med']:+10.5f} {d['w']:3d}/{d['n']:<2d} {d['p']:7.4f}")

print('\n  인접 rung 짝검정 (관측영역, 낮은 lr − 높은 lr; Δ>0 = lr 낮추면 관측 악화):')
lad = ['AUTO', 'AUTO-LR10', 'AUTO-LR100', 'AUTO-LR1000', 'AUTO-NOTRACK']
for x, y in zip(lad[1:], lad[:-1]):
    r = pair(x, y, 'avg_RMSE_observed')
    print(f'   {x:13s} − {y:13s} medΔ={r["med"]:+.5f}  악화 {r["n"]-r["w"]}/{r["n"]}  p={r["p"]:.4f}')
mono = sum(1 for e in kept if all(
    M(lad[i], e, 'avg_RMSE_observed') < M(lad[i + 1], e, 'avg_RMSE_observed')
    for i in range(4)))
print(f'  ★관측영역 5-rung 완전 단조 에피소드: {mono}/{n}')
mono2 = sum(1 for e in kept if all(
    M(lad[i], e, 'avg_RMSE_unobserved') > M(lad[i + 1], e, 'avg_RMSE_unobserved')
    for i in range(4)))
print(f'   미관측영역 5-rung 완전 단조 에피소드: {mono2}/{n}')

# seed-level consistency (검정 아님, 서술)
sl = [(c, sum(1 for e in kept for s in SEEDS
              if by[(c, e, s)][0]['avg_RMSE_observed']
              < by[('AUTO-NOTRACK', e, s)][0]['avg_RMSE_observed']))
      for c in ['AUTO', 'AUTO-LR10', 'AUTO-LR100', 'AUTO-LR1000']]
print(f'  (ep,seed) 39개 수준 관측영역 개선 건수: ' + '  '.join(f'{c}={k}/39' for c, k in sl))

# ---------------------------------------------------------------- overfit gap
print('\n=== 과적합 간극: 짝별 (미관측Δ − 관측Δ) [차, 비 아님] ===')
for c in ['AUTO', 'AUTO-LR10', 'AUTO-LR100', 'AUTO-LR1000']:
    g = [(M(c, e, 'avg_RMSE_unobserved') - M('AUTO-NOTRACK', e, 'avg_RMSE_unobserved'))
         - (M(c, e, 'avg_RMSE_observed') - M('AUTO-NOTRACK', e, 'avg_RMSE_observed'))
         for e in kept]
    w = sum(1 for x in g if x > 0)
    print(f'  {c:13s} med={median(g):+.5f}  간극>0 {w}/{n}  p={sign_p(w,n):.4f}')

print('\n  ★짝별 비의 중앙 (⚠집계의 비 아님; 에피소드별 비를 만든 뒤 중앙):')
for c in ['AUTO', 'AUTO-LR10', 'AUTO-LR100']:
    rr = [abs(M(c, e, 'avg_RMSE_observed') - M('AUTO-NOTRACK', e, 'avg_RMSE_observed'))
          / abs(M(c, e, 'avg_RMSE_unobserved') - M('AUTO-NOTRACK', e, 'avg_RMSE_unobserved'))
          for e in kept]
    print(f'   {c:13s} |관측이득|/|미관측손실| 중앙 = {median(rr):.2f}x')

# ---------------------------------------------------------------- GT-free gate
print('\n=== ★GT-free 게이트 후보 ===')
print(' (1) 에피소드별: 관측영역 Δ(AUTO−바닥) 이 nRMSE Δ 를 예측하는가')
dobs = [M('AUTO', e, 'avg_RMSE_observed') - M('AUTO-NOTRACK', e, 'avg_RMSE_observed')
        for e in kept]
dn = [M('AUTO', e) - M('AUTO-NOTRACK', e) for e in kept]
dun = [M('AUTO', e, 'avg_RMSE_unobserved') - M('AUTO-NOTRACK', e, 'avg_RMSE_unobserved')
       for e in kept]
print(f'   Spearman(Δobs, ΔnRMSE)   = {spearman(dobs, dn):+.3f}')
print(f'   Spearman(Δobs, Δunobs)   = {spearman(dobs, dun):+.3f}')
print(f'   Spearman(Δunobs, ΔnRMSE) = {spearman(dun, dn):+.3f}')

print(' (2) 셀 선택기: 에피소드마다 관측영역 최소 셀 / 최대 셀을 고르면 nRMSE 는?')
CAND = ['AUTO', 'AUTO-LR10', 'AUTO-LR100', 'AUTO-LR1000', 'AUTO-NOTRACK']
pickmin = [min(CAND, key=lambda c: M(c, e, 'avg_RMSE_observed')) for e in kept]
pickmax = [max(CAND, key=lambda c: M(c, e, 'avg_RMSE_observed')) for e in kept]
print(f'   argmin_obs 선택 분포: {dict((c, pickmin.count(c)) for c in CAND if pickmin.count(c))}')
print(f'   argmax_obs 선택 분포: {dict((c, pickmax.count(c)) for c in CAND if pickmax.count(c))}')
sel = [M(pickmin[i], e) - M('AUTO-NOTRACK', e) for i, e in enumerate(kept)]
w = sum(1 for x in sel if x < 0)
print(f'   argmin_obs 선택기 − 바닥: medΔ={median(sel):+.5f} 개선 {w}/{n} p={sign_p(w,n):.4f}'
      '   ⟹ 관측잔차 최소화 = 반예측적')

print(' (3) ★배포 게이트: 짝별 Δobs 크기가 문턱 이하이면 추적기 유지 (LR100 대상)')
for thr in (0.02, 0.04, 0.06, 0.08):
    keepep = [e for e in kept
              if abs(M('AUTO-LR100', e, 'avg_RMSE_observed')
                     - M('AUTO-NOTRACK', e, 'avg_RMSE_observed')) <= thr]
    if not keepep: continue
    dd = [M('AUTO-LR100', e) - M('AUTO-NOTRACK', e) for e in keepep]
    print(f'   thr={thr:.2f}: 통과 {len(keepep)}/{n}  그 안 nRMSE medΔ={median(dd):+.5f} '
          f'개선 {sum(1 for x in dd if x<0)}/{len(dd)}')

# ---------------------------------------------------------------- power / MDE
print('\n=== 검정력 / MDE ===')
import statistics as st
for name, a, b in [('AUTO-LR1000 vs 바닥', 'AUTO-LR1000', 'AUTO-NOTRACK'),
                   ('AUTO-LR100 vs 바닥', 'AUTO-LR100', 'AUTO-NOTRACK'),
                   ('AUTO vs 바닥', 'AUTO', 'AUTO-NOTRACK')]:
    d = [M(a, e) - M(b, e) for e in kept]
    sd = st.stdev(d)
    mde = (0.8416 + 1.96) * sd / math.sqrt(len(d))
    print(f'  {name:22s} med={median(d):+.5f} sd={sd:.5f}  MDE(80%,정규근사)={mde:.5f}')
gap = [M('AUTO', e) - M('AUTO-NOTRACK', e) for e in kept]
print(f'  추적기 총 여유(AUTO−바닥) 짝중앙 = {median(gap):+.5f}')

# ---------------------------------------------------------------- folds / LOO
print('\n=== 설정 폴드 n=4 (⛔"유의"·효과크기 금지) + LOO ===')
folds = defaultdict(list)
for e in kept: folds[e.split('_seed')[0]].append(e)
for c in ['AUTO-LR1000', 'AUTO-LR100']:
    fd = [median([M(c, e) - M('AUTO-NOTRACK', e) for e in ee])
          for _, ee in sorted(folds.items())]
    print(f'  {c:13s} 폴드Δ = ' + ' '.join(f'{x:+.5f}' for x in fd)
          + f'  ({sum(1 for x in fd if x<0)}/{len(fd)} 개선)')
for c in ['AUTO-LR1000']:
    fails = 0
    for drop in kept:
        blk = [e for e in kept if e != drop]
        r = pair(c, 'AUTO-NOTRACK', blk=blk)
        if not (r['med'] < 0 and r['w'] >= NEED[r['n']]): fails += 1
    print(f'  LOO({c}): {n} 회 중 문턱 미달 {fails}')
for k in ('avg_RMSE_observed',):
    fails = 0
    for drop in kept:
        blk = [e for e in kept if e != drop]
        r = pair('AUTO', 'AUTO-NOTRACK', k, blk=blk)
        if not (r['med'] < 0 and r['w'] >= NEED[r['n']]): fails += 1
    print(f'  LOO(PRIMARY-B 관측영역): {n} 회 중 문턱 미달 {fails}')

print('\n=== 미배제 n=16 감도 ===')
for lbl, a, b, k in [('PRIMARY-A LR1000', 'AUTO-LR1000', 'AUTO-NOTRACK', 'nRMSE'),
                     ('PRIMARY-B 관측', 'AUTO', 'AUTO-NOTRACK', 'avg_RMSE_observed'),
                     ('PRIMARY-B 미관측', 'AUTO', 'AUTO-NOTRACK', 'avg_RMSE_unobserved')]:
    r = pair(a, b, k, blk=EPS)
    print(f"  {lbl:20s} medΔ={r['med']:+.5f} {r['w']}/{r['n']} p={r['p']:.4f}")

print('\n=== 3시드 개별 재현 (PRIMARY-B 관측영역) ===')
for s in SEEDS:
    d = [by[('AUTO', e, s)][0]['avg_RMSE_observed']
         - by[('AUTO-NOTRACK', e, s)][0]['avg_RMSE_observed'] for e in kept]
    print(f'  seed {s}: med={median(d):+.5f}  개선 {sum(1 for x in d if x<0)}/{n}')
