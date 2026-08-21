"""P3 감사관의 감사 — AUDIT-C 의 C1/C2 를 확장 재검정.

1) RESTART 10분할 -> **20 순서분할**(기준선 대칭화) + 분할-평균 통계량
2) ★C_matched(= A 배치 PRIMARY 후보 '맞대결')를 **모든 분할에서** 재도출
3) S7 P-1 집계 규약 전수(중앙/평균/시드3/런풀링)
4) 대체 PRIMARY 후보의 MDE·검정력
"""
import json, os, math, itertools, statistics as st
from collections import defaultdict

ROOT = '/home/vexit/wavesplat/wavesplat/tmp'
S9 = os.path.join(ROOT, 'S9_lr_x_l1_2026-08-11', 'results.jsonl')
X2 = os.path.join(ROOT, 'X2_no_learned_2026-08-11', 'results.jsonl')
S7 = os.path.join(ROOT, 'S7_confirm_n32_2026-08-10', 'results.jsonl')


def p2(k, n):
    if n == 0:
        return float('nan')
    return sum(math.comb(n, i) for i in range(n + 1)
               if abs(i - n / 2) >= abs(k - n / 2)) / 2.0 ** n


def min_k(n, a=0.05):
    for k in range(n // 2, n + 1):
        if p2(k, n) <= a:
            return k
    return None


def binom_ge(n, k, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def regime(ep):
    b = os.path.basename(ep)
    if '_2comp_' in b or '_5comp_' in b:
        return 'sparse'
    if '_15comp_' in b or '_30comp_' in b:
        return 'crowded'
    raise ValueError(ep)


def load_rows(p):
    return [json.loads(l) for l in open(p) if l.strip()]


# ── 시드별 원자료 (중앙 집계를 나중에 분할별로) ─────────────────────────────
def by_cell_ep_seed(rows, cells):
    d = defaultdict(dict)
    for r in rows:
        if 'error' in r or r['cell'] not in cells:
            continue
        d[(r['cell'], os.path.basename(r['episode']))][r['seed']] = r
    return d


s9 = load_rows(S9)
x2 = load_rows(X2)
D9 = by_cell_ep_seed(s9, {'AUTO', 'AUTO-SEED-NULL', 'AUTO-L1'})
D2 = by_cell_ep_seed(x2, {'V2K'})
SEEDS6 = sorted({s for (c, e), d in D9.items() if c in ('AUTO', 'AUTO-SEED-NULL') for s in d})
print('AUTO/AUTO-SEED-NULL 시드 합집합:', SEEDS6)
print('V2K 시드:', sorted({s for d in D2.values() for s in d}))

EPS = sorted({e for (c, e) in D9 if c == 'AUTO-L1'})
BYREG = defaultdict(list)
for e in EPS:
    BYREG[regime(e)].append(e)
print('에피소드', {k: len(v) for k, v in BYREG.items()})


def pool(cell, ep, seeds, metric):
    d = D9.get((cell, ep)) or D2.get((cell, ep))
    if not d:
        return None
    v = [d[s][metric] for s in seeds if s in d and d[s].get(metric) is not None]
    return st.median(v) if v else None


def med_group(ep, seeds, metric):
    """AUTO/AUTO-SEED-NULL 를 합쳐 임의 시드집합의 중앙."""
    v = []
    for cell in ('AUTO', 'AUTO-SEED-NULL'):
        d = D9.get((cell, ep), {})
        for s in seeds:
            if s in d and d[s].get(metric) is not None:
                v.append(d[s][metric])
    return st.median(v) if v else None


def v2k(ep, metric):
    d = D2.get(('V2K', ep), {})
    v = [r[metric] for r in d.values() if r.get(metric) is not None]
    return st.median(v) if v else None


def signrow(rel):
    imp = sum(1 for r in rel if r < 0)
    wor = sum(1 for r in rel if r > 0)
    tie = len(rel) - imp - wor
    nz = [r for r in rel if r]
    return imp, wor, tie, p2(max(imp, wor), imp + wor) if imp + wor else 1.0, \
        (st.median(nz) if nz else 0.0), (st.median(rel) if rel else 0.0)


print('\n' + '=' * 82)
print('## 1. RESTART — 순서분할 20가지 (기준선 = A). AUDIT-C 는 10개만 봤다')
print('=' * 82)
ORD = [(A, tuple(s for s in SEEDS6 if s not in A))
       for A in itertools.combinations(SEEDS6, 3)]
for rg in ('sparse', 'crowded'):
    res = []
    for A, B in ORD:
        rel = []
        for e in BYREG[rg]:
            wa, wb = med_group(e, A, 'window_resid'), med_group(e, B, 'window_resid')
            na, nb = med_group(e, A, 'nRMSE'), med_group(e, B, 'nRMSE')
            if None in (wa, wb, na, nb):
                continue
            rel.append(((na if wa <= wb else nb) - na) / na)
        i, w, t, p, mnz, mall = signrow(rel)
        res.append((p, i, w, t, A, mnz, len(rel)))
    res.sort()
    print(f'  --- {rg}  n={len(BYREG[rg])} ---')
    for p, i, w, t, A, m, n in res:
        tag = ' <-S9 분할' if A == (1575, 1576, 1577) else ''
        print(f'    A={A}  {i:2d}/{w:2d} tie{t:2d}  n_eff={i+w:2d}  p={p:.4f}  '
              f'med_nz={m*100:+7.2f}%{"  *" if p <= .05 else "   "}{tag}')
    sig = sum(1 for r in res if r[0] <= 0.05)
    print(f'    ★유의 {sig}/20   n_eff 범위 [{min(r[1]+r[2] for r in res)}, '
          f'{max(r[1]+r[2] for r in res)}]   med_nz 범위 '
          f'[{min(r[5] for r in res)*100:+.2f}%, {max(r[5] for r in res)*100:+.2f}%]')

print('\n' + '=' * 82)
print('## 2. ★C_matched (= A 배치 PRIMARY 맞대결) 를 20 순서분할 전부에서')
print('##    T = SELECT{A, V2K}   N = SELECT{A, B}   rel = (T-N)/N   (window_resid 로 선택)')
print('=' * 82)
CM = {}
for rg in ('sparse', 'crowded'):
    res = []
    for A, B in ORD:
        rel = []
        for e in BYREG[rg]:
            wa = med_group(e, A, 'window_resid')
            wb = med_group(e, B, 'window_resid')
            wv = v2k(e, 'window_resid')
            na = med_group(e, A, 'nRMSE')
            nb = med_group(e, B, 'nRMSE')
            nv = v2k(e, 'nRMSE')
            if None in (wa, wb, wv, na, nb, nv):
                continue
            T = na if wa <= wv else nv
            N = na if wa <= wb else nb
            if N > 0:
                rel.append((T - N) / N)
        i, w, t, p, mnz, mall = signrow(rel)
        res.append((p, i, w, t, A, mnz, mall, len(rel)))
    res.sort()
    print(f'  --- {rg} ---')
    for p, i, w, t, A, mnz, mall, n in res:
        tag = ' <-S11 이 쓴 분할' if A == (1575, 1576, 1577) else ''
        print(f'    A={A}  {i:2d}/{w:2d} tie{t:2d}  n_eff={i+w:2d}  p={p:.3e}  '
              f'med_all={mall*100:+7.2f}%  med_nz={mnz*100:+7.2f}%'
              f'{"  *" if p <= .05 else "   "}{tag}')
    sig = sum(1 for r in res if r[0] <= 0.05)
    print(f'    ★유의 {sig}/20   n_eff [{min(r[1]+r[2] for r in res)},{max(r[1]+r[2] for r in res)}]'
          f'   승률 범위 [{min(r[1]/(r[1]+r[2]) for r in res):.3f},'
          f'{max(r[1]/(r[1]+r[2]) for r in res):.3f}]'
          f'   med_nz [{min(r[5] for r in res)*100:+.2f}%,{max(r[5] for r in res)*100:+.2f}%]')
    CM[rg] = res

print('\n' + '=' * 82)
print('## 3. S7 P-1 집계 규약 전수 (AUDIT-C 는 median/mean 둘만 대조했다)')
print('=' * 82)
s7 = load_rows(S7)
D7 = by_cell_ep_seed(s7, {'AUTO-NT', 'AUTO-NT-L1-1e4'})
E7 = sorted({e for (c, e) in D7 if c == 'AUTO-NT' and regime(e) == 'crowded'})
S7SEEDS = sorted({s for d in D7.values() for s in d})
K32 = min_k(32, 0.05)
print(f'  n={len(E7)} 혼잡, 시드 {S7SEEDS}, 사전등록 문턱 k>={K32}')


def s7row(agg):
    rel = []
    for e in E7:
        a = [D7[('AUTO-NT-L1-1e4', e)][s]['nRMSE'] for s in S7SEEDS]
        b = [D7[('AUTO-NT', e)][s]['nRMSE'] for s in S7SEEDS]
        if agg == 'median':
            va, vb = st.median(a), st.median(b)
        elif agg == 'mean':
            va, vb = sum(a) / 3, sum(b) / 3
        else:
            k = S7SEEDS.index(agg)
            va, vb = a[k], b[k]
        rel.append((va - vb) / vb)
    return signrow(rel)


rows = []
for agg in ['median', 'mean'] + S7SEEDS:
    i, w, t, p, mnz, mall = s7row(agg)
    rows.append((str(agg), i, w, p, mall))
    print(f'  {str(agg):10s}  {i:2d}/{i+w:<2d}  p={p:.5f}  med_rel={mall*100:+6.2f}%  '
          f'{"PASS" if i >= K32 and p <= 0.05 else "FAIL"}')
# 런 풀링
pl = []
for e in E7:
    for s in S7SEEDS:
        a = D7[('AUTO-NT-L1-1e4', e)][s]['nRMSE']
        b = D7[('AUTO-NT', e)][s]['nRMSE']
        pl.append((a - b) / b)
i, w, t, p, mnz, mall = signrow(pl)
print(f'  {"pooled96":10s}  {i:2d}/{i+w:<2d}  p={p:.3e}  med_rel={mall*100:+6.2f}%  '
      f'(문턱 다름: k>={min_k(96,0.05)})')
npass = sum(1 for r in rows if r[1] >= K32 and r[3] <= 0.05)
print(f'  ★에피소드-집계 규약 5가지 중 통과 {npass}/5')

print('\n' + '=' * 82)
print('## 4. 대체 PRIMARY 검정력 — 맞대결(H2H) 을 신선 n 에서')
print('=' * 82)
for rg in ('sparse', 'crowded'):
    res = CM[rg]
    rates = [r[1] / (r[1] + r[2]) for r in res]
    tierate = [r[3] / (r[1] + r[2] + r[3]) for r in res]
    print(f'  {rg}: 20분할 승률 중앙 {st.median(rates):.4f} '
          f'(최악 {min(rates):.4f} / 최선 {max(rates):.4f})  '
          f'동점률 중앙 {st.median(tierate):.3f}')
for rg, ntot in (('sparse', 64), ('crowded', 64), ('sparse', 96), ('crowded', 96),
                 ('sparse', 128)):
    res = CM[rg]
    rates = sorted(r[1] / (r[1] + r[2]) for r in res)
    tr = st.median([r[3] / (r[1] + r[2] + r[3]) for r in res])
    neff = int(round(ntot * (1 - tr)))
    k = min_k(neff, 0.05)
    for label, q in (('중앙승률', st.median(rates)), ('최악분할', rates[0]),
                     ('하위25%', rates[len(rates) // 4])):
        print(f'  {rg:8s} n={ntot:3d} -> n_eff={neff:3d} k>={k:3d} | {label} '
              f'{q:.4f} -> power {binom_ge(neff, k, q):.4f}')
