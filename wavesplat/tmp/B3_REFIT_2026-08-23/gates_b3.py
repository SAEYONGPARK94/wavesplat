"""B-3 단일출처 — **`bank_refit_every_s` 확증**: 배포 실측 경로의 *시간축*.

## 왜 이 배치인가
`C`(08-20)가 남긴 **유일하게 살아있는 실측 축**이다:
`bank_mmse` 는 적합을 한 번 하고 **동결**하는데, 2 s 마다 재적합하면
`frozen 0.9452 → refit@2s 0.7932`(8/9 창) 였다. **그러나 레코드 5개라 `DESCRIPTIVE` 였다.**
`STATUS_2026-08-21` 은 *"확증하려면 **새 실측 데이터**가 있어야 한다"* 고 적었다 —
⛔**그 문장은 틀렸다.** 팬을 뺀 `scene_station` 이 **같은 보유 자료에서 12 레코드**를 연다.

★**이것이 왜 학위에 중요한가**: 문헌의 유일한 틈이 **"실시간 순차 추적"** 인데(Wang 2025 는
같은 문제를 배치로 푼다) 저장소의 기여 ②c 는 **미측정**이었다. 이 배치가 그것을 잰다.
그리고 **짝 비교**라서 오늘 무너진 `ZERO_REF` 결함에 **면역**이다.

## 프로토콜 (결과 전 확정)
과거 40 s 로 초기화 → lead 1..14 s 각각에서 **① 먼저 예측·채점 ② 그 다음 그 시각의 관측을 먹인다**
(정직한 예보. 순서를 바꾸면 채점에 미래 관측이 새어 든다).
채점 = σ 정규화 RMSE, **분모는 `B-2` 와 같은 `sg`** — ⚠단 **짝 비교라 기준값은 소거된다**.

## 판정 (결과 전 확정) — 짝 부호검정, n=36 (12 레코드 × 창 1·3·5)
**PRIMARY** = `REFIT2` 의 `fc_mean`(lead 2–14 s 평균)이 `FROZEN` 보다 작은 창의 수:
- **≥26/36** (p=0.00567 ≤ 가족단위 α=0.05/7=0.0071) ⟹ ★`REFIT-HELPS (CONFIRMATORY)`
- **≥24/36** (p=0.03262) ⟹ ⚠`REFIT-HELPS (EXPLORATORY)`
- **19~23** ⟹ `UNDECIDED`
- **≤18/36** ⟹ ⛔`REFIT-NULL-OR-HURTS`

★**다중성 결과 전 기재**: 오늘 같은 12 레코드를 보는 **일곱 번째**다.
★**개입 계수기는 *가지* 단위**(오늘 B-2f 가 함수 단위로는 못 잡는다는 것을 보였다):
Phase 2 의 `_bank_fit_window` 호출 수를 센다. `FROZEN`=0 · `REFIT2`>0 이어야 한다.
"""
RECORDS = ['AA14', 'AA15a', 'AA15b', 'BS11a', 'BS11b', 'BS11c', 'BS11d',
           'BS13a', 'BS13b', 'BS13c', 'LJ18a', 'YS17']
WINDOWS = [1, 3, 5]
SEED = 42
NOISE = 0.02
ARMS = {'FROZEN': {'bank_refit_every_s': None},   # 배포 기본값
        'REFIT2': {'bank_refit_every_s': 2.0}}
ARM_ORDER = ['FROZEN', 'REFIT2']
FIELD = 'fc_mean'                 # lead 2–14 s 평균 (결과 전 확정)
N_PAIRS = len(RECORDS) * len(WINDOWS)
NOMINAL, CONFIRM = 24, 26
FAMILY_ALPHA = 0.05 / 7


def sign_test_one_sided(w, n):
    from math import comb
    return sum(comb(n, k) for k in range(w, n + 1)) / 2 ** n


def key(r):
    return (r['record'], r['win'])


def paired(rows, arm, ref, field=FIELD):
    import statistics as st
    a = {key(r): r[field] for r in rows if r['cell'] == arm and r.get(field) is not None}
    b = {key(r): r[field] for r in rows if r['cell'] == ref and r.get(field) is not None}
    ks = sorted(set(a) & set(b))
    n = len(ks)
    w = sum(1 for k in ks if a[k] < b[k])
    return {'arm': arm, 'ref': ref, 'n': n, 'better': w,
            'ties': sum(1 for k in ks if a[k] == b[k]),
            'rel_median_pct': st.median([(a[k] - b[k]) / b[k] * 100 for k in ks]) if ks else None,
            'median_arm': st.median([a[k] for k in ks]) if ks else None,
            'median_ref': st.median([b[k] for k in ks]) if ks else None,
            'p_one_sided': sign_test_one_sided(w, n) if n else None}


def gates(rows):
    ok = [r for r in rows if 'error' not in r]
    exp = len(ARMS) * N_PAIRS
    out = [('G-0 런 수', len(ok) == exp, f'{len(ok)}/{exp}'),
           ('G-0b 에러 0', len(rows) == len(ok), f'{len(rows)-len(ok)}')]
    fr = [r for r in ok if r['cell'] == 'FROZEN']
    rf = [r for r in ok if r['cell'] == 'REFIT2']
    # ★G-1 배포 실측 게이트가 두 팔 모두 bank 경로로 보냈나 (성분 0개 = bank)
    out.append(('G-1 두 팔 다 bank 경로', all((r.get('n_atoms') or 0) == 0 for r in ok),
                str(sorted({r.get('n_atoms') for r in ok}))[:60]))
    # ★G-2 가지 단위 개입 계수기
    out.append(('G-2 FROZEN 재적합 0회', all(r.get('refit_calls') == 0 for r in fr),
                str(sorted({r.get('refit_calls') for r in fr}))[:60]))
    out.append(('G-2b REFIT2 재적합 >0', all((r.get('refit_calls') or 0) > 0 for r in rf),
                str(sorted({r.get('refit_calls') for r in rf}))[:60]))
    # ★G-3 음성 대조 — lead 0 은 두 팔이 비트 동일이어야 한다(재적합 전이다)
    n0 = {key(r): r.get('lead0') for r in fr}
    n1 = {key(r): r.get('lead0') for r in rf}
    ks = sorted(set(n0) & set(n1))
    out.append(('G-3 lead 0 비트동일(음성대조)',
                bool(ks) and all(n0[k] == n1[k] for k in ks),
                f'{sum(1 for k in ks if n0[k] == n1[k])}/{len(ks)}'))
    out.append(('G-4 36 창 전건', len({key(r) for r in ok}) == N_PAIRS, ''))
    out.append(('G-5 되읽기 일치',
                all(r.get('refit_readback') == ARMS[r['cell']]['bank_refit_every_s'] for r in ok), ''))
    return out


def label(p):
    w, n = p['better'], p['n']
    if w >= CONFIRM:
        return ('REFIT-HELPS (CONFIRMATORY)',
                f'재적합이 {w}/{n} 창에서 낫다(p={p["p_one_sided"]:.4f} ≤ 가족단위 '
                f'{FAMILY_ALPHA:.4f}) ⟹ ★C 의 `DESCRIPTIVE` 가 확증으로 승격된다.')
    if w >= NOMINAL:
        return ('REFIT-HELPS (EXPLORATORY)',
                f'재적합이 {w}/{n} 창에서 낫다(p={p["p_one_sided"]:.4f}) — 명목 통과이나 '
                f'가족단위 {FAMILY_ALPHA:.4f} 미달. ⚠확증 아님.')
    if w <= n - NOMINAL:
        return ('REFIT-NULL-OR-HURTS',
                f'재적합이 {w}/{n} 뿐 ⟹ C 의 서술이 12 레코드에서 재현되지 않는다.')
    return ('UNDECIDED', f'부호로 갈리지 않는다({w}/{n}). 크기를 따로 읽어라.')
