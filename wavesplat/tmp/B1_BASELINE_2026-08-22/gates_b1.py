"""B-1 단일출처 — ⚠**사후 작성**(2026-08-23). 이 파일이 존재하는 이유가 곧 결함의 기록이다.

## 왜 지금 쓰는가
`B-1` 은 **런 0회 재집계**였고 `RESULT.md` 하나만 남겼다 ⟹ **`verdict` 도 `gates` 도 없었다.**
그런데 그 배치가 낸 수(**혼잡 62/64 · −38.58%**)가 **학위논문 제1장 기여 ①의 헤드라인**이다.
⟹ ★***논문에서 가장 많이 인용되는 수가 코드 판정 없이 산문으로만 존재했다.***
오늘 하루 종일 그 결함 부류에 물렸으므로, 여기서도 닫는다.

⚠**정직성**: 이 게이트/분석기는 **결과 뒤에** 쓰였다. 규칙을 새로 고르지 않고
`RESULT.md` 와 설계 문서(`tmp/DEGREE_ROADMAP_2026-08-22/B1_DESIGN.md`)에 **이미 적힌 것**을
코드로 옮긴다. 검사기의 `analyzer<results` 열이 이 위반을 표시한다.

## 자료 (전부 기존 배치의 원자료 — 새 런 0회)
- `SHIP` : `../W16_REBASELINE_2026-08-22/results.jsonl` (384행 = 128 에피소드 × 3 시드)
- 기준선: `../S16_BASELINE_2026-08-14/results.jsonl` (`SPEC25` 384행) —
  `S17` 이 신규 4팔 중 최강을 못 내서 **`SPEC25` 가 여전히 `BEST`** 다.

## 판정 (`RESULT.md` 문언 그대로)
혼잡 층에서 `SHIP` 이 `SPEC25` 보다 낮은 에피소드 수(에피소드당 시드 median-of-3):
- **≥ 41/64**(문턱, 단측 p<0.05) ⟹ `SHIP-BEATS-BEST-CROWDED`
- 그 미만 ⟹ `UNDECIDED`
⛔**희소 층과 전체 합산은 라벨에 넣지 않는다** — `RESULT.md §2` 가 *"희소는 부호만,
전체 합산은 통째로 인용 금지"* 라 적었고 금지 목록에도 있다. **여기서도 인쇄하지 않는다.**
"""
import os, sys, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))
W16 = os.path.abspath(os.path.join(HERE, '..', 'W16_REBASELINE_2026-08-22'))
S16 = os.path.abspath(os.path.join(HERE, '..', 'S16_BASELINE_2026-08-14'))
SHIP_ARM, BASE_ARM = 'SHIP', 'SPEC25'
FIELD = 'avg_nRMSE'
THRESHOLD = 41          # /64, RESULT.md 문언
N_EPISODES = 64         # 혼잡 층


def sign_test_one_sided(w, n):
    from math import comb
    return sum(comb(n, k) for k in range(w, n + 1)) / 2 ** n


def _arm_of(r):
    """⚠배치마다 팔 키가 다르다 — `W16` 은 `cell`, `S16` 은 `arm`.

    ★이것 자체가 이 저장소의 결함 부류다(*"같은 것을 두 이름으로"*).  여기서 흡수하되
    **어느 쪽에서 왔는지 게이트가 세도록** 한다.
    """
    return r.get('cell') if r.get('cell') is not None else r.get('arm')


def med_by_episode(rows, arm, stratum='crowded'):
    d = {}
    for r in rows:
        if 'error' in r or _arm_of(r) != arm:
            continue
        if stratum and r.get('stratum') != stratum:
            continue
        v = r.get(FIELD)
        if v is not None:
            d.setdefault(r['episode'], []).append(v)
    return {e: st.median(v) for e, v in d.items()}


def paired(ship, base):
    ks = sorted(set(ship) & set(base))
    w = sum(1 for k in ks if ship[k] < base[k])
    return {'n': len(ks), 'better': w,
            'rel_median_pct': st.median([(ship[k] - base[k]) / base[k] * 100 for k in ks]) if ks else None,
            'p_one_sided': sign_test_one_sided(w, len(ks)) if ks else None}


def gates(ship, base, n_ship_rows, n_base_rows):
    out = [('G-0 SHIP 행 수 384', n_ship_rows == 384, str(n_ship_rows)),
           ('G-0b SPEC25 행 수 384', n_base_rows == 384, str(n_base_rows)),
           ('G-1 혼잡 에피소드 64개(SHIP)', len(ship) == N_EPISODES, f'{len(ship)}'),
           ('G-1b 혼잡 에피소드 64개(SPEC25)', len(base) == N_EPISODES, f'{len(base)}'),
           # ★G-2 두 팔이 **같은 에피소드**를 봤는가 (교집합이 곧 전체여야 한다)
           ('★G-2 에피소드 집합 완전 일치', set(ship) == set(base),
            f'교집합 {len(set(ship) & set(base))} / 합집합 {len(set(ship) | set(base))}'),
           # ★G-3 시드 median-of-3 이 실제로 3에서 나왔는가는 원본 행 수로 대리 확인
           ('G-3 에피소드당 3행(384/128)', n_ship_rows == 3 * 128, '')]
    return out


def label(p):
    if p['better'] >= THRESHOLD:
        return ('SHIP-BEATS-BEST-CROWDED',
                f'혼잡 층에서 새 배포 구성이 최강 적격 기준선을 {p["better"]}/{p["n"]} 이긴다 '
                f'(상대 중앙 {p["rel_median_pct"]:.2f}%, 단측 p={p["p_one_sided"]:.3e}). '
                '⚠의무 병기: ①합성·선형 GT ②에피소드가 체크포인트 학습 분포와 겹친다 '
                '③"강화된" 기준선이 아니라 "우리가 가진 최강" 이다.')
    return ('UNDECIDED', f'{p["better"]}/{p["n"]} — 문턱 {THRESHOLD} 미달.')
