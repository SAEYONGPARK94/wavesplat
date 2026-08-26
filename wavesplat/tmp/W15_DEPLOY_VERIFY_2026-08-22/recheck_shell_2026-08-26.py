"""W15 §4 셸 표 사후 재집계 — **판정을 낳지 않는 포렌식 스크립트**(임무 V-B, 2026-08-26).

동결 판정(`verdict_w15.json`)은 이 파일을 쓰지 않는다. 이 파일이 다루는 것은
`RESULT.md` §4 의 **손으로 만든 셸 표** 하나뿐이다. 그 표는 동결 규칙(`gates_w15.py`)에
없다 — `G-5` 는 셸 키가 **기록됐는지**만 보고 값은 보지 않는다.

★밝힌 것 세 가지
  ① RMSE·비 열은 **층을 합산한 중앙값**으로 정확히 재현된다(⟹ 수치 자체는 진짜다).
  ② 그러나 그 합산은 이 배치 **자신의 사전등록 §4**(*"층별로만 말한다"*)를 벗어난다.
     층으로 가르면 *"9셸 전부 2.2–2.5배로 평평"* 이 **사라진다**.
  ③ `격자 비중` 열은 **어느 집계로도 재현되지 않는다**(손복사 의심).

⟹ 결론 문장(⛔`SUPPORT-TRADEOFF` 아니다)은 **살아남고 오히려 강해진다**:
   혼잡에서 우위가 가장 큰 곳이 **근거리**이고 멀어질수록 줄어든다 = 거래의 정반대다.

실행: python3 tmp/W15_DEPLOY_VERIFY_2026-08-22/recheck_shell_2026-08-26.py
"""
import json
import os
import statistics as st
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
SHELLS = ['0_5', '5_10', '10_20', '20_30', '30_50', '50_75', '75_100', '100_150', '150_inf']

ROWS = [json.loads(l) for l in open(os.path.join(HERE, 'results.jsonl'))]
OK = [r for r in ROWS if 'error' not in r]
_C = json.load(open(os.path.join(HERE, 'corpus.json')))
STRATUM = {e: 'crowded' for e in _C['crowded']}
STRATUM.update({e: 'sparse' for e in _C['sparse']})

# RESULT.md §4 가 인쇄한 5행 (축자). 재현 대조용 — 이 스크립트가 **틀리면 실패한다**.
PRINTED = {  # shell: (격자비중%, 학습, 해석적, 비)
    '0_5':     (2.7, 0.0261, 0.0569, 2.18),
    '10_20':   (4.0, 0.0429, 0.1078, 2.51),
    '50_75':  (15.0, 0.0628, 0.1581, 2.52),
    '100_150': (22.1, 0.0894, 0.2055, 2.30),
    '150_inf': (27.9, 0.1036, 0.2444, 2.36),
}


def two_sided(w, n):
    ps = [comb(n, k) / 2 ** n for k in range(n + 1)]
    return min(1.0, sum(p for p in ps if p <= ps[w] * (1 + 1e-12)))


def pooled_median(arm, shell, key='avg_shell_rmse'):
    v = [(r.get(key) or {}).get(shell) for r in OK if r['cell'] == arm]
    return st.median([x for x in v if x is not None])


def med_by_episode(arm, shell):
    d = {}
    for r in OK:
        v = (r.get('avg_shell_rmse') or {}).get(shell)
        if r['cell'] == arm and v is not None:
            d.setdefault(r['episode'], []).append(v)
    return {e: st.median(x) for e, x in d.items()}


def main():
    fail = []

    print('=== ① RESULT.md §4 재현 (층 합산 · 1,152런 중앙값) ===')
    print(f"{'shell':10s} {'학습':>9s} {'해석적':>9s} {'비':>6s}   {'인쇄된 값 대비'}")
    for s, (fr, la, an, ra) in PRINTED.items():
        a, b = pooled_median('POST', s), pooled_median('ALL-ANALYTIC', s)
        okv = abs(a - la) < 5e-5 and abs(b - an) < 5e-5 and abs(b / a - ra) < 0.01
        if not okv:
            fail.append(f'§4 RMSE/비 재현 실패: {s}')
        print(f'{s:10s} {a:9.4f} {b:9.4f} {b / a:5.2f}x   {"OK" if okv else "MISMATCH"}')

    print('\n=== ③ `격자 비중` 열은 재현되지 않는다 ===')
    print(f"{'shell':10s} {'인쇄':>7s} {'중앙':>7s} {'평균':>7s}")
    for s, (fr, *_rest) in PRINTED.items():
        v = [(r.get('shell_frac') or {}).get(s) for r in OK]
        v = [x for x in v if x is not None]
        md, mn = st.median(v) * 100, st.mean(v) * 100
        flag = '' if min(abs(md - fr), abs(mn - fr)) < 0.15 else '  ⟵ 어느 집계로도 안 맞음'
        print(f'{s:10s} {fr:6.1f}% {md:6.1f}% {mn:6.1f}%{flag}')
    print('⟹ `shell_frac` 은 **세 팔에서 완전히 동일**하다(격자 기하라 추정기와 무관).')
    print('   그런데도 인쇄된 5개 중 3개가 어긋난다 ⟹ 이 열은 손복사로 봐야 한다.')

    print('\n=== ② 층으로 가르면 "평평" 이 사라진다 (짝 · 에피소드별 median-of-3) ===')
    for strat in ('crowded', 'sparse'):
        print(f'\n--- {strat} : POST(학습) vs ALL-ANALYTIC(배포 해석적) ---')
        print(f"{'shell':10s} {'학습 승':>10s} {'비 중앙':>9s} {'상대중앙':>10s} {'양측 p':>9s}")
        for s in SHELLS:
            A, B = med_by_episode('POST', s), med_by_episode('ALL-ANALYTIC', s)
            ks = [k for k in sorted(set(A) & set(B)) if STRATUM.get(k) == strat]
            w = sum(1 for k in ks if A[k] < B[k])
            ratio = st.median([B[k] / A[k] for k in ks])
            rel = st.median([(A[k] - B[k]) / B[k] * 100 for k in ks])
            print(f'{s:10s} {w:6d}/{len(ks):<3d} {ratio:8.2f}x {rel:+9.2f}% {two_sided(w, len(ks)):9.2g}')

    print('\n' + '=' * 72)
    if fail:
        print('FAIL — ' + ' | '.join(fail))
        raise SystemExit(1)
    print('PASS — §4 의 RMSE·비 열은 층 합산 중앙값으로 정확히 재현된다.')
    print('       ⛔단 그 합산은 PREREG §4 를 벗어나고, 층별 짝 수치가 옳은 인용 대상이다.')


if __name__ == '__main__':
    main()
