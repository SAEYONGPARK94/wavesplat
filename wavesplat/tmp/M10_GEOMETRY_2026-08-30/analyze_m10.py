"""M-10 분석기 — ⛔**여기에 문턱·밴드·라벨·해석이 없다.** 전부 `gates_m10.py` 를 부르고,
작업 목록·행 정체성은 `jobs_m10.py` 를 부른다.

    python3 tmp/M10_GEOMETRY_2026-08-30/analyze_m10.py          # → verdict_m10.json

★**0행 가드**: 결과 파일이 없거나 `jobs_m10.expected_jobs()` 를 다 못 채우면
  **판정하지 않고 exit 1**(`analyze_m8.py:62-70` 규약). ⛔부분 자료로 라벨을 만들지 않는다.
  ⚠추가 사유가 하나 더 있다: `gates_m10.cmp_ok_from` → `thresholds(n)` 은 **n 이 작으면
  `StopIteration` 으로 죽는다**(n≤5). 완전성 가드가 그 앞을 막는다.
★**차단/범위축소 분리**: 차단 게이트가 하나라도 FAIL 이면 판정 없음(label=null).
  ⚠단 **`G-CMP` 만은 다르다** — 그 실패는 사전등록된 라벨 **`NEARFIELD-VOID`** 그 자체다
  (`PREREG §4.4`). 그때는 라벨을 적되 ⛔우열은 읽지 않는다.
★**척도 셋 병기**(`F1` 교훈 — 승수 한 척도가 순위를 뒤집는다): 짝 승수 · **절대** 밴드 RMSE ·
  **참 제로필드 대비**를 함께 낸다. ⛔`ZERO_REF=1.0` 미사용(제로필드는 밴드마다 자료에서 낸다).
⛔`label` 이 `null` 로 끝나면 **결과를 열지 마라**(`M5` 의 교훈).
"""
import glob
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for _p in (HERE, ROOT,
           os.path.join(ROOT, 'tmp', 'B11_HOS_BASELINE_2026-08-23'),
           os.path.join(ROOT, 'tmp', 'B12_HOS_EXTEND_2026-08-23'),
           os.path.join(ROOT, 'tmp', 'S16_BASELINE_2026-08-14')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gates_m10 as G                                              # noqa: E402
import jobs_m10 as J                                               # noqa: E402

OUT = os.path.join(HERE, 'verdict_m10.json')

# ★결과 문서에서 지우지 마라 (`PREREG §6`). 인용할 때 동반한다.
DISCLOSURES = [
    'blind 아님 — 사전등록 §6-1 이 결과 전 예측(NEARFIELD-HOLDS)을 적어 두었고, '
    '★AMENDMENT_1 의 스모크가 6 에피소드(48 중 12.5%, 전부 BF4 층)에서 BAND_REAL 방향을 '
    '6/6 으로 드러냈다(⛔그 수는 인용 금지 — AMENDMENT_1 §5.4).',
    '⛔BLOCKER-P1 (AMENDMENT_1 에서 정정) — PREREG 초판의 λ_p = 76.5 m 는 **BF4 전용 수**인데 코퍼스는 '
    'BF4 24 + BF5 24 다(Tp 7.0/9.0 ⟹ λ_p 76.50/126.47 m). `premise_m10.py` 전건 재도출: '
    'obs:0_5 는 BF4 ≤0.0653 λ_p · BF5 ≤0.0395 λ_p, 실측 중앙은 12 레코드에서 0.0223–0.0485 λ_p. '
    '합성 gap 중앙은 88.14 m = **0.868 λ_p**(문서의 1.15 아님)이고 실측 대비 비는 **17.9배**'
    '(문서의 "25–30배" 아님). ⟹ 밴드 선택과 판정은 무손상이나 ⛔**동기 서술의 크기는 정정 대상**이다.',
    '⛔★방향 반전 (AMENDMENT_1 §4.3, premise_m10 이 매 실행 재검사): obs:0_5 상한은 '
    'BF4 0.06536 λ_p · BF5 0.03954 λ_p 이고 실측 12 중앙은 0.0223–0.0485 λ_p 다 ⟹ '
    'BF4 층은 0/12 가 밴드 밖(보수적 대리)이지만 **BF5 층은 2/12(BS11a·YS17)가 밴드보다 멀다**. '
    '최대 거리 기준으로는 BF4 9/12 · BF5 12/12 가 벗어난다. '
    '⟹ ⛔"밴드가 실측보다 넓다" 를 **층 없이 쓰지 마라**.',
    '접두어는 obs:(최근접 관측점까지의 거리)이지 배-거리가 아니다 — 같은 표본에서 1.8–4.1배 갈린다.',
    '이 배치가 SHIP 셸의 집계 규약을 기준선 규약으로 바꿨다(§3.3) ⟹ run.py 의 avg_shell_rmse 와 '
    '**같은 수가 아니다**. ⛔두 값을 섞어 인용하지 마라.',
    '동결값 재현은 비트가 아니라 허용오차다(G-4 rel Δ ≤ 1e-6) — 이 배치는 스레드를 고정하고 '
    'B-12 는 고정하지 않았다. 비트 일치 수는 G-12 가 보고한다.',
    '혼잡 층 한정(ncomp ≥ 15). 희소 층은 이 배치에 없다 — 같은 무게로 읽지 않는다.',
    'HOS M=3 이고 mesh_verdict 가 시드별로 갈린다 ⟹ ⛔"HOS M=3" 단일 서술 금지(G-14 가 분할 보고).',
    '체크포인트가 평가와 같은 분포라는 단서는 그대로다.',
    '재채점이다 ⟹ ⛔기제 지목·인과 귀속 전면 금지. 최대 주장은 "이득의 위치"까지.',
    '⛔BLOCKER-D1 (AMENDMENT_1 에서 해소) — 이 배치의 수는 **GPU 산물**이다. '
    '`PREREG §8`/`G-10` 의 CPU 고정과 `G-4` 의 동결 재현은 동시에 성립하지 않았고'
    '(CPU 고정 시 SHIP rel Δ = 3.171e-01, GPU 에서는 비트 일치), 앵커인 B-12 동결 자료가 '
    'GPU 환경의 산물이기 때문이다. ⟹ 재현 통화 = 「스레드 고정 · GPU 허용」. '
    '행의 `cuda_available`·`cuda_off_policy`·`prior_device`(사전이 **실제로** 쓴 장치의 되읽기, '
    'G-15)를 반드시 함께 인용하라. ⛔"CPU 에서 검증했다" 는 이 배치에 대해 거짓이다.',
    '⛔BLOCKER-P1 정정 후 크기: 합성/실측 피복 격차는 **17.9–39.0배**다 '
    '(⛔"25–30배" 는 AMENDMENT_1 로 철회). 합성 gap 중앙 = 0.868 λ_p '
    '(층별 BF4 1.152 · BF5 0.697), 미터 값 88.135 m 는 불변.',
    '⚠48 에피소드 λ_p 중앙 101.485 m 는 24/24 이봉 분포의 중점이라 어느 에피소드도 갖지 않는다 '
    '⟹ ⛔단일 수로 인용하지 말고 층(BF4 76.504 / BF5 126.466 m)을 병기하라.',
]


def load_rows():
    """★무인자 로더 — `glob.glob` **무별칭** + `json.loads`(`M8b F-2` 처방 (b)).
    ⛔`gates_m10.load_rows()` 와 같은 glob 을 쓰되 **행 원천이 이 파일에서도 보이게** 둔다
    (공유 헬퍼로만 단일출처화하면 `check_gate_refs._call_loaders` 눈에 안 띈다)."""
    out = []
    for path in sorted(glob.glob(os.path.join(HERE, 'results_m10_*.jsonl'))):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    return out


def _dump(obj):
    with open(OUT, 'w') as fh:
        json.dump(obj, fh, indent=1, ensure_ascii=False, default=str)


def _ratio(a, b):
    """에피소드별 `a/b`. 공통 키만, 분모 0/None 은 뺀다."""
    return {k: a[k] / b[k] for k in (set(a) & set(b)) if b.get(k)}


def _pair(a, b):
    w, n, rel = G.paired_wins(a, b)
    return dict(better=w, n=n, rel_median=rel,
                p_one_sided=(G.sign_test_one_sided(w, n) if n else None))


def _band_stats(rows, band):
    ship = G.med_by_episode(rows, 'SHIP', 'band_' + band)
    spec = G.med_by_episode(rows, 'SPEC25', 'band_' + band)
    zsh = G.med_by_episode(rows, 'SHIP', 'zero_' + band)
    zsp = G.med_by_episode(rows, 'SPEC25', 'zero_' + band)
    ks = sorted(set(ship) & set(spec))
    return dict(
        band=band,
        primary=_pair(ship, spec),                       # SHIP < SPEC25
        ship_vs_zero=_pair(ship, zsh),                   # S-1 (서술)
        spec_vs_zero=_pair(spec, zsp),                   # 비교자 유효성
        abs_median=dict(                                 # ★척도 ② 절대 밴드 RMSE
            SHIP=(st.median([ship[k] for k in ks]) if ks else None),
            SPEC25=(st.median([spec[k] for k in ks]) if ks else None),
            ZERO=(st.median([zsh[k] for k in ks]) if ks else None)),
        vs_zero_median=dict(                             # ★척도 ③ 참 제로필드 대비
            SHIP=(st.median([ship[k] / zsh[k] for k in ks if zsh.get(k)]) if ks else None),
            SPEC25=(st.median([spec[k] / zsp[k] for k in ks if zsp.get(k)]) if ks else None)),
        n_episodes=len(ks))


def main():
    rows = load_rows()
    ok = [r for r in rows if 'error' not in r]
    want = J.expected_jobs()
    have = {J.row_identity(r) for r in ok}
    missing = [j for j in want if j not in have]

    if not rows:
        print('⛔결과 0행 — 판정하지 않는다.')
        _dump(dict(batch='M-10', label=None, why='결과 0행', n_rows=0))
        return 1
    if missing:
        print('⛔작업 미완 %d/%d — 판정하지 않는다 (예: %r)'
              % (len(missing), len(want), missing[:3]))
        _dump(dict(batch='M-10', label=None, why='작업 미완 %d/%d' % (len(missing), len(want)),
                   n_rows=len(ok), missing_sample=[list(m) for m in missing[:20]]))
        return 1

    grows = G.gates(rows)
    print('=== 게이트 ===')
    for name, good, det, blocking in grows:
        print('  [%-4s] %-64s %s' % ('PASS' if good else ('FAIL' if blocking else 'SCOPE'),
                                     name, det))
    blocked = [n.split()[0] for n, o, d, b in grows if b and not o]
    scoped = [n.split()[0] for n, o, d, b in grows if (not b) and (not o)]

    if blocked:
        print('\n⛔차단 게이트 FAIL %r — 판정하지 않는다.' % (blocked,))
        _dump(dict(batch='M-10', label=None, why='차단 게이트 FAIL', blocked=blocked,
                   n_rows=len(ok), n_error=len(rows) - len(ok),
                   gates=[[n, o, d, b] for n, o, d, b in grows],
                   disclosures=DISCLOSURES))
        return 1

    cmp_ok, cmp_d = G.cmp_ok_from(rows)
    print('  [%-4s] G-CMP 비교자 유효성 (SPEC25 < ZERO on %s): %r'
          % ('PASS' if cmp_ok else 'FAIL', G.PRIMARY_BAND, cmp_d))

    prim = _band_stats(rows, G.PRIMARY_BAND)
    far = _band_stats(rows, 'BAND_FAR')
    wide = _band_stats(rows, 'BAND_WIDE')                # 사전선언 민감도

    # SECONDARY-B: 에피소드마다 r_near = SHIP/SPEC25(near), r_far = SHIP/SPEC25(far).
    # `r_near > r_far` 를 **한 개의 짝 검정**으로 본다. ⛔두 밴드의 p 를 비교하지 않는다.
    r_near = _ratio(G.med_by_episode(rows, 'SHIP', 'band_' + G.PRIMARY_BAND),
                    G.med_by_episode(rows, 'SPEC25', 'band_' + G.PRIMARY_BAND))
    r_far = _ratio(G.med_by_episode(rows, 'SHIP', 'band_BAND_FAR'),
                   G.med_by_episode(rows, 'SPEC25', 'band_BAND_FAR'))
    sec = _pair(r_far, r_near)                            # r_far < r_near ⟺ r_near > r_far

    n = prim['primary']['n']
    thr = G.thresholds(n)
    lab = G.label(prim['primary']['better'], n, cmp_ok)
    v = dict(batch='M-10', label=lab, fold=G.fold(lab), reading=G.READING[lab],
             cmp_ok=cmp_ok, cmp=cmp_d, thresholds=thr,
             bands={k: list(v_) for k, v_ in G.BANDS.items()},
             primary_band=G.PRIMARY_BAND,
             primary=prim, secondary_profile=sec, band_far=far,
             sensitivity_band_wide=wide,
             n_rows=len(ok), n_error=len(rows) - len(ok),
             scoped_out=scoped,
             gates=[[nm, o, d, b] for nm, o, d, b in grows],
             disclosures=DISCLOSURES)
    _dump(v)

    print('\n=== 판정 ===')
    print('  %s   (접기 = %s)' % (lab, G.fold(lab)))
    print('  %s' % G.READING[lab])
    print('\n  PRIMARY-A  SHIP < SPEC25 on %s : %d/%d  p=%.4g  상대중앙 %s'
          % (G.PRIMARY_BAND, prim['primary']['better'], n,
             prim['primary']['p_one_sided'],
             ('%.2f%%' % (100 * prim['primary']['rel_median'])
              if prim['primary']['rel_median'] is not None else '--')))
    print('  문턱  NOMINAL %d · CONFIRM %d · NOT_EARN %d · MDE q80 %.4f · 바닥 p %.3e'
          % (thr['NOMINAL'], thr['CONFIRM'], thr['NOT_EARN'], thr['q80'], thr['floor_p']))
    print('  ★척도 셋: 승수 %d/%d · 절대 밴드 RMSE SHIP %s / SPEC25 %s / ZERO %s · '
          '참 제로필드 대비 SHIP %s / SPEC25 %s'
          % (prim['primary']['better'], n,
             _f(prim['abs_median']['SHIP']), _f(prim['abs_median']['SPEC25']),
             _f(prim['abs_median']['ZERO']),
             _f(prim['vs_zero_median']['SHIP']), _f(prim['vs_zero_median']['SPEC25'])))
    print('  S-1 (서술) SHIP < ZERO on %s : %d/%d'
          % (G.PRIMARY_BAND, prim['ship_vs_zero']['better'], prim['ship_vs_zero']['n']))
    print('  SECONDARY-B  r_near > r_far : %d/%d  p=%.4g'
          % (sec['better'], sec['n'], sec['p_one_sided']))
    print('  민감도 BAND_WIDE : %d/%d  p=%.4g   대비 BAND_FAR : %d/%d  p=%.4g'
          % (wide['primary']['better'], wide['primary']['n'], wide['primary']['p_one_sided'],
             far['primary']['better'], far['primary']['n'], far['primary']['p_one_sided']))
    print('\n=== 공시 (⛔지우지 마라) ===')
    for d in DISCLOSURES:
        print('  · %s' % d)
    print('\n-> %s' % OUT)
    return 0


def _f(x):
    return '--' if x is None else '%.4f' % x


if __name__ == '__main__':
    sys.exit(main())
