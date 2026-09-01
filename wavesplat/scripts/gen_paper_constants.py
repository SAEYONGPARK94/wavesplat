#!/usr/bin/env python3
"""`paper/paper_constants.tex` 생성 — ⛔논문의 수치를 **손으로 적지 마라**.

선례 = `scripts/gen_deploy_constants.py`(학위논문 배포 상수).
사유 = 이 저장소가 반복해서 대가를 치른 **「손복사 기준 상수」** 결함 부류.
`paper/wavesplat.tex` 는 철회된 헤드라인을 **라이브 22줄**에 손으로 인쇄하고 있었고,
그 22줄은 서로 독립적으로 낡을 수 있었다.

★이 스크립트는 **동결 `verdict_*.json` 에서 값을 읽고**, 아래 `PINS` 와 어긋나면
`AssertionError` 로 죽는다. ⟹ 배치가 재분석되면 논문이 **조용히** 낡지 않는다.

사용:
    python3 scripts/gen_paper_constants.py          # 검증 + 생성
    python3 scripts/gen_paper_constants.py --show   # 값만 출력
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
TMP = os.path.join(ROOT, 'tmp')
OUT = os.path.join(ROOT, 'paper', 'paper_constants.tex')


def _get(v, path):
    """경로를 따라간다. ⚠키 자체에 점이 있는 경우(`ladder` 의 `"0.0"`)를 위해
    **튜플 경로**도 받는다 — 점 분할이 그런 키를 조용히 쪼갠다."""
    cur = v
    keys = path if isinstance(path, (list, tuple)) else path.split('.')
    for k in keys:
        if isinstance(cur, list):
            cur = cur[int(k)]
        else:
            cur = cur[k]
    return cur


def _count_tests():
    """pytest --collect-only 로 실제 테스트 수를 센다(thesis 생성기와 같은 규약)."""
    import subprocess
    r = subprocess.run([sys.executable, '-m', 'pytest', 'wavesim/tests/',
                        'wavesplat/tests/', '--collect-only', '-q'],
                       cwd=ROOT, capture_output=True, text=True)
    import re as _re
    m = _re.search(r'(\d+)\s+tests? collected', r.stdout)
    return m.group(1) if m else '0'


def _prior_arch():
    """★학습 사전의 **아키텍처를 체크포인트 가중치에서** 읽는다.

    5차 감사(2026-09-02) REJECT 경계 지적:

    > *"제목이 파는 학습 사전을 논문이 두 문장으로 정의한다. 아키텍처.학습 자료.학습
    >  절차가 없다. 심사자는 재현도 과적합 판단도 할 수 없다."*

    ⛔손으로 적으면 체크포인트를 갈 때 조용히 낡는다 --- 그래서 **가중치에서 센다**.
    ⚠학습 절차(스텝 수.혼합비)는 가중치에 없다: 그것은 배치 문서에서 오고,
      그 파일이 동결 후 움직였다는 사실을 논문이 **공시**한다.
    """
    import re as _re
    import torch
    from wavesplat.optimizer.learned_init import DEFAULT_CKPT
    sd = torch.load(DEFAULT_CKPT, map_location='cpu', weights_only=False)
    blocks = len({int(m.group(1)) for k in sd
                  for m in [_re.match(r'blocks\.(\d+)\.', k)] if m})
    return {
        'PriorParams': f"{sum(v.numel() for v in sd.values()):,}",
        'PriorDim': str(sd['enc.pos'].shape[-1]),
        'PriorTokens': str(sd['enc.pos'].shape[1]),
        'PriorBlocks': str(blocks),
    }


def _ckpt_sha():
    """배포 체크포인트의 **코드에 핀된** SHA256 앞 12자리."""
    from wavesplat.optimizer.learned_init import DEFAULT_CKPT_SHA256
    return DEFAULT_CKPT_SHA256[:12]


def _k_lim():
    """학습 사전의 파수 지지 상한 --- `setdiff_prior.pt` 의 `stats['K_LIM']`.

    샘플러가 정규화 좌표를 `clamp(-1,1)` 하므로(`d3_beamform_cond_diffusion.py`)
    `|kx|,|ky| <= K_LIM` 이 **구조적** 상한이다.  ⛔손으로 적지 마라 --- 체크포인트를
    갈면 조용히 낡는다.
    """
    import torch
    ck = os.path.join(ROOT, 'results', 'direction3', 'setdiff_prior.pt')
    if not os.path.exists(ck):
        raise SystemExit(f'⛔체크포인트가 없다: {ck} — 지지 상한을 셀 수 없다')
    d = torch.load(ck, map_location='cpu', weights_only=False)
    st = d.get('stats', d)
    return f"{float(st['K_LIM']):g}"


def _ana_kmax():
    """해석적 빔포밍의 탐색 상한 --- 함수 서명 기본값에서 읽는다."""
    import inspect
    from wavesplat.optimizer import initializer as _init
    hi = inspect.signature(_init.initialize_from_frames_v2).parameters['kx_range'].default[1]
    return f"{float(hi):g}"


def load(rel):
    p = os.path.join(TMP, rel)
    if not os.path.exists(p):
        raise SystemExit(f'⛔동결 판정이 없다: {p}')
    return json.load(open(p))


# ─────────────────────────────────────────────────────────────────────────────
# ★인용 원자 — (매크로, 배치 verdict, json 경로, 형식, 핀)
#   ⛔핀은 "오늘 값" 이다. 값이 바뀌면 **의도적으로** 갱신하라(래칫과 같은 규약).
#   ⚠`kind`: 'int' | 'pct'(부호 포함 소수1) | 'p'(과학표기) | 'ratio'(a/b)
# ─────────────────────────────────────────────────────────────────────────────
ATOMS = [
    # ── 헤드라인 (혼잡 층, 최강 적격 격자 기준선 대비)
    ('BOneBetter',  'B1_BASELINE_2026-08-22/verdict_b1.json', 'crowded.better', 'int', 62),
    ('BOneN',       'B1_BASELINE_2026-08-22/verdict_b1.json', 'crowded.n', 'int', 64),
    ('BOneRel',     'B1_BASELINE_2026-08-22/verdict_b1.json', 'crowded.rel_median_pct', 'pct', -38.6),
    ('BOneP',       'B1_BASELINE_2026-08-22/verdict_b1.json', 'crowded.p_one_sided', 'p', 1.13e-16),
    # ── 비선형 참값 · 신선 시드 확장 (확증)
    ('BTwelveBetter', 'B12_HOS_EXTEND_2026-08-23/verdict_b12.json', 'ship_vs_spec.better', 'int', 43),
    ('BTwelveN',      'B12_HOS_EXTEND_2026-08-23/verdict_b12.json', 'ship_vs_spec.n', 'int', 48),
    ('BTwelveRel',    'B12_HOS_EXTEND_2026-08-23/verdict_b12.json', 'ship_vs_spec.rel_median_pct', 'pct', -40.2),
    ('BTwelveP',      'B12_HOS_EXTEND_2026-08-23/verdict_b12.json', 'ship_vs_spec.p_one_sided', 'p', 6.84e-09),
    # ★음성대조 — 비교자에게도 기술이 있는가 (F1 교훈)
    ('BTwelveCmpBetter', 'B12_HOS_EXTEND_2026-08-23/verdict_b12.json', 'spec_vs_zero.better', 'int', 47),
    ('BTwelveCmpN',      'B12_HOS_EXTEND_2026-08-23/verdict_b12.json', 'spec_vs_zero.n', 'int', 48),
    # ── 표현 절제: 파수 연속성 (혼잡)
    ('BSixteenBetter', 'B16_KGRID_2026-08-24/verdict_b16.json', 'primary.better', 'int', 46),
    ('BSixteenN',      'B16_KGRID_2026-08-24/verdict_b16.json', 'primary.n', 'int', 48),
    ('BSixteenRel',    'B16_KGRID_2026-08-24/verdict_b16.json', 'primary.rel_median_pct', 'pct', -38.6),
    ('BSixteenP',      'B16_KGRID_2026-08-24/verdict_b16.json', 'primary.p_one_sided', 'p', 4.18e-12),
    ('BSixteenDose',   'B16_KGRID_2026-08-24/verdict_b16.json', 'dose.better', 'int', 47),
    # ── 표현 절제: 파수 연속성 (희소)
    ('BNineteenBetter', 'B19_KGRID_SPARSE_2026-08-25/verdict_b19.json', 'primary.better', 'int', 14),
    ('BNineteenN',      'B19_KGRID_SPARSE_2026-08-25/verdict_b19.json', 'primary.n', 'int', 16),
    ('BNineteenRel',    'B19_KGRID_SPARSE_2026-08-25/verdict_b19.json', 'primary.rel_median_pct', 'pct', -67.4),
    ('BNineteenP',      'B19_KGRID_SPARSE_2026-08-25/verdict_b19.json', 'primary.p_one_sided', 'p', 2.09e-03),
    # ── 사전 절제: 학습 패키지 < 해석적 패키지 (L2) ★%를 margin 으로 쓸 수 있는 성분
    ('BTwentyTwoBetter', 'B22_PRIOR_ABLATION_2026-08-27/verdict_b22.json', 'comparisons.L2.better', 'int', 45),
    ('BTwentyTwoN',      'B22_PRIOR_ABLATION_2026-08-27/verdict_b22.json', 'comparisons.L2.n', 'int', 48),
    ('BTwentyTwoRel',    'B22_PRIOR_ABLATION_2026-08-27/verdict_b22.json', 'comparisons.L2.rel_median_pct', 'pct', -36.3),
    ('BTwentyTwoP',      'B22_PRIOR_ABLATION_2026-08-27/verdict_b22.json', 'comparisons.L2.p_one_sided', 'p', 6.56e-11),
    # ⚠L2 의 우변(해석적 패키지)은 **참 제로필드를 못 넘는다** — 인용 시 병기 의무
    ('BTwentyTwoAnaZero', 'B22_PRIOR_ABLATION_2026-08-27/verdict_b22.json', 'neg_vs_zero.ANA.better', 'int', 15),
    ('BTwentyTwoLearnZero', 'B22_PRIOR_ABLATION_2026-08-27/verdict_b22.json', 'neg_vs_zero.LEARN.better', 'int', 48),
    # ── 센서 모형(raydrop) 강건성
    # ── ★[2026-09-02] 유일하게 살아있는 시간 계수(4차 감사 B9).
    #   ⛔프로세스간 지연은 전부 철회됐지만, **질의 렌더 단가**는 다른 양이고 측정돼 있다.
    #   ⚠병기 의무 2건: **격자 기반 실측 경로**의 계측이고(학습 렌더러 미측정),
    #   **초기화 이후 정상 상태**만 포괄한다(제안 단계 불포함).
    ('BudgetUsPerPoint', 'B6_BUDGET_2026-08-23/verdict_b6b.json',
     'pooled.per_point_s', 'us', 137.0),
    # ── ★[2026-09-02] `T6-BUDGET` --- **배포 학습 경로**의 프레임 예산.
    #   ⛔이 배치 전까지 학습 경로의 지연은 **0건 측정**이었고 논문이 그렇게 적고 있었다.
    #   결정표가 결과 전에 지정한 **가장 보수적인 칸** = 혼잡 · CPU 전용 · 1 스레드.
    ('FrameMsLearn', 'T6_BUDGET_2026-09-02/verdict_t6.json',
     ('cells', 'CPU|LEARN|crowded', 'frame_ms_median'), 'ms2', 5.91),
    ('FramePctLearn', 'T6_BUDGET_2026-09-02/verdict_t6.json',
     ('cells', 'CPU|LEARN|crowded', 'budget_frac_at_10hz'), 'pct100', 5.9),
    ('InitSLearn', 'T6_BUDGET_2026-09-02/verdict_t6.json',
     ('cells', 'CPU|LEARN|crowded', 'init_s_median'), 'ms2', 5.18),
    ('InitSAna', 'T6_BUDGET_2026-09-02/verdict_t6.json',
     ('cells', 'CPU|ANA|crowded', 'init_s_median'), 'ms2', 26.61),
    ('InitSLearnGpu', 'T6_BUDGET_2026-09-02/verdict_t6.json',
     ('cells', 'GPU|LEARN|crowded', 'init_s_median'), 'ms2', 4.86),
    ('BudgetFrameS', 'B6_BUDGET_2026-08-23/verdict_b6b.json', 'frame_budget_s', 'sec', 0.1),
    ('BFifteenN',   'B15_RAYDROP_2026-08-24/verdict_b15.json', 'by_raydrop.slope.n', 'int', 20),
    ('BFifteenRel', 'B15_RAYDROP_2026-08-24/verdict_b15.json', 'by_raydrop.slope.rel_median_pct', 'pct', -42.6),
    ('BFifteenBetter', 'B15_RAYDROP_2026-08-24/verdict_b15.json',
     'by_raydrop.slope.ship_beats_spec', 'int', 17),
    ('BFifteenP', 'B15_RAYDROP_2026-08-24/verdict_b15.json',
     'by_raydrop.slope.p_one_sided', 'p', 1.29e-03),
    # ★음성대조: 이 코너에서 **비교자에게도 기술이 있다**(F1 부류 방지)
    ('BFifteenCmp', 'B15_RAYDROP_2026-08-24/verdict_b15.json',
     'by_raydrop.slope.spec_beats_zero', 'int', 20),
    # ── 이득의 위치: 공간(M-10) ⚠rel_median 이 **분수**다(다른 배치와 단위가 다르다)
    ('MTenBetter', 'M10_GEOMETRY_2026-08-30/verdict_m10.json', 'primary.primary.better', 'int', 45),
    ('MTenN',      'M10_GEOMETRY_2026-08-30/verdict_m10.json', 'primary.primary.n', 'int', 48),
    ('MTenRel',    'M10_GEOMETRY_2026-08-30/verdict_m10.json', 'primary.primary.rel_median', 'frac', -31.7),
    ('MTenCmp',    'M10_GEOMETRY_2026-08-30/verdict_m10.json', 'primary.spec_vs_zero.better', 'int', 48),
    ('MTenP',      'M10_GEOMETRY_2026-08-30/verdict_m10.json', 'primary.primary.p_one_sided', 'p', 6.56e-11),
    # ★[2026-09-02] **절대** nRMSE. 4차 감사: 논문 52쪽에 우리 팔의 절대 정확도가 0회였다
    #   --- 인쇄된 유일한 절대값이 **철회된 기준선의 것**이었다.
    ('MTenAbsShip', 'M10_GEOMETRY_2026-08-30/verdict_m10.json', 'primary.abs_median.SHIP', 'abs3', 0.043),
    ('MTenAbsSpec', 'M10_GEOMETRY_2026-08-30/verdict_m10.json', 'primary.abs_median.SPEC25', 'abs3', 0.063),
    ('MTenAbsZero', 'M10_GEOMETRY_2026-08-30/verdict_m10.json', 'primary.abs_median.ZERO', 'abs3', 0.339),
    # ── 이득의 위치: 시간(T-4) ⚠비교자 붕괴를 **함께** 싣는다(F1 부류 방지)
    ('TFourBetter', 'T4_GEOMETRY_2026-08-30/verdict_t4.json', 'primary.better', 'int', 47),
    ('TFourN',      'T4_GEOMETRY_2026-08-30/verdict_t4.json', 'primary.n', 'int', 48),
    ('TFourRel',    'T4_GEOMETRY_2026-08-30/verdict_t4.json', 'primary.rel_median_pct', 'pct', -46.3),
    ('TFourCmp',    'T4_GEOMETRY_2026-08-30/verdict_t4.json', 'comparator.better', 'int', 34),
    ('TFourCmpAtZero', 'T4_GEOMETRY_2026-08-30/verdict_t4.json',
     ('ladder', '0.0', 'spec_vs_zero', 'better'), 'int', 47),

    # ═════════════════════════════════════════════════════════════════════════
    # ★[2026-09-02 P9] 레거시 표·그림 처분에 필요한 원자.
    #   사유: 논문의 절제·강건성 표 다섯 개와 그림 넷이 **철회된 코퍼스** 산물이라
    #   옮긴다. 그 자리에 들어가는 대체 표는 **전부 동결 verdict 에서만** 나와야 한다.
    #   ⛔손으로 옮겨 적으면 이 저장소가 반복해 대가를 치른 결함으로 되돌아간다.
    # ═════════════════════════════════════════════════════════════════════════

    # ── 해상상태 2x2 (B-26). ⚠F1: 승수 한 척도에서만 "최강" 이다 --- **절대 정확도는 반대**.
    #    ⟹ 절대 중앙값 두 개를 **함께** 낸다(코너 순위 주장을 문면에서 불가능하게 만든다).
    ('BTwentySixBetter', 'B26_SEASTATE_2026-08-27/verdict_b26.json',
     'corners.SS-REAL.ship_vs_spec.better', 'int', 24),
    ('BTwentySixN', 'B26_SEASTATE_2026-08-27/verdict_b26.json',
     'corners.SS-REAL.ship_vs_spec.n', 'int', 24),
    ('BTwentySixRel', 'B26_SEASTATE_2026-08-27/verdict_b26.json',
     'corners.SS-REAL.ship_vs_spec.rel_median_pct', 'pct', -42.4),
    # ★F1 병기 원자 --- 비교자의 기술이 이 코너에서 붕괴한다(-16.2% -> -4.0%)
    ('BTwentySixCmpRel', 'B26_SEASTATE_2026-08-27/verdict_b26.json',
     'corners.SS-REAL.spec_vs_zero.rel_median_pct', 'pct', -4.0),
    ('BTwentySixAnchorCmpRel', 'B26_SEASTATE_2026-08-27/verdict_b26.json',
     'corners.SS-BF4.spec_vs_zero.rel_median_pct', 'pct', -16.2),
    # ★F1 병기 원자 --- 절대 정확도는 **반대 방향**(0.1366 vs 0.1176)
    ('BTwentySixAbs', 'B26_SEASTATE_2026-08-27/verdict_b26.json',
     'corners.SS-REAL.medians.ship', 'abs3', 0.137),
    ('BTwentySixAnchorAbs', 'B26_SEASTATE_2026-08-27/verdict_b26.json',
     'corners.SS-BF4.medians.ship', 'abs3', 0.118),
    ('BTwentySixCmpBetter', 'B26_SEASTATE_2026-08-27/verdict_b26.json',
     'corners.SS-REAL.spec_vs_zero.better', 'int', 23),

    # ── 실측 음성 셋 (경계 표). ⛔전부 **참 제로필드 대비 승수**이지 팔간 비교가 아니다.
    ('BSeventeenBeats', 'B17_REALNOISE_2026-08-25/verdict_b17.json',
     'beats_zero.RANGEANG.wins', 'int', 4),
    ('BSeventeenN', 'B17_REALNOISE_2026-08-25/verdict_b17.json',
     'beats_zero.RANGEANG.n', 'int', 12),
    ('BEighteenBeats', 'B18_DENSITY_NOISE_2026-08-25/verdict_b18.json',
     'primary.D_BOTH.beats_zero', 'int', 8),
    ('BEighteenN', 'B18_DENSITY_NOISE_2026-08-25/verdict_b18.json',
     'primary.D_BOTH.n', 'int', 12),
    ('BEighteenConfirm', 'B18_DENSITY_NOISE_2026-08-25/verdict_b18.json',
     'thresholds.confirm', 'int', 11),
    ('BFourteenBBeats', 'B14_OBSMODEL_2026-08-24/verdict_b14b.json',
     'h1.IID.zero_better', 'int', 8),
    ('BFourteenBN', 'B14_OBSMODEL_2026-08-24/verdict_b14b.json',
     'h1.IID.n', 'int', 12),

    # ── 앙상블 대칭 (2026-09-01 두 배치)
    ('SEnsBetter', 'S_ENS_2026-09-01/verdict_sens.json', 'contrasts.PRIMARY.better', 'int', 55),
    ('SEnsN', 'S_ENS_2026-09-01/verdict_sens.json', 'contrasts.PRIMARY.n', 'int', 64),
    ('SEnsRel', 'S_ENS_2026-09-01/verdict_sens.json',
     'contrasts.PRIMARY.rel_median_pct', 'pct', -17.9),
    ('SEnsP', 'S_ENS_2026-09-01/verdict_sens.json',
     'contrasts.PRIMARY.p_one_sided', 'p', 1.77e-09),
    # ⚠희소는 `UNDECIDED` --- **혼잡만 인용하면 층 한정이 사라진다** ⟹ 함께 낸다.
    ('SEnsSparseBetter', 'S_ENS_2026-09-01/verdict_sens.json',
     'contrasts.SEC-A.better', 'int', 35),
    ('SEnsSparseN', 'S_ENS_2026-09-01/verdict_sens.json', 'contrasts.SEC-A.n', 'int', 64),
    ('SEnsNominal', 'S_ENS_2026-09-01/verdict_sens.json', 'thresholds.NOMINAL', 'int', 40),

    ('TFiveBetter', 'T5_ENS_2026-09-01/verdict_t5.json', 'contrasts.PRIMARY.better', 'int', 28),
    ('TFiveN', 'T5_ENS_2026-09-01/verdict_t5.json', 'contrasts.PRIMARY.n', 'int', 36),
    ('TFiveRel', 'T5_ENS_2026-09-01/verdict_t5.json',
     'contrasts.PRIMARY.rel_median_pct', 'pct', -8.5),
    ('TFiveP', 'T5_ENS_2026-09-01/verdict_t5.json',
     'contrasts.PRIMARY.p_one_sided', 'p', 5.97e-04),
    ('TFiveFarBetter', 'T5_ENS_2026-09-01/verdict_t5.json', 'contrasts.SEC-A.better', 'int', 35),
    ('TFiveFarRel', 'T5_ENS_2026-09-01/verdict_t5.json',
     'contrasts.SEC-A.rel_median_pct', 'pct', -11.1),
]

FMT = {
    'int': lambda x: str(int(x)),
    'pct': lambda x: f'{x:+.1f}\\%'.replace('+', '') if x < 0 else f'{x:+.1f}\\%',
    'p': lambda x: _sci(x),
    # ⚠`M-10` 은 rel 을 **분수**로 저장한다 — 단위를 여기서 흡수한다(⛔논문에서 곱하지 마라).
    'frac': lambda x: f'{x*100:+.1f}\\%'.replace('+', '') if x < 0 else f'{x*100:+.1f}\\%',
    # ⚠**절대** nRMSE --- 퍼센트가 아니다. F1 이 가르친 것: 승수와 절대는 순위를 뒤집는다.
    'abs3': lambda x: f'{x:.3f}',
    'us': lambda x: f'{x * 1e6:.0f}',
    'sec': lambda x: f'{x:g}',
    'ms2': lambda x: f'{x:.2f}',
    'pct100': lambda x: f'{x * 100:.1f}',
}


def _sci(x):
    from math import floor, log10
    if x == 0:
        return '0'
    e = int(floor(log10(abs(x))))
    m = x / 10 ** e
    return f'{m:.2f}\\times 10^{{{e}}}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--check', action='store_true',
                    help='커밋된 paper_constants.tex 가 생성물과 같은가 (쓰지 않는다)')
    a = ap.parse_args()

    cache, rows, bad = {}, [], []
    for name, rel, path, kind, pin in ATOMS:
        v = cache.setdefault(rel, load(rel))
        try:
            raw = _get(v, path)
        except (KeyError, IndexError, TypeError) as e:
            bad.append((name, f'경로 없음 {path}: {e}'))
            continue
        # 핀 대조 — 정수는 정확히, 실수는 표시 자릿수에서
        if kind == 'int':
            ok = int(raw) == int(pin)
        elif kind in ('pct', 'frac'):
            val = raw * 100 if kind == 'frac' else raw
            ok = abs(round(val, 1) - pin) < 1e-9
        elif kind == 'abs3':
            ok = abs(round(raw, 3) - pin) < 1e-9
        elif kind == 'us':
            ok = abs(round(raw * 1e6) - pin) < 1e-9
        elif kind == 'sec':
            ok = abs(raw - pin) < 1e-12
        elif kind == 'ms2':
            ok = abs(round(raw, 2) - pin) < 1e-9
        elif kind == 'pct100':
            ok = abs(round(raw * 100, 1) - pin) < 1e-9
        else:
            ok = abs(raw - pin) / abs(pin) < 0.005
        if not ok:
            bad.append((name, f'핀 {pin} != 실측 {raw}'))
        rows.append((name, FMT[kind](raw), rel.split('/')[0]))

    if a.show or bad:
        for n, val, src in rows:
            print(f'  {n:22s} = {val:22s}  [{src}]')
    if bad:
        print('\n⛔핀 불일치:')
        for n, why in bad:
            print(f'   {n}: {why}')
        raise AssertionError(f'{len(bad)}건 — 배치가 재분석됐다면 ATOMS 의 핀을 '
                             '**의도적으로** 갱신하라(래칫 규약).')

    # ── ★배포 상수도 같은 파일에 낸다(2026-09-01, 2차 감사).
    #    사유: 논문이 `\lambda_1 = 10^{-3}` 을 **손으로** 적고 있었는데 배포값은 1e-4 다
    #    (2026-08-11 변경). 손으로 적힌 배포 상수는 조용히 낡는다 — 이 저장소의 반복 결함.
    from wavesplat.optimizer.sequential_update import (
        DEFAULT_ANALYTIC_ENSEMBLE_K, DEFAULT_LEARNED_ENSEMBLE_K, LAMBDA_L1_DEFAULT,
        VARPRO_STEPS_DEFAULT, LEARNED_INIT_ITERS_DEFAULT)
    from wavesplat.run import DEFAULT_CONFIG as _DC
    from math import log10
    dep = [
        ('DepLambdaOne', f'10^{{{int(round(log10(LAMBDA_L1_DEFAULT)))}}}'),
        ('DepVarproSteps', str(VARPRO_STEPS_DEFAULT)),
        ('DepJointFitIters', str(LEARNED_INIT_ITERS_DEFAULT)),
        ('DepAnalyticEnsK', str(DEFAULT_ANALYTIC_ENSEMBLE_K)),
        ('DepLearnedEnsK', str(DEFAULT_LEARNED_ENSEMBLE_K)),
        ('DepTrackingLr', f"{_DC['tracking_lr']:g}"),
        ('DepTrackingSteps', str(_DC['tracking_steps'])),
        ('DepMaxN', str(_DC['max_n'])),
        # ★테스트 수는 자주 바뀐다 — 손으로 적으면 반드시 낡는다(3차 감사: 196 vs 실측 261).
        ('DeployNumTestsPaper', _count_tests()),
        # ── ★[2026-09-02] 학습 제안의 **지지 상한**. 체크포인트 안에 있다.
        #   코드 감사 D8: 논문에 이 상한이 한 번도 적혀 있지 않았고, 손으로 적으면
        #   체크포인트를 갈 때 조용히 낡는다(H2 가 이미 열려 있는 축이다).
        ('DepKLim', _k_lim()),
        ('CkptSha', _ckpt_sha()),
        # ── 해석적 대안의 탐색 범위(대비용). 함수 서명의 기본값에서 읽는다.
        ('DepAnaKMax', _ana_kmax()),
    ]
    dep += list(_prior_arch().items())
    rows += [(k, v, 'DEPLOY') for k, v in dep]

    body = ['% ⛔자동 생성 — 손으로 고치지 마라.',
            '%    생성: python3 scripts/gen_paper_constants.py',
            '%    출처: 동결 verdict_*.json (배치명은 각 줄 주석에)',
            '%    ★이 파일이 존재하는 이유: paper/wavesplat.tex 가 철회된 헤드라인을',
            '%      라이브 22줄에 **손으로** 인쇄하고 있었고 그 줄들이 독립적으로 낡았다.', '']
    for n, val, src in rows:
        body.append(f'\\newcommand{{\\{n}}}{{{val}}}  % {src}')
    text = '\n'.join(body) + '\n'
    # ★2차 감사 지적: 생성기를 **안 돌리고** tex 를 손으로 고치면 무탐지였다.
    #   `--check` 는 아무것도 쓰지 않고 **커밋된 파일과 대조**만 한다.
    if a.check:
        if not os.path.exists(OUT):
            print(f'⛔없다: {OUT} — 생성기를 돌려라')
            return 1
        if open(OUT).read() != text:
            print('⛔`paper_constants.tex` 가 생성물과 다르다 (손편집됐거나 낡았다).')
            print('   수리: python3 scripts/gen_paper_constants.py')
            return 1
        print(f'✅{len(rows)} 상수 · 커밋된 파일이 생성물과 일치')
        return 0
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, 'w').write(text)
    if not a.show:
        for n, val, src in rows:
            print(f'  \\{n} = {val}')
    print(f'\n✅{len(rows)} 상수 · 핀 불일치 0 → {os.path.relpath(OUT, ROOT)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
