"""`R2c-STEREONOISE` 규칙 단일출처. ⛔`PREREG.md` 가 정본.

⛔쌍·귀무·표본 규약은 `gates_r2b` 를 **임포트**한다 — 이 배치가 바꾸는 것은
**시각 정렬 하나**(+양성 대조의 시간기준)이고, 재선언하면 그 주장이 거짓이 된다.
"""
import os
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
R2B = os.path.join(ROOT, 'tmp', 'R2B_STEREONOISE_2026-09-02')
for p in (R2B, os.path.join(ROOT, 'tmp', 'R2_STEREONOISE_2026-09-02'), ROOT,
          os.path.join(ROOT, 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)
import gates_r2b as B                                               # noqa: E402

PAIRS, N = list(B.PAIRS), B.N
NPERSEG, N_SAMPLE_PTS, SAMPLE_SEED = B.NPERSEG, B.N_SAMPLE_PTS, B.SAMPLE_SEED
HF_CUT_HZ, WAVE_BAND, GAUGE_XCORR_MIN = B.HF_CUT_HZ, B.WAVE_BAND, B.GAUGE_XCORR_MIN
N_NULL, NULL_SEED, PCTL, PEAK_FRAC = B.N_NULL, B.NULL_SEED, B.PCTL, B.PEAK_FRAC
CTRL_LAG_S, CTRL_LAG_TOL_S = B.CTRL_LAG_S, B.CTRL_LAG_TOL_S

# ★[동결 전 스모크가 바꿨다] 선행은 ±20 s 를 **0.25 s** 로 훑었다. 그런데 상관 정점이
#   **0.1 s 안에서 0.86 → 0.40** 으로 떨어진다 ⟹ 격자가 정점을 **넘어 다닌다**(실측 확인:
#   참 정점 404.233 에서 r=0.8555 인데 0.25 격자 최대는 **0.4026**).
#   ⟹ 정렬을 자료에서 알므로 **좁고 촘촘하게** 훑는다. 두 상수 모두 **의미에서 도출**한다.
LAG_HALFSPAN_S = 1.0         # 시계가 1 표본 안에서 맞으므로 ±1 s 면 넉넉하다
LAG_STEP_DIV = 20            # 공통 Nyquist 주기의 1/20 ⟹ 정점을 분해한다
# ⚠다중성이 선행보다 **낮아진다**(161 → 약 101 lag) ⟹ 귀무 최대가 내려가
#   *"널이다"* 라는 결론이 **더 어려워진다**. 방향이 우리에게 불리한 쪽이라 그대로 쓴다.


def lag_step(fs_gauge, fs_stereo):
    """★손으로 고르지 않는다 --- 공통 Nyquist 가 정한다."""
    f_nyq = min(fs_gauge, fs_stereo) / 2.0
    return 1.0 / (LAG_STEP_DIV * f_nyq)
CLOCK_TOL_SAMPLES = 1.0      # ★G-1 — 절대 시각 차가 게이지 1 표본 이내
GAUGE_DIR = os.path.join(ROOT, 'data', 'real_bergamasco', 'wavegauge')


def sign_test_one_sided(w, n):
    return sum(comb(n, k) for k in range(w, n + 1)) / 2.0 ** n


def label(n_exceed):
    """⛔결과 전 동결 판정표(PREREG §2)."""
    if n_exceed == 0:
        return ('POSITION-NULL-CONFIRMED',
                f'0/{N} 초과 — R2b 의 스펙트럼 대조가 쓴 **면허가 자료로 선다**')
    return ('POSITION-CORRESPONDENCE-DETECTED',
            f'{n_exceed}/{N} 초과 — ⛔R2b 의 위치 무관 비교 정당성이 무너진다')


def gates(rows):
    ok = [r for r in rows if 'error' not in r]
    A = []
    A.append(('G-0 쌍 완비', len(ok) == N, f'{len(ok)}/{N}'))
    A.append(('G-0b 에러 0', len(rows) == len(ok), str(len(rows) - len(ok))))

    # ★G-1 — 선행의 항진명제(`tg[0]*0.0`)를 대체한다. **자료에서 도출**한 두 절대 시각.
    bad = [r['record'] for r in ok
           if abs(r['clock_offset_s']) > CLOCK_TOL_SAMPLES / r['fs_gauge'] + 1e-9]
    A.append((f'★G-1 절대 시각 차 ≤ 게이지 {CLOCK_TOL_SAMPLES} 표본 (자료 도출)', not bad,
              (f"최대 {max(abs(r['clock_offset_s']) for r in ok):.3f} s" if ok else '--')))

    # ★G-SPAN — 선행이 어긴 조건: 탐색 구간이 도출된 정렬을 **포함**하는가
    bad = [r['record'] for r in ok if not r['span_contains_alignment']]
    A.append(('★G-SPAN lag 탐색이 도출 정렬을 포함', not bad, ';'.join(bad[:3]) or 'OK'))

    bad = [r['record'] for r in ok
           if abs(r['ctrl_lag_recovered_s'] - r['ctrl_lag_injected_s']) > CTRL_LAG_TOL_S]
    A.append((f'★G-RECOVER 양성 대조 lag 회수 ±{CTRL_LAG_TOL_S}s (절대 시각 합성)',
              not bad and all(r.get('ctrl_timebase') == 'absolute' for r in ok),
              ';'.join(bad[:3]) or 'OK'))

    nd = sum(1 for r in ok if r['ctrl_exceeds_null'])
    A.append(('★G-DETECT 양성 대조가 자기 귀무를 초과', nd == len(ok) and len(ok) == N,
              f'{nd}/{len(ok)}'))

    bad = [r['record'] for r in ok if r['gauge_xcorr_min'] < GAUGE_XCORR_MIN]
    A.append(('G-2 게이지 내부 상관 ≥ 0.9', not bad,
              f"min={min((r['gauge_xcorr_min'] for r in ok), default=float('nan')):.3f}"))
    bad = [r['record'] for r in ok
           if r['f_max_used'] > min(r['fs_gauge'], r['fs_stereo']) / 2 + 1e-9]
    A.append(('G-4 공통 Nyquist 안에서만 비교', not bad, ';'.join(bad[:3])))
    bad = [r['record'] for r in ok if r.get('fs_source') != 'Time_UTC']
    A.append(('G-5 표본율을 Time_UTC 에서 도출', not bad, ';'.join(bad[:3])))
    A.append((f'G-NULLB B={N_NULL} · 시드 {NULL_SEED} (gates_r2b 임포트)',
              all(r.get('n_null') == N_NULL and r.get('null_seed') == NULL_SEED for r in ok),
              f'{N_NULL}/{NULL_SEED}'))
    probe = blocked_by([('X-거짓', False, ''), ('Y-참', True, '')])
    A.append(('★G-BLOCK blocked_by 자가진단', probe == ['X-거짓'], repr(probe)))
    A.append(('G-COV 모든 게이트가 자료를 봤다', len(ok) > 0, f'rows={len(ok)}'))
    return A


def blocked_by(gate_rows):
    """★자가진단 대상 — 실패한 게이트 이름을 **비어있지 않게** 돌려줘야 한다."""
    return [g[0] for g in gate_rows if not g[1]]
