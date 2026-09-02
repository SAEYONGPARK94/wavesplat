"""`R2b-STEREONOISE` 규칙 단일출처. ⛔`PREREG.md` 가 정본이고 이 파일이 코드로 옮긴다.

⛔쌍·표본율 규약은 `gates_r2` 를 **임포트**한다(손복사 금지 — 저장소의 반복 결함).
"""
import os
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for p in (os.path.join(ROOT, 'tmp', 'R2_STEREONOISE_2026-09-02'), ROOT,
          os.path.join(ROOT, 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)
import gates_r2 as R2                                               # noqa: E402

PAIRS = list(R2.PAIRS)
N = R2.N
NPERSEG = R2.NPERSEG
N_SAMPLE_PTS = R2.N_SAMPLE_PTS
SAMPLE_SEED = R2.SAMPLE_SEED
HF_CUT_HZ = R2.HF_CUT_HZ
WAVE_BAND = R2.WAVE_BAND
GAUGE_XCORR_MIN = R2.GAUGE_XCORR_MIN
SYNC_TOL_SAMPLES = R2.SYNC_TOL_SAMPLES
THRESH = N                      # ★전승만 유의 (p = 0.0078)

# ── 이 배치가 새로 정하는 것 (PREREG §1·§2, 결과 전 고정) ──────────────
N_NULL = 32                     # FT 위상무작위화 반복
NULL_SEED = 20260902
PCTL = 95.0                     # "귀무를 초과한다" 의 정의
PEAK_FRAC = 0.9                 # 봉우리 피복률의 정의 (보고 전용)
CTRL_LAG_S = 3.5                # ★양성 대조에 **주입**하는 lag
CTRL_LAG_TOL_S = 0.5            # G-LAG — 회수 허용오차


def sign_test_one_sided(w, n):
    return sum(comb(n, k) for k in range(w, n + 1)) / 2.0 ** n


def label(w):
    """⛔결과 전 동결 판정표(PREREG §3)."""
    if w >= THRESH:
        return ('STEREO-NOISE-FLOOR-MEASURED',
                f'{w}/{N} — 1 Hz 초과 초과분산의 **상한**을 보고한다')
    return ('STEREO-NOISE-UNDECIDED',
            f'{w}/{N} — ⛔"잡음이 없다" 금지 · n={N} 은 전승만 유의(6/7 은 p=0.0625)')


def gates(rows):
    ok = [r for r in rows if 'error' not in r]
    A = []
    A.append(('G-0 쌍 완비', len(ok) == N, f'{len(ok)}/{N}'))
    A.append(('G-0b 에러 0', len(rows) == len(ok), str(len(rows) - len(ok))))

    # ★G-DETECT — 탐지기가 **있는 대응을 찾는가**(양성 대조). ⛔실패 시 판정 없음.
    nd = sum(1 for r in ok if r['ctrl_exceeds_null'])
    A.append(('★G-DETECT 양성 대조가 자기 귀무를 초과', nd == len(ok) and len(ok) == N,
              f'{nd}/{len(ok)}'))

    # ★G-CTRLPT — 대조점이 **탐색되는 부분격자 위**에 있는가.
    #   ⛔스모크가 이 결함을 잡았다: 전체 격자에서 고르면 참 대응점이 탐색되지 않아
    #   `G-DETECT` 가 **탐지기가 아니라 내 대조**의 실패로 죽는다.
    A.append(('★G-CTRLPT 대조점이 탐색 부분격자 위',
              all(r.get('ctrl_point_on_grid') for r in ok) and len(ok) == N,
              f"{sum(1 for r in ok if r.get('ctrl_point_on_grid'))}/{len(ok)}"))

    # ★G-LAG — 주입한 lag 를 회수했는가. ⛔없으면 G-DETECT 는 "아무 값이나 컸다" 와 같다.
    bad = [r['record'] for r in ok
           if abs(r['ctrl_lag_s'] - CTRL_LAG_S) > CTRL_LAG_TOL_S]
    A.append((f'★G-LAG 주입 lag {CTRL_LAG_S}s 를 ±{CTRL_LAG_TOL_S}s 로 회수',
              not bad, ';'.join(bad[:3]) or
              (f"최대 오차 {max(abs(r['ctrl_lag_s'] - CTRL_LAG_S) for r in ok):.2f}s"
               if ok else '--')))

    # ★G-3 — 실제 게이지로는 위치 대응이 **널**이다
    bad = [r['record'] for r in ok if r['obs_exceeds_null']]
    A.append(('★G-3 위치 대응 널 (r_max ≤ FT 귀무 p95)', not bad,
              (f"r_max {min(r['obs_rmax'] for r in ok):.3f}–{max(r['obs_rmax'] for r in ok):.3f}"
               f" vs p95 {min(r['obs_null_p95'] for r in ok):.3f}–"
               f"{max(r['obs_null_p95'] for r in ok):.3f}") if ok else '--'))

    bad = [r['record'] for r in ok if abs(r['sync_off_samples']) > SYNC_TOL_SAMPLES]
    A.append(('G-1 시각 동기 ±1 샘플', not bad, ';'.join(bad[:3])))

    bad = [r['record'] for r in ok if r['gauge_xcorr_min'] < GAUGE_XCORR_MIN]
    A.append(('G-2 게이지 내부 상관 ≥ 0.9', not bad,
              f"min={min((r['gauge_xcorr_min'] for r in ok), default=float('nan')):.3f}"))

    bad = [r['record'] for r in ok
           if r['f_max_used'] > min(r['fs_gauge'], r['fs_stereo']) / 2 + 1e-9]
    A.append(('G-4 공통 Nyquist 안에서만 비교', not bad, ';'.join(bad[:3])))

    bad = [r['record'] for r in ok if r.get('fs_source') != 'Time_UTC']
    A.append(('G-5 표본율을 Time_UTC 에서 도출 (⛔info.fps 는 거짓)', not bad, ';'.join(bad[:3])))

    A.append(('G-6 표본 시드 결과 전 고정',
              all(r.get('sample_seed') == SAMPLE_SEED for r in ok), str(SAMPLE_SEED)))
    A.append((f'G-NULLB B={N_NULL} · 시드 {NULL_SEED}',
              all(r.get('n_null') == N_NULL and r.get('null_seed') == NULL_SEED for r in ok),
              f'{N_NULL}/{NULL_SEED}'))
    A.append(('G-COV 모든 게이트가 자료를 봤다', len(ok) > 0, f'rows={len(ok)}'))
    return A


def blocked_by(gate_rows):
    """★자가진단 대상 — 실패한 게이트 이름을 **비어있지 않게** 돌려줘야 한다."""
    return [g[0] for g in gate_rows if not g[1]]
