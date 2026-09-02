"""`R2-STEREONOISE` 규칙 단일출처. ⛔`PREREG.md` 가 정본이고 이 파일이 코드로 옮긴다."""
import os
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))

# 게이지 파일 ↔ 스테레오 레코드 (파일명의 UTC 로 맺는다)
PAIRS = [
    ('WaveGauge_BS01_20111001_16:18:00.mat', 'BS11a'),
    ('WaveGauge_BS02_20111004_11:38:00.mat', 'BS11b'),
    ('WaveGauge_BS03_20111004_13:07:00.mat', 'BS11c'),
    ('WaveGauge_BS04_20111004_15:30:00.mat', 'BS11d'),
    ('WaveGauge_BS05_20130922_13:00:01.mat', 'BS13a'),
    ('WaveGauge_BS06_20130925_12:15:01.mat', 'BS13b'),
    ('WaveGauge_BS07_20130930_10:20:01.mat', 'BS13c'),
]
N = len(PAIRS)
NPERSEG = 1024
N_SAMPLE_PTS = 400
SAMPLE_SEED = 20260902          # ⛔결과 전 고정
N_PERM = 8                      # ★귀무분포 반복 (같은 다중성)
HF_CUT_HZ = 1.0                 # "고주파" 의 정의 — 결과 전 고정
WAVE_BAND = (0.05, 0.80)        # Tp/Hm0 집계 대역
THRESH = N                      # ★전승만 유의 (p = 0.0078)
SYNC_TOL_SAMPLES = 1.0          # G-1
GAUGE_XCORR_MIN = 0.90          # G-2
TP_REL_MAX = 0.10               # SEC-a 사전 기대(판정 아님)


def sign_test_one_sided(w, n):
    return sum(comb(n, k) for k in range(w, n + 1)) / 2.0 ** n


def label(w):
    """⛔결과 전 동결 판정표(PREREG §3)."""
    if w >= THRESH:
        return ('STEREO-NOISE-FLOOR-MEASURED',
                f'{w}/{N} — 재구성면 고주파 잡음의 **상한**을 보고한다')
    return ('STEREO-NOISE-UNDECIDED',
            f'{w}/{N} — ⛔"잡음이 없다" 금지 · n={N} 은 전승만 유의(6/7 은 p=0.0625)')


def gates(rows):
    ok = [r for r in rows if 'error' not in r]
    A = []
    A.append(('G-0 쌍 완비', len(ok) == N, f'{len(ok)}/{N}'))
    A.append(('G-0b 에러 0', len(rows) == len(ok), str(len(rows) - len(ok))))

    bad = [r['record'] for r in ok if abs(r['sync_off_samples']) > SYNC_TOL_SAMPLES]
    A.append(('G-1 시각 동기 ±1 샘플', not bad, ';'.join(bad[:3])))

    bad = [r['record'] for r in ok if r['gauge_xcorr_min'] < GAUGE_XCORR_MIN]
    A.append(('G-2 게이지 내부 상관 ≥ 0.9', not bad,
              f"min={min((r['gauge_xcorr_min'] for r in ok), default=float('nan')):.3f}"))

    # ★G-3 — 위치 대응이 널임을 **같은 다중성의 귀무분포**로 강제한다
    #   ⛔스모크 정정: 초판은 「~4만 번의 최대」를 **단일 쌍**의 자기상관과 비교했다 =
    #   사과와 오렌지. 이제 게이지를 순환이동시켜 같은 탐색을 돌린 **최대끼리** 비교한다.
    bad = [r['record'] for r in ok if r.get('loc_exceeds_null')]
    A.append(('★G-3 위치 대응 널 (r_max ≤ 순환이동 귀무 최대)', not bad,
              (f"r_max {min(r['loc_rmax'] for r in ok):.3f}–{max(r['loc_rmax'] for r in ok):.3f}"
               f" vs null {min(r['loc_null_max'] for r in ok):.3f}–"
               f"{max(r['loc_null_max'] for r in ok):.3f}") if ok else '--'))

    bad = [r['record'] for r in ok
           if r['f_max_used'] > min(r['fs_gauge'], r['fs_stereo']) / 2 + 1e-9]
    A.append(('G-4 공통 Nyquist 안에서만 비교', not bad, ';'.join(bad[:3])))

    bad = [r['record'] for r in ok if r.get('fs_source') != 'Time_UTC']
    A.append(('G-5 표본율을 Time_UTC 에서 도출 (⛔info.fps 는 거짓)', not bad,
              ';'.join(bad[:3])))

    A.append(('G-6 표본 시드 결과 전 고정',
              all(r.get('sample_seed') == SAMPLE_SEED for r in ok), str(SAMPLE_SEED)))
    A.append(('G-COV 모든 게이트가 자료를 봤다', len(ok) > 0, f'rows={len(ok)}'))
    return A


def blocked_by(gate_rows):
    """★자가진단 대상 — 실패한 게이트 이름을 **비어있지 않게** 돌려줘야 한다."""
    return [g[0] for g in gate_rows if not g[1]]
