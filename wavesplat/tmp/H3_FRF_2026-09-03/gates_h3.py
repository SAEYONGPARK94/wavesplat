"""`H3-FRF` 규칙 단일출처. ⛔`PREREG.md` 가 정본이고 이 파일이 코드로 옮긴다."""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
FRF = os.path.join(ROOT, 'tmp', 'frf_lidar_2026-08-01')
WS = os.path.join(FRF, 'ws')
for p in (FRF, ROOT, os.path.join(ROOT, 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)

# ── 제5장이 스스로 적은 문턱 (⛔내가 고른 수가 아니다) ─────────────
MRE_THRESH = 0.20
# ── 셀 구성 규약 (⛔`measure_dropout.py` 와 같은 값) ──────────────
N_RING = 32
AZ_STEP = 0.30
MIN_FRAMES = 30
D_REF = 125.0
# ── 대조 ─────────────────────────────────────────────────────────
N_PERM = 200                  # G-POS 귀무 반복
PERM_SEED = 20260903          # ⛔결과 전 고정
DOM_MAX = 0.20                # G-DOM: 한 파일이 전체 셀의 20% 초과 금지
N_FILES = 30
MODEL_SRC = os.path.join(ROOT, 'wavesim', 'raydrop_uniform.py')


def model_md5():
    return hashlib.md5(open(MODEL_SRC, 'rb').read()).hexdigest()[:12]


def label(mre):
    """⛔결과 전 동결 판정표(PREREG §2). 문턱은 제5장의 것이다."""
    if mre <= MRE_THRESH:
        return ('OBSMODEL-FAITHFUL',
                f'MRE = {mre:.4f} ≤ {MRE_THRESH} — H3 가 실물 LiDAR 에서 충족된다')
    return ('OBSMODEL-NOT-FAITHFUL',
            f'MRE = {mre:.4f} > {MRE_THRESH} — 이 레짐에서 이 모형은 반환율을 예측하지 못한다')


def gates(rows, summary):
    ok = [r for r in rows if 'error' not in r]
    A = []
    A.append((f'G-FILES {N_FILES} 파일 전건 · 오류 0',
              len(ok) == N_FILES and len(rows) == len(ok), f'{len(ok)}/{N_FILES}'))

    # ★G-POS 양성 대조 — 측정 반환율에 구조가 있는가
    A.append(('★G-POS 양성 대조: 측정 반환율의 십분위 퍼짐 > 섞은 귀무 p95',
              bool(summary.get('pos_spread', 0) > summary.get('pos_null_p95', 1)),
              f"퍼짐 {summary.get('pos_spread', float('nan')):.4f} vs "
              f"귀무p95 {summary.get('pos_null_p95', float('nan')):.4f}"))

    # ★G-NEG 음성 대조 — 섞은 모형이 더 나빠야 한다
    A.append(('★G-NEG 음성 대조: 섞은 모형의 MRE p5 > 실제 MRE',
              bool(summary.get('mre_shuf_p5', 0) > summary.get('mre', 1)),
              f"섞음p5 {summary.get('mre_shuf_p5', float('nan')):.4f} vs "
              f"실제 {summary.get('mre', float('nan')):.4f}"))

    A.append(('★G-IMPORT 모형을 wavesim 에서 임포트 (재구현 아님)',
              all(r.get('model_md5') == model_md5() for r in ok)
              and all(r.get('model_imported') for r in ok), model_md5()))

    A.append(('G-BIAS 파일별 유효 프레임 비율 기록 (반환율은 상향 편향)',
              all('valid_frac' in r for r in ok),
              f"{min((r['valid_frac'] for r in ok), default=float('nan')):.3f}–"
              f"{max((r['valid_frac'] for r in ok), default=float('nan')):.3f}"))

    tot = sum(r['ncell'] for r in ok) or 1
    worst = max((r['ncell'] / tot for r in ok), default=1.0)
    A.append((f'G-DOM 한 파일이 전체 셀의 {DOM_MAX:.0%} 이하', worst <= DOM_MAX,
              f'최대 {worst:.3f}'))

    A.append(('G-SCALE 전역 스케일 s 가 하나이고 기록됨',
              summary.get('scale') is not None,
              f"s = {summary.get('scale', float('nan')):.5f}"))

    probe = blocked_by([('X-거짓', False, ''), ('Y-참', True, '')])
    A.append(('★G-BLOCK blocked_by 자가진단', probe == ['X-거짓'], repr(probe)))
    A.append(('G-COV 모든 게이트가 자료를 봤다', len(ok) > 0, f'files={len(ok)}'))
    return A


def blocked_by(gate_rows):
    """★자가진단 대상 — 실패한 게이트 이름을 **비어있지 않게** 돌려줘야 한다."""
    return [g[0] for g in gate_rows if not g[1]]
