"""`R3-GEOMETRY` 규칙 단일출처. ⛔`PREREG.md` 가 정본."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
FULL = os.path.join(ROOT, 'data', 'real_bergamasco_full')

# (full 파일 접두, `_short` 레코드 키)
PAIRS = [
    ('BS_2011_2011-10-01_16-18-00_15Hz', 'BS11a'),
    ('BS_2011_2011-10-04_11-38-00_12Hz', 'BS11b'),
    ('BS_2011_2011-10-04_13-07-00_12Hz', 'BS11c'),
    ('BS_2011_2011-10-04_15-30-00_12Hz', 'BS11d'),
    ('BS_2013_2013-09-22_13-00-01_10Hz', 'BS13a'),
    ('BS_2013_2013-09-25_12-15-01_12Hz', 'BS13b'),
    ('BS_2013_2013-09-30_10-20-01_12Hz', 'BS13c'),
]
# ── 결과 전 고정 문턱 (PREREG §1·§2) ──────────────────────────────
REPROJ_MED_MAX = 0.5        # px
ROT_SPREAD_MAX = 0.02       # 회전 특이값 퍼짐
SENSOR_H_DECLARED = 12.0    # gates_b17.py:158 의 **선언값**
SENSOR_H_TOL = 0.20         # ±20%
IRJR_CHANGE_MIN = 0.5       # 프레임 간 상이 비율
RADIAL_FLAT_MAX = 0.05      # px, 반경 6분위 잔차 표준편차


def label(rows):
    """⛔결과 전 동결 판정표(PREREG §1)."""
    ok = [r for r in rows if r.get('pose_ok')]
    if not rows:
        return ('GEOMETRY-NOT-RECOVERABLE', '행 0개')
    if len(ok) == len(rows):
        return ('GEOMETRY-IS-MEASURABLE',
                f'{len(ok)}/{len(rows)} — B-17 의 선언 상수를 측정값으로 교체할 수 있다')
    if ok:
        return ('GEOMETRY-PARTIAL',
                f"{len(ok)}/{len(rows)} — 성공: {','.join(r['record'] for r in ok)}")
    return ('GEOMETRY-NOT-RECOVERABLE',
            f'0/{len(rows)} — 분해가 틀렸거나 채널이 그 뜻이 아니다')


def gates(rows):
    ok = [r for r in rows if 'error' not in r]
    A = []
    A.append(('G-0 행 완비', len(ok) == len(PAIRS), f'{len(ok)}/{len(PAIRS)}'))
    bad = [r['record'] for r in ok if not r.get('z_identical')]
    A.append(('★G-1 full Z 가 _short 와 바이트 동일', not bad, ';'.join(bad[:3])))
    bad = [r['record'] for r in ok if r.get('irjr_change', 0) <= IRJR_CHANGE_MIN]
    A.append(('G-2 iR/jR 이 프레임마다 변한다', not bad,
              f"min={min((r.get('irjr_change', 0) for r in ok), default=0):.3f}"))
    bad = [r['record'] for r in ok if r.get('irjr_fill', 1) > 0]
    A.append(('G-3 채움값 0개', not bad, ';'.join(bad[:3])))
    bad = [r['record'] for r in ok if r.get('radial_flat', 9) > RADIAL_FLAT_MAX]
    A.append(('G-4 방사 잔차 평탄 (왜곡 이미 보정됨)', not bad,
              f"max={max((r.get('radial_flat', 9) for r in ok), default=9):.4f} px"))
    bad = [r['record'] for r in ok if not r.get('K_ok')]
    A.append(('G-5 intrinsics 파싱', not bad, ';'.join(bad[:3])))
    A.append(('G-COV 모든 게이트가 자료를 봤다', len(ok) > 0, f'rows={len(ok)}'))
    return A


def blocked_by(gate_rows):
    return [g[0] for g in gate_rows if not g[1]]
