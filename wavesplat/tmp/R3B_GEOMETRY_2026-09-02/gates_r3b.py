"""`R3b-GEOMETRY` 규칙 단일출처. ⛔`PREREG.md` 가 정본.

★선행 `R3` 의 사양 오류를 고친다: `G-2` 의 문턱이 **다른 채널(`jR`)의 실측치**였다.
이제 두 채널을 **모두** 재고, 문턱은 **의미**에서 잡는다(*"정적 매핑이 아니다"* = $>0$).
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
FULL = os.path.join(ROOT, 'data', 'real_bergamasco_full')
BASE_URL = 'https://data-dataref.ifremer.fr/stereo'   # G-FILE 이 여기서 크기를 묻는다

# (full 파일 접두, `_short` 레코드 키, 원격 경로)
PAIRS = [
    ('BS_2011_2011-10-01_16-18-00_15Hz', 'BS11a', 'BS_2011/2011-10-01_16-18-00_15Hz'),
    ('BS_2011_2011-10-04_11-38-00_12Hz', 'BS11b', 'BS_2011/2011-10-04_11-38-00_12Hz'),
    ('BS_2011_2011-10-04_13-07-00_12Hz', 'BS11c', 'BS_2011/2011-10-04_13-07-00_12Hz'),
    ('BS_2011_2011-10-04_15-30-00_12Hz', 'BS11d', 'BS_2011/2011-10-04_15-30-00_12Hz'),
    ('BS_2013_2013-09-22_13-00-01_10Hz', 'BS13a', 'BS_2013/2013-09-22_13-00-01_10Hz'),
    ('BS_2013_2013-09-25_12-15-01_12Hz', 'BS13b', 'BS_2013/2013-09-25_12-15-01_12Hz'),
    ('BS_2013_2013-09-30_10-20-01_12Hz', 'BS13c', 'BS_2013/2013-09-30_10-20-01_12Hz'),
]
N = len(PAIRS)

# ── 결과 전 고정 문턱 (PREREG §2·§3) ──────────────────────────────
REPROJ_MED_MAX = 0.5        # px
ROT_SPREAD_MAX = 0.02       # 회전 특이값 퍼짐
SENSOR_H_DECLARED = 12.0    # gates_b17.py:158 의 **선언값**(⛔측정값 아님)
SENSOR_H_TOL = 0.20         # ±20%
RADIAL_FLAT_MAX = 0.05      # px
PIX_OFFSET = 1500.0         # ⭐`iR`/`jR` 저장 오프셋 — 없으면 DLT 가 깨진다
N_CORR = 4000               # DLT 대응점 수 (결과 전 고정)
N_ZCHECK = 12               # G-1 이 비교하는 표본 프레임 수
# ★G-2 — ⛔`0.5` 였던 것을 버린다. *"정적 매핑이 아니다"* 의 최소 조건은 이것뿐이다.
IRJR_CHANGE_MIN = 0.0


def label(rows):
    """⛔결과 전 동결 판정표(PREREG §2)."""
    ok = [r for r in rows if r.get('pose_ok')]
    if not rows:
        return ('GEOMETRY-NOT-RECOVERABLE', '행 0개')
    if len(ok) == len(rows) == N:
        return ('GEOMETRY-IS-MEASURABLE',
                f'{len(ok)}/{N} — B-17 의 선언 상수를 측정값으로 교체할 수 있다')
    if ok:
        return ('GEOMETRY-PARTIAL',
                f"{len(ok)}/{len(rows)} — 성공: {','.join(r['record'] for r in ok)}")
    return ('GEOMETRY-NOT-RECOVERABLE',
            f'0/{len(rows)} — 분해가 틀렸거나 채널이 그 뜻이 아니다')


def gates(rows):
    ok = [r for r in rows if 'error' not in r]
    A = []
    A.append(('G-0 행 완비', len(ok) == N, f'{len(ok)}/{N}'))
    A.append(('G-0b 에러 0', len(rows) == len(ok), str(len(rows) - len(ok))))

    # ★G-FILE — ⛔선행이 **다운로드 중 파일**에서 죽었다.
    bad = [r['record'] for r in ok
           if r.get('remote_bytes') is None or r['local_bytes'] != r['remote_bytes']]
    A.append(('★G-FILE 로컬 크기 == 원격 Content-Length', not bad, ';'.join(bad[:3])))

    bad = [r['record'] for r in ok if not r.get('z_identical')]
    A.append((f'★G-1 full Z 가 _short 와 바이트 동일 (표본 {N_ZCHECK}프레임)',
              not bad, ';'.join(bad[:3])))

    # ★G-2 — 두 채널 **모두**. 문턱은 의미에서(⛔선행의 0.5 는 다른 채널의 수였다).
    bad = [r['record'] for r in ok
           if r.get('iR_change', 0) <= IRJR_CHANGE_MIN or r.get('jR_change', 0) <= IRJR_CHANGE_MIN]
    A.append(('★G-2 iR·jR 둘 다 프레임 간 변화 > 0', not bad,
              (f"iR {min(r['iR_change'] for r in ok):.3f}–{max(r['iR_change'] for r in ok):.3f}"
               f" · jR {min(r['jR_change'] for r in ok):.3f}–"
               f"{max(r['jR_change'] for r in ok):.3f}") if ok else '--'))

    bad = [r['record'] for r in ok if r.get('irjr_fill_frac', 1.0) > 0.0]
    A.append(('G-3 채움값 0개 (유효 마스크 안)', not bad,
              f"max={max((r.get('irjr_fill_frac', 1.0) for r in ok), default=1.0):.4f}"))

    bad = [r['record'] for r in ok if r.get('radial_flat', 9) > RADIAL_FLAT_MAX]
    A.append(('G-4 방사 잔차 평탄 (왜곡 이미 보정됨)', not bad,
              f"max={max((r.get('radial_flat', 9) for r in ok), default=9):.4f} px"))

    bad = [r['record'] for r in ok if not r.get('K_ok')]
    A.append(('G-5 intrinsics 파싱', not bad, ';'.join(bad[:3])))

    A.append((f'G-OFF 화소 오프셋 {PIX_OFFSET:.0f} 을 행에 기록',
              all(r.get('pix_offset') == PIX_OFFSET for r in ok), f'{PIX_OFFSET:.0f}'))
    A.append(('G-COV 모든 게이트가 자료를 봤다', len(ok) > 0, f'rows={len(ok)}'))
    return A


def blocked_by(gate_rows):
    """★자가진단 대상 — 실패한 게이트 이름을 **비어있지 않게** 돌려줘야 한다."""
    return [g[0] for g in gate_rows if not g[1]]
