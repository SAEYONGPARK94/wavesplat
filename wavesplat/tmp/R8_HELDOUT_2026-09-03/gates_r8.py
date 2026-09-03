"""`R8-HELDOUT` 규칙 단일출처. ⛔`PREREG.md` 가 정본이고 이 파일이 코드로 옮긴다."""
import os
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
B2DIR = os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23')
for _p in (B2DIR, ROOT, os.path.join(ROOT, 'scripts')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import gates_b2 as B2                                                # noqa: E402

ARMS = dict(B2.ARMS)             # ⛔재선언하지 않는다 (SHIP/PROP) — G-ARM 이 되읽는다
ARM_ORDER = list(B2.ARM_ORDER)
RECORDS = list(B2.RECORDS)       # `LJ18b` 결과 전 배제를 상속
WIN = B2.WIN                     # 3
SEED = B2.SEED                   # 42
N = len(RECORDS)

THRESH_CONFIRM = 11              # p = 0.0032
THRESH_NOMINAL = 10              # p = 0.0193
THRESH_REVERSE = 7               # 이하이면 예측 반증
CIN_MIN = 10                     # ★양성 대조: 지지집합 안에서 기술
MIN_QPTS = 5                     # G-SPLIT — ⚠아래 §스모크 공시 참조
# ★결과 전 확인(추정기 0회, 장면만): npx 최소 8(BS11a/b/c) ⟹ 이 게이트는 현 코퍼스에서
#   **발화하지 않는다**. 기능은 「퇴화 분할 탐지」이지 「검정력 보증」이 아니다.
#   검정력은 `SEC_LARGE` 가 진다.
SEC_LARGE = ('AA14', 'AA15a', 'AA15b', 'LJ18a', 'YS17')   # 박스 ≥ 40 m, npx ≥ 55
SEC_LARGE_THRESH = 5             # n=5 는 전승만 유의 (p = 0.03125)
ALPHA = 0.05

SHIPPING_KEYS = ('varpro_steps', 'learned_init_iters', 'auto_switch_thresh')


def _shipping_expect():
    from wavesplat.run import DEFAULT_CONFIG
    return {k: DEFAULT_CONFIG[k] for k in SHIPPING_KEYS}


def sign_test_one_sided(w, n):
    return sum(comb(n, k) for k in range(w, n + 1)) / 2.0 ** n


def label(w):
    """⛔결과 전 동결 판정표(PREREG §2)."""
    p = sign_test_one_sided(w, N)
    if w >= THRESH_CONFIRM:
        return ('SUPPORT-BOUND-CONFIRMED',
                f'{w}/{N} p={p:.4f} — 실측 기술은 **관측 지지집합 안에 갇혀 있다**')
    if w >= THRESH_NOMINAL:
        return ('SUPPORT-BOUND-NOMINAL',
                f'{w}/{N} p={p:.4f} — ⛔명목만, 확증 아님')
    if w <= THRESH_REVERSE:
        return ('SUPPORT-GENERALISES',
                f'{w}/{N} p={p:.4f} — ★예측 반증: 지지집합 밖에서도 기술이 유지된다')
    return ('SUPPORT-UNDECIDED', f'{w}/{N} p={p:.4f} — ⛔어느 방향도 주장 금지')


def gates(rows, meta):
    ok = [r for r in rows if 'error' not in r]
    ship = [r for r in ok if r['cell'] == 'SHIP']
    A = []
    exp = len(ARMS) * N
    A.append(('G-0 행 수', len(ok) == exp, f'{len(ok)}/{exp}'))
    A.append(('G-0b 에러 0', len(rows) == len(ok), str(len(rows) - len(ok))))

    bad = [r['record'] for r in ok if not r.get('bit_identical_all')]
    A.append(('★G-BIT region=ALL 이 scene_station 과 비트 동일', not bad,
              ';'.join(sorted(set(bad))[:3]) or f'{len(ok)}행 전건'))

    bad = [r['record'] for r in ok
           if not (r['n_obs'] == r['n_obs_all'] == meta.get('m_cap', 600))]
    A.append(('★G-NOBS 관측수 = m_cap, 팔 간 동일', not bad,
              ';'.join(sorted(set(bad))[:3]) or f"n_obs={ok[0]['n_obs'] if ok else '--'}"))

    bad = [r['record'] for r in ok if not (r['gap_out_lam'] > r['gap_in_lam'])]
    A.append(('★G-GAP 개입 원자 확인 (gap_OUT > gap_IN)', not bad,
              (f"in {min(r['gap_in_lam'] for r in ok):.4f}–{max(r['gap_in_lam'] for r in ok):.4f} λ"
               f" / out {min(r['gap_out_lam'] for r in ok):.4f}–"
               f"{max(r['gap_out_lam'] for r in ok):.4f} λ") if ok else '--'))

    bad = [r['record'] for r in ok
           if min(r['npx_in'], r['npx_out']) < MIN_QPTS]
    A.append((f'G-SPLIT 두 영역 유효 질의점 ≥ {MIN_QPTS}', not bad,
              (f"min={min(min(r['npx_in'], r['npx_out']) for r in ok)}") if ok else '--'))

    exp_ship = meta.get('shipping_expect')
    bad = [r['record'] for r in ok if not r.get('arm_ok')]
    A.append((f'★G-ARM 팔 구성 객체 되읽기 {exp_ship}', not bad,
              ';'.join(sorted(set(bad))[:3]) or '전건'))

    bad = [r['record'] for r in ok if r.get('seed') != SEED or r.get('scene_seed') != 700 + WIN]
    A.append((f'G-SEED 시드 고정 (est={SEED}, scene={700 + WIN})', not bad,
              ';'.join(sorted(set(bad))[:3]) or '전건'))

    bad = [r['record'] for r in ok if r.get('convention') != 'R1b_mu_mem']
    A.append(('G-CONV 채점 규약 = R1b 수리판', not bad, ';'.join(sorted(set(bad))[:3]) or '전건'))

    # ★C-IN 양성 대조 — 차단이다(PREREG §3)
    cin = sum(1 for r in ship if r['ratio_in'] < 1.0)
    A.append((f'★C-IN 양성 대조: 지지집합 안 기술 ≥ {CIN_MIN}/{N}', cin >= CIN_MIN,
              f'{cin}/{len(ship)}'))

    A.append(('G-COV 게이트가 자료를 봤다', len(ok) > 0, f'rows={len(ok)}'))
    probe_ok = (blocked_by([('G-0 합성', False, ''), ('★G-BIT 합성', True, '')]) == ['G-0'])
    A.append(('G-BLOCK blocked_by 자가진단', bool(probe_ok),
              'synthetic-fail → [G-0]' if probe_ok else '⛔자가진단 실패'))
    return A


def blocked_by(gate_rows):
    """★자가진단 — 차단 게이트 중 실패한 것의 **이름**을 돌려준다."""
    block = ('G-0', 'G-0b', 'G-BIT', 'G-NOBS', 'G-GAP', 'G-SPLIT', 'G-ARM',
             'G-SEED', 'G-CONV', 'C-IN', 'G-COV')
    out = []
    for name, ok, _ in gate_rows:
        head = name.replace('★', '').split()[0]
        if head in block and not ok:
            out.append(head)
    return out
