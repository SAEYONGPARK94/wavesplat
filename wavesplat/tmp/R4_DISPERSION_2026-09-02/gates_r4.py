"""`R4-DISPERSION` 규칙 단일출처. ⛔`PREREG.md` 가 정본."""
import os, sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23'))
import gates_b2 as B2                                              # noqa: E402

CONTROL = list(B2.RECORDS)                     # 12, 양성 대조
AGULHAS = ['0805', '0845', '0925', '1000', '1105', '1150']
NT = 1024
N_TREAT = len(AGULHAS)
THRESH = N_TREAT                               # ★전승만 유의 (p=0.0156)
CTRL_MEDIAN_MIN = 0.40                         # ★G-CTRL — 자가 고장 검사
MIN_VALID_FRAC = 0.15                          # G-COV, 결과 전 규칙
MIN_RINGS = 3


def sign_test_one_sided(w, n):
    return sum(comb(n, k) for k in range(w, n + 1)) / 2.0 ** n


def label(w):
    if w >= THRESH:
        return ('DISPERSION-REGIME-DIFFERS',
                f'{w}/{N_TREAT} — 두 코퍼스는 **같은 종류의 자료가 아니다**')
    return ('DISPERSION-UNDECIDED',
            f'{w}/{N_TREAT} — ⛔"같다" 금지 · n={N_TREAT} 은 전승만 유의(5/6 은 p=0.109)')


def gates(rows):
    ok = [r for r in rows if 'error' not in r and not r.get('void')]
    ctl = [r for r in ok if r['group'] == 'control']
    trt = [r for r in ok if r['group'] == 'agulhas']
    A = []
    A.append(('G-0 행 완비', len(ctl) + len(trt) == len(rows),
              f'ctl={len(ctl)} trt={len(trt)}'))
    med = (sorted(r['sheetfrac_mean'] for r in ctl)[len(ctl) // 2] if ctl else 0.0)
    A.append(('★G-CTRL 양성 대조 통과 (중앙 ≥ 0.40)', med >= CTRL_MEDIAN_MIN,
              f'중앙={med:.3f}'))
    A.append(('G-SAME 같은 함수·NT·창 규약',
              all(r.get('nt') == NT for r in ok) and len({r.get('method') for r in ok}) == 1,
              str({r.get('method') for r in ok})))
    bad = [r['record'] for r in ok if r.get('n_rings', 0) < MIN_RINGS]
    A.append(('G-RING 유효 링 ≥ 3', not bad, ';'.join(bad[:3])))
    A.append(('G-COV 유효율 기록 · 0.15 미만은 VOID',
              all('valid_frac' in r for r in ok),
              f"VOID={sum(1 for r in rows if r.get('void'))}"))
    A.append(('G-NT 두 군 모두 NT=1024', all(r.get('nt') == NT for r in ok), str(NT)))
    A.append(('G-COVG 게이트가 자료를 봤다', len(ok) > 0, f'rows={len(ok)}'))
    return A


def blocked_by(gate_rows):
    return [g[0] for g in gate_rows if not g[1]]
