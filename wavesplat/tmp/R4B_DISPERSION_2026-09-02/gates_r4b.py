"""`R4b-DISPERSION` 규칙 단일출처. ⛔`PREREG.md` 가 정본이고 이 파일이 코드로 옮긴다."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
if os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23') not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23'))
import scene_station as SS                                          # noqa: E402

CONTROL = list(SS.RECORDS_12.keys())
AGULHAS = ['0805', '0845', '0925', '1000', '1105', '1150']
N_CTRL, N_TREAT = len(CONTROL), len(AGULHAS)

NT = 1024
MIN_VALID_FRAC = 0.15          # G-COV — 결과 전 규칙
N_NULL = 256                   # G-NULLB
NULL_SEED = 20260902           # ⛔결과 전 고정
NULL_MIN_SHIFT = 8             # 참 밴드 ±3 과 겹치지 않게
PCTL = 95.0                    # "자를 통과한다" 의 정의
CTRL_PASS_MIN = 10             # ★G-CTRL2 — PREREG §3 이 사유를 적었다(⛔12 아님)
NULL_SANE_MULT = 3.0           # G-NULLSANE — 귀무 중앙 ≤ 3 × (14/nf)
MIN_RINGS = 3


def label(ctrl_pass, treat_pass):
    """⛔결과 전 동결 판정표(PREREG §3)."""
    if ctrl_pass < CTRL_PASS_MIN:
        return (None, f'대조 {ctrl_pass}/{N_CTRL} < {CTRL_PASS_MIN} — 자가 고장. 판정 없음')
    if treat_pass == 0:
        return ('DISPERSION-REGIME-DIFFERS',
                f'남극 0/{N_TREAT} 통과 — 선형 분산 구조가 **우연 이상으로 없다**')
    return ('DISPERSION-PARTIAL',
            f'남극 {treat_pass}/{N_TREAT} 통과 — ⛔통과 레코드를 명시할 것')


def gates(rows):
    ok = [r for r in rows if 'error' not in r and not r.get('void')]
    ctl = [r for r in ok if r['group'] == 'control']
    trt = [r for r in ok if r['group'] == 'agulhas']
    A = []
    A.append(('G-0 행 완비', len(ok) == N_CTRL + N_TREAT, f'{len(ok)}/{N_CTRL + N_TREAT}'))
    A.append(('G-0b 에러 0', not [r for r in rows if 'error' in r],
              str(len([r for r in rows if 'error' in r]))))

    cp = sum(1 for r in ctl if r['passes'])
    A.append((f'★G-CTRL2 대조 ≥{CTRL_PASS_MIN} 이 자기 귀무 초과', cp >= CTRL_PASS_MIN,
              f'{cp}/{len(ctl)}'))

    md5s = {r['sheet_md5'] for r in ok}
    A.append(('G-SAME 두 군에 같은 함수 소스', len(md5s) == 1,
              (md5s.pop() if len(md5s) == 1 else f'{len(md5s)}종')))

    bad = [r['record'] for r in ok if r['null_median'] > NULL_SANE_MULT * r['chance']]
    A.append((f'★G-NULLSANE 귀무 중앙 ≤ {NULL_SANE_MULT}× 우연수준', not bad,
              (f"최대비 {max(r['null_median'] / r['chance'] for r in ok):.2f}" if ok else '--')))

    A.append((f'G-NULLB B={N_NULL} · 시드 {NULL_SEED}',
              all(r['n_null'] == N_NULL and r['null_seed'] == NULL_SEED for r in ok),
              f'{N_NULL}/{NULL_SEED}'))

    bad = [r['record'] for r in ok if r['n_rings'] < MIN_RINGS]
    A.append((f'G-RING 유효 링 ≥{MIN_RINGS}', not bad, ';'.join(bad[:3])))

    A.append(('G-COV 유효율 기록 · <0.15 VOID',
              all('valid_frac' in r for r in ok)
              and not [r for r in ok if r['valid_frac'] < MIN_VALID_FRAC],
              f"min={min((r['valid_frac'] for r in ok), default=float('nan')):.2f}"))

    A.append((f'G-NT 두 군 모두 NT={NT}', all(r['nt'] == NT for r in ok), str(NT)))

    # ★G-ANCHOR-R4 — 리팩터가 관측값을 **한 비트도** 바꾸지 않았음을 자료로 증명한다.
    #   ⛔손으로 옮긴 수가 아니라 선행 배치의 **동결 행**과 대조한다.
    anc = [r for r in ok if r.get('anchor_r4') is not None]
    A.append(('★G-ANCHOR-R4 관측 sheetfrac 이 R4 동결 행과 비트 동일',
              len(anc) == len(ok) and all(r['anchor_r4'] for r in ok),
              f'{sum(1 for r in ok if r.get("anchor_r4"))}/{len(ok)}'))
    A.append(('G-BLOCK 자가진단', len(ok) > 0, f'rows={len(ok)}'))
    return A


def blocked_by(gate_rows):
    """★자가진단 대상 — 실패한 게이트 이름을 **비어있지 않게** 돌려줘야 한다."""
    return [g[0] for g in gate_rows if not g[1]]
