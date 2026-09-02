"""`R4c-DISPERSION` 규칙 단일출처. ⛔`PREREG.md` 가 정본.

⛔상수는 선행 `gates_r4b` 를 **임포트**한다 — 이 배치가 바꾸는 것은 **중복 계수 하나뿐**이고,
문턱을 하나라도 재선언하면 그 주장이 거짓이 된다.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
if os.path.join(ROOT, 'tmp', 'R4B_DISPERSION_2026-09-02') not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, 'tmp', 'R4B_DISPERSION_2026-09-02'))
import gates_r4b as B                                               # noqa: E402

CONTROL, AGULHAS = list(B.CONTROL), list(B.AGULHAS)
N_CTRL, N_TREAT = B.N_CTRL, B.N_TREAT
NT, MIN_VALID_FRAC = B.NT, B.MIN_VALID_FRAC
N_NULL, NULL_SEED, NULL_MIN_SHIFT = B.N_NULL, B.NULL_SEED, B.NULL_MIN_SHIFT
PCTL, CTRL_PASS_MIN, MIN_RINGS = B.PCTL, B.CTRL_PASS_MIN, B.MIN_RINGS
BAND = 3                        # ±3 빈 — `run_r4b.py` 의 `range(-3, 4)` 와 같다
OVERLAP_TOL = 2 * BAND          # 두 밴드 공유 조건 |2(jl+s)| ≤ 6 (mod nf)
R4B_ROWS = os.path.join(ROOT, 'tmp', 'R4B_DISPERSION_2026-09-02', 'results_r4b.jsonl')


def label(ctrl_pass, treat_pass):
    """⛔결과 전 동결 판정표(PREREG §2)."""
    if ctrl_pass < CTRL_PASS_MIN:
        return (None, f'대조 {ctrl_pass}/{N_CTRL} < {CTRL_PASS_MIN} — 자가 고장. 판정 없음')
    if treat_pass == 0:
        return ('DISPERSION-REGIME-HOLDS',
                f'처치 0/{N_TREAT} — R4b 판정은 계수 인공물의 산물이 아니다')
    return ('DISPERSION-ARTEFACT-DEPENDENT',
            f'처치 {treat_pass}/{N_TREAT} — ⛔R4b 의 서술을 원고에서 철회해야 한다')


def _med(v):
    v = sorted(v)
    if not v:
        return float('nan')
    return v[len(v) // 2] if len(v) % 2 else (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2


def gates(rows):
    ok = [r for r in rows if 'error' not in r and not r.get('void')]
    ctl = [r for r in ok if r['group'] == 'control']
    trt = [r for r in ok if r['group'] == 'agulhas']
    A = []
    A.append(('G-0 행 완비', len(ok) == N_CTRL + N_TREAT, f'{len(ok)}/{N_CTRL + N_TREAT}'))
    A.append(('G-0b 에러 0', not [r for r in rows if 'error' in r],
              str(len([r for r in rows if 'error' in r]))))

    # ★G-OBS — 중복 제거가 관측을 건드리지 않았음을 **자료로** 증명
    A.append(('★G-OBS 관측이 R4b 와 비트 동일',
              all(r.get('obs_identical') for r in ok) and len(ok) == N_CTRL + N_TREAT,
              f"{sum(1 for r in ok if r.get('obs_identical'))}/{len(ok)}"))

    cp = sum(1 for r in ctl if r['passes'])
    A.append((f'★G-CTRL2 대조 ≥{CTRL_PASS_MIN} 이 자기 귀무 초과 (R4b 와 같은 상수)',
              cp >= CTRL_PASS_MIN, f'{cp}/{len(ctl)}'))

    # ★G-LOWER — 중복 제거는 귀무를 낮추거나 같게 해야 한다. 올라가면 내 기제 서술이 틀렸다.
    bad = [r['record'] for r in ok if r['null_p95'] > r['null_p95_r4b'] + 1e-15]
    A.append(('★G-LOWER 중복 제거가 귀무를 낮추거나 같게 한다', not bad,
              (f"최대 상승 {max(r['null_p95'] - r['null_p95_r4b'] for r in ok):.3e}"
               if ok else '--')))

    # ★G-OVL — 자기중첩률의 **방향**만 재현한다 (⛔크기 문턱은 걸지 않는다)
    mc, mt = _med([r['overlap_frac'] for r in ctl]), _med([r['overlap_frac'] for r in trt])
    A.append(('★G-OVL 자기중첩률 대조중앙 < 처치중앙', bool(mc < mt),
              f'대조 {mc:.3f} vs 처치 {mt:.3f}'))

    A.append((f'G-DRAWS 귀무 {N_NULL} draw 전부 저장',
              all(len(r.get('null_draws', [])) == N_NULL for r in ok), str(N_NULL)))
    A.append((f'G-SEED 시드 {NULL_SEED} · B={N_NULL} · pctl {PCTL} 가 R4b 와 동일',
              all(r.get('null_seed') == NULL_SEED and r.get('n_null') == N_NULL
                  and r.get('pctl') == PCTL for r in ok), f'{NULL_SEED}/{N_NULL}/{PCTL}'))
    bad = [r['record'] for r in ok if r['n_rings'] < MIN_RINGS]
    A.append((f'G-RING 유효 링 ≥{MIN_RINGS}', not bad, ';'.join(bad[:3])))
    A.append((f'G-NT 두 군 모두 NT={NT}', all(r['nt'] == NT for r in ok), str(NT)))

    # ★G-BLOCK — 선행 `R4b` 가 **하지 않은** 자가진단을 실제로 한다.
    probe = blocked_by([('X-거짓', False, ''), ('Y-참', True, '')])
    A.append(('★G-BLOCK blocked_by 자가진단 (⛔선행 R4b 는 이것을 안 했다)',
              probe == ['X-거짓'], repr(probe)))
    A.append(('G-COV 모든 게이트가 자료를 봤다', len(ok) > 0, f'rows={len(ok)}'))
    return A


def blocked_by(gate_rows):
    """★자가진단 대상 — 실패한 게이트 이름을 **비어있지 않게** 돌려줘야 한다."""
    return [g[0] for g in gate_rows if not g[1]]
