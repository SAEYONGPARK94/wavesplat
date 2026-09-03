"""`R4d-DISPERSION` 규칙 단일출처. ⛔`PREREG.md` 가 정본.

⛔상수는 전부 `gates_r4c` 에서 임포트한다 — 이 배치가 바꾸는 것은 **분산면 한 줄**이고,
하나라도 재선언하면 *"바꾼 것은 하나뿐"* 이 거짓이 된다.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
R4C = os.path.join(ROOT, 'tmp', 'R4C_DISPERSION_2026-09-03')
for p in (R4C, os.path.join(ROOT, 'tmp', 'R4B_DISPERSION_2026-09-02'),
          ROOT, os.path.join(ROOT, 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)
import gates_r4c as C                                               # noqa: E402

CONTROL, AGULHAS = list(C.CONTROL), list(C.AGULHAS)
N_CTRL, N_TREAT = C.N_CTRL, C.N_TREAT
NT, MIN_VALID_FRAC, MIN_RINGS = C.NT, C.MIN_VALID_FRAC, C.MIN_RINGS
N_NULL, NULL_SEED, NULL_MIN_SHIFT = C.N_NULL, C.NULL_SEED, C.NULL_MIN_SHIFT
PCTL, CTRL_PASS_MIN, BAND, OVERLAP_TOL = C.PCTL, C.CTRL_PASS_MIN, C.BAND, C.OVERLAP_TOL
R4C_ROWS = os.path.join(R4C, 'results_r4c.jsonl')
AGULHAS_DEPTH = 1000.0        # 남극은 심해 --- `tanh → 1`. 행에 기록해 확인한다.


def label(ctrl_pass, treat_pass):
    """⛔결과 전 동결 판정표(PREREG §2). ★`R4c` 와 **같은가**만 묻는다."""
    if ctrl_pass == N_CTRL and treat_pass == 0:
        return ('DEEPWATER-EQUIVALENT',
                f'대조 {ctrl_pass}/{N_CTRL} · 처치 {treat_pass}/{N_TREAT} — '
                f'자 확장이 기존 판정을 바꾸지 않는다')
    return ('TOOL-CHANGE-MOVES-VERDICT',
            f'대조 {ctrl_pass}/{N_CTRL} · 처치 {treat_pass}/{N_TREAT} — '
            f'⛔R4b·R4c 를 재검토해야 한다')


def gates(rows, r4c):
    ok = [r for r in rows if 'error' not in r and not r.get('void')]
    ctl = [r for r in ok if r['group'] == 'control']
    A = []
    A.append(('G-0 행 완비', len(ok) == N_CTRL + N_TREAT, f'{len(ok)}/{N_CTRL + N_TREAT}'))
    A.append(('G-0b 에러 0', not [r for r in rows if 'error' in r],
              str(len([r for r in rows if 'error' in r]))))

    A.append(('★G-TANH 처치가 실제로 tanh 를 쓴다',
              all(r.get('uses_tanh') and r.get('tanh_med') is not None for r in ok),
              f"tanh 중앙 {min(r['tanh_med'] for r in ok):.6f}–"
              f"{max(r['tanh_med'] for r in ok):.6f}" if ok else '--'))

    bad = [r['record'] for r in ctl if not r.get('biteq_r4c')]
    A.append((f'★G-BITEQ 대조 {N_CTRL} 의 관측이 R4c 와 비트 동일', not bad,
              f"{sum(1 for r in ctl if r.get('biteq_r4c'))}/{len(ctl)}"))

    A.append(('★G-DEPTH 수심이 scene_station 에서 왔고 행에 기록',
              all(isinstance(r.get('depth'), float) for r in ok)
              and all(r.get('depth_src') == 'scene_station.RECORDS_12'
                      for r in ctl), '기록됨'))

    cp = sum(1 for r in ctl if r['passes'])
    A.append((f'G-CTRL2 대조 ≥{CTRL_PASS_MIN} 초과 (gates_r4c 임포트)',
              cp >= CTRL_PASS_MIN, f'{cp}/{len(ctl)}'))
    A.append((f'G-DRAWS 귀무 {N_NULL} draw 전부 저장',
              all(len(r.get('null_draws', [])) == N_NULL for r in ok), str(N_NULL)))
    A.append((f'G-SEED 시드 {NULL_SEED} · B={N_NULL} · pctl {PCTL}',
              all(r.get('null_seed') == NULL_SEED and r.get('n_null') == N_NULL
                  and r.get('pctl') == PCTL for r in ok), f'{NULL_SEED}/{N_NULL}/{PCTL}'))
    bad = [r['record'] for r in ok if r['n_rings'] < MIN_RINGS]
    A.append((f'G-RING 유효 링 ≥{MIN_RINGS}', not bad, ';'.join(bad[:3])))
    probe = blocked_by([('X-거짓', False, ''), ('Y-참', True, '')])
    A.append(('★G-BLOCK blocked_by 자가진단', probe == ['X-거짓'], repr(probe)))
    A.append(('G-COV 모든 게이트가 자료를 봤다', len(ok) > 0 and len(r4c) > 0,
              f'rows={len(ok)} r4c={len(r4c)}'))
    return A


def blocked_by(gate_rows):
    """★자가진단 대상 — 실패한 게이트 이름을 **비어있지 않게** 돌려줘야 한다."""
    return [g[0] for g in gate_rows if not g[1]]
