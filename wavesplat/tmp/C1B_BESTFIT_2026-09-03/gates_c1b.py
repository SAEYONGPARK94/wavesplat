"""`C1b-BESTFIT` 규칙 단일출처. ⛔`PREREG.md` 가 정본이고 이 파일이 코드로 옮긴다."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
C1 = os.path.join(ROOT, 'tmp', 'C1_MODELGAP_2026-09-03')
for _p in (C1,):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import gates_c1 as C                                                 # noqa: E402

HOS, LIN, STRIDE = C.HOS, C.LIN, C.STRIDE
GAP_MIN = C.GAP_MIN
N_CROWDED, N_SPARSE, N_CTRL_MIN = C.N_CROWDED, C.N_SPARSE, C.N_CTRL_MIN
is_crowded = C.is_crowded
CTRL_TOL = 1e-10            # ⛔결과 전 고정 (수치 LS 이므로 정확히 0 을 요구하지 않는다)
THRESH_ALL, THRESH_MOST = 48, 44


def label(w):
    """⛔결과 전 동결 판정표(PREREG §3)."""
    if w >= THRESH_ALL:
        return ('GAP-SURVIVES-REFIT',
                f'{w}/{N_CROWDED} — ★`C1` 의 거리는 **진폭·위상 탓이 아니다**')
    if w >= THRESH_MOST:
        return ('GAP-MOSTLY-SURVIVES', f'{w}/{N_CROWDED} — 예외를 명시한다')
    return ('GAP-IS-COEFFICIENT-DRIFT',
            f'{w}/{N_CROWDED} — ★`C1` 의 읽기를 좁혀야 한다')


def gates(rows, meta):
    ok = [r for r in rows if 'error' not in r]
    ctrl = [r for r in ok if r['group'] == 'linear']
    crow = [r for r in ok if r['group'] == 'hos' and r['crowded']]
    spar = [r for r in ok if r['group'] == 'hos' and not r['crowded']]
    A = []
    bad = [r['episode'] for r in ctrl if not (r['rel_ls'] <= CTRL_TOL)]
    A.append((f'★G-CTRL 선형 rel_ls ≤ {CTRL_TOL:g} 전건', not bad and len(ctrl) > 0,
              (f"{len(ctrl)}개, max {max((r['rel_ls'] for r in ctrl), default=float('nan')):.3e}"
               if not bad else ';'.join(bad[:3]))))
    A.append(('★G-PAIR C1 과 에피소드 전건 맺힘', bool(meta.get('paired')),
              str(meta.get('pair_msg'))))
    bad = [r['episode'] for r in ok if not (r['rel_ls'] <= r['rel_gap_c1'] * (1 + 1e-9))]
    A.append(('★G-LE rel_ls ≤ rel_gap (재적합이 나빠질 수 없다)', not bad,
              ';'.join(sorted(set(bad))[:3]) or '전건'))
    A.append((f'G-N 혼잡 {N_CROWDED} · 희소 {N_SPARSE} · 대조 ≥ {N_CTRL_MIN}',
              len(crow) == N_CROWDED and len(spar) == N_SPARSE and len(ctrl) >= N_CTRL_MIN,
              f'{len(crow)}/{len(spar)}/{len(ctrl)}'))
    A.append(('★G-NORUN 추정기 임포트 0', bool(meta.get('norun')), str(meta.get('norun_msg'))))
    A.append(('G-0b 에러 0', len(rows) == len(ok), str(len(rows) - len(ok))))
    A.append(('G-COV 게이트가 자료를 봤다', len(ok) > 0, f'rows={len(ok)}'))
    probe = (blocked_by([('G-CTRL 합성', False, ''), ('★G-LE 합성', True, '')]) == ['G-CTRL'])
    A.append(('G-BLOCK blocked_by 자가진단', bool(probe),
              'synthetic-fail → [G-CTRL]' if probe else '⛔자가진단 실패'))
    return A


def blocked_by(gate_rows):
    block = ('G-CTRL', 'G-PAIR', 'G-LE', 'G-N', 'G-NORUN', 'G-0b', 'G-COV')
    out = []
    for name, ok, _ in gate_rows:
        head = name.replace('★', '').split()[0]
        if head in block and not ok:
            out.append(head)
    return out
