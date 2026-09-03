"""`C2-STOKES` 규칙 단일출처. ⛔`PREREG.md` 가 정본이고 이 파일이 코드로 옮긴다."""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
C1 = os.path.join(ROOT, 'tmp', 'C1_MODELGAP_2026-09-03')
if C1 not in sys.path:
    sys.path.insert(0, C1)
import gates_c1 as C                                                 # noqa: E402

HOS, LIN, STRIDE = C.HOS, C.LIN, C.STRIDE
is_crowded = C.is_crowded
N_CROWDED, N_SPARSE, N_CTRL_MIN = 48, 16, 8
THRESH_YES, THRESH_PARTIAL = 44, 25


def label(w):
    """⛔결과 전 동결 판정표(PREREG §2)."""
    if w >= THRESH_YES:
        return ('GAP-IS-SECOND-ORDER',
                f'{w}/{N_CROWDED} — ★이탈은 **모수 0의 이론**으로 설명된다 ⟹ 고리가 '
                '열리는 곳은 표현이 아니라 **분산관계 제약**이다')
    if w >= THRESH_PARTIAL:
        return ('SECOND-ORDER-PARTIAL', f'{w}/{N_CROWDED}')
    return ('GAP-NOT-SECOND-ORDER',
            f'{w}/{N_CROWDED} — ⛔감사의 자유적합 소견이 이론으로는 재현되지 않는다')


def _src(fn):
    return open(os.path.join(HERE, fn), encoding='utf-8').read()


def gates(rows, meta):
    ok = [r for r in rows if 'error' not in r]
    crow = [r for r in ok if r['group'] == 'hos' and r['crowded']]
    spar = [r for r in ok if r['group'] == 'hos' and not r['crowded']]
    ctrl = [r for r in ok if r['group'] == 'linear']
    A = []
    # ★G-CTRL — 선형에서는 **반드시 나빠져야** 한다(부호까지 지정된 대조)
    bad = [r['episode'] for r in ctrl if not (r['rms_st2'] > r['rms_lin'])]
    A.append(('★G-CTRL 선형 전건에서 ST2 가 더 나쁘다', not bad and len(ctrl) > 0,
              ';'.join(sorted(set(bad))[:3]) or f'{len(ctrl)}개 전건'))
    src = _src('run_c2.py')
    imp = bool(re.search(r'from\s+wavesplat\.model\.stokes2\s+import\s+Stokes2Transfer', src))
    A.append(('★G-IMPORT Stokes2Transfer 를 임포트(재구현 0)', imp,
              'import 확인' if imp else '⛔임포트 없음'))
    nofit = not re.search(r'lstsq|\.fit\(|torch\.optim|minimize\(', src)
    A.append(('★G-NOFIT ST2 에 적합된 모수 0개', bool(nofit),
              '적합 호출 0' if nofit else '⛔적합 호출 발견'))
    norun = not re.search(r'make_updater|run_estimation', src)
    A.append(('★G-NORUN 추정기 런 0회', bool(norun),
              'make_updater/run_estimation 0' if norun else '⛔추정기 호출 발견'))
    A.append((f'G-N 혼잡 {N_CROWDED} · 희소 {N_SPARSE} · 대조 ≥ {N_CTRL_MIN}',
              len(crow) == N_CROWDED and len(spar) == N_SPARSE and len(ctrl) >= N_CTRL_MIN,
              f'{len(crow)}/{len(spar)}/{len(ctrl)}'))
    bad = [r['episode'] for r in ok if r.get('depth') is None]
    A.append(('G-DEPTH 수심을 행에 기록', not bad,
              (f"{sorted(set(r['depth'] for r in ok))}" if not bad else ';'.join(bad[:3]))))
    A.append(('G-0b 에러 0', len(rows) == len(ok), str(len(rows) - len(ok))))
    A.append(('G-COV 게이트가 자료를 봤다', len(ok) > 0, f'rows={len(ok)}'))
    probe = (blocked_by([('G-CTRL 합성', False, ''), ('★G-NOFIT 합성', True, '')]) == ['G-CTRL'])
    A.append(('G-BLOCK blocked_by 자가진단', bool(probe),
              'synthetic-fail → [G-CTRL]' if probe else '⛔자가진단 실패'))
    return A


def blocked_by(gate_rows):
    block = ('G-CTRL', 'G-IMPORT', 'G-NOFIT', 'G-NORUN', 'G-N', 'G-DEPTH', 'G-0b', 'G-COV')
    out = []
    for name, ok, _ in gate_rows:
        head = name.replace('★', '').split()[0]
        if head in block and not ok:
            out.append(head)
    return out
