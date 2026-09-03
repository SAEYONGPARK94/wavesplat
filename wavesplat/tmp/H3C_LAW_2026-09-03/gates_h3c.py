"""`H3c-LAW` 규칙 단일출처. ⛔`PREREG.md` 가 정본이고 이 파일이 코드로 옮긴다."""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
H3 = os.path.join(ROOT, 'tmp', 'H3_FRF_2026-09-03')

MRE_THRESH = 0.20                # ⛔`H3` 에서 상속 (thesis 제5장 자신의 수)
N_PERM = 200                     # `H3` 와 같은 반복수
PERM_SEED = 20260903             # ⛔결과 전 고정
N_CELLS = 236423
D_REF = 125.0
M_LOOKS = 1                      # ⛔판정에 쓰는 값. M 스윕은 보고 전용.
REPRO_TOL = 1e-9
FIT_TOL = 1e-6
EPS_MODULE = 1e-9        # `raydrop_threshold.received_power` 의 정칙화 항 (코드에서 읽는다)


def norm_ws(s):
    """★공백 정규화 — 2026-09-01 `check_paper_disclosures.py` 의 처방 이식.

    검사기가 **줄바꿈을 못 넘어 「있는 것을 없다」고** 판정한 결함이 `H3b` 에서 재발했다.
    """
    import re
    return re.sub(r'\s+', ' ', s or '').strip()

# ⛔Model B 배제 사유 — `wavesim/raydrop_threshold.py` 독스트링 축자
EXCL_B = ("raydrop_slope's specular gate never fires at this geometry "
          "(required facet slope r/h = 3..40 vs wave-slope sd ~0.08) "
          "and falls back to constant-P")


def frozen_md5():
    out = {}
    for f in ('cells_h3.npz', 'verdict_h3.json'):
        p = os.path.join(H3, f)
        out[f] = hashlib.md5(open(p, 'rb').read()).hexdigest()[:12]
    return out


def label(mre_c, has_sig_c, mre_a):
    """⛔결과 전 동결 판정표(PREREG §2)."""
    if mre_c <= MRE_THRESH and has_sig_c:
        return ('THRESHOLD-LAW-FAITHFUL',
                f'MRE_C = {mre_c:.4f} ≤ {MRE_THRESH} — ★틀린 것은 **부류**였고 '
                '옳은 것이 이미 저장소에 있다')
    if has_sig_c and mre_c < mre_a:
        return ('LAW-FAMILY-MATTERS-NOT-ENOUGH',
                f'MRE_C = {mre_c:.4f} < MRE_A = {mre_a:.4f} 이나 문턱 {MRE_THRESH} 초과 — '
                '부류는 문제의 일부이나 충실하지 않다')
    return ('LAW-FAMILY-NOT-THE-ISSUE',
            f'MRE_C = {mre_c:.4f} vs MRE_A = {mre_a:.4f} — ⛔"부류를 바꾸면 된다" 반증. '
            '복사량 A 쪽을 봐야 한다')


def gates(m):
    A = []
    A.append(('★G-FROZEN 동결 md5 일치', bool(m.get('frozen_ok')), str(m.get('frozen_msg'))))
    A.append(('★G-REPRO Model A 가 verdict_h3 를 재현', bool(m.get('repro_ok')),
              f"relerr={m.get('repro_relerr')}"))
    A.append(('★G-FIT 두 모형이 같은 규칙(중앙값 정합)으로 적합', bool(m.get('fit_ok')),
              f"A={m.get('fit_a')} C={m.get('fit_c')}"))
    A.append(('★G-IMPORT-A received_power 가 모듈 식과 비트 동일 (임포트 증명)',
              bool(m.get('import_a_ok')), str(m.get('import_a_msg'))))
    A.append(('★G-IMPORT-B H3 의 P 와의 차이가 엡실론 해석적 상한 이하',
              bool(m.get('import_b_ok')),
              f"relerr={m.get('import_relerr'):.3e} ≤ bound={m.get('eps_bound'):.3e}"
              if m.get('import_relerr') is not None else '--'))
    A.append((f'G-NEG 치환 귀무 {N_PERM}회 · 시드 {PERM_SEED} · 두 모형 동일',
              bool(m.get('neg_ok')), str(m.get('neg_msg'))))
    A.append(('G-EXCL Model B 배제 사유가 코드 독스트링 인용',
              bool(m.get('excl_ok')), (m.get('excl_msg') or '')[:60]))
    A.append((f'G-CELLS 셀 수 = {N_CELLS}', m.get('n_cells') == N_CELLS,
              str(m.get('n_cells'))))
    probe_ok = (blocked_by([('G-FROZEN 합성', False, ''), ('★G-FIT 합성', True, '')])
                == ['G-FROZEN'])
    A.append(('G-BLOCK blocked_by 자가진단', bool(probe_ok),
              'synthetic-fail → [G-FROZEN]' if probe_ok else '⛔자가진단 실패'))
    return A


def blocked_by(gate_rows):
    block = ('G-FROZEN', 'G-REPRO', 'G-FIT', 'G-IMPORT-A', 'G-IMPORT-B',
             'G-NEG', 'G-EXCL', 'G-CELLS')
    out = []
    for name, ok, _ in gate_rows:
        head = name.replace('★', '').split()[0]
        if head in block and not ok:
            out.append(head)
    return out
