"""`C1-MODELGAP` 규칙 단일출처. ⛔`PREREG.md` 가 정본이고 이 파일이 코드로 옮긴다."""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
HOS = os.path.join(ROOT, 'tmp', 'HOSGATE_2026-08-13', 'data_hos')
LIN = os.path.join(ROOT, 'data')

STRIDE = 25                 # ⛔결과 전 고정
GAP_MIN = 0.10              # PRIMARY 문턱 (max|Δ| / σ)
N_CROWDED = 48
N_SPARSE = 16
N_CTRL_MIN = 8
THRESH_ALL = 48
THRESH_MOST = 44

# 열 규약 — ⛔손으로 적지 않는다. `G-COLS` 가 소스에서 읽어 대조한다.
COLS = ('A', 'omega', 'kx', 'ky', 'phi', 'Q')


def cols_from_source():
    """★`G-COLS` — `wavesim/wave_generator.py` 의 `WaveComponent` 필드 순서를 읽는다."""
    src = open(os.path.join(ROOT, 'wavesim', 'wave_generator.py'), encoding='utf-8').read()
    m = re.search(r'class WaveComponent:.*?(?=\n\s*@property|\nclass |\ndef )', src, re.S)
    if not m:
        return None
    return tuple(re.findall(r'^\s{4}(\w+)\s*:\s*float', m.group(0), re.M))


def is_crowded(name):
    return ('15comp' in name) or ('30comp' in name)


def label(w):
    """⛔결과 전 동결 판정표(PREREG §3)."""
    if w >= THRESH_ALL:
        return ('HOS-LEAVES-THE-MODEL-CLASS',
                f'{w}/{N_CROWDED} — ★사슬이 도는 코퍼스는 **닫힌 고리가 아니다**')
    if w >= THRESH_MOST:
        return ('HOS-MOSTLY-LEAVES', f'{w}/{N_CROWDED} — 예외 에피소드를 명시한다')
    return ('HOS-GAP-UNDECIDED', f'{w}/{N_CROWDED}')


def gates(rows, meta):
    ok = [r for r in rows if 'error' not in r]
    ctrl = [r for r in ok if r['group'] == 'linear']
    crow = [r for r in ok if r['group'] == 'hos' and r['crowded']]
    spar = [r for r in ok if r['group'] == 'hos' and not r['crowded']]
    A = []
    bad = [r['episode'] for r in ctrl if r['max_abs_delta'] != 0.0]
    A.append(('★G-CTRL 선형 코퍼스 max|Δ| == 정확히 0.0', not bad and len(ctrl) > 0,
              (f'{len(ctrl)}개 전건 0.0' if not bad else ';'.join(bad[:3]))))
    A.append(('★G-COLS 열 규약을 소스에서 읽어 대조', meta.get('cols') == list(COLS),
              str(meta.get('cols'))))
    A.append((f'G-N 혼잡 {N_CROWDED} · 희소 {N_SPARSE} · 대조 ≥ {N_CTRL_MIN}',
              len(crow) == N_CROWDED and len(spar) == N_SPARSE and len(ctrl) >= N_CTRL_MIN,
              f'{len(crow)}/{len(spar)}/{len(ctrl)}'))
    A.append(('★G-NORUN 추정기 임포트 0', bool(meta.get('norun')), str(meta.get('norun_msg'))))
    A.append(('G-0b 에러 0', len(rows) == len(ok), str(len(rows) - len(ok))))
    A.append(('G-COV 게이트가 자료를 봤다', len(ok) > 0, f'rows={len(ok)}'))
    probe = (blocked_by([('G-CTRL 합성', False, ''), ('★G-COLS 합성', True, '')]) == ['G-CTRL'])
    A.append(('G-BLOCK blocked_by 자가진단', bool(probe),
              'synthetic-fail → [G-CTRL]' if probe else '⛔자가진단 실패'))
    return A


def blocked_by(gate_rows):
    block = ('G-CTRL', 'G-COLS', 'G-N', 'G-NORUN', 'G-0b', 'G-COV')
    out = []
    for name, ok, _ in gate_rows:
        head = name.replace('★', '').split()[0]
        if head in block and not ok:
            out.append(head)
    return out
