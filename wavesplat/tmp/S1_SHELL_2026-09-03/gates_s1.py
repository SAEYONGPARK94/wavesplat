"""`S1-SHELL` 규칙 단일출처. ⛔`PREREG.md` 가 정본이고 이 파일이 코드로 옮긴다."""
import os
import re
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
SENS = os.path.join(ROOT, 'tmp', 'S_ENS_2026-09-01')

SHELLS = ('0_5', '5_10', '10_20', '20_30', '30_50', '50_75',
          '75_100', '100_150', '150_inf')
NEAR = ('0_5', '5_10')                 # ⛔결과 전 고정
FAR = ('75_100', '100_150', '150_inf')  # ⛔결과 전 고정
ARM_BASE, ARM_ENS = 'LEARN-M1', 'LEARN-M3'
N_EPISODES = 64
ALPHA = 0.05
SEED_EXPECT = 1626


def sign_test_one_sided(w, n):
    return sum(comb(n, k) for k in range(w, n + 1)) / 2.0 ** n


def min_w(n, alpha=ALPHA):
    """★`G-THRESH` — α 를 넘기는 **최소 정수**를 전수로 계산한다(⛔산문 신뢰 금지)."""
    for w in range(n // 2, n + 1):
        if sign_test_one_sided(w, n) <= alpha:
            return w
    return n + 1


def label(w_far, w_near, n, thr):
    """⛔결과 전 동결 판정표(PREREG §2)."""
    far_ok, near_ok = w_far >= thr, w_near >= thr
    if far_ok and w_near <= n - thr:
        return ('ENSEMBLE-BUYS-FAR-SELLS-NEAR',
                f'먼 {w_far}/{n} · 근 {w_near}/{n} (문턱 {thr}) — ★교환이다')
    if far_ok and near_ok:
        return ('ENSEMBLE-GAINS-EVERYWHERE',
                f'먼 {w_far}/{n} · 근 {w_near}/{n} — 교환이 아니다')
    if (not far_ok) and near_ok:
        return ('ENSEMBLE-BUYS-NEAR', f'먼 {w_far}/{n} · 근 {w_near}/{n} — ⛔예상과 반대')
    return ('SHELL-UNDECIDED', f'먼 {w_far}/{n} · 근 {w_near}/{n} (문턱 {thr})')


def _sens_src():
    out = ''
    for fn in ('analyze_sens.py', 'gates_sens.py', 'verdict_sens.json'):
        p = os.path.join(SENS, fn)
        if os.path.exists(p):
            out += open(p, encoding='utf-8', errors='replace').read()
    return out


def gates(rows, meta):
    ok = [r for r in rows if 'error' not in r]
    crow = [r for r in ok if r['stratum'] == 'crowded']
    A = []
    A.append(('★G-FROZEN S_ENS 결과 md5 일치', bool(meta.get('frozen_ok')),
              str(meta.get('frozen_msg'))))
    src = _sens_src()
    n_shell = len(re.findall(r'avg_shell_rmse', src))
    n_near = len(re.findall(r'avg_RMSE_near', src))
    A.append(('★G-UNUSED S_ENS 가 셸 키를 0회 썼다(= 새 질문)',
              n_shell == 0 and n_near == 0, f'shell {n_shell} · near {n_near}'))
    A.append((f'★G-PAIR 짝 {2 * N_EPISODES} 전건 맺힘', len(ok) == 2 * N_EPISODES,
              f'{len(ok)}/{2 * N_EPISODES}'))
    A.append(('★G-THRESH 문턱을 전수 계산해 기록', meta.get('thr') == min_w(N_EPISODES),
              f"{meta.get('thr')} (p={sign_test_one_sided(min_w(N_EPISODES), N_EPISODES):.4f})"))
    bad = [r['episode'] for r in ok if r.get('seed') != SEED_EXPECT]
    A.append((f'G-SEED 시드 {SEED_EXPECT} 하나', not bad,
              ';'.join(sorted(set(bad))[:3]) or str(SEED_EXPECT)))
    A.append(('★G-NORUN 추정기 임포트 0', bool(meta.get('norun')), str(meta.get('norun_msg'))))
    A.append(('G-0b 에러 0', len(rows) == len(ok), str(len(rows) - len(ok))))
    A.append((f'G-CROWDED 혼잡 {N_EPISODES}', len(crow) == N_EPISODES, str(len(crow))))
    probe = (blocked_by([('G-PAIR 합성', False, ''), ('★G-THRESH 합성', True, '')]) == ['G-PAIR'])
    A.append(('G-BLOCK blocked_by 자가진단', bool(probe),
              'synthetic-fail → [G-PAIR]' if probe else '⛔자가진단 실패'))
    return A


def blocked_by(gate_rows):
    block = ('G-FROZEN', 'G-UNUSED', 'G-PAIR', 'G-THRESH', 'G-SEED', 'G-NORUN',
             'G-0b', 'G-CROWDED')
    out = []
    for name, ok, _ in gate_rows:
        head = name.replace('★', '').split()[0]
        if head in block and not ok:
            out.append(head)
    return out
