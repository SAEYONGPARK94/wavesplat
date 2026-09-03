"""`R7-FLOOR` 규칙 단일출처. ⛔`PREREG.md` 가 정본이고 이 파일이 코드로 옮긴다."""
import os
from itertools import permutations

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))

R1B = os.path.join(ROOT, 'tmp', 'R1B_SCORINGFRAME_2026-09-02')
R2B = os.path.join(ROOT, 'tmp', 'R2B_STEREONOISE_2026-09-02')

RECORDS = ['BS11a', 'BS11b', 'BS11c', 'BS11d', 'BS13a', 'BS13b', 'BS13c']
N = len(RECORDS)
ALPHA = 0.05                    # ⛔결과 전 고정
HF_CUT_HZ = 1.0                 # `R2b.G.HF_CUT_HZ` 를 되읽어 대조한다(G-BAND)

# ★§1 노출 표 — 내가 동결 전에 **본** 값. `G-EXPOSE` 가 이것을 자료와 대조한다.
EXPOSED = {
    'BS11a': dict(sg=0.2376, hf_mm=129.0, zero=0.9574, ship=0.8149, prop=2.8375),
    'BS11b': dict(sg=0.0918, hf_mm=15.6,  zero=1.0422, ship=0.8053, prop=1.4355),
    'BS11c': dict(sg=0.1428, hf_mm=12.8,  zero=0.7508, ship=0.5487, prop=0.6669),
    'BS11d': dict(sg=0.1976, hf_mm=19.7,  zero=0.9176, ship=0.5850, prop=0.8077),
    'BS13a': dict(sg=0.1296, hf_mm=66.3,  zero=0.7040, ship=0.7194, prop=0.8235),
    'BS13b': dict(sg=0.1466, hf_mm=56.9,  zero=0.7823, ship=0.6473, prop=0.6572),
    'BS13c': dict(sg=0.2053, hf_mm=76.7,  zero=1.1500, ship=0.8554, prop=3.5796),
}
EXPOSE_TOL = 0.0051             # 소수 4자리 반올림 + 노출표는 4자리로 적었다


def _rank(v):
    """평균 순위(동점 처리)."""
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da > 0 and db > 0 else float('nan')


def perm_p_one_sided(a, b):
    """전수 순열 단측 p (ρ 가 관측 이상일 확률). ⛔표본이 아니라 7! 전수."""
    obs = spearman(a, b)
    hit = tot = 0
    for pb in permutations(b):
        tot += 1
        if spearman(a, list(pb)) >= obs - 1e-12:
            hit += 1
    return obs, hit / tot


def mde_rho(a):
    """α 를 넘기는 **최소 ρ** — 전수 순열 귀무의 **지지집합 안에서** 정확히 찾는다.

    ⚠초판은 `rs[int(n*(1-alpha))]` 분위였는데 **틀렸다**(0.6786 을 냈다).
    Spearman 의 ρ 는 이산이라 분위 인덱스가 지지값에 안 떨어진다.
    n=7 의 정답은 **0.714286**(p=0.04405)이고, 이 함수가 그것을 낸다.
    ⛔이 정정은 **결과 0행**에서 이뤄졌다(PREREG §2 의 값과 일치시키는 방향).
    """
    base = list(range(len(a)))
    rs = sorted(spearman(a, list(pb)) for pb in permutations(base))
    n = len(rs)
    for v in sorted(set(rs)):
        if sum(1 for r in rs if r >= v - 1e-12) / n <= ALPHA:
            return v
    return float('nan')


def label(rho, p, c1_p):
    """⛔결과 전 동결 판정표(PREREG §2)."""
    sig = p <= ALPHA
    c1_ok = c1_p > ALPHA
    if sig and c1_ok:
        return ('FLOOR-ERODES-SKILL',
                f'rho={rho:.3f} p={p:.4f} — 참조면 잡음이 실측 기술을 갉아먹는다')
    if sig:
        return ('FLOOR-CONFOUNDED',
                f'rho={rho:.3f} p={p:.4f} — ⛔무기술 기준선도 같이 움직인다(C-1 p={c1_p:.4f})')
    return ('FLOOR-UNDECIDED',
            f'rho={rho:.3f} p={p:.4f} — ⛔"참조 잡음은 무관하다" 금지 · n=7 은 rho≥MDE 만 유의')


def gates(rows, meta):
    ok = [r for r in rows if 'error' not in r]
    A = []
    A.append(('G-PAIR 7 레코드 전건 맺힘', len(ok) == N, f'{len(ok)}/{N}'))
    A.append(('G-0b 에러 0', len(rows) == len(ok), str(len(rows) - len(ok))))
    A.append(('★G-FROZEN 동결 결과 md5 일치', bool(meta.get('frozen_ok')),
              meta.get('frozen_msg', '')))
    A.append(('★G-NEWRUN 추정기 런 0회 (임포트 0)', bool(meta.get('no_estimator')),
              meta.get('no_estimator_msg', '')))
    bad = [r['record'] for r in ok if r.get('pass_') != 'judge']
    A.append(('G-JUDGE R1b judge 패스만', not bad, ';'.join(bad[:3])))
    bad = [r['record'] for r in ok if not (HF_CUT_HZ > 1.0 / r['Tp_gauge'])]
    A.append(('★G-BAND 1 Hz 가 파랑 대역 밖 (1/Tp < 1 Hz)', not bad,
              f"1/Tp max={max(1.0 / r['Tp_gauge'] for r in ok):.3f} Hz" if ok else '--'))
    A.append(('★G-MDE MDE 를 전수 순열로 계산해 기록',
              meta.get('mde') is not None and meta['mde'] == meta['mde'],
              f"{meta.get('mde')}"))
    bad = [r['record'] for r in ok if not r.get('expose_ok')]
    A.append(('G-EXPOSE 노출표가 자료와 일치', not bad, ';'.join(bad[:3])))
    A.append(('G-COV 게이트가 자료를 봤다', len(ok) > 0, f'rows={len(ok)}'))
    # ★G-BLOCK — `blocked_by()` 자가진단. **합성 실패행**을 넣어 이름을 실제로 잡는지 본다.
    #   ⛔이름 전체를 비교하는 버그(2026-09-01 `T5-ENS` 사전등록에서 잡힌 부류)를 막는다.
    probe_ok = (blocked_by([('G-PAIR 합성', False, ''), ('★G-BAND 합성', True, '')])
                == ['G-PAIR'])
    A.append(('G-BLOCK blocked_by 자가진단', bool(probe_ok),
              'synthetic-fail → [G-PAIR]' if probe_ok else '⛔자가진단 실패'))
    return A


def blocked_by(gate_rows):
    """★자가진단 — 차단 게이트 중 실패한 것의 **이름**을 돌려준다."""
    block = ('G-PAIR', 'G-0b', 'G-FROZEN', 'G-NEWRUN', 'G-JUDGE', 'G-BAND',
             'G-MDE', 'G-EXPOSE', 'G-COV')
    out = []
    for name, ok, _ in gate_rows:
        head = name.replace('★', '').split()[0]
        if head in block and not ok:
            out.append(head)
    return out
