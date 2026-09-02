"""`R1b-SCORINGFRAME` 규칙 단일출처. ⛔`PREREG.md` 가 정본이고 이 파일이 그것을 코드로 옮긴다.

⛔팔·레코드·시드는 `gates_b2` 를 **임포트**한다(손복사 금지 — 저장소의 반복 결함).
"""
import os
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for p in (os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23'), ROOT,
          os.path.join(ROOT, 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)
import gates_b2 as B2                                              # noqa: E402

RECORDS = list(B2.RECORDS)
ARMS = dict(B2.ARMS)          # ⛔재선언하지 않는다
ARM_ORDER = list(B2.ARM_ORDER)
WIN = B2.WIN
SEED = B2.SEED
LEAD_MAX = 14.0               # 동결 배치 전건의 값

# ── PRIMARY (PREREG §4, 결과 전 동결) ────────────────────────────────
N = 12
THRESH_SURVIVES = 10          # 단측 부호검정 p = 0.0193
THRESH_BREAKS = 7
# ── `G-INVAR` 이 훑는 채점창 길이(프레임). ⛔`i0` 는 고정한다 ────────
NLEAD_GRID = (20, 50, 80, None)      # None = 레코드의 14 s 상당
ORACLE_TOL = 1e-6             # ★개정 1: int16 packed 저장의 반올림(≈5e-8)의 20배. 판정 문턱 아님
FROZEN_TOL = 1e-12            # 옛 규약이 동결 결과를 재현해야 하는 상대오차
DEPTH_FIX_COMMIT = '97e8b591'  # 수심 17→30 정정 커밋. 앵커 수심은 그 **직전 블롭**에서 파싱한다
N_DEPTH_CHANGED = 7            # ★BS 7. ⛔손으로 적은 게 아니라 `G-DEPTH` 가 자료로 확인한다
PASSES = ('anchor', 'judge')   # anchor = B-2 시절 수심 · judge = 현행 수심

# ★출하 기대값 — ⛔손복사 아님: 배포 `DEFAULT_CONFIG` 에서 읽는다(PREREG §7b).
SHIPPING_KEYS = ('varpro_steps', 'learned_init_iters', 'auto_switch_thresh',
                 'learned_ensemble_k')


def _shipping_expect():
    from wavesplat.run import DEFAULT_CONFIG
    return {k: DEFAULT_CONFIG[k] for k in SHIPPING_KEYS}


SHIPPING_EXPECT = _shipping_expect()


def sign_test_one_sided(w, n):
    return sum(comb(n, k) for k in range(w, n + 1)) / 2.0 ** n


def label(w):
    """⛔결과 전 동결 판정표(PREREG §4)."""
    if w >= THRESH_SURVIVES:
        return ('REAL-SKILL-SURVIVES-FIX',
                f'{w}/{N} — 실측 판정은 demean 규약의 산물이 아니다')
    if w <= THRESH_BREAKS:
        return ('FIX-BREAKS-REAL-SKILL',
                f'{w}/{N} — ⛔수리가 판정을 죽였다; B-2 계수 전면 재검토')
    return ('SCORINGFRAME-UNDECIDED',
            f'{w}/{N} — ⛔"차이 없다" 금지 · MDE 병기')


def gates(rows, probes):
    """차단 게이트 (PREREG §6). `rows` = 결과행, `probes` = 런 0회 진단."""
    ok = [r for r in rows if 'error' not in r]
    anc = [r for r in ok if r['pass_'] == 'anchor']
    jud = [r for r in ok if r['pass_'] == 'judge']
    exp = len(ARMS) * len(RECORDS) * len(PASSES)
    A = []

    A.append(('G-0 런 수', len(ok) == exp, f'{len(ok)}/{exp}'))
    A.append(('G-0b 에러 0', len(rows) == len(ok), str(len(rows) - len(ok))))

    # ★G-ORACLE — 참 장이 새 규약에서 0 점, 옛 규약에서는 0 이 아니다(항진 방지)
    orc_new = [p['oracle_new'] for p in probes]
    orc_old = [p['oracle_old'] for p in probes]
    A.append(('G-ORACLE 새 규약에서 참 장 = 0',
              len(orc_new) == N and all(v <= ORACLE_TOL for v in orc_new),
              f'max={max(orc_new) if orc_new else float("nan"):.3e}'))
    A.append(('G-ORACLE-NEG 옛 규약에서는 0 이 아니다 (항진 방지)',
              len(orc_old) == N and all(v > ORACLE_TOL for v in orc_old),
              f'min={min(orc_old) if orc_old else float("nan"):.3e}'))

    # ★G-INVAR — 새 규약의 lead0 기준이 채점창 길이에 불변, 옛 규약은 아니다
    inv_new = [p['invar_new_spread'] for p in probes]
    inv_old = [p['invar_old_spread'] for p in probes]
    A.append(('G-INVAR 새 규약 기준이 채점창 길이에 불변',
              len(inv_new) == N and all(abs(v - 1.0) <= 1e-9 for v in inv_new),
              f'max={max(inv_new) if inv_new else float("nan"):.9f}'))
    A.append(('G-INVAR-NEG 옛 규약은 불변이 아니다 (항진 방지)',
              len(inv_old) == N and all(v > 1.0 + 1e-9 for v in inv_old),
              f'범위 {min(inv_old):.3f}–{max(inv_old):.3f}' if inv_old else '--'))

    # G-SCENE — 장면이 동결 B-2 와 비트 동일
    bad = [r['record'] for r in ok if not r.get('scene_match')]
    A.append((f'G-SCENE 장면이 동결 B-2 와 비트 동일 ({len(ok)}행)', not bad, ';'.join(bad[:3])))

    # ★G-FROZEN — **anchor 패스**의 옛 규약 채점이 동결 B-2 를 재현
    #   ⛔judge 패스는 정정된 수심이라 재현할 **의무가 없다**(선행 R1 이 여기서 막혔다).
    worst = max([r.get('frozen_relerr', 1.0) for r in anc], default=1.0)
    A.append((f'★G-FROZEN anchor 가 동결 B-2 를 재현 ({len(anc)}행)',
              len(anc) == len(ARMS) * len(RECORDS) and worst <= FROZEN_TOL,
              f'max relerr={worst:.3e}'))

    # ★G-DEPTH — 앵커 수심이 git 블롭에서 나왔고, 현행과 정확히 BS 7 에서 다르다
    chg = sorted({r['record'] for r in ok if r['depth_anchor'] != r['depth_now']})
    A.append((f'★G-DEPTH 앵커 수심 = git 블롭 · 현행과 {N_DEPTH_CHANGED} 레코드에서 상이',
              len(chg) == N_DEPTH_CHANGED
              and all(r.get('depth_src') == f'git:{DEPTH_FIX_COMMIT}^' for r in ok),
              f'{len(chg)}개 {chg[:3]}'))

    # ★G-IDENT — 수심이 같은 5 레코드에서 두 패스가 비트 동일
    same = {r['record'] for r in ok if r['depth_anchor'] == r['depth_now']}
    pairs, bad = 0, []
    for a in anc:
        if a['record'] not in same:
            continue
        j = [x for x in jud if x['record'] == a['record'] and x['cell'] == a['cell']]
        if not j:
            bad.append(a['record']); continue
        pairs += 1
        if a['curve_old'] != j[0]['curve_old'] or a['curve_new'] != j[0]['curve_new']:
            bad.append(a['record'])
    A.append((f'★G-IDENT 수심 동일 {len(same)} 레코드에서 두 패스가 비트 동일',
              pairs == len(same) * len(ARMS) and not bad, f'짝{pairs} 불일치{sorted(set(bad))[:3]}'))

    # G-ARM — 팔 구성 객체 되읽기 · 빈 dict 팔 0
    bad = [f"{r['cell']}" for r in ok if not r.get('arm_ok')]
    A.append((f'★G-ARM 팔 구성 객체 되읽기 (SHIP 면제 없음) {SHIPPING_EXPECT}',
              not bad, ';'.join(sorted(set(bad))[:3])))

    # G-ENV — 단일 PID · 단일 code_md5 · 스레드 1
    A.append(('G-ENV 단일 PID · 단일 code_md5 · 1스레드',
              len({r['pid'] for r in ok}) == 1
              and len({repr(sorted(r['code_md5'].items())) for r in ok}) == 1
              and all(r.get('threads') == 1 for r in ok),
              f"pid={len({r['pid'] for r in ok})}"))

    # G-COV — 커버리지 0 게이트 금지
    A.append(('G-COV 모든 게이트가 실제로 자료를 봤다',
              len(ok) > 0 and len(probes) == N and len(anc) > 0 and len(jud) > 0,
              f'probes={len(probes)} anchor={len(anc)} judge={len(jud)}'))
    return A


def blocked_by(gate_rows):
    """★자가진단 대상 — 실패한 게이트 이름을 **비어있지 않게** 돌려줘야 한다."""
    return [g[0] for g in gate_rows if not g[1]]
