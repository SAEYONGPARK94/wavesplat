"""`R6-CONF` 규칙 단일출처. ⛔`PREREG.md` 가 정본. 상수는 전부 임포트(재선언 0)."""
import hashlib
import json
import os
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
B2D = os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23')
R3B = os.path.join(ROOT, 'tmp', 'R3B_GEOMETRY_2026-09-02')
for p in (B2D, R3B, ROOT, os.path.join(ROOT, 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)
import gates_b2 as B2                                               # noqa: E402
import gates_r3b as R3                                              # noqa: E402

RECORDS = [rec for _, rec, _ in R3.PAIRS]        # ★기하가 복원된 7 레코드
N = len(RECORDS)
WIN, SEED = B2.WIN, B2.SEED
BASE_ARMS = dict(B2.ARMS)                        # {'SHIP': {}, 'PROP': {...}}
# ★`CVAR` = 미클램프 conf 를 자기 최대로 나눈 **계측기**. ⛔배포 후보가 아니다(PREREG §4).
ARM_ORDER = ['SHIP-C1', 'SHIP-CVAR', 'PROP-C1', 'PROP-CVAR']
FROZEN_TOL = 1e-12
VARY_MIN_RATIO = 1.5                             # G-VARY: 탐침이 실제로 변동해야 한다
R3B_VERDICT = os.path.join(R3B, 'verdict_r3b.json')
R1B_ROWS = os.path.join(ROOT, 'tmp', 'R1B_SCORINGFRAME_2026-09-02', 'results_r1b.jsonl')
SHIPPING_KEYS = ('varpro_steps', 'learned_init_iters', 'auto_switch_thresh',
                 'learned_ensemble_k')


def shipping_expect():
    from wavesplat.run import DEFAULT_CONFIG
    return {k: DEFAULT_CONFIG[k] for k in SHIPPING_KEYS}


def d_ref_default():
    """⛔손복사 금지 — `observation_weights` 의 기본값을 되읽는다."""
    import inspect
    from wavesplat.model.confidence import observation_weights
    return float(inspect.signature(observation_weights).parameters['d_ref'].default)


def measured_geometry():
    raw = open(R3B_VERDICT, 'rb').read()
    v = json.loads(raw.decode('utf-8'))
    out = {r[0]: dict(x_s=float(r[1]), y_s=float(r[2]), h=float(r[3]))
           for r in v['secondary']['cam_center']}
    return out, hashlib.sha256(raw).hexdigest()[:16]


def base_of(arm):
    return 'SHIP' if arm.startswith('SHIP') else 'PROP'


def geo_of(arm):
    return arm.endswith('CVAR')


def sign_test_one_sided(w, n):
    return sum(comb(n, k) for k in range(w, n + 1)) / 2.0 ** n


def label(n_saturated):
    """⛔결과 전 동결 판정표(PREREG §3). ⚠부호검정이 아니다 — 자료의 성질이다."""
    if n_saturated == N:
        return ('CONF-SATURATES-AT-STATION-RANGE',
                f'{N}/{N} 레코드에서 포화 비율 1.000 — 배포 신뢰가중은 이 사거리에서 **상수**다')
    return ('CONF-PARTIALLY-VARIES',
            f'{n_saturated}/{N} 만 포화 — ⛔변동하는 레코드를 명시할 것')


def gates(rows, geo_sha):
    ok = [r for r in rows if 'error' not in r]
    by = {(r['cell'], r['record']): r for r in ok}
    A = []
    A.append(('G-0 런 수', len(ok) == len(ARM_ORDER) * N, f'{len(ok)}/{len(ARM_ORDER) * N}'))
    A.append(('G-0b 에러 0', len(rows) == len(ok), str(len(rows) - len(ok))))

    # ★G-VARY — 탐침이 실제로 변동하는가. ⛔변동 안 하면 SEC 가 아무것도 못 가린다.
    var_ok = [r['record'] for r in ok if r.get('conf_stats')
              and r['conf_stats']['conf_max'] / max(r['conf_stats']['conf_min'], 1e-12)
              >= VARY_MIN_RATIO]
    A.append((f'★G-VARY 탐침 가중이 실제로 변동 (max/min ≥ {VARY_MIN_RATIO})',
              len(set(var_ok)) == N, f'{len(set(var_ok))}/{N}'))

    # ★G-READ — **기록**한다. ⚠어느 쪽도 실패가 아니다(PREREG §5) — 그것이 측정 대상이다.
    rd = {}
    for base in ('SHIP', 'PROP'):
        rd[base] = sum(1 for k in RECORDS
                       if (f'{base}-C1', k) in by and (f'{base}-CVAR', k) in by
                       and by[(f'{base}-C1', k)]['curve'] != by[(f'{base}-CVAR', k)]['curve'])
    A.append(('★G-READ 경로별 비트 변화 기록 (⚠실패 아님)', True,
              f"SHIP {rd['SHIP']}/{N} · PROP {rd['PROP']}/{N}"))

    A.append(('★G-NOGT 신뢰가중 코드 경로에 참 장이 없다',
              all(r.get('nogt_ok') for r in ok), f"{sum(1 for r in ok if r.get('nogt_ok'))}/{len(ok)}"))
    A.append(('★G-GEOMSRC 기하가 R3b verdict 에서 파싱됨',
              bool(geo_sha) and all(r.get('geom_sha') == geo_sha for r in ok), str(geo_sha)))

    anc = [r for r in ok if r['cell'] == 'SHIP-C1']
    worst = max([r.get('anchor_relerr', 1.0) for r in anc], default=1.0)
    A.append((f'G-ANCHOR SHIP-C1 이 동결 R1b(judge/옛규약) 를 재현 ({len(anc)}행)',
              len(anc) == N and worst <= FROZEN_TOL, f'max relerr={worst:.3e}'))

    dr = d_ref_default()
    A.append(('G-CONF observation_weights 임포트 · d_ref 기본값 일치',
              all(r.get('d_ref') == dr for r in ok), f'd_ref={dr}'))

    exp = shipping_expect()
    A.append((f'G-ARM 출하 되읽기 {exp}', all(r.get('shipping') == exp for r in ok), ''))
    A.append(('G-ENV 단일 PID · 단일 code_md5 · 시드 42',
              len({r['pid'] for r in ok}) == 1
              and len({json.dumps(r['code_md5'], sort_keys=True) for r in ok}) == 1
              and all(r.get('seed') == SEED for r in ok), f"pid={len({r['pid'] for r in ok})}"))
    probe = blocked_by([('X-거짓', False, ''), ('Y-참', True, '')])
    A.append(('★G-BLOCK blocked_by 자가진단', probe == ['X-거짓'], repr(probe)))
    A.append(('G-COV 모든 게이트가 자료를 봤다', len(ok) > 0, f'rows={len(ok)}'))
    return A


def blocked_by(gate_rows):
    """★자가진단 대상 — 실패한 게이트 이름을 **비어있지 않게** 돌려줘야 한다."""
    return [g[0] for g in gate_rows if not g[1]]
