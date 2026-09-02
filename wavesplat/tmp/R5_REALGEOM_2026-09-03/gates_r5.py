"""`R5-REALGEOM` 규칙 단일출처. ⛔`PREREG.md` 가 정본.

⛔코퍼스·층·잡음 규약·추정기 구성·판정 규칙을 **전부 `gates_b17` 에서 임포트**한다.
이 배치가 바꾸는 것은 **센서 기하 입력 하나뿐**이고, 상수를 하나라도 재선언하면 그 주장이 거짓이 된다.
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
B17 = os.path.join(ROOT, 'tmp', 'B17_REALNOISE_2026-08-25')
R3B = os.path.join(ROOT, 'tmp', 'R3B_GEOMETRY_2026-09-02')
for p in (B17, ROOT, os.path.join(ROOT, 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)
import gates_b17 as B                                               # noqa: E402

RECORDS, LARGE_BOX, SMALL_BOX = B.RECORDS, B.LARGE_BOX, B.SMALL_BOX
WIN, SEED, N = B.WIN, B.SEED, B.N
SIGMA0, SIGMA_R, D_REF, COS_MIN, MM_TARGET = B.SIGMA0, B.SIGMA_R, B.D_REF, B.COS_MIN, B.MM_TARGET
ESTIMATOR_CFG = dict(B.ESTIMATOR_CFG)
STRATUM_SURVIVE_RATIO = B.STRATUM_SURVIVE_RATIO
STRATUM_COLLAPSE_RATIO = B.STRATUM_COLLAPSE_RATIO
STRATUM_SMALL_CAP = B.STRATUM_SMALL_CAP

# ⛔`run_b17._one` 이 읽는 진단 상수들 — **재선언하지 않고 그대로 노출**한다.
DERIVED_READBACK = dict(B.DERIVED_READBACK)
ROUTER_ATTRS = list(B.ROUTER_ATTRS)
ATTR_MISSING = B.ATTR_MISSING
ARMS = dict(B.ARMS)

ANCHOR_ARM = 'RANGEANG'          # 선언 기하 — 동결 B-17 을 재현해야 한다
TREAT_ARM = 'RANGEANG-MEAS'      # 측정 기하 (BS 7 만 실제로 다르다)
REF_ARM = B.REF_ARM              # 'CONST' — ⛔재실행하지 않고 동결 행을 읽는다
ARM_ORDER = [ANCHOR_ARM, TREAT_ARM]
FROZEN_TOL = 1e-12
R3B_VERDICT = os.path.join(R3B, 'verdict_r3b.json')
B17_GLOB = os.path.join(B17, 'results_b17_*.jsonl')


def measured_geometry():
    """★`R3b` 동결 verdict 에서 **파싱**한다. ⛔손으로 좌표를 적지 않는다."""
    raw = open(R3B_VERDICT, 'rb').read()
    v = json.loads(raw.decode('utf-8'))
    out = {}
    for row in v['secondary']['cam_center']:
        out[row[0]] = dict(x_s=float(row[1]), y_s=float(row[2]), h=float(row[3]))
    return out, hashlib.sha256(raw).hexdigest()[:16]


def arm_spec(arm, record, geo):
    """팔 사양. ⛔`B-17` 의 `RANGEANG` 사양을 **복사**하고 기하만 갈아 끼운다."""
    spec = dict(B.ARMS['RANGEANG'])
    if arm == ANCHOR_ARM:
        return spec, 'declared'
    g = geo.get(record)
    if g is None:
        return spec, 'declared(측정 없음)'
    spec.update(name=TREAT_ARM, x_s=g['x_s'], y_s=g['y_s'], h=g['h'])
    return spec, 'measured'


def stratum_label(large_w, small_w):
    """⛔결과 전 동결 판정표(PREREG §2). 비율 상수는 `gates_b17` 임포트."""
    lr, sr = large_w / len(LARGE_BOX), small_w / len(SMALL_BOX)
    if lr - sr >= STRATUM_SURVIVE_RATIO and small_w <= STRATUM_SMALL_CAP:
        return ('GEOM-STRATUM-SURVIVES',
                f'대형 {large_w}/{len(LARGE_BOX)} · 소형 {small_w}/{len(SMALL_BOX)} '
                f'· gap {lr - sr:+.3f} ⟹ 측정된 기하에서도 층이 남는다')
    if lr - sr <= STRATUM_COLLAPSE_RATIO:
        return ('GEOM-STRATUM-BREAKS',
                f'gap {lr - sr:+.3f} ≤ {STRATUM_COLLAPSE_RATIO} ⟹ '
                f'⛔B-17 의 결론이 선언 상수의 산물이었다')
    return ('GEOM-STRATUM-UNDECIDED', f'gap {lr - sr:+.3f} — 두 문턱 사이')


def gates(rows, const_rows, geo_sha):
    ok = [r for r in rows if 'error' not in r]
    anc = {r['record']: r for r in ok if r['cell'] == ANCHOR_ARM}
    trt = {r['record']: r for r in ok if r['cell'] == TREAT_ARM}
    A = []
    A.append(('G-0 런 수', len(ok) == 2 * N, f'{len(ok)}/{2 * N}'))
    A.append(('G-0b 에러 0', len(rows) == len(ok), str(len(rows) - len(ok))))

    worst = max([r.get('anchor_relerr', 1.0) for r in anc.values()], default=1.0)
    A.append((f'★G-ANCHOR RANGEANG 가 동결 B-17 을 재현 ({len(anc)}행)',
              len(anc) == N and worst <= FROZEN_TOL, f'max relerr={worst:.3e}'))

    A.append(('★G-GEOMSRC 측정 기하가 R3b verdict 에서 파싱됨',
              bool(geo_sha) and all(r.get('geom_sha') == geo_sha for r in ok),
              str(geo_sha)))

    same = [k for k in RECORDS if k in anc and k in trt
            and trt[k].get('geom_kind', '').startswith('declared')]
    bad = [k for k in same if anc[k]['curve'] != trt[k]['curve']]
    A.append((f'★G-IDENT5 측정 없는 {len(same)} 레코드에서 두 팔이 비트 동일',
              len(same) == N - len(SMALL_BOX) and not bad, f'동일{len(same)} 불일치{bad[:3]}'))

    diff = [k for k in SMALL_BOX if k in anc and k in trt
            and anc[k]['sigma_stats'] != trt[k]['sigma_stats']]
    A.append((f'★G-DIFF7 BS {len(SMALL_BOX)} 에서 sigma_stats 가 다르다',
              len(diff) == len(SMALL_BOX), f'{len(diff)}/{len(SMALL_BOX)}'))

    A.append(('G-RULE 판정 규칙이 gates_b17 임포트',
              (STRATUM_SURVIVE_RATIO, STRATUM_COLLAPSE_RATIO, STRATUM_SMALL_CAP)
              == (B.STRATUM_SURVIVE_RATIO, B.STRATUM_COLLAPSE_RATIO, B.STRATUM_SMALL_CAP),
              f'{STRATUM_SURVIVE_RATIO}/{STRATUM_COLLAPSE_RATIO}/{STRATUM_SMALL_CAP}'))

    A.append((f'G-CONST 기준 {REF_ARM} 12행을 동결 B-17 에서 읽었다 (런 0회)',
              len(const_rows) == N, f'{len(const_rows)}/{N}'))

    # ⚠스레드는 **1로 고정하지 않는다** — `run_b17.py` 가 고정하지 않았으므로 바꾸면
    #   축약 순서가 달라져 `G-ANCHOR` 의 비트 재현이 깨질 수 있다. **일관성만** 요구한다.
    thr = {r.get('threads') for r in ok}
    A.append(('G-ENV 단일 src_md5 · 단일 스레드수 · 시드 42',
              len({json.dumps(r.get('src_md5'), sort_keys=True) for r in ok}) == 1
              and len(thr) == 1
              and all(r.get('seed') == SEED for r in ok),
              f"src_md5 {len({json.dumps(r.get('src_md5'), sort_keys=True) for r in ok})}종 "
              f"· threads {sorted(thr)}"))

    probe = blocked_by([('X-거짓', False, ''), ('Y-참', True, '')])
    A.append(('★G-BLOCK blocked_by 자가진단', probe == ['X-거짓'], repr(probe)))
    A.append(('G-COV 모든 게이트가 자료를 봤다', len(ok) > 0 and len(const_rows) > 0,
              f'rows={len(ok)} const={len(const_rows)}'))
    return A


def blocked_by(gate_rows):
    """★자가진단 대상 — 실패한 게이트 이름을 **비어있지 않게** 돌려줘야 한다."""
    return [g[0] for g in gate_rows if not g[1]]
