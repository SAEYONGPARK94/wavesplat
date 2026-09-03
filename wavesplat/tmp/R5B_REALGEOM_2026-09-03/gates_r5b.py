"""`R5b-REALGEOM` 규칙 단일출처. ⛔`PREREG.md` 가 정본.

⛔상수·층·규칙을 전부 `gates_b17` 에서 임포트한다. 이 배치가 바꾸는 것은
**축 배정 하나**(+수준 통제 팔 하나)이고, 재선언하면 그 주장이 거짓이 된다.
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
B17 = os.path.join(ROOT, 'tmp', 'B17_REALNOISE_2026-08-25')
R3B = os.path.join(ROOT, 'tmp', 'R3B_GEOMETRY_2026-09-02')
R5 = os.path.join(ROOT, 'tmp', 'R5_REALGEOM_2026-09-03')
for p in (B17, ROOT, os.path.join(ROOT, 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)
import gates_b17 as B                                               # noqa: E402

RECORDS, LARGE_BOX, SMALL_BOX = B.RECORDS, B.LARGE_BOX, B.SMALL_BOX
WIN, SEED = B.WIN, B.SEED
SIGMA0, SIGMA_R, D_REF, COS_MIN, MM_TARGET = B.SIGMA0, B.SIGMA_R, B.D_REF, B.COS_MIN, B.MM_TARGET
ESTIMATOR_CFG = dict(B.ESTIMATOR_CFG)
DERIVED_READBACK, ROUTER_ATTRS, ATTR_MISSING = dict(B.DERIVED_READBACK), list(B.ROUTER_ATTRS), B.ATTR_MISSING
STRATUM_SURVIVE_RATIO = B.STRATUM_SURVIVE_RATIO
STRATUM_COLLAPSE_RATIO = B.STRATUM_COLLAPSE_RATIO
STRATUM_SMALL_CAP = B.STRATUM_SMALL_CAP

FIX_ARM, MM_ARM = 'RANGEANG-MEASFIX', 'RANGEANG-MEASFIX-MM'
ARM_ORDER = [FIX_ARM, MM_ARM]
ANCHOR_ARM = 'RANGEANG'              # ⛔재실행하지 않는다 — R5 동결 행을 읽는다
REF_ARM = B.REF_ARM                  # 'CONST' — 동결 B-17 행
TREAT_RECORDS = list(SMALL_BOX)      # ★측정 기하가 있는 7 = 소형 박스
MM_TOL = 1e-6
R3B_VERDICT = os.path.join(R3B, 'verdict_r3b.json')
R5_ROWS = os.path.join(R5, 'results_r5_*.jsonl')
B17_GLOB = os.path.join(B17, 'results_b17_*.jsonl')


def measured_geometry():
    """★축 정정 — `PREREG §0`. `cam_center = (nc-X, nc-Y, z)` 인데 장면은
    `P[:,0]=nc-Y`, `P[:,1]=nc-X` 이므로 **`x_s = C[1]`, `y_s = C[0]`** 이 옳다."""
    raw = open(R3B_VERDICT, 'rb').read()
    v = json.loads(raw.decode('utf-8'))
    out = {r[0]: dict(x_s=float(r[2]), y_s=float(r[1]), h=float(r[3]),
                      raw_center=[float(r[1]), float(r[2]), float(r[3])])
           for r in v['secondary']['cam_center']}
    return out, hashlib.sha256(raw).hexdigest()[:16]


def arm_spec(arm, record, geo):
    spec = dict(B.ARMS['RANGEANG'])
    g = geo[record]
    spec.update(name=arm, x_s=g['x_s'], y_s=g['y_s'], h=g['h'], mm=(arm == MM_ARM))
    return spec


def stratum_label(large_w, small_w):
    """⛔결과 전 동결 판정표(PREREG §2). 비율 상수는 `gates_b17` 임포트."""
    lr, sr = large_w / len(LARGE_BOX), small_w / len(SMALL_BOX)
    if lr - sr >= STRATUM_SURVIVE_RATIO and small_w <= STRATUM_SMALL_CAP:
        return ('GEOM-STRATUM-SURVIVES-FIXED',
                f'대형 {large_w}/{len(LARGE_BOX)}(승계) · 소형 {small_w}/{len(SMALL_BOX)} '
                f'· gap {lr - sr:+.3f}')
    if lr - sr <= STRATUM_COLLAPSE_RATIO:
        return ('GEOM-STRATUM-BREAKS-FIXED',
                f'gap {lr - sr:+.3f} ⟹ ⛔B-17 결론이 선언 상수의 산물이었다')
    return ('GEOM-STRATUM-UNDECIDED-FIXED', f'gap {lr - sr:+.3f} — 두 문턱 사이')


def gates(rows, geo, geo_sha, r5_rows, const_rows, anchor_ok):
    ok = [r for r in rows if 'error' not in r]
    fx = {r['record']: r for r in ok if r['cell'] == FIX_ARM}
    mm = {r['record']: r for r in ok if r['cell'] == MM_ARM}
    A = []
    A.append(('G-0 런 수', len(ok) == 2 * len(TREAT_RECORDS), f'{len(ok)}/{2 * len(TREAT_RECORDS)}'))
    A.append(('G-0b 에러 0', len(rows) == len(ok), str(len(rows) - len(ok))))

    # ★G-AXIS — 이 배치의 존재 이유. 행에서 되읽는다.
    bad = [r['record'] for r in ok
           if not (abs(r['arm_spec']['x_s'] - r['raw_center'][1]) < 1e-12
                   and abs(r['arm_spec']['y_s'] - r['raw_center'][0]) < 1e-12)]
    A.append(('★G-AXIS x_s=C[1] · y_s=C[0] (되읽기)', not bad, ';'.join(bad[:3]) or 'OK'))

    # ★G-AXIS-DIFF — 정정이 실제로 발화했는가
    diff = [k for k in TREAT_RECORDS if k in fx and k in r5_rows
            and fx[k]['sigma_stats'].get('cos_med') != r5_rows[k]['sigma_stats'].get('cos_med')]
    A.append((f'★G-AXIS-DIFF cos_med 가 R5 반사 팔과 다르다 ({len(TREAT_RECORDS)}행)',
              len(diff) == len(TREAT_RECORDS), f'{len(diff)}/{len(TREAT_RECORDS)}'))

    bad = [k for k, r in mm.items()
           if abs(r['sigma_stats'].get('mm_scale_applied', float('nan')) - 1.0) > 1.0]
    A.append((f'★G-MM 평균분산 정합 (목표 {MM_TARGET})',
              len(mm) == len(TREAT_RECORDS) and all(r['arm_spec'].get('mm') for r in mm.values()),
              f'{len(mm)}/{len(TREAT_RECORDS)}'))

    A.append(('G-GEOMSRC 기하가 R3b verdict 에서 파싱',
              bool(geo_sha) and all(r.get('geom_sha') == geo_sha for r in ok), str(geo_sha)))
    A.append((f'G-INHERIT 대형 {len(LARGE_BOX)} 승계 · 앵커가 R5 동결과 비트 동일',
              bool(anchor_ok), str(anchor_ok)))
    A.append(('G-RULE 층 규칙이 gates_b17 임포트',
              (STRATUM_SURVIVE_RATIO, STRATUM_COLLAPSE_RATIO, STRATUM_SMALL_CAP)
              == (B.STRATUM_SURVIVE_RATIO, B.STRATUM_COLLAPSE_RATIO, B.STRATUM_SMALL_CAP),
              f'{STRATUM_SURVIVE_RATIO}/{STRATUM_COLLAPSE_RATIO}/{STRATUM_SMALL_CAP}'))
    A.append((f'G-CONST 기준 {REF_ARM} 을 동결 B-17 에서 읽었다',
              len(const_rows) == len(RECORDS), f'{len(const_rows)}/{len(RECORDS)}'))
    thr = {r.get('threads') for r in ok}
    A.append(('G-ENV 단일 src_md5 · 단일 스레드수 · 시드 42',
              len({json.dumps(r.get('src_md5'), sort_keys=True) for r in ok}) == 1
              and len(thr) == 1 and all(r.get('seed') == SEED for r in ok), f'threads {sorted(thr)}'))
    probe = blocked_by([('X-거짓', False, ''), ('Y-참', True, '')])
    A.append(('★G-BLOCK blocked_by 자가진단', probe == ['X-거짓'], repr(probe)))
    A.append(('G-COV 모든 게이트가 자료를 봤다', len(ok) > 0 and len(r5_rows) > 0,
              f'rows={len(ok)} r5={len(r5_rows)}'))
    return A


def blocked_by(gate_rows):
    """★자가진단 대상 — 실패한 게이트 이름을 **비어있지 않게** 돌려줘야 한다."""
    return [g[0] for g in gate_rows if not g[1]]
