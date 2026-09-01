#!/usr/bin/env python3
"""`T5-ENS` 전제 검증 — ⛔**런 0회 · 추정기 0회**. 실패하면 `AssertionError`.

`PREREG.md §1` 의 P1–P7 과 `§7.1 G-SCENE(b)` 의 **원천 항등**을 재도출해
`premise_t5.json` 으로 낸다. 게이트가 그 json 을 읽는다.

⚠**줄번호로 단언하지 않는다**(심사 C 처방) — 문자열/AST 로 단언한다.
줄번호는 파일이 한 줄만 움직여도 거짓이 되고, 그때 죽는 것은 배치다.
"""
import ast
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))       # …/wavesplat/wavesplat
B2 = os.path.abspath(os.path.join(HERE, '..', 'B2_REAL_2026-08-23'))
B3 = os.path.abspath(os.path.join(HERE, '..', 'B3_REFIT_2026-08-23'))
T3 = os.path.abspath(os.path.join(HERE, '..', 'T3_NOWCAST_2026-08-30'))
SU = os.path.join(ROOT, 'wavesplat', 'optimizer', 'sequential_update.py')
LI = os.path.join(ROOT, 'wavesplat', 'optimizer', 'learned_init.py')
RUN = os.path.join(ROOT, 'wavesplat', 'run.py')

# ★`T3` 가 동결한 공유 모듈 md5 — 이것이 같으면 **장면 구성이 T3 와 같다**(원천 항등).
T3_FROZEN_SHARED = {
    '../B3_REFIT_2026-08-23/gates_b3.py': '69272726ecb80da533e7c58c4fcb7b67',
    '../B2_REAL_2026-08-23/scene_station.py': '68dcd2073c4ec60897645faa29b16172',
    '../B2_REAL_2026-08-23/zero_ref.py': '9d3dada84a71722dc81089bcd7fe3f80',
}
FAILS = []


def chk(name, ok, detail=''):
    print(f'  [{"OK " if ok else "FAIL"}] {name}   {detail}')
    if not ok:
        FAILS.append(name)
    return ok


def src(p):
    return open(p, encoding='utf-8').read()


def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def find_func(path, name):
    """함수 소스만 잘라낸다 — 줄번호 대신 이름으로 찾는다."""
    tree = ast.parse(src(path))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(src(path), node) or ''
    return ''


def main():
    out = {'batch': 'T5-ENS', 'runs': 0, 'estimator_calls': 0, 'premises': {}}
    print('── PREREG §1 전제 (런 0회) ──────────────────────────────────────')

    su, li = src(SU), src(LI)

    # P1 — 패키지에 learned median-of-k 경로가 없다
    mk = find_func(SU, 'make_updater')
    p1a = "init_method') == 'auto'" in mk or 'init_method\') == "auto"' in mk
    p1b = "route != 'v2_konly'" in su
    p1c = "cfg['init_method'] = 'v2_konly'" in su
    chk('P1a make_updater 래퍼는 init_method==auto 에만', p1a)
    chk('P1b 래퍼가 route != v2_konly 에서 접힌다', p1b)
    chk('P1c 멤버 루프가 init_method 를 v2_konly 로 하드 세팅', p1c)
    out['premises']['P1_no_learned_ensemble_path'] = bool(p1a and p1b and p1c)

    # P2 — bank 앙상블 기본값 3
    p2 = 'DEFAULT_REAL_BANK_ENSEMBLE_K = 3' in su
    chk('P2 DEFAULT_REAL_BANK_ENSEMBLE_K = 3', p2)
    out['premises']['P2_bank_ensemble_default_3'] = bool(p2)

    # P5 — learned_diffusion_seed 는 패키지 정규 키
    p5 = "config.get('learned_diffusion_seed'" in li or \
         "cfg.get('learned_diffusion_seed'" in su
    chk('P5 learned_diffusion_seed 가 패키지 config 키', p5)
    out['premises']['P5_dseed_is_package_key'] = bool(p5)

    # P6 — analytic_ensemble_k 는 learned 경로에서 무연산 (P1a 의 귀결)
    chk('P6 analytic_ensemble_k 무연산 (P1a 귀결)', p1a)
    out['premises']['P6_analytic_k_inert_for_learned'] = bool(p1a)

    # P7 — ★config['device'] 가 제안 경로에 도달하지 않는다
    pa = find_func(LI, 'propose_atoms')
    p7a = "device = 'cuda' if torch.cuda.is_available() else 'cpu'" in pa
    # 호출 지점에 device= 인자가 없다
    p7b = False
    for node in ast.walk(ast.parse(li)):
        if isinstance(node, ast.Call) and getattr(node.func, 'id', '') == 'propose_atoms':
            p7b = all(k.arg != 'device' for k in node.keywords)
    chk('P7a propose_atoms 가 스스로 장치를 고른다', p7a)
    chk('P7b 호출 지점에 device= 인자가 없다', p7b)
    out['premises']['P7_device_does_not_reach_proposal'] = bool(p7a and p7b)

    # ── ★G-SCENE(b): 원천 항등 ────────────────────────────────────────────
    print('── G-SCENE(b) 원천 항등 (⛔M8 처방②) ───────────────────────────')
    shared_ok = True
    for rel, want in T3_FROZEN_SHARED.items():
        p = os.path.abspath(os.path.join(HERE, rel))
        got = md5(p) if os.path.exists(p) else 'MISSING'
        ok = (got == want)
        shared_ok &= ok
        chk(f'공유 모듈 md5 == T3 동결  {os.path.basename(p)}', ok, got[:8])

    # 장면 상수표에서 depth 를 직접 읽는다(추정기 0회 · 자료 로드 0회)
    sys.path.insert(0, B2)
    import scene_station as SS
    recs = getattr(SS, 'RECORDS_12', None)
    depth_ok = isinstance(recs, dict) and len(recs) == 12
    depths = {k: (v.get('depth') if isinstance(v, dict) else None)
              for k, v in (recs or {}).items()}
    chk('장면 상수표 RECORDS_12 12 레코드', depth_ok, f'{sorted(depths)}')
    chk('전 레코드 depth 정의됨', all(d is not None for d in depths.values()), f'{depths}')
    out['scene_depths'] = depths
    out['scene_source_identity'] = bool(shared_ok and depth_ok
                                        and all(d is not None for d in depths.values()))
    chk('★scene_source_identity', out['scene_source_identity'])

    # ── 문턱·검정력 재도출 (문서와 대조) ──────────────────────────────────
    print('── 문턱 재도출 ────────────────────────────────────────────────')
    sys.path.insert(0, HERE)
    import gates_t5 as G
    t, m = G.thresholds(), G.mde()
    out['thresholds'], out['mde'] = t, m
    out['power'] = {q: round(G.power_at(q), 4) for q in (0.806, 0.70, 0.60, 0.50)}
    chk('NOMINAL == 24', t['NOMINAL'] == 24, str(t['NOMINAL']))
    chk('HOLM_MIN_WINS == [25, 24]', t['HOLM_MIN_WINS'] == [25, 24], str(t['HOLM_MIN_WINS']))
    chk('NOT_EARN == 11', t['NOT_EARN'] == 11, str(t['NOT_EARN']))
    chk('MDE q80 == 0.742', m['q80'] == 0.742, str(m['q80']))
    chk('검정력 q=0.70 ≈ 0.6091', abs(out['power'][0.70] - 0.6091) < 5e-4,
        str(out['power'][0.70]))
    out['expected_rows'], out['expected_runs'] = G.expected_rows(), G.expected_runs()
    chk('기대 행/런 == 72/144',
        (out['expected_rows'], out['expected_runs']) == (72, 144),
        f'{out["expected_rows"]}/{out["expected_runs"]}')

    with open(os.path.join(HERE, 'premise_t5.json'), 'w') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, sort_keys=True)
    print('──────────────────────────────────────────────────────────────────')
    if FAILS:
        raise AssertionError(f'전제 검증 실패 {len(FAILS)}건: {FAILS}')
    print('✅전제 전건 통과 · 런 0회 · 추정기 0회 → premise_t5.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
