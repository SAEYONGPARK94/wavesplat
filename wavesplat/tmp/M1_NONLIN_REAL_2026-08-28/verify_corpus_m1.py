"""M1 생성 산출물 검증 — ⛔**런 0회 · GPU 0회 · 판정 0건**. 읽기 전용.

왜 필요한가: `PREREG §5` 는 `gates_data` 를 *"생성 후 · 추정 전"* 이라 적었는데
`§7 실행 순서` 에는 그 단계가 **없다** — `gates_data` 는 `analyze_m1.py`(7단계) 안에서만
불린다. ⟹ 손상된 코퍼스가 **288 런을 다 태운 뒤에야** 드러난다.
이 스크립트가 그 창을 닫는다: **5단계와 6단계 사이**에서 돌린다.

⛔이름이 `gates*`·`run*`·`analyze*` 가 아니다(`tmp/_audit/check_batch_artifacts.py:77` JUDGE 규약)
  — 판정을 낳지 않는 사후 검증 스크립트이기 때문이다(`B2_REAL/depth_recheck.py` 선례).
⛔문턱·판정 상수를 **하나도 새로 정의하지 않는다**. 게이트는 동결된 `gates_m1.gates_data()`
  를 그대로 부르고, 나머지는 파일에서 잰 사실(존재·열림·md5·계수)뿐이다.

    python3 tmp/M1_NONLIN_REAL_2026-08-28/verify_corpus_m1.py
    → exit 0 = 추정(6단계) 진행 가능 · exit 1 = ⛔진행 금지
"""
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import gates_m1 as G                                           # noqa: E402

FAIL, WARN = [], []


def chk(name, ok, msg='', warn_only=False):
    tag = 'PASS' if ok else ('⚠WARN' if warn_only else '⛔FAIL')
    print(f'[{tag}] {name}: {msg}')
    if not ok:
        (WARN if warn_only else FAIL).append(name)
    return ok


# ── A. 동결 md5 (S13·S13b 가 물린 창: 배치 도중 다른 세션의 임포트 경로 수정) ────
def a_frozen():
    drift, missing, okn = [], [], 0
    for line in open(os.path.join(HERE, 'FROZEN_md5.txt'), encoding='utf-8'):
        m = re.match(r'^([0-9a-f]{32})\s+(.+)$', line.strip())
        if not m:
            continue
        h, rel = m.groups()
        for cand in (os.path.join(HERE, rel), os.path.join(ROOT, rel)):
            cand = os.path.normpath(cand)
            if os.path.exists(cand):
                break
        else:
            missing.append(rel)
            continue
        cur = hashlib.md5(open(cand, 'rb').read()).hexdigest()
        if cur == h:
            okn += 1
        else:
            drift.append((rel, h, cur))
    chk('A-1 동결 md5 전건 일치', not drift and not missing,
        f'OK {okn} · 이동 {[d[0] for d in drift]} · 부재 {missing}')
    for rel, h, cur in drift:
        print(f'        ⟹ {rel}\n           동결 {h}\n           현재 {cur}')
    return drift


# ── B. 48 파일 존재 · 열림 · 잘리지 않음 ────────────────────────────────────────
def b_files():
    import h5py
    import numpy as np
    eps, _ = G.select()
    chk('B-0 select() 48 에피소드', len(eps) == 2 * G.N_EP, f'{len(eps)}/{2*G.N_EP}')
    have = [e for e in eps if os.path.exists(e['path'])]
    chk('B-1 48 파일 존재 (⛔부족 = 생성 미완)', len(have) == 2 * G.N_EP,
        f'{len(have)}/{2*G.N_EP}')
    sizes, broken, nfr = [], [], set()
    for e in have:
        try:
            with h5py.File(e['path'], 'r') as f:
                gt = f['frames/gt_wave_field']
                nfr.add(int(gt.shape[0]))
                # ★잘린 h5 는 헤더가 열려도 **마지막 청크**에서 죽는다 — 실제로 읽는다
                _ = np.asarray(gt[-1])
                _ = np.asarray(f['frames/lidar_points'][-1])
                _ = np.asarray(f['frames/lidar_valid_mask'][-1])
                _ = dict(f['config'].attrs)['Hs']
            sizes.append(os.path.getsize(e['path']))
        except Exception as exc:                                # noqa: BLE001
            broken.append((os.path.basename(e['path']), repr(exc)[:90]))
    chk('★B-2 마지막 프레임까지 실제 판독 (부분 파일 탐지)', not broken, f'손상 {broken}')
    if sizes:
        lo, hi = min(sizes), max(sizes)
        chk('★B-3 파일 크기 산포 <= 5% (잘린 파일 = 작다)', hi and (hi - lo) / hi <= 0.05,
            f'{lo}–{hi} bytes ({(hi-lo)/hi*100:.2f}%)')
    chk('B-4 프레임 수 단일값', len(nfr) <= 1, str(sorted(nfr)))
    return have


# ── C. 매니페스트 무결성 (⛔러너가 `os.path.exists` 로 skip 하므로 파일만 봐선 부족) ──
def c_manifest(have):
    man = G._load_manifest()
    eps, _ = G.select()
    want = {f"{e['gt_arm']}/{G.fname(e['ncomp'], e['seed'])}" for e in eps}
    got = set(man)
    okrec = {k for k, v in man.items() if v.get('status') == 'ok'}
    chk('C-1 매니페스트 키 == 기대 48', got == want,
        f'{len(got)}/{len(want)} · 누락 {sorted(want-got)[:4]} · 잉여 {sorted(got-want)[:4]}')
    chk('★C-2 status==ok 48 (⛔파일은 있는데 항목이 없으면 = 중단된 쓰기)',
        len(okrec) == 2 * G.N_EP, f'{len(okrec)}/{2*G.N_EP} · '
        f'error {[k for k, v in man.items() if v.get("status") != "ok"]}')
    onfile = {os.path.relpath(e['path'], G.DATA).replace(os.sep, '/') for e in have}
    chk('★C-3 디스크 파일 ⊆ ok 매니페스트 (부분 파일 = 여기서 갈린다)',
        onfile <= okrec, f'항목 없는 파일 {sorted(onfile - okrec)}')
    return man


# ── D. 동결 게이트 그대로 (문턱·논리는 gates_m1 이 유일하게 정의한다) ───────────
def d_gates(man):
    print('\n── gates_runfree (런 0회) ─────────────────────────────────────')
    for n, ok, msg in G.gates_runfree():
        chk(n, ok, msg)
    print('\n── gates_data (생성 후 · 추정 전) ─────────────────────────────')
    for n, ok, msg in G.gates_data(manifest=man):
        chk(n, ok, msg)
    p = os.path.join(HERE, 'pc1_m1.json')
    print('\n── gates_pc1 ──────────────────────────────────────────────────')
    for n, ok, msg in G.gates_pc1(json.load(open(p)) if os.path.exists(p) else None):
        chk(n, ok, msg)


# ── E. mesh_verdict 3분할 (PREREG §1.3 의 표본 서술 — ⛔"HOS M=3" 단일 서술 금지) ──
def e_mesh(man):
    from collections import Counter
    from wavesim import mesh_adequacy
    eps, _ = G.select()
    lin = Counter(mesh_adequacy.check(G.components(e['ncomp'], e['seed']),
                                      G.cfg_m1(False), hos_order=1)['mesh_verdict']
                  for e in eps if e['gt_arm'] == 'LIN')
    hos = Counter(e['mesh_verdict'] for e in eps if e['gt_arm'] == 'HOS')   # screen = M=3
    snap = Counter(v.get('mesh_verdict_snapped') for v in man.values()
                   if v.get('status') == 'ok')
    attr = Counter((v.get('gt_arm'), v.get('attr_mesh_verdict')) for v in man.values()
                   if v.get('status') == 'ok')
    print(f'\n★표본 서술(§1.3, 재도출): LIN(M=1 채점) {dict(lin)} · HOS(M=3 채점) {dict(hos)}')
    print(f'★스냅 후 판정(M=3) {dict(snap)} · 저장 attrs {dict(attr)}')
    chk('E-1 3분할이 산출됐다 (⛔보고 시 이 표를 그대로 옮길 것)', bool(lin and hos), '')


# ── F. 추정 전 위생 (스모크 잔재 · 결과 파일) ───────────────────────────────────
def f_hygiene():
    import glob
    sm = glob.glob(os.path.join(HERE, 'smoke*')) + \
        ([G.SMOKE] if os.path.exists(G.SMOKE) else [])
    chk('F-1 스모크 잔재 0 (§2.6: 삭제 후 본 배치가 seed 1681 을 재생성한다)',
        not sm, str([os.path.basename(x) for x in sm]))
    res = sorted(glob.glob(os.path.join(HERE, 'results_m1_*.jsonl')))
    print(f'[INFO] 기존 결과 샤드 {len(res)} 개 (있으면 러너가 이어서 돈다: '
          f'{[os.path.basename(x) for x in res]})')


def main():
    print(f'M1 코퍼스 검증 — {os.path.basename(HERE)}\n' + '=' * 66)
    a_frozen()
    have = b_files()
    man = c_manifest(have)
    d_gates(man)
    e_mesh(man)
    f_hygiene()
    print('=' * 66)
    print(f'⛔FAIL {len(FAIL)} · ⚠WARN {len(WARN)}')
    if FAIL:
        print('FAIL:', FAIL)
        print('⟹ ⛔6단계(추정)로 가지 마라.')
    else:
        print('✅6단계(추정) 진행 가능.')
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
