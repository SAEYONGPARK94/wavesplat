"""S9 실행 전 게이트 (PREREG §7).

⛔하나라도 FAIL 이면 `run_s9.py` 를 **돌리지 않는다.** 결과를 본 뒤 완화하지 않는다.

G-0 채점기 자기시험 (S8 데이터로 S8 판정 재현)   ← S8 이 3.4h 잃은 결함의 0원 처방
G-1 독립성 5축                                  ← S1 게이트를 **import** (복사 금지)
G-2 메시 ADEQUATE 96/96
G-3 손잡이 3종 ON 96/96
G-4 시드 충돌 0                                 ← ⚠목록을 적지 않고 **스캔해서 유도**
G-5 코드 출처 md5 기록
G-6 셀 cfg 유일성 + 실측 파라미터
G-7 ★제어 축 결백 (S2_ALPHA 미설정)              ← 사용자 지시

★전 상수는 `s9_constants.py` 가 유도한다. ⛔이 파일에 n·시드·문턱을 적지 않는다.
이 스크립트는 S9 결과 파일을 읽지 않는다(존재하지 않는다).
"""
import glob
import hashlib
import json
import os
import subprocess
import sys

import s9_constants as C

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
S1DIR = os.path.abspath(os.path.join(HERE, '..', 'S1_indep_sparse_2026-08-08'))
K6 = os.path.abspath(os.path.join(HERE, '..', 'K6_run_2026-08-07'))
for p in (K6, S1DIR):
    sys.path.insert(0, p)

import gate as S1GATE                      # noqa: E402  복사 금지, import
from independence_gate import report       # noqa: E402  frozen 독립성 규약

import run_s9 as RUN                       # noqa: E402  에피소드·셀의 단일 출처

CODE_FILES = ['wavesplat/run.py', 'wavesplat/evaluation/metrics.py',
              'wavesplat/optimizer/initializer.py', 'wavesplat/optimizer/sequential_update.py',
              'wavesim/wave_generator.py', 'wavesim/pipeline.py', 'wavesim/config.py',
              'wavesim/data_writer.py', 'generate_dataset.py', 'configs/estimation.yaml']


def basenames(rg):
    return [os.path.basename(e) for e in RUN.episodes_for(rg)]


def g0_selftest():
    """★채점기가 **알려진 정답**(S8)을 재현하는가."""
    print('\n### G-0 채점기 자기시험 (S8 데이터로 S8 판정 재현)')
    f = os.path.join(HERE, 'G0_selftest_on_S8.txt')
    if not os.path.exists(f):
        print('  ⛔ G0_selftest_on_S8.txt 없음 — analyze_s9.py --s8map 을 먼저 돌려라')
        return False
    body = open(f).read()
    checks = [
        ('희소 22/32 미달 재현', '개선 22/32' in body and '미달' in body),
        ('혼잡 27/32 통과 재현', '개선 27/32' in body and '✅통과' in body),
        ('혼잡 Δ중앙 -0.02650 재현', '-0.02650' in body),
        ('희소 원p 0.05010 재현', '0.05010' in body),
        ('all_resid 혼잡 -9.94% 재현', '-9.94%' in body),
        ('워커 누출 0 재현', '비트 불일치 0개' in body),
        ('lr all_resid 혼잡 4/32 +7.55% 재현', '+7.55%' in body),
    ]
    ok = True
    for name, passed in checks:
        print(f'  {"✅" if passed else "⛔"} {name}')
        ok &= passed
    return ok


def g4_collisions():
    """⚠S8 AMENDMENT_2 처방: 상수를 적지 말고 **유도**한다. 기존 시드는 스캔해서 얻는다."""
    print('\n### G-4 시드 충돌 0 (기존 시드를 **스캔해서 유도** — 목록을 적지 않는다)')
    import re
    new_names = set(basenames('sparse')) | set(basenames('crowded'))
    mine = set(C.DATA_DIR.values())
    existing_seeds, bad = set(), []
    for d in glob.glob(os.path.join(REPO, 'data*')) + glob.glob(os.path.join(REPO, 'tmp', '*')):
        if not os.path.isdir(d) or os.path.basename(d) in mine:
            continue
        for p in glob.glob(os.path.join(d, '**', '*.h5'), recursive=True):
            m = re.search(r'seed(\d+)', os.path.basename(p))
            if m:
                existing_seeds.add(int(m.group(1)))
            if os.path.basename(p) in new_names:
                bad.append(os.path.relpath(p, REPO))
    want = set(C.SEEDS['sparse']) | set(C.SEEDS['crowded'])
    overlap = want & existing_seeds
    got = {int(re.search(r'seed(\d+)', n).group(1)) for n in new_names}
    print(f'  기존 시드 (스캔): {sorted(existing_seeds)}')
    print(f'  이 배치 시드    : {sorted(want)}  (실측 파일명에서 {sorted(got)})')
    print(f'  시드 교집합     : {sorted(overlap) if overlap else "0  ✅"}')
    print(f'  파일명 교집합   : {bad[:3] if bad else "0  ✅"}')
    print(f'  시드 유도 일치  : {"✅" if got == want else f"⛔ {got ^ want}"}')
    return not overlap and not bad and got == want


def g5_code_provenance():
    print('\n### G-5 코드 출처 (미커밋 변경 위에서 실행 중임을 기록)')
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO,
                          capture_output=True, text=True).stdout.strip()
    diff = subprocess.run(['git', 'diff'] + CODE_FILES, cwd=REPO,
                          capture_output=True, text=True).stdout
    md5 = hashlib.md5(diff.encode()).hexdigest()
    print(f'  HEAD = {head[:12]} · 미커밋 diff md5 = {md5}')
    json.dump({'head': head, 'diff_md5': md5},
              open(os.path.join(HERE, 'G5_code_provenance.json'), 'w'), indent=1)
    return True                              # 기록 의무이지 차단 조건이 아니다


def g6_cells():
    print('\n### G-6 셀 cfg 유일성 + 실측 파라미터')
    seen, ok = {}, True
    for c in RUN.CELLS:
        extra = RUN.S5.S4.EXTRA.get(c, {})
        key = (json.dumps(extra, sort_keys=True),
               tuple(RUN.SEEDS_BY_CELL.get(c, RUN.SEEDS)))
        print(f'  {c:16s} lambda_l1={extra.get("lambda_l1", RUN.L1_SHIP):.0e} '
              f'tracking_lr={extra.get("tracking_lr", RUN.TRACK_LR):.5f} '
              f'seeds={RUN.SEEDS_BY_CELL.get(c, RUN.SEEDS)}')
        if key in seen and not (c.endswith('DUP') or seen[key].endswith('DUP')):
            print(f'    ⛔ {c} 가 {seen[key]} 와 완전히 같다 (DUP 아닌데 중복)')
            ok = False
        seen[key] = c
    return ok


def g7_control_axis():
    """★사용자 지시(2026-08-11): 제어 축 종료. 계산조차 하지 않는다.

    ⚠**원문 문자열 검색으로 하지 않는다.** 이 파일과 `run_s9.py` 는 금지된 이름들을
    주석·docstring 에서 **설명하려고** 언급한다 ⟹ 텍스트 검색은 자기 주석에 걸려 위양성을 낸다
    (실제로 초안이 그렇게 FAIL 했다). **AST 로 실행 코드만** 본다.
    docstring 은 `ast.get_docstring` 으로 떼어내고, 주석은 AST 에 애초에 없다.
    """
    import ast
    print('\n### G-7 ★제어 축 결백 (S2_ALPHA 미설정) — AST 기반, 주석/docstring 제외')
    path = os.path.join(HERE, 'run_s9.py')
    tree = ast.parse(open(path).read())

    def strings_in(node, skip_docstring=True):
        """실행 코드 안의 문자열 상수 + 속성/이름 식별자."""
        out = []
        # ⚠`clean=True`(기본)는 들여쓰기를 정규화해서 **원문과 달라진다** ⟹ 비교가 항상 실패하고
        # docstring 이 걸러지지 않는다. 반드시 `clean=False` 로 원문을 받는다.
        doc = ast.get_docstring(node, clean=False) if skip_docstring and isinstance(
            node, (ast.FunctionDef, ast.Module, ast.ClassDef)) else None
        for n in ast.walk(node):
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                if doc is not None and n.value == doc:
                    continue
                out.append(n.value)
            elif isinstance(n, ast.Attribute):
                out.append(n.attr)
            elif isinstance(n, ast.Name):
                out.append(n.id)
        return out

    # 1) os.environ['S2_ALPHA'] = ... 형태의 **대입**이 있는가
    sets_alpha = False
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant) \
                        and t.slice.value == 'S2_ALPHA':
                    sets_alpha = True
    # 2) R7.run_one / run_s7.run_one 을 **호출**하는가
    calls_s7_runone = any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == 'run_one'
        and isinstance(n.func.value, ast.Name) and n.func.value.id in ('R7', 'run_s7')
        for n in ast.walk(tree))
    # 3) _extra_keys_s9 의 **실행 코드**에 alpha 가 등장하는가
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef)
               and n.name == '_extra_keys_s9'), None)
    alpha_in_fn = any('alpha' in s.lower() for s in strings_in(fn)) if fn else True
    # 4) BANNED_ENV 를 실제로 pop 하는가
    pops = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == 'pop' for n in ast.walk(tree))

    checks = [
        ('run_s9 이 S2_ALPHA 를 설정하지 않음', not sets_alpha),
        ('run_s9 이 BANNED_ENV 를 매 잡마다 제거', pops and 'BANNED_ENV' in strings_in(tree)),
        ('run_s7.run_one 을 호출하지 않음 (그것이 S2_ALPHA 를 켠다)', not calls_s7_runone),
        ('_extra_keys_s9 실행 코드에 alpha 없음', not alpha_in_fn),
        ('환경에 S2_ALPHA 미설정', os.environ.get('S2_ALPHA') is None),
    ]
    ok = True
    for name, passed in checks:
        print(f'  {"✅" if passed else "⛔"} {name}')
        ok &= passed
    print('  ⚠런타임 증거는 채점기 §0 이 본다: 레코드의 `s2_alpha_env` 와 `alpha_*` 키 개수.')
    return ok


def main():
    print('S9 실행 전 게이트 (PREREG §7)')
    print(f'  n: {C.N} · 시드: '
          f'{ {k: f"{min(v)}-{max(v)}" for k, v in C.SEEDS.items()} }')
    print(f'  PRIMARY 문턱: ' + ', '.join(
        f'{rg} {C.primary_k(C.N[rg])}/{C.N[rg]}' for rg in ('sparse', 'crowded')))

    missing = [e for e in RUN.EPISODES if not os.path.exists(os.path.join(REPO, e))]
    if missing:
        print(f'\n⛔ 에피소드 결측 {len(missing)}개 — 생성이 끝나지 않았다. 예: {missing[:2]}')
        sys.exit(2)

    ok = True
    summary = {}
    for rg in ('crowded', 'sparse'):
        eps = basenames(rg)
        label = f'신규 {rg}{len(eps)} (시드 {min(C.SEEDS[rg])}-{max(C.SEEDS[rg])})'
        r = report(C.DATA_DIR[rg], eps, label)
        if r is None:
            print(f'  ⛔ {label}: 파일 결측')
            ok = False
            continue
        _axes, lo, floor = r
        summary[label] = (lo, floor)
        ok &= (lo == len(eps))
        ok &= S1GATE.block_report(C.DATA_DIR[rg], eps, label)
        ok &= S1GATE.mesh_and_provenance(C.DATA_DIR[rg], eps, label)

    ok &= g0_selftest()
    ok &= g4_collisions()
    g5_code_provenance()
    ok &= g6_cells()
    ok &= g7_control_axis()

    print('\n' + '=' * 70)
    for k, (lo, floor) in summary.items():
        print(f'  {k}: 가장 좁은 축 {lo}종, 바닥 p = {floor:.2e}')
    print('  ⚠보증 = 길이 블록당 (블록 간 독립성은 길이가 달라 측정 불가) — S1 F8 승계')
    print(f'GATE: {"PASS ✅ — S9 실행 허가" if ok else "FAIL ⛔ — S9 를 돌리지 않는다"}')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
