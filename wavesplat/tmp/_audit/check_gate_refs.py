#!/usr/bin/env python3
"""게이트 **외부 참조 키** 검사기 — *"공허한 FAIL/PASS 를 사전등록 시점에 잡는다"* (2026-08-28).

★**결함 부류 (M2 `G-13` 이 노출시켰다)**

    게이트가 참조하는 키가 **실제 외부 동결 자료에 없어서**, 그 게이트는 자료에 대고는
    처음부터 통과(또는 평가)가 **불가능**하다. 자가진단은 **픽스처가 그 키를 갖고 있어서** 통과한다.

실측 사례 (`tmp/M2_LOCTUNE_2026-08-28/gates_m2.py:536-553`):
    `G-13`(단일요인)이 `B-26` 동결 행에서 `cap_effective_config` / `effective_config` 를 찾아
    참조 `ref` 로 쓴다. 그런데 `B-26` 의 `SPEC25` 동결 행 288개는 **그 키를 하나도 안 싣는다**
    (`cap_loc_radius`·`cap_depth` 처럼 **평탄화된** `cap_*` 키만 있다) ⟹ `ref = {}` ⟹
    `bool(ref) and not diff` 가 **영원히 False**. ★자료에 대고는 통과 불가능한 게이트였다.

★**핵심 통찰 — 이것은 사전등록 시점에 잡을 수 있었다.**
    참조는 **외부 동결 파일**에서 온다. 그 파일은 사전등록 시점에 **이미 존재**한다.
    ⟹ *"외부 동결 자료를 읽는 게이트는, 사전등록 시점에 그 실제 파일에 대고 한 번 돌려
       참조가 비지 않음을 확인해야 한다."*
    이 검사기가 그 확인을 기계화한다. **런 0회 · 읽기 전용 · GPU 0회.**

★**두 방향, 그리고 어느 쪽이 더 나쁜가**
  · **공허한 FAIL** — 없는 키가 조건을 False 로 만든다. 아프지만 **탐지된다**(M2 가 그랬다).
  · **공허한 PASS** — 없는 키가 필터를 비워 `not bad` 를 참으로 만든다. ⟹ **구조적으로 탐지 불가.**
    ★이쪽이 더 나쁘다. 이 검사기는 **양쪽을 같은 방식으로 잡는다**(키의 부재를 보지, 결과를 안 본다).
  ⛔**단 이 검사기는 어느 쪽인지 판정하지 않는다** — 그것은 사람이 그 줄을 읽어야 한다.
    (조건식 모양으로 자동 분류하려다 규칙이 두 줄 만에 틀리는 것을 확인했다. 저장소 규율:
     *"기계가 확신할 수 없는 것을 확신한 척하지 마라."*)

────────────────────────────────────────────────────────────────────────────
검사 절차 (배치 하나당)

  0. 검사 대상 = `gates_*.py` **+ `run_*.py` · `analyze_*.py`**(= 저장소가 이미
     `check_batch_artifacts._judge_files()` 에서 「판정을 낳는 코드」로 정의한 것과 같은 집합).
     ★게이트만 보면 **외부 동결 행을 러너가 읽는 배치**를 통째로 놓친다 — 소급 적용 중
     `M3`(`run_m3.frozen_b26_rows`)에서 실제로 그랬다. ⛔러너는 **임포트하지 않는다**
     (`run_m3.py:40` 이 최상위에서 `os.chdir` 을 한다) — 정적으로만 판다.
  1. `gates_*.py` 만 임포트한다(그 파일 자신이 `sys.path` 를 세우므로 배치 디렉터리만 얹는다).
  2. **외부 결과 파일 발견** — 네 경로를 전부 쓴다(하나만 쓰면 조용히 놓친다):
     (a) 모듈 전역에 들어 있는 경로 문자열(중첩 dict/list/tuple 안까지 3단계 하강).
         → `gates_b21.FROZEN_SRC` 처럼 **dict 값 안에 숨은 glob** 을 잡는다.
     (b) 소스의 **모든 식**을 작은 평가기로 정적 평가한다(`os.path.join/dirname/abspath`,
         모듈 상수, `<별칭>.<상수>`, 그리고 **함수 지역 경로 변수**).
         → `gates_b25.py:156` 인라인 glob · `analyze_b8.py:9` 모듈 최상위 컴프리헨션 ·
           `analyze_s8.py:528` 지역 변수 조립 · `run_m3.py` 의 `G.B26_RESULTS_GLOB` 을 잡는다.
     (c) **무인자 로더를 실제로 호출**하고 `open`/`glob.glob` 을 계측해 **연 파일을 기록**한다.
         → 경로 조립이 아무리 복잡해도 잡는다. ★정적 분석의 취약함을 여기서 메운다.
     (d) 0개로 펼쳐진 외부 패턴도 기록한다 → `⚠대조0`.
     `<외부>` = **배치 디렉터리 밖**. 배치 자신의 `results*.jsonl` 은 사전등록 시점에
     존재하지 않으므로 **일부러 제외**한다(그건 다른 결함 부류다 — 아래 「보지 않는 것」).
  3. **원천별로** 그 파일들의 모든 행의 키 합집합 = `외부 키 우주`.
     ⚠합집합 하나로 뭉개면 안 된다 — 한 함수가 외부 파일과 자기 파일을 **둘 다** 읽는
     경우가 있다(`analyze_s13b.load()`). 원천(파일)을 따라다녀야 오탐이 안 난다.
  4. **정적 추적** — *외부 자료에서 온 행 변수* 에 대한 키 리터럴 **읽기**
     (`r.get('K')` · `r['K']` · `'K' in r`) 만 모은다. 자세한 규칙은 `_trace()` 독스트링.
  5. `누락 = 접근한 키 − 그 원천의 키 우주` ⟹ ⛔**공허 참조**로 보고한다.
     ⛔**값 접근**(항상 `None`)과 ⚠**존재 검사만**(필터가 no-op)을 **등급으로 나눈다** —
     `'error' not in r` 는 이 저장소의 표준 방어 관용구라 부재가 정상이고, 그것을 ⛔로
     찍으면 *"경고가 항상 켜져 있어 신호가 죽는"* 결함이 된다. 종료코드는 ⛔에만 반응한다.

⚠**0건 대조는 `OK` 가 아니다** — 외부를 읽는 함수를 찾았는데 파일을 하나도 못 열었으면
  `⚠대조0` 경고를 낸다(`check_batch_artifacts.py` 가 M-5 에서 배운 것과 같은 규율).

────────────────────────────────────────────────────────────────────────────
★**보지 않는 것**(= 이 검사기로 *"검사했다"* 고 말할 수 없는 범위)
  · **배치 자신의 `results*.jsonl`** — 사전등록 시점에 없다. 게다가 팔마다 키가 달라
    (`SHIP` 행에는 `loc_radius` 가 없다) 합집합 대조가 오탐을 쏟는다.
  · **팔 단위 부재** — 키가 *어떤* 행에는 있고 필터가 고르는 행에는 없는 경우.
    합집합 대조는 이것을 놓친다(**안전한 방향**: 침묵이지 오탐이 아니다).
  · 다른 모듈(`gates_b11.med_by_episode` 등)로 넘어간 뒤의 접근. 한 모듈 안에서만 추적한다.
    ⚠**로더 *발견* 은 2026-08-29 부터 모듈을 넘어간다**(F-2 수리, `_reads_data` 폐포).
    넘어가지 못하는 것은 4단계의 **키 추적**뿐이다 — 둘을 섞어 읽지 마라.
  · 키가 **변수**인 접근(`r.get(k)`). 리터럴만 본다.
  · ★**픽스처 제조기**(`_fixture*`·`synthetic_rows`)가 여는 파일. **일부러 안 본다** —
    반환 행이 합성물이라 그 키 집합을 「외부 키 우주」로 삼으면 오탐이 쏟아진다(실측 17건).
    ⟹ 어떤 배치의 **유일한** 외부 읽기가 픽스처 함수 안에 있으면 이 검사기는 침묵한다.
  · **호출 그래프 폐포의 한계** — 깊이 `CLOSURE_DEPTH`(=3)까지 · `fn.__globals__` 로 이름이
    풀리는 호출만(동적 디스패치 `getattr(mod, name)()`·함수 표는 못 따라간다) ·
    stdlib/site-packages 안으로는 들어가지 않는다.
  · **인자가 필요한 로더**(`load(path)`)는 호출하지 않는다 — 무인자 함수만 부른다.
  · `⚠원천미식별` 로 찍힌 배치 — 외부 경로는 찾았는데 그것을 행으로 읽는 지점을 못 짚었다.
    ★`OK` 가 아니다(M3 가 그 상태였다). 실측 잔여 4건은 손으로 확인했다:
    설정 문서(`corpus_router_targets.json`) · `sys.path` 디렉터리 · 저장소 전체 `**` 센서스.

★**소급 적용 결과 (2026-08-28, tmp/ 전체 100배치)**
  ⛔값 접근 공허 참조 = **1건뿐**, `M2` `G-13`(이 검사기를 만들게 한 그 결함).
  ⟹ 같은 부류가 다른 배치의 **판정을 오염시킨 사례는 발견되지 않았다**(발견 못 함 ≠ 없음:
     위 「보지 않는 것」 이 범위다). ⚠존재검사 등급은 전부 `'error'` 방어 관용구였다.
  ★소급 적용이 **이 검사기 자신의 오탐 5부류**를 깎았다(전부 회귀 시험으로 고정):
    ①`readers` 전이 폐포가 `aggregate()` 까지 삼킴 ②`<module>` 스코프가 함수 본문을 walk
    ③`r['K'] = v` 쓰기를 읽기로 셈 ④`json.load` 문서를 행으로 오인 ⑤한 함수가 읽는
    **두 파일**을 한 우주로 뭉갬. ⟹ ★*"검사기도 자기 자신에게 소급 적용해야 한다."*

사용:
    python3 tmp/_audit/check_gate_refs.py                 # tmp/ 전체
    python3 tmp/_audit/check_gate_refs.py tmp/M2_LOCTUNE_2026-08-28
    python3 tmp/_audit/check_gate_refs.py --selftest      # 자가진단(일부러 깨진 픽스처)
    python3 tmp/_audit/check_gate_refs.py -v tmp/...      # 로더·파일·키 우주까지 출력
종료코드 1 = 공허 참조가 하나라도 있음.
"""
import ast
import builtins
import glob as globmod
import importlib.util
import inspect
import io
import json
import os
import re
import signal
import sys
import sysconfig
import textwrap
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = os.path.dirname(HERE)
ROOT = os.path.dirname(TMP)

CALL_TIMEOUT_S = 20          # 무인자 로더 하나당 벽시계 상한
MAX_ROWS_PER_FILE = 200000   # 안전판
CONTAINER_DEPTH = 3          # 전역 컨테이너 하강 깊이

# 호출해도 되는 무인자 함수에서 **제외**할 이름 접두어.
# ⛔`gates*`/`selftest`/`main` 은 게이트 전체를 돌리므로 부작용·비용이 크다.
# ⛔**픽스처 제조기는 로더가 아니다 (2026-08-29, F-2 수리에 딸린 필수 항목).**
#   `_fixture*`·`synthetic_rows` 는 **합성 행을 만든다**(`gates_m8.py:1241` 이 스스로
#   *"⛔판정 자료가 아니다"* 라 적었다). 앵커 값을 싣느라 외부 동결 파일을 **읽기는 하지만**
#   반환하는 행은 제조물이므로, 그 키 집합을 「외부 키 우주」로 삼으면 **없는 키가 아닌 것을
#   없다고 보고**한다. F-2 폐포를 넣자 이 부류가 후보로 들어와 오탐 **17건**(B19·B21·B22·M8)이
#   났고, 전건이 배치 자신의 실제 결과 행에는 **존재하는 키**였다(실측 확인).
#   ★같은 이유로 `selftest` 는 처음부터 이 목록에 있었다 — `_fixture` 는 그 짝이다.
#   실측 근거: 저장소의 픽스처 제조기 이름은 `_fixture*`(7종)·`synthetic_rows` 뿐이다.
NO_CALL = ('gates', 'selftest', '_selftest', 'main', 'report', 'run', 'label', 'print',
           'fixture', '_fixture', 'synthetic')

# 행 변수에서 파생된 컬렉션으로 인정하는 호출(순서/형만 바꾸는 것들)
PASSTHRU = {'sorted', 'list', 'tuple', 'reversed', 'filter', 'set'}


# ═══════════════════════════════════════════════════════════════════════════
# 0. 유틸
# ═══════════════════════════════════════════════════════════════════════════
class _Timeout(Exception):
    pass


def _alarm(_s, _f):
    raise _Timeout()


def _is_external(path, batchdir):
    """`path` 가 **배치 디렉터리 밖**인가 (그리고 실재하는가)."""
    try:
        p = os.path.realpath(path)
    except Exception:
        return False
    b = os.path.realpath(batchdir)
    return not (p == b or p.startswith(b + os.sep))


def _looks_like_data(path):
    return path.endswith('.jsonl') or path.endswith('.json')


def _norm(p):
    """★경로 철자를 통일한다 — `..` 를 남긴 glob 결과와 `abspath` 결과가 **같은 파일의
    두 이름**으로 세어져 `외부파일 1` 이 `2` 로 찍혔다(자가진단 [1] 이 잡았다)."""
    try:
        return os.path.realpath(p)
    except Exception:
        return p


def _expand(pattern):
    """glob 이면 펼치고, 아니면 존재할 때만 돌려준다. 경로는 전부 정규화한다."""
    if any(c in pattern for c in '*?['):
        return [_norm(p) for p in sorted(globmod.glob(pattern))]
    return [_norm(pattern)] if os.path.isfile(pattern) else []


def _row_keys(path):
    """파일 하나에서 **행 dict 들의 키 합집합**과 행 수.

    `.jsonl` = 줄마다 한 행. `.json` = 컨테이너를 하강해 dict 를 모은다
    (동결 md5 표처럼 행이 아닌 것도 들어오지만 **키 우주를 넓히는 방향** = 안전).
    """
    keys, n = set(), 0
    try:
        if path.endswith('.jsonl'):
            with open(path, encoding='utf-8', errors='replace') as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(r, dict):
                        keys |= set(r)
                        n += 1
                    if n >= MAX_ROWS_PER_FILE:
                        break
        else:
            with open(path, encoding='utf-8', errors='replace') as fh:
                obj = json.load(fh)
            stack, depth = [(obj, 0)], 0
            while stack:
                o, depth = stack.pop()
                if depth > CONTAINER_DEPTH + 2:
                    continue
                if isinstance(o, dict):
                    keys |= set(o)
                    n += 1
                    stack += [(v, depth + 1) for v in o.values()]
                elif isinstance(o, (list, tuple)):
                    stack += [(v, depth + 1) for v in o]
    except Exception:
        return set(), 0
    return keys, n


# ═══════════════════════════════════════════════════════════════════════════
# 1. 모듈 임포트 (배치 디렉터리를 sys.path 에 얹고)
# ═══════════════════════════════════════════════════════════════════════════
def _import(path):
    d = os.path.dirname(os.path.abspath(path))
    for p in (d, ROOT, TMP):
        if p not in sys.path:
            sys.path.insert(0, p)
    name = '_gref_' + os.path.basename(path)[:-3]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ═══════════════════════════════════════════════════════════════════════════
# 2. 외부 결과 파일 발견 (세 경로)
# ═══════════════════════════════════════════════════════════════════════════
def _globals_paths(consts, batchdir):
    """(a) 모듈 전역(중첩 컨테이너 포함)에 든 **외부 경로 문자열**.

    → `(전역이름 -> set(실재 파일), set(0개로 펼쳐진 외부 패턴))`
    ⚠**0개로 펼쳐진 패턴도 반환한다** — 그것이 `⚠대조0` 의 근거다. 종전 판은 이것을 버려서
      *"파일이 없으면 검사기가 조용해진다"* 는 자기 결함이 있었다(자가진단 [3] 이 잡았다).
    """
    found, empty = {}, set()
    for gname, gval in list(consts.items()):
        if gname.startswith('__'):
            continue
        files, stack = set(), [(gval, 0)]
        seen_ext = False
        while stack:
            v, dep = stack.pop()
            if dep > CONTAINER_DEPTH:
                continue
            if isinstance(v, str):
                if _looks_like_data(v) and os.path.isabs(v) and _is_external(v, batchdir):
                    seen_ext = True
                    got = set(_expand(v))
                    files |= got
                    if not got:
                        empty.add(v)
            elif isinstance(v, dict):
                stack += [(x, dep + 1) for x in list(v.keys()) + list(v.values())]
            elif isinstance(v, (list, tuple, set)):
                stack += [(x, dep + 1) for x in v]
        if seen_ext:
            found[gname] = files
    return found, empty


def _extdirs(consts, batchdir):
    """전역 중 **배치 밖 디렉터리** 를 가리키는 것 (`B22 = .../B22_.../`)."""
    out = {}
    for gname, gval in consts.items():
        if (isinstance(gval, str) and os.path.isabs(gval) and os.path.isdir(gval)
                and _is_external(gval, batchdir) and not gname.startswith('__')):
            out[gname] = gval
    return out


def _static_joins(tree, consts, batchdir):
    """(b) 소스의 `os.path.join(<전역>, '<리터럴>' …)` 을 안전하게 평가한다.

    ⚠`gates_b25.py:156` 의 `glob.glob(os.path.join(B22, 'results_b22*.jsonl'))` 처럼
      **전역에 묶이지 않은 인라인 경로** 는 (a) 로는 절대 안 잡힌다.
    """
    files, empty = set(), set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'join'):
            continue
        parts = []
        for a in node.args:
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                parts.append(a.value)
            elif isinstance(a, ast.Name) and isinstance(consts.get(a.id), str):
                parts.append(consts[a.id])
            elif (isinstance(a, ast.Attribute) and isinstance(a.value, ast.Name)
                  and isinstance(consts.get(f'{a.value.id}.{a.attr}'), str)):
                parts.append(consts[f'{a.value.id}.{a.attr}'])
            else:
                parts = None
                break
        if not parts:
            continue
        p = os.path.join(*parts)
        if os.path.isabs(p) and _looks_like_data(p) and _is_external(p, batchdir):
            got = set(_expand(p))
            files |= got
            if not got:
                empty.add(p)
    return files, empty


# ★F-2 수리 (2026-08-29) — 「자료를 읽는가」의 증거 패턴.
#   ⛔종전 정규식은 `glob\.glob\s*\(` 로 **수신자 이름까지** 박아 놨다 ⟹ `_g.glob(` 를 놓쳤다.
#   수신자를 풀어 `\bglob\s*\(` 로 바꾼다(`glob.glob(`·`_g.glob(`·`Path.glob(` 전부 매치).
#   ⛔`open(` 은 **일부러 넣지 않는다** — 쓰기(`open(p,'w')`) 함수까지 호출 대상이 되어
#     검사기가 자료를 **덮어쓸** 수 있다. 읽기 증거만 쓴다.
_READ_SRC = re.compile(r'json\.loads?\s*\(|\.jsonl|\bglob\s*\(')
CLOSURE_DEPTH = 3            # 호출 그래프를 따라가는 깊이 상한


# stdlib·site-packages 경로 — 폐포가 여기로 넘어가면 안 된다(`json.load` 안까지 파는 짓).
_LIBDIRS = tuple(sorted({os.path.realpath(q) + os.sep
                         for q in (sysconfig.get_paths().get(k) for k in
                                   ('stdlib', 'platstdlib', 'purelib', 'platlib'))
                         if q}))


def _fn_src(fn):
    """**제1자 코드**(저장소·배치)에 정의된 함수의 소스. stdlib·site-packages 면 `None`.

    ⛔경계를 `ROOT` 로 잡으면 안 된다 — 그러면 **자가진단 픽스처(tempdir)가 폐포를 못 타서**
      회귀 시험이 조용히 무력해진다(2026-08-29 에 실제로 그렇게 만들었다가 잡았다).
      막아야 하는 것은 *"저장소 밖"* 이 아니라 *"제3자 라이브러리"* 다.
    """
    try:
        f = inspect.getsourcefile(fn)
        if not f:
            return None
        r = os.path.realpath(f)
        if any(r.startswith(q) for q in _LIBDIRS):
            return None
        return inspect.getsource(fn)
    except Exception:
        return None


def _reads_data(fn, depth=0, seen=None):
    """`fn` **또는 그것이 부르는** 저장소-내 함수가 자료를 읽는가 (호출 그래프 폐포).

    ⛔**F-2 = 이 검사기의 사각지대였다 (2026-08-29 수리).**
      종전 후보 선정은 함수 **자기 본문의 문자 매칭**뿐이었다
      (`json.load(` · `.jsonl` · `glob.glob(`). ⟹ jsonl 읽기를 **공유 헬퍼로
      단일출처화한 배치가 통째로 안 보였다**: `gates_m8b.source_rows()` 는
      `G8._load_jsonl(p)` 와 별칭 `_g.glob(...)` 을 쓰므로 셋 중 어느 것도 매치하지 않아
      **후보에 들지도 못했고**, 그 배치는 `⚠원천미식별` 로 조용히 빠졌다.
      ★***저장소가 권장하는 규율(단일출처)과 탐지 규칙이 정면으로 어긋나 있었다*** —
      **잘 짤수록 안 보이는 검사기**는 방벽으로서 치명적이다.
    ⟹ 이름을 따라가지 말고 **호출 그래프**를 따라간다. 이름 해석은 `fn.__globals__`
      기준이라 `G8._load_jsonl` 처럼 **다른 모듈로 넘어간** 호출도 정확히 풀린다.
    ⚠**좁음은 유지한다** — 저장소 밖 함수(stdlib·numpy)는 따라가지 않고, 깊이는
      `CLOSURE_DEPTH` 로 막고, `NO_CALL` 접두어 배제는 그대로다. 그래서 후보가
      폭증하지 않는다(실측: 100배치에서 새 로더 **1개**, 벽시계 +0.4 s).
    """
    seen = set() if seen is None else seen
    if not callable(fn) or id(fn) in seen or depth > CLOSURE_DEPTH:
        return False
    seen.add(id(fn))
    src = _fn_src(fn)
    if src is None:
        return False
    if _READ_SRC.search(src):
        return True
    try:
        sub = ast.parse(textwrap.dedent(src))
    except Exception:
        return False
    g = getattr(fn, '__globals__', {})
    for n in ast.walk(sub):
        if not isinstance(n, ast.Call):
            continue
        if isinstance(n.func, ast.Name):
            tgt = g.get(n.func.id)
        elif isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name):
            base = g.get(n.func.value.id)
            tgt = getattr(base, n.func.attr, None) if base is not None else None
        else:
            continue
        if _reads_data(tgt, depth + 1, seen):
            return True
    return False


def _call_loaders(mod, tree, batchdir, module_src='', verbose=False):
    """(c) **무인자 로더를 실제로 호출**하고 연 파일을 계측한다.

    반환 `{함수이름: {'files': set, 'rows': int, 'keys': set, 'err': str|None}}`
    ★정적 분석이 취약하다는 지적에 대한 답이다 — 경로 조립이 어떻든 `open`/`glob` 은 못 피한다.
    """
    cands = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        a = node.args
        need = len(a.args) - len(a.defaults)
        if need > 0 or a.posonlyargs or a.kwonlyargs and any(
                d is None for d in a.kw_defaults):
            continue
        if node.name.lower().startswith(NO_CALL):
            continue
        # ★호출 대상을 **좁게** 고른다. `ast.dump` 부분문자열 매칭은 `h5py.File` 을 읽는
        #   `mesh_census()` 같은 **무거운 함수**까지 끌어들였다(M1 이 GPU 를 쓰는 중이라
        #   무거운 CPU/디스크 작업은 금지다). `json`/`jsonl`/`glob` 을 실제로 쓰는 것만.
        # ⛔**F-2 수리**: 그 좁힘을 **자기 본문 문자 매칭**으로 했더니 읽기를 공유 헬퍼로
        #   단일출처화한 로더가 통째로 안 보였다 ⟹ `_reads_data` 의 **호출 그래프 폐포**로
        #   바꾼다(자기 본문 매칭은 그 안의 첫 조건으로 그대로 남는다).
        seg = ast.get_source_segment(module_src, node) or ''
        if not (_READ_SRC.search(seg) or _reads_data(getattr(mod, node.name, None))):
            continue
        cands.append(node.name)

    out = {}
    real_open, real_io_open, real_glob = builtins.open, io.open, globmod.glob
    for name in cands:
        fn = getattr(mod, name, None)
        if not callable(fn):
            continue
        opened, attempted = set(), set()

        def _rec(p):
            try:
                if isinstance(p, (str, bytes, os.PathLike)):
                    opened.add(_norm(os.fspath(p)))
            except Exception:
                pass

        def op(file, *a, **k):
            _rec(file)
            return real_open(file, *a, **k)

        def iop(file, *a, **k):
            _rec(file)
            return real_io_open(file, *a, **k)

        def gl(pat, *a, **k):
            res = real_glob(pat, *a, **k)
            for p in res:
                _rec(p)
            # ★패턴 자체도 기록한다 — 0개로 펼쳐지면 `opened` 가 비어 **로더가 아예 안 보인다**.
            #   자가진단 [3](외부 파일이 사라진 상황)이 정확히 그 침묵을 잡았다.
            try:
                if isinstance(pat, (str, bytes, os.PathLike)):
                    attempted.add(os.fspath(pat))
            except Exception:
                pass
            return res

        err, val = None, None
        builtins.open, io.open, globmod.glob = op, iop, gl
        # 모듈이 `import glob` 로 잡은 참조도 갈아끼운다
        patched = [m for m in (mod,) if getattr(m, 'glob', None) is globmod]
        try:
            signal.signal(signal.SIGALRM, _alarm)
            signal.alarm(CALL_TIMEOUT_S)
            val = fn()
        except BaseException as e:                       # noqa: BLE001 — 무엇이든 삼킨다
            err = f'{type(e).__name__}: {e}'
        finally:
            signal.alarm(0)
            builtins.open, io.open, globmod.glob = real_open, real_io_open, real_glob
            del patched
        ext = {p for p in opened if _is_external(p, batchdir) and _looks_like_data(p)}
        # 외부를 **겨냥했으나 0개로 펼쳐진** 패턴 (⚠대조0 의 근거)
        ext_empty = {q for q in attempted
                     if _looks_like_data(q) and _is_external(q, batchdir) and not _expand(q)}
        if not ext and not ext_empty:
            continue
        keys, nrow = set(), 0
        stack = [(val, 0)]
        while stack:
            o, dep = stack.pop()
            if dep > CONTAINER_DEPTH + 2:
                continue
            if isinstance(o, dict):
                keys |= {k for k in o if isinstance(k, str)}
                nrow += 1
                stack += [(v, dep + 1) for v in o.values()]
            elif isinstance(o, (list, tuple, set)):
                stack += [(v, dep + 1) for v in o]
        out[name] = dict(files=ext, empty=ext_empty, rows=nrow, keys=keys, err=err)
        if verbose:
            print(f'      · 호출 {name}() → 외부파일 {len(ext)} · 행 {nrow}'
                  + (f' · ⚠0개로 펼쳐진 외부 패턴 {len(ext_empty)}' if ext_empty else '')
                  + (f' · {err}' if err else ''))
    return out


def _static_consts(tree, filepath, alias_consts=None):
    """★**임포트하지 않고** 모듈 최상위의 경로 상수를 평가한다.

    ⛔`run_*.py` 를 임포트하면 안 된다 — `run_m3.py:40` 은 모듈 최상위에서 `os.chdir(ROOT)`
      를 하고, 러너들은 무거운 의존을 끌어온다(그리고 지금 M1 이 GPU 를 쓰고 있다).
      그런데 **외부 동결 행을 읽는 코드가 러너에 있는 배치가 실제로 있다**
      (`run_m3.frozen_b26_rows()` 가 `G.B26_RESULTS_GLOB` 를 읽는다) ⟹ 정적으로 판다.

    지원: 문자열 상수 · `__file__` · `os.path.join/dirname/abspath/realpath` · 이미 평가된 이름 ·
          `<별칭>.<이름>`(같은 배치의 다른 모듈에서 온 상수, `alias_consts` 로 주입).
    """
    env = dict(alias_consts or {})
    env['__file__'] = os.path.abspath(filepath)

    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            return n.value
        if isinstance(n, ast.Name):
            v = env.get(n.id)
            return v if isinstance(v, str) else None
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
            v = env.get(f'{n.value.id}.{n.attr}')
            return v if isinstance(v, str) else None
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            fn = n.func.attr
            args = [ev(a) for a in n.args]
            if any(a is None for a in args):
                return None
            try:
                if fn == 'join':
                    return os.path.join(*args)
                if fn == 'dirname':
                    return os.path.dirname(args[0])
                if fn in ('abspath', 'realpath', 'normpath'):
                    return getattr(os.path, fn)(args[0])
            except Exception:
                return None
        return None

    for _ in range(3):
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                    and isinstance(node.targets[0], ast.Name):
                v = ev(node.value)
                if v is not None:
                    env[node.targets[0].id] = v
    env.pop('__file__', None)
    return env


def _make_eval(consts):
    """`os.path.join/dirname/abspath` 와 이름/속성 상수만 아는 아주 작은 평가기."""
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            return n.value
        if isinstance(n, ast.Name):
            v = consts.get(n.id)
            return v if isinstance(v, str) else None
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
            v = consts.get(f'{n.value.id}.{n.attr}')
            return v if isinstance(v, str) else None
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            fn, args = n.func.attr, [ev(a) for a in n.args]
            if any(a is None for a in args) or not args:
                return None
            try:
                if fn == 'join':
                    return os.path.join(*args)
                if fn == 'dirname':
                    return os.path.dirname(args[0])
                if fn in ('abspath', 'realpath', 'normpath'):
                    return getattr(os.path, fn)(args[0])
            except Exception:
                return None
        return None
    return ev


def _ext_exprs(tree, consts, batchdir):
    """소스의 **모든 식** 중 외부 자료 경로로 평가되는 것 → `({노드id: 라벨}, {라벨: 파일들})`.

    ★상수 이름(`B19_RESULT_GLOB`)뿐 아니라 **인라인 `os.path.join(G.B7, 'results_b7.jsonl')`**
      까지 잡는다. `analyze_b8.py:9` 처럼 **모듈 최상위 컴프리헨션**이 외부 동결 행을 읽는
      배치가 13개 있었고, 이름만 보던 종전 판은 그것을 전부 `원천미식별` 로 흘렸다.
    """
    # ★함수 **지역** 경로 변수도 푼다 — `analyze_s8.py:528` 의
    #   `s7 = os.path.join(HERE, '..', 'S7_…', 'results.jsonl')` 처럼 지역에서 조립한 뒤
    #   `open(s7)` 로 읽는 형태가 있고, 모듈 전역만 보면 그 배치는 `원천미식별` 로 샌다.
    #   ⚠이름이 서로 다른 값으로 두 번 이상 대입되면 **버린다**(모호하면 추적하지 않는다).
    env = dict(consts)
    for _ in range(2):
        ev0, seen = _make_eval(env), {}
        for n in ast.walk(tree):
            if (isinstance(n, ast.Assign) and len(n.targets) == 1
                    and isinstance(n.targets[0], ast.Name)):
                v = ev0(n.value)
                if isinstance(v, str):
                    t = n.targets[0].id
                    seen.setdefault(t, set()).add(v)
        for t, vs in seen.items():
            if len(vs) == 1 and t not in consts:
                env[t] = next(iter(vs))
            elif len(vs) > 1:
                env.pop(t, None)
    ev = _make_eval(env)
    node_lab, lab_files = {}, {}
    for n in ast.walk(tree):
        if not isinstance(n, (ast.Call, ast.Name, ast.Attribute)):
            continue
        v = ev(n)
        if not (isinstance(v, str) and _looks_like_data(v) and os.path.isabs(v)
                and _is_external(v, batchdir)):
            continue
        f = set(_expand(v))
        if not f:
            continue
        if isinstance(n, ast.Name):
            lab = n.id
        elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
            lab = f'{n.value.id}.{n.attr}'
        else:
            lab = 'path:' + os.path.basename(v)
        node_lab[id(n)] = lab
        lab_files.setdefault(lab, set()).update(f)
    return node_lab, lab_files


def _alias_map(tree, batchdir, consts_of):
    """`import gates_m3 as G` → `{'G.<이름>': 값}` (같은 배치의 모듈만)."""
    out = {}
    for node in ast.walk(tree):
        mods = []
        if isinstance(node, ast.Import):
            mods = [(a.name, a.asname or a.name) for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            mods = [(node.module, node.module)]
        for mod, alias in mods:
            base = mod.split('.')[0]
            if not os.path.exists(os.path.join(batchdir, base + '.py')):
                continue
            for k, v in consts_of(base).items():
                if isinstance(v, str):
                    out[f'{alias}.{k}'] = v
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 3. 외부를 읽는 함수 (정적 호출그래프) — 인자를 받아 호출 못 하는 것까지 덮는다
# ═══════════════════════════════════════════════════════════════════════════
JSON_PARSE = re.compile(r'json\.loads?\s*\(')


def _row_sources(tree, src, extnames, extdirs, called_ok):
    """**외부 동결 행을 실제로 돌려주는** 모듈-수준 함수 집합.

    ⛔**초판은 여기서 틀렸다** — *"외부 경로를 만지는 함수의 전이 폐포"* 로 잡았더니
      `gates()`·`aggregate()`·`selftest()` 까지 전부 들어와, **배치 자신의 행**에 대한 키 접근이
      외부 자료로 오귀속됐다. 소급 적용 1차에서 `'error'`·`'stratum'`·`'auto_selected'` 등
      **오탐 30여 건**이 나왔고 전부 이 사유였다. ⟹ 정의를 좁힌다:

      `F` 가 행 원천 ⟺
        · `F` 가 외부 경로 전역/디렉터리 전역을 **직접** 참조하고,
          `F` 자신이나 `F` 가 부르는 배치-내 함수가 **json 을 파싱**한다, 또는
        · `F` 를 실제로 **호출했더니** 외부 파일을 열고 dict 를 돌려줬다(계측 결과).

    ⚠`aggregate(rows)` 는 이제 들어오지 않는다 — 외부 전역을 만지지 않기 때문이다.
      `gates_b21.posctrl_ref` 는 들어온다(`POSCTRL_SRC` 를 직접 참조 + `_load_jsonl` 이 파싱).
    """
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    seg = {n: (ast.get_source_segment(src, f) or '') for n, f in funcs.items()}
    parses = {n for n in funcs if JSON_PARSE.search(seg[n])}
    marker = set(extnames) | set(extdirs)
    out = set(called_ok) & set(funcs)
    def _touches(node):
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and n.id in marker:
                return True
            # ★`G.B26_RESULTS_GLOB` — 별칭으로 **다른 모듈의 상수**를 쓰는 형태.
            #   `run_m3.frozen_b26_rows` 가 정확히 이 모양이라 종전 판은 M3 를 통째로 놓쳤다.
            if (isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                    and f'{n.value.id}.{n.attr}' in marker):
                return True
        return False

    for name, node in funcs.items():
        if not _touches(node):
            continue
        callees = {n.func.id for n in ast.walk(node)
                   if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        if name in parses or (callees & parses):
            out.add(name)
    return out, parses


# ═══════════════════════════════════════════════════════════════════════════
# 4. 정적 추적 — 외부 행 변수에 대한 키 리터럴 접근
# ═══════════════════════════════════════════════════════════════════════════
def _key_accesses(node, var):
    """`node` 서브트리 안에서 `var` 에 대한 **리터럴 키 접근** → {(키, 줄, 종류)}.

    종류 `'val'` = 값을 꺼내 쓴다(`r['K']` · `r.get('K')`) ⟹ 없으면 **항상 None/KeyError**.
    종류 `'in'`  = 존재만 본다(`'K' in r`) ⟹ 없으면 **필터가 no-op**.
    ★둘을 갈라 보고한다. `'error' not in r` 는 이 저장소의 표준 방어 관용구라
      동결 파일에 `error` 키가 없는 것이 **정상**이고, 이것을 ⛔로 찍으면
      *"경고가 항상 켜져 있어 신호가 죽는"* 결함이 된다.
    """
    out = set()
    for n in ast.walk(node):
        # r.get('K') / r.get('K', d)
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr in ('get', 'setdefault', 'pop')
                and isinstance(n.func.value, ast.Name) and n.func.value.id == var
                and n.args and isinstance(n.args[0], ast.Constant)
                and isinstance(n.args[0].value, str)):
            out.add((n.args[0].value, n.lineno, 'val'))
        # r['K']
        # ⛔`r['K'] = v` 는 **읽기가 아니라 쓰기**다 — 없는 것이 정상이다.
        #   `run_w5.load_reused:197` 의 `r['_src'] = 'W4'` 가 이 사유로 오탐이었다.
        if (isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
                and n.value.id == var and isinstance(n.slice, ast.Constant)
                and isinstance(n.slice.value, str)
                and not isinstance(n.ctx, (ast.Store, ast.Del))):
            out.add((n.slice.value, n.lineno, 'val'))
        # 'K' in r
        if (isinstance(n, ast.Compare) and len(n.ops) == 1
                and isinstance(n.ops[0], (ast.In, ast.NotIn))
                and isinstance(n.left, ast.Constant) and isinstance(n.left.value, str)
                and isinstance(n.comparators[0], ast.Name)
                and n.comparators[0].id == var):
            out.add((n.left.value, n.lineno, 'in'))
    return out


def _trace(tree, readers, parses=frozenset(), extexpr=None):
    """외부 자료에서 온 **행** 에 대한 키 접근을 모은다 → {(키, 함수, 줄, 종류, 원천들)}.

    ★**규칙 (오탐을 줄이려고 일부러 좁게 잡았다 — 소급 적용이 두 번 깎았다)**

    `rowset` = 외부 행들의 컬렉션. 씨앗은 `readers`(= 행 원천 함수)의 **호출**이고,
    ★**어느 원천에서 왔는지를 함께 들고 다닌다** — 배치가 외부 파일을 여러 개 읽을 때
      (에피소드 매니페스트 + 결과 jsonl) **엉뚱한 파일의 키 우주와 대조**하면 오탐이 난다
      (소급 2차에서 `gates_b21._fixture_row` 의 `'stratum'` 이 정확히 그 오탐이었다).
      파생: `Name` · `IfExp`(어느 가지든) · 컴프리헨션(어느 generator 든) ·
      `sorted/list/tuple/…`(첫 인자) · 슬라이스 · `a + b`(★**양쪽 다** rowset 일 때만 —
      외부 행과 새 행을 섞은 컬렉션에서 새 행 전용 키를 오탐하지 않기 위함).
      `fr = frozen_rows() if frozen is None else frozen` 은 `IfExp` 규칙이 잡는다 =
      **M2 의 바로 그 모양**(자가진단이 픽스처를 넣는 경로이므로 반드시 추적해야 한다).

    `row` = 행 하나. ★**루프/컴프리헨션 노드 안으로만** 범위를 제한한다.
      ⛔`(함수, 변수명)` 단위로 잡으면 안 된다 — `gates_m2.gates_run` 은 같은 이름 `r` 을
        `for r in fr`(외부)와 `for r in new`(자기 결과) **둘 다** 쓴다. 이름 단위 추적은
        자기 결과 전용 키를 통째로 오탐한다. **실측으로 확인한 오탐이다.**
      행을 인자로 넘기는 1홉(`_snap(r)`)도 ★**그 루프 안에서 실제로 넘긴 경우에만** 따라간다
        (`gates_b20._snap` 오탐이 여기서 나왔다 — 같은 함수 안 다른 루프의 `r` 이었다).
    """
    extexpr = extexpr or {}
    used = set()

    def _ids(e):
        """식 안에서 **외부 자료 경로로 평가되는 부분식**의 라벨들."""
        return {extexpr[id(n)] for n in ast.walk(e) if id(n) in extexpr}

    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    # ★모듈 스코프는 **최상위 문장만** 본다. 종전에는 `tree` 를 통째로 walk 해서
    #   함수 본문의 지역 변수까지 `<module>` 의 rowset 으로 새어 들어갔고,
    #   그 오염이 다시 인자 전파를 타고 `aggregate()` 를 오염시켰다(소급 1차 오탐의 절반).
    modtop = ast.Module(body=[n for n in tree.body
                              if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                                    ast.ClassDef))],
                        type_ignores=[])
    rowset = {k: {} for k in list(funcs) + ['<module>']}   # scope -> {var: frozenset(src)}

    def srcs(scope, e):
        """`e` 가 외부 rowset 이면 그 **원천 함수 이름들**, 아니면 빈 집합."""
        if isinstance(e, ast.Call):
            f = e.func
            if isinstance(f, ast.Name):
                if f.id in readers:
                    return frozenset({f.id})
                if scope in readers and f.id in parses:
                    return frozenset({scope})
                if f.id in PASSTHRU and e.args:
                    return srcs(scope, e.args[0])
            if isinstance(f, ast.Attribute) and f.attr in readers:
                return frozenset({f.attr})
            return frozenset()
        if isinstance(e, ast.Name):
            return rowset[scope].get(e.id, frozenset())
        if isinstance(e, ast.IfExp):
            return srcs(scope, e.body) | srcs(scope, e.orelse)
        if isinstance(e, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            out = frozenset()
            for g in e.generators:
                out |= srcs(scope, g.iter)
            # ★`[json.loads(l) for l in open(<외부>)]` — 이 저장소의 분석기 표준형이고
            #   **모듈 최상위**에 그대로 놓여 있다(`analyze_b8.py:9`·`analyze_b1.py:9`).
            if (isinstance(e.elt, ast.Call) and isinstance(e.elt.func, ast.Attribute)
                    and e.elt.func.attr == 'loads'):
                for g in e.generators:
                    out |= frozenset(_ids(g.iter))
            return out
        if isinstance(e, ast.Subscript) and isinstance(e.slice, ast.Slice):
            return srcs(scope, e.value)
        if isinstance(e, ast.BinOp) and isinstance(e.op, ast.Add):
            a, b = srcs(scope, e.left), srcs(scope, e.right)
            return (a | b) if (a and b) else frozenset()
        return frozenset()

    scopes = [('<module>', modtop)] + [(n, f) for n, f in funcs.items()]

    # ── 고정점: rowset 지역변수 · rowset 파라미터
    for _ in range(6):
        before = {s: dict(v) for s, v in rowset.items()}
        for scope, node in scopes:
            for n in ast.walk(node):
                if (isinstance(n, ast.Assign) and len(n.targets) == 1
                        and isinstance(n.targets[0], ast.Name)):
                    got = srcs(scope, n.value)
                    if got:
                        t = n.targets[0].id
                        rowset[scope][t] = rowset[scope].get(t, frozenset()) | got
                if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                        and n.func.id in funcs):
                    callee = funcs[n.func.id]
                    names = [a.arg for a in callee.args.args]
                    for i, a in enumerate(n.args):
                        if i < len(names) and srcs(scope, a):
                            rowset[n.func.id][names[i]] = (
                                rowset[n.func.id].get(names[i], frozenset())
                                | srcs(scope, a))
                    for kw in n.keywords:
                        if kw.arg in names and srcs(scope, kw.value):
                            rowset[n.func.id][kw.arg] = (
                                rowset[n.func.id].get(kw.arg, frozenset())
                                | srcs(scope, kw.value))
        if rowset == before:
            break

    hits = set()
    # (함수이름, 파라미터명) -> 원천들   ← 루프 **안에서** 행을 넘긴 1홉만 등록된다
    row_params = {}

    def _json_row_sites(scope, node):
        """행 원천 함수 **안에서** `r = json.loads(line)` 로 만들어지는 행.

        ★로더가 `for r in <rowset>` 모양이 아니라 **줄 단위로 파싱**하는 것이 이 저장소의
          표준형이다(`_b19_rows`·`_load_jsonl`·`run_m3.frozen_b26_rows`). 그 안에서 하는
          키 접근(`r.get('filter_seed')` 등)이야말로 **외부 동결 자료에 직접 닿는 지점**이다.
          범위는 그 대입을 감싸는 **가장 안쪽 루프**(없으면 함수 전체)로 제한한다.
        """
        parent = {}
        for n in ast.walk(node):
            for c in ast.iter_child_nodes(n):
                parent[c] = n
        for n in ast.walk(node):
            if not (isinstance(n, ast.Assign) and len(n.targets) == 1
                    and isinstance(n.targets[0], ast.Name)
                    and isinstance(n.value, ast.Call)
                    and isinstance(n.value.func, ast.Attribute)
                    and n.value.func.attr == 'loads'):
                # ⛔`json.load(...)` (문서 하나) 는 **행이 아니다**. `analyze_s17.py:247` 의
                #   `doc = json.load(open(thin19.json))` 를 행으로 오인해
                #   `doc['episodes']` 를 공허 참조로 찍었다(오탐).
                continue
            # 조상 루프/`with` 에서 **어느 파일을 읽는지** 를 식별자로 되짚는다.
            scope_node, cur, ids, in_loop = node, parent.get(n), set(), False
            while cur is not None:
                if isinstance(cur, ast.For):
                    if not in_loop:
                        scope_node, in_loop = cur, True
                    ids |= _ids(cur.iter)
                elif isinstance(cur, ast.With):
                    for it in cur.items:
                        ids |= _ids(it.context_expr)
                cur = parent.get(cur)
            ids |= _ids(n.value)   # `json.loads(open(<외부>).read())` 류
            src = ids
            # ★루프 밖이거나, 읽는 파일이 **외부로 해소되지 않으면** 추적하지 않는다.
            #   `analyze_s13b.load()` 는 외부 `S13_JSONL` 과 **자기 배치의** `S13B_JSONL` 을
            #   한 함수에서 읽는다 — 파일을 구분하지 않으면 자기 행의 키(`est_seed`)를
            #   외부 우주와 대조해 오탐이 난다(실측 확인).
            if in_loop and src:
                yield scope_node, n.targets[0].id, frozenset(src)

    def take(scope, node, var, src):
        used.update(src)
        for k, ln, kind in _key_accesses(node, var):
            hits.add((k, scope, ln, kind, src))
        # 이 루프 안에서 `f(var)` 로 넘긴 배치-내 함수의 파라미터를 1홉 따라간다
        for n in ast.walk(node):
            if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id in funcs):
                continue
            names = [a.arg for a in funcs[n.func.id].args.args]
            for i, a in enumerate(n.args):
                if i < len(names) and isinstance(a, ast.Name) and a.id == var:
                    key = (n.func.id, names[i])
                    row_params[key] = row_params.get(key, frozenset()) | src

    for scope, node in scopes:
        for sub, var, psrc in _json_row_sites(scope, node):
            take(scope, sub, var, psrc)
        for n in ast.walk(node):
            if (isinstance(n, ast.For) and isinstance(n.target, ast.Name)
                    and srcs(scope, n.iter)):
                take(scope, n, n.target.id, srcs(scope, n.iter))
            if isinstance(n, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
                for g in n.generators:
                    if isinstance(g.target, ast.Name) and srcs(scope, g.iter):
                        take(scope, n, g.target.id, srcs(scope, g.iter))

    for (fname, pname), src in list(row_params.items()):
        for k, ln, kind in _key_accesses(funcs[fname], pname):
            hits.add((k, fname, ln, kind, src))
    for h in hits:
        used |= set(h[4])
    return hits, used


# ═══════════════════════════════════════════════════════════════════════════
# 5. 배치 하나 검사
# ═══════════════════════════════════════════════════════════════════════════
JUDGE_AUX = re.compile(r'^(run|analyze)_.*\.py$')   # 판정을 낳는 배치 코드 (게이트 외)


def check_batch(batchdir, verbose=False):
    """→ dict(gates=…, missing=[…], ext_files=n, ext_rows=n, ext_keys=set, note=str)

    ★검사 대상은 `gates_*.py` **만이 아니다**. `run_*.py`·`analyze_*.py` 도 본다 —
      `run_m3.frozen_b26_rows()` 처럼 **외부 동결 행을 러너가 읽는 배치가 실제로 있고**,
      게이트만 보면 그 배치는 `외부참조 없음` 으로 **조용히 통과**한다(= 이 검사기의
      사각지대였다. 소급 적용 중 M3 에서 발견). 러너는 ⛔**임포트하지 않고** 정적으로 판다.
    """
    batchdir = os.path.abspath(batchdir)
    gates = sorted(globmod.glob(os.path.join(batchdir, 'gates_*.py')))
    aux = sorted(q for q in globmod.glob(os.path.join(batchdir, '*.py'))
                 if JUDGE_AUX.match(os.path.basename(q)))
    res = dict(batch=os.path.basename(batchdir), gates=[], aux=[], missing=[], ext_files=0,
               ext_rows=0, ext_keys=set(), readers=set(), empties=set(), note='', err=None)
    if not gates and not aux:
        res['note'] = '판정 코드 없음'
        return res
    all_keys, all_files, readers_all, missing = set(), set(), set(), []
    filekeys, nrows = {}, 0
    consts_cache = {}

    def consts_of(base):
        """배치-내 모듈 `base` 의 최상위 경로 상수(정적)."""
        if base in consts_cache:
            return consts_cache[base]
        consts_cache[base] = {}                     # 재귀 방지
        q = os.path.join(batchdir, base + '.py')
        try:
            t = ast.parse(open(q, encoding='utf-8', errors='replace').read())
            consts_cache[base] = _static_consts(t, q)
        except Exception:
            pass
        return consts_cache[base]

    for path in gates + aux:
        is_gate = path in gates
        (res['gates'] if is_gate else res['aux']).append(os.path.basename(path))
        try:
            src = open(path, encoding='utf-8', errors='replace').read()
            tree = ast.parse(src)
        except Exception as e:
            res['err'] = f'파싱 실패 {os.path.basename(path)}: {e}'
            continue

        mod, called = None, {}
        consts = _static_consts(tree, path, _alias_map(tree, batchdir, consts_of))
        if is_gate:
            # ★게이트만 임포트한다(런타임 계측이 정적 분석의 취약함을 메운다).
            try:
                mod = _import(path)
                consts = dict(consts, **{k: v for k, v in vars(mod).items()
                                         if not k.startswith('__')})
            except BaseException as e:                   # noqa: BLE001
                res['err'] = (f'임포트 실패 {os.path.basename(path)}: '
                              f'{type(e).__name__}: {e}')
                if verbose:
                    traceback.print_exc()

        nodelab, labfiles = _ext_exprs(tree, consts, batchdir)
        gpaths, gempty = _globals_paths(consts, batchdir)
        extdirs = _extdirs(consts, batchdir)
        files = set().union(*gpaths.values()) if gpaths else set()
        jfiles, jempty = _static_joins(tree, consts, batchdir)
        files |= jfiles
        for _f in labfiles.values():
            files |= _f
        empties = set(gempty) | set(jempty)
        if mod is not None:
            called = _call_loaders(mod, tree, batchdir, module_src=src, verbose=verbose)
            for info in called.values():
                files |= info['files']
                empties |= info['empty']
        res['empties'] |= empties

        called_rows = {n for n, i in called.items() if i['rows'] > 0}
        readers, parses = _row_sources(tree, src, set(gpaths), set(extdirs), called_rows)
        # ★원천별 파일 매핑 — 배치가 외부 파일을 **여러 갈래**로 읽을 때
        #   (에피소드 매니페스트 vs 결과 jsonl) 엉뚱한 우주와 대조하면 오탐이 난다.
        src_files, fnodes = {}, {n.name: n for n in tree.body
                                 if isinstance(n, ast.FunctionDef)}
        for rname in readers:
            f = set(called.get(rname, {}).get('files', ()))
            if not f and rname in fnodes:
                node = fnodes[rname]
                for n in ast.walk(node):
                    if isinstance(n, ast.Name) and n.id in gpaths:
                        f |= gpaths[n.id]
                    if (isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                            and f'{n.value.id}.{n.attr}' in gpaths):
                        f |= gpaths[f'{n.value.id}.{n.attr}']
                jf, _je = _static_joins(node, consts, batchdir)
                f |= jf
            if f:
                src_files[rname] = f
        # ★경로 **식** 라벨도 원천 키다 — 그 라벨이 가리키는 **그 파일들**과만 대조한다.
        for lab, lf in labfiles.items():
            src_files[lab] = set(src_files.get(lab, set())) | lf
        readers_all |= readers
        if verbose:
            print(f'      · {os.path.basename(path)}: 외부 파일 {len(files)} · '
                  f'행 원천 {sorted(readers)}')

        for f in sorted(files):
            if f not in filekeys:
                k, n = _row_keys(f)
                filekeys[f] = k
                nrows += n
        all_files |= files
        all_keys |= set().union(*[filekeys[f] for f in files]) if files else set()
        if not readers and not nodelab:
            continue

        keycache = {}

        def universe(srcnames, _files=files, _sf=src_files, _kc=keycache):
            """이 원천들이 실제로 읽는 파일들의 **키 합집합**. 매핑이 없으면 전체로 후퇴."""
            f = set()
            for sn in srcnames:
                f |= _sf.get(sn, set())
            if not f:
                f = _files
            key = frozenset(f)
            if key not in _kc:
                u = set()
                for q in sorted(f):
                    u |= filekeys.get(q, set())
                _kc[key] = (u, len(f))
            return _kc[key]

        byk = {}
        hits, used = _trace(tree, readers, parses, nodelab)
        readers_all |= used
        for key, scope, line, kind, srcnames in sorted(hits, key=lambda h: h[:4]):
            u, nf = universe(srcnames)
            if key in u or not u:
                continue
            k = (key, scope, line)
            prev = byk.get(k)
            byk[k] = (('val' if (kind == 'val' or (prev and prev[0] == 'val')) else 'in'),
                      tuple(sorted(srcnames)), len(u), nf)
        for (key, scope, line), (kind, srcnames, nu, nf) in sorted(byk.items()):
            missing.append(dict(gate=os.path.basename(path), key=key, func=scope,
                                line=line, kind=kind, src=list(srcnames),
                                universe=nu, files=nf))

    res.update(ext_files=len(all_files), ext_rows=nrows, ext_keys=all_keys,
               readers=readers_all, missing=missing)
    return res


# ═══════════════════════════════════════════════════════════════════════════
# 6. 자가진단 — ★일부러 깨진 픽스처에서 발화하는가 + 오탐 회귀
# ═══════════════════════════════════════════════════════════════════════════
_FX_EXT = '''\
{"arm": "SPEC25", "episode": "e0", "avg_nRMSE": 0.11, "cap_loc_radius": 25.0}
{"arm": "SPEC25", "episode": "e1", "avg_nRMSE": 0.12, "cap_loc_radius": 25.0}
{"arm": "SHIP", "episode": "e0", "avg_nRMSE": 0.07, "cap_varpro_steps": 400}
'''

_FX_BROKEN = '''\
import glob, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(HERE, "..", "EXTB", "results_ext_*.jsonl")

def frozen_rows():
    rows = []
    for p in sorted(glob.glob(EXT)):
        for line in open(p):
            rows.append(json.loads(line))
    return rows

def gates_run(rows, frozen=None):
    fr = frozen_rows() if frozen is None else frozen
    out = []
    ref = {}
    for r in fr:
        if r.get("arm") == "SPEC25" and r.get("cap_effective_config"):
            ref = r["cap_effective_config"]
            break
    out.append(("G-13", bool(ref), ""))
    bad = [r for r in fr if r.get("obs_noise_sigma") not in (None, 0.02)]
    out.append(("G-14", not bad, ""))
    for r in rows:
        if r.get("filter_class") != "LETKFGridSpec":
            out.append(("G-16", False, ""))
    return out
'''

_FX_CLEAN = '''\
import glob, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(HERE, "..", "EXTB", "results_ext_*.jsonl")

def frozen_rows():
    rows = []
    for p in sorted(glob.glob(EXT)):
        for line in open(p):
            rows.append(json.loads(line))
    return rows

def gates_run(rows, frozen=None):
    fr = frozen_rows() if frozen is None else frozen
    ref = {}
    for r in fr:
        if r.get("arm") == "SPEC25" and r.get("cap_loc_radius"):
            ref = r["cap_loc_radius"]
            break
    out = [("G-13", bool(ref), "")]
    for r in rows:
        if r.get("filter_class") != "LETKFGridSpec":
            out.append(("G-16", False, ""))
    return out
'''


# ★F-2 회귀 픽스처 (2026-08-29) — **jsonl 읽기를 공유 헬퍼로 단일출처화한 배치**.
#   `gates_singlesrc.frozen_rows()` 는 `SIO.load_jsonl` 과 별칭 `_g.glob` 만 쓴다 ⟹
#   종전의 문자 매칭(`json.load(`·`.jsonl`·`glob.glob(`)에 **하나도 걸리지 않는다**.
#   ⟹ 로더 0개 = `⚠원천미식별` = **공허 참조 `cap_ghost_key` 를 놓친다**.
#   ★그리고 같은 파일의 `_fixture_rows()` 는 **로더로 세면 안 된다**(합성 행 제조기).
_FX_SHARED_IO = """\
import glob
import json


def find(pattern):
    return sorted(glob.glob(pattern))


def load_jsonl(path):
    return [json.loads(x) for x in open(path) if x.strip()]
"""

_FX_SINGLESRC = """\
import os

import shared_io as SIO

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(HERE, "..", "EXTB", "results_ext_*.jsonl")


def frozen_rows():
    rows = []
    for q in SIO.find(EXT):
        rows += SIO.load_jsonl(q)
    return rows


def alias_glob_rows():
    import glob as _g
    out = []
    for q in sorted(_g.glob(EXT)):
        for x in open(q):
            if x.strip():
                out.append({"arm": x[8:14]})
    return out


def _fixture_rows():
    return [dict(arm="SHIP", episode=r.get("episode")) for r in frozen_rows()]


def gates_run(rows, frozen=None):
    fr = frozen_rows() if frozen is None else frozen
    ref = {}
    for r in fr:
        if r.get("arm") == "SPEC25" and r.get("cap_ghost_key"):
            ref = r["cap_ghost_key"]
            break
    out = [("G-1", bool(ref), "")]
    for r in fr:
        if r.get("cap_loc_radius"):
            out.append(("G-2", True, ""))
    return out
"""


_FX_RUNNER_GATES = '''\
import os
HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(HERE, "..", "EXTB", "results_ext_*.jsonl")
PC_ARM = "SPEC25"
'''

_FX_RUNNER = '''\
import glob, json, os
import gates_runner as G
HERE = os.path.dirname(os.path.abspath(__file__))

def frozen_ref():
    """게이트가 아니라 **러너** 가 외부 동결 행을 읽는다 (M3 가 이 모양이다)."""
    out = {}
    for p in sorted(glob.glob(G.EXT)):
        for line in open(p):
            r = json.loads(line)
            if "error" in r:
                continue
            if r.get("arm") == G.PC_ARM:
                out[r["episode"]] = (r["avg_nRMSE"], r.get("cap_n_ensemble"))
    return out

def ref_local():
    """경로를 **함수 지역 변수**로 조립해서 읽는다 (analyze_s8.py:528 의 모양)."""
    q = os.path.join(HERE, "..", "EXTB", "results_ext_0.jsonl")
    out = {}
    for line in open(q):
        r = json.loads(line)
        out[r["episode"]] = r["cap_mystery_key"]
    return out
'''


_FX_MIXED_LOCAL = '''\
{"arm": "SHIP", "episode": "e0", "avg_nRMSE": 0.07, "est_seed": 43}
'''

_FX_MIXED_DOC = '''{"episodes": [{"episode": "e0"}]}'''

_FX_MIXED = '''\
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(HERE, "..", "EXTB", "results_ext_0.jsonl")
OWN = os.path.join(HERE, "results_own.jsonl")
DOC = os.path.join(HERE, "thin.json")

def load():
    """★한 함수가 **외부 파일과 자기 파일을 둘 다** 읽는다 (analyze_s13b 의 모양)."""
    out = []
    for line in open(EXT):
        r = json.loads(line)
        out.append((r["arm"], r["avg_nRMSE"]))
    for line in open(OWN):
        r = json.loads(line)
        out.append((r["est_seed"], r["avg_nRMSE"]))   # est_seed 는 자기 파일에만 있다
    doc = json.load(open(DOC))
    return out, doc["episodes"]                       # 문서 하나 = 행이 아니다

def tag():
    """`r['K'] = v` 는 쓰기다 (run_w5.load_reused 의 모양)."""
    out = []
    for line in open(EXT):
        r = json.loads(line)
        r["_src"] = "EXT"
        out.append(r)
    return out
'''


def selftest(verbose=True):
    import shutil
    import tempfile
    base = tempfile.mkdtemp(prefix='gref_selftest_')
    ok = True

    def say(s):
        if verbose:
            print(s)

    def check(tag, cond, detail=''):
        nonlocal ok
        ok = ok and bool(cond)
        say(f'  {"✅" if cond else "⛔"} {tag}' + (f'  — {detail}' if detail else ''))

    try:
        os.makedirs(os.path.join(base, 'EXTB'))
        with open(os.path.join(base, 'EXTB', 'results_ext_0.jsonl'), 'w') as fh:
            fh.write(_FX_EXT)
        for name, src in (('BROKEN', _FX_BROKEN), ('CLEAN', _FX_CLEAN)):
            d = os.path.join(base, name)
            os.makedirs(d)
            with open(os.path.join(d, f'gates_{name.lower()}.py'), 'w') as fh:
                fh.write(src)

        say('[1] 일부러 깨진 픽스처 — 외부 자료에 없는 키를 참조한다')
        r = check_batch(os.path.join(base, 'BROKEN'))
        keys = {m['key'] for m in r['missing']}
        check('외부 파일을 찾았다', r['ext_files'] == 1, f"{r['ext_files']}개")
        check('로더를 외부-읽기 함수로 인식', 'frozen_rows' in r['readers'], str(sorted(r['readers'])))
        check('★공허 FAIL 형(`cap_effective_config`) 을 잡았다',
              'cap_effective_config' in keys, str(sorted(keys)))
        check('★공허 PASS 형(`obs_noise_sigma`, `not bad` 가 항상 참) 을 잡았다',
              'obs_noise_sigma' in keys, str(sorted(keys)))
        check('⛔오탐 없음: 자기 결과 행(`rows`)의 `filter_class` 는 잡지 않는다',
              'filter_class' not in keys, str(sorted(keys)))
        check('⛔오탐 없음: 외부에 실제로 있는 키는 잡지 않는다',
              not ({'arm', 'cap_loc_radius', 'avg_nRMSE'} & keys), str(sorted(keys)))

        say('[2] 성한 픽스처 — 있는 키만 참조한다')
        r2 = check_batch(os.path.join(base, 'CLEAN'))
        check('공허 참조 0건', not r2['missing'],
              str([m['key'] for m in r2['missing']]))
        check('그래도 대조는 실제로 일어났다(침묵 아님)',
              r2['ext_files'] == 1 and r2['ext_keys'], f"키 {len(r2['ext_keys'])}개")

        say('[3] ★자기 무력화 시험 — 외부 파일이 사라지면 침묵하지 말고 경고해야 한다')
        d3 = os.path.join(base, 'GHOST')
        os.makedirs(d3)
        with open(os.path.join(d3, 'gates_ghost.py'), 'w') as fh:
            fh.write(_FX_BROKEN.replace('EXTB', 'NOSUCHDIR'))
        r3 = check_batch(d3)
        check('외부 파일 0개인데 **침묵하지 않는다** = `⚠대조0` 경고 대상',
              r3['ext_files'] == 0 and (r3['empties'] or r3['readers']),
              f"파일 {r3['ext_files']} · 0개 패턴 {len(r3['empties'])} · 리더 {sorted(r3['readers'])}")

        say('[6] ★오탐 회귀 — 소급 적용이 실제로 뱉었던 오탐 세 부류')
        d6 = os.path.join(base, 'MIXED')
        os.makedirs(d6)
        with open(os.path.join(d6, 'gates_mixed.py'), 'w') as fh:
            fh.write(_FX_MIXED)
        with open(os.path.join(d6, 'results_own.jsonl'), 'w') as fh:
            fh.write(_FX_MIXED_LOCAL)
        with open(os.path.join(d6, 'thin.json'), 'w') as fh:
            fh.write(_FX_MIXED_DOC)
        r6b = check_batch(d6)
        k6 = {m['key'] for m in r6b['missing'] if m['kind'] == 'val'}
        check('⛔`est_seed`(자기 파일에만 있는 키) 오탐 없음 — 파일별로 우주를 가른다',
              'est_seed' not in k6, str(sorted(k6)))
        check('⛔`episodes`(`json.load` 문서) 오탐 없음 — 문서는 행이 아니다',
              'episodes' not in k6, str(sorted(k6)))
        check("⛔`_src`(`r['K'] = v` 쓰기) 오탐 없음 — 대입은 접근이 아니다",
              '_src' not in k6, str(sorted(k6)))
        check('그래도 외부 파일은 실제로 읽었다(침묵 아님)', r6b['ext_files'] >= 1,
              f"{r6b['ext_files']}개 · 원천 {sorted(r6b['readers'])}")

        say('[5] ★러너가 외부를 읽는 배치 — 게이트만 보면 조용히 통과한다 (M3 부류)')
        d5 = os.path.join(base, 'RUNNER')
        os.makedirs(d5)
        with open(os.path.join(d5, 'gates_runner.py'), 'w') as fh:
            fh.write(_FX_RUNNER_GATES)
        with open(os.path.join(d5, 'run_runner.py'), 'w') as fh:
            fh.write(_FX_RUNNER)
        r5 = check_batch(d5)
        k5 = {m['key'] for m in r5['missing'] if m['kind'] == 'val'}
        check('러너를 **임포트하지 않고** 외부 파일을 찾았다(별칭 `G.EXT` 해소)',
              r5['ext_files'] == 1, f"{r5['ext_files']}개")
        check('러너 안의 행 원천을 인식', 'frozen_ref' in r5['readers'], str(sorted(r5['readers'])))
        check('★러너의 공허 참조(`cap_n_ensemble`) 를 잡았다',
              'cap_n_ensemble' in k5, str(sorted(k5)))
        check('★**함수 지역 변수**로 조립한 경로도 푼다(`cap_mystery_key`)',
              'cap_mystery_key' in k5, str(sorted(k5)))
        check('⛔오탐 없음: 있는 키(`arm`·`episode`·`avg_nRMSE`)는 잡지 않는다',
              not ({'arm', 'episode', 'avg_nRMSE'} & k5), str(sorted(k5)))
        check("`'error' in r` 는 ⚠(존재검사) 등급으로만 나온다",
              any(m['key'] == 'error' and m['kind'] == 'in' for m in r5['missing']),
              str([(m['key'], m['kind']) for m in r5['missing']]))

        say('[7] ★F-2 회귀 — **읽기를 단일출처화한 배치가 보이는가**')
        d7 = os.path.join(base, 'SINGLESRC')
        os.makedirs(d7)
        with open(os.path.join(d7, 'shared_io.py'), 'w') as fh:
            fh.write(_FX_SHARED_IO)
        with open(os.path.join(d7, 'gates_singlesrc.py'), 'w') as fh:
            fh.write(_FX_SINGLESRC)
        r7 = check_batch(d7)
        k7 = {m['key'] for m in r7['missing'] if m['kind'] == 'val'}
        check('★**폐포** — 본문에 `json`·`glob`·`.jsonl` 이 한 글자도 없고 공유 헬퍼'
              '(`SIO.find`/`SIO.load_jsonl`)만 부르는 로더를 본다 (종전 = `⚠원천미식별`)',
              'frozen_rows' in r7['readers'], str(sorted(r7['readers'])))
        check('★**별칭** — `import glob as _g` 뒤 `_g.glob(` 도 본다 '
              '(종전 정규식은 `glob.glob(` 로 수신자까지 박아 놨다)',
              'alias_glob_rows' in r7['readers'], str(sorted(r7['readers'])))
        check('그래서 외부 파일·행을 실제로 읽었다(침묵 아님)',
              r7['ext_files'] == 1 and r7['ext_keys'], f"파일 {r7['ext_files']} · 키 {len(r7['ext_keys'])}개")
        check('★그 덕에 공허 참조(`cap_ghost_key`) 를 잡는다 — 종전에는 놓쳤다',
              'cap_ghost_key' in k7, str(sorted(k7)))
        check('⛔오탐 없음: 있는 키(`cap_loc_radius`)는 잡지 않는다',
              'cap_loc_radius' not in k7, str(sorted(k7)))
        check('★⛔**픽스처 제조기(`_fixture_rows`)는 로더로 세지 않는다** '
              '— 세면 합성 키 우주로 오탐 17건이 난다(B19·B21·B22·M8 실측)',
              '_fixture_rows' not in r7['readers'], str(sorted(r7['readers'])))

        say('[4] ★M2 실물에 대고 — `G-13` 의 두 키를 실제로 잡는가')
        m2 = os.path.join(TMP, 'M2_LOCTUNE_2026-08-28')
        if os.path.isdir(m2):
            r4 = check_batch(m2)
            k4 = {m['key'] for m in r4['missing'] if m['kind'] == 'val'}
            check('`cap_effective_config` · `effective_config` 둘 다 잡았다',
                  {'cap_effective_config', 'effective_config'} <= k4, str(sorted(k4)))
        else:
            say('     (M2 배치 없음 — 건너뜀)')
        m3 = os.path.join(TMP, 'M3_SUPPORT_2026-08-28')
        if os.path.isdir(m3):
            r6 = check_batch(m3)
            check('★M3 회귀: 러너가 읽는 외부 원천이 **보인다**(종전에는 0이라 침묵했다)',
                  len(r6['readers']) > 0 and r6['ext_files'] > 0,
                  f"원천 {sorted(r6['readers'])} · 파일 {r6['ext_files']}")
    finally:
        shutil.rmtree(base, ignore_errors=True)
    say('\n' + ('✅자가진단 통과' if ok else '⛔자가진단 실패'))
    return 0 if ok else 1


# ═══════════════════════════════════════════════════════════════════════════
# 7. main
# ═══════════════════════════════════════════════════════════════════════════
def _one(d, verbose=False):
    """`--one` 모드 — 배치 하나를 검사하고 결과를 JSON 으로 뱉는다(부모가 파싱한다)."""
    r = check_batch(d, verbose=verbose)
    r = dict(r)
    r['ext_keys'] = sorted(r['ext_keys'])
    r['readers'] = sorted(r['readers'])
    r['empties'] = sorted(r['empties'])
    print('@@GREF@@' + json.dumps(r, ensure_ascii=False))
    return 0


def _run_isolated(d, verbose=False):
    """★배치마다 **새 프로세스**에서 검사한다.

    ⛔같은 프로세스에서 여러 배치를 임포트하면 `sys.modules['gates_b26']` 이 캐시돼
      **다른 배치의 헬퍼가 조용히 재사용**된다. 그러면 검사기가 *"검사했다"* 고 찍으면서
      **엉뚱한 자료를 봤을 수** 있다 — 이 검사기가 잡으려는 결함과 같은 부류의 자기 결함이다.
    """
    import subprocess
    cmd = [sys.executable, os.path.abspath(__file__), '--one', d] + (['-v'] if verbose else [])
    try:
        pr = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return None, '시간초과(600s)'
    if verbose and pr.stdout:
        for ln in pr.stdout.splitlines():
            if not ln.startswith('@@GREF@@'):
                print(ln)
    for ln in pr.stdout.splitlines():
        if ln.startswith('@@GREF@@'):
            r = json.loads(ln[len('@@GREF@@'):])
            r['ext_keys'] = set(r['ext_keys'])
            r['readers'] = set(r['readers'])
            r['empties'] = set(r['empties'])
            return r, None
    return None, (pr.stderr.strip().splitlines() or ['출력 없음'])[-1][:200]


def main(argv):
    verbose = '-v' in argv or '--verbose' in argv
    args = [a for a in argv[1:] if not a.startswith('-')]
    if '--selftest' in argv:
        return selftest()
    if '--one' in argv:
        return _one(os.path.abspath(args[0]), verbose=verbose)
    if args:
        dirs = [os.path.abspath(a) for a in args]
    else:
        # ★`gates_*.py` 만 보면 **러너에 판정 자료 읽기가 있는 배치**를 통째로 놓친다(M3).
        dirs = sorted({os.path.dirname(p) for pat in ('gates_*.py', 'run_*.py', 'analyze_*.py')
                       for p in globmod.glob(os.path.join(TMP, '*', pat))})

    print('게이트 외부 참조 검사 — *"게이트가 참조하는 키가 외부 동결 자료에 실제로 있는가"*')
    print('⛔사전등록 시점에 돌려야 하는 검사다. 외부 동결 파일은 그때 **이미 존재**한다.\n')
    print(f'{"배치":42s} {"외부파일":>6s} {"외부행":>7s} {"리더":>4s} {"공허참조":>8s}  비고')
    print('─' * 100)
    bad, soft, warned, errs, silent, unmapped = [], [], [], [], [], []
    for d in dirs:
        if verbose:
            print(f'  ▸ {os.path.basename(d)}')
        r, ferr = _run_isolated(d, verbose=verbose)
        if r is None:
            errs.append((os.path.basename(d), ferr))
            print(f"{os.path.basename(d)[:42]:42s} {'':6s} {'':7s} {'':4s} {'!!':>8s}  ⚠검사 실패: {ferr}")
            continue
        if not r['gates'] and not r['aux']:
            continue
        note = r['note']
        if r['err']:
            note = (note + ' ' + r['err']).strip()
            errs.append((r['batch'], r['err']))
        if (r['readers'] or r['empties']) and r['ext_files'] == 0:
            note = ('⚠대조0(외부를 겨냥했는데 파일을 하나도 못 열었다) ' + note).strip()
            warned.append(r['batch'])
        if not r['readers'] and r['ext_files']:
            # ★침묵을 경고로 승격 — 외부 파일은 찾았는데 **행 원천 함수를 못 짚었다**.
            #   M3 가 정확히 이 상태였고(러너가 읽었다) 그때 표는 `외부참조 없음` 이라 찍었다.
            note = (f"⚠원천미식별(외부 파일 {r['ext_files']}개를 찾았으나 "
                    f"행 원천 함수를 못 짚었다) " + note).strip()
            unmapped.append(r['batch'])
        elif not r['readers'] and not r['empties']:
            note = (note + ' 외부참조 없음').strip()
            silent.append(r['batch'])
        nval = len([m for m in r['missing'] if m['kind'] == 'val'])
        nin = len(r['missing']) - nval
        mark = (f'⛔{nval}' if nval else (f'⚠{nin}' if nin
                else ('0' if (r['readers'] or r['empties']) else '--')))
        print(f"{r['batch'][:42]:42s} {r['ext_files']:6d} {r['ext_rows']:7d} "
              f"{len(r['readers']):4d} {mark:>8s}  {note}")
        if [m for m in r['missing'] if m['kind'] == 'val']:
            bad.append(r)
        elif r['missing']:
            soft.append(r)

    if bad:
        print('\n⛔**공허 참조 — 게이트가 외부 동결 자료에 없는 키를 참조한다.**')
        print('   ★그 게이트는 자료에 대고는 처음부터 통과(또는 평가)가 불가능하다.')
        print('   ⚠**어느 방향인지는 사람이 그 줄을 읽어야 한다**:')
        print('     · 조건을 False 로 만들면 **공허 FAIL** — 아프지만 탐지된다.')
        print('     · 필터를 비워 조건을 True 로 만들면 **공허 PASS** — ★구조적으로 탐지 불가.')
        for r in bad:
            print(f"\n  {r['batch']}   (외부 키 우주 {len(r['ext_keys'])}개 · 파일 {r['ext_files']}개)")
            for m in sorted([m for m in r['missing'] if m['kind'] == 'val'],
                            key=lambda m: (m['gate'], m['line'])):
                print(f"    {m['gate']}:{m['line']}  {m['func']}()  →  '{m['key']}' 없음"
                      f"  (값 접근 ⟹ 항상 None; 원천 {'/'.join(m['src']) or '?'} "
                      f"· 파일 {m['files']}개 · 키 {m['universe']}개)")
    softall = soft + [r for r in bad if any(m['kind'] == 'in' for m in r['missing'])]
    if softall:
        print('\n⚠**존재 검사만 하는 누락 키** — `\'K\' in r` 가 **영원히 거짓**이다.\n'
              '   대개 방어 관용구(`\'error\' not in r`)라 무해하지만, 그 필터는 no-op 이다:')
        for r in softall:
            ks = sorted({m['key'] for m in r['missing'] if m['kind'] == 'in'})
            sites = [f"{m['gate']}:{m['line']}" for m in r['missing'] if m['kind'] == 'in']
            print(f"  {r['batch']}: {', '.join(ks)}   ({len(sites)}곳)")
    if warned:
        print('\n⚠**대조 0건** — 외부를 읽는 함수를 찾았는데 파일을 하나도 못 열었다.\n'
              '   ★`OK` 가 아니다. 경로가 바뀌었거나 자료가 아직 없다는 뜻이고,\n'
              '   그 상태에서는 이 검사기가 **아무것도 검증하지 못했다**: ' + ', '.join(warned))
    if unmapped:
        print('\n⚠**행 원천 미식별** — 외부 결과 파일 경로는 찾았는데 그것을 **행으로 읽는\n'
              '   함수를 못 짚었다**. 경로 상수만 있고 안 쓰는 것일 수도 있고, 이 검사기가\n'
              '   못 따라가는 모양일 수도 있다. ★`OK` 로 읽지 마라(M3 가 이 상태였다): '
              + ', '.join(unmapped))
    if errs:
        print('\n⚠**임포트/파싱 실패** — 그 배치는 검사되지 않았다(침묵을 경고로 승격):')
        for k, v in errs:
            print(f'  {k}: {v}')
    print(f'\n공허 참조(값 접근) 있는 배치: {len(bad)}/{len([d for d in dirs])}  ·  '
          f'외부참조 없어 대상 밖: {len(silent)}  ·  ⚠원천미식별: {len(unmapped)}  ·  '
          f'대조0 경고: {len(warned)}  ·  실패: {len(errs)}')
    print('⛔이 검사기가 **보지 않는 것**: 배치 자신의 `results*.jsonl` · 팔 단위 키 부재 ·\n'
          '   다른 모듈로 넘어간 뒤의 **키 추적**(로더 *발견* 은 2026-08-29 부터 넘어간다) ·\n'
          '   변수 키(`r.get(k)`) · **픽스처 제조기가 여는 파일**(`_fixture*`·`synthetic_rows`,\n'
          '   일부러 제외 — 합성 행이라 우주로 쓰면 오탐 17건) · 인자 있는 로더 ·\n'
          '   동적 디스패치와 깊이 %d 넘는 호출 사슬. 「검사했다」의 범위를 넘지 마라.' % CLOSURE_DEPTH)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
