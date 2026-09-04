#!/usr/bin/env python3
"""사전등록 방벽 — **`T11-EXTACC` 가 하루에 네 번 낸 결함 부류를 결과 0행 시점에 잡는다** (2026-09-04).

⛔이 검사기는 `check_batch_artifacts.py`(존재·짝·mtime·md5·자기동결·`{}` 팔) ·
`check_gate_coverage.py`(**verdict 기준** 사전등록 게이트 미구현 — 결과 이후) ·
`check_gate_refs.py`(외부 자료 키 부재) · `check_gate_blindspots.py`(필터 항진) 가 **이미 보는
것을 다시 보지 않는다.** 여기서 보는 것은 전부 **소스와 사전등록 문서만으로, 결과가 나오기 전에**
판정 가능한 부류다. 사건 근거는 `tmp/T11_EXTACC_2026-09-03/AMENDMENT_2_PREDATA.md`(A2) ·
`AMENDMENT_3_POSTRESULT.md`(A3) · `AMENDMENT_1_PREDATA.md`(A1) 와 `tmp/_audit/PREREG_CHECKLIST.md`.

검사 항목 (배치 디렉터리 하나당 · 항목 ID 는 `PREREG_CHECKLIST.md` 와 같다)
  C1  MDE        사전등록 문서에 MDE/q80/등가한계 가 **수치와 함께** 있고 gates 에 상수가 있다      (A2 §3)
  C2  PRED       gates.label() 이 내는 **모든 라벨**에 사전등록 확률이 붙어 있다 (C2b: 확률을 **코드가** 계산한다) (A2 §2 · A3 §5)
  C3  LABELS     gates.label() 반환 라벨이 사전등록 문서에 있고, 사전등록 표의 라벨을 gates 가 낼 수 있다 (18216fc^: PREREG v2 ↔ gates v1)
  C4  POSLABEL   「기각 실패」 다리(≤k · 널 확률 > α)에 **양성 이름**의 라벨이 없다 — 표(PREREG)와 코드(label()) 둘 다 (A2 §1)
  C5  DECOR      분석기가 만드는 게이트 이름을 **실제 `blocked_by()` 에 넣어** 전부 차단됨을 확인 (장식 게이트 · T11 V3 방법 · A2 §7 G-COV)
  C6  TAUTO      항진 후보 — (a) 러너가 `k=G.C`/숫자 리터럴을 쓰고 분석기가 **같은 것**과 비교 (b) 게이트 조건이 기록값의 존재/불리언 그대로
                 (c) 같은 시점 같은 식을 두 키로 기록                                           (A2 §7 G-ENV · A1 §4 · A3 §0 · §2)
  C7  FROZEN     `FROZEN*.txt` 가 러너·게이트·분석기의 **패키지 임포트 경로**를 덮고(직접=FAIL · 깊이2=WARN), 학습 팔이면 **체크포인트 지문**이 있다 (A2 §7·§8)
  C8  GATESIMPL  사전등록 표의 게이트가 분석기 소스에 **전부** 있다 (결과 0행 판 — verdict 판은 `check_gate_coverage`)
  C9  DISCLOSE   사전등록 §금지/범위/공시 의 ⛔ 항목 토큰이 분석기 소스(공시)에 전부 있다 (N−1 · A3 §1 = 사전등록 위반)
  C10 GATELOSS   git 이전 판본에 있던 게이트가 사라졌는데 사전등록 문서 어디에도 그 이름이 없다 (N−1 · A3 §3 G-DENSITY)
  C11 SMOKE      스모크 경로/산출물/언급이 있다                                                  (feedback_smoke_before_long_runs)
  C12 PROV       추정기를 만드는 러너가 `arm_provenance()` 를 행에 넣고 분석기가 읽는다         (CLAUDE.md 「팔 정체는 되읽기로」)
  C13 MULT       부차 통계량이 둘 이상이면 보정/무보정 공시가 있다                               (A2 §4)
  C14 BLIND      blind/노출 상태를 적었다                                                        (PREREG v2 §6 · T5-ENS)
  C15 PRIMARY    PRIMARY 통계량 · 방향 · n 이 문서에 있다                                        (feedback_prespecify_aggregation_and_direction)
  C16 SEED       시드(축)를 언급했다                                                             (DISCIPLINE_seed_axes · CLAUDE.md S13)
  C17 KEYS       분석기가 읽는 행/meta 키를 러너가 어딘가에서 쓴다 (⚠팔 단위 누락은 못 본다 — A2 §9.2 는 게이트가 잡았다)
  C18 BELT       라벨 인쇄가 `blocked` 에 매달려 있다                                            (A3 §6 · T5-ENS blocked_by 항상 빈 리스트)

등급: FAIL(종료코드 1) · WARN · OK · N/A(**미검증** — 자료가 없어 검사하지 못했다. ⛔통과가 아니다).

사용:
    python3 tmp/_audit/check_prereg.py tmp/T11_EXTACC_2026-09-03                 # 작업트리
    python3 tmp/_audit/check_prereg.py tmp/T11_EXTACC_2026-09-03 --rev 18216fc^   # 과거 판본 (git archive 로 물질화)
    python3 tmp/_audit/check_prereg.py --selftest                                 # 정상 픽스처 + 변이 15종
⛔실행은 `systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 -- python3 …` 로 감싼다.
   이 검사기는 추정기를 돌리지 않고 GPU 를 쓰지 않는다. gates 모듈 임포트는 `blocked_by` 호출을 위해서만 하며
   실패하면 AST 폴백으로 내려간다. 임시 파일은 `CHECK_PREREG_TMP` 환경변수 디렉터리(없으면 시스템 임시)에 만든다.

★보지 않는 것 (= 이 검사기로 "검사했다" 고 말할 수 없는 것)
  · **효과를 판정에서 쟀는가**(A2 §9.1 클램프 예측 반전) — 의미론. 체크리스트 M-항목.
  · **팔 단위 키 누락**(A2 §9.2 `t_origin`) — 정적 키 집합은 합집합이다. 배치 게이트(`G-TORIGIN`)의 일이다.
  · 사전등록 확률의 **산술 정합**(단일 q 이항모형에서 동시에 성립 가능한가 — A2 §2 가 잡은 것) — 표의 존재만 본다.
  · 항진 후보 (a) 는 `G.C`/숫자 리터럴의 **문법적 동일성**만 잡는다. 문자열 선택자(`arm='SHIP'`)는 일부러 뺀다.
    같은 값을 다른 경로로 들고 오면(`LI.M_OBS_CAP` ↔ `G.deployed_cap()`) 못 본다.
  · 동결 임포트 덮음은 **깊이 2** 까지, 파일이 실제로 값을 움직이는지는 판정하지 않는다(WARN 목록을 사람이 읽는다).
  · git 이 없는 사본(픽스처)에서 C10 은 N/A 다. `--rev` 물질화는 진짜 저장소 이력을 쓴다.
  · C9 는 토큰 **부분문자열** 검사다 — 짧은 토큰(`E2`)은 우연히 맞을 수 있다.
"""
import ast
import glob
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
from math import comb

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')          # ⛔GPU 금지

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))        # …/wavesplat/wavesplat (패키지·scripts·tmp 의 뿌리)
ROOT_OVERRIDE = [ROOT]
TMPBASE = os.environ.get('CHECK_PREREG_TMP') or None

GATE_RE = re.compile(r'G-[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*')
LABEL_RE = re.compile(r'[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+')
LABEL_TOKEN_RE = re.compile(r'`([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+)`')
NEG_MARK = ('NO-', 'NOT-', 'NEITHER', 'UNDECIDED', 'UNDETECTED', 'INDISTINGUISHABLE', 'NULL',
            'INSUFFICIENT', 'BLOCKED', 'UNANSWERABLE', 'UNDER-POWERED', '-NULL')


# ────────────────────────────────────────────────────────────────────────────
# 파일 발견
# ────────────────────────────────────────────────────────────────────────────
def _read(p):
    try:
        return open(p, encoding='utf-8', errors='replace').read()
    except OSError:
        return ''


def prereg_docs(d):
    """사전등록 **문서** = `PREREG*.md`(DISCARDED 제외) + `AMENDMENT*.md` 중 결과 **이전** 것.
    ⛔`*POST*` 이름의 개정문은 결과 이후이므로 사전등록 측 검사에서 **뺀다**(A3 가 그렇게 자기 이름에 박았다)."""
    out = []
    for p in sorted(glob.glob(os.path.join(d, 'PREREG*.md'))):
        if 'DISCARD' not in os.path.basename(p).upper():
            out.append(p)
    for p in sorted(glob.glob(os.path.join(d, 'AMENDMENT*.md'))):
        if 'POST' not in os.path.basename(p).upper():
            out.append(p)
    return out


def code_files(d):
    return (sorted(glob.glob(os.path.join(d, 'gates*.py'))),
            sorted(glob.glob(os.path.join(d, 'analyze*.py'))),
            sorted(glob.glob(os.path.join(d, 'run*.py'))))


def _parse(p):
    try:
        return ast.parse(_read(p))
    except SyntaxError:
        return None


def _parse_src(src):
    try:
        return ast.parse(src)
    except SyntaxError:
        return None


def _dump(node):
    return ast.dump(node, annotate_fields=False, include_attributes=False)


def _str_head(node):
    """문자열 상수 또는 f-string 의 **앞머리 리터럴**."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr) and node.values:
        v = node.values[0]
        if isinstance(v, ast.Constant) and isinstance(v.value, str):
            return v.value
    return None


def gate_name(s):
    if not s:
        return None
    m = GATE_RE.search(s.replace('★', '').replace('⚠', '').replace('⛔', ''))
    return m.group(0) if m else None


# ────────────────────────────────────────────────────────────────────────────
# 분석기 게이트 이름 · 조건식 (AST)
# ────────────────────────────────────────────────────────────────────────────
def analyzer_gates_tree(t):
    """`(이름, 조건, 상세)` 튜플 또는 `{'name'/'id'/'gate': …, 'ok'/'pass': …}` dict 에서 (이름, 조건식).
    ⛔독스트링·주석의 게이트 이름은 세지 않는다 — 튜플/dict 의 자리에 있는 것만 (사라진 게이트를 설명하는
    독스트링이 「장식 게이트」오탐을 내는 것을 막는다)."""
    out = []
    if t is None:
        return out

    def _is_str(x):
        return isinstance(x, ast.Constant) and isinstance(x.value, str)

    for n in ast.walk(t):
        # 튜플형 `('G-X …', cond, detail)` — ⛔둘째가 문자열이면 차단 목록 튜플(`('G-A','G-B',…)`)이므로 제외
        if isinstance(n, ast.Tuple) and len(n.elts) >= 2 and not _is_str(n.elts[1]):
            nm = gate_name(_str_head(n.elts[0]))
            if nm:
                out.append((nm, n.elts[1]))
        # 호출형 `A('G-X …', cond, detail)` — T5·S_ENS·E1·M4 가 gates_*.py 안에서 이렇게 만든다
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and len(n.args) >= 2 and not _is_str(n.args[1]):
            nm = gate_name(_str_head(n.args[0]))
            if nm:
                out.append((nm, n.args[1]))
        elif isinstance(n, ast.Dict):
            keys = [k.value if isinstance(k, ast.Constant) else None for k in n.keys]
            for key in ('name', 'id', 'gate'):
                if key in keys:
                    nm = gate_name(_str_head(n.values[keys.index(key)]))
                    if nm:
                        cond = None
                        for ck in ('ok', 'pass', 'pass_', 'passed'):
                            if ck in keys:
                                cond = n.values[keys.index(ck)]
                        out.append((nm, cond))
                    break
    return out


def analyzer_gates(paths):
    out = []
    for p in paths:
        out += analyzer_gates_tree(_parse(p))
    return out


# ────────────────────────────────────────────────────────────────────────────
# gates 모듈 — label() · blocked_by()
# ────────────────────────────────────────────────────────────────────────────
_IMPORT_SEQ = [0]


def import_gates(path):
    """gates 모듈을 **고유 이름**으로 임포트한다(`--rev` 사본·픽스처가 캐시를 공유하지 않게). 실패하면 None."""
    _IMPORT_SEQ[0] += 1
    name = f'_check_prereg_gates_{_IMPORT_SEQ[0]}'
    d = os.path.dirname(os.path.abspath(path))
    added = d not in sys.path
    if added:
        sys.path.insert(0, d)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    except BaseException:                                        # noqa: BLE001  (SystemExit 포함)
        return None
    finally:
        if added and d in sys.path:
            sys.path.remove(d)


def label_functions(tree):
    return [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and 'label' in n.name.lower()
            and any(isinstance(m, ast.Return) for m in ast.walk(n))]


def _labels_in_return(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.IfExp):
        return _labels_in_return(node.body) + _labels_in_return(node.orelse)
    return []


def gates_labels(gpaths):
    """gates 의 `*label*()` 함수가 `return` 하는 문자열 상수 집합 (⛔`None`·변수는 못 본다)."""
    labs = set()
    for p in gpaths:
        t = _parse(p)
        if t is None:
            continue
        for fn in label_functions(t):
            for n in ast.walk(fn):
                if isinstance(n, ast.Return) and n.value is not None:
                    labs |= {s for s in _labels_in_return(n.value) if LABEL_RE.fullmatch(s)}
    return labs


def label_paths(fn):
    """label() 의 각 `return 'LABEL'` 에 이르는 경로 조건.
    반환: ({label: [ [term…], … ]}, 이김변수집합) · term = (이름|'cmp', negated) · ('?', None) = 판정 불가.
    「이김 변수」= `x = w >= T` 처럼 GtE/Gt 비교로 대입된 이름."""
    win = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Compare) and len(n.targets) == 1 \
                and isinstance(n.targets[0], ast.Name) and n.value.ops \
                and isinstance(n.value.ops[0], (ast.GtE, ast.Gt)):
            win.add(n.targets[0].id)

    def terms(test, neg=False):
        if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And) and not neg:
            return sum((terms(v) for v in test.values), [])
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            return terms(test.operand, not neg)
        if isinstance(test, ast.Name):
            return [(test.id, neg)]
        if isinstance(test, ast.Compare) and test.ops:
            if isinstance(test.ops[0], (ast.GtE, ast.Gt)):
                return [('cmp', neg)]
            if isinstance(test.ops[0], (ast.LtE, ast.Lt)):
                return [('cmp', not neg)]
        return [('?', None)]

    def negate(t):
        # 단일 항의 부정만 확정; 연언의 부정은 판정 불가
        if len(t) == 1 and t[0][1] is not None:
            return [(t[0][0], not t[0][1])]
        return [('?', None)]

    out = {}

    def add(label, ctx):
        out.setdefault(label, []).append(list(ctx))

    def walk(body, ctx):
        for st in body:
            if isinstance(st, ast.If):
                t = terms(st.test)
                walk(st.body, ctx + t)
                if st.orelse:
                    walk(st.orelse, ctx + negate(t))
                ctx = ctx + negate(t)               # 이후 문장 = 앞 if 가 return 으로 빠지지 않은 경우
            elif isinstance(st, ast.Return) and st.value is not None:
                if isinstance(st.value, ast.IfExp):
                    t = terms(st.value.test)
                    for lab in _labels_in_return(st.value.body):
                        add(lab, ctx + t)
                    for lab in _labels_in_return(st.value.orelse):
                        add(lab, ctx + negate(t))
                else:
                    for lab in _labels_in_return(st.value):
                        add(lab, ctx)
    walk(fn.body, [])
    return out, win


def is_negative_name(lab):
    return any(m in lab for m in NEG_MARK)


def blocked_by_fn(gpaths):
    """실제 `blocked_by` 를 돌려준다. 임포트 실패 시 AST 로 차단 목록을 뽑아 **긴 이름 우선 접두** 규약을 흉내 낸다.
    반환: (callable(name)->list|None, mode) · mode ∈ {'live','ast',None}."""
    for p in gpaths:
        mod = import_gates(p)
        fn = getattr(mod, 'blocked_by', None) if mod else None
        if callable(fn):
            def live(name, _fn=fn):
                for row in ((name, False, ''), {'name': name, 'ok': False, 'pass_': False, 'detail': ''}):
                    try:
                        return list(_fn([row]))
                    except Exception:                                # noqa: BLE001
                        continue
                return None
            return live, 'live'
    for p in gpaths:
        t = _parse(p)
        if t is None:
            continue
        for fn in ast.walk(t):
            if isinstance(fn, ast.FunctionDef) and fn.name == 'blocked_by':
                block = []
                for n in ast.walk(fn):
                    if isinstance(n, (ast.Tuple, ast.List, ast.Set)):
                        ss = [e.value for e in n.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
                        if ss and all(GATE_RE.fullmatch(s) for s in ss):
                            block += ss
                if block:
                    order = sorted(set(block), key=len, reverse=True)

                    def ast_fn(name, _order=order):
                        nm = name.replace('★', '').strip()
                        return [b for b in _order if nm.startswith(b)][:1]
                    return ast_fn, 'ast'
    return None, None


# ────────────────────────────────────────────────────────────────────────────
# 러너 ↔ 분석기 (항진 후보 · 키)
# ────────────────────────────────────────────────────────────────────────────
def runner_writes(rpaths):
    """러너가 행/meta 에 쓰는 `(key, value_node)` 목록 — `dict(k=v)` · `{'k': v}` · `.update(k=v)`.
    같은 dict 호출 안의 묶음(groups)도 돌려준다(중복 식 탐지용)."""
    pairs, groups = [], []
    for p in rpaths:
        t = _parse(p)
        if t is None:
            continue
        for n in ast.walk(t):
            grp = []
            if isinstance(n, ast.Call) and ((isinstance(n.func, ast.Name) and n.func.id == 'dict')
                                            or (isinstance(n.func, ast.Attribute) and n.func.attr == 'update')):
                grp = [(kw.arg, kw.value) for kw in n.keywords if kw.arg]
            elif isinstance(n, ast.Dict):
                grp = [(k.value, v) for k, v in zip(n.keys, n.values)
                       if isinstance(k, ast.Constant) and isinstance(k.value, str)]
            if grp:
                pairs += grp
                groups.append(grp)
    return pairs, groups


def _get_key(node):
    """`X.get('k')` · `X['k']` → ('X', 'k') (X 가 Name 일 때만). 아니면 None."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'get' \
            and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str) \
            and isinstance(node.func.value, ast.Name):
        return node.func.value.id, node.args[0].value
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) \
            and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
        return node.value.id, node.slice.value
    return None


def analyzer_reads(apaths):
    out = set()
    for p in apaths:
        t = _parse(p)
        if t is None:
            continue
        for n in ast.walk(t):
            gk = _get_key(n)
            if gk and gk[0] in ('r', 'row', 'meta'):
                out.add(gk[1])
    return out


def _is_const_like(node):
    """항진 대조에 쓰는 「상수」 = 모듈 속성(`G.SEED`) 또는 **숫자/불리언** 리터럴. ⛔문자열은 선택자(`arm='SHIP'`)라 뺀다."""
    if isinstance(node, ast.Attribute):
        return True
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool)) and not isinstance(node.value, str)


def analyzer_const_compares(apaths):
    """`X.get('k') == <G.C | 숫자>` 형 비교 → (k, dump(other))."""
    out = []
    for p in apaths:
        t = _parse(p)
        if t is None:
            continue
        for n in ast.walk(t):
            if isinstance(n, ast.Compare) and len(n.comparators) == 1 and isinstance(n.ops[0], (ast.Eq, ast.NotEq)):
                for a, b in ((n.left, n.comparators[0]), (n.comparators[0], n.left)):
                    gk = _get_key(a)
                    if gk and _is_const_like(b):
                        out.append((gk[1], _dump(b)))
    return out


def _is_bare_record_test(cond):
    """게이트 조건 전체가 `X.get('k')` · `bool(X.get('k'))` · `not X.get('k')` · `X['k']` 인가 → 키 또는 None."""
    if cond is None:
        return None
    node = cond
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        node = node.operand
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'bool' and node.args:
        node = node.args[0]
    gk = _get_key(node)
    return gk[1] if gk else None


# ────────────────────────────────────────────────────────────────────────────
# 임포트 경로 → 파일 (동결 덮음)
# ────────────────────────────────────────────────────────────────────────────
def _mod_to_path(root, mod, name=None):
    cands = []
    if mod:
        parts = mod.split('.')
        if name:
            cands.append(os.path.join(root, *parts, name + '.py'))
        cands.append(os.path.join(root, *parts) + '.py')
        cands.append(os.path.join(root, *parts, '__init__.py'))
    elif name:
        cands.append(os.path.join(root, 'scripts', name + '.py'))
        cands.append(os.path.join(root, name + '.py'))
    for c in cands:
        if os.path.isfile(c):
            return os.path.realpath(c)
    return None


def imports_of(path, root):
    """이 파일이 (모듈·함수 어디서든) 임포트하는 **저장소 안** 파일들."""
    t = _parse(path)
    out = set()
    if t is None:
        return out
    for n in ast.walk(t):
        if isinstance(n, ast.Import):
            for a in n.names:
                p = _mod_to_path(root, a.name) or _mod_to_path(root, None, a.name.split('.')[0])
                if p:
                    out.add(p)
        elif isinstance(n, ast.ImportFrom) and not n.level and n.module:
            base = _mod_to_path(root, n.module)
            for a in n.names:
                p = _mod_to_path(root, n.module, a.name)
                if p:
                    out.add(p)
                elif base:
                    out.add(base)
    return out


def frozen_entries(d):
    """`FROZEN*.txt` → {realpath|raw: (hash, raw)} · 미해결 md5 경로 · 64-hex 지문 줄 수."""
    ent, unresolved, sha_lines = {}, [], 0
    for ff in sorted(glob.glob(os.path.join(d, 'FROZEN*.txt'))):
        for ln in open(ff, encoding='utf-8', errors='replace'):
            if ln.startswith('#'):
                continue
            parts = ln.split()
            if len(parts) != 2 or not re.fullmatch(r'[0-9a-f]{32}|[0-9a-f]{64}', parts[0]):
                continue
            raw = parts[1]
            if len(parts[0]) == 64:
                sha_lines += 1
                ent[raw] = (parts[0], raw)
                continue
            p = None
            for base in (d, os.path.dirname(d), os.path.dirname(os.path.dirname(d)), ROOT_OVERRIDE[0]):
                q = os.path.normpath(os.path.join(base, raw))
                if os.path.isfile(q):
                    p = os.path.realpath(q)
                    break
            if p:
                ent[p] = (parts[0], raw)
            else:
                ent[raw] = (parts[0], raw)
                unresolved.append(raw)
    return ent, unresolved, sha_lines


# ────────────────────────────────────────────────────────────────────────────
# 사전등록 문서 파싱
# ────────────────────────────────────────────────────────────────────────────
def md_tables(text):
    """마크다운 표 → [ [row cells…], … ] (헤더 포함, 구분선 제외)."""
    tables, cur = [], []
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith('|'):
            cells = [c.strip() for c in s.strip('|').split('|')]
            if all(re.fullmatch(r':?-{2,}:?', c) for c in cells if c):
                continue
            cur.append(cells)
        elif cur:
            tables.append(cur)
            cur = []
    if cur:
        tables.append(cur)
    return tables


def prereg_gate_names(text):
    """표의 **첫 칸**에 적힌 `G-…` 이름(`check_gate_coverage.prereg_gates` 와 같은 규약)."""
    out = set()
    for tb in md_tables(text):
        for row in tb:
            if row:
                nm = gate_name(row[0])
                if nm:
                    out.add(nm)
    return out


def label_tables(text):
    """헤더에 '라벨'/'label' 이 있는 표."""
    return [tb for tb in md_tables(text) if tb and re.search(r'라벨|label', ' '.join(tb[0]), re.I)]


def prereg_labels_strict(text):
    """라벨 표의 행에 적힌 백틱 대문자-대시 토큰."""
    out = set()
    for tb in label_tables(text):
        for row in tb[1:]:
            for c in row:
                out |= set(LABEL_TOKEN_RE.findall(c))
    return {x for x in out if not x.startswith('G-')}


def prereg_labels_broad(text):
    """문서 어디든 백틱으로 적힌 대문자-대시 토큰(라벨이 아닌 배치코드·팔 이름도 섞인다 — **포함 여부**에만 쓴다)."""
    return {x for x in LABEL_TOKEN_RE.findall(text) if not x.startswith('G-')}


def sign_tail_le(k, n):
    return sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n


def sign_tail_ge(k, n):
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


def prereg_n(text):
    m = re.search(r'PRIMARY[^\n]*?\bn\s*=\s*(\d+)', text) or re.search(r'\bn\s*=\s*(\d+)\b', text)
    return int(m.group(1)) if m else None


# ────────────────────────────────────────────────────────────────────────────
# 검사
# ────────────────────────────────────────────────────────────────────────────
class R:
    def __init__(self):
        self.rows = []

    def add(self, cid, status, msg):
        self.rows.append((cid, status, msg))

    def status(self, cid):
        return [s for c, s, _ in self.rows if c == cid]


def _git(args, cwd):
    return subprocess.run(['git'] + args, cwd=cwd, capture_output=True, text=True)


def check(d, git_rev=None, git_relpath=None):
    d = os.path.abspath(d.rstrip(os.sep))
    res = R()
    docs = prereg_docs(d)
    text = '\n'.join(_read(p) for p in docs)
    gpaths, apaths, rpaths = code_files(d)
    if not docs:
        res.add('C0', 'FAIL', 'PREREG*.md 없음 — 이하 전부 판정 불가')
        return res
    if not gpaths or not apaths:
        res.add('C0', 'FAIL', f'gates*.py {len(gpaths)} · analyze*.py {len(apaths)} — 판정 코드가 없다 (check_batch_artifacts B·C)')
    gsrc = '\n'.join(_read(p) for p in gpaths)
    asrc = '\n'.join(_read(p) for p in apaths)
    rsrc = '\n'.join(_read(p) for p in rpaths)
    amend_text = '\n'.join(_read(p) for p in docs if 'AMENDMENT' in os.path.basename(p).upper())

    # ── C1 MDE ────────────────────────────────────────────────────────────
    mde_lines = [ln for ln in text.splitlines()
                 if re.search(r'\bMDE\b|\bq[_ ]?80\b|등가\s*한계|equivalence\s+margin', ln)
                 and re.search(r'0\.\d+|\d+\s*/\s*\d+', ln)]
    g_has_const = bool(re.search(r'^\s*\w*(MDE|Q80|EQUIV)\w*\s*(,\s*\w+)*\s*=', gsrc, re.M))
    if not mde_lines:
        res.add('C1', 'FAIL', 'MDE / q80 / 등가한계 가 **수치와 함께** 사전등록 문서에 없다 (A2 §3: "배치 전체에 0회")')
    elif not g_has_const:
        res.add('C1', 'WARN', f'문서에는 있으나({len(mde_lines)}줄) gates 에 MDE/Q80 상수가 없다 — 손계산 값이 verdict 로 안 들어간다')
    else:
        res.add('C1', 'OK', f'문서 {len(mde_lines)}줄 · gates 상수 있음')

    # ── C2 PRED · C3 LABELS ───────────────────────────────────────────────
    glabels = gates_labels(gpaths)
    strict = prereg_labels_strict(text)
    broad = prereg_labels_broad(text)
    if not glabels:
        res.add('C2', 'N/A', 'gates 에 문자열 라벨을 return 하는 *label* 함수가 없다 — 라벨 확률 검사 불가')
        res.add('C3', 'N/A', '동상')
    else:
        have, miss = [], []
        for lab in sorted(glabels):
            num = r'(?:\b0\.\d+\b|\b\d{1,3}(?:\.\d+)?\s*%)'
            pat = re.compile(re.escape(lab) + r'[^\n|]{0,40}?' + num + r'|' + num + r'[^\n|]{0,40}?' + re.escape(lab))
            (have if pat.search(text) else miss).append(lab)
        if not have:
            res.add('C2', 'FAIL', f'gates 라벨 {sorted(glabels)} 중 **어느 것에도** 사전등록 확률이 없다 (A2 §2 · feedback_prereg_needs_power…)')
        elif miss:
            res.add('C2', 'WARN', f'확률 없는 라벨 {miss} (A3 §5 부류: 음성 라벨 둘을 하나로 뭉갬)')
        else:
            res.add('C2', 'OK', f'{len(have)}/{len(glabels)} 라벨에 확률')
        fn_prob = re.search(r'def\s+\w*(label_prob|power|p_ge|prob_label|label_dist)\w*\s*\(', gsrc + '\n' + asrc)
        if fn_prob:
            res.add('C2b', 'OK', f'코드 계산 `{fn_prob.group(0).strip()}`')
        else:
            res.add('C2b', 'WARN', '라벨 확률/검정력을 계산하는 함수(label_prob*/power*/p_ge)가 gates·analyze 에 없다 — 표가 손계산이다')
        only_g = sorted(glabels - broad)
        only_p = sorted(strict - glabels)
        if only_g:
            res.add('C3', 'FAIL', f'gates.label() 이 내는데 사전등록 문서 어디에도 없는 라벨 {only_g}'
                    + (f' · 표에만 있는 라벨 {only_p}' if only_p else ''))
        elif only_p:
            res.add('C3', 'WARN', f'사전등록 표에 있으나 gates 가 낼 수 없는 라벨 {only_p} — 개정으로 대체됐는지 확인')
        else:
            res.add('C3', 'OK', f'라벨 집합 일치 {sorted(glabels)}')

    # ── C4 POSLABEL ───────────────────────────────────────────────────────
    n = prereg_n(text)
    c4 = []
    # ⛔이항 귀무(q=.5)는 **부호검정을 선언한 배치**에만 적용한다 — 첫 census 에서 T10 의 "≥ 8/12 파일이 물리 기준 충족"
    #   (파일당 귀무가 .5 가 아니다)에 귀무 확률을 붙여 거짓 FAIL 을 냈다. 미선언이면 표의 수치 검사는 건너뛴다.
    sign_test = bool(re.search(r'부호검정|sign[\s-]?test', text, re.I))
    for tb in (label_tables(text) if sign_test else []):
        for row in tb[1:]:
            labs = [x for c in row for x in LABEL_TOKEN_RE.findall(c) if not x.startswith('G-')]
            if not labs:
                continue
            les = [int(x) for c in row for x in re.findall(r'(?:≤|<=)\s*(\d+)', c)]
            # `≥ 8/12` 처럼 칸 안에 분모가 있으면 그것이 그 행의 n 이다
            kn = [(int(a), int(b)) for c in row for a, b in re.findall(r'(?:≥|>=)\s*(\d+)\s*/\s*(\d+)', c)]
            # `(?!\d)` 가 없으면 `≥ 10/12` 에서 역추적으로 `1` 을 잡는다(첫 census 에서 E1·M4 에 "≥1" 거짓 FAIL)
            ges = [int(x) for c in row for x in re.findall(r'(?:≥|>=)\s*(\d+)(?!\d)(?!\s*/)', c)]
            for lab in labs:
                for k, n_row in kn:
                    p_ = sign_tail_ge(k, n_row)
                    if p_ > 0.05:
                        c4.append(('FAIL', f'`{lab}` 의 이김 문턱 ≥{k}/{n_row} 는 P(W≥{k}|.5)={p_:.4f} > 0.05 — 검정이 아니다'))
                if ges and n:
                    for k in ges:
                        p_ = sign_tail_ge(k, n)
                        if p_ > 0.05:
                            c4.append(('FAIL', f'`{lab}` 의 이김 문턱 ≥{k} 는 n={n} 에서 P(W≥{k}|.5)={p_:.4f} > 0.05 — 검정이 아니다'))
                if les and not ges and not kn and not is_negative_name(lab):
                    p_ = sign_tail_le(max(les), n) if n else None
                    if p_ is None or p_ > 0.05:
                        detail = f'P(W≤{max(les)}|n={n},.5)={p_:.4f}' if p_ is not None else 'n 미상'
                        if lab in glabels:
                            c4.append(('FAIL', f'`{lab}` 은 기각-실패 다리(≤{max(les)}, {detail})에 붙은 **양성 이름** 라벨이고 gates 가 그대로 낸다 (A2 §1 BEYOND-REACH 부류)'))
                        elif lab in amend_text:
                            c4.append(('WARN', f'`{lab}` 은 기각-실패 다리({detail})의 양성 라벨 — 개정문이 언급하므로 대체된 것으로 본다(표는 PREREG 에 남아 있다)'))
                        else:
                            c4.append(('FAIL', f'`{lab}` 은 기각-실패 다리({detail})의 양성 라벨인데 gates 도 안 내고 개정문도 언급하지 않는다'))
    for p in gpaths:
        t = _parse(p)
        if t is None:
            continue
        for fn in label_functions(t):
            paths_, win = label_paths(fn)
            for lab, ctxs in paths_.items():
                if is_negative_name(lab) or not LABEL_RE.fullmatch(lab):
                    continue
                for ctx in ctxs:
                    pos = any((nm == 'cmp' or nm in win) and ng is False for nm, ng in ctx)
                    unk = any(ng is None for _, ng in ctx)
                    if not pos and not unk:
                        c4.append(('FAIL', f'{os.path.basename(p)}:{fn.name}() 의 `{lab}` 경로에 이김 검정(≥) 항이 없다 — 기각 실패에 양성 이름'))
                    elif not pos:
                        c4.append(('WARN', f'{os.path.basename(p)}:{fn.name}() 의 `{lab}` 경로를 정적으로 판정 못 함(복합 조건) — 사람이 볼 것'))
    if c4:
        for st, msg in c4:
            res.add('C4', st, msg)
    elif glabels or n:
        res.add('C4', 'OK', '양성 이름 라벨은 전부 이김 검정 다리에 있다' + ('' if sign_test else ' (⚠부호검정 미선언 — 표의 귀무 확률은 미검사, label() 구조만 봤다)'))
    else:
        res.add('C4', 'N/A', '라벨·n 을 못 찾아 미검증')

    # ── C5 DECOR · C8 GATESIMPL ───────────────────────────────────────────
    # ★게이트 튜플은 analyze_*.py 뿐 아니라 gates_*.py 에도 산다(첫 census 에서 4배치가 "구현 0" 으로 오판됐다)
    ag = analyzer_gates(gpaths + apaths)
    anames = sorted({nm for nm, _ in ag})
    bb, mode = blocked_by_fn(gpaths)
    if not anames:
        res.add('C5', 'N/A', '분석기에서 `(이름, 조건, …)` 형 게이트를 못 찾았다 — 장식 게이트 검사 불가')
    elif bb is None:
        res.add('C5', 'N/A', 'gates 에 blocked_by() 가 없거나 차단 목록을 못 뽑았다')
    else:
        deco = [nm for nm in anames if not bb(nm)]
        if deco:
            res.add('C5', 'FAIL', f'[{mode}] 차단 목록에 안 잡히는 게이트 {deco} — FAIL 해도 라벨을 못 막는 **장식 게이트** (A2 §7 G-COV · T5-ENS blocked_by)')
        else:
            res.add('C5', 'OK', f'[{mode}] 분석기 게이트 {len(anames)}개 전부 blocked_by 가 잡는다')
    pg = prereg_gate_names(text)
    if not pg:
        res.add('C8', 'N/A', '사전등록 표에 G-… 게이트가 없다')
    else:
        missing = sorted(g for g in pg if g not in anames)
        if missing:
            res.add('C8', 'FAIL', f'사전등록 표의 게이트 {missing} 가 분석기 소스에 없다 (사전등록 {len(pg)} · 구현 {len(anames)})')
        else:
            res.add('C8', 'OK', f'사전등록 {len(pg)} ⊆ 구현 {len(anames)}')

    # ── C6 TAUTO ──────────────────────────────────────────────────────────
    cmp_ = analyzer_const_compares(apaths)
    if rpaths:
        pairs, groups = runner_writes(rpaths)
        wmap = {}
        for k, v in pairs:
            if _is_const_like(v):
                wmap.setdefault(k, set()).add(_dump(v))
        hits = sorted({f'{k} ← {dump_[:70]}' for k, dump_ in cmp_ if dump_ in wmap.get(k, ())})
        if hits:
            res.add('C6a', 'FAIL', f'러너가 상수로 쓰고 분석기가 **같은 상수**와 비교 (실패 불가): {hits} (A2 §7 G-ENV · A3 §2 seed)')
        else:
            res.add('C6a', 'OK', f'상수 왕복 없음 (분석기 상수 비교 {len(cmp_)}건 대조)')
        dups = []
        for grp in groups:
            seen = {}
            for k, v in grp:
                dv = _dump(v)
                if isinstance(v, (ast.Call, ast.Attribute, ast.BinOp)) and dv in seen:
                    dups.append(f'{seen[dv]}=={k}')
                seen.setdefault(dv, k)
        if dups:
            res.add('C6c', 'WARN', f'같은 dict 에 같은 식을 두 키로 기록 {sorted(set(dups))} — 하나를 둘로 센 것 (A3 §0 threads/threads_readback)')
        else:
            res.add('C6c', 'OK', '중복 되읽기 없음')
    else:
        res.add('C6a', 'N/A', '러너(run*.py)가 없어 러너↔분석기 상수 왕복을 대조할 수 없다 (⛔통과 아님)')
        res.add('C6c', 'N/A', '동상')
    bare = sorted({f'{nm}←{k}' for nm, cond in ag if (k := _is_bare_record_test(cond))})
    if bare:
        res.add('C6b', 'FAIL', f'게이트 조건 전체가 기록값의 존재/불리언 그대로 {bare} — 러너가 리터럴을 쓰면 항진, 키가 없으면 통과 (A1 §4 G-SIZE·G-DISJOINT·G-ENV)')
    elif ag:
        res.add('C6b', 'OK', '기록값 존재/불리언만으로 판정하는 게이트 없음')

    # ── C7 FROZEN ─────────────────────────────────────────────────────────
    ent, unresolved, sha_lines = frozen_entries(d)
    if not ent:
        res.add('C7', 'FAIL', 'FROZEN*.txt 가 없거나 해시 줄이 0개 (A2 §8: 결과 0행 시점에 동결)')
    else:
        root = ROOT_OVERRIDE[0]
        own = {os.path.realpath(p) for p in gpaths + apaths + rpaths}
        direct = set()
        for p in gpaths + apaths + rpaths:
            direct |= imports_of(p, root)
        direct -= own
        trans = set()
        for p in direct:
            trans |= imports_of(p, root)
        trans -= direct | own
        miss_d = sorted(os.path.relpath(p, root) for p in direct if p not in ent)
        miss_t = sorted(os.path.relpath(p, root) for p in trans if p not in ent)
        if miss_d:
            res.add('C7', 'FAIL', f'러너·게이트·분석기가 **직접** 임포트하는 저장소 파일이 동결 목록에 없다 {miss_d} (A2 §7: varpro·spectral_gaussian 누락 부류)')
        else:
            res.add('C7', 'OK', f'직접 임포트 {len(direct)}개 전부 동결' + (f' (미해결 경로 {unresolved})' if unresolved else ''))
        if miss_t:
            res.add('C7t', 'WARN', f'깊이-2 전이 임포트 중 미동결 {miss_t} — 값을 움직이는 파일인지 사람이 판정 (A2 §7 은 varpro 가 그랬다)')
        learned = any(os.path.basename(p) == 'learned_init.py' for p in direct | trans | set(ent))
        if learned and sha_lines == 0:
            res.add('C7s', 'FAIL', '학습 팔(learned_init) 경로인데 체크포인트 지문(64-hex sha256) 줄이 동결 목록에 없다 (A2 §7 "확산 체크포인트 누락")')
        elif learned:
            res.add('C7s', 'OK', f'체크포인트 지문 {sha_lines}줄')
        if not git_rev:
            rs = [f for f in glob.glob(os.path.join(d, 'results*.jsonl')) if 'SMOKE' not in os.path.basename(f)]
            fz = glob.glob(os.path.join(d, 'FROZEN*.txt'))
            if rs and fz and min(os.path.getmtime(f) for f in fz) > min(os.path.getmtime(f) for f in rs):
                res.add('C7m', 'WARN', '가장 이른 FROZEN 이 가장 이른 결과보다 늦다(mtime) — 결과 0행 동결이 아닐 수 있다 (checkout 이 mtime 을 갱신했을 수도 있다)')

    # ── C9 DISCLOSE ───────────────────────────────────────────────────────
    vsrc = '\n'.join(_read(p) for p in glob.glob(os.path.join(d, 'verdict*.json')))
    tokens, sect = set(), ''
    for ln in text.splitlines():
        if ln.startswith('#'):
            sect = ln
            continue
        # ⛔표 행은 공시가 아니다 — 첫 실행에서 `§5 차단 게이트 (⛔사후 완화 금지 …)` 표의 `G-WRAPPER` 행이
        #   토큰(`prov['is_ensemble_wrapper'] is True`·`SequentialUpdater`)을 내어 **통과 상태의 T11 을 FAIL** 시켰다
        #   (= 검사기가 「있는 것을 없다고」 한 사례). 산문·불릿의 ⛔ 항목만 센다.
        if ln.lstrip().startswith('|'):
            continue
        if re.search(r'금지|범위|공시', sect) and '⛔' in ln:
            if ln.count('`') % 2:                      # 닫히지 않은 백틱(여러 줄 경로) — 그 뒤는 버린다
                ln = ln[:ln.rfind('`')]
            tokens |= {t for t in re.findall(r'`([^`]+)`', ln) if len(t) <= 40}
            tokens |= {w for w in re.findall(r'\b[A-Za-z][A-Za-z0-9_]{3,}\b', re.sub(r'`[^`]*`', ' ', ln))
                       if w.lower() not in ('blind', 'true', 'false', 'none', 'primary')}
    # ★공시 토큰만 남긴다: 대문자 라벨/팔/배치코드(`E2`·`BAND-MISMATCH`) 와 소문자 낱말(`solver`).
    #   경로·글롭·확장자·snake_case 함수명·혼합대소문자(`C1c`)는 **공시 내용이 아니라 참조**라 뺀다
    #   (첫 census 오탐: `results*.jsonl`·`RESULT.md`·`check_batch_artifacts`·`matlab`(경로 조각)).
    tokens = {t for t in tokens if not t.startswith('G-')
              and (re.fullmatch(r'[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*', t) or re.fullmatch(r'[a-z]{4,}', t))}
    if not tokens:
        res.add('C9', 'N/A', '사전등록에 §금지/범위/공시 절의 ⛔ 항목이 없거나 공시 토큰을 못 뽑았다')
    else:
        miss = sorted(t for t in tokens if t not in asrc and t.lower() not in asrc.lower() and t not in vsrc)
        strong = [t for t in miss if len(t) >= 3]
        if strong:
            res.add('C9', 'FAIL', f'사전등록 ⛔ 의무 공시 토큰이 분석기(공시)에 없다 {miss} (A3 §1 = 사전등록 위반 · N−1)')
        elif miss:
            res.add('C9', 'WARN', f'짧은 코드 토큰만 누락 {miss} — 공시 문장이 떨어졌는지 사람이 확인 (T11 A3 §1 의 `E2` 가 이 부류였다)')
        else:
            res.add('C9', 'OK', f'⛔ 토큰 {len(tokens)}개 전부 분석기에 있음')

    # ── C10 GATELOSS (git) ────────────────────────────────────────────────
    try:
        gtop = _git(['rev-parse', '--show-toplevel'], ROOT)
        if gtop.returncode != 0:
            raise RuntimeError('no git')
        gtop = gtop.stdout.strip()
        rel = git_relpath or os.path.relpath(d, gtop)
        prior, found = set(), False
        for ap in apaths:
            apath = os.path.join(rel, os.path.basename(ap)).replace(os.sep, '/')
            log = _git(['log', '--format=%H', git_rev or 'HEAD', '--', apath], gtop).stdout.split()
            if not log:
                continue
            found = True
            for h in (log[1:] if git_rev else log):          # --rev: 물질화된 판 자신은 제외
                src = _git(['show', f'{h}:{apath}'], gtop).stdout
                prior |= {nm for nm, _ in analyzer_gates_tree(_parse_src(src))}
        if not found:
            res.add('C10', 'N/A', 'git 이력이 없다(사본/미추적) — 소실 게이트 검사 불가')
        else:
            lost = sorted(prior - set(anames))
            undoc = [g for g in lost if g not in text]
            doc = [g for g in lost if g in text]
            if undoc:
                res.add('C10', 'WARN', f'이전 판본에 있던 게이트가 사라졌고 사전등록 문서 어디에도 그 이름이 없다 {undoc} (A3 §3 G-DENSITY 부류)'
                        + (f' · 언급 있음 {doc}' if doc else ''))
            elif lost:
                res.add('C10', 'OK', f'소실 게이트 {lost} 는 문서가 언급한다')
            else:
                res.add('C10', 'OK', f'소실 게이트 없음 (이전 판본 이름 {len(prior)}개 대조)')
    except Exception as e:                                           # noqa: BLE001
        res.add('C10', 'N/A', f'git 대조 불가 ({type(e).__name__})')

    # ── C11 SMOKE ─────────────────────────────────────────────────────────
    if re.search(r'smoke|스모크', rsrc, re.I) or glob.glob(os.path.join(d, '*SMOKE*')) or re.search(r'스모크|smoke', text, re.I):
        res.add('C11', 'OK', '스모크 경로/산출물/언급 있음')
    else:
        res.add('C11', 'WARN', '스모크 흔적 없음 — 긴 배치 전 몇 케이스를 먼저 돌려라 (feedback_smoke_before_long_runs)')

    # ── C12 PROV ──────────────────────────────────────────────────────────
    if rpaths and re.search(r'make_updater|SequentialUpdater|\.initialize\(', rsrc):
        if 'arm_provenance' not in rsrc:
            res.add('C12', 'FAIL', '러너가 추정기를 만들면서 `arm_provenance()` 를 행에 넣지 않는다 (CLAUDE.md: E1-EXTBASE 빈 dict 팔 사고)')
        elif not re.search(r"\bprov\b|resolved_init_method", asrc):
            res.add('C12', 'WARN', '러너는 provenance 를 넣는데 분석기가 읽지 않는다')
        else:
            res.add('C12', 'OK', '러너 기록 + 분석기 되읽기')
    elif not rpaths:
        res.add('C12', 'N/A', '러너 없음')
    else:
        res.add('C12', 'N/A', '러너가 추정기를 만들지 않는다(재집계/오라클 배치)')

    # ── C13 MULT ──────────────────────────────────────────────────────────
    secs = {s.upper() for s in re.findall(r'\bSEC-[A-Z]\b|\bsecondary\b', text, re.I)}
    if len(secs) >= 2:
        if re.search(r'Holm|Bonferroni|무보정|보정하지|uncorrected|FWER|다중비교|다중검정', text):
            res.add('C13', 'OK', f'부차 통계량 {sorted(secs)} · 보정/무보정 공시 있음')
        else:
            res.add('C13', 'WARN', f'부차 통계량 {sorted(secs)} 인데 Holm/Bonferroni/무보정 공시가 0회 (A2 §4)')
    else:
        res.add('C13', 'N/A', '부차 통계량이 둘 미만')

    # ── C14 BLIND ─────────────────────────────────────────────────────────
    if re.search(r'blind|노출', text, re.I):
        res.add('C14', 'OK', 'blind/노출 상태 언급 있음')
    else:
        res.add('C14', 'WARN', 'blind/사전노출 상태를 적지 않았다')

    # ── C15 PRIMARY ───────────────────────────────────────────────────────
    #   키워드 부재만 FAIL. 방향·n 은 표기 규약이 배치마다 달라(표 안의 `≥`, "12개") 부재 시 WARN.
    prim = [ln for ln in text.splitlines() if re.search(r'PRIMARY|주\s*가설|1차\s*가설|\bH1\b', ln)]
    dir_re = r'[<>≤≥]|작을수록|클수록|낮을수록|높을수록|lower|higher'
    has_dir = any(re.search(dir_re, ln) for ln in prim) or \
        any(re.search(dir_re, ' '.join(c for row in tb[1:] for c in row)) for tb in label_tables(text))
    has_n = n is not None or bool(re.search(r'\b\d+\s*(개|블록|레코드|에피소드|파일|짝|셀)\b', '\n'.join(prim)))
    if not prim:
        res.add('C15', 'FAIL', 'PRIMARY(주 가설) 를 이름으로 못 박은 줄이 없다 (feedback_prespecify_aggregation_and_direction)')
    elif not (has_dir and has_n):
        res.add('C15', 'WARN', f'PRIMARY 줄 {len(prim)} 은 있으나 ' + ('방향 기호 없음 ' if not has_dir else '') + ('n 없음' if not has_n else ''))
    else:
        res.add('C15', 'OK', f'PRIMARY 줄 {len(prim)} · 방향 · n={n}')

    # ── C16 SEED ──────────────────────────────────────────────────────────
    if re.search(r'시드|seed', text, re.I):
        res.add('C16', 'OK', '시드 언급 있음 (⚠어느 축인지는 사람이 확인)')
    else:
        res.add('C16', 'WARN', '시드(축) 언급 없음')

    # ── C17 KEYS ──────────────────────────────────────────────────────────
    if rpaths:
        wkeys = {k for k, _ in runner_writes(rpaths)[0]}
        rkeys = analyzer_reads(apaths)
        miss = sorted(k for k in rkeys if k not in wkeys)
        if miss:
            res.add('C17', 'WARN', f'분석기가 읽는데 러너가 어디서도 안 쓰는 키 {miss} (⚠팔 단위 누락은 이 검사가 못 본다)')
        else:
            res.add('C17', 'OK', f'분석기 읽기 키 {len(rkeys)}개 전부 러너가 쓴다')
    else:
        res.add('C17', 'N/A', '러너 없음')

    # ── C18 BELT ──────────────────────────────────────────────────────────
    belt = None
    for p in apaths:
        t = _parse(p)
        if t is None:
            continue
        for nd in ast.walk(t):
            if not isinstance(nd, ast.Assign):
                continue
            v = nd.value
            calls = [c for c in ast.walk(v) if isinstance(c, ast.Call) and
                     ((isinstance(c.func, ast.Attribute) and 'label' in c.func.attr.lower()) or
                      (isinstance(c.func, ast.Name) and 'label' in c.func.id.lower()))]
            if not calls:
                continue
            if isinstance(v, ast.IfExp) and 'block' in _dump(v.test).lower():
                belt = True
            elif belt is None:
                belt = False
    if belt is None:
        res.add('C18', 'N/A', '분석기에서 label(...) 호출 대입을 못 찾았다')
    elif belt:
        res.add('C18', 'OK', '라벨 인쇄가 blocked 에 매달려 있다')
    else:
        res.add('C18', 'WARN', '라벨 인쇄가 blocked 조건 없이 대입된다 — FAIL 게이트가 있어도 라벨이 나온다')
    return res


# ────────────────────────────────────────────────────────────────────────────
# 출력 · rev 물질화
# ────────────────────────────────────────────────────────────────────────────
def report(res, title):
    print(f'══ check_prereg :: {title}')
    counts = {'FAIL': 0, 'WARN': 0, 'OK': 0, 'N/A': 0}
    for cid, st, msg in res.rows:
        counts[st] += 1
        print(f'  [{st:4s}] {cid:4s} {msg}')
    print(f'  요약: FAIL {counts["FAIL"]} · WARN {counts["WARN"]} · OK {counts["OK"]} · N/A {counts["N/A"]}'
          f'   (⛔N/A 는 미검증이지 통과가 아니다)')
    return 1 if counts['FAIL'] else 0


def materialize(batch_dir, rev):
    gtop = _git(['rev-parse', '--show-toplevel'], ROOT).stdout.strip()
    rel = os.path.relpath(os.path.abspath(batch_dir), gtop)
    tmpd = tempfile.mkdtemp(prefix='check_prereg_rev_', dir=TMPBASE)
    ar = subprocess.run(['git', 'archive', rev, '--', rel], cwd=gtop, capture_output=True, check=True)
    subprocess.run(['tar', '-x', '-C', tmpd], input=ar.stdout, check=True)
    return os.path.join(tmpd, rel), rel, tmpd


# ────────────────────────────────────────────────────────────────────────────
# 자가시험 — 정상 픽스처 + 변이 (⛔"문자열이 있다" 만 보고 통과시키는 항진 방지)
# ────────────────────────────────────────────────────────────────────────────
GOOD_PREREG = """# 픽스처 사전등록

## §2 결정표
**PRIMARY**: `A < ZERO` 짝 부호검정, `n = 24`. **≥17** 이면 유의.

| `A` | `B` | 라벨 |
|---|---|---|
| ≥17 | ≤12 | `A-WINS` |
| 그 밖 | | `NEITHER-DETECTED` |

## §3 예측
q_a = 0.75, q_b = 0.55 · `A-WINS` 0.70 · `NEITHER-DETECTED` 0.30
MDE: 검정력 0.80 → q80 = 0.760

SEC-A `B<ZERO` · SEC-B 층. p 는 **무보정**.
⚠완전 blind 아님. 시드 = 20260904 + i.

## §5 게이트
| # | 게이트 |
|---|---|
| `G-ONE` | 되읽기 |
| `G-TWO` | 되읽기 |
| `G-BLOCK` | 자가진단 |

## §6 ⛔범위와 금지
- ⛔`E2` 와 비교 금지 · solver 미명시.
"""

GOOD_GATES = """import os
HERE = os.path.dirname(os.path.abspath(__file__))
N_FILES = 24
THRESH_WIN = 17
MDE_Q80 = 0.760
PRED_Q = (0.75, 0.55)
SEED = 20260904


def label(w_a, w_b):
    a_win = w_a >= THRESH_WIN
    b_win = w_b >= THRESH_WIN
    if a_win and not b_win:
        return 'A-WINS'
    return 'NEITHER-DETECTED'


def label_probs(qa, qb):
    return {'A-WINS': qa * (1 - qb), 'NEITHER-DETECTED': 1 - qa * (1 - qb)}


def blocked_by(rows):
    block = ('G-ONE', 'G-TWO', 'G-BLOCK')
    order = sorted(block, key=len, reverse=True)
    out = []
    for name, ok, _ in rows:
        if ok:
            continue
        nm = name.replace('★', '').strip()
        for b in order:
            if nm.startswith(b):
                out.append(b)
                break
    return out
"""

GOOD_ANALYZE = """import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_x as G


def main():
    rows = [json.loads(x) for x in open(os.path.join(HERE, 'results_x.jsonl'))]
    meta = json.load(open(os.path.join(HERE, 'meta_x.json')))
    A = []
    A.append(('★G-ONE t0 되읽기 == 0 · prov 존재', all(abs(r.get('t0', 1)) < 1e-9 and r.get('prov') is not None for r in rows), ''))
    A.append(('G-TWO 시드 집합 되읽기', {r['seed'] for r in rows} == {G.SEED + i for i in range(1, 25)}, ''))
    probe = G.blocked_by([('★G-ONE 합성', False, '')])
    A.append(('G-BLOCK 자가진단', probe == ['G-ONE'], ''))
    blocked = G.blocked_by(A)
    lab = G.label(1, 1) if not blocked else None
    v = {'label': lab, 'gates': A, 'mde': {'q80': G.MDE_Q80},
         'disclosures': ['⛔E2 와 비교 금지', '⛔solver 미명시'],
         'threads': meta.get('threads')}
    json.dump(v, open(os.path.join(HERE, 'verdict_x.json'), 'w'))
"""

GOOD_RUN = """import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_x as G
from wavesplat.optimizer.sequential_update import make_updater


def main(smoke=False):
    rows = []
    for i in range(1, 25):
        upd = make_updater({})
        rows.append(dict(arm='A', seed=G.SEED + i, t0=float(min([0.0])), prov=upd.arm_provenance()))
    json.dump(dict(threads=int(os.cpu_count()), n=len(rows)), open('meta_x.json', 'w'))
"""


def _write(p, s):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, 'w', encoding='utf-8').write(s)


def _md5(p):
    import hashlib
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def _fixture(base, name, prereg=GOOD_PREREG, gates=GOOD_GATES, analyze=GOOD_ANALYZE, run=GOOD_RUN, frozen=None):
    root = os.path.join(base, name)
    d = os.path.join(root, 'tmp', 'BATCH_X')
    _write(os.path.join(root, 'wavesplat', 'optimizer', 'sequential_update.py'), 'def make_updater(c):\n    return None\n')
    _write(os.path.join(root, 'wavesplat', 'optimizer', 'learned_init.py'), 'M_OBS_CAP = 400\n')
    _write(os.path.join(d, 'PREREG.md'), prereg)
    _write(os.path.join(d, 'gates_x.py'), gates)
    _write(os.path.join(d, 'analyze_x.py'), analyze)
    _write(os.path.join(d, 'run_x.py'), run)
    lines = [f'{_md5(os.path.join(d, f))}  {f}' for f in ('PREREG.md', 'gates_x.py', 'analyze_x.py', 'run_x.py')]
    lines.append(f'{_md5(os.path.join(root, "wavesplat/optimizer/sequential_update.py"))}  wavesplat/optimizer/sequential_update.py')
    if frozen is not None:
        lines = frozen(lines, root, d)
    _write(os.path.join(d, 'FROZEN_md5.txt'), '\n'.join(lines) + '\n')
    return root, d


def selftest():
    base = tempfile.mkdtemp(prefix='check_prereg_selftest_', dir=TMPBASE)
    ok = True
    out = []

    def chk(tag, cond, detail=''):
        nonlocal ok
        ok = ok and bool(cond)
        out.append(f'  {"✅" if cond else "⛔"} {tag}' + (f'  — {detail}' if detail else ''))

    def run(root, d):
        ROOT_OVERRIDE[0] = root
        try:
            return check(d)
        finally:
            ROOT_OVERRIDE[0] = ROOT

    try:
        # [0] 침묵 시험 — 정상 픽스처에서 FAIL/WARN 0 (검사기가 「있는 것을 없다고」 하지 않는가)
        root, d = _fixture(base, 'GOOD')
        r0 = run(root, d)
        bad = [(c, s, m) for c, s, m in r0.rows if s in ('FAIL', 'WARN')]
        chk('[0] 정상 픽스처에 FAIL/WARN 0', not bad, '; '.join(f'{c}:{s}:{m[:60]}' for c, s, m in bad))
        na = [c for c, s, _ in r0.rows if s == 'N/A']
        out.append(f'      (정상 픽스처의 N/A = {na} — git 이력 없음 등, 통과로 세지 않는다)')

        muts = []
        # [1] C1 MDE 삭제
        muts.append(('C1', 'FAIL', dict(prereg=GOOD_PREREG.replace('MDE: 검정력 0.80 → q80 = 0.760', ''))))
        # [2] C2 라벨 확률 삭제
        muts.append(('C2', 'FAIL', dict(prereg=GOOD_PREREG.replace('`A-WINS` 0.70 · `NEITHER-DETECTED` 0.30', '예측 없음'))))
        # [3] C5 장식 게이트 — 분석기가 G-THREE 를 만드는데 차단 목록에 없다
        muts.append(('C5', 'FAIL', dict(analyze=GOOD_ANALYZE.replace(
            "    probe = G.blocked_by", "    A.append(('G-THREE 장식', len(rows) > 0, ''))\n    probe = G.blocked_by"))))
        # [4] C6a 상수 왕복 — 러너 meta 에 seed=G.SEED, 분석기가 == G.SEED
        muts.append(('C6a', 'FAIL', dict(
            run=GOOD_RUN.replace('dict(threads=int(os.cpu_count()), n=len(rows))',
                                 'dict(threads=int(os.cpu_count()), n=len(rows), seed=G.SEED)'),
            analyze=GOOD_ANALYZE.replace("    A.append(('G-BLOCK 자가진단'",
                                         "    A.append(('G-TWO2 시드', meta.get('seed') == G.SEED, ''))\n    A.append(('G-BLOCK 자가진단'"),
            gates=GOOD_GATES.replace("('G-ONE', 'G-TWO', 'G-BLOCK')", "('G-ONE', 'G-TWO', 'G-TWO2', 'G-BLOCK')"))))
        # [5] C7 동결 목록에서 임포트 경로 삭제
        muts.append(('C7', 'FAIL', dict(frozen=lambda ls, root, d: [x for x in ls if 'sequential_update' not in x])))
        # [6] C8 사전등록 표에 유령 게이트
        muts.append(('C8', 'FAIL', dict(prereg=GOOD_PREREG.replace('| `G-BLOCK` | 자가진단 |', '| `G-BLOCK` | 자가진단 |\n| `G-GHOST` | 없음 |'))))
        # [7] C4 기각-실패 다리에 양성 라벨 (표 + 코드 둘 다)
        muts.append(('C4', 'FAIL', dict(
            prereg=GOOD_PREREG.replace('| 그 밖 | | `NEITHER-DETECTED` |', '| ≤12 | ≤12 | `BEYOND-REACH` |\n| 그 밖 | | `NEITHER-DETECTED` |')
            .replace('`A-WINS` 0.70 · `NEITHER-DETECTED` 0.30', '`A-WINS` 0.60 · `BEYOND-REACH` 0.20 · `NEITHER-DETECTED` 0.20'),
            gates=GOOD_GATES.replace("    return 'NEITHER-DETECTED'",
                                     "    if w_a <= 12 and w_b <= 12:\n        return 'BEYOND-REACH'\n    return 'NEITHER-DETECTED'"))))
        # [8] C9 의무 공시 두 건 삭제 (= T11 A3 §1: `E2` 와 `solver` 가 함께 떨어졌다)
        muts.append(('C9', 'FAIL', dict(analyze=GOOD_ANALYZE.replace("'⛔E2 와 비교 금지', '⛔solver 미명시'", "'⛔없음'"))))
        # [9] C6c 같은 식을 두 키로
        muts.append(('C6c', 'WARN', dict(run=GOOD_RUN.replace(
            'dict(threads=int(os.cpu_count()), n=len(rows))',
            'dict(threads=int(os.cpu_count()), threads_readback=int(os.cpu_count()), n=len(rows))'))))
        # [10] C3 라벨 집합 불일치
        muts.append(('C3', 'FAIL', dict(gates=GOOD_GATES.replace("return 'A-WINS'", "return 'OTHER-LABEL'"))))
        # [11] C18 인쇄 벨트 없음
        muts.append(('C18', 'WARN', dict(analyze=GOOD_ANALYZE.replace('lab = G.label(1, 1) if not blocked else None', 'lab = G.label(1, 1)'))))
        # [12] C7s 학습 팔인데 체크포인트 지문 없음
        muts.append(('C7s', 'FAIL', dict(
            run=GOOD_RUN.replace('from wavesplat.optimizer.sequential_update import make_updater',
                                 'from wavesplat.optimizer.sequential_update import make_updater\nfrom wavesplat.optimizer.learned_init import M_OBS_CAP'),
            frozen=lambda ls, root, d: ls + [f'{_md5(os.path.join(root, "wavesplat/optimizer/learned_init.py"))}  wavesplat/optimizer/learned_init.py'])))
        # [13] C6b 게이트 조건이 기록값 그대로
        muts.append(('C6b', 'FAIL', dict(analyze=GOOD_ANALYZE.replace(
            "    A.append(('G-BLOCK 자가진단'", "    A.append(('G-TWO 크기', meta.get('size_ok'), ''))\n    A.append(('G-BLOCK 자가진단'"))))
        # [14] C12 provenance 미기록
        muts.append(('C12', 'FAIL', dict(run=GOOD_RUN.replace(', prov=upd.arm_provenance()', ''))))
        # [15] C15 PRIMARY 방향·n 삭제
        muts.append(('C15', 'FAIL', dict(prereg=GOOD_PREREG
                                         .replace('**PRIMARY**: `A < ZERO` 짝 부호검정, `n = 24`. **≥17** 이면 유의.', '검정은 나중에 정한다.')
                                         .replace('| ≥17 | ≤12 | `A-WINS` |', '| 이김 | 짐 | `A-WINS` |'))))
        # [16] C9 짧은 코드 토큰(`E2`) 하나만 탈락 → WARN (FAIL 아님 — 늑대 방지 등급)
        muts.append(('C9', 'WARN', dict(analyze=GOOD_ANALYZE.replace("'⛔E2 와 비교 금지', ", ''))))

        base_fail = {c for c, s, _ in r0.rows if s == 'FAIL'}
        for i, (cid, want, kw) in enumerate(muts, 1):
            root, d = _fixture(base, f'MUT{i}', **kw)
            r = run(root, d)
            got = r.status(cid)
            others = sorted({c for c, s, _ in r.rows if s == 'FAIL' and not c.startswith(cid[:2])} - base_fail)
            chk(f'[{i:2d}] 변이 → {cid} {want}', want in got and not others,
                f'{cid}={got}' + (f' · 다른 FAIL {others}' if others else ''))
    finally:
        shutil.rmtree(base, ignore_errors=True)
    print('\n'.join(out))
    print('\n' + ('✅자가시험 통과' if ok else '⛔자가시험 실패'))
    return 0 if ok else 1


def main(argv):
    if '--selftest' in argv:
        return selftest()
    args = list(argv[1:])
    rev = None
    if '--rev' in args:
        i = args.index('--rev'); rev = args[i + 1]; del args[i:i + 2]
    if '--root' in args:
        i = args.index('--root'); ROOT_OVERRIDE[0] = os.path.abspath(args[i + 1]); del args[i:i + 2]
    if not args:
        print(__doc__)
        return 2
    rc = 0
    for a in args:
        if rev:
            d, rel, tmpd = materialize(a, rev)
            try:
                rc |= report(check(d, git_rev=rev, git_relpath=rel), f'{os.path.basename(a.rstrip(os.sep))} @ {rev}')
            finally:
                shutil.rmtree(tmpd, ignore_errors=True)
        else:
            rc |= report(check(a), os.path.basename(a.rstrip(os.sep)))
    return rc


if __name__ == '__main__':
    sys.exit(main(sys.argv))
