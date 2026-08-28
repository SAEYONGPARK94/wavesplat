"""M-8b 단일출처 — ★`M8` 동결 432행의 **재분석**. ⛔신규 판정 런 0회.

    python3 tmp/M8b_LADDER_2026-08-28/gates_m8b.py             # 자가진단
    python3 tmp/M8b_LADDER_2026-08-28/gates_m8b.py --refcheck  # 외부 참조를 실제 파일에 대고

────────────────────────────────────────────────────────────────────────────
★**이 파일이 바꾸는 것은 앵커 술어 하나뿐이다.**

`M8` 은 432런을 에러 0 으로 완주했고 게이트 **38 중 37 이 PASS** 했다. 유일한 차단 FAIL 이
`G-ANCHOR-B`(*"R0(BANK, seed 42) == `B-2` 동결 `SHIP` 비트 12/12"*)이고, 진단
(`../M8_LADDER_2026-08-28/AMENDMENT_1.md`)이 그 원인을 **양방향 비트 재현**으로 확정했다:

  · `B-2` 동결 `SHIP` 열은 BS 7 레코드를 **수심 17.0 m** 로 돌았다 — 저장소가
    **2026-08-23 에 오류로 확정하고 30.0 m 로 정정**한 값이다(`scene_station.py:37-39`).
  · `B-2` 앵커 파일 생성(08-23 06:52)이 정정 커밋 `97e8b59`(13:30)보다 **6시간 38분 앞선다**.
  · `M8` 은 **정정값 30.0** 으로 돌았다 = 두 배치 중 **옳은 쪽**이다.

⟹ ★**옛 술어는 사전등록 시점에 이미 거짓이었다.** (그 사실은 사전등록보다 **이틀 먼저**
`tmp/B2_REAL_2026-08-23/AMENDMENT_ZEROREF.md:106` 에 *"`SHIP` −2.7%~+1.2%"* 로 적혀 있었다.)

★**교체 — 옛것보다 약하지 않고 강하다** (⛔게이트를 고쳐 통과시키는 것이 아니다):

  `G-ANCHOR-B1`  ★**조건부 앵커** — `B-2` **자신의 수심 상수**를 복원하면 12/12 비트동일,
                 **그리고** 정정 수심에서는 `M8` 배치 행과 12/12 비트동일(양방향).
                 ⟹ 옛 술어가 **눈감고 있던 축(수심)을 명시적으로 검사**한다.
  `G-ANCHOR-B2`  432행의 `depth` 가 **정정값**이고 팔·시드에 걸쳐 레코드당 **유일**하다.
                 ⟹ 옛 술어에 아예 없던 검사다.
  `G-ANCHOR-SCENE` ★★**신설 구조 게이트** — ***앵커 게이트는 값을 비교하기 전에
                 「장면 상수 일치」를 먼저 검사한다.*** 장면 상수를 행에 싣지 않은 외부
                 자료는 **무조건 비트 앵커로 쓸 수 없다**(그 경우 조건부 앵커를 요구한다).
                 ⟵ 진단이 낸 저장소 처방(`AMENDMENT_1.md §5.4`)의 기계화.

⛔**그 밖의 모든 불변은 글자 그대로 승계한다** — 팔 12 · 사다리 9계단 6고리 · PRIMARY(L3) ·
문턱 · 라벨 4갈래 · `READING` · MDE · 채점 기준. **`gates_m8` 에서 *읽는다*. 손복사 0건.**
(이 파일에는 문턱 숫자도 팔 이름 목록도 라벨 문자열도 **새로 적히지 않는다**.)

⛔**`M8` 의 게이트·결과는 수정하지 않는다.** `M8` 은 *"판정 없음"* 으로 닫는다.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
REPO = os.path.abspath(os.path.join(ROOT, '..'))
SOURCE_DIR = os.path.join(ROOT, 'tmp', 'M8_LADDER_2026-08-28')
for _p in (HERE, SOURCE_DIR, ROOT,
           os.path.join(ROOT, 'scripts'),
           os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gates_m8 as G8                                            # noqa: E402

# ── ★불변 승계 (⛔재정의가 아니라 **참조**다. 값을 여기 적지 않는다) ──────────
BATCH = 'M8b_LADDER_2026-08-28'
SOURCE_BATCH = G8.BATCH
SOURCE_GLOB = os.path.join(SOURCE_DIR, 'results_m8_*.jsonl')
DIRECTION = G8.DIRECTION
RECORDS = G8.RECORDS
EST_SEEDS = G8.EST_SEEDS
ANCHOR_SEED = G8.ANCHOR_SEED
ARM_ORDER = G8.ARM_ORDER
RUNGS = G8.RUNGS
LINKS = G8.LINKS
FIELD = G8.FIELD
FIELD_SECOND = G8.FIELD_SECOND
READING = G8.READING
SCOPING = G8.SCOPING          # 결과 전 선언한 범위축소 4개 — 그대로 승계
THRESHOLD_PRIMARY = G8.THRESHOLD_PRIMARY
N_PRIMARY = G8.N_PRIMARY
COVERAGE = {}

# ── 교체되는 게이트 (이름을 상수로 둔다 — 문서와 코드가 어긋나지 않게) ────────
REPLACED_GATE = 'G-ANCHOR-B'
NEW_GATES = ('G-ANCHOR-SCENE', 'G-ANCHOR-B1', 'G-ANCHOR-B2')

# ── 외부 자료 ────────────────────────────────────────────────────────────────
#  ①`M8` 동결 결과 432행  ②`B-2` 동결 `SHIP`(=`G8.FROZEN_B2`)  ③`M4` 동결(=`G8.FROZEN_M4_GLOB`)
#  ④★이 배치가 만든 **앵커 재현 자료**(BANK · seed 42 · 12 레코드 × 수심 2종 = 24 런)
ANCHOR_JSONL = os.path.join(HERE, 'anchor_b2depth.jsonl')
#  ⑤★`B-2` **자신의** 장면 소스 — git 에서 꺼낸 사본. ⛔수심 상수를 손으로 적지 않는다.
B2ERA_SRC = os.path.join(HERE, 'scene_station_b2era.py')
B2ERA_GIT_REV = 'ca19192'          # `B-2` 사전등록 동결 커밋 (results.jsonl 생성 2분 전)
B2ERA_GIT_PATH = 'wavesplat/tmp/B2_REAL_2026-08-23/scene_station.py'
B2ERA_BLOB = '0f7c6053affb7348e5e85addddd74a851f6c7d24'

# ★**장면 상수** = 장면에서 나와 **추정기로 들어가지만 장면 해시에는 안 들어가는** 양.
#   `scene_station.py:104,187` — `depth` 는 관측 추출·질의 격자·참값에 **쓰이지 않고**
#   반환 dict 에 실려 나가 `run_m8.py:265,281` 이 **모든 팔에 같은 인자로** 넘긴다.
#   ⟹ `obs_hash`·`q_hash`·`gt_hash` 가 같아도 추정 결과는 갈릴 수 있다. 그것이 이 배치가
#   존재하는 이유다. ⛔이 목록을 늘릴 때는 새 사전등록이 필요하다.
SCENE_CONST_KEYS = ('depth',)


# ─────────────────────────────────────────────────────────────────────────────
# 1. 수심 상수의 **두 출처** (⛔둘 다 손복사가 아니다)
# ─────────────────────────────────────────────────────────────────────────────
def _parse_depths(src_text):
    """`RECORDS_12 = {rec: dict(..., depth=X, ...)}` 를 **AST 로** 읽는다.

    ⛔`exec` 하지 않는다(임포트 부작용 없이 상수만 꺼낸다). ⛔정규식도 아니다."""
    import ast
    try:
        tree = ast.parse(src_text)
    except SyntaxError:
        return {}
    out = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(getattr(t, 'id', None) == 'RECORDS_12' for t in node.targets):
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        for k, v in zip(node.value.keys, node.value.values):
            try:
                rec = ast.literal_eval(k)
            except ValueError:
                continue
            if isinstance(v, ast.Call):
                for kw in v.keywords:
                    if kw.arg == 'depth':
                        try:
                            out[rec] = float(ast.literal_eval(kw.value))
                        except ValueError:
                            pass
            elif isinstance(v, ast.Dict):
                for k2, v2 in zip(v.keys, v.values):
                    if getattr(k2, 'value', None) == 'depth':
                        try:
                            out[rec] = float(ast.literal_eval(v2))
                        except ValueError:
                            pass
    return out


def b2era_depths():
    """`B-2` 가 **실제로 돌린** 수심 상수. ⛔없으면 **빈 dict**(공허 PASS 금지)."""
    if not os.path.exists(B2ERA_SRC):
        return {}
    return _parse_depths(open(B2ERA_SRC, encoding='utf-8', errors='replace').read())


def current_depths():
    """현행(정정) 수심 상수 — `scene_station.RECORDS_12` **단일출처**에서 임포트한다."""
    try:
        from scene_station import RECORDS_12
    except Exception:                                             # noqa: BLE001
        return {}
    return {k: float(v['depth']) for k, v in RECORDS_12.items() if k in RECORDS}


def b2era_provenance():
    """git 이 그 사본을 확인해 주는가. ⛔git 이 없거나 어긋나면 **False**(관용 없음)."""
    try:
        blob = subprocess.run(['git', '-C', REPO, 'rev-parse',
                               f'{B2ERA_GIT_REV}:{B2ERA_GIT_PATH}'],
                              capture_output=True, text=True, timeout=30)
        local = subprocess.run(['git', '-C', REPO, 'hash-object', B2ERA_SRC],
                               capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as e:            # noqa: BLE001
        return False, f'git 실행 실패 {e!r}'
    b, l_ = blob.stdout.strip(), local.stdout.strip()
    ok = (b == B2ERA_BLOB and l_ == B2ERA_BLOB)
    return ok, f'{B2ERA_GIT_REV}:blob={b[:12]} 사본={l_[:12]} 기대={B2ERA_BLOB[:12]}'


# ─────────────────────────────────────────────────────────────────────────────
# 2. 자료 로더 (⛔비면 빈 목록 — `bool(...) and ...` 로만 쓴다)
# ─────────────────────────────────────────────────────────────────────────────
def source_rows():
    """`M8` 동결 432행. ★이 배치는 이 파일들을 **읽기만** 한다."""
    import glob as _g
    rows = []
    for p in sorted(_g.glob(SOURCE_GLOB)):
        rows += G8._load_jsonl(p)
    return rows


def anchor_rows():
    """앵커 재현 자료 24행 (`anchor_b2depth.py` 산출). ⛔없으면 빈 목록."""
    return [r for r in G8._load_jsonl(ANCHOR_JSONL) if 'error' not in r]


def _bank_anchor_batch(ok):
    """배치의 `BANK` · seed 42 행 → {record: row}. (= 옛 `G-ANCHOR-B` 가 보던 그 행들)"""
    out = {}
    for r in ok:
        if r.get('arm') == 'BANK' and G8._int(r.get('est_seed')) == ANCHOR_SEED:
            out.setdefault(r['record'], r)
    return out


def _same_bits(a, b):
    """두 행의 `nowcast`·`leadmean` 이 **비트 동일**한가 (파이썬 `!=` = 옛 술어 축자)."""
    return a.get(FIELD) == b.get(FIELD) and a.get(FIELD_SECOND) == b.get(FIELD_SECOND)


def _f(x):
    """⛔`float(None)` 이 `TypeError` 로 게이트를 죽이지 않게. 결측은 `None` 이다."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _raw_rows_by_record(pattern):
    """★외부 동결 파일의 **원본 행**을 레코드별로 (⛔로더의 투영이 아니다).

    `G8.b2_ship_ref()`·`G8.m4_prop_ref()` 는 키를 **골라서** 담기 때문에, 그것으로는
    *"원천이 장면 상수를 싣는가"* 를 물을 수 없다(로더가 떨어뜨린 것과 원천에 없는 것이
    구분되지 않는다). ⟹ 파일을 직접 읽는다. 필터는 **레코드 이름**뿐이고(`record`/`unit`),
    그 이름은 실측 레코드에만 쓰인다(`M4` 의 합성 층은 `BF4_…` 이름을 쓴다).
    자가진단이 이 집합이 `G8` 의 로더와 같은 레코드를 덮는지 대조한다."""
    import glob as _g
    out = {}
    for p in sorted(_g.glob(pattern)) or ([pattern] if os.path.exists(pattern) else []):
        for r in G8._load_jsonl(p):
            if 'error' in r:
                continue
            rec = r.get('record') or r.get('unit')
            if rec in RECORDS:
                out.setdefault(rec, []).append(r)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. 게이트 — ★`gates_m8.gates()` 를 **그대로 부르고** 앵커 술어 하나만 갈아 끼운다
# ─────────────────────────────────────────────────────────────────────────────
def gates(rows):
    base = list(G8.gates(rows))                 # ⛔재구현 금지 — 승계는 호출로 한다
    cov = dict(G8.COVERAGE)
    ok = [r for r in rows if 'error' not in r]

    new, ncov = [], {}

    def A(gid, good, msg, n):
        new.append((gid, bool(good), str(msg)))
        ncov[gid.split()[0]] = int(n)

    # ── ★G-ANCHOR-SCENE — **값 비교 전에 장면 상수부터 본다** ────────────────
    #    (a) 배치 자신이 장면 상수를 행에 싣는가
    #    (b) 각 외부 앵커 원천이 그것을 싣는가 → 싣으면 **일치**를 요구,
    #        안 싣으면 **무조건 비트 앵커로 쓸 수 없다** ⟹ 조건부 앵커 자료를 요구한다
    stamped_batch = [G8.key(r) for r in ok
                     if any(_f(r.get(k)) is None for k in SCENE_CONST_KEYS)]
    bat_const = {}
    for r in ok:
        d = bat_const.setdefault(r['record'], {})
        for k in SCENE_CONST_KEYS:
            d.setdefault(k, set()).add(_f(r.get(k)))
    srcs, seen = [], 0
    for name, pattern, cond_file in (
            ('B-2 SHIP', G8.FROZEN_B2, ANCHOR_JSONL),
            (f'M4 {G8.M4_ANCHOR_ARM}', G8.FROZEN_M4_GLOB, None)):
        byrec = _raw_rows_by_record(pattern)
        if len(byrec) != len(RECORDS):
            srcs.append((name, f'원천 레코드 {len(byrec)}/{len(RECORDS)} ⛔'))
            continue
        seen += sum(len(v) for v in byrec.values())
        stamps = all(k in r for rs in byrec.values() for r in rs
                     for k in SCENE_CONST_KEYS)
        if stamps:
            bad = [rec for rec, rs in byrec.items()
                   if rec in bat_const
                   and any({_f(r.get(k)) for r in rs} != bat_const[rec][k]
                           for k in SCENE_CONST_KEYS)]
            srcs.append((name, f'스탬프O 불일치={bad[:3]}' if bad else '스탬프O 일치'))
        else:
            has_cond = bool(cond_file) and bool(anchor_rows())
            srcs.append((name, '스탬프X ⟹ 무조건 앵커 불가 · 조건부 앵커='
                         + ('있음' if has_cond else '없음 ⛔')))
    bad_src = [s for s in srcs if '⛔' in s[1] or '불일치=' in s[1]]
    A('G-ANCHOR-SCENE ★앵커는 값 비교 **전에** 장면 상수 일치를 검사한다 '
      f'(상수={list(SCENE_CONST_KEYS)}; 안 싣는 원천은 무조건 앵커 불가)',
      bool(ok) and not stamped_batch and len(srcs) == 2 and not bad_src,
      f'배치 미스탬프={len(stamped_batch)} · ' + ' | '.join(f'{a}: {b}' for a, b in srcs),
      seen + len(ok))

    # ── ★G-ANCHOR-B1 — 조건부 앵커, **양방향** ──────────────────────────────
    #    (a) `B-2` 자신의 수심을 복원하면 `B-2` 동결과 12/12 비트동일
    #    (b) 정정 수심에서는 `M8` 배치 행과 12/12 비트동일  ⟸ 하네스 동일성의 증거
    #    (c) 수심이 같은 레코드에서는 (a)행과 (b)행이 서로 비트동일 (결정성)
    ref2 = G8.b2_ship_ref()
    d_b2, d_now = b2era_depths(), current_depths()
    arow = anchor_rows()
    bat = _bank_anchor_batch(ok)
    by = {}
    for r in arow:
        if r.get('arm') == 'BANK' and G8._int(r.get('est_seed')) == ANCHOR_SEED:
            by[(str(r.get('anchor_tag')), r.get('record'))] = r
    prov_ok, prov_msg = b2era_provenance()
    bad, seen = [], 0
    if not (ref2 and d_b2 and d_now and arow and bat):
        bad.append('참조/자료 부재')
    else:
        for rec in RECORDS:
            a2, an = by.get(('b2', rec)), by.get(('m8', rec))
            c2, cb = ref2.get(rec), bat.get(rec)
            if not (a2 and an and c2 and cb):
                bad.append((rec, 'missing')); continue
            seen += 1
            if _f(a2.get('depth')) is None or _f(a2.get('depth')) != _f(d_b2.get(rec)):
                bad.append((rec, 'b2 depth', a2.get('depth'))); continue
            if _f(an.get('depth')) is None or _f(an.get('depth')) != _f(d_now.get(rec)):
                bad.append((rec, 'now depth', an.get('depth'))); continue
            if not _same_bits(a2, c2):
                bad.append((rec, 'a: B-2 동결과 불일치')); continue
            if not _same_bits(an, cb):
                bad.append((rec, 'b: M8 배치와 불일치')); continue
            same_depth = float(d_b2[rec]) == float(d_now[rec])
            if same_depth and not _same_bits(a2, an):
                bad.append((rec, 'c: 같은 수심인데 두 재현이 다르다'))
    A('G-ANCHOR-B1 ★조건부 앵커 — (a) B-2 수심 복원 시 B-2 동결과 비트 12/12 · '
      '(b) 정정 수심에서 M8 배치와 비트 12/12 · (c) 수심 동일 레코드는 두 재현이 비트동일',
      bool(ref2) and bool(arow) and prov_ok and not bad and seen == len(RECORDS),
      f'재현={len(arow)}행 · git={prov_msg} · {str(bad[:3])}', seen)

    # ── ★G-ANCHOR-B2 — 배치가 **정정 수심** 위에 있고 레코드당 유일하다 ──────
    bad, seen = [], 0
    per = {}
    for r in ok:
        per.setdefault(r['record'], set()).add(_f(r.get('depth')))
    for rec in RECORDS:
        vs = per.get(rec)
        if not vs:
            bad.append((rec, 'no rows')); continue
        seen += 1
        if len(vs) != 1:
            bad.append((rec, 'depth 가 팔·시드에 걸쳐 유일하지 않다',
                        sorted(str(x) for x in vs)))
        elif next(iter(vs)) is None or next(iter(vs)) != _f(d_now.get(rec)):
            bad.append((rec, next(iter(vs)), d_now.get(rec)))
    # ⛔공허 방지: 정정이 실제로 **무언가를 바꿨어야** 한다(= 배치가 옛 상수 위가 아니다)
    moved = sorted(r for r in RECORDS
                   if _f(d_b2.get(r)) is not None and _f(d_now.get(r)) is not None
                   and _f(d_b2[r]) != _f(d_now[r]))
    A('G-ANCHOR-B2 ★432행의 depth 가 **정정값**이고 팔·시드에 걸쳐 레코드당 유일 '
      '(그리고 정정이 실제로 일부 레코드를 움직였다)',
      bool(d_now) and bool(d_b2) and not bad and seen == len(RECORDS) and bool(moved),
      f'정정된 레코드={moved} · {str(bad[:3])}', seen * len(EST_SEEDS) * len(ARM_ORDER))

    # ── 조립: 옛 앵커 술어를 뺀 자리에 셋을 넣고 `G-COV` 를 **다시 센다** ────
    idx = [i for i, (g, _o, _m) in enumerate(base) if g.split()[0] == REPLACED_GATE]
    cut = idx[0] if idx else len(base)
    body = [t for i, t in enumerate(base) if i != cut and t[0].split()[0] != 'G-COV']
    cov.pop(REPLACED_GATE, None)
    cov.pop('G-COV', None)
    cov.update(ncov)
    out = body[:cut] + new + body[cut:]
    zero = sorted(k for k, v in cov.items() if v <= 0)
    out.append(('G-COV ★커버리지 0 게이트 없음 (항등적으로 참/거짓인 검사 금지)',
                not zero, str(zero)))
    cov['G-COV'] = len(cov)
    COVERAGE.clear()
    COVERAGE.update(cov)
    return out


def blocking_failures(gate_rows):
    return [gid for gid, ok_, _ in gate_rows if not ok_ and gid.split()[0] not in SCOPING]


def verdict(rows, gate_rows=None, field=FIELD):
    """★판정은 `gates_m8.verdict` 이 낸다 — 라벨·문턱·집계를 여기서 다시 쓰지 않는다."""
    gate_rows = gates(rows) if gate_rows is None else gate_rows
    v = G8.verdict(rows, gate_rows=gate_rows, field=field)
    v.update(batch=BATCH, source_batch=SOURCE_BATCH,
             reanalysis_of=os.path.relpath(SOURCE_DIR, os.path.dirname(HERE)),
             replaced_gate=REPLACED_GATE, new_gates=list(NEW_GATES))
    return v


# ─────────────────────────────────────────────────────────────────────────────
# 4. 자가진단 — ★합성 행으로만 (판정 자료 0회 열람)
# ─────────────────────────────────────────────────────────────────────────────
def _fixture():
    """`gates_m8._fixture(anchored=True)` 를 승계하고 **이 배치가 바꾼 축만** 고친다.

    ① `depth` 를 레코드별 **정정값**으로 (M8 픽스처는 전 행 17.0 이었다)
    ② `BANK`·seed 42 행을 **앵커 재현 자료의 정정-수심 행**과 맞춘다
       (= 옛 픽스처는 `B-2` 값을 실었고, 그것이 바로 성립 불가였던 술어다)
    """
    rows = G8._fixture(anchored=True)
    d_now = current_depths()
    by = {r['record']: r for r in anchor_rows()
          if str(r.get('anchor_tag')) == 'm8' and r.get('arm') == 'BANK'}
    for r in rows:
        if r['record'] in d_now:
            r['depth'] = float(d_now[r['record']])
        if (r['arm'] == 'BANK' and G8._int(r['est_seed']) == ANCHOR_SEED
                and r['record'] in by):
            c = by[r['record']]
            r[FIELD] = c[FIELD]
            r[FIELD_SECOND] = c[FIELD_SECOND]
            r['nowcast_bits'] = G8.bits(c[FIELD])
            r['leadmean_bits'] = G8.bits(c[FIELD_SECOND])
    return rows


def synthetic_rows():
    """★`tmp/_audit/check_gate_blindspots.py` 훅 (결과 0행 시점 검사)."""
    return _fixture()


def selftest():
    fails = []

    def chk(cond, msg):
        if not cond:
            fails.append(msg)

    # ① ★승계가 진짜 승계인가 — `gates_m8` 자신의 자가진단이 통과해야 한다
    print('--- 승계 확인: gates_m8.selftest() ---')
    chk(G8.selftest() == 0, 'gates_m8 자가진단이 실패한다 (승계 불가)')
    print('--- 여기서부터 M8b ---')

    # ② 불변이 **참조**로 승계됐다 (값을 다시 적지 않았다)
    chk(RECORDS is G8.RECORDS and ARM_ORDER is G8.ARM_ORDER, '불변이 사본이다')
    chk(READING is G8.READING and SCOPING is G8.SCOPING, 'READING/SCOPING 이 사본이다')
    chk(THRESHOLD_PRIMARY == G8.THRESHOLD_PRIMARY == 10, '문턱 승계')
    # ⛔라벨·문턱 문자열이 이 파일에 **손복사**되지 않았다 (토막으로 적어 자기적발 회피)
    src = open(__file__, encoding='utf-8').read()
    for tok in ('PRIOR-ATOMS' + '-EARN', 'SKILL-DIES' + '-AT-',
                'LADDER' + '-VOID', 'GAP-NOT' + '-FACTORIZABLE'):
        chk(src.count(tok) == 0, f'⛔라벨 문자열 {tok!r} 이 이 파일에 손복사됐다')
    # ★외부 원천의 **원본 행 로더**가 `G8` 의 투영 로더와 같은 레코드를 덮는가 (필터 드리프트)
    chk(set(_raw_rows_by_record(G8.FROZEN_B2)) == set(G8.b2_ship_ref()),
        '원본 로더(B-2)가 G8 로더와 다른 레코드 집합을 낸다')
    chk(set(_raw_rows_by_record(G8.FROZEN_M4_GLOB)) == set(G8.m4_prop_ref()),
        '원본 로더(M4)가 G8 로더와 다른 레코드 집합을 낸다')

    # ③ 수심 상수의 두 출처 — 손복사가 아니라 **파싱·임포트**에서 온다
    d2, dn = b2era_depths(), current_depths()
    chk(len(d2) >= len(RECORDS), f'B-2 시대 수심 파싱 {len(d2)}')
    chk(len(dn) == len(RECORDS), f'현행 수심 임포트 {len(dn)}')
    moved = [r for r in RECORDS if d2.get(r) != dn.get(r)]
    chk(len(moved) == 7, f'정정된 레코드 수 {len(moved)} (기대 7 = BS 전건)')
    chk(all(r.startswith('BS') for r in moved), f'정정 레코드가 BS 밖이다: {moved}')
    prov_ok, prov_msg = b2era_provenance()
    chk(prov_ok, f'git 계보 확인 실패: {prov_msg}')

    # ④ 깨끗한 픽스처가 **전 게이트**를 통과한다 (환경 의존 `G-PATCH-*` 제외)
    good = _fixture()
    g = dict((k.split()[0], ok_) for k, ok_, _ in gates(good))
    env = {k for k in g if k.startswith('G-PATCH')}
    chk(all(v for k, v in g.items() if k not in env),
        '깨끗한 픽스처 FAIL: ' + str([k for k, v in g.items() if not v and k not in env]))
    chk(REPLACED_GATE not in g, f'{REPLACED_GATE} 가 아직 남아 있다')
    for k in NEW_GATES:
        chk(k in g, f'{k} 가 게이트 목록에 없다')
    clean_cov = dict(COVERAGE)
    zerocov = sorted(k for k, v in clean_cov.items() if v <= 0)
    chk(not zerocov, f'⛔커버리지 0 게이트: {zerocov}')
    chk(g.get('G-COV') is True, 'G-COV 가 깨끗한 픽스처에서 PASS 해야 한다')

    # ⑤ ★팔 삭제 민감도 — 팔 하나를 지우면 **반드시** 누가 FAIL 한다
    for arm in ARM_ORDER:
        rows = [r for r in good if r['arm'] != arm]
        bad = [gid.split()[0] for gid, ok_, _ in gates(rows)
               if not ok_ and not gid.startswith('G-PATCH')]
        chk(bad, f'{arm} 삭제가 어느 게이트에도 안 걸린다')
        chk([x for x in bad if x not in ('G-0', 'G-0e', 'G-SEED')],
            f'{arm} 삭제가 런 수/표본 게이트에만 걸린다: {bad}')
    # BANK 를 지우면 **앵커 게이트가** 걸려야 한다 (새 게이트의 팔-인지)
    bad = [gid.split()[0] for gid, ok_, _ in gates([r for r in good
                                                    if r['arm'] != 'BANK'])if not ok_]
    chk('G-ANCHOR-B1' in bad, 'BANK 삭제가 G-ANCHOR-B1 에 안 걸린다')

    # ⑥ ★변형 하네스 — 새 게이트마다 결함을 심으면 **그 게이트가** FAIL 해야 한다
    def mutate(fn, gate_id, base=None):
        rows = [dict(r) for r in (base or good)]
        fn(rows)
        gg = dict((k.split()[0], ok_) for k, ok_, _ in gates(rows))
        chk(gg.get(gate_id) is False, f'변형이 {gate_id} 에 안 걸린다')

    mutate(lambda rs: [r.__setitem__('depth', 17.0) for r in rs], 'G-ANCHOR-B2')
    mutate(lambda rs: rs[0].__setitem__('depth', 99.0), 'G-ANCHOR-B2')
    mutate(lambda rs: [r.pop('depth', None) for r in rs], 'G-ANCHOR-SCENE')
    mutate(lambda rs: [r.__setitem__('depth', 31.0) for r in rs], 'G-ANCHOR-SCENE')
    mutate(lambda rs: [r.__setitem__(FIELD, 0.123) for r in rs
                       if r['arm'] == 'BANK' and G8._int(r['est_seed']) == ANCHOR_SEED],
           'G-ANCHOR-B1')
    mutate(lambda rs: [r.__setitem__(FIELD_SECOND, 0.123) for r in rs
                       if r['arm'] == 'BANK' and G8._int(r['est_seed']) == ANCHOR_SEED],
           'G-ANCHOR-B1')

    # ⑦ ★공허 참조 방어 — 참조가 비면 새 게이트는 **FAIL** 이어야 한다 (PASS 아님)
    global ANCHOR_JSONL, B2ERA_SRC
    sv = (ANCHOR_JSONL, B2ERA_SRC, G8.FROZEN_B2, G8.FROZEN_M4_GLOB)
    ANCHOR_JSONL = os.path.join(HERE, '__none__.jsonl')
    gg = dict((k.split()[0], ok_) for k, ok_, _ in gates(good))
    chk(gg.get('G-ANCHOR-B1') is False, '앵커 자료가 없어도 G-ANCHOR-B1 이 통과한다')
    chk(gg.get('G-ANCHOR-SCENE') is False,
        '조건부 앵커가 없어도 G-ANCHOR-SCENE 이 통과한다')
    ANCHOR_JSONL = sv[0]
    B2ERA_SRC = os.path.join(HERE, '__none__.py')
    gg = dict((k.split()[0], ok_) for k, ok_, _ in gates(good))
    chk(gg.get('G-ANCHOR-B1') is False, 'B-2 시대 소스가 없어도 G-ANCHOR-B1 이 통과한다')
    chk(gg.get('G-ANCHOR-B2') is False, 'B-2 시대 소스가 없어도 G-ANCHOR-B2 가 통과한다')
    B2ERA_SRC = sv[1]
    G8.FROZEN_M4_GLOB = os.path.join(HERE, '__none__*.jsonl')
    gg = dict((k.split()[0], ok_) for k, ok_, _ in gates(good))
    chk(gg.get('G-ANCHOR-SCENE') is False, 'M4 참조가 비어도 G-ANCHOR-SCENE 이 통과한다')
    G8.FROZEN_M4_GLOB = sv[3]
    G8.FROZEN_B2 = os.path.join(HERE, '__none__.jsonl')
    gg = dict((k.split()[0], ok_) for k, ok_, _ in gates(good))
    chk(gg.get('G-ANCHOR-B1') is False, 'B-2 참조가 비어도 G-ANCHOR-B1 이 통과한다')
    G8.FROZEN_B2 = sv[2]

    # ⑧ 앵커 재현 자료 자체의 형식 (⛔판정 수치는 보지 않는다 — 개수·태그·스탬프만)
    ar = anchor_rows()
    chk(len(ar) == 2 * len(RECORDS), f'앵커 자료 {len(ar)}행 (기대 {2 * len(RECORDS)})')
    chk({str(r.get('anchor_tag')) for r in ar} == {'b2', 'm8'}, '앵커 태그 두 종')
    chk(all(r.get('arm') == 'BANK' for r in ar), '앵커 자료가 BANK 팔이 아니다')
    chk(all(G8._int(r.get('est_seed')) == ANCHOR_SEED for r in ar), '앵커 시드')
    chk(all(r.get('cuda_available') is False for r in ar), '⛔앵커 자료가 GPU 에서 돌았다')
    chk(all(r.get('harness_run_md5') for r in ar), '앵커 자료에 하네스 md5 스탬프 없음')

    gates(good)     # ⚠COVERAGE 를 깨끗한 픽스처의 것으로 되돌린다
    print(f'selftest(M8b): {"PASS" if not fails else "FAIL"}  ({len(fails)} 실패)')
    for f in fails:
        print('  -', f)
    if not fails:
        print(f'  교체 = {REPLACED_GATE} → {list(NEW_GATES)} · 게이트 {len(clean_cov)}개 · '
              f'승계 불변(팔 {len(ARM_ORDER)} · 고리 {[l[0] for l in LINKS]} · '
              f'문턱 {THRESHOLD_PRIMARY}/{N_PRIMARY})')
        print('  커버리지(깨끗한 픽스처): '
              + ', '.join(f'{k}={v}' for k, v in sorted(clean_cov.items())))
    return 0 if not fails else 1


def refcheck():
    ref2, ref4 = G8.b2_ship_ref(), G8.m4_prop_ref()
    d2, dn = b2era_depths(), current_depths()
    ar, sr = anchor_rows(), source_rows()
    prov_ok, prov_msg = b2era_provenance()
    print(f'SOURCE(M8 432행) = {SOURCE_GLOB}\n  행={len(sr)}  '
          f'에러={len([r for r in sr if "error" in r])}')
    def _stamped(pattern):
        byrec = _raw_rows_by_record(pattern)      # ★원본 행 (⛔로더의 투영이 아니다)
        return (bool(byrec) and all(k in r for rs in byrec.values() for r in rs
                                    for k in SCENE_CONST_KEYS))
    print(f'FROZEN_B2        = {G8.FROZEN_B2}\n  SHIP ref={len(ref2)}/{len(RECORDS)}  '
          f'장면상수 스탬프(원본 행)={_stamped(G8.FROZEN_B2)}')
    print(f'FROZEN_M4        = {G8.FROZEN_M4_GLOB}\n  ref={len(ref4)}/{len(RECORDS)}  '
          f'장면상수 스탬프(원본 행)={_stamped(G8.FROZEN_M4_GLOB)}')
    print(f'B2ERA_SRC        = {B2ERA_SRC}\n  수심 파싱={len(d2)}  git={prov_msg}')
    print(f'현행 수심(임포트) = {len(dn)}  정정된 레코드='
          f'{sorted(r for r in RECORDS if d2.get(r) != dn.get(r))}')
    print(f'ANCHOR_JSONL     = {ANCHOR_JSONL}\n  행={len(ar)}  '
          f'태그={sorted({str(r.get("anchor_tag")) for r in ar})}')
    gates(_fixture())
    zero = sorted(k for k, v in COVERAGE.items() if v <= 0)
    print('커버리지: ' + ', '.join(f'{k}={v}' for k, v in sorted(COVERAGE.items())))
    print(f'커버리지 0 게이트 = {zero}')
    okc = (len(ref2) == len(RECORDS) and len(ref4) == len(RECORDS) and len(d2) >= 12
           and len(dn) == 12 and len(ar) == 2 * len(RECORDS) and prov_ok and not zero
           and len(sr) == G8.N_RUNS)
    if not okc:
        print('\n⛔refcheck 실패 — 위에서 비어 있는 것을 먼저 만들어야 한다.')
    return 0 if okc else 1


if __name__ == '__main__':
    if '--refcheck' in sys.argv:
        sys.exit(refcheck())
    sys.exit(selftest())
