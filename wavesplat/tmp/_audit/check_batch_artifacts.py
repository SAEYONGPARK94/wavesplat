#!/usr/bin/env python3
"""배치 산출물 검사기 — **산문 처방이 두 번 실패해서 만든 기계적 방벽** (2026-08-23).

오늘 같은 결함이 세 번 났다:
  · W15  — 동결 분석기를 안 돌리고 라벨을 손으로 붙였다(`verdict_w15.json` 부재)
  · W17  — 분석기 자체가 없었고 `verdict` 에 `gates`·`label` 이 없었다
  · W12  — verdict 는 있으나 게이트 결과가 안 들어갔다

검사 항목 (배치 디렉터리 하나당):
  A. `PREREG.md` 존재
  B. `gates_*.py` 존재
  C. `analyze_*.py` 존재                     ← 판정이 코드에서 나왔다는 증거
  R. **`results*.jsonl` 존재**                ← ★D·E 의 **전제**. 없으면 무엇을 판정했는지 없다.
  D. `verdict_*.json` 존재 + `label` 키 보유  ← 라벨이 손으로 붙지 않았다는 증거
  E. `verdict_*.json` 이 `gates` 키 보유      ← 게이트가 실제로 돌았다는 증거
  F. `FROZEN_md5*.txt` 존재
  G. `analyze_*.py` 가 `results*.jsonl` 보다 **먼저** 쓰였는가 (mtime)
  H. **결과 파일마다 짝이 되는 verdict 이 있는가** (디렉터리 단위가 아니라 **분석 단위**)
  I. **`FROZEN_md5*.txt` 에 적힌 해시가 현재 파일과 일치하는가** (동결 위반 탐지)
  J. ★**팔이 빈 dict `{}` 로 정의되지 않았는가** (2026-08-23 R2 토론 F-7)
     `'ARM': {}` 은 **런타임 `DEFAULT_CONFIG` 를 스냅샷**한다 ⟹ 그 배치가 **그날의 기본값에
     묶이고**, 기본값이 나중에 바뀌면 **결과를 재현할 수도, 무엇을 쟀는지 알 수도 없다.**
     실제 사고: `gates_b7.py:46` 의 `'SHIP-TRACK': {}` 가 **당일 배포됐다 철회된 값**으로 돌아
     논문 문안에 들어갈 뻔했다.  ⟹ **팔은 키를 명시하거나 최소한 되읽기를 결과에 남겨야 한다.**

⚠**RV-5(2026-08-23)가 지적한 이 검사기의 사각지대**를 H·I 로 메웠다:
  · 종전에는 **디렉터리 단위**라 `results_b2b/c/d/e.jsonl` 처럼 **한 디렉터리에 여러 분석**이
    있으면 하나만 있어도 OK 로 셌다 ⟹ 라벨만 붙고 분석기가 없는 배치를 못 잡았다.
  · 종전에는 **동결물 md5 를 전혀 안 봤다** ⟹ `gates_b3b.py` 가 결과 뒤에 바뀐 것을 못 잡았다.

★★★**2026-08-26 수리 5건 — 전부 "검사기가 검사하지 않는다" 부류다.**
저장소는 2026-08-23 야간 감사에서 *"침묵을 경고로 승격하라"* 를 처방했는데
**이 검사기 자신이 그 처방을 세 군데에서 어기고 있었다.**

  ①**[M-4] 한 번도 안 돌아간 배치를 "인용 가능" 으로 통과시켰다.**
    분석기를 **결과 0행**에서 돌리면 exit 0 으로 `verdict_*.json` 을 쓰고 `label`·`gates` 키를
    갖춘다 ⟹ D·E 가 **결과 0개에서 충족**된다. 유일한 잔여 신호(`analyzer<results` 의 `--`)는
    종료코드에도 요약줄에도 반영되지 않았다.
    ⟹ **`results*.jsonl` 존재를 D·E 의 전제로 걸었다.** 결과 0개면 verdict 유무와 무관하게 `인용 불가`.

  ②**[M-5] 0건 대조를 `OK` 로 찍었다 — 그리고 그것이 거의 전부였다.**
    FROZEN 의 경로는 **배치 디렉터리 기준**으로 적혀 있는데(`../../wavesplat/run.py`,
    `gates_w7.py`) 검사기는 `dirname(dirname(d))` 한 곳만 봤다 ⟹ 실측 **동결 항목 207개 중
    201개가 조용히 SKIP** 되고 그 위에 `md5 OK`·`★자기동결 OK` 가 찍혔다.
    ⟹ (a) **경로 해석을 고쳤다**(배치 기준 → 저장소 기준 순) ⟹ 스킵 201 → 6.
       (b) **그래도 남는 스킵·0건 대조는 `OK` 가 아니라 경고**로 표에 드러낸다(`⚠m/n`).

  ③**[STATUS 08-24 ②] 샤딩된 결과의 짝을 못 봤다.**
    `results_b15_0..5.jsonl` 의 접미사가 `b15_0`…이라 `verdict_b15.json` 과 짝이 안 맞았다.
    ⟹ 접미사 끝의 `_<숫자>` **샤드 인덱스를 벗기고** 짝을 맞춘다.

  ④**[STATUS 08-24 ①] 축이 둘인 배치의 라벨을 `--` 로만 찍었다.**
    `B-14` 는 `h1_label`/`h2_label` 로 낸다. ⛔**통과로 바꾸지 않는다**(규약은 규약이다) —
    다만 `⚠축N` 으로 **왜 `--` 인지**를 표에 드러낸다.

  ⑥**[V3 반증 2026-08-26] ①의 수리가 파일 *존재* 만 봤다.**
    가짜 배치로 반증했다: **0바이트 `results_x.jsonl` + 손으로 쓴 `verdict_x.json`** 이
    `인용 가능` 으로 찍혔다 ⟹ ① 이 막으려던 바로 그 시나리오(*분석기가 결과 0행에서
    verdict 을 쓴다*)가 분석기가 빈 결과 파일까지 남기면 되살아난다.
    ⟹ **행이 하나라도 있는 결과 파일**을 요구한다(`_hasrow`). 샤드 하나가 비는 것은
       정상이므로 판정은 배치 단위로 하되 빈 파일 수를 `N⚠빈k` 로 드러낸다.

  ⑤**사전등록만 있고 안 돌아간 배치가 목록에 들지도 못했다.**
    기본 실행이 `results*.jsonl` 을 가진 디렉터리만 봤으므로 **결과 0행 배치는 표에 0줄**이었다
    ⟹ 구조적으로 검사되지 않았다. **`PREREG*.md` 를 가진 디렉터리도 목록에 넣고 `미실행` 표시.**

사용:  python3 tmp/_audit/check_batch_artifacts.py [tmp/BATCH_DIR ...]
인자가 없으면 `tmp/` 아래 `results*.jsonl` **또는** `PREREG*.md` 를 가진 디렉터리를 전부 검사한다.
종료코드 1 = D 또는 E 또는 H 위반이 하나라도 있음(= 인용 불가 배치가 있음).
"""
import ast, glob, hashlib, json, os, re, sys

COLS = ['PREREG', 'gates', 'analyze', 'results', 'verdict.label', 'verdict.gates',
        'FROZEN', 'analyzer<results', 'results~verdict', '★자기동결', 'md5', 'no-{}-arm']

SHARD = re.compile(r'_\d+$')          # `results_b15_3.jsonl` 의 샤드 인덱스
JUDGE = re.compile(r'^(gates|run|analyze).*\.py$')   # 판정을 낳는 배치-자체 코드


def _judge_files(d):
    """이 배치에서 **동결돼 있어야 하는 자기 파일**의 절대경로 집합.

    ★2026-08-26 V5 우회 시험이 뚫은 구멍이다. 종전 `own_cell` 은 자기 파일이
    **한 건이라도** 대조되면 `OK` 였다 ⟹ 두 가지가 그대로 통과했다:
      (b2) `gates_x.py` 를 동결 목록에서 빼고 **없는 경로**(`gates_GHOST.py`)를 적은 뒤
           `PREREG.md` 하나만 동결 ⟹ `⚠1/2` 경고만 뜨고 **인용 가능**.
      (d)  판정 코드를 아예 동결 목록에 **적지 않으면** 경고조차 없다.
    ⟹ *"동결 뒤에 움직였다"* 만 막고 *"동결한 적이 없다"* 는 안 막았다. 후자가 더 나쁘다 —
      전자는 최소한 탐지되지만 후자는 **구조적으로 탐지 불가**다.
    ⟹ 배치 안의 `gates*`·`run*`·`analyze*` 와 **그것들이 임포트하는 배치-내 모듈**을 요구한다
      (`scene_station.py`·`noise_models.py`·`zero_ref.py` 처럼 이름이 규약 밖이어도 잡힌다).
    ⚠사후 포렌식 스크립트(`depth_recheck.py`·`audit_replay_*.py`)는 판정을 낳지 않으므로 제외다.
    """
    need = {p for p in glob.glob(os.path.join(d, '*.py')) if JUDGE.match(os.path.basename(p))}
    for p in sorted(need):
        try:
            tree = ast.parse(open(p, encoding='utf-8', errors='replace').read())
        except Exception:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
                mods = [n.module]
            else:
                continue
            for mod in mods:
                q = os.path.join(d, mod.split('.')[0] + '.py')
                if os.path.exists(q):
                    need.add(q)
    return {os.path.realpath(p) for p in need}


def _resolve(d, tgt):
    """`FROZEN_md5*.txt` 의 경로를 실제 파일로 푼다.

    ⛔**종전에는 `dirname(dirname(d))` 한 곳만 봤다.** 그런데 이 저장소의 FROZEN 파일은
    **배치 디렉터리에서 `md5sum` 을 돌려 만든 것**이라 경로가 배치 기준이다
    (`gates_w7.py`, `../../wavesplat/optimizer/sequential_update.py`).
    ⟹ 거의 모든 대조가 `SKIPPED(missing)` 이었고 검사기는 그것을 `OK` 로 보고했다.
    일부 배치(B-17)는 헤더에 *"저장소 루트 기준 상대경로"* 라 적고 그렇게 쓴다.
    하위 분석 디렉터리(`S16/phase1_read`)는 **부모 배치의** `PREREG.md` 를 동결한다.
    ⟹ **배치 기준 → 부모 배치 기준 → 저장소 기준 순으로 시도**한다.
    """
    if os.path.isabs(tgt):
        return tgt if os.path.exists(tgt) else None
    for base in (d, os.path.dirname(d), os.path.dirname(os.path.dirname(d))):
        p = os.path.normpath(os.path.join(base, tgt))
        if os.path.exists(p):
            return p
    return None


def _hasrow(f):
    """결과 파일에 **행이 하나라도** 있는가. 앞부분만 읽어 O(1) 로 판정한다.

    ⚠`_resolve` 는 **없는 파일에 `None`** 을 준다 ⟹ `None` 을 받으면 곧바로 거짓이다
    (2026-08-26: 그 경우 `open(None)` 으로 죽어 재집계 반증 시험이 크래시했다).
    """
    if not f:
        return False
    try:
        with open(f, 'rb') as fh:
            return bool(fh.read(4096).strip())
    except OSError:
        return False


def _sfx(f, pre):
    """`results_b15_3.jsonl` → `b15` (샤드 인덱스 제거), `verdict_b15.json` → `b15`."""
    b = os.path.basename(f)
    b = b[len(pre):] if b.startswith(pre) else b
    b = b.split('.')[0].lstrip('_')
    return SHARD.sub('', b)


def check(d):
    d = d.rstrip(os.sep) or '.'
    has = lambda pat: bool(glob.glob(os.path.join(d, pat)))
    verdicts = glob.glob(os.path.join(d, 'verdict*.json'))
    lab = gts = False
    axis_labels = set()
    src_decl = []
    for f in verdicts:
        try:
            v = json.load(open(f))
        except Exception:
            continue
        if isinstance(v, dict):
            lab |= any(k in v for k in ('label', 'primary_label', 'verdict'))
            gts |= 'gates' in v
            # ★2026-08-26 (V4 F-7): **재집계 배치**(자기 런 0회, 다른 배치의 동결 결과를
            #   다시 집계)는 `results*.jsonl` 이 없는 것이 **정상**이다. ①의 엄격화가
            #   `B1_BASELINE_2026-08-22`(살아있는 헤드라인)를 오탐으로 죽였다.
            #   ⛔되돌리지 않는다 — verdict 이 `source_results` 로 **계보를 선언**하면
            #   여기서 그 파일들이 **존재하고 행이 있는지 직접 확인**한다. 선언만으로는
            #   통과하지 못한다(존재하지 않거나 비면 그대로 인용 불가).
            sr = v.get('source_results')
            if isinstance(sr, (list, tuple)):
                src_decl += [str(x) for x in sr]
            # ④ 축이 둘인 배치: `h1_label`/`h2_label`. 통과로 세지 않고 사유만 드러낸다.
            axis_labels |= {k for k in v if k.endswith('_label')}
    an = sorted(glob.glob(os.path.join(d, 'analyze*.py')))
    rs = sorted(f for f in glob.glob(os.path.join(d, 'results*.jsonl'))
                if 'ABANDONED' not in os.path.basename(f))   # 폐기 선언된 결과는 짝을 요구하지 않는다
    # ⑤ 규약 밖 이름(`smoke_results.jsonl`, `g0_results.jsonl`)은 **결과로 세지 않는다** —
    #    다만 "미실행" 이라 **거짓말하지는 않는다**. 별도 경고로 드러낸다.
    offspec = [f for f in glob.glob(os.path.join(d, '*results*.jsonl'))
               if f not in rs and 'ABANDONED' not in os.path.basename(f)]
    order = bool(an and rs) and min(os.path.getmtime(a) for a in an) < max(os.path.getmtime(r) for r in rs)

    # ★① D·E 의 전제 = 결과가 실제로 있는가. 결과 0행에서도 분석기는 exit 0 으로
    #    `verdict` 을 쓴다(적대적 검증에서 B17·B18·B19 가 그렇게 통과했다).
    # ⛔**2026-08-26 V3 반증**: `bool(rs)` 는 **파일 존재**만 봤다 ⟹ 분석기가 **빈**
    #    `results*.jsonl` 을 만들고 `verdict` 을 쓰면 M-4 수리를 그대로 통과했다
    #    (가짜 배치 시험에서 `0바이트 결과 + 가짜 verdict` 이 `인용 가능` 으로 찍혔다).
    #    ⟹ **행이 하나라도 있는 파일**을 요구한다. 샤드 중 하나가 비는 것은 정상이므로
    #    판정은 **배치 단위**(어느 한 파일이라도 행이 있으면 OK)로 하되 빈 파일 수는 드러낸다.
    empty_rs = [f for f in rs if not _hasrow(f)]
    has_rs = len(empty_rs) < len(rs)
    # 재집계 계보: 선언된 원천이 **전부** 존재하고 행이 있어야 한다(하나라도 어기면 불가).
    src_bad = [x for x in src_decl if not _hasrow(_resolve(d, x))]
    has_src = bool(src_decl) and not src_bad
    has_rs = has_rs or has_src
    lab_ok, gts_ok = lab and has_rs, gts and has_rs

    # H. 결과 파일마다 짝 verdict — `results_b2b.jsonl` ↔ `verdict_b2b.json` 식의 접미사 매칭
    rs_sfx = {_sfx(f, 'results') for f in rs}
    vd_sfx = {_sfx(f, 'verdict') for f in verdicts}
    # 결과가 하나뿐이면 접미사가 달라도 짝으로 인정한다(`results.jsonl` ↔ `verdict_w17.json`).
    # 결과가 여럿이면 **각각** 짝이 있어야 한다 — 그게 RV-5 가 지적한 사각지대다.
    # 접미사 없는 `results.jsonl` 은 아무 verdict 과도 짝으로 인정한다(관례상 배치 본 결과).
    need = {x for x in rs_sfx if x}
    paired = has_rs and (bool(verdicts) if len(rs_sfx) <= 1 else (bool(verdicts) and need <= vd_sfx))
    # 재집계 배치는 자기 결과 파일이 없으므로 접미사 짝을 요구할 수 없다 — verdict 존재로 족하다.
    if has_src and not rs:
        paired = bool(verdicts)

    # I. 동결 해시 대조
    # ★★**두 부류를 가른다**(2026-08-23 전수조사): `md5 OK` 한 열로는
    #   ⚠*"배포코드가 그 뒤에 바뀐 정상 상태"* 와
    #   ⛔*"판정 규칙이 사후에 움직인 이상 상태"* 가 **같은 `--`** 로 보인다.
    # 배치 자신의 `gates_*`·`run_*`·`analyze_*` 가 움직인 것만이 심각하다.
    # ★**같은 경로가 뒤에 다시 찍혀 있으면 뒤엣것이 유효하다**(재동결).
    # 원칙: *드리프트가 있었다는 사실은 지우지 않는다* ⟹ 원본 줄을 지우는 대신
    # 재동결 줄을 **덧붙인다**. 그러면 이 열이 영원히 `--` 로 남으므로 신호가 죽는다.
    # ⟹ 검사기가 **마지막 기록**을 현재 동결로 읽는다(앞선 줄은 이력으로 보존).
    # ⚠V5: 재동결하면 이 열이 `OK` 로 돌아간다 ⟹ **표에서 드리프트 사실이 사라진다**
    #   (규약은 원본 줄을 보존하라고 하지만 그건 *산문*이고 검사기는 마지막 줄만 읽는다).
    #   ⟹ 같은 경로가 **다른 해시로** 두 번 이상 적힌 것을 세어 `재동결` 로 계속 드러낸다.
    latest, seen = {}, {}
    for ff in sorted(glob.glob(os.path.join(d, 'FROZEN*.txt'))):
        for ln in open(ff):
            parts = ln.split()
            if len(parts) != 2 or len(parts[0]) != 32 or ln.startswith('#'):
                continue
            key = os.path.basename(parts[1])
            seen.setdefault(key, set()).add(parts[0])
            latest[parts[1]] = parts[0]
    refrozen = sorted(k for k, v in seen.items() if len(v) > 1)
    drift, own_drift, skipped = [], [], []
    own_frozen = set()
    n_cmp = n_own_cmp = 0
    for tgt, h in sorted(latest.items()):
        p = _resolve(d, tgt)
        if p is None:
            skipped.append(tgt)
            continue
        n_cmp += 1
        own = os.path.abspath(p).startswith(os.path.abspath(d) + os.sep)
        n_own_cmp += own
        if own:
            own_frozen.add(os.path.realpath(p))
        if hashlib.md5(open(p, 'rb').read()).hexdigest() != h:
            drift.append(os.path.basename(p))
            if own:
                own_drift.append(os.path.basename(p))
    # ★V5: **동결 목록에서 빠진 자기 판정코드** — "움직였다" 가 아니라 "적은 적이 없다".
    unfrozen = sorted(os.path.basename(q) for q in _judge_files(d) if q not in own_frozen)

    # ★② "0건 대조" 와 "부분 스킵" 은 `OK` 가 아니라 **경고**다.
    if not latest:
        # FROZEN 파일 자체가 없으면 F 열이 이미 `--` 로 말한다. 있는데 해시 줄이 0개면
        # **그것이야말로 침묵**이므로 경고로 올린다.
        md5_cell = own_cell = ('⚠0줄' if glob.glob(os.path.join(d, 'FROZEN*.txt')) else '--')
    elif drift:
        md5_cell = '--'
        own_cell = '--' if own_drift else ('OK' if n_own_cmp else '⚠0건')
    elif not n_cmp:
        md5_cell = own_cell = f'⚠0/{len(latest)}'
    elif skipped:
        md5_cell = f'⚠{n_cmp}/{len(latest)}'
        own_cell = 'OK' if n_own_cmp else '⚠0건'
    else:
        md5_cell = 'OK'
        own_cell = 'OK' if n_own_cmp else '⚠0건'
    # ⚠자기 파일이 몇 건 대조됐든, **판정을 낳는 자기 코드가 동결 목록에 없으면** OK 가 아니다.
    if own_cell == 'OK' and unfrozen:
        own_cell = f'⚠미동결{len(unfrozen)}'

    # J. 빈 dict 팔 탐지
    empty_arm = True
    for gf in glob.glob(os.path.join(d, 'gates*.py')):
        txt = open(gf).read()
        if re.search(r"['\"][A-Za-z0-9_.+-]+['\"]\s*:\s*\{\s*\}", txt):
            empty_arm = False

    if rs:
        rs_cell = f'{len(rs)}⚠빈{len(empty_rs)}' if empty_rs else str(len(rs))
    elif offspec:
        rs_cell = '⚠규약밖'
    elif src_decl:
        # 재집계 배치 — "미실행" 이라 **거짓말하지 않는다**. 선언 원천이 깨졌으면 그것을 드러낸다.
        rs_cell = f'재집계{len(src_decl)}' if has_src else f'⛔원천깨짐{len(src_bad)}'
    else:
        rs_cell = '⚠미실행'
    lab_cell = ('OK' if lab_ok else
                ('⚠결과0' if lab else
                 (f'⚠축{len(axis_labels)}' if axis_labels else '--')))
    gts_cell = 'OK' if gts_ok else ('⚠결과0' if gts else '--')

    # ④ 경고는 **규약 라벨이 아예 없을 때만** 낸다. `label` 을 이미 낸 배치의
    #    부차 라벨(`secondary_label`·`dose_label`)까지 경고하면 신호가 잡음에 묻힌다.
    info = dict(drift=sorted(set(drift) - set(own_drift)), own_drift=sorted(set(own_drift)),
                unfrozen=unfrozen, refrozen=refrozen,
                skipped=skipped, n_cmp=n_cmp, n_frozen=len(latest),
                axis_labels=[] if lab else sorted(axis_labels), offspec=[os.path.basename(f) for f in offspec],
                no_results=not has_rs, has_verdict=bool(verdicts))
    cells = [has('PREREG*.md'), has('gates*.py'), bool(an), rs_cell, lab_cell, gts_cell,
             has('FROZEN*.txt'), order, paired, own_cell, md5_cell, empty_arm]
    # ★★2026-08-26 (사용자 결정: "막고 다시 하자") — **동결 증거를 판정에 넣는다.**
    # ⛔종전: `ok = lab_ok and gts_ok and paired` 뿐이라, 이 검사기가 스스로
    #   *"판정 규칙이 사후에 이동했다 — 가장 심각하다"* 고 인쇄한 배치가 **인용 가능**으로
    #   통과했다(V3 가 `W9_PENALTY_BUDGET_2026-08-22` 로 직접 확인). 최상위 경고가
    #   종료코드에 안 걸리는 것은 **경고를 인쇄만 하는 방벽** = 저장소가 반복해서 대가를
    #   치른 부류다(08-23 "침묵을 경고로 승격하라" 가 여기엔 적용되지 않았다).
    # ⟹ 두 가지를 **차단 사유**로 승격한다:
    #   (a) `own_drift` = 배치 **자신의** gates/run/analyze 가 동결 뒤에 움직였다.
    #       판정 규칙이 결과를 보고 이동했을 수 있다 ⟹ 그 판정은 인용할 수 없다.
    #   (b) 자기 파일을 **한 건도 대조하지 못했다**(`⚠0건`/`⚠0줄`/`⚠0/n`).
    #       "위반 없음" 이 아니라 **"검사한 적 없음"** 이다.
    # ⚠배포코드만 움직인 `drift` 는 **차단하지 않는다** — 런은 옛 코드로 돌았고
    #   무효가 되는 것은 재실행뿐이다(그 구분은 이 파일이 이미 옳게 적어 두었다).
    # ⚠재집계 배치는 자기 결과가 없을 뿐 자기 동결은 똑같이 요구된다.
    frozen_ok = (not own_drift) and own_cell == 'OK'
    ok = lab_ok and gts_ok and paired and frozen_ok
    info['frozen_block'] = (not frozen_ok) and lab_ok and gts_ok and paired
    return cells, ok, info


def _fmt(x):
    return f'{("OK" if x else "--") if isinstance(x, bool) else x:>9s}'


def main(argv):
    if argv[1:]:
        dirs = [a.rstrip(os.sep) for a in argv[1:]]
    else:
        # ★⑤ 결과가 없는(=안 돌아간) 배치도 **목록에 든다**. 종전에는 표에 0줄이었다.
        dirs = sorted({os.path.dirname(f) for f in glob.glob('tmp/**/results*.jsonl', recursive=True)} |
                      {os.path.dirname(f) for f in glob.glob('tmp/**/PREREG*.md', recursive=True)})
    bad = no_res = 0
    print(f"{'배치':44s} " + ' '.join(f'{c[:9]:>9s}' for c in COLS))
    drifted, own_drifted, skips, axes, offspecs, unrun = {}, {}, {}, {}, {}, []
    frozen_blocked, unfrozen_own, refroze = [], {}, {}
    for d in dirs:
        cells, ok, info = check(d)
        name = os.path.basename(d)
        if not ok:
            bad += 1
        if info['no_results']:
            no_res += 1
            unrun.append(f"{name}{' (verdict 있음!)' if info['has_verdict'] else ''}")
        if info['drift']:
            drifted[name] = info['drift']
        if info['own_drift']:
            own_drifted[name] = info['own_drift']
        if info.get('unfrozen'):
            unfrozen_own[name] = info['unfrozen']
        if info.get('refrozen'):
            refroze[name] = info['refrozen']
        if info.get('frozen_block'):
            frozen_blocked.append(name)
        if info['skipped']:
            skips[name] = (info['n_cmp'], info['n_frozen'], info['skipped'])
        if info['axis_labels']:
            axes[name] = info['axis_labels']
        if info['offspec']:
            offspecs[name] = info['offspec']
        print(f"{name[:44]:44s} " + ' '.join(_fmt(x) for x in cells))

    if own_drifted:
        print('\n⛔**★자기동결 위반 — 배치 자신의 파일이 동결 뒤에 움직였다.**\n'
              '   ⚠판정은 **이름이 아니라 경로**로 한다 — 배치 디렉터리 안이면\n'
              '   `obs_models.py`·`scene_station.py` 처럼 규약 밖 이름도 포함된다.\n'
              '   판정 규칙이 사후에 이동했다는 뜻이므로 **가장 심각하다**:')
        for k, v in own_drifted.items():
            print(f'  {k}: ' + ', '.join(v))
    if drifted:
        print('\n동결 이동(md5 불일치). ⚠배포코드(sequential_update.py·run.py)만 움직인 것은 **정상**이다\n'
              '(런은 옛 코드로 돌았다; 재실행만 무효). 위의 ★자기동결 목록만이 심각하다:')
        for k, v in drifted.items():
            print(f'  {k}: ' + ', '.join(v))
    if skips:
        print('\n⚠**동결 대조 스킵**(파일을 못 찾아 대조하지 못했다 — 종전에는 이것이 `OK` 로 찍혔다).\n'
              '   `⚠0/n` = **0건 대조**(그 배치의 md5 열은 아무것도 검증하지 못했다):')
        for k, (c, n, sk) in skips.items():
            print(f'  {k}: {c}/{n} 대조 · 미해결 경로 ' + ', '.join(sk[:4]) + (' …' if len(sk) > 4 else ''))
    if axes:
        print('\n⚠**verdict 라벨 키가 규약 밖**(축이 둘인 배치 등). 통과로 세지 않는다 — '
              '`label`/`primary_label`/`verdict` 중 하나를 함께 내라:')
        for k, v in axes.items():
            print(f'  {k}: ' + ', '.join(v))
    if offspecs:
        print('\n⚠**결과 파일 이름이 규약(`results*.jsonl`) 밖**이라 이 검사기가 못 본다:')
        for k, v in offspecs.items():
            print(f'  {k}: ' + ', '.join(v[:5]) + (' …' if len(v) > 5 else ''))
    if refroze:
        print('\n⚠**재동결 이력 — 드리프트가 있었고 사후에 다시 동결됐다.**\n'
              '   ★자기동결/md5 열은 `OK` 로 돌아가지만 **사실은 지워지지 않는다.**\n'
              '   재동결 사유가 해당 `FROZEN*.txt` 헤더에 적혀 있어야 하고, 판정 불변이\n'
              '   기계적으로 확인됐을 때만 인용 가능하다(동결 갱신 자체는 증거가 아니다):')
        for k, v in sorted(refroze.items()):
            print(f'  {k}: ' + ', '.join(v))
    if unfrozen_own:
        print('\n⛔**★동결 목록 누락 — 판정을 낳는 배치 자체 코드가 동결된 적이 없다.**\n'
              '   ★자기동결 위반(움직였다)보다 **탐지가 더 어렵다** — 기준선이 없으므로\n'
              '   md5 열은 영원히 `OK` 를 찍는다. 대상 = `gates*`·`run*`·`analyze*` 와\n'
              '   그것들이 임포트하는 배치-내 모듈(`scene_station.py` 부류):')
        for k, v in sorted(unfrozen_own.items()):
            print(f'  {k}: ' + ', '.join(v))
    if unrun:
        print(f'\n⚠**결과 0개(미실행 또는 결과가 다른 배치에 있음) — {len(unrun)}개.**\n'
              '   ★`verdict 있음!` 은 **결과 없이 라벨이 붙었다**는 뜻이다(M-4 부류):')
        for k in unrun:
            print(f'  {k}')
    if frozen_blocked:
        print(f'\n⛔**동결 증거 부재/이동으로 인용 불가 — {len(frozen_blocked)}개.**\n'
              '   라벨·게이트·짝은 갖췄으나 배치 **자신의** 파일이 동결 뒤에\n'
              '   움직였거나(★자기동결 위반), **한 건도 대조되지 않았다**(⚠0건), 또는\n'
              '   **판정 코드를 동결 목록에 적은 적이 없다**(⚠미동결n = V5 우회 시험이 뚫은 구멍).\n'
              '   ⟹ 판정 규칙이 결과를 보고 이동했을 가능성을 배제할 수 없다:')
        for k in frozen_blocked:
            print(f'  {k}')
    print(f"\n인용 불가(라벨·게이트·짝·동결 증거 없음): {bad}/{len(dirs)}   "
          f"(그중 결과 0개: {no_res} · 동결 사유만: {len(frozen_blocked)})")
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
