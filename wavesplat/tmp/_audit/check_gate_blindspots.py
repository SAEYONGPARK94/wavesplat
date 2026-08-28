#!/usr/bin/env python3
"""게이트 **부분집합 눈멂** 검사기 — *"팔 이름을 단 게이트가 그 팔의 행을 안 본다"* (2026-08-28).

★**결함 부류 (`M4` `G-13` 이 노출시켰다)** — `check_gate_refs.py` 가 **구조적으로 못 잡는 쪽**

    게이트가 참조하는 **키는 실제로 있다**. 그런데 게이트가 세는 **집합**이, 러너가 행에
    찍는 스탬프 때문에 그 팔을 **정의상 배제**한다 ⟹ 판정이 자료와 무관하게 **항등적**이다.

실측 사례 (`tmp/M4_CAPACITY_2026-08-28/gates_m4.py:480-487`):
    `G-13 real SHIP-K-INL` 이 `REAL_ARMS` 를 돌며 팔별 행 수를 세는데, 세는 대상이
    `real = [r for r in ok if r['stratum'] == 'real' and not r.get('pc')]` 다.
    그런데 `SHIP-K-INL` 행을 만드는 곳은 저장소에 하나뿐이고(`run_m4.py:224,256`)
    거기서 **전부 `pc=True` 를 박는다** ⟹ `0/12` 은 **항등적으로 참인 실패**.
    ★어떤 실행 순서·어떤 샤딩으로도 통과시킬 수 없었다. `check_gate_refs` 는 `'pc'` 키가
    **존재하므로** 침묵한다 — 부재가 아니라 **필터 항진**이기 때문이다.

★**방법 — 삭제 민감도** (런 0회 · 읽기 전용 · GPU 0회)

    base = gates(rows)
    for v in {행에 실제로 있는 arm/stratum 값}:
        out = gates([r for r in rows if r[key] != v])
        게이트 **이름에 v 가 토큰으로** 들어 있는데 `(ok, msg)` 가 base 와 **비트동일**
        ⟹ 그 게이트는 v 의 행을 **보지 않는다**.

    ⟹ 이름이 주장하는 범위와 실제로 읽는 범위의 **불일치**를 잰다. 게이트 본문을 파싱하지
       않으므로 조건식이 아무리 복잡해도 통한다(정적 분석의 취약함을 여기서 피한다).

★**두 등급 — 자동으로 가른다** (⛔에만 종료코드가 반응한다)
  · ⛔**항등 게이트** (`ok=False`) — 그 팔의 행이 **있는데도** 안 보고 실패한다.
    ⟹ 자료가 무엇이든 통과 불가. `M4 G-13` 이 정확히 이것이다.
  · ⚠**존재 가드 없음** (`ok=True`) — 위반 0건을 세는 게이트가 **그 팔의 행이 통째로
    사라져도 똑같이 통과**한다. 지금 자료에서는 무해하지만(행이 실제로 있다),
    *"검사했다"* 의 근거가 **행의 존재를 다른 게이트에 의존**한다는 뜻이다.
    처방은 한 줄이다 — `M4 G-15` 의 `len(inl) > 0 and …` 처럼 **분모를 조건에 넣는 것**.

⛔**내가 시도했다가 버린 판별자**(기록): 팔의 행 *값* 을 흔들어(내용 민감도) 두 등급을
   가르려 했더니 **`M4` 자신을 오분류**했다 — 변이가 `pc=True` 를 뒤집어 필터를 통과시켜
   버려서 "내용 민감" 으로 보였다. ★**필터 스탬프를 흔들면 그것은 내용 변화가 아니라
   존재 변화다.** 그래서 삭제 민감도 하나만 쓴다.

★**행을 어디서 읽는가 — 분석기 자신의 로더를 쓴다**
   `analyze_*.py` 의 무인자 로더(`load_rows`/`load`/…)를 찾아 **그것을 호출**한다.
   ⛔`results*.jsonl` glob 을 재구현하면 **의도표**가 되고 오탐이 난다 — 실측:
   `M4` 는 `B-22` 동결 행 144개를 **수입**하는데 glob 만 쓰면 `G-13s syn SHIP-K` 가
   **없는 결함으로** 찍혔다(초판 프로토타입의 오탐 1건, 이렇게 고쳤다).
   ★사전등록 시점(결과 0행)에는 `gates_*.synthetic_rows()` 를 쓴다 — 배치가 그 훅을
   제공하면 **결과가 나오기 전에** 이 검사가 돌아간다(그것이 이 부류의 올바른 시점이다).

────────────────────────────────────────────────────────────────────────────
★**보지 않는 것**(= 이 검사기로 *"검사했다"* 고 말할 수 없는 범위)
  · **행이 0개인 팔** — 삭제할 것이 없으면 민감도를 잴 수 없다. *"선언된 팔인데 행이
    아예 없다"* 는 팔별 행 수 게이트(`G-13` 부류)의 일이고, 그 게이트 자신이 이 결함에
    걸리면 아무도 안 본다 ⟹ ★그래서 배치 안 자가진단(합성 행 만족가능성)이 따로 필요하다.
  · **`gates(rows)` 로 부를 수 없는 배치** — 인자를 더 받거나 `rows` 이름이 아니면 건너뛴다.
    2026-08-28 실측에서 40배치 중 **27개만** 검사됐다. ⛔**나머지 13개는 `OK` 가 아니다.**
  · **`arm`·`stratum` 밖의 축**(레코드·시드·창) · 이름에 팔이 안 적힌 게이트 ·
    게이트 이름이 팔 이름과 **토큰으로** 일치하지 않는 표기.
  · 통계 함수(`paired`/`holm`)에 같은 결함이 있는지 — 게이트 목록만 본다.

사용:
    python3 tmp/_audit/check_gate_blindspots.py                     # tmp/ 전체
    python3 tmp/_audit/check_gate_blindspots.py tmp/M4_CAPACITY_2026-08-28
    python3 tmp/_audit/check_gate_blindspots.py --selftest          # 일부러 깨진 픽스처
종료코드 1 = ⛔항등 게이트가 하나라도 있음(⚠존재 가드 없음은 0).
"""
import glob as globmod
import importlib.util
import inspect
import json
import os
import re
import subprocess
import sys

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')      # ⛔GPU 금지

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = os.path.dirname(HERE)
ROOT = os.path.dirname(TMP)

GROUP_KEYS = ('arm', 'stratum')     # 게이트 이름이 실제로 다는 축
MAX_VALUES = 30                     # 한 축의 값이 이보다 많으면 (레코드 축 등) 건너뛴다
BATCH_TIMEOUT_S = 900
LOADER_NAMES = ('load_rows', 'load', 'read_rows', 'rows')
SYNTH_NAMES = ('synthetic_rows', '_synthetic_rows', 'fake_rows')
ENTRY_RE = re.compile(r'^gates?(_run|_after|_post)?$')
FIRST_ARG = ('rows', 'res', 'results', 'r')


def _tokens(name):
    return set(re.split(r'[\s(),:=/\[\]\'"·]+', str(name)))


def _import(path):
    d = os.path.dirname(path)
    if d not in sys.path:
        sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location(
        'x_' + os.path.basename(path)[:-3], path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    try:
        spec.loader.exec_module(m)
    except SystemExit as e:
        # ⚠임포트만으로 `sys.exit` 하는 모듈이 있다(`analyze_m3.py`·`gates_w10.py` 는
        #   최상위에서 판정하고 끝낸다). 프로세스를 죽게 두면 **배치가 통째로 침묵**한다.
        raise RuntimeError(f'임포트 중 sys.exit({e.code})')
    return m


def _noarg(f):
    try:
        return all(p.default is not inspect.Parameter.empty
                   for p in inspect.signature(f).parameters.values())
    except (TypeError, ValueError):
        return False


def _own(mod, f):
    return callable(f) and getattr(f, '__module__', None) == mod.__name__


def _rows_from(mod, names):
    for nm in names:
        f = getattr(mod, nm, None)
        if not _own(mod, f) or not _noarg(f):
            continue
        got = f()
        if isinstance(got, tuple) and got and isinstance(got[0], list):
            got = got[0]           # `load_rows()` 가 (rows, n_imported) 를 준다
        if isinstance(got, list) and got and isinstance(got[0], dict):
            return got, nm
    return None, None


def _glob_rows(d):
    rows = []
    for p in sorted(globmod.glob(os.path.join(d, 'results*.jsonl'))):
        for ln in open(p):
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
    return rows


def _load_rows(d, rec, gfiles):
    """① 분석기 로더 → ② 사전등록 훅 `synthetic_rows()` → ③ glob 폴백."""
    for p in sorted(globmod.glob(os.path.join(d, 'analyze_*.py'))):
        try:
            m = _import(p)
        except Exception as e:                                    # noqa: BLE001
            rec['notes'].append(f'{os.path.basename(p)} 임포트 실패: {repr(e)[:80]}')
            continue
        try:
            got, nm = _rows_from(m, LOADER_NAMES)
        except Exception as e:                                    # noqa: BLE001
            rec['notes'].append(f'{os.path.basename(p)} 로더 실패: {repr(e)[:70]}')
            got = None
        if got:
            rec['loader'] = f'{os.path.basename(p)}:{nm}()'
            return got
    rows = _glob_rows(d)
    if rows:
        rec['loader'] = 'glob(results*.jsonl)'
        return rows
    for p in gfiles:                       # ★결과 0행 = 사전등록 시점
        try:
            got, nm = _rows_from(_import(p), SYNTH_NAMES)
        except Exception as e:                                    # noqa: BLE001
            rec['notes'].append(f'{os.path.basename(p)} 합성행 훅 실패: {repr(e)[:70]}')
            continue
        if got:
            rec['loader'] = f'{os.path.basename(p)}:{nm}() ★사전등록 시점'
            return got
    rec['loader'] = None
    return []


def _entry_fns(mod):
    """`gates*(rows, …)` 꼴 진입 함수 — 나머지 인자는 전부 기본값이 있어야 부른다."""
    out = []
    for nm in sorted(dir(mod)):
        if not ENTRY_RE.match(nm):
            continue
        f = getattr(mod, nm)
        if not _own(mod, f):
            continue
        try:
            ps = list(inspect.signature(f).parameters.values())
        except (TypeError, ValueError):
            continue
        if not ps or ps[0].name not in FIRST_ARG:
            continue
        if any(p.default is inspect.Parameter.empty for p in ps[1:]):
            continue
        out.append((nm, f))
    return out


def _norm(res):
    out = []
    for t in res:
        t = tuple(t)
        out.append((str(t[0]), bool(t[1]), str(t[2]) if len(t) > 2 else ''))
    return out


def check_batch(d):
    rec = dict(batch=os.path.basename(d), rows=0, loader=None, fns=[],
               findings=[], notes=[], err=None)
    os.chdir(ROOT)
    gfiles = sorted(globmod.glob(os.path.join(d, 'gates_*.py')))
    if not gfiles:
        return rec
    rows = _load_rows(d, rec, gfiles)
    rec['rows'] = len(rows)
    if not rows:
        rec['notes'].append('행 0개 — 검사 불가(⛔`OK` 아님)')
        return rec
    mods = []
    for p in gfiles:
        try:
            mods.append((os.path.basename(p), _import(p)))
        except Exception as e:                                    # noqa: BLE001
            rec['notes'].append(f'{os.path.basename(p)} 임포트 실패: {repr(e)[:90]}')
    vals = {}
    for k in GROUP_KEYS:
        vs = sorted({str(r.get(k)) for r in rows if r.get(k) is not None})
        if 0 < len(vs) <= MAX_VALUES:
            vals[k] = vs
    for fname, mod in mods:
        for nm, f in _entry_fns(mod):
            tag = f'{fname}:{nm}'
            try:
                base = _norm(f(rows))
            except Exception as e:                                # noqa: BLE001
                rec['notes'].append(f'{tag} 기준 호출 실패(검사 안 됨): {repr(e)[:80]}')
                continue
            rec['fns'].append(dict(fn=tag, n_gates=len(base)))
            for k, vs in vals.items():
                for v in vs:
                    sub = [r for r in rows if str(r.get(k)) != v]
                    n_del = len(rows) - len(sub)
                    if not n_del:
                        continue
                    try:
                        out = _norm(f(sub))
                    except Exception:                             # noqa: BLE001
                        continue        # 예외 = 민감(자료가 없어진 것을 알아챘다)
                    if len(out) != len(base):
                        continue        # 게이트 목록 구조가 바뀌었다 = 민감
                    for (n0, o0, m0), (n1, o1, m1) in zip(base, out):
                        if n0 != n1 or v not in _tokens(n0) or (o0, m0) != (o1, m1):
                            continue
                        rec['findings'].append(dict(
                            fn=tag, gate=n0, key=k, value=v, ok=o0,
                            msg=m0[:70], n_rows=n_del,
                            grade=('항등' if not o0 else '존재가드없음')))
    return rec


# ═══════════════════════════════════════════════════════════════════════════
def _selftest():
    """일부러 깨진 픽스처 — ★`M4` 를 축소 재현하고, 정상 게이트는 안 찍히는지 본다."""
    import tempfile
    fails = []
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, 'results_x_0.jsonl'), 'w') as fh:
            for a, pc in (('A', False), ('A', False), ('PCARM', True), ('PCARM', True)):
                fh.write(json.dumps(dict(stratum='real', arm=a, unit=f'u{a}',
                                         est_seed=42, pc=pc, val=1.0)) + '\n')
        open(os.path.join(td, 'gates_x.py'), 'w').write(
            'def gates(rows):\n'
            '    ok = [r for r in rows if not r.get("pc")]\n'
            '    out = []\n'
            '    for a in ("A", "PCARM"):\n'
            '        n = sum(1 for r in ok if r["arm"] == a)\n'
            '        out.append((f"G-1 {a} 행수", n == 2, f"{n}/2"))\n'
            '    allr = [r for r in rows if r["arm"] == "PCARM"]\n'
            '    out.append(("G-2 PCARM 값", bool(allr) and all(r["val"] == 1.0 '
            'for r in allr), f"{len(allr)}"))\n'
            '    bad = [r for r in rows if r["arm"] == "GONE"]\n'
            '    out.append(("G-3 GONE 위반 0", not bad, f"{len(bad)}"))\n'
            '    return out\n')
        rec = check_batch(td)
        got = {(f['gate'], f['value'], f['grade']) for f in rec['findings']}
    want_bad = ('G-1 PCARM 행수', 'PCARM', '항등')
    if want_bad not in got:
        fails.append(f'M4 부류를 못 잡았다: {sorted(got)}')
    if any(g[0] == 'G-2 PCARM 값' for g in got):
        fails.append('정상 게이트를 찍었다(오탐): G-2')
    if any(g[0] == 'G-1 A 행수' for g in got):
        fails.append('민감한 게이트를 찍었다(오탐): G-1 A')
    print('자가진단: ' + ('PASS' if not fails else '\n  FAIL '.join([''] + fails)))
    return 1 if fails else 0


def _one(d):
    print('@@GBS@@' + json.dumps(check_batch(os.path.abspath(d)), ensure_ascii=False))
    return 0


def _isolated(d):
    """★배치마다 새 프로세스 — `sys.modules['gates_b26']` 캐시로 **다른 배치의 헬퍼를
    조용히 재사용**하면, 검사기가 *"검사했다"* 고 찍으면서 엉뚱한 자료를 본다."""
    cmd = [sys.executable, os.path.abspath(__file__), '--one', d]
    try:
        pr = subprocess.run(cmd, capture_output=True, text=True,
                            timeout=BATCH_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return dict(batch=os.path.basename(d), rows=-1, loader=None, fns=[],
                    findings=[], notes=[], err=f'시간초과({BATCH_TIMEOUT_S}s)')
    for ln in pr.stdout.splitlines():
        if ln.startswith('@@GBS@@'):
            return json.loads(ln[len('@@GBS@@'):])
    return dict(batch=os.path.basename(d), rows=-1, loader=None, fns=[], findings=[],
                notes=[], err=(pr.stderr.strip().splitlines() or ['출력 없음'])[-1][:200])


def main(argv):
    if '--selftest' in argv:
        return _selftest()
    args = [a for a in argv[1:] if not a.startswith('-')]
    if '--one' in argv:
        return _one(args[0])
    dirs = ([os.path.abspath(a) for a in args] if args else
            sorted({os.path.dirname(p)
                    for p in globmod.glob(os.path.join(TMP, '*', 'gates_*.py'))}))
    print('게이트 부분집합 눈멂 검사 — *"팔 이름을 단 게이트가 그 팔의 행을 실제로 보는가"*')
    print('⛔`check_gate_refs` 는 **키 부재**를 본다. 이 검사기는 **필터 항진**을 본다.\n')
    recs = [_isolated(d) for d in dirs]
    hard = [r for r in recs if any(f['grade'] == '항등' for f in r['findings'])]
    soft = [r for r in recs if any(f['grade'] == '존재가드없음' for f in r['findings'])]
    checked = [r for r in recs if r['fns']]
    skipped = [r for r in recs if not r['fns'] and (r['notes'] or r['err'])]

    for r in recs:
        if not r['findings']:
            continue
        print(f"  {r['batch']}   (행 {r['rows']} · {r['loader']})")
        for f in sorted(r['findings'], key=lambda x: (x['grade'], x['gate'])):
            mk = '⛔' if f['grade'] == '항등' else '⚠'
            print(f"    {mk}[{f['grade']}] {f['fn']} | {f['gate']}")
            print(f"        {f['key']}={f['value']} 행 {f['n_rows']}개를 지워도 "
                  f"ok={f['ok']} msg={f['msg']!r} 가 **비트동일**")
    if hard:
        print('\n⛔**항등 게이트** — 그 팔의 행이 있는데도 안 본다 ⟹ 자료가 무엇이든 통과 불가.')
        print('   ★결과를 보기 **전에** 고쳐야 하는 부류다(개정은 집계 전 동결).')
    if soft:
        print('\n⚠**존재 가드 없음** — 위반 0건 게이트가 그 팔의 행이 사라져도 똑같이 통과한다.')
        print('   지금 자료에서는 무해하다(행이 실제로 있다). 처방은 조건에 **분모**를 넣는 것:')
        print("   `bool(rows_of_arm) and not bad` (본보기 = `gates_m4.py:526` 의 `len(inl) > 0`).")
    if skipped:
        print(f'\n⚠**검사되지 않은 배치 {len(skipped)}개** — ⛔`OK` 로 읽지 마라:')
        for r in skipped:
            why = (r['err'] or (r['notes'][0] if r['notes'] else '?'))
            print(f"   {r['batch']}: {why[:96]}")
    print(f"\n검사된 배치 {len(checked)}/{len(recs)} · ⛔항등 {len(hard)} · ⚠가드없음 {len(soft)}")
    print('⛔이 검사기가 **보지 않는 것**: 행이 0개인 팔 · `gates(rows)` 로 못 부르는 배치 ·'
          ' arm/stratum 밖의 축 · 이름에 팔이 안 적힌 게이트. 「검사했다」의 범위를 넘지 마라.')
    return 1 if hard else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
