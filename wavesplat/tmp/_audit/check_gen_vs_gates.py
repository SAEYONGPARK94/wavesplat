#!/usr/bin/env python3
"""생성-게이트 지연 검사기 — **"위반 위에서 완주한 생성"** 을 사후에 잡는다 (2026-08-29 신설).

## 결함 부류 (저장소 반복 결함 *"계산되는데 검사되지 않는다"* 의 다섯 번째 사례)
`M3_SUPPORT_2026-08-28`: 생성기 `gen_m3.py:_diag` 가 `gt_Hs_4std` 를 **매니페스트와 로그에
이미 쓰고 있었는데** 아무것도 그 값을 `G-8` 문턱(0.25)과 비교하지 않았다.
`gen_m3.log` 09:09:40 `[32/48] … Hs4std=0.889` — **위반이 화면에 찍힌 뒤 46분(16 에피소드)을
더 돌았고**, 러너의 데이터 게이트에서야 정지했다(추정 0회 = 결과가 존재하지 않았다).
앞선 네 사례: `RMSE_near` · 셸 지표 · 채점 창 · `M6` 경보. 진단서 =
`tmp/M3_SUPPORT_2026-08-28/AMENDMENT_1.md` 부록 A §A6 (P2).

## 무엇을 검사하나 — ⛔**이 파일은 문턱을 하나도 소유하지 않는다**
술어는 전부 **배치 자신의 동결 `gates_data()`** 다(손복사 상수 0 — 저장소 반복 결함 1위 회피).
대상 배치(규약: `tmp/*/gates_*.py` 에 `def gates_data` + `def select` 가 있고 같은 디렉터리에
`gen_manifest*.json` 이 있는 = **자기 생성 + 데이터 게이트** 배치)마다:

  C1. **완결성** — `select()` 가 요구하는 에피소드가 디스크에 다 있는가.
      부분 결손 = ⚠미완(생성 진행 중이거나 자료 삭제 — P2 판단 불가), 전부 부재 = SKIP(보관 삭제).
  C2. ★**위반-완주** — 자료가 완결인데 `gates_data()` 가 FAIL 이면, 생성은 위반 **위에서**
      완주한 것이다(생성기가 그 값을 이미 계산했으므로) = **P2 사건, 유일한 ⛔**.
      발화 시 주입 `exists` 프리픽스 이분탐색(mtime 순)으로 첫 위반 에피소드를 지목하고
      낭비(에피소드 수·벽시계)를 산출한다. ⚠이분탐색은 단조 위반(max 형 술어)에 정확하고,
      비단조 술어(평균형)에는 "한 전이점" 만 준다 — 메시지에 그렇게 적는다.
  C3. **배제 원장** — 매니페스트(생성 원장)에 있는데 `select()` 밖인 에피소드
      (= 생성됐다가 배제된 것)는 그 배치의 `FROZEN_REJECTED` 에 원장화돼 있어야 한다.
      원장화됨 = OK 로 **한 줄 공시**(사건의 잔재가 계속 보이게), 없음 = ⚠.
      반대 방향(선택됐는데 매니페스트에 기록 없음 — 재개/skip 재사용)은 정보 표시만 한다
      (`B-26` 에 5건 실측: 이전 부분 실행의 파일 재사용, 무해 관행).

## 소급 census (2026-08-29 실측 — 이 검사기를 만들며 전수 조사한 결과, 동결)
- 규약 배치 3개(B-26 · M1 · M3) 로그의 인쇄값 전수: **사건 1건 = `M3|SPR5|30|seed1702` 가 유일**.
  B-26 194값 max 0.190 < 0.25 · M1 50값 max 0.0166 · M3 50값 중 위반 1(0.2592, [32/48]).
- 비규약 gen 로그 전수(S1·S7·S8·S9·B12·HOSGATE·C_ENCOUNTER·s4·discreteness·indep_data):
  **자기 게이트 위반 0** — 데이터 게이트 자체가 없던 시대다. ⚠단 `HOSGATE` 는
  `mesh=INADEQUATE` 3건이 로그에 찍히고 계속 돌아 **셋 다 채점됐다**(results.jsonl 각 3행;
  BF4_2comp_seed1585 · BF5_2comp_seed1585 · BF5_2comp_seed1586). 당시 게이트가 없었으므로
  "위반" 은 아니고 **P1 부류**다 — `DISCIPLINE_gen_gates_2026-08-29.md` 참조.
  `C_ENCOUNTER` 는 반대의 본보기: 생성기가 G1 을 **인라인으로** 검사했다(64/64 OK).

## 오탐 실측 (2026-08-29, 규약 3배치): ⛔ 0 · ⚠ 0 (M3 는 (라) 적용·재생성 완료 후 기준).
## 비용 실측: 전체 ≈ 9 s (gates_data 가 GT 필드를 되읽는 값이 지배; 이분탐색은 ⛔에서만).
## 종료코드: **⛔(위반-완주)에만 1.** ⚠·SKIP·정보는 0 (상시 경고는 신호를 죽인다).
"""
import glob
import importlib
import json
import os
import re
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))


# ── 발견 ──────────────────────────────────────────────────────────────────────
def discover():
    """규약 배치 = gates_*.py(gates_data+select) + gen_manifest*.json 동거. 반환 [(dir, 모듈명)].
    ⚠규약 밖(gates_data 는 있는데 자기 생성이 없는 배치, 예: M2)은 별도 목록으로 돌려
    침묵 스킵이 되지 않게 한다."""
    hits, outside = [], []
    for gf in sorted(glob.glob(os.path.join(ROOT, 'tmp', '*', 'gates_*.py'))):
        try:
            txt = open(gf, encoding='utf-8', errors='replace').read()
        except OSError:
            continue
        if 'def gates_data' not in txt:
            continue
        d = os.path.dirname(gf)
        mod = os.path.basename(gf)[:-3]
        if 'def select' in txt and glob.glob(os.path.join(d, 'gen_manifest*.json')):
            hits.append((d, mod))
        else:
            outside.append((d, mod))
    return hits, outside


def _load_manifest(bdir):
    man = {}
    for p in sorted(glob.glob(os.path.join(bdir, 'gen_manifest*.json'))):
        try:
            man.update(json.load(open(p)))
        except (OSError, ValueError):
            pass
    return man


def _tok(s):
    """토큰 지문 — 매니페스트 키 형식이 배치마다 다르다(B-26 = 파일명 전체 ·
    M1 = 'LIN/episode_….h5' 인데 select() 의 episode 는 'LIN|15comp_seed1681' ·
    M3 = '…@CELL'). ⟹ 키의 토큰 집합이 **에피소드 path 의 토큰 집합에 부분집합**이면
    같은 에피소드로 본다(seed·arm/cell 토큰이 유일성을 보장한다). 08-29 초판이
    문자열 정규화 대조로 M1 에서 오탐 96건을 낸 것의 수리 — 자가진단 픽스처가 지킨다."""
    return {t for t in re.findall(r'[A-Za-z0-9]+', str(s).lower()) if len(t) > 1}


def _ledgered(rec, key, frozen_rejected):
    """배제 원장 대조: FROZEN_REJECTED 의 어느 튜플이든 그 원소 전부가
    매니페스트 레코드 값(또는 키 문자열)에 나타나면 원장화된 것으로 본다.
    (형식이 배치마다 다르다 — M1 (ncomp,seed) · M3 (cell,ncomp,seed).)"""
    vals = {str(v) for v in rec.values() if isinstance(v, (str, int, float))}
    for t in frozen_rejected or ():
        t = t if isinstance(t, (tuple, list)) else (t,)
        if all(str(x) in vals or str(x) in key for x in t):
            return True
    return False


# ── 핵심 판정 (실배치·자가진단 공용 — I/O 는 전부 주입 가능) ─────────────────
def audit_batch(mod, name, manifest, exists=os.path.exists, mtime=None):
    """반환: [(level, text)] — level ∈ {'OK','정보','⚠','⛔','SKIP'}."""
    mtime = mtime or (lambda p: os.path.getmtime(p) if os.path.exists(p) else 0.0)
    out = []
    sel = mod.select()
    eps = sel[0] if isinstance(sel, tuple) else sel
    paths = [e.get('path') for e in eps]
    if any(p is None for p in paths):
        out.append(('⚠', f'{name}: select() 에 path 없음 — 규약 이탈, 검사 불가'))
        return out
    have = [p for p in paths if exists(p)]

    # C1 완결성
    if not have:
        out.append(('SKIP', f'{name}: 자료 전부 부재 ({len(paths)} 선택) — 보관 삭제로 보고 건너뜀'))
        return out
    if len(have) < len(paths):
        out.append(('⚠', f'{name}: 미완 {len(have)}/{len(paths)} — 생성 진행 중이거나 부분 삭제. '
                          f'P2(위반-완주) 판단은 완결 후에만 가능하다'))
        return out

    # C2 위반-완주 (유일한 ⛔)
    gd = mod.gates_data()
    fails = [(n, msg) for n, ok, msg in gd if not ok]
    if fails:
        loc = _localize(mod, {n for n, _ in fails}, sorted(paths, key=lambda p: (mtime(p), p)),
                        mtime)
        for n, msg in fails:
            out.append(('⛔', f'{name}: 자료 완결인데 데이터 게이트 FAIL — 생성이 위반 위에서 '
                              f'완주했다 (P2). [{n}] {msg}{loc}'))
    else:
        out.append(('OK', f'{name}: 완결 {len(paths)} · 데이터 게이트 {len(gd)}/{len(gd)}'))

    # C3 배제 원장
    man_ok = {k: v for k, v in manifest.items() if isinstance(v, dict) and v.get('status') == 'ok'}
    path_toks = [(e, _tok(e['path'])) for e in eps]
    matched_eps = set()
    excess = {}
    for k, v in man_ok.items():
        kt = _tok(k)
        hit = [e for e, pt in path_toks if kt <= pt]
        if hit:
            matched_eps.update(id(e) for e in hit)
        else:
            excess[k] = v
    missing = [e for e, _ in path_toks if id(e) not in matched_eps]
    fr = getattr(mod, 'FROZEN_REJECTED', ())
    for k, rec in sorted(excess.items()):
        if _ledgered(rec, k, fr):
            out.append(('OK', f'{name}: 생성-후-배제 1건 원장화됨 — {k} ∈ FROZEN_REJECTED'))
        else:
            out.append(('⚠', f'{name}: 매니페스트에 있는데 select() 밖 + FROZEN_REJECTED 에도 '
                             f'없음 — 원장 없는 배제: {k}'))
    if missing:
        out.append(('정보', f'{name}: 선택 {len(missing)}건이 매니페스트 미기록 '
                            f'(재개/skip 재사용 — B-26 에서 5건 실측된 무해 관행)'))
    return out


def _localize(mod, fail_names, paths_mtime_order, mtime):
    """⛔에서만: 주입 exists 프리픽스 이분탐색으로 첫 위반 에피소드를 지목한다.
    술어는 여전히 배치 자신의 gates_data — 이 함수는 exists 만 갈아 끼운다."""
    n = len(paths_mtime_order)

    def fails_at(k):
        allowed = set(paths_mtime_order[:k])
        try:
            gd = mod.gates_data(exists=lambda p: p in allowed)
        except Exception:                                          # noqa: BLE001
            return None
        return any(nm in fail_names and not ok for nm, ok, _ in gd)

    lo, hi = 1, n                        # fails_at(n) == True (완결 전수에서 FAIL 확인됨)
    while lo < hi:
        mid = (lo + hi) // 2
        r = fails_at(mid)
        if r is None:
            return ' · 지목 불가(프리픽스 평가 예외)'
        lo, hi = (lo, mid) if r else (mid + 1, hi)
    p = paths_mtime_order[lo - 1]
    waste_ep = n - lo
    try:
        waste_min = (mtime(paths_mtime_order[-1]) - mtime(p)) / 60.0
        wm = f' · 이후 {waste_ep} 에피소드 ≈{waste_min:.0f}분 더 생성됨'
    except Exception:                                              # noqa: BLE001
        wm = f' · 이후 {waste_ep} 에피소드 더 생성됨'
    return (f' · 첫 위반 프리픽스(이분탐색; max 형 술어에 정확) = {lo}/{n} '
            f'{os.path.basename(str(p))}{wm}')


# ── 실행 ──────────────────────────────────────────────────────────────────────
def main():
    os.chdir(ROOT)
    hits, outside = discover()
    print(f'생성-게이트 지연 검사 — 규약 배치 {len(hits)}개 '
          f'(gates_data+select+gen_manifest), 규약 밖 {len(outside)}개')
    worst = 0
    for bdir, modname in hits:
        rel = os.path.relpath(bdir, ROOT)
        sys.path.insert(0, bdir)
        try:
            t0 = time.time()
            mod = importlib.import_module(modname)
            rows = audit_batch(mod, rel, _load_manifest(bdir))
            el = time.time() - t0
        except Exception as exc:                                   # noqa: BLE001
            rows, el = [('⚠', f'{rel}: 임포트/실행 실패 {exc!r} — ★침묵 스킵 아님, 사람이 볼 것')], 0
        finally:
            sys.path.remove(bdir)
        for lv, txt in rows:
            print(f'  [{lv}] {txt}' + (f'  ({el:.1f}s)' if lv == 'OK' and '게이트' in txt else ''))
            worst = max(worst, 2 if lv == '⛔' else 1 if lv == '⚠' else 0)
    for bdir, modname in outside:
        print(f'  [정보] {os.path.relpath(bdir, ROOT)}: gates_data 는 있으나 자기 생성이 없다'
              f'(gen_manifest 부재) — 이 검사기 대상 아님 ({modname})')
    print('요약: ' + ('⛔위반-완주 있음' if worst == 2 else
                     '⚠경고 있음(종료코드 0 — 출력을 읽어라)' if worst == 1 else '✅전부 통과'))
    return 1 if worst == 2 else 0


# ── 자가진단 (run_all.sh §0 규약 — 방벽이 스스로 틀리는 것을 막는 회귀 픽스처) ──
def selftest():
    import types

    def fx(n_exist, devs, man_extra=None, frozen=()):
        """규약 모양의 가짜 배치: 에피소드 4, dev 술어 0.25 (픽스처 상수 — 실배치와 무관)."""
        eps = [dict(episode=f'ep{i}', path=f'/fx/episode_ep{i}.h5', cell='C', ncomp=15,
                    seed=100 + i) for i in range(4)]
        on_disk = {e['path'] for e in eps[:n_exist]}

        def gates_data(read=None, exists=None):
            ex = exists or (lambda p: p in on_disk)
            have = [e for e in eps if ex(e['path'])]
            vals = [devs[int(e['episode'][2:])] for e in have]
            return [('G-6 개수', len(have) == 4, f'{len(have)}/4'),
                    ('G-8 max dev', bool(vals) and max(vals) <= 0.25,
                     f'max {max(vals):.3f}' if vals else '')]

        # ⚠키 형식을 일부러 select() 의 'episode' 와 다르게 둔다 — M1 오탐 96건의 회귀 픽스처
        #   (매니페스트 'LIN/episode_….h5' vs episode 'LIN|15comp_seed1681').
        man = {f'FX/EPISODE_{e["episode"]}.H5':
               dict(status='ok', **{k: e[k] for k in ('cell', 'ncomp', 'seed')})
               for e in eps[:n_exist]}
        man.update(man_extra or {})
        mod = types.SimpleNamespace(select=lambda: (eps, []), gates_data=gates_data,
                                    FROZEN_REJECTED=frozen)
        return mod, man, (lambda p: p in on_disk), (lambda p: float(p[-4]) if p[-4].isdigit() else 0.0)

    def levels(rows):
        return [lv for lv, _ in rows]

    ok = True

    def chk(label, cond):
        nonlocal ok
        print(f'  [{"PASS" if cond else "FAIL"}] {label}')
        ok &= cond

    mtime = lambda p: float(re.search(r'(\d+)', os.path.basename(p)).group(1))  # noqa: E731
    # 1 깨끗: 완결·통과·원장 일치 → OK 만
    m, man, ex, _ = fx(4, [0.1, 0.1, 0.1, 0.1])
    chk('깨끗 → OK 만', levels(audit_batch(m, 'fx', man, ex, mtime)) == ['OK'])
    # 2 위반-완주: ep2(3번째 생성) 가 0.30 → ⛔ + 이분탐색이 3/4 지목
    m, man, ex, _ = fx(4, [0.1, 0.1, 0.30, 0.1])
    rows = audit_batch(m, 'fx', man, ex, mtime)
    chk('위반-완주 → ⛔', '⛔' in levels(rows))
    chk('이분탐색 = 3/4 지목', any('3/4' in t and 'ep2' in t for _, t in rows))
    # 3 미완: 3/4 만 존재 → ⚠미완, ⛔ 없음
    m, man, ex, _ = fx(3, [0.1, 0.1, 0.30, 0.1])
    rows = audit_batch(m, 'fx', man, ex, mtime)
    chk('미완 → ⚠, ⛔ 없음', levels(rows) == ['⚠'] and '미완' in rows[0][1])
    # 4 배제 원장화: 매니페스트 여분 1 + FROZEN_REJECTED 일치 → OK 공시
    extra = {'epX': dict(status='ok', cell='C', ncomp=15, seed=999)}
    m, man, ex, _ = fx(4, [0.1] * 4, man_extra=extra, frozen=(('C', 15, 999),))
    rows = audit_batch(m, 'fx', man, ex, mtime)
    chk('배제 원장화 → OK 공시', any(lv == 'OK' and 'FROZEN_REJECTED' in t for lv, t in rows)
        and '⚠' not in levels(rows))
    # 5 원장 없는 배제 → ⚠
    m, man, ex, _ = fx(4, [0.1] * 4, man_extra=extra, frozen=())
    chk('원장 없는 배제 → ⚠', any(lv == '⚠' and '원장 없는 배제' in t
                                    for lv, t in audit_batch(m, 'fx', man, ex, mtime)))
    print('자가진단', '통과' if ok else '실패')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(selftest() if '--selftest' in sys.argv else main())
