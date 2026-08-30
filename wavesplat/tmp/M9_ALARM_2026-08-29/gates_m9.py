"""M-9 게이트 단일출처 — 비동기 재적합 no-op 경보 수리의 판정 규약.

(2026-08-30 작성 · ⛔**결과 전 동결** — 예상표·계약·라벨·셀 정의의 유일한 권위.
 동결 후 이 파일을 고치면 그 판정은 무효다. md5 는 FROZEN_md5.txt — 수리 착지 후
 `sequential_update.py` 줄은 **덮어쓰지 말고 새 줄을 덧붙인다**.)

# 무엇을 재는가 (PREREG.md §3)
RG-1  `M6` 동결 360행 재생(**런 0회**): 수리된 경보가 같은 표본에서 무엇을 잡는지를
      **진실 열 = 하네스 되읽기**(`applied_after_lead`·`applied_pre_wait`, M6 의 ③채널)와
      **발화 열 = 제품 술어** 의 독립 채널 대조로 잰다 — 경보의 자기 계수로 경보를
      채점하지 않는다.
RG-0  라이브 스모크 4셀의 수리 전/후 **수치 비트 동일**(경고 억제) — 관측성 밖 누수 감지.
RG-2  (a) skip 소거 (b) (2,0) 순서 함정 (c) 건강 침묵 — RG-0 셀 3개가 게이트로 겸하고,
      pytest E절 이식본은 `rg2_tests_snippet_e.py`(수리 커밋에서 이동).
RG-3  `bank_refit_health()` 권위(②) — 새 키 2개 + 기존 키·`effective` 의미 불변 +
      오프라인 늦착지 54런 플래그.

# ★함정 차단 (PREREG §2-① off-by-one · "같은 기능 두 곳 구현" 부류)
 1) 발화 열은 post 모드에서 **오직 제품 임포트 함수** `noop_alarm_due(n_triggers, n_applied)`
    가 낸다(모듈 수준 순수 함수, 이번 사건 **포함** 계수). 검사기는 술어를 판정에 재구현하지
    않는다.
 2) 이 파일의 `_contract_due` 는 **예상표의 유도식**일 뿐이다(pre 모드 = 동결 수치 재확인
    전용). 제품 함수가 계약에서 이탈하면 M-3(프로브·사건별 대조)와 M-4~M-7(동결 수치)이
    죽는다 — 유도식이 이탈을 가려 줄 수 없는 구조다.
 3) off-by-one 함정: 동결형 = "**이번 사건 포함** 트리거 수 ≥ 2 ∧ applied == 0".
    프로브 `(2,0)→True` 가 증가-전-계수 구현을 직접 잡고, `mutate_m9.py` 가 그 변형을
    먹여 게이트가 실제로 죽는지 확인한다((2,0) 부류 15런 = 구 경보가 유일하게 잡던 것).

# 동결 예상표 (PREREG §3 RG-1 — 2026-08-30 동결 jsonl 재유도로 재확인 완료)
  포착 131/131 · 무경고 0 · 건강 대역 0/72 · 대역별 [0,0,20,33,33,33,33,33]=185 ·
  늦착지 발화 54 {P2:13, P3:23, P4:18} (구 오발 3 포함) · 구 경보 재생 일치 288/288 ·
  구 발화 18 {P2:10, P3:8} · 구 오발 3 = (AA14,3)(BS11b,1)(BS13b,3).
"""
import copy
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
M6_DIR = os.path.join(ROOT, 'tmp', 'M6_PACING_2026-08-28')
for _p in (M6_DIR,
           os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23'),
           os.path.join(ROOT, 'tmp', 'B3_REFIT_2026-08-23'),
           os.path.join(ROOT, 'tmp', 'bankmmse_port_2026-07-29'),
           ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import gates_m6 as G6  # noqa: E402  단일출처: ARM_ORDER·RECORDS·WINDOWS·BASE_PINS·TAU·SEED·NOISE

# ── 동결 입력·경로 ───────────────────────────────────────────────────────────
M6_JSONL = os.path.join(M6_DIR, 'results_m6.jsonl')
M6_JSONL_MD5 = '54224699e20670bd60ed8ea7f29e1697'      # 2026-08-30 실측 핀 (동결 자료)
SEQ_UPD_PATH = os.path.join(ROOT, 'wavesplat', 'optimizer', 'sequential_update.py')
SEQ_UPD_MD5_PRE = '1b473e066cb02e43f39a9887a090c7d4'   # 수리 **전** — post 에서는 달라야 정상

RG1_REPORT = os.path.join(HERE, 'rg1_report.json')
RG0_REPORT = {'pre': os.path.join(HERE, 'rg0_pre.json'),
              'post': os.path.join(HERE, 'rg0_post.json')}
MUT_REPORT = os.path.join(HERE, 'mutation_report.json')
VERDICT = os.path.join(HERE, 'verdict_m9.json')

ASYNC_ORDER = [a for a in G6.ARM_ORDER if a.startswith('ASYNC5_')]   # 8 대역, 러너 순서

# ── 동결 계약 (수리가 지켜야 하는 인터페이스) ────────────────────────────────
# 제품 술어: 모듈 수준 순수 함수  noop_alarm_due(n_triggers, n_applied) -> bool
#   n_triggers = 이번 사건 **포함** 트리거 사건 수(실행 kick + skip),  n_applied = 그 시점 적용 수.
PREDICATE_NAME = 'noop_alarm_due'
PREDICATE_PROBES = [  # (입력, 기대) — I/O 예시이지 재구현이 아니다
    ((1, 0), False),   # 첫 트리거는 침묵 (아직 판단 근거 없음)
    ((2, 0), True),    # ★off-by-one 함정의 직접 검사: (2,0) 부류 15런의 보존
    ((2, 1), False),   # 적용이 있으면 침묵
    ((3, 0), True),
    ((2, 5), False),
    ((17, 0), True),
]
ALARM_WARN_TOKEN = 'bank_refit_async'   # 수리 후 경고 문안도 이 토큰을 포함해야 한다(동결 인터페이스)
HEALTH_KEYS_BASE = {'enabled', 'kicks', 'applied', 'skipped', 'in_flight', 'effective'}
HEALTH_KEYS_NEW = {'first_apply_data_t', 'last_fit_wall_s'}          # ② 가 추가하는 키
LATE_LEAD_MIN = 10.0    # RG-3 오프라인 늦착지 규칙(결과 전 동결): first_apply_lead ≥ 10 = 플래그

# ── 동결 예상표 (결과 전 · 2026-08-30 동결 jsonl 재유도로 재확인) ────────────
EXPECTED = dict(
    n_rows=360, n_async=288, events_per_run=2,
    old_replay_match=288,
    old_fires=18, old_fires_by_band={'ASYNC5_P2': 10, 'ASYNC5_P3': 8},
    old_false_fires=[('AA14', 3), ('BS11b', 1), ('BS13b', 3)],
    full_noop=131, capture=131, silent_noop=0,
    healthy_fires=0, healthy_n=72,
    fires_by_band=[0, 0, 20, 33, 33, 33, 33, 33], fires_total=185,
    late_fires=54, late_fires_by_band={'ASYNC5_P2': 13, 'ASYNC5_P3': 23, 'ASYNC5_P4': 18},
)

# ── RG-0 셀 (결과 전 동결 — 레코드·창·구성·조인 지점·기대값) ─────────────────
RG0_RECORD, RG0_WIN = 'AA14', 1
RG0_CELLS = ['SYNC5', 'ASYNC_NOLAND', 'ASYNC_LAND10', 'ASYNC_LAND6']
RG0_CELL_CFG = {   # ⛔빈 dict 팔 금지 — 매 셀이 두 키를 명시한다 (BASE_PINS 는 러너가 별도 명시)
    'SYNC5':        dict(bank_refit_every_s=5.0, bank_refit_async=False),
    'ASYNC_NOLAND': dict(bank_refit_every_s=5.0, bank_refit_async=True),
    'ASYNC_LAND10': dict(bank_refit_every_s=5.0, bank_refit_async=True),
    'ASYNC_LAND6':  dict(bank_refit_every_s=5.0, bank_refit_async=True),
}
# 하네스가 이 lead 의 update **직전에** 배경 스레드를 join 한다 → 착지/kick 이 결정적이 된다.
RG0_JOIN_BEFORE_LEAD = {'ASYNC_LAND10': 10.0, 'ASYNC_LAND6': 6.0}
# 발화 기대 — pre(구 코드)/post(수리 후)가 갈리는 곳이 정확히 NOLAND(skip 소거 부류)다.
RG0_EXPECT_FIRE = {
    'pre':  {'SYNC5': 0, 'ASYNC_NOLAND': 0, 'ASYNC_LAND10': 1, 'ASYNC_LAND6': 0},
    'post': {'SYNC5': 0, 'ASYNC_NOLAND': 1, 'ASYNC_LAND10': 1, 'ASYNC_LAND6': 0},
}
# 계수기 기대 (pre == post — §5-2: 스케줄링은 이번에 안 고친다. LAND10 의 applied=0 은
# 완료-미수집 적합을 새 kick 이 덮어쓰는 현행 거동이다: M6 오발 3건의 first_apply=14 가 그 증거)
RG0_EXPECT_COUNTERS = {
    'SYNC5':        dict(kicks=0, skipped=0, applied=0, first_apply=None),
    'ASYNC_NOLAND': dict(kicks=1, skipped=1, applied=0, first_apply=None),
    'ASYNC_LAND10': dict(kicks=2, skipped=0, applied=0, first_apply=None),
    'ASYNC_LAND6':  dict(kicks=2, skipped=0, applied=1, first_apply=6.0),
}


# ── 입력 로더 (0행 가드) ─────────────────────────────────────────────────────
def load_rows(path=M6_JSONL):
    blob = open(path, 'rb').read()
    md5 = hashlib.md5(blob).hexdigest()
    rows = [json.loads(ln) for ln in blob.decode().splitlines() if ln.strip()]
    if not rows:
        raise SystemExit('⛔0행 가드: 동결 입력이 비었다 — ' + path)
    return rows, md5


# ── 러너 모형 (변형 하네스의 단일출처 — 게이트가 아니라 러너를 모형한다) ────
def expected_jobs():
    """M6 러너가 만들었어야 하는 행 정체 360개 (팔×레코드×창) — gates_m6 에서 유도."""
    return [(arm, rec, win) for arm in G6.ARM_ORDER
            for rec in G6.RECORDS for win in G6.WINDOWS]


def row_identity(r):
    return (r['cell'], r['record'], r['win'])


# ── RG-1 사건 재구성 (PREREG §3 규칙 그대로) ─────────────────────────────────
def trigger_events(r):
    """동결 행 → 트리거 사건열 [(n_triggers_incl, kind, applied_at_check), ...].

    사건 1 = kick@lead 5 (검사 시점 applied = `applied_after_lead["4.0"]`),
    사건 2 = kick|skip@lead 10 (검사 시점 applied = `applied_after_lead["9.0"]` —
    lead 10 update 안에서 kick 검사가 collect 보다 **앞**이므로(:1316-1319 → :1321)
    lead 9 종료 시점 값이 검사 시점 값이다). 전제 = G-2 실측 k+s ≡ 2."""
    k, s = r['async_kicks'], r['async_skipped']
    if k + s != EXPECTED['events_per_run'] or k < 1:
        raise ValueError(f'k+s!=2 또는 k<1: {row_identity(r)} (k={k}, s={s})')
    a4 = r['applied_after_lead'].get('4.0')
    a9 = r['applied_after_lead'].get('9.0')
    if a4 is None or a9 is None:
        raise ValueError(f'applied_after_lead 키 4.0/9.0 부재: {row_identity(r)}')
    if a4 != 0:
        raise ValueError(f'사건1 시점 applied!=0: {row_identity(r)} (a4={a4})')
    return [(1, 'kick', a4), (2, 'kick' if k == 2 else 'skip', a9)]


def old_alarm_reference(events):
    """⚠구 계약의 참조 재현 — **은퇴 대상** 계약이며 SD-1 재구성 검증 전용.

    구 코드: 검사는 조기 반환 **뒤**(= 실행 kick 에서만), 전제 = 증가 전 `kicks ≥ 1`,
    한 번만. 신 술어는 절대 여기 두지 않는다 — post 모드의 발화 열은 제품 임포트다."""
    kicks = 0
    for _n, kind, applied in events:
        if kind != 'kick':
            continue                     # skip 은 검사 자체가 없다 (조기 반환)
        if kicks >= 1 and applied == 0:
            return True
        kicks += 1
    return False


def _contract_due(n_triggers, n_applied):
    """§2-① 동결 계약의 **예상표 유도식** (pre 모드 전용).

    ⛔판정의 발화 열이 아니다 — post 모드에서 발화 열은 오직 제품 `noop_alarm_due` 다.
    제품이 이 식과 다르면 M-3/M-3b/M-4~M-7 이 죽는다(유도식이 이탈을 못 가린다)."""
    return n_triggers >= 2 and n_applied == 0


def replay_alarm(events, predicate):
    """사건열을 술어에 차례로 먹인다 — 인스턴스당 1회(once-only) 규약 포함."""
    for n, _kind, applied in events:
        if predicate(n, applied):
            return True
    return False


def import_product_predicate():
    """제품 술어 임포트. 없으면 (None, 사유) = 수리 미착지(pre 모드)."""
    try:
        mod = __import__('wavesplat.optimizer.sequential_update',
                         fromlist=[PREDICATE_NAME])
    except Exception as e:                                  # noqa: BLE001
        return None, f'임포트 실패: {e!r}'
    fn = getattr(mod, PREDICATE_NAME, None)
    if fn is None:
        return None, f'{PREDICATE_NAME} 부재 (수리 미착지 = pre 모드)'
    return fn, f'wavesplat.optimizer.sequential_update.{PREDICATE_NAME}'


# ── RG-1 판정 (단일출처 — regress/mutate 가 이 함수만 부른다) ────────────────
def rg1_run(rows, md5, predicate=None, predicate_src=None):
    if not rows:
        raise SystemExit('⛔0행 가드: rg1_run 에 빈 행 목록')
    mode = 'post' if predicate is not None else 'pre'
    gates, derived = [], {}

    def g(name, ok, note=''):
        gates.append(dict(name=name, ok=ok, note=str(note)))

    g('M-0a 동결 입력 md5 핀', md5 == M6_JSONL_MD5, md5)
    ids = sorted(row_identity(r) for r in rows)
    g('M-0b 행 정체 = 러너 모형(expected_jobs 다중집합)',
      ids == sorted(expected_jobs()), f'{len(rows)}행')
    g('M-0c 에러 행 0', not any('error' in r for r in rows),
      sum(1 for r in rows if 'error' in r))
    asy = [r for r in rows if r['cell'].startswith('ASYNC5_')]
    g('M-0d ASYNC 층 288', len(asy) == EXPECTED['n_async'], len(asy))

    evs, ev_err = {}, []
    for r in asy:
        try:
            evs[id(r)] = trigger_events(r)
        except ValueError as e:
            ev_err.append(str(e))
    g('M-1 사건 재구성 전건 (k+s=2 · 키 4.0/9.0 · 사건1 applied=0)',
      not ev_err, '; '.join(ev_err[:2]))
    ok_rows = [r for r in asy if id(r) in evs]

    # M-2 ★SD-1: 구 경보를 같은 재구성으로 재생 → 동결 발화열과 전건 일치 = 재구성 검증
    old = {id(r): old_alarm_reference(evs[id(r)]) for r in ok_rows}
    n_match = sum(1 for r in ok_rows if old[id(r)] == bool(r['noop_warning_fired']))
    g('M-2 ★구 경보 재생 = 동결 발화열 (재구성 검증)',
      n_match == EXPECTED['old_replay_match'] and len(ok_rows) == EXPECTED['n_async'],
      f'{n_match}/{len(ok_rows)}')
    of_band = {c: sum(1 for r in ok_rows if r['cell'] == c and old[id(r)])
               for c in ASYNC_ORDER}
    of_band = {c: n for c, n in of_band.items() if n}
    ff = sorted((r['record'], r['win']) for r in ok_rows
                if old[id(r)] and r['applied_pre_wait'] > 0)
    g('M-2b 구 발화 분해 {P2:10, P3:8} · 오발 3건 정체',
      of_band == EXPECTED['old_fires_by_band']
      and ff == sorted(EXPECTED['old_false_fires']),
      f'{of_band} · 오발 {ff}')

    # 발화 열 — post: 제품 술어 / pre: 계약 유도식(예상표 재확인 전용)
    if mode == 'post':
        import inspect
        probes = [(inp, want, bool(predicate(*inp))) for inp, want in PREDICATE_PROBES]
        probe_ok = all(got == want for _i, want, got in probes)
        shape_ok = (inspect.isfunction(predicate)
                    and len(inspect.signature(predicate).parameters) == 2
                    and predicate.__module__ == 'wavesplat.optimizer.sequential_update')
        g('M-3 제품 술어: 모듈 순수함수·2인자·프로브 6/6',
          probe_ok and shape_ok,
          f'src={predicate_src} probes={[(i, w, g_) for i, w, g_ in probes if w != g_]}')
        fire = {id(r): replay_alarm(evs[id(r)], predicate) for r in ok_rows}
        contract = {id(r): replay_alarm(evs[id(r)], _contract_due) for r in ok_rows}
        n_dis = sum(1 for r in ok_rows if fire[id(r)] != contract[id(r)])
        g('M-3b 제품 술어 = 계약 유도 (288행 사건별)', n_dis == 0, f'불일치 {n_dis}')
    else:
        g('M-3 제품 술어: 모듈 순수함수·2인자·프로브 6/6', None, '수리 미착지 (pre)')
        g('M-3b 제품 술어 = 계약 유도 (288행 사건별)', None, '수리 미착지 (pre)')
        fire = {id(r): replay_alarm(evs[id(r)], _contract_due) for r in ok_rows}

    fire_src = ('제품 noop_alarm_due' if mode == 'post' else '계약 유도식(pre — 예상표 재확인)')
    noop = [r for r in ok_rows if r['applied_pre_wait'] == 0]
    n_noop = len(noop) if noop else None                      # 커버리지 0 → None (0 금지)
    n_cap = sum(1 for r in noop if fire[id(r)]) if noop else None
    g('M-4 전면 no-op 포착 131/131 · 무경고 0',
      n_noop == EXPECTED['full_noop'] and n_cap == EXPECTED['capture'],
      f'포착 {n_cap}/{n_noop} · 발화열={fire_src}')

    healthy = [r for r in ok_rows if r['cell'] in ('ASYNC5_P0.5', 'ASYNC5_P1')]
    n_h = len(healthy) if healthy else None
    n_hf = sum(1 for r in healthy if fire[id(r)]) if healthy else None
    g('M-5 건강 대역(0.5×·1×) 발화 0/72 (신호 보존)',
      n_h == EXPECTED['healthy_n'] and n_hf == EXPECTED['healthy_fires'],
      f'발화 {n_hf}/{n_h}')

    by_band = []
    for c in ASYNC_ORDER:
        band = [r for r in ok_rows if r['cell'] == c]
        by_band.append(sum(1 for r in band if fire[id(r)]) if band else None)
    g('M-6 대역별 발화 = [0,0,20,33,33,33,33,33] (총 185)',
      by_band == EXPECTED['fires_by_band'],
      f'{by_band} 총 {sum(b for b in by_band if b is not None)}')

    late_fire = {id(r) for r in ok_rows if fire[id(r)] and r['applied_pre_wait'] > 0}
    late_land = {id(r) for r in ok_rows if r['applied_pre_wait'] > 0
                 and r.get('first_apply_lead') is not None
                 and r['first_apply_lead'] >= LATE_LEAD_MIN}
    lf_band = {c: sum(1 for r in ok_rows if id(r) in late_fire and r['cell'] == c)
               for c in ASYNC_ORDER}
    lf_band = {c: n for c, n in lf_band.items() if n}
    g('M-7 늦착지 발화 54 {P2:13,P3:23,P4:18} ∧ = 늦착지 집합(원리적 오발의 전량 공시)',
      len(late_fire) == EXPECTED['late_fires']
      and lf_band == EXPECTED['late_fires_by_band'] and late_fire == late_land,
      f'{len(late_fire)}건 {lf_band} 집합동일={late_fire == late_land}')

    eff_ok = all(bool(r['health_pre_wait']['effective']) == (r['applied_pre_wait'] > 0)
                 for r in ok_rows)
    late_eff = ([bool(r['health_pre_wait']['effective']) for r in ok_rows
                 if id(r) in late_land] or [None])
    g('M-8 RG-3 오프라인: first_apply≥10 플래그 54 · 그 전건 구 effective True '
      '· effective⟺applied>0',
      len(late_land) == EXPECTED['late_fires'] and all(late_eff) and eff_ok,
      f'플래그 {len(late_land)} · effective동치 {eff_ok}')

    derived = dict(fires_by_band=by_band, fire_src=fire_src,
                   n_capture=n_cap, n_noop=n_noop, healthy_fires=n_hf,
                   late_fires=len(late_fire), old_fires_by_band=of_band)
    return dict(mode=mode, gates=gates, derived=derived, md5_jsonl=md5,
                predicate_src=predicate_src)


# ── RG-0 판정 ────────────────────────────────────────────────────────────────
def rg0_gates(tag, rep, pre_rep=None):
    """rep = {'cells': {...}, 'meta': {...}} (regress_m9._rg0_run_cells 산출)."""
    gates = []

    def g(name, ok, note=''):
        gates.append(dict(name=name, ok=ok, note=str(note)))

    cells = rep.get('cells', {})
    g('R-0 셀 4/4 존재', sorted(cells) == sorted(RG0_CELLS), sorted(cells))
    bad_fire, bad_cnt = [], []
    for c in RG0_CELLS:
        if c not in cells:
            continue
        cc = cells[c]
        if cc['fired_n'] != RG0_EXPECT_FIRE[tag][c]:
            bad_fire.append((c, cc['fired_n']))
        want = RG0_EXPECT_COUNTERS[c]
        got = dict(kicks=cc['counters']['kicks'], skipped=cc['counters']['skipped'],
                   applied=cc['counters']['applied'], first_apply=cc['first_apply'])
        if got != want:
            bad_cnt.append((c, got))
    g(f'R-1 발화 수 = 기대({tag})', not bad_fire, bad_fire)
    g('R-2 계수기 = 기대 (kicks/skipped/applied/first_apply — 스케줄링 불변 §5-2)',
      not bad_cnt, bad_cnt)
    if tag == 'post':
        if pre_rep is None:
            g('R-3 ★수치 비트 동일 (per_lead + 장 md5, pre 대조)', None, 'rg0_pre.json 부재')
        else:
            diff = []
            for c in RG0_CELLS:
                a, b = pre_rep['cells'].get(c), cells.get(c)
                if a is None or b is None:
                    diff.append((c, '셀 부재'))
                elif a['per_lead'] != b['per_lead'] or a['field_md5s'] != b['field_md5s']:
                    diff.append((c, '불일치'))
            g('R-3 ★수치 비트 동일 (per_lead + 장 md5, pre 대조)', not diff, diff)
        md5_now = rep['meta'].get('md5_sequential_update')
        g('R-3b 수리 착지 확인 (모듈 md5 ≠ pre 핀)', md5_now != SEQ_UPD_MD5_PRE, md5_now)
    else:
        g('R-3 ★수치 비트 동일 (post 에서 대조)', None, '보류')
        md5_now = rep['meta'].get('md5_sequential_update')
        g('R-3b pre 모듈 md5 = 핀', md5_now == SEQ_UPD_MD5_PRE, md5_now)
    key_bad = []
    for c in RG0_CELLS:
        if c not in cells:
            continue
        keys = set(cells[c]['health_keys'])
        if not HEALTH_KEYS_BASE <= keys:
            key_bad.append((c, 'base 키 결손'))
        has_new = bool(HEALTH_KEYS_NEW & keys)
        if tag == 'pre' and has_new:
            key_bad.append((c, '⚠pre 인데 새 키 존재 — 수리가 이미 착지?'))
        if tag == 'post' and not HEALTH_KEYS_NEW <= keys:
            key_bad.append((c, 'post 인데 새 키 결손'))
    g('R-4 health 키 계약 (base 항상 · new pre부재/post존재)', not key_bad, key_bad)
    if tag == 'post':
        sem_bad = []
        h6 = cells.get('ASYNC_LAND6', {}).get('health', {})
        hno = cells.get('ASYNC_NOLAND', {}).get('health', {})
        t0 = rep['meta'].get('t0')
        if h6:
            fat = h6.get('first_apply_data_t')
            if fat is None or t0 is None or abs(fat - (t0 + 5.0)) > 1e-3:
                sem_bad.append(('LAND6 first_apply_data_t', fat, '기대', (t0 or 0) + 5.0))
            if not (h6.get('last_fit_wall_s') or 0) > 0:
                sem_bad.append(('LAND6 last_fit_wall_s', h6.get('last_fit_wall_s')))
            if h6.get('effective') is not True:
                sem_bad.append(('LAND6 effective 의미 불변(k>0∧a>0=True)', h6.get('effective')))
        if hno and hno.get('first_apply_data_t') is not None:
            sem_bad.append(('NOLAND first_apply_data_t 는 None 이어야', hno.get('first_apply_data_t')))
        if hno and hno.get('effective') is not False:
            sem_bad.append(('NOLAND effective 의미 불변(a=0=False)', hno.get('effective')))
        g('R-5 RG-3 라이브 의미 (착지 자료시각=트리거1 창 상한 · 단가>0 · effective 불변)',
          not sem_bad, sem_bad)
    return gates


# ── 라벨 (결과 전 동결 — PREREG §6-3 의 갈래 포함) ───────────────────────────
def label_final(rg1, rg0_pre=None, rg0_post=None, mutation=None):
    def _all_ok(gates):
        return all(x['ok'] for x in gates if x['ok'] is not None)

    def _fail(gates, prefix):
        return any(x['ok'] is False for x in gates if x['name'].startswith(prefix))

    if rg1 is None:
        return 'HARNESS-NOT-RUN', 'rg1_report.json 부재 — regress_m9.py rg1 부터.'
    if _fail(rg1['gates'], 'M-1') or _fail(rg1['gates'], 'M-2'):
        return ('REPLAY-RULE-MISMATCH (STOP)',
                '재구성 규칙이 동결 자료와 불일치 — 수리를 착지시키지 말고 '
                'PREREG §3 규칙부터 고쳐 재동결하라 (§6-3).')
    if rg1['mode'] == 'pre':
        if not _all_ok(rg1['gates']):
            return ('EXPECTED-TABLE-DRIFT (STOP)',
                    'pre 재유도가 동결 예상표와 불일치 — 예상표/자료 어느 쪽이 움직였는지 규명.')
        if mutation is not None and not mutation.get('all_caught'):
            return ('MUTATION-ESCAPE (STOP)',
                    '변형이 게이트를 통과 — 게이트가 공허하다. 수리 착지 금지.')
        return ('HARNESS-READY (PRE-REPAIR)',
                'RG-1 오프라인 전건 + 예상표 재확인. 수리 착지 가능 '
                '(착지 후 rg1 → rg0 post → analyze).')
    # post
    if _fail(rg1['gates'], 'M-3') or _fail(rg1['gates'], 'M-4') \
            or _fail(rg1['gates'], 'M-5') or _fail(rg1['gates'], 'M-6') \
            or _fail(rg1['gates'], 'M-7'):
        return ('PREDICATE-CONTRACT-VIOLATION (STOP)',
                '제품 술어가 동결 계약/예상표에서 이탈 — 수리를 수정하라.')
    if rg0_post is not None and _fail(rg0_post['gates'], 'R-3 '):
        return ('OBSERVABILITY-LEAK (STOP)',
                '수리 전/후 수치 비트 불일치 — 수리가 관측성 밖으로 샜다. '
                '즉시 중지·원인 규명 (§6-3).')
    every = list(rg1['gates'])
    for rep in (rg0_pre, rg0_post):
        if rep is not None:
            every += rep['gates']
    complete = (rg0_pre is not None and rg0_post is not None
                and mutation is not None and mutation.get('all_caught'))
    if _all_ok(every) and complete and not any(x['ok'] is None for x in every):
        return ('ALARM-FIXED (CONFIRMATORY)',
                'RG-1 131/131·0/72 + RG-0 비트 동일 + RG-2 셀 3종 + RG-3 키·의미 + 변형 전건 포착.')
    missing = [n for n, v in (('rg0_pre', rg0_pre), ('rg0_post', rg0_post),
                              ('mutation', mutation)) if v is None]
    bad = [x['name'] for x in every if x['ok'] is False]
    return ('ALARM-FIXED (EXPLORATORY)',
            f'부분 통과 — 미실행 {missing} · FAIL {bad[:4]}')


# ── 자가진단 (합성 사건으로 이 파일 자신을 검사 — 동결 자료 불요) ────────────
def selftest():
    # 재구성: (2,0)/(1,1) 부류 합성 행
    r20 = dict(cell='ASYNC5_P2', record='X', win=1, async_kicks=2, async_skipped=0,
               applied_after_lead={'4.0': 0, '9.0': 0}, applied_pre_wait=0,
               noop_warning_fired=True, health_pre_wait={'effective': False})
    r11 = dict(r20, async_kicks=1, async_skipped=1, noop_warning_fired=False)
    e20, e11 = trigger_events(r20), trigger_events(r11)
    assert e20 == [(1, 'kick', 0), (2, 'kick', 0)]
    assert e11 == [(1, 'kick', 0), (2, 'skip', 0)]
    # 구 계약: (2,0) 은 잡고 (1,1) 은 침묵 — M6 결함의 최소 재현
    assert old_alarm_reference(e20) is True and old_alarm_reference(e11) is False
    # 신 계약 유도식: 둘 다 잡고, 적용 있으면 침묵, 사건1 침묵
    assert replay_alarm(e20, _contract_due) and replay_alarm(e11, _contract_due)
    assert not replay_alarm([(1, 'kick', 0), (2, 'kick', 1)], _contract_due)
    for (inp, want) in PREDICATE_PROBES:
        assert _contract_due(*inp) == want, ('프로브와 유도식 자체 모순', inp)
    # off-by-one 변형은 (2,0) 에서 침묵해야 하고(함정의 존재 증명) 유도식과 달라야 한다
    bad = lambda n, a: (n - 1) >= 2 and a == 0                     # noqa: E731
    assert not replay_alarm(e20, bad) and not replay_alarm(e11, bad)
    # 러너 모형
    jobs = expected_jobs()
    assert len(jobs) == EXPECTED['n_rows'] and len(set(jobs)) == EXPECTED['n_rows']
    assert sum(1 for a, _r, _w in jobs if a.startswith('ASYNC5_')) == EXPECTED['n_async']
    # 라벨 갈래
    lab, _ = label_final(None)
    assert lab == 'HARNESS-NOT-RUN'
    return '자가진단 OK — 재구성·구/신 계약·프로브·러너 모형·라벨 갈래'


def deepcopy_rows(rows):
    return copy.deepcopy(rows)


if __name__ == '__main__':
    print(selftest())
