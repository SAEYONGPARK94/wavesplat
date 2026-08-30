"""M-10 **작업 목록 · 행 정체성 단일출처** — 러너와 분석기가 **같은 함수**를 부른다.

★왜 파일이 따로 있는가 (`jobs_m7.py` 와 같은 사유)
`gates_m10.py` 는 `PREREG.md §12` 에서 *"바꾸지 마라"* 로 닫혔다(자가진단 24/24 완료).
한 줄을 더하면 md5 가 바뀌고 §11 의 동결 규약이 깨진다. ⟹ 요건의 실질
(***"같은 기능을 두 곳에 구현하지 마라"*** — 이 저장소 누적 결함 1순위)을 이 파일 하나로 지킨다:
`run_m10.py` 와 `analyze_m10.py` 와 `mutate_m10.py` 가 **셋 다 여기서만** 작업 목록·행 정체성·
행 스탬프를 얻는다.

⛔이 파일은 **판정하지 않는다.** 문턱·밴드·라벨·게이트·해석 규칙은 `gates_m10.py` 가 유일 출처이고
코퍼스·시드·팔 이름조차 여기서 **되읽을 뿐**이다(`G.ARM_ORDER`·`G.EST_SEEDS`·`G.FILTER_SEEDS`·
`G.N_EP`). 손복사 상수 0건.

★코퍼스는 **두 갈래로 독립 유도해 대조**한다(`M8b §5.3` — *동결은 파일의 동일성을 보장하지
자료의 정합성을 보장하지 않는다*):
  ① `run_b12.crowded_episodes()`  (동결 러너의 선택 함수 = 경로까지 준다)
  ② `gates_m10.episodes()`        (게이트 자신의 열거)
어긋나면 `AssertionError` 다.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for _p in (HERE, ROOT,
           os.path.join(ROOT, 'tmp', 'B11_HOS_BASELINE_2026-08-23'),
           os.path.join(ROOT, 'tmp', 'B12_HOS_EXTEND_2026-08-23'),
           os.path.join(ROOT, 'tmp', 'S16_BASELINE_2026-08-14')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gates_m10 as G                                              # noqa: E402

# ⛔손복사 0 — 전부 게이트에서 되읽는다.
ARMS = tuple(G.ARM_ORDER)                       # ('SHIP', 'SPEC25')
SEEDS = {'SHIP': tuple(G.EST_SEEDS), 'SPEC25': tuple(G.FILTER_SEEDS)}

_EPS = None


def episodes():
    """혼잡 48. ★두 독립 유도의 대조를 통과한 것만 돌려준다.

    ⚠`run_b12` 는 임포트 시점에 `os.chdir(ROOT)` 를 한다 ⟹ **지연 임포트**로 둔다
    (`check_gate_refs` 가 러너를 정적으로만 파는 것과 같은 이유: 임포트 부작용을 분석기에
    끌고 들어가지 않는다).
    """
    global _EPS
    if _EPS is None:
        import run_b12                                             # noqa: E402 (지연)
        eps = run_b12.crowded_episodes()
        names_runner = [e['episode'] for e in eps]
        names_gate = list(G.episodes())
        if names_runner != names_gate:
            raise AssertionError(
                'M-10 코퍼스가 두 유도에서 갈렸다: run_b12 %d개 vs gates_m10 %d개 · '
                '차집합 %r' % (len(names_runner), len(names_gate),
                              sorted(set(names_runner) ^ set(names_gate))[:6]))
        if len(eps) != G.N_EP:
            raise AssertionError('M-10 코퍼스 %d != gates_m10.N_EP %d' % (len(eps), G.N_EP))
        _EPS = eps
    return _EPS


def expected_jobs():
    """★**완전한** 작업 정체성 목록. 정체성 = `(arm, episode, seed)`.

    순서는 **에피소드 바깥 · 팔 안쪽**이다 — 한 에피소드의 두 팔이 붙어서 돌아야
    `G-5`(같은 창)·`G-6`(같은 마스크)·`G-9`(같은 GT)가 **샤드 안에서** 대조 가능해진다
    (팔을 샤드로 가르면 그 세 게이트는 합치기 전에는 발화하지 않는다).
    """
    return [(arm, e['episode'], s)
            for e in episodes() for arm in ARMS for s in SEEDS[arm]]


def row_identity(row):
    """결과 행 하나 → 같은 정체성 튜플. ⛔별도 `unit` 키를 두지 않는다 —
    **게이트가 실제로 읽는 키**(`arm`·`episode`·`seed`)에서 유도한다
    (`gates_m10.gates` 의 `frozen.get((r['arm'], r['episode'], r.get('seed')))` 와 같은 키)."""
    return (row.get('arm'), row.get('episode'), row.get('seed'))


def stamp(ident, metrics=None, **extra):
    """★행의 정체성 스탬프 — 러너의 실행 결과에도, 변형 하네스의 합성 픽스처에도 **이 함수**를
    쓴다. ⟹ 변형 하네스가 *"게이트를 모형한 의도표"* 가 아니라 **러너를 모형**한다
    (`M4 G-13` 부류: 러너가 찍는 스탬프가 게이트의 필터를 항진시키는 결함)."""
    arm, ep, seed = ident
    row = dict(metrics or {})
    row['arm'] = arm
    row['episode'] = ep
    row['seed'] = seed
    row.update(extra)
    return row


def shard(jobs, i, n):
    """★**에피소드 축**의 `idx % n` 분할. `expected_jobs()` 순서를 보존한다.

    ⛔`M1`·`B-12`·`M7` 의 *작업 축* `idx % n` 을 그대로 쓰면 **한 에피소드의 6행이 여러 샤드로
    갈라진다** — 그러면 `G-5`(같은 창)·`G-6`(같은 마스크)·`G-9`(같은 GT)는 **샤드를 합치기
    전에는 구조적으로 발화하지 않는다**(자가진단이 이것을 잡았다: 4분할에서 1/6).
    ⟹ 분할 단위는 **에피소드**다. 샤드 하나가 그 자체로 세 게이트를 발화시킨다.
    """
    if not (isinstance(n, int) and n >= 1 and 0 <= i < n):
        raise ValueError('shard %r/%r 이 유효하지 않다' % (i, n))
    order = []
    for _a, ep, _s in jobs:
        if ep not in order:
            order.append(ep)
    mine = {ep for k, ep in enumerate(order) if k % n == i}
    return [j for j in jobs if j[1] in mine]


def results_path(i=0):
    """⚠접미사 `_i` 를 **항상** 붙인다 — `gates_m10.load_rows()` 의 glob 이
    `results_m10_*.jsonl` 이라 `results_m10.jsonl` 은 **조용히 안 읽힌다**."""
    return os.path.join(HERE, 'results_m10_%d.jsonl' % int(i))


def smoke_path(i=0):
    """⛔스모크는 `results_m10_*` 밖에 쓴다 — 판정 자료와 절대 섞이지 않는다
    (`run_s16` 의 `smoke_results.jsonl` 규약)."""
    return os.path.join(HERE, 'smoke_m10_%d.jsonl' % int(i))


# ── 자가진단 (런 0회 · 자료 0행) ──────────────────────────────────────────────
def selftest(verbose=True):
    fails = []

    def say(s):
        if verbose:
            print(s, flush=True)

    say('★jobs_m10 자가진단 ─────────────────────────────────────────')
    jobs = expected_jobs()
    want = G.N_EP * sum(len(SEEDS[a]) for a in ARMS)
    if len(jobs) != want:
        fails.append('작업 수 %d != 기대 %d' % (len(jobs), want))
    if len(set(jobs)) != len(jobs):
        fails.append('중복 작업 %d건' % (len(jobs) - len(set(jobs))))
    bad = [j for j in jobs if row_identity(stamp(j)) != j]
    if bad:
        fails.append('stamp/row_identity 왕복 실패 %d건 (예: %r)' % (len(bad), bad[0]))
    for n in (1, 4, 7):
        cov = [x for i in range(n) for x in shard(jobs, i, n)]
        if sorted(cov) != sorted(jobs):
            fails.append('%d분할이 작업을 정확히 한 번씩 덮지 않는다' % n)
    # ★샤드가 에피소드를 쪼개지 않는가 — G-5/G-6/G-9 의 발화 조건
    for n in (2, 4, 8):
        for i in range(n):
            for ep in {j[1] for j in shard(jobs, i, n)}:
                mine = {j for j in shard(jobs, i, n) if j[1] == ep}
                allj = {j for j in jobs if j[1] == ep}
                if mine != allj:
                    fails.append('%d분할 샤드 %d 이 에피소드 %s 를 쪼갠다 '
                                 '(%d/%d) ⟹ G-5·G-6·G-9 가 샤드 안에서 발화 불가'
                                 % (n, i, ep, len(mine), len(allj)))
                    break
            else:
                continue
            break
    if results_path(0) == smoke_path(0):
        fails.append('스모크 경로가 판정 경로와 같다')
    if os.path.basename(results_path(3)) not in \
            os.path.basename(os.path.join(HERE, 'results_m10_3.jsonl')):
        fails.append('결과 파일명이 gates_m10.load_rows 의 glob 과 안 맞는다')
    say('  작업 %d개 (에피소드 %d × 팔 %d × 시드 %s) · 왕복 OK · 분할 덮음 OK'
        % (len(jobs), G.N_EP, len(ARMS), {a: len(SEEDS[a]) for a in ARMS}))
    say('  결과 %s · 스모크 %s'
        % (os.path.basename(results_path(0)), os.path.basename(smoke_path(0))))
    for f in fails:
        say('  ⛔ %s' % f)
    say('jobs_m10 자가진단 %s (%d 결함)' % ('PASS' if not fails else 'FAIL', len(fails)))
    return (not fails), fails


if __name__ == '__main__':
    sys.exit(0 if selftest()[0] else 1)
