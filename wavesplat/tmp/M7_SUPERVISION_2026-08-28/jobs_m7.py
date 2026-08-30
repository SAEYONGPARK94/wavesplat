"""M7 **작업 목록 · 행 정체성 단일출처** — 러너와 분석기가 **같은 함수**를 부른다.

## 왜 이 파일이 따로 있는가 (⚠임무 지시와의 차이를 여기 적는다)

임무는 *"`expected_jobs()`/`row_identity()` 를 `gates_m7` 에 단일출처화하라"* 였다.
⛔**할 수 없었다**: `gates_m7.py` 는 `FROZEN_md5.txt` 에 `7c4cc145…` 로 **이미 동결**돼 있고
`AMENDMENT_1.md §4` 가 축자로 *"게이트 16개 … 불변"* 이라 적었다. 한 줄을 더해도 md5 가 바뀌어
**`G-15` 가 드리프트를 보고**한다(그리고 그것이 옳다 — 동결은 동결이다).
⟹ ★**요건의 실질**(*"같은 기능을 두 곳에 구현하지 마라"*, 저장소 누적 결함 1순위)은
이 파일 하나로 지킨다: `run_m7.py` 와 `analyze_m7.py` 가 **둘 다 여기서만** 작업 목록과
행 정체성을 얻는다. 문턱·팔·검정·해석 규칙은 **여전히 `gates_m7` 만** 소유한다
(이 파일은 `CELL_ORDER`·`REAL_RECORDS`·`DIFFUSION_SEEDS`·`EST_SEEDS` 를 **되읽을 뿐**이다).
⟹ 이 파일은 `AMENDMENT_1 §7 Step 0c` 의 동결 목록에 **반드시 함께 들어가야 한다**
(`check_batch_artifacts._judge_files()` 가 `run_*`·`analyze_*` 의 배치-내 임포트를 요구한다).

## 무엇을 소유하는가
* `expected_jobs(exp)` — E1/E2/E3 의 **완전한** 작업 정체성 목록(코퍼스에서 열거).
* `row_identity(row)` — 결과 행 하나 → 같은 정체성 튜플. **역함수 관계**를 자가진단이 검사한다.
* `stamp(ident, metrics)` — ★행의 **정체성 스탬프**. 러너가 실행 결과에, 자가진단이 합성
  픽스처에 **같은 함수**로 찍는다. ⟹ *"게이트를 모형한 의도표"* 가 아니라 **러너를 모형**한다.
* `shard(jobs, i, n)` — `idx % n` 분할(`M1` 규약과 같다).

⛔이 파일은 판정을 내리지 않는다. 문턱·라벨·검정은 `gates_m7.py` 가 유일한 출처다.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
_PATHS = (HERE, ROOT,
          os.path.join(ROOT, 'scripts'),
          os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23'),
          os.path.join(ROOT, 'tmp', 'B12_HOS_EXTEND_2026-08-23'),
          os.path.join(ROOT, 'tmp', 'B11_HOS_BASELINE_2026-08-23'),
          os.path.join(ROOT, 'tmp', 'S16_BASELINE_2026-08-14'),
          os.path.join(ROOT, 'tmp', 'M1_NONLIN_REAL_2026-08-28'),
          os.path.join(ROOT, 'tmp', 'B26_SEASTATE_2026-08-27'),
          os.path.join(ROOT, 'tmp', 'bankmmse_port_2026-07-29'))
for _p in _PATHS:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gates_m7 as G                                                    # noqa: E402

EXPERIMENTS = ('e1', 'e2', 'e3')

# 학습 5칸 + 참조 배포 체크포인트 = `learned_ckpt` 한 키로 갈리는 팔들 (PREREG §1.1)
LEARNED_ARMS = tuple(G.CELL_ORDER) + (G.REF_SHIPPED,)
REAL_ARMS = LEARNED_ARMS + (G.REF_BANK,)          # BANK 는 체크포인트 무관(배포 실측 경로)
SPEC_ARM = 'SPEC25'                               # 합성 외부 기준선 (S16 정의를 그대로 import)
ZERO_ARM = 'ZERO'                                 # ⛔런 0회 — SPEC25 행의 `zero_nRMSE` 에서 합성
SYNTH_ARMS = LEARNED_ARMS + (SPEC_ARM,)

# 체크포인트 경로 규약 — ★임무 A(`train_m7.py`)와 맞춘 것. 여기가 유일한 정의다.
CKPT_DIR_REL = os.path.join('results', 'direction3')


def cell_ckpt_rel(cell):
    """칸 이름 → 체크포인트 저장소-상대 경로. `SHIPPED` 만 배포본을 가리킨다."""
    if cell == G.REF_SHIPPED:
        from wavesplat.optimizer.learned_init import DEFAULT_CKPT_REL
        return DEFAULT_CKPT_REL
    if cell == G.REF_BANK:
        return None                                # bank_mmse 경로 — 체크포인트를 쓰지 않는다
    if cell not in G.CELLS:
        raise KeyError(f'알 수 없는 칸: {cell!r}')
    return os.path.join(CKPT_DIR_REL, f'm7_{cell}.pt')


def cell_ckpt(cell):
    rel = cell_ckpt_rel(cell)
    return None if rel is None else os.path.join(ROOT, rel)


def pretrain_ckpt(name):
    """Phase-A 사전학습 체크포인트(`PRE-BASE-1` 등) 절대경로."""
    return os.path.join(ROOT, CKPT_DIR_REL, f'm7_{name}.pt')


# ── 코퍼스 열거 (전부 동결 배치의 선택 함수를 **그대로 부른다**) ────────────────
def real_records():
    """E1 = `scene_station.RECORDS_12`. `gates_b2.RECORDS` 와 전건 일치해야 한다."""
    import gates_b2
    if tuple(gates_b2.RECORDS) != tuple(G.REAL_RECORDS):
        raise AssertionError('gates_b2.RECORDS != gates_m7.REAL_RECORDS — 코퍼스가 갈렸다')
    if int(gates_b2.WIN) != int(G.REAL_WIN_PRIMARY):
        raise AssertionError('gates_b2.WIN != gates_m7.REAL_WIN_PRIMARY')
    return tuple(G.REAL_RECORDS)


def e2_episodes():
    """E2 = `B-12` 혼잡 HOS 48 에피소드. 선택 규칙을 `run_b12` 에서 그대로 쓴다."""
    import run_b12
    eps = run_b12.crowded_episodes()
    if len(eps) != G.E2_N_EP:
        raise AssertionError(f'E2 코퍼스 {len(eps)} != {G.E2_N_EP}')
    return eps


def e3_episodes():
    """E3 = `M1` 자료(LIN 24 + HOS 24). ⛔생성 0회 — 동결 자료를 읽기만 한다."""
    import gates_m1
    eps, _rej = gates_m1.select()
    if len(eps) != G.E3_N_EP:
        raise AssertionError(f'E3 코퍼스 {len(eps)} != {G.E3_N_EP}')
    return eps


_EP_CACHE = {}


def episodes(exp):
    if exp not in _EP_CACHE:
        _EP_CACHE[exp] = e2_episodes() if exp == 'e2' else e3_episodes()
    return _EP_CACHE[exp]


# ── 작업 목록 · 행 정체성 ──────────────────────────────────────────────────────
def expected_jobs(exp):
    """`exp` 의 **완전한** 작업 정체성 목록. 정체성 = `(exp, cell, unit, seed)`.

    * E1 : unit = 레코드, seed = 확산 시드. `BANK` 는 확산과 무관하므로 레코드당 1회.
    * E2/E3 : unit = 에피소드 키, seed = 추정 시드(`SPEC25` 는 필터 시드).
    """
    if exp == 'e1':
        out = [('e1', c, r, s)
               for c in LEARNED_ARMS for r in real_records() for s in G.DIFFUSION_SEEDS]
        out += [('e1', G.REF_BANK, r, None) for r in real_records()]
        return out
    if exp in ('e2', 'e3'):
        keys = [e['episode'] for e in episodes(exp)]
        return [(exp, c, k, s) for c in SYNTH_ARMS for k in keys
                for s in (G.EST_SEEDS if c != SPEC_ARM else _filter_seeds())]
    raise KeyError(f'exp={exp!r} — {EXPERIMENTS} 중 하나여야 한다')


def _filter_seeds():
    import gates_b11
    return tuple(gates_b11.FILTER_SEEDS)


def row_identity(row):
    """결과 행 → 정체성 튜플. ⛔행에 별도 `unit` 키를 두지 않는다(중복 기재 금지) —
    `record`/`episode` 라는 **게이트가 읽는 바로 그 키**에서 유도한다."""
    exp = row.get('exp')
    if exp == 'e1':
        return ('e1', row.get('cell'), row.get('record'), row.get('diffusion_seed'))
    if exp in ('e2', 'e3'):
        seed = row.get('filter_seed') if row.get('cell') == SPEC_ARM else row.get('est_seed')
        return (exp, row.get('cell'), row.get('episode'), seed)
    return (exp, row.get('cell'), row.get('record') or row.get('episode'), None)


def stamp(ident, metrics=None, **extra):
    """★정체성 스탬프 — 러너의 실행 결과에도, 자가진단의 합성 픽스처에도 **이 함수**를 쓴다.

    ⟹ *"게이트가 세는 집합이 팔을 구조적으로 비우는가"* 를 **러너를 모형해** 시험할 수 있다
    (`M4 G-13` 부류: 러너가 찍는 스탬프가 게이트의 필터를 항진시키는 결함).
    ⚠정체성 키는 `metrics` 를 **덮어쓴다** — 러너가 팔 이름을 `'BASE|ds0'` 처럼 꾸며 부르는
    경우가 있고, 행에 남아야 하는 것은 **정규 칸 이름**이다.
    """
    exp, cell, unit, seed = ident
    row = dict(metrics or {})
    row['exp'] = exp
    row['cell'] = cell
    if exp == 'e1':
        row['record'] = unit
        row['diffusion_seed'] = seed
        row['win'] = G.REAL_WIN_PRIMARY
    else:
        row['episode'] = unit
        if cell == SPEC_ARM:
            row['filter_seed'] = seed
        else:
            row['est_seed'] = seed
    row.update(extra)
    return row


def census_to_manifest(census):
    """`train_m7.py` 가 쓰는 `stream_census.json` → `gates_m7.gates()` 의 **학습-측** 매니페스트 키.

    ⚠**이 사상은 `train_m7._gate_selfcheck()` 를 그대로 거울로 삼는다**(같은 키·같은 출처).
    ⛔거기서 import 하지 않는 이유는 하나다: `train_m7` 은 import 시점에 `torch`·`gen_m7`
    (→`gen_mixed`)을 끌고 오고 **GPU 를 잡는다** — 분석기는 GPU 0회로 돌아야 한다.
    ⟹ 중복의 위험은 **두 독립 유도의 대조**로 막는다: 평가-측 `analyze_m7.fs_facts()` 가
    체크포인트 sha256·학습 횟수를 **파일에서 다시 읽고**, `P-8` 이 원장의 주장과 대조한다.
    """
    st = (census or {}).get('stream') or {}
    runs = (census or {}).get('runs') or {}
    hard, base = st.get('HARD') or {}, st.get('BASE') or {}
    man = dict(
        hard_scenes_n=hard.get('n_scenes'),
        hard_scenes_below_repr=hard.get('below_repr'),
        hard_kHs_median=hard.get('kHs_median'),
        base_kHs_median=base.get('kHs_median'),
        anchors_excluded=hard.get('anchors_excluded'),
        eval_records=list(G.REAL_RECORDS),          # ⛔평가에서는 사후배제하지 않는다(G-8)
        train_scene_seeds_sample=list((census or {}).get('train_scene_seeds_sample') or []),
    )
    for k in ('lam0_finetune_steps', 'ckpt_sha256', 'shipped_sha256',
              'n_pretrain', 'n_finetune', 'pretrain_sha256'):
        if k in (census or {}):
            man[k] = census[k]
    # ★분모가 살아 있는 표현-필터 측정(`gen_m7.probe_repr_filter`) — `below_repr` 은 재시도
    #   경로라 구조적으로 0 이다. 정직한 수는 이쪽이고, 분석기가 **공시**한다.
    for tag, row in runs.items():
        pr = (row or {}).get('probe') or {}
        if (row or {}).get('data') == 'HARD' and pr:
            man['hard_probe'] = pr
        if (row or {}).get('data') == 'BASE' and pr:
            man['base_probe'] = pr
    man['census_runs'] = {t: {k: r.get(k) for k in
                              ('phase', 'cell', 'data', 'seed', 'lambda_field', 'steps',
                               'ckpt_sha256', 'out')} for t, r in runs.items()}
    return {k: v for k, v in man.items() if v is not None}


def shard(jobs, i, n):
    """`M1`·`B-12` 와 같은 `idx % nshard` 분할. 순서는 `expected_jobs` 순서를 보존한다."""
    if not (isinstance(n, int) and n >= 1 and 0 <= i < n):
        raise ValueError(f'shard {i}/{n} 이 유효하지 않다')
    return [j for k, j in enumerate(jobs) if k % n == i]


def results_path(exp, shard_i=None, nshard=1):
    base = f'results_m7_{exp}' + ('' if (nshard or 1) <= 1 else f'_{shard_i}')
    return os.path.join(HERE, base + '.jsonl')


def all_results_globs():
    return [os.path.join(HERE, f'results_m7_{e}*.jsonl') for e in EXPERIMENTS]


# ── 자가진단 (런 0회 · GPU 0회) ───────────────────────────────────────────────
def selftest(verbose=True):
    def say(s):
        if verbose:
            print(s, flush=True)
    fails = []
    say('★jobs_m7 자가진단 ─────────────────────────────────────────')
    exp_n = {'e1': len(G.REAL_RECORDS) * (len(LEARNED_ARMS) * len(G.DIFFUSION_SEEDS) + 1),
             'e2': G.E2_N_EP * (len(LEARNED_ARMS) * len(G.EST_SEEDS)
                                + len(_filter_seeds())),
             'e3': G.E3_N_EP * (len(LEARNED_ARMS) * len(G.EST_SEEDS)
                                + len(_filter_seeds()))}
    for exp in EXPERIMENTS:
        jobs = expected_jobs(exp)
        if len(jobs) != exp_n[exp]:
            fails.append(f'{exp}: 작업 수 {len(jobs)} != 기대 {exp_n[exp]}')
        if len(set(jobs)) != len(jobs):
            fails.append(f'{exp}: 중복 작업 {len(jobs) - len(set(jobs))}건')
        # 역함수: stamp -> row_identity 왕복
        bad = [j for j in jobs if row_identity(stamp(j)) != j]
        if bad:
            fails.append(f'{exp}: stamp/row_identity 왕복 실패 {len(bad)}건 (예: {bad[0]})')
        # 분할이 정확히 한 번씩 덮는가
        cov = [x for i in range(8) for x in shard(jobs, i, 8)]
        if sorted(cov) != sorted(jobs):
            fails.append(f'{exp}: 8분할이 작업을 정확히 한 번씩 덮지 않는다')
        say(f'  {exp}: {len(jobs)} 작업 · 왕복 OK · 8분할 덮음 OK')
    # 코퍼스 정합
    try:
        real_records()
        say(f'  코퍼스: 실측 {len(G.REAL_RECORDS)} · E2 {G.E2_N_EP} · E3 {G.E3_N_EP}')
    except AssertionError as exc:
        fails.append(str(exc))
    # 체크포인트 경로 규약
    rels = {c: cell_ckpt_rel(c) for c in G.CELL_ORDER}
    if len(set(rels.values())) != len(G.CELL_ORDER):
        fails.append(f'칸별 체크포인트 경로가 유일하지 않다: {rels}')
    if cell_ckpt_rel(G.REF_BANK) is not None:
        fails.append('BANK 는 체크포인트를 가지면 안 된다')
    say(f'  체크포인트 규약: {rels[G.CELL_ORDER[0]]} … (칸 {len(rels)}개, SHIPPED 별도)')
    for f in fails:
        say(f'  ⛔ {f}')
    say(f'jobs_m7 자가진단 {"PASS" if not fails else "FAIL"} ({len(fails)} 결함)')
    return (not fails), fails


if __name__ == '__main__':
    sys.exit(0 if selftest()[0] else 1)
