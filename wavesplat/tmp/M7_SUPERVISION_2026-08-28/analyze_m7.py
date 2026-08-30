"""M7 동결 분석기 — ⛔**결과보다 먼저 쓰였다.** 결과를 보고 고치지 않는다.

    python3 tmp/M7_SUPERVISION_2026-08-28/analyze_m7.py            # 판정
    python3 tmp/M7_SUPERVISION_2026-08-28/analyze_m7.py --selftest # 런 0회 자가진단

## 이 파일이 존재하는 진짜 이유 — ★**게이트 결함 4건을 여기서 막는다**

`AMENDMENT_1 §5.5` 가 `gates_m7.py` 를 결과 0행에서 탐침해 결함 넷을 찾았다. ⛔**게이트는
고치지 않는다**(동결). 대신 **러너·분석기의 계약**으로 막는다 — 그 계약이 아래 `PRECHECKS` 다.

| | 결함 | 여기서 막는 방법 |
|---|---|---|
| **F-1** | `G-9` 가 **빈 리스트에서 공허 PASS** (`all(... for s in [])`) | `P-1` 이 `train_scene_seeds_sample` **≥1** 을 **라벨 전에** 요구. 없으면 판정 거부 |
| **F-2** | `contrasts()` 가 결측 칸을 조용히 건너뛰어 **n 12→11 이어도 `label()` 이 돈다** | `P-2` 가 세 대비 전부 `n == 12` 를 요구. 아니면 **`INCOMPLETE`**, 라벨 없음 |
| **F-3** | E2 에 `SPEC25`/`ZERO` 가 없으면 `S-1` 이 **무효 라벨로 조용히 착지** | `ZERO` 를 `zero_nRMSE` 열에서 **합성**(B-12 규약)하고 `P-3` 이 `neg_control['n'] == 48` 을 **선행** 검사 |
| **F-4** | `mechanism()` 이 조용히 `{}` | `P-4` 가 **공시**한다(바 없음이므로 차단은 아니고 판정문에 실린다) |

## 그 밖의 규율
* ★**0행 가드** — 유효 행 0이면 `verdict` 을 쓰지 않고 `exit 1`(`check_batch_artifacts` M-4/V3).
* ★**차단 게이트 FAIL 이면 판정 거부** — `BLOCK_E1` / `BLOCK_S1`.
* ★**커버리지 0 게이트 금지** — 세는 대상 행이 **한 줄도 없으면** 계수기를 `0` 이 아니라
  **`None`** 으로 둔다 ⟹ 게이트가 조용히 통과하는 대신 **소리 내어 FAIL** 한다.
* ★**`DATA` 축 결과에 `EXPLORATORY` 를 기계로 붙인다**(`AMENDMENT_1 §2.3`).
* ★**분기표 초과 공시 문언을 판정문에 자동 삽입**한다 — ⛔손으로 옮겨 적지 않고
  `AMENDMENT_1.md §1.3` 에서 **읽어 온다**(md5 를 `FROZEN_md5.txt` 와 대조한 뒤).
"""
import glob
import hashlib
import json
import os
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import jobs_m7 as J                                                     # noqa: E402
import gates_m7 as G                                                    # noqa: E402

os.chdir(ROOT)

AMEND = os.path.join(HERE, 'AMENDMENT_1.md')
FROZEN = sorted(glob.glob(os.path.join(HERE, 'FROZEN_md5*.txt')))
M1_VERDICT = os.path.join(ROOT, 'tmp/M1_NONLIN_REAL_2026-08-28/verdict_m1.json')
CENSUS = os.path.join(HERE, 'stream_census.json')       # train_m7.py 가 쓴다
GEN_PARITY = os.path.join(HERE, 'gen_parity.json')      # gen_m7.py --parity 가 쓴다
TRAIN_MANIFESTS = ('gen_manifest_m7.json', 'train_manifest_m7.json', 'manifest_m7.json')

# ── ★결과 등급 (AMENDMENT_1 §2.3, 결과 전 고정) ──────────────────────────────
GRADES = {
    'T-1_loss': 'PREREGISTERED',
    'T-2_data': 'EXPLORATORY',              # ★DATA 축 — 확증·기여·우선순위 승격 금지
    'T-3_interaction': 'EXPLORATORY',       # 〃 (분포 성분을 포함한다)
    'neg_control(BASE vs BASE-B)': 'PREREGISTERED',
    'abort_rule': 'PREREGISTERED',
    'S-1': 'PREREGISTERED',
    'E2_hard_bar(R1)': 'PREREGISTERED',
}
# 라벨의 **하중**이 분포 축에 실리는 것들 — 등급이 라벨 자체에 붙는다.
DATA_BORNE_LABELS = ('DISTRIBUTION-EARNS', 'BOTH-ADDITIVE', 'INTERACTION')
DATA_AXIS_BANS = (
    '⛔확증(CONFIRMATORY) 승격 금지',
    '⛔논문 기여 목록 승격 금지',
    '⛔후속 배치의 우선순위 근거로 인용 금지(`B-23` 금지 항목과 같은 규율)',
    '허용 최대치 = "탐색적으로 방향이 잡혔다. 확증은 신선 시드·새 사전등록으로만."',
)

# ── 차단 게이트 (FAIL 이면 그 판정을 내지 않는다) ────────────────────────────
BLOCK_E1 = ('G-1', 'G-2', '★G-3', '★G-4', '★G-5', 'G-5b', '★G-6', '★G-7', '★G-8',
            'G-9', '★G-10', '★G-11', 'G-12', 'G-13', 'G-15')
BLOCK_S1 = ('G-1', 'G-2', '★G-3', '★G-4', '★G-6', '★G-7', 'G-9', '★G-10', '★G-11', 'G-15')
BLOCK_E3 = ('G-2', '★G-3', 'G-14', 'G-15')


def gid(name):
    """게이트 표시명 → 식별자(`★G-3 배포 …` → `★G-3`)."""
    return name.split()[0]


# ── 입력 ─────────────────────────────────────────────────────────────────────
def load_rows():
    rows = []
    for pat in J.all_results_globs():
        for p in sorted(glob.glob(pat)):
            for line in open(p):
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def _read_json(p):
    if not os.path.exists(p):
        return {}
    try:
        d = json.load(open(p))
    except json.JSONDecodeError:
        return {}
    return d if isinstance(d, dict) else {}


def load_train_manifest():
    """학습-측 매니페스트 = **임무 A 의 산출물에서 유도한다**(손복사 0).

    * `stream_census.json`(`train_m7._merge_census`) → `jobs_m7.census_to_manifest`
    * `gen_parity.json`(`gen_m7.py --parity`) → `gen_parity_maxabs`
    ⚠키가 없으면 **조용히 넘어가지 않는다** — 해당 게이트가 `None` 을 받아 **FAIL** 한다.
    """
    man = J.census_to_manifest(_read_json(CENSUS))
    gp = _read_json(GEN_PARITY)
    if 'gen_parity_maxabs' in gp:
        man['gen_parity_maxabs'] = gp['gen_parity_maxabs']
        man['gen_parity_detail'] = gp.get('detail')
    for name in TRAIN_MANIFESTS:                        # 명시 매니페스트가 있으면 덮어쓴다
        man.update(_read_json(os.path.join(HERE, name)))
    return man


def _sha256(path):
    if not (path and os.path.exists(path)):
        return None
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def frozen_drift():
    """`FROZEN_md5*.txt` 의 각 줄을 현재 파일과 대조. **경로 기준이 섞여 있다** —
    배치 기준 → 부모 → 저장소 기준 순으로 푼다(`check_batch_artifacts._resolve` 와 같은 규약).
    ⚠적혀 있는데 **찾을 수 없는 파일도 드리프트로 센다**(대조 불가 = 확인 불가)."""
    latest, detail = {}, []
    for f in FROZEN:
        for ln in open(f):
            parts = ln.split()
            if ln.startswith('#') or len(parts) != 2 or len(parts[0]) != 32:
                continue
            latest[parts[1]] = parts[0]
    n_bad = 0
    for tgt, h in sorted(latest.items()):
        p = None
        for base in (HERE, os.path.dirname(HERE), ROOT):
            q = os.path.normpath(os.path.join(base, tgt))
            if os.path.exists(q):
                p = q
                break
        if p is None:
            n_bad += 1
            detail.append([tgt, 'MISSING'])
            continue
        cur = hashlib.md5(open(p, 'rb').read()).hexdigest()
        if cur != h:
            n_bad += 1
            detail.append([tgt, 'DRIFT'])
    return n_bad, len(latest), detail


def excess_disclosure():
    """★분기표 초과 공시 문언을 `AMENDMENT_1.md §1.3` 에서 **읽어 온다**(손복사 0).

    반환 `(text, ok, note)`. md5 가 `FROZEN_md5.txt` 와 어긋나거나 추출이 실패하면
    `ok=False` — 그러면 `P-7` 이 판정을 거부한다(공시는 **동반 의무**다).
    """
    if not os.path.exists(AMEND):
        return None, False, f'{AMEND} 없음'
    want = None
    for f in FROZEN:
        for ln in open(f):
            parts = ln.split()
            if len(parts) == 2 and parts[1].endswith('AMENDMENT_1.md'):
                want = parts[0]
    cur = hashlib.md5(open(AMEND, 'rb').read()).hexdigest()
    if want and cur != want:
        return None, False, f'AMENDMENT_1.md md5 불일치 {cur[:8]} != {want[:8]}'
    src = open(AMEND, encoding='utf-8').read().splitlines()
    body, grab = [], False
    for ln in src:
        if ln.startswith('### 1.3'):
            grab = True
            continue
        if grab:
            if ln.startswith('>'):
                body.append(ln.lstrip('>').strip())
            elif body and ln.strip() and not ln.startswith('>'):
                break
    txt = re.sub(r'\s+', ' ', ' '.join(x for x in body if x)).replace('*', '').strip()
    if len(txt) < 80:
        return None, False, f'§1.3 공시 문언 추출 실패(길이 {len(txt)})'
    return txt, True, f'AMENDMENT_1.md §1.3 에서 추출 (md5 {cur[:8]})'


# ── 매니페스트 조립 (⛔의도표 금지 — 행과 파일에서 **되읽는다**) ─────────────
def fs_facts():
    """파일 계통 사실. 자가진단은 이 dict 을 **픽스처로 대체**한다."""
    from wavesplat.optimizer.learned_init import DEFAULT_CKPT
    cells = {c: _sha256(J.cell_ckpt(c)) for c in G.CELL_ORDER}
    pre = sorted(glob.glob(os.path.join(ROOT, J.CKPT_DIR_REL, 'm7_PRE-*.pt')))
    nb, ntot, detail = frozen_drift()
    m1 = None
    if os.path.exists(M1_VERDICT):
        try:
            m1 = json.load(open(M1_VERDICT)).get('pair_key')
        except json.JSONDecodeError:
            m1 = None
    return dict(ckpt_sha256={k: v for k, v in cells.items() if v},
                n_pretrain=len(pre),
                n_finetune=sum(1 for v in cells.values() if v),
                shipped_sha256=_sha256(DEFAULT_CKPT),
                frozen_drift=nb, frozen_n=ntot, frozen_detail=detail,
                m1_verdict_frozen=bool(m1), m1_pair_key=m1)


def build_manifest(by_exp, train_man, fs=None):
    """⛔**커버리지 0 게이트 금지** — 세는 대상 행이 없으면 계수기를 `None` 으로 둔다.
    (`0` 으로 두면 게이트가 **자료 없이 통과**한다 = `M4 G-13`·`G-9` 와 같은 부류.)"""
    fs = fs_facts() if fs is None else dict(fs)
    e1 = by_exp.get('e1', [])
    allr = [r for rs in by_exp.values() for r in rs]
    # ★순서가 규율이다: 학습-측 매니페스트(주장) → **파일 되읽기**(fs) → **행 되읽기**.
    #   뒤엣것이 이긴다 ⟹ 매니페스트가 무엇을 적었든 sha256·계수기는 **실측이 권위**다.
    man = dict(train_man)
    man['_census_claim'] = {k: train_man[k] for k in
                            ('n_pretrain', 'n_finetune', 'shipped_sha256', 'ckpt_sha256')
                            if k in train_man}
    man.update(fs)

    learned_any = [r for r in allr if r.get('cell') in J.LEARNED_ARMS]
    exp_sha = dict(fs.get('ckpt_sha256') or {})
    exp_sha[G.REF_SHIPPED] = fs.get('shipped_sha256')
    man['rows_ckpt_mismatch'] = (None if not learned_any else sum(
        1 for r in learned_any
        if not (r.get('ckpt_sha256') and exp_sha.get(r['cell'])
                and r['ckpt_sha256'] == exp_sha[r['cell']])))

    e1_learned = [r for r in e1 if r.get('cell') in J.LEARNED_ARMS]
    man['rows_route_not_learned'] = (None if not e1_learned else sum(
        1 for r in e1_learned
        if r.get('route_init_method') != 'learned' or r.get('route_bank')
        or not (r.get('n_atoms') or 0) > 0))

    e1_bank = [r for r in e1 if r.get('cell') == G.REF_BANK]
    man['bank_rows_wrong_route'] = (None if not e1_bank else sum(
        1 for r in e1_bank
        if not (r.get('route_bank') and r.get('route_init_method') == 'bank_mmse'
                and (r.get('n_atoms') or 0) == 0)))

    man['rows_missing_add_calls'] = (None if not allr else
                                     sum(1 for r in allr if r.get('add_calls') is None))

    man['eval_records'] = sorted({r.get('record') for r in e1 if r.get('record')})
    zsrc = {r.get('zero_ref_source') for r in e1 if r.get('zero_ref_source')}
    man['zero_ref_source'] = (zsrc.pop() if len(zsrc) == 1 else None)
    return man


# ── ★선행 검사 (F-1 … F-4 + 커버리지) ───────────────────────────────────────
def coverage(by_exp):
    """러너의 작업 목록(`jobs_m7.expected_jobs`)과 실제 행을 **같은 정체성**으로 맞춘다."""
    out = {}
    for exp in J.EXPERIMENTS:
        rs = by_exp.get(exp, [])
        if not rs:
            out[exp] = dict(n_rows=0, n_expected=None, missing=None, extra=None, complete=None)
            continue
        want = set(J.expected_jobs(exp))
        got = {J.row_identity(r) for r in rs}
        out[exp] = dict(n_rows=len(rs), n_expected=len(want),
                        missing=len(want - got), extra=len(got - want),
                        complete=(want <= got))
    return out


def prechecks(man, by_exp, res, s1, mech, disc_ok, disc_note, cov, negc=None, r1=None):
    """★라벨을 내기 **전에** 도는 검사. `block` 이 참인 항목이 FAIL 이면 판정을 거부한다."""
    P = []
    seeds = man.get('train_scene_seeds_sample')
    P.append(('P-1 [F-1] 학습 시드 표본 ≥1 (없으면 G-9 가 공허 PASS)',
              bool(seeds) and len(seeds) >= 1, f'{len(seeds) if seeds else 0}개',
              ('e1', 's1')))
    ns = {k: res[k]['n'] for k in G.FAMILY} if res else {}
    P.append((f'P-2 [F-2] 세 대비 전부 n == {G.N_REAL} (MDE 는 n={G.N_REAL} 에서 동결)',
              bool(ns) and all(v == G.N_REAL for v in ns.values()), str(ns), ('e1',)))
    nc = (s1 or {}).get('neg_control') or {}
    P.append((f'P-3 [F-3] E2 음성대조 n == {G.S1_N_EP} (SPEC25 · ZERO 합성 확인)',
              nc.get('n') == G.S1_N_EP, f"n={nc.get('n')} better={nc.get('better')}",
              ('s1',)))
    # ★F-3 과 **같은 부류의 결함 둘**을 자가진단의 삭제 민감도가 새로 찾았다(AMENDMENT_1 §5.5
    #   목록에 없다). 둘 다 *"팔이 통째로 없으면 규칙이 공허하게 만족된다"* 이다.
    #   · `neg_control` 은 `aborts_inference = bool(n and …)` ⟹ **`BASE-B` 가 0행이면 중단 규칙이
    #     발화하지 않는다** = 음성대조 없이 요인 귀속이 나간다.
    #   · `r1_regression` 은 `SHIPPED` 가 0행이면 `bar=None` ⟹ 모든 칸이 `passed=False` 로 떨어져
    #     `TRADE`/`UNINTERPRETABLE-NULL` 로 **사유를 잘못 붙인다**(바를 깬 것이 아니라 바가 없다).
    nn = (negc or {}).get('n')
    P.append((f'P-3b [F-3 부류] 음성대조 BASE vs BASE-B 가 n == {G.N_REAL} '
              '(0행이면 중단 규칙이 공허하게 만족된다)',
              nn == G.N_REAL, f'n={nn}', ('e1',)))
    ns_ship = (r1 or {}).get('shipped_n_ep')
    P.append((f'P-3c [F-3 부류] R1 바의 기준(SHIPPED)이 실측됐다 n == {G.E2_N_EP} '
              '(0행이면 바가 깨진 것이 아니라 없는 것이다)',
              ns_ship == G.E2_N_EP and (r1 or {}).get('shipped_mean') is not None,
              f"n_ep={ns_ship} mean={(r1 or {}).get('shipped_mean')}", ('e1',)))
    P.append(('P-4 [F-4] 기제 부차 측정됨(varpro 목적함수) — ⚠바 없음, 공시 전용',
              bool(mech), f'{sorted(mech)}' if mech else '{} ⟹ **기제 부차 미측정** 공시',
              ()))
    P.append(('P-5 커버리지: 러너 작업 목록과 행이 일치',
              bool(cov['e1']['complete'] or cov['e2']['complete'] or cov['e3']['complete']),
              json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'n_expected'}
                          for k, v in cov.items()}, ensure_ascii=False), ()))
    P.append(('P-5a E1 커버리지 완전', bool(cov['e1']['complete']),
              str(cov['e1']), ('e1',)))
    P.append(('P-5b E2 커버리지 완전', bool(cov['e2']['complete']),
              str(cov['e2']), ('s1',)))
    counters = ('rows_ckpt_mismatch', 'rows_route_not_learned', 'bank_rows_wrong_route',
                'rows_missing_add_calls')
    nones = [k for k in counters if man.get(k) is None]
    P.append(('P-6 커버리지 0 계수기 없음(있으면 그 게이트는 자료 없이 통과한다)',
              not nones, f'None 계수기: {nones}' if nones else '없음', ('e1',)))
    P.append(('P-7 분기표 초과 공시 문언 확보(AMENDMENT_1 §1.3, 손복사 0)',
              bool(disc_ok), disc_note, ('e1', 's1')))
    # ★P-8 — **두 독립 유도의 대조**: 학습 원장의 주장 vs 파일 되읽기.
    #   `census_to_manifest` 가 `train_m7._gate_selfcheck` 의 거울이라 둘이 갈릴 수 있다.
    #   갈리면 조용히 한쪽을 쓰지 말고 **판정을 멈춘다**.
    claim = man.get('_census_claim') or {}
    diff = [k for k in ('n_pretrain', 'n_finetune', 'shipped_sha256')
            if k in claim and claim[k] != man.get(k)]
    ck_c, ck_f = claim.get('ckpt_sha256') or {}, man.get('ckpt_sha256') or {}
    diff += [f'ckpt:{c}' for c in sorted(set(ck_c) | set(ck_f))
             if ck_c.get(c) != ck_f.get(c)]
    P.append(('P-8 학습 원장의 주장 == 체크포인트 파일 되읽기 (두 독립 유도)',
              not diff, f'불일치 {diff}' if diff else
              f"n_pre={man.get('n_pretrain')} n_ft={man.get('n_finetune')} sha 일치",
              ('e1', 's1')))
    # ★P-9 — 공시 전용. `G-6` 이 세는 `below_repr` 은 재시도 경로라 **구조적으로 0** 이다
    #   (`gen_m7.probe_repr_filter` 독스트링 축자). 분모가 살아 있는 수를 여기서 드러낸다.
    hp = man.get('hard_probe') or {}
    P.append(('P-9 표현 필터의 **정직한 분모**(probe) 존재 — ⚠공시 전용, 차단 아님',
              bool(hp.get('n_scored')),
              (f"HARD probe n={hp.get('n_scored')} below={hp.get('below')} "
               f"frac_below_min={hp.get('frac_below_min')} p10={hp.get('p10')}"
               if hp else '⚠HARD probe 없음 — G-6 의 0 은 재시도 경로의 항등식일 수 있다'),
              ()))
    return [dict(name=n, passed=bool(ok), msg=str(m), blocks=list(b)) for n, ok, m, b in P]


# ── ★등급 부착 (AMENDMENT_1 §2.3) ────────────────────────────────────────────
def grade_label(label):
    """`DATA` 축 등급을 **기계로** 붙인다. ⛔결과를 보고 정하지 않는다."""
    borne = label in DATA_BORNE_LABELS
    grade = ('EXPLORATORY (라벨의 하중이 DATA 축에 실린다)' if borne
             else 'PREREGISTERED (LOSS 축) + EXPLORATORY (DATA 축)')
    return dict(label=label, reported_label=f'{label} · DATA 축 = EXPLORATORY',
                grade=grade, data_axis_is_load_bearing=borne,
                per_test=dict(GRADES), bans=list(DATA_AXIS_BANS))


# ── E2 : ZERO 합성 (F-3) + E3 기술 집계 ─────────────────────────────────────
def synth_zero(rows):
    """⛔런 0회 — `B-12` 규약: `ZERO` 는 기준선 행의 `zero_nRMSE` 열이다."""
    out = []
    for r in rows:
        if r.get('cell') == J.SPEC_ARM and r.get('zero_nRMSE') is not None:
            out.append(dict(cell=J.ZERO_ARM, exp=r.get('exp'), episode=r['episode'],
                            filter_seed=r.get('filter_seed'),
                            avg_nRMSE=float(r['zero_nRMSE']), synthesized_from='SPEC25.zero_nRMSE'))
    return out


def descriptive_e3(rows):
    """E3 = **보고 전용**. ⛔바 없음 · HARD 팔에게는 in-family(out-of-realisation)."""
    def med(cell, sub):
        per = {}
        for r in sub:
            if r.get('cell') == cell:
                per.setdefault(r['episode'], []).append(r['avg_nRMSE'])
        return {k: st.median(v) for k, v in per.items()}
    out = {}
    for arm in ('LIN', 'HOS'):
        sub = [r for r in rows if str(r.get('episode', '')).startswith(arm + '|')]
        if not sub:
            continue
        spec = med(J.SPEC_ARM, sub)
        blk = {}
        for c in list(G.CELL_ORDER) + [G.REF_SHIPPED]:
            m = med(c, sub)
            ks = sorted(set(m) & set(spec))
            blk[c] = dict(n=len(ks), better=sum(1 for k in ks if m[k] < spec[k]),
                          median=(st.median(m.values()) if m else None))
        out[arm] = dict(n_ep=len(spec), spec_median=(st.median(spec.values()) if spec else None),
                        per_cell=blk)
    out['note'] = ('⛔보고 전용 — 바 없음. HARD 로 학습한 칸에게 이 코너는 **in-family** '
                   '(학습 앵커가 실측 중앙값이고 M1 이 그 코너다). 확증에 쓰지 마라.')
    return out


# ── 본체 ─────────────────────────────────────────────────────────────────────
def analyze(rows, train_man=None, fs=None, write=True, verbose=True):
    def say(s):
        if verbose:
            print(s, flush=True)

    ok = [r for r in rows if 'error' not in r]
    if not ok:                                              # ★0행 가드
        say('⛔유효 0행 — verdict 을 쓰지 않는다 (0행에서 판정 금지).')
        return None, 1

    by_exp = {e: [r for r in ok if r.get('exp') == e] for e in J.EXPERIMENTS}
    e1 = by_exp['e1']
    e2 = list(by_exp['e2'])
    zero_rows = synth_zero(e2)
    e2z = e2 + zero_rows
    e3 = list(by_exp['e3']) + synth_zero(by_exp['e3'])

    train_man = load_train_manifest() if train_man is None else dict(train_man)
    man = build_manifest(by_exp, train_man, fs=fs)
    gates = [dict(name=n, passed=bool(p), msg=str(m)) for n, p, m in G.gates(man)]
    gmap = {gid(g['name']): g['passed'] for g in gates}

    res = G.contrasts(e1) if e1 else None
    hol = G.holm(res) if res else None
    negc = G.neg_control(e1) if e1 else None
    r1 = G.r1_regression(e2z) if e2 else None
    s1 = G.s1_shift_survives(e2z) if e2 else None
    mech = G.mechanism(e1) if e1 else {}
    disc, disc_ok, disc_note = excess_disclosure()
    cov = coverage(by_exp)
    pre = prechecks(man, by_exp, res, s1, mech, disc_ok, disc_note, cov, negc=negc, r1=r1)

    def blocked(scope, block_gates):
        bad_pre = [p['name'] for p in pre if scope in p['blocks'] and not p['passed']]
        bad_g = [g for g in block_gates if not gmap.get(g, False)]
        return bad_pre, bad_g

    # ── PRIMARY (E1) ────────────────────────────────────────────────────────
    bp, bg = blocked('e1', BLOCK_E1)
    if not e1:
        primary = dict(status='INCOMPLETE', reason='E1 행 0개 (사용자 승인 전이면 정상)',
                       blocked_by=dict(prechecks=bp, gates=bg))
    elif bp or bg:
        primary = dict(status='INCOMPLETE',
                       reason='선행 검사 또는 차단 게이트 FAIL ⟹ **라벨을 내지 않는다**',
                       blocked_by=dict(prechecks=bp, gates=bg))
    else:
        lab, txt = G.label(res, hol, negc, r1)
        primary = dict(status='LABELED', label_text=txt, **grade_label(lab))
    primary.update(contrasts={k: {kk: vv for kk, vv in v.items() if kk != 'per_record'}
                              for k, v in (res or {}).items()},
                   per_record={k: v['per_record'] for k, v in (res or {}).items()},
                   holm=hol, neg_control=negc, mde=G.mde(),
                   skill_vs_true_zero=(_skill(e1) if e1 else None),
                   mechanism=mech, mechanism_measured=bool(mech))

    # ── S-1 (E2) ────────────────────────────────────────────────────────────
    bp1, bg1 = blocked('s1', BLOCK_S1)
    if not e2:
        s1v = dict(status='INCOMPLETE', reason='E2 행 0개')
    elif bp1 or bg1:
        s1v = dict(status='INCOMPLETE', reason='선행 검사 또는 차단 게이트 FAIL',
                   blocked_by=dict(prechecks=bp1, gates=bg1))
    else:
        l1, t1 = G.label_s1(s1)
        s1v = dict(status='LABELED', label=l1, label_text=t1, grade=GRADES['S-1'])
    s1v.update(detail=s1, r1_hard_bar=r1, n_zero_rows=len(zero_rows))

    # ── E3 (보고 전용) ──────────────────────────────────────────────────────
    bg3 = [g for g in BLOCK_E3 if not gmap.get(g, False)]
    e3v = (dict(status='INCOMPLETE', reason='E3 행 0개 또는 차단 게이트 FAIL',
                blocked_by=bg3) if (not by_exp['e3'] or bg3)
           else dict(status='REPORTED', **descriptive_e3(e3)))

    verdict = dict(
        batch='M7_SUPERVISION_2026-08-28',
        question='넷이 실측에서 자신 있게 틀리는 것은 학습 *분포* 때문인가 *손실* 때문인가',
        n_rows=len(rows), n_ok=len(ok),
        n_by_exp={k: len(v) for k, v in by_exp.items()},
        coverage=cov,
        label=(primary.get('reported_label')
               or f'M7 :: PRIMARY = {primary["status"]} ({primary.get("reason", "")})'),
        excess_disclosure=disc, excess_disclosure_ok=disc_ok,
        excess_disclosure_note=disc_note,
        grades=dict(GRADES), data_axis_bans=list(DATA_AXIS_BANS),
        prechecks=pre, prechecks_pass=sum(1 for p in pre if p['passed']),
        gates=gates, gates_pass=sum(1 for g in gates if g['passed']), gates_total=len(gates),
        blocking=dict(E1=list(BLOCK_E1), S1=list(BLOCK_S1), E3=list(BLOCK_E3)),
        manifest={k: v for k, v in man.items() if k != 'frozen_detail'},
        frozen_detail=man.get('frozen_detail'),
        primary=primary, s1=s1v, e3=e3v,
        disclosures=_disclosures(mech, man, cov))

    if write:
        _write(verdict, by_exp)
    if verbose:
        _print(verdict)
    hard_fail = (primary['status'] != 'LABELED' and s1v['status'] != 'LABELED')
    return verdict, (1 if hard_fail else 0)


def _skill(e1):
    from zero_ref import zero_ref_nowcast
    zr = {r: float(zero_ref_nowcast(r, win=G.REAL_WIN_PRIMARY)) for r in G.REAL_RECORDS}
    out = G.skill_vs_true_zero(e1, zr)
    out['_note'] = ('⛔DESCRIPTIVE — 귀무가 0.5 라는 근거가 없다. '
                    '⛔`ZERO_REF=1.0` 금지(참값 레코드별 0.3375–1.1559).')
    out['_zero_ref'] = zr
    return out


def _disclosures(mech, man, cov):
    d = []
    if not mech:
        d.append('⚠**기제 부차 미측정** — 행에 `varpro_obj_before` 가 없다(F-4 공시).')
    else:
        missing = [c for c in G.ARMS_REAL if c not in mech]
        if missing:
            d.append(f'⚠기제 부차 일부 미측정: {missing} (F-4 공시).')
    if (man.get('hard_scenes_below_repr') or 0) == 0 and man.get('hard_scenes_n'):
        hp = man.get('hard_probe') or {}
        d.append(f"HARD 학습 스트림 표현 필터: {man['hard_scenes_n']} 장면 중 미달 0건(G-6). "
                 '⚠**그 0 은 재시도 경로의 항등식에 가깝다** — 분모가 살아 있는 측정은 '
                 f"`gen_m7.probe_repr_filter` 이고 그 값은 "
                 f"{('n=%s below=%s frac_below_min=%s' % (hp.get('n_scored'), hp.get('below'), hp.get('frac_below_min'))) if hp else '**없음**'} 이다.")
    d.append('⚠공시(PREREG §8-3): HARD 장면의 약 10%는 목표 에너지의 40%가 '
             '빔포밍 조건자 대역(|k|≤0.6) **밖**이다 — 결함이 아니라 표현의 상한.')
    d.append('⚠공시(PREREG §8-1): HARD 앵커는 E1 이 평가하는 바로 그 12 레코드의 '
             '스칼라 2개(중앙 Hs, Tp_disp)에서 유도됐다. 누출에서 자유로운 검사는 **E2** 다.')
    d.append('⚠공시(PREREG §8-2): `DATA` 요인은 **묶음**(해상상태 + 2차 비선형 관측)이다 — '
             '양성이어도 어느 쪽인지 말할 수 없다.')
    d.append('⚠공시(PREREG §8-10): 이 배치는 **제안 단계만** 개입한다. 실측 과적합의 측정된 '
             '발생 지점은 **시험 시점**(VARPRO·추적기)이다(`W12`).')
    for e in J.EXPERIMENTS:
        if cov[e]['complete'] is False:
            d.append(f'⛔{e.upper()} 커버리지 불완전: 결측 {cov[e]["missing"]} · '
                     f'초과 {cov[e]["extra"]}.')
    return d


def _write(verdict, by_exp):
    json.dump(verdict, open(os.path.join(HERE, 'verdict_m7.json'), 'w'),
              ensure_ascii=False, indent=1)
    # ★결과 파일마다 짝 verdict (`check_batch_artifacts` H) — 각각 `label`·`gates` 를 갖는다.
    pair = {'e1': ('primary', verdict['primary']), 'e2': ('s1', verdict['s1']),
            'e3': ('e3', verdict['e3'])}
    for e, (key, blk) in pair.items():
        if not glob.glob(os.path.join(HERE, f'results_m7_{e}*.jsonl')):
            continue
        lab = (verdict['label'] if e == 'e1'
               else (f"{blk.get('label')} · 등급 {blk.get('grade')}" if blk.get('label')
                     else f"{e.upper()} = {blk.get('status')} ({blk.get('reason', '')})"))
        json.dump(dict(batch=verdict['batch'], experiment=e, label=lab,
                       excess_disclosure=verdict['excess_disclosure'],
                       gates=verdict['gates'], prechecks=verdict['prechecks'],
                       coverage=verdict['coverage'][e], **{key: blk}),
                  open(os.path.join(HERE, f'verdict_m7_{e}.json'), 'w'),
                  ensure_ascii=False, indent=1)


def _print(v):
    p = lambda s='': print(s, flush=True)
    p('\n=== 선행 검사 (F-1…F-4 · 커버리지) ===')
    for q in v['prechecks']:
        p(f'  [{"PASS" if q["passed"] else "FAIL"}] {q["name"]}: {q["msg"][:120]}')
    p(f'\n=== 게이트 {v["gates_pass"]}/{v["gates_total"]} ===')
    for g in v['gates']:
        if not g['passed']:
            p(f'  ⛔FAIL {g["name"]}: {g["msg"][:110]}')
    pr = v['primary']
    p(f'\n=== E1 (PRIMARY) — {pr["status"]} ===')
    if pr['status'] == 'LABELED':
        p(f'  라벨 : {pr["reported_label"]}')
        p(f'  등급 : {pr["grade"]}')
        p(f'  해설 : {pr["label_text"]}')
    else:
        p(f'  사유 : {pr.get("reason")}  {pr.get("blocked_by", "")}')
    for k in G.FAMILY:
        c = pr['contrasts'].get(k)
        if c:
            h = (pr['holm'] or {}).get(k, {})
            p(f'  {k:16s} n={c["n"]} wins-={c["wins_negative"]} p={c["p"]:.6f} '
              f'thr={h.get("threshold")} reject={h.get("reject")} '
              f'median={c["median_delta"]} [{v["grades"].get(k)}]')
    nc = pr.get('neg_control')
    if nc:
        p(f'  음성대조 BASE vs BASE-B: {nc["wins"]}/{nc["n"]} (최대편 {nc["max_side"]}, '
          f'중단문턱 {G.NEG_CONTROL_ABORT}) aborts={nc["aborts_inference"]}')
    p(f'  MDE: {pr["mde"]}')
    s = v['s1']
    p(f'\n=== E2 (S-1 + 하드 바) — {s["status"]} ===')
    if s['status'] == 'LABELED':
        p(f'  라벨 : {s["label"]} [{s["grade"]}]  — {s["label_text"][:90]}')
    else:
        p(f'  사유 : {s.get("reason")}  {s.get("blocked_by", "")}')
    if s.get('r1_hard_bar'):
        r1 = s['r1_hard_bar']
        p(f'  R1 바(SHIPPED {r1["shipped_mean"]} + {G.R1_MARGIN}): all_pass={r1["all_pass"]}')
        for c, d in r1['per_cell'].items():
            p(f'    {c:8s} mean={d["mean"]} n_ep={d["n_ep"]} passed={d["passed"]}')
    if s.get('detail'):
        d = s['detail']
        p(f'  음성대조 SPEC25<ZERO: {d["neg_control"]["better"]}/{d["neg_control"]["n"]} '
          f'(문턱 {G.S1_NEG_MIN}) ok={d["neg_control"]["ok"]} · '
          f'B12 문턱 패리티={d["threshold_parity_with_B12"]}')
        for c, dd in d['per_cell'].items():
            p(f'    {c:8s} {dd["better"]}/{dd["n"]} nominal={dd["nominal"]} '
              f'confirmatory={dd["confirmatory"]}')
    p(f'\n=== E3 (보고 전용) — {v["e3"]["status"]} ===')
    p('\n=== ★분기표 초과 공시 (동반 의무) ===')
    p(f'  {v["excess_disclosure"] or "⛔확보 실패: " + str(v["excess_disclosure_note"])}')
    p('\n=== 공시 ===')
    for d in v['disclosures']:
        p(f'  · {d}')
    p(f'\n★판정 : {v["label"]}')


# ══════════════════════════════════════════════════════════════════════════════
# 자가진단 — 런 0회 · GPU 0회 · 파일 쓰기 0회
# ══════════════════════════════════════════════════════════════════════════════
FIX_SHA = {c: f'{i:064x}' for i, c in enumerate(G.CELL_ORDER, start=1)}
SHIPPED_SHA = '636c72a797ee6d0b725449293d5cd230dbd64c8eb03e345da4561b0c265239d9'


def _fix_fs():
    return dict(ckpt_sha256=dict(FIX_SHA), n_pretrain=3, n_finetune=5,
                shipped_sha256=SHIPPED_SHA, frozen_drift=0, frozen_n=31, frozen_detail=[],
                m1_verdict_frozen=True, m1_pair_key=['hold', 'hold'])


def _fix_train_man():
    return dict(train_scene_seeds_sample=[2_000_000_001, 2_000_000_002, 2_100_000_000],
                gen_parity_maxabs=0.0, hard_scenes_n=400, hard_scenes_below_repr=0,
                hard_kHs_median=0.244, base_kHs_median=0.093,
                anchors_excluded=['LJ18a'], lam0_finetune_steps=G.FINETUNE_STEPS,
                eval_records=list(G.REAL_RECORDS),
                hard_probe=dict(n=400, n_scored=400, below=0, frac_below_min=0.0,
                                median=0.998, p10=0.593, min=0.506),
                # 학습 원장의 **주장** — `_fix_fs()` 의 파일 되읽기와 일치해야 P-8 이 통과한다
                n_pretrain=3, n_finetune=5, shipped_sha256=SHIPPED_SHA,
                ckpt_sha256=dict(FIX_SHA))


def _fix_rows(loss_effect=-0.02, data_effect=0.0, seed=7):
    """★픽스처는 **러너의 작업 목록과 스탬프**로 만든다(`jobs_m7.expected_jobs`/`stamp`).
    ⛔게이트를 모형해 손으로 행을 짜면 그것이 「의도표」다."""
    import random
    rnd = random.Random(seed)
    sha = dict(FIX_SHA)
    sha[G.REF_SHIPPED] = SHIPPED_SHA
    base_r = {r: 0.60 + 0.02 * i for i, r in enumerate(G.REAL_RECORDS)}
    rows = []
    for ident in J.expected_jobs('e1'):
        _, cell, rec, ds = ident
        bank = cell == G.REF_BANK
        eff = (loss_effect if cell in ('LOSS', 'BOTH') else 0.0) + \
              (data_effect if cell in ('DATA', 'BOTH') else 0.0)
        val = base_r[rec] + eff + 0.001 * (ds or 0) + rnd.uniform(-1e-4, 1e-4)
        rows.append(J.stamp(ident, dict(
            nowcast=val, leadmean=val * 1.1, curve=[val], Hs=0.7, sigma=0.2,
            n_atoms=0 if bank else 24, add_calls=0 if bank else 1, add_atoms=0,
            varpro_fired=not bank, varpro_obj_before=None if bank else 1.0,
            varpro_obj_after=None if bank else 0.4,
            ckpt_path=None if bank else 'results/direction3/x.pt',
            ckpt_sha256=None if bank else sha[cell],
            route_init_method='bank_mmse' if bank else 'learned',
            route_real_gate=True if bank else False, route_bank=bank,
            zero_ref=1.0, zero_ref_source='tmp/B2_REAL_2026-08-23/zero_ref.py',
            cuda_available=True, wall_s=4.2)))
    for exp in ('e2', 'e3'):
        for ident in J.expected_jobs(exp):
            _, cell, ep, sd = ident
            spec = cell == J.SPEC_ARM
            eff = (loss_effect * 0.1 if cell in ('LOSS', 'BOTH') else 0.0)
            val = (0.30 if spec else 0.18 + eff) + rnd.uniform(-1e-3, 1e-3)
            extra = dict(zero_nRMSE=0.55, loc_radius=25.0, n_ensemble=600,
                         background='jonswap') if spec else dict(
                ckpt_sha256=sha[cell], ckpt_path='results/direction3/x.pt')
            rows.append(J.stamp(ident, dict(
                avg_nRMSE=val, n_atoms=0 if spec else 24, add_calls=0 if spec else 1,
                varpro_obj_before=None if spec else 1.0,
                varpro_obj_after=None if spec else 0.4,
                cuda_available=True, wall_s=8.1, **extra)))
    return rows


def _run(rows, train_man=None, fs=None):
    return analyze(rows, train_man=_fix_train_man() if train_man is None else train_man,
                   fs=_fix_fs() if fs is None else fs, write=False, verbose=False)[0]


def selftest(verbose=True):
    def say(s):
        if verbose:
            print(s, flush=True)
    fails = []
    say('★analyze_m7 자가진단 ─────────────────────────────────────')
    ok_j, f_j = J.selftest(verbose=False)
    fails += list(f_j)

    # [1] 정상 픽스처 — 전 게이트 PASS · 전 선행검사 PASS · 라벨 발화
    rows = _fix_rows()
    v = _run(rows)
    bad_g = [g['name'] for g in v['gates'] if not g['passed']]
    bad_p = [p['name'] for p in v['prechecks'] if not p['passed']]
    if bad_g:
        fails.append(f'[1] 픽스처에서 FAIL 한 게이트: {bad_g}')
    if bad_p:
        fails.append(f'[1] 픽스처에서 FAIL 한 선행검사: {bad_p}')
    if v['primary']['status'] != 'LABELED' or v['s1']['status'] != 'LABELED':
        fails.append(f'[1] 라벨 미발화: E1={v["primary"]["status"]} S1={v["s1"]["status"]}')
    say(f'  [1] 정상 픽스처: 게이트 {v["gates_pass"]}/{v["gates_total"]} · '
        f'선행 {v["prechecks_pass"]}/{len(v["prechecks"])} · '
        f'E1={v["primary"].get("label")} · S1={v["s1"].get("label")}')

    # [2] ★커버리지 0 — 빈 매니페스트·빈 행에서 **공허하게 통과하는 게이트**를 전부 드러낸다
    empty = [dict(name=n, passed=bool(p)) for n, p, _ in G.gates({})]
    vac = [g['name'] for g in empty if g['passed']]
    if [gid(x) for x in vac] != ['G-9']:
        fails.append(f'[2] 공허 통과 게이트 집합이 바뀌었다: {vac} (기대 G-9 하나)')
    v0 = _run([], train_man={}, fs=dict(_fix_fs(), ckpt_sha256={}, n_pretrain=0, n_finetune=0))
    if v0 is not None:
        fails.append('[2] 0행에서 verdict 을 냈다 — 0행 가드 실패')
    say(f'  [2] 커버리지 0: gates({{}}) 공허 통과 = {vac} · 0행 ⟹ verdict 없음 ✓')

    # [3] ★삭제 민감도 — 팔 하나를 지우면 **반드시 누가 FAIL** 한다
    fired = []
    for cell in list(G.CELL_ORDER) + [G.REF_SHIPPED, G.REF_BANK, J.SPEC_ARM]:
        sub = [r for r in rows if r.get('cell') != cell]
        vv = _run(sub)
        hit = ([g['name'] for g in vv['gates'] if not g['passed']]
               + [p['name'] for p in vv['prechecks'] if not p['passed']])
        if not hit:
            fails.append(f'[3] 팔 {cell} 을 통째로 지웠는데 아무도 FAIL 하지 않았다')
        else:
            fired.append(f'{cell}->{gid(hit[0])}')
    say(f'  [3] 삭제 민감도: {len(fired)}/{len(fired)} 발화 ({", ".join(fired[:4])} …)')

    # [4] ★F-1 회귀 — 시드 표본이 비면 G-9 는 통과하는데 P-1 이 판정을 거부한다
    v1 = _run(rows, train_man=dict(_fix_train_man(), train_scene_seeds_sample=[]))
    g9 = [g for g in v1['gates'] if gid(g['name']) == 'G-9'][0]
    p1 = [p for p in v1['prechecks'] if p['name'].startswith('P-1')][0]
    if not g9['passed']:
        fails.append('[4] F-1 픽스처에서 G-9 가 FAIL 했다 — 결함 재현이 안 된다')
    if p1['passed'] or v1['primary']['status'] == 'LABELED' or v1['s1']['status'] == 'LABELED':
        fails.append('[4] F-1: 빈 시드 표본인데 P-1 이 통과했거나 라벨이 나왔다')
    say('  [4] F-1: G-9 공허 PASS 재현 · P-1 FAIL · 라벨 거부 ✓')

    # [5] ★F-2 회귀 — 한 칸·한 레코드를 지우면 n=11 ⟹ INCOMPLETE (라벨 없음)
    drop = G.REAL_RECORDS[0]
    sub = [r for r in rows if not (r.get('exp') == 'e1' and r.get('cell') == 'LOSS'
                                   and r.get('record') == drop)]
    v2 = _run(sub)
    ns = {k: c['n'] for k, c in v2['primary']['contrasts'].items()}
    p2 = [p for p in v2['prechecks'] if p['name'].startswith('P-2')][0]
    if ns.get('T-1_loss') != G.N_REAL - 1:
        fails.append(f'[5] F-2 픽스처가 n 을 줄이지 못했다: {ns}')
    if p2['passed'] or v2['primary']['status'] != 'INCOMPLETE':
        fails.append(f'[5] F-2: n={ns} 인데 P-2 통과 또는 라벨 발화')
    say(f'  [5] F-2: 결측 칸 1개 ⟹ n={ns["T-1_loss"]} · P-2 FAIL · PRIMARY=INCOMPLETE ✓')

    # [6] ★F-3 회귀 — SPEC25 가 없으면 label_s1 은 조용히 착지하는데 P-3 이 거부한다
    sub = [r for r in rows if not (r.get('exp') == 'e2' and r.get('cell') == J.SPEC_ARM)]
    v3 = _run(sub)
    p3 = [p for p in v3['prechecks'] if p['name'].startswith('P-3')][0]
    silent = G.label_s1(G.s1_shift_survives([r for r in sub if r.get('exp') == 'e2']))[0]
    if silent != 'BASELINE-NO-SKILL-HERE':
        fails.append(f'[6] F-3 결함 재현 실패 — 조용한 착지 라벨이 {silent}')
    if p3['passed'] or v3['s1']['status'] == 'LABELED':
        fails.append('[6] F-3: SPEC25 부재인데 P-3 통과 또는 S-1 라벨 발화')
    # ZERO 합성이 실제로 작동하는지도 확인(정상 픽스처에서)
    if v['s1']['n_zero_rows'] != G.S1_N_EP * 3:
        fails.append(f'[6] ZERO 합성 행 수 이상: {v["s1"]["n_zero_rows"]}')
    say(f'  [6] F-3: SPEC25 부재 ⟹ 조용한 라벨 `{silent}` 재현 · P-3 FAIL · S-1 거부 · '
        f'정상 시 ZERO 합성 {v["s1"]["n_zero_rows"]}행 ✓')

    # [6b] ★F-3 부류(자가진단이 새로 찾은 둘) — 팔이 통째로 없으면 규칙이 공허하게 만족된다
    sub = [r for r in rows if r.get('cell') != 'BASE-B']
    e1b = [r for r in sub if r.get('exp') == 'e1']
    if G.neg_control(e1b)['aborts_inference']:
        fails.append('[6b] BASE-B 0행인데 중단 규칙이 발화했다 — 결함 재현 실패')
    v5 = _run(sub)
    p3b = [p for p in v5['prechecks'] if p['name'].startswith('P-3b')][0]
    if p3b['passed'] or v5['primary']['status'] == 'LABELED':
        fails.append('[6b] P-3b 가 BASE-B 부재를 막지 못했다')
    sub = [r for r in rows if not (r.get('exp') == 'e2' and r.get('cell') == G.REF_SHIPPED)]
    v6 = _run(sub)
    p3c = [p for p in v6['prechecks'] if p['name'].startswith('P-3c')][0]
    if p3c['passed'] or v6['primary']['status'] == 'LABELED':
        fails.append('[6b] P-3c 가 SHIPPED 부재를 막지 못했다')
    say('  [6b] F-3 부류 신규 2건: BASE-B 0행 ⟹ 중단 규칙 공허 만족 재현 · P-3b 차단 / '
        'SHIPPED 0행 ⟹ R1 바 사유 오귀속 · P-3c 차단 ✓')

    # [7] ★F-4 회귀 — varpro 목적함수가 없으면 mechanism()=={} ⟹ **공시**(차단 아님)
    sub = [dict(r, varpro_obj_before=None) if r.get('exp') == 'e1' else r for r in rows]
    v4 = _run(sub)
    p4 = [p for p in v4['prechecks'] if p['name'].startswith('P-4')][0]
    if v4['primary']['mechanism_measured'] or p4['passed']:
        fails.append('[7] F-4: 기제가 비었는데 P-4 가 통과했다')
    if v4['primary']['status'] != 'LABELED':
        fails.append('[7] F-4: 공시 항목인데 판정을 차단했다(P-4 는 차단이 아니다)')
    if not any('기제 부차 미측정' in d for d in v4['disclosures']):
        fails.append('[7] F-4: 공시 문장이 판정문에 실리지 않았다')
    say('  [7] F-4: mechanism()=={} ⟹ P-4 FAIL(공시) · 라벨은 그대로 발화 ✓')

    # [7b] ★P-8 / P-9 — 두 독립 유도의 대조 · 표현 필터의 정직한 분모
    v7 = _run(rows, train_man=dict(_fix_train_man(),
                                   ckpt_sha256=dict(FIX_SHA, BASE='0' * 64)))
    p8 = [p for p in v7['prechecks'] if p['name'].startswith('P-8')][0]
    if p8['passed'] or v7['primary']['status'] == 'LABELED':
        fails.append('[7b] P-8 이 원장/파일 sha 불일치를 막지 못했다')
    v8 = _run(rows, train_man={k: v for k, v in _fix_train_man().items() if k != 'hard_probe'})
    p9 = [p for p in v8['prechecks'] if p['name'].startswith('P-9')][0]
    g6 = [g for g in v8['gates'] if gid(g['name']) == '★G-6'][0]
    if p9['passed'] or not g6['passed']:
        fails.append('[7b] P-9: probe 부재인데 공시가 안 붙었거나 G-6 이 FAIL 했다')
    if v8['primary']['status'] != 'LABELED':
        fails.append('[7b] P-9 는 공시 전용인데 판정을 차단했다')
    say('  [7b] P-8: 원장 주장 vs 파일 되읽기 불일치 ⟹ 차단 · '
        'P-9: probe 부재 ⟹ 공시(차단 아님, G-6 은 그대로 PASS) ✓')

    # [8] ★EXPLORATORY 자동 부착 — 도달 가능한 모든 라벨에서 등급이 실린다
    labs = set()
    for le in (-0.05, 0.0):
        for de in (-0.05, 0.0):
            vv = _run(_fix_rows(loss_effect=le, data_effect=de))
            if vv['primary']['status'] != 'LABELED':
                fails.append(f'[8] loss={le} data={de} 에서 라벨 미발화')
                continue
            labs.add(vv['primary']['label'])
            if 'DATA 축 = EXPLORATORY' not in vv['primary']['reported_label']:
                fails.append(f'[8] {vv["primary"]["label"]} 에 EXPLORATORY 가 안 붙었다')
            if vv['grades']['T-2_data'] != 'EXPLORATORY':
                fails.append('[8] T-2_data 등급이 EXPLORATORY 가 아니다')
    for lab in DATA_BORNE_LABELS:
        if not grade_label(lab)['data_axis_is_load_bearing']:
            fails.append(f'[8] {lab} 이 DATA-하중 라벨로 분류되지 않았다')
    say(f'  [8] 등급 자동 부착: 도달 라벨 {sorted(labs)} 전부 `DATA 축 = EXPLORATORY` ✓')

    # [9] ★분기표 초과 공시 문언이 **읽혀서** 판정문에 실리는가 (손복사 0)
    txt, dok, note = excess_disclosure()
    if not dok or '분기표' not in (txt or '') or 'EXPLORATORY' in (txt or '')[:20]:
        fails.append(f'[9] 초과 공시 추출 실패: {note}')
    if v['excess_disclosure'] != txt:
        fails.append('[9] 공시 문언이 판정문에 실리지 않았다')
    say(f'  [9] 초과 공시: {note} · 길이 {len(txt or "")}자 · 판정문 삽입 ✓')

    # [10] 차단 게이트 이름이 실제 게이트 이름 집합의 부분집합인가 (오탈자 = 조용한 무력화)
    names = {gid(n) for n, _, _ in G.gates({})}
    for nm, s in (('BLOCK_E1', BLOCK_E1), ('BLOCK_S1', BLOCK_S1), ('BLOCK_E3', BLOCK_E3)):
        miss = [g for g in s if g not in names]
        if miss:
            fails.append(f'[10] {nm} 에 존재하지 않는 게이트 이름: {miss}')
    say(f'  [10] 차단 게이트 이름 정합: E1 {len(BLOCK_E1)} · S1 {len(BLOCK_S1)} · '
        f'E3 {len(BLOCK_E3)} (전체 {len(names)}) ✓')

    say('────────────────────────────────────────────────────────────')
    for f in fails:
        say(f'  ⛔ {f}')
    say(f'analyze_m7 자가진단 {"PASS" if not fails else "FAIL"} ({len(fails)} 결함)')
    return (not fails), fails


def main(argv):
    if '--selftest' in argv:
        return 0 if selftest()[0] else 1
    ok, fails = selftest(verbose=False)
    print('★분석 전 자가진단 ' + ('PASS' if ok else 'FAIL'), flush=True)
    if not ok:
        for f in fails:
            print(f'  ⛔ {f}', flush=True)
        print('⛔ 자가진단 실패 ⟹ 판정을 내지 않는다.', flush=True)
        return 1
    rows = load_rows()
    print(f'결과 행 {len(rows)}개 로드', flush=True)
    _v, rc = analyze(rows)
    return rc


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
