"""`T18-FACTORIAL` 분석기 — ⛔러너보다 **먼저** 쓰였다. ★**이 파일이 행 스키마를 정의한다.**

════════════════════════════════════════════════════════════════════════════
행 스키마 (사전등록 §6.3) — ⛔열이 빠지면 게이트가 **발화 불가**다.
  러너는 아래를 **되읽기로** 채운다. ⛔러너가 스스로 쓴 리터럴을 그대로 다시 비교하는 열은
  게이트가 되지 못한다(항진) — 그런 열에는 「되읽기 지점」을 함께 적었다.

  식별   episode · bf · ncomp · gen_seed · arm · est_seed · layer('PRIMARY'|'SEC-C')
  채점   avg_nRMSE · zero_nRMSE(★`gates_b19.zero_nrmse` 재계산, ⛔상수 1.0 아님)
         zero_n_frames · Hs_true · Tp_true
  창     n_frames_scored(= `run.py` 의 `tracking_frames`) · scored_t_first · scored_t_last
         episode_total_frames
  ★개입 k_snap_dk · k_snap_fired · k_snap_cells · k_snap_shift_median
         k_snap_shift_max · k_snap_n_atoms
           ★넷 모두 `sequential_update._snap_k_to_grid` 가 **돌려준 record 되읽기**다.
             ⛔러너가 넣은 `cfg['k_snap_dk']` 가 아니다 — 뒤의 둘은 `run.py` 요약이
             버리므로 러너가 그 메서드를 **관측 훅으로 감싸** 잡는다(거동 불변).
  ★원자 atoms_source · atoms_md5 · n_atoms_injected · n_pad_dropped · n_pool_expected
         propose_calls(★런 안에서 확산 제안이 돈 횟수 = 주입의 직접 증거)
  팔 정체 arm_provenance(★`upd.arm_provenance()` **전체** — 의도가 아니라 되읽기)
         auto_selected · auto_n_selected · auto_real_gate · ensemble_k · updater_class
         effective(★`gates_b19.effective_from_updater` — 객체 파생 속성)
  배포상수 deploy_readback(9종, ★`upd.config` **객체 되읽기**)
  ★용량  pool_cap_readback(`learned_init.resolve_pool_cap(upd.config)`)
         growth_ceiling_readback(`sequential_update.resolve_growth_ceiling(upd.config)`)
         max_n_alias(`upd.max_n` — 별칭 property 되읽기)
           ★2026-09-05 `a6eb5ba` 가 `max_n` 을 **두 이름으로 쪼갰다** ⟹ 사전등록 §6.1 의
             `G-MAXN` 은 한 역할만 봤다. 개정 = `AMENDMENT_1_PREDATA.md`.
  ★성장  grow_calls · grow_branch_entries · grow_threshold_pass · n_grow_atoms
           ⛔**호출이 아니라 분기를 센다** — `sequential_update.py:1675-1681` 의 주석이
             축자로 요구한 것이다(*"A per-call counter does NOT detect that … Count the BRANCH."*).
  기타   varpro_fired · varpro_steps · varpro_obj_before · varpro_obj_after
         n_gaussians · depth(★`upd.depth` = `loader.depth` 되읽기) · mesh_verdict
         wall_s · git_head · import_md5 · ckpt_sha256 · code_md5_row

  ⛔`(x or 기본값)` 을 쓰지 마라 — `0.0`·`None` 은 falsy 다. 이 파일은 `is None` 으로 가른다.
════════════════════════════════════════════════════════════════════════════

★판정 (사전등록 §0.2 · §3)
  d_i = [ln y(A-Q) − ln y(A-C)] − [ln y(L-Q) − ln y(L-C)]      (⛔단일출처 = `G.interaction_contrast`)
  W₊  = #{i : d_i > 0} 의 **짝 부호검정**(양측). d_i > 0 = 초가법.
  ★그리고 사전등록은 **결과 전에** `P(COMBINATION-SUPERADDITIVE) = 0.000000` 이라 적었다
    (가법 널이 43/48 에피소드에서 관측 범위 밖). ⟹ **등록된 산출물은 축 II** 다.
    ⛔결과 뒤에 이 문단을 다시 쓰지 마라.

사용:
    python3 analyze_t18.py                       # 본 판정 (results_t18*.jsonl)
    python3 analyze_t18.py --smoke --out DIR     # ★스모크: 배치 **밖**에 verdict 을 쓴다
"""
import argparse
import glob
import hashlib
import json
import math
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gates_t18 as G                                              # noqa: E402

ROWS_MAIN = os.path.join(HERE, 'results_t18*.jsonl')
ROWS_SMOKE = os.path.join(HERE, 'results_t18_SMOKE.jsonl')
META_MAIN = os.path.join(HERE, 'meta_t18.json')
META_SMOKE = os.path.join(HERE, 'meta_t18_SMOKE.json')
FROZEN = os.path.join(HERE, 'FROZEN_md5.txt')
FROZEN_A = os.path.join(HERE, 'FROZEN_md5_ANALYSIS.txt')
# ★측정면 / 분석면 분리 (사전등록 §9.1 · `T11 A3 §2`). ⛔러너가 이 목록을 다시 적지 않는다.
ANALYSIS_SIDE = ('analyze_t18.py',)


# ── 작은 도우미 ─────────────────────────────────────────────────────────────
def _num(x):
    """숫자면 float, 아니면 `None`. ⛔`or` 를 쓰지 않는 이유: `0.0` 은 falsy 다."""
    if x is None or isinstance(x, bool):
        return None
    return float(x) if isinstance(x, (int, float)) else None


def _md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def load_rows(pattern):
    out = []
    for p in sorted(glob.glob(pattern)):
        if p.endswith('_SMOKE.jsonl') and 'SMOKE' not in pattern:
            continue
        with open(p, encoding='utf-8') as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                out.append(r)
    return out


def load_meta(smoke):
    """샤드마다 meta 가 하나씩 나온다 ⟹ **합쳐서** 본다(⛔하나만 보면 나머지가 무검사다)."""
    pats = [META_SMOKE] if smoke else sorted(
        p for p in glob.glob(os.path.join(HERE, 'meta_t18*.json')) if 'SMOKE' not in p)
    metas = [json.load(open(p, encoding='utf-8')) for p in pats if os.path.exists(p)]
    if not metas:
        return None
    thr = {m.get('threads_readback') for m in metas}
    return dict(n_meta=len(metas), files=[os.path.basename(p) for p in pats],
                threads_readback=(1 if thr == {1} else sorted(thr, key=str)),
                md5_stable=all(m.get('md5_stable') for m in metas),
                git_head=sorted({m.get('git_head') for m in metas}),
                ckpt_sha256=sorted({m.get('ckpt_sha256') for m in metas}),
                modes=sorted({m.get('mode') for m in metas}))


def med3_layer(rows, layer, seeds):
    """층 안에서 `(arm, episode) -> median-of-3`. ⛔불완전 셀은 **버리지 않고** 돌려준다."""
    sub = [r for r in rows if r.get('layer') == layer and 'error' not in r]
    return G.med3(sub, seeds=seeds)


def contrast_rows(m, episodes):
    """네 칸이 **전부 있는** 에피소드에서 `d_i`·`r_i` 를 만든다. ⛔한 칸이라도 없으면 뺀다."""
    out = []
    for e in episodes:
        y = {a: m.get((a, e)) for a in G.CELL_ARMS}
        if any(v is None or not (v > 0) for v in y.values()):
            continue
        out.append(dict(episode=e,
                        d=G.interaction_contrast(y['A-Q'], y['A-C'], y['L-Q'], y['L-C']),
                        ln_r=math.log(y['A-Q']) - math.log(y['A-C']), **y))
    return out


def sign_split(vals, tie_abs=G.TIE_ABS):
    """`(W₊, n_tied, n_eff)` — 동점(`|d| < tie_abs`)은 빼고 검정한다(사전등록 §3.1)."""
    tied = [v for v in vals if abs(v) < tie_abs]
    kept = [v for v in vals if abs(v) >= tie_abs]
    return sum(1 for v in kept if v > 0), len(tied), len(kept)


def w_pair(m, a, b, episodes):
    """`W[a < b]` — 에피소드별 짝 비교(작을수록 좋다). 반환 `(w, n)`."""
    w = n = 0
    for e in episodes:
        x, y = m.get((a, e)), m.get((b, e))
        if x is None or y is None:
            continue
        n += 1
        w += int(x < y)
    return w, n


def w_vs_zero(m, a, zero, episodes):
    w = n = 0
    for e in episodes:
        x, z = m.get((a, e)), zero.get(e)
        if x is None or z is None:
            continue
        n += 1
        w += int(x < z)
    return w, n


# ─────────────────────────────────────────────────────────────────────────────
# 게이트
# ─────────────────────────────────────────────────────────────────────────────
def build_gates(rows, meta, mode, m_prim, eps_seen, zero_seen):
    """★사전등록 §6 의 게이트 25개. 반환 `[(이름, ok, 상세), …]`.

    ⛔사후 완화 금지. ⛔러너 리터럴 대조 금지 — 전부 **행의 되읽기 값**을 본다.
    ⚠스모크에서는 표본 모양 게이트 셋(`G-CORPUS`·`G-N`·`G-MESH`)의 **범위가 스모크 부분집합**
      으로 좁혀지고, 이름에 `[스모크 범위]` 를 박는다. ⛔조용히 완화하지 않는다.
    """
    A = []
    ok_rows = [r for r in rows if 'error' not in r]
    smoke = (mode == 'SMOKE')
    tag = ' [스모크 범위]' if smoke else ''
    corpus = G.B22.episodes()
    corpus_eps = [e['episode'] for e in corpus]
    scope_eps = corpus_eps[:G.SMOKE_N_EP] if smoke else corpus_eps

    # ── ★★G-ANCHOR-B22 / G-ANCHOR-B16 — 오늘의 패키지가 격자를 재현하는가 ──────
    def anchor(batch, pairs):
        fz = G.frozen_med3(batch)
        cen, bad = {'T1': 0, 'T2': 0}, []
        for arm, fa in pairs:
            for e in scope_eps:
                mine, theirs = m_prim.get((arm, e)), fz.get((fa, e))
                if mine is None:
                    continue                 # 아직 안 돈 셀 — `G-N` 이 잡는다
                g = G.anchor_grade(mine, theirs)
                if g is None:
                    bad.append(f'{arm}~{fa}:{e[:22]} {mine!r} vs {theirs!r}')
                else:
                    cen[g] += 1
        return cen, bad

    cen22, bad22 = anchor('B22', (('SHIP', 'SHIP'), ('L-C', 'LEARN'), ('A-C', 'ANA-N')))
    # ★ZERO 교차검증 — 앵커 행 집합의 일부다(같은 함수·같은 창·같은 에피소드).
    fz_zero, zbad = G.frozen_zero(), []
    for e, z in zero_seen.items():
        fzv = fz_zero.get(e)
        if fzv is None or fzv == 0:
            zbad.append(f'{e[:22]}:동결 ZERO 없음')
        elif abs(z / fzv - 1.0) > G.ZERO_RTOL:
            zbad.append(f'{e[:22]}:{z:.6f} vs {fzv:.6f}')
    A.append((f'★★G-ANCHOR-B22{tag} `SHIP`·`L-C`·`A-C` 가 `B-22` 동결 `SHIP`·`LEARN`·`ANA-N` 을 '
              f'에피소드 median-of-3 로 재현(T1=비트동일 · T2=상대차≤{G.ANCHOR_T2_RTOL:g}) '
              f'+ 참 제로필드 교차검증',
              not bad22 and not zbad and (cen22['T1'] + cen22['T2']) > 0,
              f"T1 {cen22['T1']} · T2 {cen22['T2']} · 실패 {bad22[:2]} · ZERO {zbad[:2]}"))
    cen16, bad16 = anchor('B16', (('SHIP', 'SHIP-NOW'), ('L-Q', 'KGRID1')))
    A.append((f'★★G-ANCHOR-B16{tag} `SHIP`·`L-Q` 가 `B-16` 동결 `SHIP-NOW`·`KGRID1` 을 재현',
              not bad16 and (cen16['T1'] + cen16['T2']) > 0,
              f"T1 {cen16['T1']} · T2 {cen16['T2']} · 실패 {bad16[:2]}"))

    # ── ★★G-LATTICE — 동결끼리의 교차 등식 (런 0회 · 행 없이도 판정 가능) ──────
    lat = G.lattice_check()
    A.append(('★★G-LATTICE 동결끼리 `B16 SHIP-NOW == B22 SHIP` (⛔거짓이면 2×2 격자 자체가 '
              '정합하지 않는다) + 부수로 `B22 LEARN == B22 SHIP`(주입==인라인)',
              lat['ship_now_vs_ship']['fail'] == 0 and lat['learn_vs_ship']['fail'] == 0
              and lat['n'] == G.N_EP, json.dumps(lat, ensure_ascii=False)))

    # ── ★★G-SNAPFIRE — 개입이 **먹었는가** ("설정을 넣었다 ≠ 설정이 먹었다") ────
    snap_bad, dk_dev = [], []
    for r in ok_rows:
        want = G.expected_dk(r['arm'])
        got, fired = _num(r.get('k_snap_dk')), bool(r.get('k_snap_fired'))
        if want is None:
            if fired or got is not None:
                snap_bad.append(f"{r['arm']}/{r['episode'][:18]} 연속인데 fired={fired} dk={got}")
        else:
            if not fired or got is None:
                snap_bad.append(f"{r['arm']}/{r['episode'][:18]} 스냅인데 fired={fired} dk={got}")
            else:
                dk_dev.append(abs(got - want))
                if abs(got - want) > 0:
                    snap_bad.append(f"{r['arm']}/{r['episode'][:18]} dk {got} != {want}")
            if r.get('k_snap_cells') is None:
                snap_bad.append(f"{r['arm']}/{r['episode'][:18]} cells 없음")
    A.append((f'★★G-SNAPFIRE 계수기 되읽기 — 스냅 팔 {list(G.QUANT_ARMS)} **전건 발화** · '
              f'연속 팔 {list(G.CONT_ARMS)} 전건 `None` · `dk` == `B16.DK1` '
              f'(⚠손 반올림 리터럴, `2π/240` 재도출 편차 {abs(G.DK1 - G.DK1_REDERIVE):.3g})',
              not snap_bad and len(ok_rows) > 0,
              f'위반 {snap_bad[:3]} · dk 편차 max {(max(dk_dev) if dk_dev else 0):.3g}'))

    # ── ★★G-ATOMSRC — 원자 출처 + **캐시 누출** 탐지 ──────────────────────────
    src_bad, md5_by_arm = [], {}
    for r in ok_rows:
        want = G.ARMS[r['arm']]['atoms']
        if r.get('atoms_source') != want:
            src_bad.append(f"{r['arm']} atoms_source={r.get('atoms_source')} != {want}")
        if want is None:
            if r.get('atoms_md5') is not None:
                src_bad.append(f"{r['arm']} 주입 없어야 하는데 md5 있음")
            if r.get('propose_calls') is not None and int(r['propose_calls']) < 1:
                src_bad.append(f"{r['arm']} 인라인인데 propose_calls=0")
        else:
            if r.get('atoms_md5') is None:
                src_bad.append(f"{r['arm']} md5 없음")
            md5_by_arm.setdefault(r['arm'], {})[(r['episode'], r['est_seed'])] = r['atoms_md5']
    leak = []
    for a, d in md5_by_arm.items():
        if len(set(d.values())) != len(d):
            leak.append(f'{a}: {len(d)}셀 → md5 {len(set(d.values()))}종 (캐시 누출)')
    A.append(('★★G-ATOMSRC `atoms_source` 가 팔 규약과 일치 · `SHIP` 은 주입 0 · ★주입 팔 안에서 '
              '`atoms_md5` 가 (에피소드,시드)마다 **전부 다르다**(`T11 v1 BLOCKER-2` 캐시 누출 탐지)',
              not src_bad and not leak and len(ok_rows) > 0,
              f'출처 {src_bad[:3]} · 누출 {leak}'))

    # ── ★★G-PAIRSRC — 열 안의 유일한 차이가 `k_snap_dk` 임을 되읽기로 증명 ─────
    pair_bad, pair_n = [], 0
    idx = {(r['arm'], r['episode'], r['est_seed']): r for r in ok_rows}
    for c_arm, q_arm in G.SNAP_PAIRS:
        for (a, e, s) in [k for k in idx if k[0] == c_arm]:
            q = idx.get((q_arm, e, s))
            if q is None:
                continue
            pair_n += 1
            if idx[(a, e, s)].get('atoms_md5') != q.get('atoms_md5'):
                pair_bad.append(f'{c_arm}~{q_arm}/{e[:18]}/s{s}')
    A.append(('★★G-PAIRSRC 같은 에피소드·시드에서 `A-C`↔`A-Q` · `L-C`↔`L-Q` 의 `atoms_md5` 가 '
              '**동일** ⟹ 열 안의 유일한 차이가 `k_snap_dk` 임을 되읽기로 증명',
              not pair_bad and pair_n > 0, f'짝 {pair_n} · 불일치 {pair_bad[:3]}'))

    # ── ★★G-FRAME — 다섯 팔이 **같은 채점 프레임**에서 채점됐는가 ──────────────
    fr_bad = []
    for e in eps_seen:
        sig = {}
        for r in ok_rows:
            if r['episode'] != e:
                continue
            sig.setdefault((r.get('n_frames_scored'), r.get('scored_t_first'),
                            r.get('scored_t_last')), set()).add(r['arm'])
        if len(sig) > 1:
            fr_bad.append(f'{e[:22]}: {[(k, sorted(v)) for k, v in list(sig.items())[:2]]}')
    A.append(('★★G-FRAME 한 에피소드의 **다섯 팔 전부**가 `n_frames_scored`·`scored_t_first`·'
              '`scored_t_last` 동일 (⛔저장소의 옛 헤드라인을 죽인 1순위 사유가 프레임 불일치다)',
              not fr_bad and len(eps_seen) > 0, f'불일치 {fr_bad[:2]}'))

    # ── ★★G-PROV — 팔 정체는 **의도가 아니라 되읽기** ─────────────────────────
    prov_bad = []
    for r in ok_rows:
        p = r.get('arm_provenance')
        if not isinstance(p, dict):
            prov_bad.append(f"{r['arm']}/{r['episode'][:16]} provenance 없음")
            continue
        if p.get('resolved_init_method') != G.ROUTE_EXPECTED:
            prov_bad.append(f"{r['arm']} resolved={p.get('resolved_init_method')}")
        if p.get('real_domain_gate_fired'):
            prov_bad.append(f"{r['arm']} 실측 게이트 발화 (합성인데)")
        if not p.get('resolved_is_final'):
            prov_bad.append(f"{r['arm']} 해소 전 스냅샷")
        if p.get('is_bank_mmse'):
            prov_bad.append(f"{r['arm']} bank_mmse 로 갔다")
    A.append((f'★★G-PROV `upd.arm_provenance()` 전건 `resolved_init_method == '
              f'{G.ROUTE_EXPECTED!r}` · 실측 게이트 **미발화** · 해소 완료 '
              f'(⛔`W-12`/`D3` 재발 방지 · `E1` 빈 dict 사고 부류)',
              not prov_bad and len(ok_rows) > 0, f'{prov_bad[:3]}'))

    # ── ★G-DEPLOY (⚠**개정 A2**) — 세 조항. ⚠(b)·(c) 는 항진에 가깝고 (a) 가 진짜 자다 ──
    dep_bad = set()
    #  (a) ★진짜 트립와이어 — **패키지**의 배포 상수가 핀과 같은가(행을 읽지 않는다).
    #      `B-16` 이후 12일 · `B-22` 이후 9일이 지났고 저장소는 그 사이에 배포 상수를 바꾼
    #      전례가 있다(08-22 세 개). 이것이 깨지면 앵커도 깨진다.
    pkg = G.B22.deploy_constants()
    for k, v in G.DEPLOY_EXPECTED.items():
        if pkg.get(k) != v:
            dep_bad.add(f'(a) 패키지 {k}={pkg.get(k)!r} != 핀 {v!r}')
    #  (b) 행 되읽기가 **그 팔이 등록한** 값과 같은가 (⚠`init_method` 항은 항진)
    for r in ok_rows:
        d, want = r.get('deploy_readback'), G.arm_deploy_expected(r['arm'])
        if not isinstance(d, dict):
            dep_bad.add('(b) deploy_readback 없음')
            continue
        for k, v in want.items():
            if d.get(k) != v:
                dep_bad.add(f"(b) {r['arm']} {k}={d.get(k)!r} != 등록 {v!r}")
    #  (c) ★핀을 벗어나는 팔·키가 **등록된 이탈과 정확히 일치**하는가 — ⛔허용이 아니라 **요구**다.
    dev, dev_want = G.deploy_deviation_audit(), G.deploy_deviation_registered()
    if dev != dev_want:
        dep_bad.add(f'(c) 이탈 {dev} != 등록 {dev_want}')
    A.append(('★G-DEPLOY 배포 상수 9종 — (a) **패키지** 단일출처가 `DEPLOY_EXPECTED` 핀과 일치'
              '(★진짜 트립와이어) · (b) 행의 `upd.config` 되읽기가 **그 팔이 등록한** 값과 일치 · '
              '(c) ★핀을 벗어나는 팔·키가 **등록된 이탈과 정확히 같다** — 사전등록 §1.2 가 결과 '
              '전에 적은 `ANA-N` 의 `learned_energy_thresh = 0.0` 을 **요구**한다(⛔허용이 아니다). '
              '⚠(b) 의 `init_method` 항은 러너 강제값 되읽기라 **항진**이다(§6.4). '
              '⚠개정 사유 = 사전등록 §6.1 이 §1.2 의 등록된 이탈을 예외로 적지 않아 **자기모순**'
              '이었고 **스모크가 그것을 잡았다**(`AMENDMENT_2_PREDATA.md`)',
              not dep_bad and len(ok_rows) > 0,
              f'{sorted(dep_bad)[:4]} · 이탈 {dev}'))

    # ── ★G-DEPTH — `B-2` 수심 사건 부류 ───────────────────────────────────────
    dep_by_ep, d_bad = {}, []
    for r in ok_rows:
        d = _num(r.get('depth'))
        if d is None or not math.isfinite(d) or d <= 0:
            d_bad.append(f"{r['arm']}/{r['episode'][:16]} depth={r.get('depth')}")
        else:
            dep_by_ep.setdefault(r['episode'], set()).add(round(d, 9))
    split = {e: sorted(v) for e, v in dep_by_ep.items() if len(v) > 1}
    A.append(('★G-DEPTH `upd.depth`(= `loader.depth`) 되읽기가 유한·양수이고 **에피소드 안에서 '
              '팔 간 동일** (⛔`B-2` 수심 17→30 사건 부류: 값이 바뀌면 하류 판정이 뒤집혔다)',
              not d_bad and not split and len(dep_by_ep) > 0,
              f'이상 {d_bad[:2]} · 갈림 {list(split.items())[:2]} · '
              f'값 {sorted({x for v in dep_by_ep.values() for x in v})[:4]}'))

    # ── ★★G-MAXN — ⚠**개정됨**(`AMENDMENT_1_PREDATA.md`): 역할이 **둘**이다 ────
    cap_bad, caps = [], {'pool': set(), 'grow': set(), 'alias': set()}
    pin = G.DEPLOY_EXPECTED['max_n']
    for r in ok_rows:
        pc, gc, al = (r.get('pool_cap_readback'), r.get('growth_ceiling_readback'),
                      r.get('max_n_alias'))
        for nm, v in (('pool', pc), ('grow', gc), ('alias', al)):
            if v is None:
                cap_bad.append(f"{r['arm']} {nm} 되읽기 없음")
            else:
                caps[nm].add(v)
        if gc is not None and al is not None and gc != al:
            cap_bad.append(f"{r['arm']} 별칭 {al} != 성장천장 {gc}")
    for nm in ('pool', 'grow', 'alias'):
        if len(caps[nm]) > 1:
            cap_bad.append(f'{nm} 이 팔 간에 갈린다 {sorted(caps[nm])}')
        elif caps[nm] and next(iter(caps[nm])) != pin:
            cap_bad.append(f'{nm} {sorted(caps[nm])} != 배포 핀 {pin}')
    A.append((f'★★G-MAXN 다섯 팔이 **풀링 상한**(`resolve_pool_cap`)과 **성장 천장**'
              f'(`resolve_growth_ceiling`)을 **둘 다** 배포값 {pin} 으로 공유하고 `upd.max_n` '
              f'별칭이 후자와 일치 (⚠2026-09-05 `a6eb5ba` 가 한 키를 둘로 쪼갰다 ⟹ 사전등록의 '
              f'한-역할 판은 개정됐다. 갈리면 이 배치는 **성장과 교락된 3인자 설계**가 된다)',
              not cap_bad and len(ok_rows) > 0,
              f'{cap_bad[:3]} · pool {sorted(caps["pool"])} · grow {sorted(caps["grow"])}'))

    # ── ★G-CORPUS — ⛔손목록 금지 ─────────────────────────────────────────────
    want_eps = set(scope_eps)
    dupes = [k for k, v in _count([(r['arm'], r['episode'], r['est_seed'], r.get('layer'))
                                   for r in ok_rows]).items() if v > 1]
    # ★`B-16` 표본과 집합 동일 — 동결 행에서 **되읽어** 댄다(⛔손목록도 선언도 아니다)
    b16_eps = {r['episode'] for r in G.frozen_rows('B16')}
    b16_same = (b16_eps == set(corpus_eps))
    A.append((f'★G-CORPUS{tag} 표본을 `gates_b22.episodes()` 에서 **임포트**하고 실물 해소 · '
              f'중복 행 0 · `B-16` 동결 행의 에피소드 집합과 **동일**',
              set(eps_seen) == want_eps and not dupes and b16_same,
              f'행 에피소드 {len(eps_seen)} / 기대 {len(want_eps)} · '
              f'누락 {sorted(want_eps - set(eps_seen))[:2]} · 중복 {dupes[:2]} · '
              f'B16 집합 동일 {b16_same}'))

    # ── ★G-MESH — ⛔"HOS M=3 비선형 참값" 단일 서술 금지 ───────────────────────
    #   ★행의 `mesh_verdict`(러너가 HDF5 에서 되읽은 값)를 **분석기가 HDF5 에서 다시 센 값**과
    #     대조하고, 본 판정에서는 `MESH_EXPECTED` **핀**과도 댄다(⛔러너 리터럴 왕복 아님).
    census = _count([v for v in {r['episode']: r.get('mesh_verdict')
                                 for r in ok_rows}.values() if v])
    want_mesh = _count(_mesh_of([e for e in corpus if e['episode'] in want_eps]))
    pin_ok = True if smoke else (want_mesh == G.MESH_EXPECTED)
    A.append((f'★G-MESH{tag} 행의 `mesh_verdict` 집계 == HDF5 재집계 (본 판정에서는 '
              f'`gates_b22.MESH_EXPECTED` 핀과도 일치) — ⚠`DEGRADED` 셀은 사실상 2차 '
              f'비선형이다. ⛔표본을 *"HOS M=3 비선형 참값"* 으로 **단일 서술하지 마라**',
              census == want_mesh and pin_ok and bool(census),
              f'행 {census} vs HDF5 {want_mesh} vs 핀 {G.MESH_EXPECTED} (핀대조 {pin_ok})'))

    # ── ★G-SIBLING — 형제 동결 행과의 교집합 ─────────────────────────────────
    alias, sib_keys = {}, set()
    for a in G.ARM_ORDER:
        for b, fa in G.ARMS[a]['anchors']:
            alias.setdefault(a, []).append((b, fa))
    for b in ('B16', 'B22'):
        for r in G.frozen_rows(b):
            sib_keys.add(G.sibling_key(r['episode'], (b, r['arm']), r['est_seed'],
                                       r.get('k_snap_dk'), r.get('atoms_md5')))
    expected_hit, unexpected, aq_hit = 0, [], 0
    for r in ok_rows:
        for b, fa in alias.get(r['arm'], []):
            k = G.sibling_key(r['episode'], (b, fa), r['est_seed'],
                              r.get('k_snap_dk'), r.get('atoms_md5'))
            expected_hit += int(k in sib_keys)
        if not alias.get(r['arm']):
            for b in ('B16', 'B22'):
                for fa in {x['arm'] for x in G.frozen_rows(b)}:
                    k = G.sibling_key(r['episode'], (b, fa), r['est_seed'],
                                      r.get('k_snap_dk'), r.get('atoms_md5'))
                    if k in sib_keys:
                        aq_hit += 1
                        unexpected.append(f"{r['arm']}~{b}/{fa}/{r['episode'][:16]}")
    A.append(('★G-SIBLING `(episode, arm→앵커별칭, est_seed, k_snap_dk, atoms_md5)` 교집합 — '
              '★앵커 팔은 **의도적 재현**이라 예외 목록에 명시하고, ★`A-Q` 는 **전건 신규**여야 한다',
              aq_hit == 0, f'앵커 별칭 적중 {expected_hit} · ⛔비앵커 충돌 {unexpected[:3]}'))

    # ── G-FINITE ─────────────────────────────────────────────────────────────
    fin_bad = [f"{r['arm']}/{r['episode'][:16]}" for r in ok_rows
               if not (_num(r.get('avg_nRMSE')) is not None
                       and math.isfinite(_num(r['avg_nRMSE'])) and _num(r['avg_nRMSE']) > 0
                       and _num(r.get('zero_nRMSE')) is not None
                       and math.isfinite(_num(r['zero_nRMSE'])) and _num(r['zero_nRMSE']) > 0)]
    err = [r for r in rows if 'error' in r]
    A.append(('G-FINITE 전 행 `avg_nRMSE`·`zero_nRMSE` 가 유한·양수 (⛔`NaN` 은 조용히 패배가 '
              '되고, `d_i` 는 로그라 0 이면 정의되지 않는다) · 예외 행 0',
              not fin_bad and not err and len(ok_rows) > 0,
              f'이상 {fin_bad[:3]} · 예외행 {len(err)}'))

    # ── G-N ──────────────────────────────────────────────────────────────────
    cnt = _count([(r.get('layer'), r['arm']) for r in ok_rows])
    if smoke:
        want_cnt = {(G.LAYER_PRIMARY, a): G.SMOKE_N_EP * len(G.EST_SEEDS)
                    for a in G.ARM_ORDER}
    else:
        want_cnt = {(G.LAYER_PRIMARY, a): G.N_EP * len(G.EST_SEEDS) for a in G.ARM_ORDER}
        want_cnt.update({(G.LAYER_SECC, a): G.SECC_N_EP * len(G.SECC_SEEDS)
                         for a in G.CELL_ARMS})
    n_desc = (f'스모크 {len(G.ARM_ORDER)}팔×{G.SMOKE_N_EP}×{len(G.EST_SEEDS)}='
              f'{sum(want_cnt.values())}' if smoke else
              f'PRIMARY {len(G.ARM_ORDER)}×{G.N_EP}×{len(G.EST_SEEDS)}={G.N_RUNS_PRIMARY} · '
              f'`SEC-C` {len(G.CELL_ARMS)}×{G.SECC_N_EP}×{len(G.SECC_SEEDS)}='
              f'{G.N_RUNS_SECC} · 합 {G.N_RUNS_TOTAL}')
    A.append((f'G-N{tag} 팔·층별 행 수 = {n_desc}',
              cnt == want_cnt, f'{ {f"{k[0]}/{k[1]}": v for k, v in sorted(cnt.items())} }'))

    # ── G-ENV — 측정면/분석면 분리 · 동결 대조 ────────────────────────────────
    env_bad = []
    seeds_seen = {r.get('layer'): sorted({x['est_seed'] for x in ok_rows
                                          if x.get('layer') == r.get('layer')})
                  for r in ok_rows}
    if seeds_seen.get(G.LAYER_PRIMARY) not in (None, list(G.EST_SEEDS)):
        env_bad.append(f"PRIMARY 시드 {seeds_seen.get(G.LAYER_PRIMARY)} != {list(G.EST_SEEDS)}")
    if seeds_seen.get(G.LAYER_SECC) not in (None, list(G.SECC_SEEDS)):
        env_bad.append(f"SEC-C 시드 {seeds_seen.get(G.LAYER_SECC)} != {list(G.SECC_SEEDS)}")
    if isinstance(meta, dict):
        if meta.get('threads_readback') != 1:
            env_bad.append(f"threads_readback={meta.get('threads_readback')}")
        if not meta.get('md5_stable', False):
            env_bad.append('배치 중 md5 가 움직였다')
    else:
        env_bad.append('meta 없음')
    frozen_bad, frozen_n = [], 0
    for path, side in ((FROZEN, '측정면'), (FROZEN_A, '분석면')):
        if not os.path.exists(path):
            frozen_bad.append(f'{side} FROZEN 없음')
            continue
        for line in open(path, encoding='utf-8'):
            line = line.strip()
            if not line or line.startswith('#') or ' ' not in line:
                continue
            h, rel = line.split(None, 1)
            if h.startswith('sha256:') or rel.startswith('sha256:'):
                continue
            p = _resolve(rel)
            if p is None:
                frozen_bad.append(f'{rel} 미해결')
                continue
            frozen_n += 1
            if _md5(p) != h:
                frozen_bad.append(f'{rel} md5 불일치')
    ck = {r.get('ckpt_sha256') for r in ok_rows if r.get('ckpt_sha256')}
    if len(ck) > 1:
        env_bad.append(f'체크포인트 지문이 갈린다 {sorted(ck)}')
    A.append((f'G-ENV 스레드 되읽기 1 · 층별 추정기 시드 집합 · 체크포인트 `sha256` 단일 · '
              f'`FROZEN*.txt` {frozen_n}항 대조 (★측정면/분석면 **분리**)',
              not env_bad and not frozen_bad,
              f'{env_bad[:3]} · 동결 {frozen_bad[:3]} · ckpt {sorted(ck)[:1]}'))

    # ── 기록 전용 6 (⛔막지 않는다) ───────────────────────────────────────────
    dose = {}
    for a in G.QUANT_ARMS:
        v = [(_num(r.get('k_snap_shift_median')), _num(r.get('k_snap_shift_max')),
              r.get('k_snap_cells'), r.get('k_snap_n_atoms'))
             for r in ok_rows if r['arm'] == a]
        v = [x for x in v if x[0] is not None]
        if v:
            dose[a] = dict(n=len(v), shift_median=round(st.median([x[0] for x in v]), 6),
                           shift_max=round(max(x[1] for x in v if x[1] is not None), 6),
                           cells_median=st.median([x[2] for x in v if x[2] is not None]),
                           atoms_median=st.median([x[3] for x in v if x[3] is not None]))
    A.append(('G-DOSE [기록 전용] 팔별 `k_snap` 이동량·셀 수 = `SEC-DOSE` 원자료. '
              '⚠**두 열의 개입 용량이 같다는 보장이 없다**(사전등록 §9.3-4) — 크게 갈리면 '
              '「교호작용」의 일부는 **용량 차이**이고 그때 라벨은 `EXPLORATORY` 다',
              True, json.dumps(dose, ensure_ascii=False)))
    nsel = {a: dict(median=st.median(v), sat=sum(1 for x in v if x >= G.DEPLOY_EXPECTED['max_n']),
                    n=len(v))
            for a in G.ARM_ORDER
            for v in [[r['auto_n_selected'] for r in ok_rows if r['arm'] == a
                       and r.get('auto_n_selected') is not None]] if v}
    A.append(('G-NSEL [기록 전용] `auto_n_selected` 팔별 분포 = §1.2 **개수 일치**의 되읽기 '
              '(⛔`ANA` 를 칸으로 썼다면 여기서 6배 차이가 났을 것이다)',
              True, json.dumps(nsel, ensure_ascii=False)))
    vp = {a: dict(fired=sum(1 for r in ok_rows if r['arm'] == a and r.get('varpro_fired')),
                  n=sum(1 for r in ok_rows if r['arm'] == a)) for a in G.ARM_ORDER}
    A.append(('G-VARPRO [기록 전용] `varpro_fired` 네 칸 전건 True 기대 · 목적함수 전/후',
              True, json.dumps(vp, ensure_ascii=False)))
    ceil = G.ceil_by_episode()
    z_rel = {a: [m_prim[(a, e)] / zero_seen[e] for e in eps_seen
                 if (a, e) in m_prim and e in zero_seen] for a in G.ARM_ORDER}
    n_over = sum(1 for e in eps_seen if ('A-Q', e) in m_prim and e in zero_seen
                 and e in ceil and m_prim[('A-Q', e)] / zero_seen[e] > ceil[e])
    ceil_seen = [ceil[e] for e in eps_seen if e in ceil]
    A.append(('G-CEIL [기록 전용] `z_i`·`CEIL_i`·`n_over_ceiling` = 축 II 원자료 '
              '(★**병기 의무 대상** — 초가법 라벨은 `n_over_ceiling` 없이 인용 금지)',
              True,
              'z 중앙 ' + json.dumps({a: round(st.median(vv), 4)
                                     for a, vv in z_rel.items() if vv})
              + ' · CEIL 중앙 '
              + (f'{st.median(ceil_seen):.4f}' if ceil_seen else 'n/a')
              + f' · n_over_ceiling {n_over}'))
    wall = {a: round(st.median([r['wall_s'] for r in ok_rows if r['arm'] == a
                                and r.get('wall_s') is not None]), 2)
            for a in G.ARM_ORDER
            if any(r['arm'] == a and r.get('wall_s') is not None for r in ok_rows)}
    A.append(('G-WALL [기록 전용] 팔별 `wall_s` 중앙 — ⛔**팔 간 비교 금지**'
              '(`torch.set_num_threads(1)` 은 BLAS 를 제약하지 않는다)',
              True, json.dumps(wall, ensure_ascii=False)))
    grow = {a: dict(calls=sum(r.get('grow_calls', 0) or 0 for r in ok_rows if r['arm'] == a),
                    branch=sum(r.get('grow_branch_entries', 0) or 0
                               for r in ok_rows if r['arm'] == a),
                    passed=sum(r.get('grow_threshold_pass', 0) or 0
                               for r in ok_rows if r['arm'] == a),
                    atoms=sum(r.get('n_grow_atoms', 0) or 0 for r in ok_rows if r['arm'] == a))
            for a in G.ARM_ORDER}
    A.append(('G-GROW [기록 전용] ★**성장 분기 진입 횟수**를 센다 — ⛔호출 횟수가 아니다 '
              '(`sequential_update.py:1675-1681` 이 축자로 그렇게 요구했다: *"A per-call counter '
              'does NOT detect that … Count the BRANCH."*). ⚠팔 간 진입률이 크게 갈리면 '
              '「교호작용」의 일부가 **성장**이다 ⟹ 라벨을 `EXPLORATORY` 로 강등',
              True, json.dumps(grow, ensure_ascii=False)))
    return A


def _count(seq):
    out = {}
    for x in seq:
        out[x] = out.get(x, 0) + 1
    return out


def _mesh_of(eps):
    """에피소드의 `mesh_verdict` 를 **HDF5 에서 직접** 읽는다(행이 아니라 원본)."""
    import h5py
    out = []
    for e in eps:
        with h5py.File(e['path'], 'r') as h:
            v = h['config'].attrs.get(G.B22.MESH_ATTR)
        out.append(v.decode() if isinstance(v, bytes) else str(v))
    return out


def _resolve(rel):
    """`FROZEN*.txt` 의 경로를 **배치 → tmp → 저장소** 순으로 푼다."""
    for base in (HERE, os.path.dirname(HERE), G.ROOT):
        p = os.path.join(base, rel)
        if os.path.exists(p):
            return p
    return None


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--out', default=None,
                    help='verdict 을 쓸 디렉터리 (⛔스모크는 배치 **밖**에 쓴다)')
    args = ap.parse_args()
    mode = 'SMOKE' if args.smoke else 'MAIN'
    rows = load_rows(ROWS_SMOKE if args.smoke else ROWS_MAIN)
    meta = load_meta(args.smoke)
    outdir = args.out if args.out else HERE
    os.makedirs(outdir, exist_ok=True)

    ok_rows = [r for r in rows if 'error' not in r]
    eps_seen = sorted({r['episode'] for r in ok_rows})
    zero_seen = {}
    for r in ok_rows:
        z = _num(r.get('zero_nRMSE'))
        if z is not None:
            zero_seen.setdefault(r['episode'], z)

    m_prim, bad_prim = med3_layer(rows, G.LAYER_PRIMARY, G.EST_SEEDS)
    m_secc, bad_secc = med3_layer(rows, G.LAYER_SECC, G.SECC_SEEDS)
    A = build_gates(rows, meta, mode, m_prim, eps_seen, zero_seen)

    # ── PRIMARY (축 I) ───────────────────────────────────────────────────────
    cr = contrast_rows(m_prim, eps_seen)
    d = [x['d'] for x in cr]
    w_plus, n_tied, n_eff = sign_split(d)
    kstar = G.k_two(n_eff) if n_eff > 0 else None
    # ── 축 II (결정적 통계량 · 검정 아님) ────────────────────────────────────
    ceil = G.ceil_by_episode()
    ln_r = [x['ln_r'] for x in cr]
    z_i = [m_prim[('A-Q', x['episode'])] / zero_seen[x['episode']] for x in cr
           if x['episode'] in zero_seen]
    ceil_i = [ceil[x['episode']] for x in cr if x['episode'] in ceil]
    med_ln_r = st.median(ln_r) if ln_r else float('nan')
    med_z = st.median(z_i) if z_i else float('nan')
    med_ceil = st.median(ceil_i) if ceil_i else float('nan')
    inert = G.t_inert(med_ln_r) if ln_r else False
    ceil_hit = G.t_ceil(med_z, med_ceil) if z_i and ceil_i else False
    n_over_ceiling = sum(1 for x in cr if x['episode'] in ceil and x['episode'] in zero_seen
                         and m_prim[('A-Q', x['episode'])] / zero_seen[x['episode']]
                         > ceil[x['episode']])
    # ── 양성대조 · SEC ───────────────────────────────────────────────────────
    w_ctrl, n_ctrl = w_vs_zero(m_prim, G.CTRL_ARM, zero_seen, eps_seen)
    w_a, n_a = w_pair(m_prim, 'A-C', 'A-Q', eps_seen)      # SEC-A: 사전-OFF 열의 양자화 효과
    w_b, n_b = w_pair(m_prim, 'L-Q', 'A-Q', eps_seen)      # SEC-B: 양자화 열의 사전 효과
    w_ac0, n_ac0 = w_vs_zero(m_prim, 'A-C', zero_seen, eps_seen)
    # SEC-IDENT — ⚠**항진**(로그의 항등식). 유용한 목적은 부호 뒤집힘 세기 하나뿐이다.
    n_ident_flip = 0
    for x in cr:
        alt = math.log(x['A-Q'] * x['L-C'] / (x['A-C'] * x['L-Q']))
        n_ident_flip += int((alt > 0) != (x['d'] > 0))
    # SEC-C — 신선 시드 45·46·47 · 앞 16
    secc_eps = sorted({e for (_a, e) in m_secc})
    cr_c = contrast_rows(m_secc, secc_eps)
    w_c, tie_c, neff_c = sign_split([x['d'] for x in cr_c])
    sign_agree = (None if not cr or not cr_c else
                  ((w_plus * 2 >= n_eff) == (w_c * 2 >= neff_c)))
    strat = {}
    for key in ('bf', 'ncomp', 'mesh_verdict'):
        by = {}
        for x in cr:
            v = next((r.get(key) for r in ok_rows if r['episode'] == x['episode']), None)
            by.setdefault(v, []).append(x['d'])
        strat[key] = {str(k): dict(n=len(v), w_plus=sum(1 for z in v if z > 0))
                      for k, v in sorted(by.items(), key=lambda t: str(t[0]))}

    anchor_ok = all(o for n, o, _ in A if n.replace('★', '').startswith(('G-ANCHOR', 'G-LATTICE')))
    lab_raw = G.label(anchor_ok, w_ctrl, w_plus, n_eff, inert, ceil_hit)
    blocked = G.blocked_by(A)
    all_ok = all(bool(o) for _n, o, _d in A)

    # ── G-COV · G-BLOCK (마지막 — 자기 자신을 포함해 자가진단한다) ────────────
    # ⚠`G-COV`·`G-BLOCK` 은 **아직 append 되기 전**이다(자기 자신을 검사하는 게이트라
    #   마지막에 붙는다) ⟹ 이름을 명시적으로 더한다. ⛔빼면 자기 자신을 「미구현」으로 신고한다.
    emitted = [n.replace('★', '').replace('⚠', '').replace('⛔', '').strip().split()[0]
               for n, _o, _d in A] + ['G-COV', 'G-BLOCK']
    cov_bad = []
    missing = sorted(set(G.GATE_NAMES) - set(emitted))
    extra = sorted(set(emitted) - set(G.GATE_NAMES) - {'G-COV', 'G-BLOCK'})
    if missing:
        cov_bad.append(f'등록됐으나 분석기가 안 낸 게이트 {missing}')
    if extra:
        cov_bad.append(f'등록 안 된 게이트 이름 {extra}')
    try:
        G.verify_registered()
    except AssertionError as e:                                    # noqa: BLE001
        cov_bad.append(f'등록 상수 드리프트: {e}')
    disclosures = _disclosures(mode, w_a, n_a, w_b, n_b, w_c, neff_c, sign_agree,
                              med_ln_r, med_z, med_ceil, n_over_ceiling, w_ac0, n_ac0,
                              lab_raw, bad_prim, bad_secc)
    # ★병기 의무 자가검사 (사전등록 §3.4) — ⛔기계로 넣는다
    if not any('SEC-A' in s and 'SEC-B' in s for s in disclosures):
        cov_bad.append('축 I 진술에 SEC-A/SEC-B 병기 없음')
    if lab_raw.startswith('SUBADDITIVE') and not any('median ln r' in s for s in disclosures):
        cov_bad.append('아가법 라벨에 §3.2 원자료 병기 없음')
    if lab_raw == 'COMBINATION-SUPERADDITIVE' and not any('n_over_ceiling' in s
                                                          for s in disclosures):
        cov_bad.append('초가법 라벨에 n_over_ceiling 병기 없음')
    if ('DETECTED' in lab_raw or 'UNDER-POWERED' in lab_raw) and \
            not any('q80' in s for s in disclosures):
        cov_bad.append('음성 라벨에 MDE 병기 없음')
    A.append(('★G-COV 사전등록 §6 게이트 표 ↔ 구현 **1:1** · 등록 상수 드리프트 0 · '
              '★§3.4 **병기 의무** 자가검사(축 I 진술에 `SEC-A`/`SEC-B` · 아가법에 축 II 원자료 · '
              '초가법에 `n_over_ceiling` · 음성에 MDE) · 게이트가 자료를 봤다',
              not cov_bad and len(ok_rows) > 0, f'{cov_bad if cov_bad else "이상 없음"} · '
              f'rows={len(ok_rows)}'))
    missed, folded = G.selfcheck_blocked_by(emitted + list(G.GATE_NAMES))
    A.append(('★G-BLOCK `blocked_by()` 자가진단 — **분석기가 실제로 낸 이름 전건**(자기 자신 포함) '
              '+ 등록된 전 이름을 합성 위반 튜플로 넣어 빈 리스트가 아님을 확인 [긴 이름 우선] '
              '(⛔`T5-ENS` 의 *"항상 빈 리스트"* 재발 방지)',
              not missed and not folded, f'미포착 {missed} · 접힘 {folded}'))

    # ★라벨 인쇄 벨트 — ⛔차단이 하나라도 있거나 게이트가 하나라도 FAIL 이면 인쇄하지 않는다.
    blocked = G.blocked_by(A)
    all_ok = all(bool(o) for _n, o, _d in A)
    # ★인쇄 벨트 — ⛔`blocked` 에 **직접** 매달린 호출이어야 한다(`check_prereg C18`).
    lab = (G.label(anchor_ok, w_ctrl, w_plus, n_eff, inert, ceil_hit)
           if (not blocked and all_ok and mode == 'MAIN') else None)

    v = {
        'batch': 'T18-FACTORIAL', 'mode': mode,
        'kind': ('CONFIRMATORY (축 I) + 구조 판별(축 II) — ⚠**blind 아님**: 네 칸 중 셋의 '
                 '에피소드별 값이 설계자에게 열려 있고 이 배치의 상수 전부가 거기서 유도됐다. '
                 '미노출은 `A-Q` 열 하나뿐이다'),
        'n_rows': len(ok_rows), 'n_error_rows': len(rows) - len(ok_rows),
        'n_runs_expected': (G.N_RUNS_SMOKE if mode == 'SMOKE' else G.N_RUNS_TOTAL),
        'gates': [{'name': n, 'pass_': bool(o), 'detail': str(dd)} for n, o, dd in A],
        'gates_pass': f'{sum(1 for _n, o, _d in A if o)}/{len(A)}', 'blocked_by': blocked,
        'primary': {
            'stat': 'W+ = #{i : d_i > 0}, d_i = ln(y(A-Q)·y(L-C)/(y(A-C)·y(L-Q)))',
            'direction': '양측 (두 꼬리에 서로 다른 라벨)',
            'w_plus': w_plus, 'n': len(cr), 'n_tied': n_tied, 'n_eff': n_eff,
            'k_star': kstar, 'tie_abs': G.TIE_ABS,
            'p_two_sided': G.sign_p_two(w_plus, n_eff) if n_eff else None,
            'median_d': (round(st.median(d), 6) if d else None),
            'q80': G.MDE['I_q80']},
        'axis_II': {'median_ln_r': (None if not ln_r else round(med_ln_r, 6)),
                    'eq_band': round(G.EQ_BAND, 6), 'T_INERT': bool(inert),
                    'median_z': (None if not z_i else round(med_z, 6)),
                    'median_CEIL': (None if not ceil_i else round(med_ceil, 6)),
                    'T_CEIL': bool(ceil_hit), 'n_over_ceiling': n_over_ceiling},
        'positive_control': {'stat': f'W[{G.CTRL_ARM} < ZERO]', 'w': w_ctrl, 'n': n_ctrl,
                             'thresh': G.THRESH_CTRL,
                             'p': G.sign_p_one(w_ctrl, n_ctrl) if n_ctrl else None,
                             'pass': w_ctrl >= G.THRESH_CTRL},
        'SEC': {
            'SEC-A (사전-OFF 열의 양자화 효과, 무보정 단측)':
                {'stat': 'W[A-C < A-Q]', 'w': w_a, 'n': n_a, 'thresh': G.THRESH_SEC,
                 'p': G.sign_p_one(w_a, n_a) if n_a else None, 'q80': G.MDE['SEC_q80']},
            'SEC-B (양자화 열의 사전 효과, 무보정 단측)':
                {'stat': 'W[L-Q < A-Q]', 'w': w_b, 'n': n_b, 'thresh': G.THRESH_SEC,
                 'p': G.sign_p_one(w_b, n_b) if n_b else None, 'q80': G.MDE['SEC_q80']},
            'SEC-IDENT (⚠항진 — 로그의 항등식)': {'n_ident_flip': n_ident_flip},
            'SEC-C (신선 시드 45·46·47 · 앞 16, 무보정)':
                {'w_plus': w_c, 'n_eff': neff_c, 'n_tied': tie_c,
                 'k_star': G.k_two(neff_c) if neff_c else None,
                 'sign_agrees_with_axis_I': sign_agree, 'q80': G.MDE['SECC_q80']},
            'SEC-CEIL': {'median_z': (None if not z_i else round(med_z, 6)),
                         'median_CEIL': (None if not ceil_i else round(med_ceil, 6)),
                         'n_over_ceiling': n_over_ceiling},
            'SEC-DOSE': next((dd for n, _o, dd in A if n.startswith('G-DOSE')), None),
            'SEC-STRAT (⛔기술통계 — 층 n 이 작아 라벨을 만들지 않는다)': strat},
        'A-C_vs_ZERO (⚠사전-OFF 열은 무기술 열이다)': {'w': w_ac0, 'n': n_ac0},
        'thresholds': {'I': G.THRESH_I, 'SEC': G.THRESH_SEC, 'CTRL': G.THRESH_CTRL,
                       'SECC': G.THRESH_SECC,
                       'N_EFF_UNPOWERED': G.N_EFF_UNPOWERED},
        'mde': G.MDE, 'n_eff_table': {str(k): val for k, val in G.n_eff_table().items()},
        'design_constants': {k: (round(val, 6) if isinstance(val, float) else val)
                             for k, val in G.design_constants().items() if k != 'episodes'},
        'prediction': {'inputs': G.PRED_INPUTS, 'q_registered': G.q_registered(),
                       'label_probs': G.label_probs(),
                       'null_label_probs': G.null_label_probs(),
                       'detect_power_H_syn': G.detect_power_syn(),
                       'sensitivity': {str(k): val for k, val in G.sensitivity().items()},
                       'family_cost': G.family_cost_table()},
        'incomplete_cells': {'PRIMARY': bad_prim[:5], 'SEC-C': bad_secc[:5]},
        'label_raw': lab_raw, 'label': lab,
        'label_msg': lab or ('SMOKE — ⛔판정 아님' if mode == 'SMOKE' else 'BLOCKED'),
        'label_kind': G.LABEL_KIND.get(lab_raw),
        'disclosures': disclosures,
    }
    out_p = os.path.join(outdir, 'verdict_t18.json' if mode == 'MAIN'
                         else 'verdict_t18_SMOKE.json')
    json.dump(v, open(out_p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    print(f"T18-FACTORIAL [{mode}] :: {v['label_msg']}")
    print(f'  축 I  W₊ = {w_plus}/{n_eff} (동점 {n_tied}, k*={kstar}) '
          f'⟹ {lab_raw}   [MDE q80={G.MDE["I_q80"]}]')
    print(f'  ★병기 의무 — SEC-A(A-C<A-Q) {w_a}/{n_a} · SEC-B(L-Q<A-Q) {w_b}/{n_b} '
          f'(둘 다 무보정, q80={G.MDE["SEC_q80"]})')
    print(f'  축 II  median ln r = {med_ln_r:.6f} (밴드 ±{G.EQ_BAND:.6f}) T_INERT={inert} · '
          f'median z = {med_z:.4f} vs CEIL {med_ceil:.4f} T_CEIL={ceil_hit} · '
          f'n_over_ceiling={n_over_ceiling}')
    print(f'  양성대조 {G.CTRL_ARM}<ZERO {w_ctrl}/{n_ctrl} (문턱 {G.THRESH_CTRL}) · '
          f'⚠A-C<ZERO {w_ac0}/{n_ac0} = 사전-OFF 열은 **무기술 열**')
    print(f'  SEC-C {w_c}/{neff_c} 부호일치={sign_agree} (q80={G.MDE["SECC_q80"]}) · '
          f'SEC-IDENT flip={n_ident_flip}')
    print(f"  gates {v['gates_pass']} · blocked_by {blocked}")
    for n, o, dd in A:
        print(f"   [{'OK ' if o else 'FAIL'}] {n[:100]:100s} {str(dd)[:150]}")
    print(f'  ⟹ {out_p}')
    return 0 if (all_ok and not blocked) else 1


def _disclosures(mode, w_a, n_a, w_b, n_b, w_c, neff_c, sign_agree, med_ln_r, med_z,
                 med_ceil, n_over, w_ac0, n_ac0, lab, bad_prim, bad_secc):
    """⛔사전등록 §9.2(금지)·§9.3(공시)을 **기계로** 넣는다. ⚠사람이 지우면 `G-COV` 가 죽는다."""
    q80 = G.MDE
    return [
        f'⛔[§0.3-1 · §9.2-1] **외부 기준선 대비 우위를 재지 못한다** — 네 팔이 전부 우리 팔 안의 '
        f'절제이고 외부 기준선 행이 **0개**다. 어떤 결과도 *"우리가 남보다 낫다"* 로 읽지 마라. '
        f'이 배치가 내는 것은 **귀속**이다.',
        f'⛔[§0.3-2 · §9.2-5] **합성 혼잡 층 한정** — 희소·실측·비혼잡으로 외삽 금지.',
        f'★[§3.4] **축 I 의 모든 진술에 병기**: `SEC-A`(사전-OFF 열의 양자화 효과) '
        f'{w_a}/{n_a} · `SEC-B`(양자화 열의 사전 효과) {w_b}/{n_b} — ⛔둘 다 **무보정**이고 '
        f'⛔유의해도 확증으로 승격하지 마라(§9.2-6, 새 사전등록으로만).',
        f'★[§3.2 원자료] median ln r = {med_ln_r:.6f} (등가 밴드 ±{G.EQ_BAND:.6f}) · '
        f'median z = {med_z:.4f} · median CEIL = {med_ceil:.4f} · n_over_ceiling = {n_over}.',
        f'⛔[§9.2-9] **`d_i` 의 중앙값을 「효과 크기」로 단독 인용 금지** — 천장이 그것을 압축한다.',
        f'⛔[§0.3-3 · §3.6] ★**이 설계는 자기가 만들어진 질문에 답하지 못할 공산이 크다고 결과 '
        f'전에 스스로 적었다**: 가법 널이 43/48 에피소드에서 관측 범위 밖이라 등록 예측 아래 '
        f'`P(COMBINATION-SUPERADDITIVE)` = 인쇄 정밀도로 **0.000000** 이다. ⟹ 등록된 산출물은 '
        f'축 I 이 아니라 **축 II**(`A-Q` 의 착지)다. ⛔결과 뒤에 이 문단을 다시 쓰지 마라. '
        f'⚠단 **검정력이 없지는 않다** — 반대 가설(반쪽 synergy) 아래 검출확률 '
        f'{G.detect_power_syn():.4f}. ⛔이 둘을 바꿔 쓰지 마라.',
        f'⛔[§9.2-4] **아가법 결과를 「기제」로 읽지 마라** — §0.4 가 결과 전에 그것을 **척도**로 '
        f'예측했다. MDE 병기: 축 I q80={q80["I_q80"]} · SEC q80={q80["SEC_q80"]} · '
        f'`SEC-C` q80={q80["SECC_q80"]} (⛔`M8b` 선례 — **섞지 마라**).',
        f'⚠[§3.4 · §9.3-7] **`A-C` 는 참 제로필드를 이기지 못한다** — 이 배치 실측 {w_ac0}/{n_ac0} '
        f'⟹ 사전-OFF 열 전체가 **무기술 열**이다. 모든 진술에 병기한다.',
        f'⚠[§3.4] `SEC-C`(신선 시드 45·46·47 · 앞 16) {w_c}/{neff_c} · 축 I 과 부호일치 '
        f'{sign_agree} — ⛔부호가 갈리면 라벨은 `EXPLORATORY` 로 강등된다. ⛔독립 확증 아님.',
        f'⚠[§9.3-1] ★★**blind 아님, 그리고 그 정도가 크다.** 설계자는 네 칸 중 **셋의 '
        f'에피소드별 값을 전부 열었고** `MED_Q1`·`SD_Q1`·`MED_PR`·`RHO`·`CEIL` 이 거기서 '
        f'유도됐다 ⟹ ⛔*"out-of-sample"·"held-out"·"blind"* 자구 **금지**. 미노출은 `A-Q` 열 하나뿐.',
        f'⚠[§9.3-2] **등록 예측 자체가 노출의 산물이다** — ⛔*"독립 예측"* 금지. '
        f'⚠[§9.3-3] `ρ` = {G.PRED_INPUTS["RHO"]} 는 **미관측 대리값**(사전-ON 열의 상관을 '
        f'사전-OFF 열에 빌렸다) ⟹ 민감도를 4열로 적었다. ⛔사후 검정력 주장에 쓰지 마라.',
        f'★⚠[§9.3-4 · §11-2] **개입 용량이 두 열에서 같다는 보장이 없다** — `SEC-DOSE` 를 병기 '
        f'의무로 걸었다. ⛔크게 갈리면 「교호작용」의 일부는 **용량 차이**이고 라벨은 `EXPLORATORY` 다. '
        f'⚠그리고 스모크에서 그 값을 **먼저 봤다** — 그만큼 추가 노출이다(사전등록 §11-2 가 요구한 기재).',
        f'⚠[§9.3-5] `DK1 = {G.DK1}` 은 **손 반올림 리터럴**이다(`gates_b16` 에서 임포트, '
        f'`2π/240` 재도출 편차 {abs(G.DK1 - G.DK1_REDERIVE):.3g}). **등방 스냅**이라 y 축에서는 '
        f'기준선 격자보다 고운 격자를 준 셈 = **우리 팔에 유리**하다.',
        f'⚠[§9.3-6] **자국 코퍼스는 배포 체크포인트의 학습 봉투다** ⟹ 결과는 *"자국은 사전의 '
        f'학습 분포다"* 와 **분리되지 않는다**.',
        f'⚠[§9.2-2] ⛔**「사전」 인자를 단일 노브라고 쓰지 마라** — `ANA-N` 은 원자 출처 **와** '
        f'`learned_energy_thresh = 0` 의 묶음이다. ⛔*"제안원 하나만 다르다"* 는 개수가 어긋나는 '
        f'`ANA` 팔에서만 참이었다(§11-5: 둘 중 **덜 나쁜 교락**을 골랐다).',
        f'⛔[§9.2-3] **`B-16 −38.607%` 와 `B-22 −48.586%` 를 이 배치의 계수와 산술 결합하지 마라** '
        f'(`P8` 감사가 `\\BOneRel`/`\\BSixteenRel` 에서 같은 부류를 잡았다).',
        f'⛔[§9.2-7 · §9.2-8] 이 배치는 `B-16`·`B-22`·`B-20b` 의 판정을 **대체하지 않는다**'
        f'(교호작용을 재지 주변효과의 크기를 다시 재지 않는다) · **배포 권고 아님**(네 칸 중 '
        f'셋이 절제 팔이다). ⛔[§9.2-10] 총량 표현 금지.',
        f'⚠[§10] 이 배치는 **전부 합성**이다 — 라우터가 `auto_switch_thresh = 0` 때문에 항상 '
        f'`learned` 로 해소되고 실측 게이트는 발화하지 않는다(`G-PROV` 되읽기). ⛔어떤 결과도 '
        f'실측 팔(`bank_mmse`)에 대한 진술이 아니고, ⛔라우팅 동작을 고치지 않는다(`W-12` 살아 있다).',
        f'⚠[§11-6 · G-GROW] **성장 분기가 팔마다 다른 횟수로 발화할 수 있다** — `max_n` 은 팔 간 '
        f'동일하지만(`G-MAXN`) 시작 원자 수와 잔차가 다르면 진입 횟수가 갈린다. ⛔갈리면 '
        f'「교호작용」의 일부가 **성장**이다.',
        f'⚠[개정 A1] **`G-MAXN` 은 사전등록 판보다 넓다** — 2026-09-05 `a6eb5ba` 가 `max_n` 을 '
        f'`pool_cap`(풀링 상한)과 `growth_ceiling`(성장 천장) **두 이름으로 쪼갰다**. 사전등록 '
        f'§6.1 은 한 역할만 되읽었다 ⟹ **결과 0행 시점의 사전 개정**으로 둘 다 되읽는다. '
        f'⛔이 개정은 §5 의 결정(제3인자를 넣지 않는다)을 바꾸지 않는다.',
        f'⚠[§9.3-8] 추정기 시드는 PRIMARY 층에서 **median-of-3 한 벌**뿐이다. 불완전 셀: '
        f'PRIMARY {len(bad_prim)} · SEC-C {len(bad_secc)}.',
        f'⚠[§9.3-9] `n_over_ceiling` 이 큰 초가법 결과는 「관측된 적 없는 나쁨」에 의존한다 — '
        f'그때는 ⛔*"보수적"* 이라는 §0.4 의 논증을 **인용하지 마라**.',
        f'⚠[§6.4] **항진으로 표기한 검사 둘**: `SEC-IDENT`(로그의 항등식 — ⛔증거 아님, '
        f'쓸모는 부호 뒤집힘 세기 하나) · `G-DEPLOY` 의 `init_method` 항(러너 강제값 되읽기).',
        f'⚠[§13] 이 배치는 **7자리 시드 밑변을 선언하지 않는다**(추정기 시드는 두 자리 리터럴 '
        f'여섯 개뿐) ⟹ `check_seed_bands.py` 의 대역 원장에 나타나지 않는다. ⛔그것은 '
        f'*"검사됐다"* 가 아니라 *"검사 대상이 아니다"* 라는 뜻이고, 진짜 방어는 `G-SIBLING` 이다.',
    ] + ([f'⛔[§7] **이것은 스모크다** — `results_t18_SMOKE.jsonl` 은 본 판정에 넣지 않는다. '
          f'라벨은 인쇄하지 않고, `G-CORPUS`·`G-N`·`G-MESH` 는 **스모크 부분집합으로 범위가 '
          f'좁혀졌다**(이름에 표기). ⛔조용한 완화가 아니다.'] if mode == 'SMOKE' else [])


if __name__ == '__main__':
    sys.exit(main())
