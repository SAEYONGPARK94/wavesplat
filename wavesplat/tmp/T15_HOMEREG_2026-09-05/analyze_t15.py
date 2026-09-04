"""`T15-HOMEREG` 분석기 — ⛔**러너보다 먼저** 쓰인다. 이 파일이 **행 스키마를 정의**한다.

## 러너가 채워야 할 행 (⛔여기서 정의하고, 러너는 따르기만 한다)
```
layer('crowded'|'sparse'|'SMOKE') · arm · block · fi · bf · ncomp · ep_seed · path_md5
nrmse            ★ZERO 행은 산술(`√mean(gt²)/Hs` 채점창 평균) — 추정기 런 0회
seed · wall_s
n_frames_scored · scored_t_first · scored_t_last · episode_total_frames   (G-FRAME)
n_init_frames · init_t_last · tp_true · hs_true · depth_readback · t_split (G-NOLEAK/G-DEPTH)
has_gt · auto_selected · auto_n_selected · auto_real_gate · ensemble_k · n_gaussians
varpro_fired · varpro_k_shift · init_iters · avg_rmse_near
knobs_before · knobs_after        ★살아있는 모듈 상태 10키 (G-KNOB/G-RESTORE)
proposal_md5 · proposal_device · n_gathered · cap_proposal · n_raw · n_sel
n_grow_calls · n_grow_atoms · rng_md5_before · rng_md5_after
zero_nrmse       ★모든 행에 같은 값 — 분석기가 h5 에서 **독립 재계산**해 대조한다 (G-ZERO)
```
⛔`(x or 기본값)` 자구 금지 — `0.0` 은 falsy 다(`T14` 사건).
"""
import hashlib, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import consts_t15 as C                                            # noqa: E402
import gates_t15 as G                                             # noqa: E402

# ★러너는 **두 프로세스**로 돈다(전층 4팔 / 부분층 5팔). ⛔행을 손으로 잇지 않는다 —
#   분석기가 **양쪽 파일을 직접** 읽고 `G-ENV` 가 **양쪽 meta** 를 대조한다
#   (`T12 AMENDMENT_2 §2`: SEC-R 112행이 손 이어붙이기라 측정면 검증 밖에 있었다).
SHARDS = (('results_t15_FULL.jsonl', 'meta_t15_FULL.json'),
          ('results_t15_SUB.jsonl', 'meta_t15_SUB.json'))
FROZEN = os.path.join(HERE, 'FROZEN_md5.txt')
FROZEN_A = os.path.join(HERE, 'FROZEN_md5_ANALYSIS.txt')
LAYERS = ('crowded', 'sparse')
ALL_ARMS = ('ZERO',) + tuple(C.EST_ARMS)


def _frozen(path):
    if not os.path.exists(path):
        return None
    out = {}
    for line in open(path, encoding='utf-8'):
        if line.strip() and not line.startswith('#'):
            h, k = line.split(None, 1)
            out[k.strip()] = h.strip()
    return out


def zero_nrmse_h5(path):
    """★`ZERO` 를 **분석기가 독립 재계산**한다 — 러너 코드를 한 줄도 공유하지 않는다.

    정본 = `tmp/S16_BASELINE_2026-08-14/run_s16.py:240,266`
    `zero_nRMSE = mean_{t > 2·Tp} [ sqrt(mean(gt²)) / Hs ]`,
    채점창이 `run.py:272 get_tracking_frames(tp_factor=2.0)` 와 **같은 부등호**(`>`)다.
    """
    import h5py
    with h5py.File(path, 'r') as f:
        hs = float(f['gt_params'].attrs['Hs_true'])
        tp = float(f['gt_params'].attrs['Tp_true'])
        ts = np.asarray(f['frames/timestamps'])
        sel = ts > C.TP_FACTOR * tp
        gt = np.asarray(f['frames/gt_wave_field'][sel])
    per = np.sqrt((gt ** 2).reshape(gt.shape[0], -1).mean(axis=1)) / max(hs, 1e-6)
    return float(per.mean()), int(sel.sum()), hs, tp


def _sib_rows():
    """형제 배치의 동결 결과행 (⛔읽기만 한다)."""
    out = []
    for rel in C.SIBLINGS:
        p = os.path.join(C.ROOT, 'tmp', rel)
        if not os.path.exists(p):
            continue
        for line in open(p, encoding='utf-8'):
            if line.strip():
                d = json.loads(line)
                out.append((rel, d))
    return out


def main():
    rows, metas = [], {}
    for rf, mf in SHARDS:
        rp, mp = os.path.join(HERE, rf), os.path.join(HERE, mf)
        rows += [json.loads(x) for x in open(rp, encoding='utf-8') if x.strip()]
        metas[mf] = json.load(open(mp, encoding='utf-8'))
    meta = metas[SHARDS[0][1]]
    P = [r for r in rows if r.get('layer') in LAYERS]
    by = {L: {a: {r['block']: r for r in P if r['layer'] == L and r['arm'] == a}
              for a in ALL_ARMS} for L in LAYERS}
    corp = C.corpus()
    sub = {L: {n for n, _ in v} for L, v in C.sub_layer().items()}
    dep = C.deployed_knobs()

    def wins(d, ref):
        cm = sorted(set(d) & set(ref))
        return sum(1 for b in cm if d[b]['nrmse'] < ref[b]['nrmse']), len(cm)

    # PRIMARY = 층별 F2 < SHIP (해침 다리) · 양성대조 = 층별 SHIP < ZERO
    W = {L: wins(by[L]['F2'], by[L]['SHIP']) for L in LAYERS}
    W_ctrl = {L: wins(by[L]['SHIP'], by[L]['ZERO']) for L in LAYERS}
    W_arm = {L: {a: wins(by[L][a], by[L]['SHIP']) for a in C.EST_ARMS if a != 'SHIP'}
             for L in LAYERS}

    def rel(a, L, blocks=None):
        s = by[L]['SHIP']
        ks = sorted(set(by[L][a]) & set(s)) if blocks is None \
            else sorted(set(by[L][a]) & set(s) & set(blocks))
        v = [(by[L][a][b]['nrmse'] - s[b]['nrmse']) / s[b]['nrmse']
             for b in ks if s[b]['nrmse'] > 0]
        return v

    floor = {L: float(np.median(np.abs(rel('FLOOR-VPLR', L)))) if by[L]['FLOOR-VPLR']
             else None for L in LAYERS}
    effect = {L: float(np.median(rel('F2', L))) if by[L]['F2'] else None for L in LAYERS}
    within = {L: (floor[L] is not None and effect[L] is not None
                  and abs(effect[L]) <= floor[L]) for L in LAYERS}

    A = []
    # ── 처치 정체 ──────────────────────────────────────────────────────────
    tp_ok, mine, theirs = C.treat_pin()
    A.append(('★★G-TREATPIN f=2 처치값을 `consts_t12.TREAT["A-MAPSCALE"]` 에서 **임포트**해 '
              'assert 로 핀 — ⛔러너는 그 리터럴을 다시 적지 않는다(손복사 0)',
              tp_ok, f'mine {mine} · T12 {theirs}'))

    kbad, kdet = [], {}

    def _ne(a, b):
        if a is None or b is None:
            return a is not b
        if isinstance(a, str) or isinstance(b, str):
            return a != b
        return abs(float(a) - float(b)) > 1e-12
    for r in rows:
        kb = r.get('knobs_before')
        if kb is None:
            kbad.append(f"{r['arm']}:{r['block'][:14]}:knobs_before=None"); continue
        diff = frozenset(k for k in C.KNOB_KEYS if _ne(kb.get(k), dep.get(k)))
        kdet.setdefault(r['arm'], set()).update(diff)
        if diff != C.EXPECT_DIFF.get(r['arm'], frozenset()):
            kbad.append(f"{r['arm']}:{r['block'][:14]}:{sorted(diff)}")
    A.append(('★★G-KNOB 팔 설정 직후 **살아있는 모듈 상태** 10키 되읽기 — 정확히 그 팔의 노브만 다르다',
              not kbad, f'위반 {len(kbad)}' + (f' 예 {kbad[:2]}' if kbad else '')))
    rbad = [f"{r['arm']}:{r['block'][:14]}" for r in rows
            if r.get('knobs_after') is None
            or any(_ne((r.get('knobs_after') or {}).get(k), dep.get(k))
                   for k in C.KNOB_KEYS)]
    A.append(('★★G-RESTORE 팔 종료 후 10키가 배포값으로 복귀 (⛔프로세스 전역 오염 차단)',
              not rbad, f'위반 {len(rbad)}'))

    # ── 프레임 정합 (★저장소가 헤드라인을 잃은 그 사유) ───────────────────
    fbad = []
    for L in LAYERS:
        for b in sorted(by[L]['SHIP']):
            got = [(a, by[L][a][b]) for a in ALL_ARMS if b in by[L][a]]
            key = {(r['n_frames_scored'], r['scored_t_first'], r['scored_t_last'])
                   for _, r in got}
            if len(key) != 1:
                fbad.append(f'{L}:{b[:18]}:{len(key)}종')
    A.append(('★★G-FRAME 네 팔 전부 동일 채점 프레임 집합 (n_frames_scored·첫·끝 t 일치)',
              not fbad and bool(P), f'불일치 블록 {len(fbad)}' + (f' 예 {fbad[:2]}' if fbad else '')))

    # ── 형제 배치와의 독립 (★T13 사고의 재발 방지) ────────────────────────
    sib = _sib_rows()
    my_seeds = {r['seed'] for r in rows if r.get('seed') is not None}
    sib_seeds = {d['seed'] for _, d in sib if d.get('seed') is not None}
    my_trip = {(r['block'], r.get('seed'), r.get('proposal_md5')) for r in rows
               if r.get('proposal_md5')}
    sib_trip = {(d.get('block'), d.get('seed'), d.get('proposal_md5')) for _, d in sib
                if d.get('proposal_md5')}
    sib_vals = {round(float(d['nrmse']), 12) for _, d in sib
                if isinstance(d.get('nrmse'), (int, float)) and np.isfinite(d['nrmse'])}
    my_vals = {round(float(r['nrmse']), 12) for r in rows
               if isinstance(r.get('nrmse'), (int, float)) and np.isfinite(r['nrmse'])}
    # ⛔`my_seeds <= {SEED15+i}` 는 **넣지 않는다** — 러너가 그 식으로 쓴 값을 분석기가 같은
    #   식과 대조하는 것이라 **실패할 수 없다**(`check_prereg C6a`). 시드가 실제로 무엇이었는지는
    #   `G-SEED` 가 **적용 시점 되읽기 + 독립 재계산**으로 답한다. 여기서는 형제와의 **교집합**만 건다.
    A.append(('★★G-SIBLING 시드 서로소 · (block, seed, proposal_md5) 교집합 0 · **nRMSE 값 비트일치 0** '
              f'[형제 {len(sib)}행 · ⛔`T13` 이 `T12` 와 24행 비트 동일이면서 "독립" 이라 적은 사고]',
              not (my_seeds & sib_seeds) and not (my_trip & sib_trip)
              and not (my_vals & sib_vals) and bool(sib),
              f'시드교집합 {len(my_seeds & sib_seeds)} · 삼중교집합 {len(my_trip & sib_trip)} · '
              f'값비트일치 {len(my_vals & sib_vals)} · 관측 대역 '
              f'{min(my_seeds) if my_seeds else "-"}–{max(my_seeds) if my_seeds else "-"} '
              f'[기술]'))

    # ── 표본 ───────────────────────────────────────────────────────────────
    names = {L: [n for n, _ in corp[L]] for L in LAYERS}
    paths = [p for L in LAYERS for _, p in corp[L]]
    dup = len(paths) != len(set(paths))
    got = {L: set(by[L]['SHIP']) for L in LAYERS}
    A.append(('★G-CORPUS 128 을 `W15/corpus.json` 에서 읽고 실물 해소 128/128 · 중복 0 · 행과 일치 '
              '(⛔손목록 금지)',
              all(len(names[L]) == C.N_LAYER for L in LAYERS) and not dup
              and all(os.path.exists(p) for p in paths)
              and all(got[L] == set(names[L]) for L in LAYERS),
              f'혼잡 {len(names["crowded"])} · 희소 {len(names["sparse"])} · 중복 {dup} · '
              f'행 { {L: len(got[L]) for L in LAYERS} }'))
    nbad = {}
    for L in LAYERS:
        for a in ALL_ARMS:
            want = C.N_LAYER if a in ('ZERO',) + C.FULL_ARMS else C.N_SUB // 2
            if len(by[L][a]) != want:
                nbad[f'{L}/{a}'] = f'{len(by[L][a])}≠{want}'
        for a in C.SUB_ARMS:
            if set(by[L][a]) - sub[L]:
                nbad[f'{L}/{a}/subset'] = len(set(by[L][a]) - sub[L])
    A.append((f'★G-N 전층 팔 층당 {C.N_LAYER}행 · 스윕 팔 층당 {C.N_SUB // 2}행(부분층 안) · '
              f'합 {C.N_ROWS} (⚠문턱 23/41 은 n={C.N_LAYER} **전용**이다)',
              not nbad and len(P) == C.N_ROWS,
              f'행 {len(P)}/{C.N_ROWS} · 위반 {nbad or "없음"}'))
    nf = [r['block'] for r in rows
          if r.get('nrmse') is None or not np.isfinite(float(r['nrmse']))]
    A.append(('★G-FINITE 모든 행의 nRMSE 가 유한 (⛔NaN 은 조용히 패배가 된다)',
              not nf, f'비유한 {len(nf)}'))

    # ── 양성대조 (⛔차단이 아니라 **제외**한다 — PREREG §8 의 자기 서술) ──
    ctrl_ok = G.judged_layers(W_ctrl)
    A.append(('★G-CTRL 층별 양성대조 SHIP<ZERO 를 **양층 전건 계산** (실패 층은 판정에서 제외하고 인쇄)',
              all(W_ctrl[L][1] == C.N_LAYER for L in LAYERS),
              ' · '.join(f'{L} {W_ctrl[L][0]}/{W_ctrl[L][1]}'
                         f'{"" if L in ctrl_ok else " ⛔미복제→제외"}' for L in LAYERS)))

    # ── 잡음 바닥이 실제로 발화했는가 ──────────────────────────────────────
    fl_diff = kdet.get('FLOOR-VPLR', set())
    fl_same = sum(1 for L in LAYERS for b in by[L]['FLOOR-VPLR']
                  if b in by[L]['SHIP']
                  and abs(by[L]['FLOOR-VPLR'][b]['nrmse'] - by[L]['SHIP'][b]['nrmse']) < 1e-12)
    fl_n = sum(len(by[L]['FLOOR-VPLR']) for L in LAYERS)
    A.append(('★G-FLOOR FLOOR-VPLR 이 `vp_lr` **하나만** 다르고 no-op 이 아니다 '
              '(⛔기본인자 바인딩 때문에 순진한 개입은 조용히 no-op 이 된다)',
              fl_diff == frozenset({'vp_lr'}) and fl_n > 0 and fl_same < fl_n,
              f'측정된 차이 {sorted(fl_diff)} · SHIP 과 비트동일 {fl_same}/{fl_n}'))

    # ── 팔 정체 · 누출 ─────────────────────────────────────────────────────
    k_dep = C.deployed_ensemble_k()
    abad = [f"{r['arm']}:{r['block'][:14]}" for r in rows if r['arm'] != 'ZERO'
            and (r.get('auto_selected') != 'learned' or r.get('auto_real_gate') is not False
                 or r.get('ensemble_k') != k_dep or r.get('n_gaussians') is None)]
    A.append((f'★G-ARM resolved==learned · 실측게이트 **미발화** · ensemble_k=={k_dep} [되읽기·임포트]',
              not abad, f'위반 {len(abad)}'))
    lbad = [f"{r['arm']}:{r['block'][:14]}" for r in rows
            if r.get('init_t_last') is None or r.get('scored_t_first') is None
            or r.get('t_split') is None
            or not (r['init_t_last'] <= r['t_split'] < r['scored_t_first'])]
    A.append(('★G-NOLEAK 초기창(t≤2·Tp)과 채점창(t>2·Tp)이 **행에서** 서로소 · 제안 캐시 사전주입 0',
              not lbad and not any(m.get('cfg_had_learned_atoms') for m in metas.values()),
              f'위반 {len(lbad)} · 캐시역류 '
              f'{ {k: m.get("cfg_had_learned_atoms") for k, m in metas.items()} }'))
    pbad = []
    for L in LAYERS:
        for a in C.EST_ARMS:
            fps = [r.get('proposal_md5') for r in by[L][a].values()]
            if not fps or not all(fps) or len(set(fps)) != len(fps):
                pbad.append(f'{L}/{a}')
    A.append(('★G-ATOMS 팔·층마다 제안 원자 지문이 전건 상이 · 결측 0 (config 캐시 누출 탐지)',
              not pbad, f'위반 {pbad or "없음"}'))
    cap = C.deployed_cap()
    # ⛔`n_gathered <= cap` 은 **거짓 전제**다 — `learned_init.py:143` 이 캡을 **부분표집**으로
    #   집행하므로 수집량은 캡보다 클 수 있고, 그것이 정상이다(`T12` 공시의 예산 비대칭).
    #   그래서 여기서 거는 것은 ①캡이 **임포트값과 같다** ②수집이 **결측이 아니다** 셋뿐이다.
    ng = [r['n_gathered'] for r in rows if r.get('n_gathered') is not None]
    cbad = [f"{r['arm']}" for r in rows if r['arm'] != 'ZERO'
            and (r.get('cap_proposal') != cap or r.get('n_gathered') is None)]
    A.append((f'★G-CAP 관측 캡을 배포 모듈에서 **임포트**({cap}) · n_gathered 결측 0 (⛔손복사 0)',
              not cbad and bool(ng),
              f'위반 {len(cbad)} · n_gathered 중앙 {float(np.median(ng)) if ng else "-"} '
              f'· 캡 발화 {sum(1 for v in ng if v > cap)}/{len(ng)}'))

    # ── 수심 · 제로 ────────────────────────────────────────────────────────
    dbad = [r['block'] for r in rows if r.get('depth_readback') is None
            or abs(float(r['depth_readback']) - C.DEPTH_HOME) > 1e-9
            or _ne((r.get('knobs_before') or {}).get('MAP_DEPTH'), C.DEPTH_HOME)]
    A.append((f'★G-DEPTH 자국 수심 되읽기 == {C.DEPTH_HOME} · 조건화 맵 DEPTH 도 같다 '
              '(⛔`B-2` 수심 사건 부류)', not dbad, f'위반 {len(dbad)}'))
    zbad, zchk = [], 0
    for L in LAYERS:
        want = dict(corp[L])
        for b, r in by[L]['ZERO'].items():
            z, nsel, hs, tp = zero_nrmse_h5(want[b]); zchk += 1
            if (abs(z - float(r['nrmse'])) > 1e-9 or nsel != r['n_frames_scored']
                    or abs(hs - float(r['hs_true'])) > 1e-9
                    or abs(tp - float(r['tp_true'])) > 1e-9):
                zbad.append(f'{L}:{b[:18]}')
    zx = [f"{r['arm']}:{r['block'][:14]}" for L in LAYERS for a in ALL_ARMS
          for b, r in by[L][a].items()
          if b in by[L]['ZERO']
          and abs(float(r['zero_nrmse']) - float(by[L]['ZERO'][b]['nrmse'])) > 1e-12]
    A.append(('★★G-ZERO 분석기가 h5 에서 **독립 재계산**한 제로필드가 행과 일치 '
              '(√mean(gt²)/Hs · 채점창 t>2·Tp) · **두 샤드**의 zero_nrmse 도 일치',
              not zbad and not zx and zchk > 0,
              f'대조 {zchk} · 불일치 {len(zbad)} · 샤드간 {len(zx)}'))

    # ── 환경 · 동결 ────────────────────────────────────────────────────────
    now = C.freeze_map(); rec = meta.get('code_md5')
    rec = rec if isinstance(rec, dict) else {}
    # ⛔`m['seed'] == C.SEED15` 도 넣지 않는다 — 러너가 `C.SEED15` 를 적고 분석기가 같은 상수와
    #   대조하는 자기대조다(`check_prereg C6a`). 시드는 `G-SEED` 가 **적용 시점**에서 답한다.
    meta_agree = all(m.get('code_md5') == rec and m.get('threads_readback') == 1
                     and m.get('md5_stable') is True for m in metas.values())
    fz0, fza = _frozen(FROZEN), _frozen(FROZEN_A)
    run_keys = sorted(k for k in now if k not in C.ANALYSIS_SIDE)
    ana_keys = sorted(k for k in now if k in C.ANALYSIS_SIDE)
    d_meta = [k for k in run_keys if now[k] != rec.get(k)]
    d_fz0 = None if fz0 is None else [k for k in run_keys if now[k] != fz0.get(k)]
    d_fza = None if fza is None else sorted(k for k in set(ana_keys) | set(fza)
                                            if now.get(k) != fza.get(k))
    A.append(('★G-ENV 측정면 3중 대조(현재==meta==결과0행동결) · 분석면 동결 · **두 샤드 meta 합치** '
              '· 스레드[되읽기]==1  (⛔시드는 여기서 안 센다 — `G-SEED` 몫)',
              bool(rec) and not d_meta and d_fz0 == [] and d_fza == []
              and meta_agree,
              f'측정면 {len(run_keys)}키 · 런대비 {len(d_meta)} · 0행동결대비 '
              f'{"N/A" if d_fz0 is None else len(d_fz0)} · 분석면 '
              f'{"N/A" if d_fza is None else len(d_fza)} · 샤드meta합치 {meta_agree}'))

    def _seed_fp(seed):
        st0 = np.random.get_state()
        try:
            np.random.seed(int(seed))
            st = np.random.get_state()
            return hashlib.md5(np.asarray(st[1]).tobytes() + str(st[2]).encode()).hexdigest()
        finally:
            np.random.set_state(st0)
    sdbad = [f"{r['arm']}:{r['block'][:14]}" for r in rows if r['arm'] != 'ZERO'
             and (r.get('seed_applied') != r.get('seed')
                  or r.get('rng_md5_seeded') != _seed_fp(r['seed']))]
    A.append(('★★G-SEED 추정기에 **적용된** 시드를 `run.set_seed` 훅에서 되읽고, 그 직후 전역 RNG '
              '지문을 분석기가 **독립 재계산**해 대조 (⛔러너 리터럴 vs 분석기 리터럴은 실패할 수 '
              '없는 게이트다)', not sdbad,
              f'위반 {len(sdbad)}/{sum(1 for r in rows if r["arm"] != "ZERO")}'))

    probe = G.blocked_by([('★★G-NOLEAK 합성', False, ''), ('★G-N 합성', False, ''),
                          ('★★G-RESTORE 합성', False, ''), ('★G-TREATPIN 합성', False, ''),
                          ('★G-ENV 합성', True, '')])
    A.append(('G-BLOCK blocked_by 자가진단 (긴 이름 우선)',
              probe == ['G-NOLEAK', 'G-N', 'G-RESTORE', 'G-TREATPIN'], f'합성 → {probe}'))
    A.append(('G-COV 게이트가 자료를 봤다', len(rows) > 0 and len(P) > 0,
              f'rows={len(rows)} · PRIMARY={len(P)}'))

    blocked = G.blocked_by(A)
    all_ok = all(bool(o) for _, o, _ in A)          # ★인쇄 벨트 (PREREG §8)
    ok_layers = G.judged_layers(W_ctrl)
    lab = G.label(W, W_ctrl, within) if (not blocked and all_ok) else None

    # ── 기술통계 (⛔라벨을 만들지 않는다) ──────────────────────────────────
    curve = {L: {a: {'w': W_arm[L][a][0], 'n': W_arm[L][a][1],
                     'p_hurt': G.sign_p_hurt(*W_arm[L][a]),
                     'median_rel': (float(np.median(rel(a, L))) if by[L][a] else None)}
                 for a in C.EST_ARMS if a != 'SHIP'} for L in LAYERS}
    glob_w = wins({b: r for L in LAYERS for b, r in by[L]['F2'].items()},
                  {b: r for L in LAYERS for b, r in by[L]['SHIP'].items()})
    cells = {}
    for L in LAYERS:
        for b, r in by[L]['F2'].items():
            k = f"{L}/{r['bf']}/{r['ncomp']}"
            s = by[L]['SHIP'].get(b)
            if not s:
                continue
            cells.setdefault(k, [0, 0])
            cells[k][1] += 1
            if r['nrmse'] < s['nrmse']:
                cells[k][0] += 1
    med = {L: {a: float(np.median([r['nrmse'] for r in by[L][a].values()]))
               for a in ALL_ARMS if by[L][a]} for L in LAYERS}

    v = {'batch': 'T15-HOMEREG', 'kind': 'CONFIRMATORY',
         'grade': 'CONFIRMATORY (⚠노출된 코퍼스 — PREREG §3·§7.3: 사전의 **학습 봉투**다)',
         'prereg': 'PREREG.md (결과 0행·러너 0줄 시점 동결)',
         'n_rows': len(rows), 'n_primary': len(P),
         'gates': [{'name': nm, 'pass_': bool(o), 'detail': d} for nm, o, d in A],
         'gates_pass': f'{sum(1 for _, o, _ in A if o)}/{len(A)}', 'blocked_by': blocked,
         'primary': {L: {'stat': 'F2 < SHIP', 'w': W[L][0], 'n': W[L][1],
                         'p_hurt_one_sided': G.sign_p_hurt(*W[L]),
                         'thresh_hurt': C.THRESH_HURT, 'thresh_help': C.THRESH_HELP,
                         'holm_alpha': 0.025} for L in LAYERS},
         'control': {L: {'stat': 'SHIP < ZERO', 'w': W_ctrl[L][0], 'n': W_ctrl[L][1],
                         'replicated': L in ctrl_ok} for L in LAYERS},
         'layers_judged': ok_layers,
         'equivalence': {L: {'floor_median_abs_rel': floor[L],
                             'effect_median_rel': effect[L],
                             'within_floor': within[L]} for L in LAYERS},
         'floor_moved': {L: (floor[L] is not None and floor[L] > 0.0) for L in LAYERS},
         'curve': curve,
         'global_128_descriptive': {'w': glob_w[0], 'n': glob_w[1]},
         'cells_descriptive': {k: {'w': x[0], 'n': x[1]} for k, x in sorted(cells.items())},
         'medians': med,
         'knob_diffs_measured': {a: sorted(s) for a, s in sorted(kdet.items())},
         'mde': C.MDE, 'prediction': {'q': C.PRED_Q, 'label_probs': G.label_probs(),
                                      'null_label_probs': G.label_probs(0.5)},
         'proposal_device': sorted({r.get('proposal_device') for r in rows
                                    if r.get('proposal_device')}),
         'label': lab, 'label_msg': lab or 'BLOCKED',
         'disclosures': [
             '⛔`f` 는 **노브 스케일 인자**다 — `T11`·`T14` 의 `s`(자료 재라벨)와 같은 기호로 '
             '쓰지 마라. 저건 좌표를 옮겼고 이건 상수를 옮긴다',
             '⛔**어떤 팔도 배포 구성이 아니다** — 이겨도 배포 권고가 아니다(배포는 별도 사전등록)',
             '⚠★**자국 코퍼스는 배포 체크포인트의 학습 봉투다**(`gen_mixed.py:78` 학습 장면 앵커 == '
             '`pipeline.py:35 SEA_STATE`, `nc∈{1,2,5,15,30}` 동일) ⟹ ⛔*"out-of-sample"·'
             '"held-out"·"blind"* 자구 금지. 결과는 *"자국은 사전의 학습 분포다"* 와 분리되지 않는다',
             '⛔**전역 128 수치를 헤드라인으로 쓰지 마라**(상시지시) — 층별로만. '
             '이 verdict 의 `global_128_descriptive` 는 기술통계다',
             f'⛔음성 라벨(`CHANGE-UNDETECTED`·`NO-HARM-*`)을 *"안전하다/무해하다"* 로 읽는 것을 '
             f'금지한다 — **검정력의 진술**이다. MDE 병기 의무: q80={C.MDE["q80"]} · '
             f'기각 불가 대역 W={C.MDE["unrejectable_w"]}',
             '⛔`T12` 의 회수율 ρ 나 `T14` 의 R 과 병치 금지 — 정의가 다르다',
             '⚠형제 배치와의 서로소는 **시드만으로 증명되지 않는다** — 시드 대역 '
             '(T11/T12 20260905–928 · `T13`-FRESH 20261005–032 · T14 20260915–948 vs '
             '이 배치 20261101–228)에 더해 `G-SIBLING` 이 (block, seed, proposal_md5) 교집합과 '
             '**nRMSE 값 비트일치**까지 0 으로 확인한다. ⛔`T13` 이 `T12` 와 24행 비트 동일이면서 '
             '*"독립"* 이라 적었다가 철회한 사고의 재발 방지다',
             '⚠**스윕은 64 부분층에서만** 잰다 — 곡선을 128 에 외삽하지 마라',
             '⛔`wall_s` 를 팔 간 비교에 쓰지 마라 · `proposal_device` 는 config 가 정하지 않는다',
             '⚠`FLOOR-VPLR` 은 `T12` 에서 측정된 널이다 — 이 코퍼스에서 다시 널이라는 보장은 없다. '
             '`floor_moved` 를 함께 읽어라',
             '⚠셀 8개(n=16)는 **기술통계 전용** — MDE q<=0.0975 로 사실상 검정력이 없다. ⛔셀 라벨 금지',
             '⚠`ensemble_collapsed=True`(요청 3 vs 실효 1)는 **자국 학습 경로의 정상값**이고 결함이 '
             '아니다(`sequential_update.py:2090-2126 _effective_k`)',
             '⚠자국 nRMSE 는 `Hs` 로 나눈다 — **실측 하네스는 σ 로 나눈다**(비 3.99–4.06). '
             '⛔실측 nRMSE 와 직접 비교 금지(`metrics.py:33-46` 축자 경고)',
             '⛔`G-CTRL` 은 차단이 아니라 **제외**다(PREREG §8 의 자기 서술) — 양층 다 미복제면 '
             '라벨은 `ANCHOR-DID-NOT-REPLICATE` 이고 그때 다른 라벨을 인쇄하지 않는다',
         ]}
    json.dump(v, open(os.path.join(HERE, 'verdict_t15.json'), 'w'),
              ensure_ascii=False, indent=1)

    print(f"T15-HOMEREG :: {v['label_msg']}   (판정 층 {ok_layers or '없음'})")
    for L in LAYERS:
        print(f"  PRIMARY[{L:7s}] F2<SHIP {W[L][0]:2d}/{W[L][1]:2d} "
              f"p_hurt={G.sign_p_hurt(*W[L]):.4g}  (해침<={C.THRESH_HURT} 이김>={C.THRESH_HELP})"
              f"   양성대조 SHIP<ZERO {W_ctrl[L][0]:2d}/{W_ctrl[L][1]:2d}"
              f"{'' if L in ctrl_ok else '  ⛔미복제'}")
    for L in LAYERS:
        print(f"  등가[{L:7s}] 바닥 {floor[L]} · 효과 {effect[L]} · 바닥안 {within[L]}")
    print('  비용 곡선 (해침 승수 · 중앙 상대차):')
    for L in LAYERS:
        print(f"    {L:7s} " + ' · '.join(
            f"{a} {curve[L][a]['w']}/{curve[L][a]['n']}"
            f"({curve[L][a]['median_rel']:+.4f})" if curve[L][a]['median_rel'] is not None
            else f"{a} -" for a in C.EST_ARMS if a != 'SHIP'))
    print(f"  중앙 nRMSE {med}")
    print(f"  측정된 노브 차이 {v['knob_diffs_measured']}")
    print(f"  전역 128 [기술통계] {glob_w[0]}/{glob_w[1]}  · 제안 device {v['proposal_device']}")
    print(f"  gates {v['gates_pass']}  blocked_by={blocked or '없음'}")
    for nm, o, d in A:
        print(f"   [{'OK ' if o else 'FAIL'}] {nm:78s} {d}")


if __name__ == '__main__':
    main()
