"""`T12-ISOLATE` 분석기 — ⛔**러너보다 먼저** 쓰인다. 이 파일이 **행 스키마를 정의**한다.

## 러너가 채워야 할 행 (여기서 정의한다)
  layer('PRIMARY'|'SECR'|'SMOKE') · arm · block(★아카이브 전체 경로) · angle · hs
  nrmse · sigma4 · frame_index · seed · wall_s · n_offered · n_frames · dt · tq · e_shape
  t_origin · gt_std · gt_value · gt_std_after
  knobs_before · knobs_after   (★살아있는 모듈/객체 상태 10키 — G-KNOB/G-RESTORE)
  proposal_md5 · proposal_device · n_gathered · cap_proposal
  n_raw · n_sel · n_atoms_installed · n_grow_calls · n_grow_atoms · suppress_radius
  vp_lr · vp_k_shift · vp_clamp_hits
  rng_md5_before · rng_md5_after      (★G-RNGSTREAM)
  prov(arm_provenance 통째) · rescale_s · x_scale · t_scale · disp_relerr
  kp_true · kmed_raw · kmed_sel · kmed_final · ratio    (★§8.2 매개열, 추가 런 0회)
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import consts_t12 as C                                            # noqa: E402
import gates_t12 as G                                             # noqa: E402

ROWS = os.path.join(HERE, 'results_t12.jsonl'); META = os.path.join(HERE, 'meta_t12.json')
FROZEN = os.path.join(HERE, 'FROZEN_md5.txt')
FROZEN_A = os.path.join(HERE, 'FROZEN_md5_ANALYSIS.txt')
OUTDIR = HERE            # ★verdict 출력만 분리(스모크 하네스가 HERE 를 안 흔들게)


def _frozen(path):
    if not os.path.exists(path):
        return None
    out = {}
    for line in open(path, encoding='utf-8'):
        if line.strip() and not line.startswith('#'):
            h, k = line.split(None, 1)
            out[k.strip()] = h.strip()
    return out


def main():
    rows = [json.loads(x) for x in open(ROWS, encoding='utf-8') if x.strip()]
    meta = json.load(open(META, encoding='utf-8'))
    P = [r for r in rows if r.get('layer') == 'PRIMARY']
    R = [r for r in rows if r.get('layer') == 'SECR']
    by = {a: {r['block']: r for r in P if r['arm'] == a} for a in C.ARMS}
    byr = {a: {r['block']: r for r in R if r['arm'] == a} for a in C.SECR_ARMS}

    def wins(d, zero):
        cm = sorted(set(d) & set(zero))
        return sum(1 for b in cm if d[b]['nrmse'] < zero[b]['nrmse']), len(cm)

    W = {a: wins(by[a], by['ZERO']) for a in C.ARMS if a != 'ZERO'}
    Wr = {a: wins(byr[a], byr['ZERO']) for a in C.SECR_ARMS if a != 'ZERO'}
    w_pri, n_pri = W.get(C.PRIMARY_ARM, (0, 0))
    sec = {a: W.get(a, (0, 0)) for a in C.SEC8}
    w_secmax = max([v[0] for v in sec.values()] or [0])

    dep = C.deployed()
    A = []

    # ── 표본·층 ────────────────────────────────────────────────────────────
    sz = os.path.getsize(C.ZIP) if os.path.exists(C.ZIP) else -1
    A.append((f'★G-SIZE 분석기가 직접 잰 크기 == {C.WANT_BYTES}', sz == C.WANT_BYTES, str(sz)))
    want = set(C.sample()); got = {r['block'] for r in P}
    exposed, mats = C.exposed_blocks()
    A.append(('★G-SAMPLE 표본==행 · 아카이브 실존 · 사전감사 스모크 교집합 0 [namelist 직접]',
              got == want and len(want) == C.N_PRIMARY and not (want - set(mats))
              and not (want & set(exposed))
              and C.file_md5(C.SAMPLE_JSON) == C.SAMPLE_MD5_PIN,
              f'표본 {len(want)} · 행 {len(got)} · 노출교집합 {len(want & set(exposed))} · '
              f'md5핀 {C.file_md5(C.SAMPLE_JSON) == C.SAMPLE_MD5_PIN}'))
    hold = set(C.heldout_blocks()); gotr = {r['block'] for r in R}
    A.append(('★G-HELDOUT 여집합 28 · 표본/노출과 교집합 0 [분석기가 namelist 에서 계산]',
              len(hold) == C.N_SECR and not (hold & want) and not (hold & set(exposed))
              and (gotr == hold if R else True),
              f'held-out {len(hold)} · 행 {len(gotr)} · 표본교집합 {len(hold & want)}'))
    per = {a: len(by[a]) for a in C.ARMS}
    perr = {a: len(byr[a]) for a in C.SECR_ARMS}
    A.append((f'★G-N PRIMARY 팔별 {C.N_PRIMARY}행({len(C.ARMS)}팔) · SEC-R 팔별 {C.N_SECR}행 '
              f'(⚠문턱 17 은 n=24 전용 · 19 는 n=28 전용)',
              all(v == C.N_PRIMARY for v in per.values()) and len(P) == C.N_PRIMARY * len(C.ARMS)
              and (all(v == C.N_SECR for v in perr.values())
                   and len(R) == C.N_SECR * len(C.SECR_ARMS) if R else True),
              f'PRIMARY {len(P)}행 · SEC-R {len(R)}행 · n_pri={n_pri}'))
    nf = [r['block'] for r in rows if r.get('nrmse') is None or not np.isfinite(r['nrmse'])]
    A.append(('★G-FINITE 모든 행의 nRMSE 가 유한 (⛔NaN 은 조용히 패배가 된다)',
              not nf, f'비유한 {len(nf)}'))

    # ── 개입이 실제로 걸렸는가 (★이 배치의 심장) ──────────────────────────
    kbad, kdet = [], {}
    for r in rows:
        kb = r.get('knobs_before') or {}
        def _ne(a, b):                 # ★수치는 오차 비교, 지문(문자열)은 동등 비교
            if a is None or b is None:
                return False
            if isinstance(a, str) or isinstance(b, str):
                return a != b
            return abs(float(a) - float(b)) > 1e-12
        diff = frozenset(k for k in C.KNOB_KEYS if _ne(kb.get(k), dep.get(k)))
        exp = C.EXPECT_DIFF.get(r['arm'], frozenset())
        kdet.setdefault(r['arm'], set()).update(diff)
        if diff != exp:
            kbad.append(f"{r['arm']}:{r['block'][-14:]}")
    A.append(('★★G-KNOB 팔 설정 직후 **살아있는 상태** 10키 되읽기 — 정확히 그 팔의 노브만 다르다',
              not kbad, f'위반 {len(kbad)}' + (f' 예 {kbad[:2]}' if kbad else '')))
    def _neq(a, b):
        if a is None or b is None:
            return True
        return (a != b) if isinstance(a, str) or isinstance(b, str) \
            else abs(float(a) - float(b)) > 1e-12
    rbad = [f"{r['arm']}:{r['block'][-14:]}" for r in rows
            if any(_neq((r.get('knobs_after') or {}).get(k), dep.get(k))
                   for k in C.KNOB_KEYS)]
    A.append(('★★G-RESTORE 팔 종료 후 10키가 배포값으로 복귀 (⛔프로세스 전역 오염 차단)',
              not rbad, f'위반 {len(rbad)}'))

    noop, effect = [], {}
    for a in C.TREAT_ARMS:
        same = sum(1 for b in by[a]
                   if b in by['SHIP'] and abs(by[a][b]['nrmse'] - by['SHIP'][b]['nrmse']) < 1e-12
                   and by[a][b].get('proposal_md5') == by['SHIP'][b].get('proposal_md5'))
        effect[a] = f'{len(by[a]) - same}/{len(by[a])} 다름'
        if by[a] and same == len(by[a]):
            noop.append(a)
    # ★차단은 **PRIMARY 팔**에 대해서만. SEC 팔이 NO-OP 이면 배치를 막는 게 아니라
    #   그 팔을 `NO-OP`(널 아님)으로 보고한다 — PREREG §6 의 *"NO-OP 로 보고(널 아님)"* 조항.
    #   (AMENDMENT_1_PREDATA §1: 사전등록 표의 통과조건 열과 그 조항이 서로 어긋났다)
    A.append(('★★G-EFFECT 개입 계수기 · PRIMARY 팔 no-op 이면 차단 · SEC 는 NO-OP 보고',
              C.PRIMARY_ARM not in noop
              and all(r.get('n_grow_calls') is not None for a in C.TREAT_ARMS
                      for r in by[a].values()),
              f'NO-OP 팔 {noop or "없음"}'))
    pbad = []
    for a in C.TREAT_ARMS:
        for b, r in by[a].items():
            s = by['SHIP'].get(b)
            if not s:
                continue
            same = r.get('proposal_md5') == s.get('proposal_md5')
            if a in C.PRE_PROPOSAL and same:
                pbad.append(f'{a}:same')
            if a in C.POST_PROPOSAL and not same:
                pbad.append(f'{a}:diff')
    propid_block = bool(meta.get('propid_blocking', True))
    A.append((('★★G-PROPID 제안 이전 팔=전건 상이 · 이후 팔=전건 동일'
               if propid_block else
               '⚠G-PROPID [비차단 — GPU 재현성 실패로 강등, PREREG §2.5 분기]'),
              (not pbad) if propid_block else True, f'위반 {len(pbad)}'))
    rng_moved = [r['block'] for r in rows
                 if r.get('rng_md5_before') != r.get('rng_md5_after')]
    A.append(('★★G-RNGSTREAM 팔 전후 전역 numpy 상태 md5 되읽기 (before==after ⟹ 스트림 미진행)',
              all(r.get('rng_md5_before') for r in rows),
              f'진행한 행 {len(rng_moved)}/{len(rows)}'))
    gf = sum(1 for r in by['SHIP'].values() if (r.get('n_grow_calls') or 0) > 0)
    A.append((f'★★G-GROWFIRE SHIP 성장 발화 행 (< {C.GROWFIRE_MIN} 이면 C-* 는 NOT-EXERCISED)',
              all(r.get('n_grow_calls') is not None for r in by['SHIP'].values()),
              f'발화 {gf}/{len(by["SHIP"])}'))

    # ── 팔 정체·누출 ───────────────────────────────────────────────────────
    leak = [r for r in rows if r['arm'] not in ('ZERO', 'ORACLE')
            and (r.get('gt_std') is None or r.get('gt_std_after') is None
                 or abs(r['gt_std']) > 1e-12 or abs(r['gt_std_after']) > 1e-12
                 or abs((r.get('gt_value') or -9) - C.GT_CONST) > 1e-12)]
    A.append(('★★G-NOLEAK gt 가 상수(std 0) · 값==GT_CONST · initialize 후에도 [되읽기]',
              not leak, f'위반 {len(leak)}'))
    abad = []
    for a in C.ARMS:
        if a in ('ZERO', 'ORACLE'):
            continue
        fps = [r.get('proposal_md5') for r in by[a].values()]
        if not fps or len(set(fps)) != len(fps) or not all(fps):
            abad.append(a)
    A.append(('★★G-ATOMS 팔마다 제안 원자 지문이 전건 상이 (config 캐시 누출 탐지)',
              not abad, f'위반 팔 {abad or "없음"}'))
    k_dep = dep['learned_ensemble_k']
    arm_bad = [f"{r['arm']}" for r in rows if r['arm'] not in ('ZERO', 'ORACLE')
               and ((r.get('prov') or {}).get('resolved_init_method') != 'learned'
                    or (r.get('prov') or {}).get('real_domain_gate_fired') is not False
                    or (r.get('prov') or {}).get('ensemble_k') != k_dep)]
    A.append((f'★★G-ARM resolved==learned · 실측게이트 미발화 · ensemble_k=={k_dep} [되읽기]',
              not arm_bad, f'위반 {len(arm_bad)}'))
    pin_bad = sorted(k for k in C.SHIP_PIN if dep.get(k) != C.SHIP_PIN[k])
    A.append(('★G-SHIPPIN 배포 상수 17종을 **임포트해** 사전등록 핀과 대조 (⛔손복사 0)',
              not pin_bad, f'불일치 {pin_bad or "없음"} (17종)'))

    # ── 앵커 (T11 재현) ────────────────────────────────────────────────────
    t11 = {}
    if os.path.exists(C.T11_RESULTS):
        for line in open(C.T11_RESULTS, encoding='utf-8'):
            if line.strip():
                d = json.loads(line)
                t11[(d['arm'], d['block'])] = d['nrmse']
    dmax, npair = 0.0, 0
    for a in ('SHIP', 'RESCALE'):
        for b, r in by[a].items():
            if (a, b) in t11:
                dmax = max(dmax, abs(r['nrmse'] - t11[(a, b)])); npair += 1
    bit = npair == 2 * C.N_PRIMARY and dmax <= 1e-9
    sign_ok = (W.get('RESCALE', (0, 0))[0] >= C.THRESH_PRIMARY
               and W.get('SHIP', (99, 0))[0] <= 12)
    A.append(('★★G-ANCHOR SHIP·RESCALE 48값 vs T11 동결 결과 (BIT 아니면 부호 재현 필수)',
              bit or sign_ok,
              f'짝 {npair}/48 · max|Δ| {dmax:.3g} · {"BIT" if bit else "부호" if sign_ok else "실패"}'))

    # ── 되읽기 규약 ────────────────────────────────────────────────────────
    tb = [r for r in rows if r.get('t_origin') is None or abs(r['t_origin']) > 1e-9]
    A.append(('★G-TORIGIN 창 시작 t == 0 [변화 탐지기]', not tb, f'위반 {len(tb)}'))
    db = []
    for r in rows:
        es = r.get('e_shape') or [0]
        if (r.get('n_offered') != C.DENSITY * int(C.WIN_S) or r.get('n_frames') != int(C.WIN_S)
                or es[0] < C.J0 + int(C.WIN_S) or r.get('dt') is None or r.get('tq') is None
                or abs(r['tq'] - (int(C.WIN_S) - 1) * r['dt']) > 1e-9):
            db.append(r['block'])
    A.append((f'★G-DENSITY 제공=={C.DENSITY}x{int(C.WIN_S)} · 창이 자료 안 · tq==(N-1)dt [되읽기]',
              not db, f'위반 {len(db)}'))
    rs = [r for r in rows if r['arm'] == 'RESCALE']
    rsb = [r['block'] for r in rs
           if r.get('x_scale') is None or abs(r['x_scale'] - 1.0 / C.RESCALE_S) > 1e-9
           or r.get('disp_relerr') is None or r['disp_relerr'] > 1e-6]
    A.append((f'★G-RESCALE 측정 배율이 s={C.RESCALE_S} 와 일치 · ω=√(gk) 보존 [변화 탐지기]',
              not rsb, f'위반 {len(rsb)}'))

    # ── 환경·동결 ──────────────────────────────────────────────────────────
    now = C.freeze_map(); rec = meta.get('code_md5') or {}
    fz0, fza = _frozen(FROZEN), _frozen(FROZEN_A)
    run_keys = sorted(k for k in now if k not in C.ANALYSIS_SIDE)
    ana_keys = sorted(k for k in now if k in C.ANALYSIS_SIDE)
    d_meta = [k for k in run_keys if now[k] != rec.get(k)]
    d_fz0 = None if fz0 is None else [k for k in run_keys if now[k] != fz0.get(k)]
    d_fza = None if fza is None else sorted(k for k in set(ana_keys) | set(fza)
                                            if now.get(k) != fza.get(k))
    A.append(('★G-FREEZE 측정면 3중 대조(현재==meta==결과0행동결) · 분석면 동결',
              bool(rec) and not d_meta and d_fz0 == [] and d_fza == [],
              f'측정면 {len(run_keys)}키 · 런대비 {len(d_meta)} · 0행동결대비 '
              f'{"N/A" if d_fz0 is None else len(d_fz0)} · 분석면 '
              f'{"N/A" if d_fza is None else len(d_fza)}'))
    want_seed = {C.SEED + i for i in range(1, C.N_PRIMARY + 1)}
    sbad = sorted(a for a in C.ARMS if {r.get('seed') for r in by[a].values()} != want_seed)
    A.append(('G-ENV 스레드[되읽기]==1 · **행 시드 집합 되읽기** · ckpt sha256 2개',
              meta.get('threads_readback') == 1 and not sbad
              and now.get('sha256:' + C.CKPTS[0]) == C.CKPT_SHA_PIN[C.CKPTS[0]],
              f"threads={meta.get('threads_readback')} 시드위반팔 {len(sbad)}"))
    probe = G.blocked_by([('★★G-NOLEAK 합성', False, ''), ('★G-N 합성', False, ''),
                          ('★★G-RESTORE 합성', False, ''), ('★G-RESCALE 합성', False, ''),
                          ('G-ENV 합성', True, '')])
    A.append(('G-BLOCK blocked_by 자가진단 (긴 이름 우선)',
              probe == ['G-NOLEAK', 'G-N', 'G-RESTORE', 'G-RESCALE'], f'합성 → {probe}'))
    A.append(('G-COV 게이트가 자료를 봤다', len(rows) > 0, f'rows={len(rows)}'))

    blocked = G.blocked_by(A)
    all_ok = all(bool(o) for _, o, _ in A)      # ★인쇄 벨트 (PREREG §4.3)
    lab = G.label(w_pri, w_secmax) if (not blocked and all_ok) else None

    # ── 기술통계 (⛔라벨을 만들지 않는다) ──────────────────────────────────
    rho = {}
    for a in C.RECOVERY_ARMS:
        vals = []
        for b in sorted(set(by[a]) & set(by['SHIP']) & set(by['RESCALE'])):
            den = by['SHIP'][b]['nrmse'] - by['RESCALE'][b]['nrmse']
            if den > 1e-12:
                vals.append((by['SHIP'][b]['nrmse'] - by[a][b]['nrmse']) / den)
        rho[a] = dict(median=float(np.median(vals)) if vals else None,
                      q1=float(np.percentile(vals, 25)) if vals else None,
                      q3=float(np.percentile(vals, 75)) if vals else None, n=len(vals))
    strata, clusters = {}, {}
    for a in C.ARMS:
        if a == 'ZERO':
            continue
        for nm, pr in (('hs_low', lambda h: h <= C.HS_SPLIT), ('hs_high', lambda h: h > C.HS_SPLIT)):
            cm = [b for b in sorted(set(by[a]) & set(by['ZERO'])) if pr(by[a][b]['hs'])]
            strata[f'{a}_{nm}'] = {'w': sum(1 for b in cm
                                            if by[a][b]['nrmse'] < by['ZERO'][b]['nrmse']),
                                   'n': len(cm)}
        agg = {}
        for b in sorted(set(by[a]) & set(by['ZERO'])):
            k = str(by[a][b]['angle']); agg.setdefault(k, [0, 0]); agg[k][1] += 1
            if by[a][b]['nrmse'] < by['ZERO'][b]['nrmse']:
                agg[k][0] += 1
        clusters[a] = {k: {'w': v[0], 'n': v[1]} for k, v in sorted(agg.items())}
    mediation = {a: {k: float(np.median([r[k] for r in by[a].values() if r.get(k) is not None]))
                     for k in ('kp_true', 'kmed_raw', 'kmed_sel', 'kmed_final', 'ratio')
                     if any(r.get(k) is not None for r in by[a].values())}
                 for a in C.ARMS if by[a]}
    not_exercised = list(C.GROWTH_ARMS) if gf < C.GROWFIRE_MIN else []

    v = {'batch': 'T12-ISOLATE', 'kind': 'CONFIRMATORY',
         'grade': 'CONFIRMATORY (⚠노출된 표본 위의 확증 — PREREG §2.1)',
         'prereg': 'PREREG.md + REACHABILITY.md (결과 0행·러너 0줄 시점 동결)',
         'n_rows': len(rows), 'n_primary': len(P), 'n_secr': len(R),
         'gates': [{'name': nm, 'pass_': bool(o), 'detail': d} for nm, o, d in A],
         'gates_pass': f'{sum(1 for _, o, _ in A if o)}/{len(A)}', 'blocked_by': blocked,
         'primary': {'stat': f'{C.PRIMARY_ARM} < ZERO', 'w': w_pri, 'n': n_pri,
                     'p_one_sided': G.sign_p(w_pri, n_pri), 'thresh': C.THRESH_PRIMARY},
         'sec_family': {a: {'w': sec[a][0], 'n': sec[a][1],
                            'p_one_sided': G.sign_p(*sec[a])} for a in C.SEC8},
         'sec_family_thresh': C.THRESH_SEC, 'sec_max': w_secmax,
         'anchors': {a: {'w': W[a][0], 'n': W[a][1]} for a in ('SHIP', 'RESCALE', 'ORACLE')
                     if a in W},
         'secr_exploratory': {a: {'w': Wr[a][0], 'n': Wr[a][1],
                                  'p_one_sided': G.sign_p(*Wr[a])} for a in Wr},
         'not_exercised': not_exercised, 'grow_fire_ship': gf, 'no_op_arms': noop,
         'knob_diffs_measured': {a: sorted(v2) for a, v2 in sorted(kdet.items())},
         'effect_vs_ship': effect, 'rho_recovery': rho,
         'medians': {a: float(np.median([r['nrmse'] for r in by[a].values()]))
                     for a in C.ARMS if by[a]},
         'strata': strata, 'angle_clusters': clusters, 'mediation': mediation,
         'mde': C.MDE, 'prediction': {'q': C.PRED_Q, 'label_probs': G.label_probs(),
                                      'null_label_probs': G.label_probs(
                                          {a: 0.5 for a in C.PRED_Q})},
         'proposal_device': sorted({r.get('proposal_device') for r in rows
                                    if r.get('proposal_device')}),
         'label': lab, 'label_msg': lab or 'BLOCKED',
         'disclosures': [
             '⛔RESCALE 을 실제 성능으로 인용 금지 — 단위를 바꾼 사고실험이지 배포 구성이 아니다',
             '⛔ORACLE 은 추정기가 아니다(참 파수를 받는다) — 정보 존재의 상한',
             '⛔처치 팔은 전부 배포 구성이 아니다 — 어떤 팔이 이겨도 **배포 권고가 아니다**',
             '⛔「단일 상수」 금지 — 정본 표현은 **「단일 서브시스템의 절대 스케일」** '
             '(PREREG §1.1 의 동반 이동: A-BAND 는 polish 물리 스텝을, C-GROWGRID 는 DC블록·'
             '억제반경·refine 격자를 함께 옮긴다)',
             '⛔E2 와 대각선 비교 금지 · 저들 논문 수치와 비교 금지 · solver 미명시 '
             '(T11 PREREG §6 의무 공시 — T11 AMENDMENT_3 §1 의 삭제 사고를 재발시키지 않는다)',
             '⚠예산 비대칭: 제안 단계는 프레임당 8,000 관측 중 캡 400 만 쓴다 — '
             '"고른 400" 과 "있는 것 전부" 는 다르다',
             '⚠Tp 가 PRIMARY 층 전 파일 10.0 s ⟹ 한 주기 조건의 진술이다',
             '⚠완전 blind 아님 · 노출 표본(PREREG §2.1) ⟹ ⛔"독립 확증"·"blind"·'
             '"out-of-sample" 자구 금지. 처치 9팔은 미실행이었으나 앵커 4팔 값은 알려져 있었다',
             '⚠조건화 빔포머 DEPTH=50 하드코딩(d3_beamform_net.py:30)은 config depth 가 '
             '닿지 않는 곳이다 — E-MAPDEPTH 로 재기만 하고 배포 기본값을 바꾸지 않는다',
             '⛔E-MAPDEPTH·F-STEEP 에 회수율 ρ 를 인쇄하지 마라 — 스케일 축이 아니다 '
             '(PREREG §3). 그 둘의 승리는 다른 문장이다',
             '⛔ρ 를 팔 간에 더하지 마라 — 합이 1 이 될 이유가 없다 (P8 MAJOR 선례)',
             '⛔성장 발화가 희소하면 C-MINDK·C-GROWGRID 는 NOT-EXERCISED 이지 널이 아니다',
             '⛔D-VPLR 이 널이면 「no-op」과 구별해야 한다 — vp_lr 되읽기와 vp_k_shift 가 '
             '그 구별의 근거다 (기본 인자 바인딩 때문에 순진한 개입은 조용히 no-op 이 된다)',
             '⛔proposal_device 는 config 가 정하지 않는다(propose_atoms 가 cuda-if-available) · '
             '⛔wall_s 를 팔 간 비교에 쓰지 마라',
             f'⛔음성 결과 인용 시 MDE 병기 의무: PRIMARY q80={C.MDE["primary_q80"]} · '
             f'SEC family q80={C.MDE["sec_q80"]}(참 승률 0.70 축을 77% 확률로 놓친다) · '
             f'기각 불가 대역 W={C.MDE["unrejectable_w"]}',
             '⛔NO-SINGLE-AXIS-DETECTED 는 **미검출**이다 — "인수분해되지 않는다"·"상호작용"·'
             '"어느 상수도 원인이 아니다"·"도달 불가" 로 읽는 것을 금지한다 (M8b 선례)',
             '⛔다중비교: PRIMARY 1 + SEC family 8(Holm) = 보정 대상 9. SEC-R·층·각도·ρ·'
             '매개열은 **무보정 기술통계**이며 라벨을 만들지 않는다 '
             '(층 n=12 는 MDE q≥0.869, 각도 n=4 는 만장일치도 p=0.0625 ⟹ 클러스터 주장 불가)',
         ]}
    json.dump(v, open(os.path.join(OUTDIR, 'verdict_t12.json'), 'w'),
              ensure_ascii=False, indent=1)
    print(f"T12-ISOLATE :: {v['label_msg']}")
    print(f"  PRIMARY {C.PRIMARY_ARM} < ZERO  {w_pri}/{n_pri}  p={v['primary']['p_one_sided']}"
          f"  (문턱 {C.THRESH_PRIMARY})")
    print(f"  SEC family (문턱 {C.THRESH_SEC}, Holm m=8):")
    for a in C.SEC8:
        print(f"     {a:12s} {sec[a][0]:2d}/{sec[a][1]:2d}  p={G.sign_p(*sec[a]):.4g}")
    print(f"  앵커 {v['anchors']}")
    if R:
        print(f"  SEC-R [EXPLORATORY] {{a: (w,n)}} = "
              f"{ {a: Wr[a] for a in Wr} }")
    print(f"  중앙 nRMSE {v['medians']}")
    print(f"  회수율 ρ 중앙 { {a: (round(x['median'],3) if x['median'] is not None else None) for a,x in rho.items()} }")
    print(f"  성장 발화(SHIP) {gf}/{len(by['SHIP'])} · NOT-EXERCISED {not_exercised or '없음'}")
    print(f"  측정된 노브 차이 {v['knob_diffs_measured']}")
    print(f"  제안 device {v['proposal_device']}")
    print(f"  gates {v['gates_pass']}")
    for nm, o, d in A:
        print(f"   [{'OK ' if o else 'FAIL'}] {nm:72s} {d}")


if __name__ == '__main__':
    main()
