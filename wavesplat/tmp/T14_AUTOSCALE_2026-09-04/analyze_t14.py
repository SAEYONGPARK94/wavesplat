"""`T14-AUTOSCALE` 분석기 — ⛔러너보다 **먼저**. 이 파일이 **행 스키마를 정의**한다.

행 필수 열 (PREREG §6.5): block · tier · arm · angle · hs · sigma4 · frame_index · seed
  n_frames · density · dt · tq · e_shape · mean_convention · axis_convention
  nrmse · nrmse_explicit · n_offered · n_gathered · cap_proposal
  gt_std · gt_value · gt_std_after · proposal_md5 · atoms_md5 · proposal_device
  vp_clamp_hits · n_atoms_installed · prov · wall_s
  ★s_rule · s_readback · s_blind · s_c05 · s_c20 · x_scale · t_scale · disp_relerr
  ★kp_hat · kp_hat_stage1 · L_obs_x · L_obs_y · L_TRAIN · K_TRAIN · s_clipped
  ★kd_min · use_stokes2 · use_neural_corrector · invert_maxdiff · deploy_consts
"""
import json, os, re, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_t14 as G                                             # noqa: E402
import scale_rules_t14 as SR                                      # noqa: E402

ROWS = os.path.join(HERE, 'results_t14.jsonl'); META = os.path.join(HERE, 'meta_t14.json')
FROZEN = os.path.join(HERE, 'FROZEN_md5.txt')
OUTDIR = HERE
BANNED = ('target', 'gt', 'eta', 'Ew', 'hs', 'angle', 'sigma4', 'block')


def _frozen(p):
    if not os.path.exists(p):
        return None
    out = {}
    for line in open(p, encoding='utf-8'):
        if line.strip() and not line.startswith('#'):
            h, k = line.split(None, 1)
            out[k.strip()] = h.strip()
    return out


def main():
    rows = [json.loads(x) for x in open(ROWS, encoding='utf-8') if x.strip()]
    meta = json.load(open(META, encoding='utf-8'))
    t1, t2, t3, anc = G.tiers()
    by = {}
    for r in rows:
        by.setdefault((r['tier'], r['arm']), {})[r['block']] = r

    def wins(arm, ref='ZERO', tier='T1'):
        d, z = by.get((tier, arm), {}), by.get((tier, ref), {})
        cm = sorted(set(d) & set(z))
        return sum(1 for b in cm if d[b]['nrmse'] < z[b]['nrmse']), len(cm)

    w_ap, n_ap = wins('AUTO-AP'); w_bd, n_bd = wins('AUTO-BAND')
    w_fx, n_fx = wins('FIX2'); w_sh, n_sh = wins('SHIP')
    A = []
    est = [r for r in rows if r['arm'] != 'ZERO']
    auto = [r for r in rows if r['arm'] in G.AUTO_ARMS]

    # ── 6.1 `s` 가 참값을 안 봤음 ─────────────────────────────────────────
    bb = [f"{r['arm']}:{r['block'][-12:]}" for r in auto
          if r.get('s_blind') is None or r['s_blind'] != r.get('s_rule')]
    A.append(('★★G-S-BLIND 관측 5배열만 별도 프로세스에 넘겨 규칙 재실행 ⟹ s 비트 동일 [행동 검사]',
              bool(auto) and not bb, f'위반 {len(bb)}/{len(auto)}'))
    import inspect
    src_bad = []
    for nm, fn in SR.RULES.items():
        sig = list(inspect.signature(fn).parameters)
        srcs = inspect.getsource(fn)
        if sig != ['x', 'y', 't', 'z', 'depth']:
            src_bad.append(f'{nm}:sig{sig}')
        for tok in BANNED:
            if re.search(r'\b' + tok + r'\b', srcs):
                src_bad.append(f'{nm}:{tok}')
    A.append(('★G-S-SRC 규칙이 정확히 5인자 · 소스에 표적/메타 토큰 0회 [⚠단독 항진 위험 — '
              'G-S-BLIND 가 하중을 진다]', not src_bad, f'위반 {src_bad or "없음"}'))
    rb = []
    for r in est:
        s = r.get('s_rule')
        if s is None or r.get('x_scale') is None or r.get('t_scale') is None:
            rb.append(r['arm']); continue
        # ⛔`(x or 1)` 을 쓰지 마라 — **0.0 은 falsy** 라 「완벽함」이 위반으로 센다
        #   (AMENDMENT_1 §1: 스모크에서 7/14 행이 그 이유로 FAIL 했다)
        dre = r.get('disp_relerr')
        if abs(1.0 / r['x_scale'] - s) > 1e-9 * max(1, s) \
                or abs(r['t_scale'] - 1.0 / np.sqrt(s)) > 1e-9 \
                or dre is None or dre > 1e-9:
            rb.append(f"{r['arm']}:{r['block'][-12:]}")
    A.append(('★★G-S-READBACK 변환된 좌표에서 s 되읽기 (1/median(x\'/x)==s · median(t\'/t)==1/√s)',
              not rb, f'위반 {len(rb)}'))
    # ★**비클리핑** 규칙으로 잰다 — 안전 레일이 등변성을 깬다(AMENDMENT_1 §2)
    # ★★그리고 두 규칙의 등변성은 **같지 않다**(AMENDMENT_1 §3):
    #   `AUTO-AP` 는 정확히 등변(순수 기하) · `AUTO-BAND` 는 2단 줌의 **1단이 절대 격자**라
    #   근사적이다. 허용오차는 **코드에서 나온다** — 2단 격자 `2·3.5/(120−1) = 5.88%`.
    #   ⛔자료에서 고르지 않았다.
    BAND_TOL = 7.0 / 119.0
    eb = []
    for r in auto:
        if r.get('s_c20') is None or r.get('s_raw') is None:
            eb.append(f"{r['arm']}:missing"); continue
        tol = 1e-6 if r['arm'] == 'AUTO-AP' else BAND_TOL
        if abs(r['s_c20'] * 2.0 / r['s_raw'] - 1) > tol:
            eb.append(f"{r['arm']}:c20")
        if r['arm'] == 'AUTO-AP' and (r.get('s_c05') is None
                                      or abs(r['s_c05'] * 0.5 / r['s_raw'] - 1) > 1e-6):
            eb.append(f"{r['arm']}:c05")
    A.append((f'★G-EQUIVAR 등변성 — AUTO-AP 는 1e-6(정확) · AUTO-BAND 는 코드유래 {BAND_TOL:.4f} '
              f'(c=2.0 방향) [런 0회]', not eb, f'위반 {eb or "없음"}'))
    # ── ★신설 G-BAND-FRAGILE (기록 전용 — AMENDMENT_1 §3) ────────────────
    #   1단 격자 간격 `2·2.0/119 = 0.0336`. 장면 k_p 가 그보다 크지 않으면 규칙이 무너진다.
    STAGE1_DK = 4.0 / 119.0
    frag = [{'block': r['block'][-14:], 'kp': r.get('kp_hat'),
             'kp_over_dk1': (r['kp_hat'] / STAGE1_DK) if r.get('kp_hat') else None,
             'dev_c05': (abs(r['s_c05'] * 0.5 / r['s_raw'] - 1)
                         if r.get('s_c05') and r.get('s_raw') else None)}
            for r in auto if r['arm'] == 'AUTO-BAND']
    A.append(('⚠G-BAND-FRAGILE [기록 전용] BAND 규칙의 1단 격자 여유 k_p/Δk₁ 와 c=0.5 편차',
              True, f'k_p/Δk₁ 중앙 '
                    f'{np.median([f["kp_over_dk1"] for f in frag if f["kp_over_dk1"]]):.2f} '
                    f'· c=0.5 편차 중앙 '
                    f'{np.median([f["dev_c05"] for f in frag if f["dev_c05"] is not None]):.3f}'
              if frag else '자료 없음'))
    try:
        sys.path.insert(0, os.path.join(G.ROOT, 'scripts'))
        from d40_learned_acquisition import pool_xy
        P = np.asarray(pool_xy())                      # ★(N, 2) 배열이다
        lt = max(float(np.ptp(P[:, 0])), float(np.ptp(P[:, 1])))
    except Exception as e:                                        # noqa: BLE001
        lt = None
    A.append(('★G-LTRAIN pool_xy() 재호출로 L_TRAIN 재도출 == 동결값 (⛔손복사 금지)',
              lt is not None and abs(lt - SR.L_TRAIN) < 0.01,
              f'재생성 {None if lt is None else round(lt, 3)} vs 동결 {SR.L_TRAIN}'))
    try:                    # ★원 절차 = Tp 에서 **분산관계로** k_p 를 직접 유도(AMENDMENT_1 §4)
        from scipy.optimize import brentq
        from wavesplat.model.dispersion import G as _g
        Tp = np.linspace(5.0, 11.0, 20001)
        kk = np.array([brentq(lambda k, w=2 * np.pi / tp: _g * k * np.tanh(k * 50.0) - w * w,
                              1e-6, 10.0) for tp in Tp])
        kt = float(np.exp(np.mean(np.log(kk))))
    except Exception:                                             # noqa: BLE001
        kt = None
    A.append(('★G-KTRAIN 사전 밴드에서 K_TRAIN 기하평균 재생성 == 동결값 (⛔손복사 금지)',
              kt is not None and abs(kt / SR.K_TRAIN - 1) < 0.02,   # ★원 추정의 MC ±2σ
              f'재생성 {None if kt is None else round(kt, 5)} vs 동결 {SR.K_TRAIN}'))
    import d3_beamform_net as bn
    from wavesplat.optimizer.initializer import MIN_DK
    pin = (abs(bn.DEPTH - 50.0) < 1e-12 and abs(bn.KLIM - 0.6) < 1e-12
           and bn.NK == 64 and abs(MIN_DK - 0.15) < 1e-12)
    A.append(('G-SRC-PIN 소스 리터럴 대조 (DEPTH=50 · KLIM=0.6 · NK=64 · MIN_DK=0.15)',
              pin, f'DEPTH={bn.DEPTH} KLIM={bn.KLIM} NK={bn.NK} MIN_DK={MIN_DK}'))

    # ── 6.2 역변환·앵커 ───────────────────────────────────────────────────
    ib = [f"{r['arm']}:{r['block'][-12:]}" for r in est
          if r.get('invert_maxdiff') is None
          or r['invert_maxdiff'] > G.INVERT_TOL_REL * r['sigma4']]
    A.append((f'★G-INVERT 암묵/명시 역변환 두 경로 차 ≤ {G.INVERT_TOL_REL}·σ4 [필드 차이로 잰다]',
              not ib, f'위반 {len(ib)}/{len(est)}'))
    pb = [r['arm'] for r in est if r.get('use_stokes2') is not False
          or r.get('use_neural_corrector') is not False]
    A.append(('G-PURE use_stokes2 · use_neural_corrector 둘 다 False [모델 객체 되읽기]',
              not pb, f'위반 {len(pb)}'))
    kdb = [r['block'] for r in auto if r.get('kd_min') is None or r['kd_min'] < G.KD_MIN]
    kmin = min([r['kd_min'] for r in auto if r.get('kd_min') is not None] or [-1])
    A.append(('★G-DEEP 모든 행 k̂_p·depth ≥ π (⟹ tanh ≥ 0.9963, 수심 미재척도가 무해)',
              not kdb, f'위반 {len(kdb)} · kd_min {kmin:.1f}'))
    sb = [r['block'] for r in rows if r['arm'] == 'SHIP'
          and (r.get('x_scale') != 1.0 or r.get('t_scale') != 1.0)]
    A.append(('★G-SHIPID SHIP 은 항등 재라벨 — x_scale==1.0 ∧ t_scale==1.0 **정확히**',
              not sb, f'위반 {len(sb)}'))
    t11s = {}
    if os.path.exists(G.SMOKE_T11):
        for line in open(G.SMOKE_T11, encoding='utf-8'):
            if line.strip():
                d = json.loads(line)
                t11s[(d['arm'], d['block'])] = d['nrmse']
    grade, abad, npair = 'T1', [], 0
    for b in anc:
        for arm, ref in (('SHIP', 'SHIP'), ('FIX2', 'RESCALE')):
            r = by.get(('ANCHOR', arm), {}).get(b)
            v = t11s.get((ref, b))
            if r is None or v is None:
                continue
            npair += 1
            if r['nrmse'] != v:
                grade = 'T2'
                if abs(r['nrmse'] / v - 1) > 1e-9:
                    abad.append(f'{arm}:{b[-12:]}')
    A.append((f'★★G-ANCHOR-T11 앵커 2파일의 SHIP·FIX2 가 T11 스모크와 일치 [등급 {grade}]',
              npair == 4 and not abad, f'짝 {npair}/4 · 불일치 {len(abad)} · 등급 {grade}'))
    from wavesplat.optimizer.sequential_update import (
        DEFAULT_LEARNED_ENSEMBLE_K as KDEP, VARPRO_STEPS_DEFAULT, LAMBDA_L1_DEFAULT,
        LEARNED_INIT_ITERS_DEFAULT, AUTO_SWITCH_THRESH_DEFAULT, DEFAULT_ANALYTIC_ENSEMBLE_K)
    want_dep = {'lambda_l1': float(LAMBDA_L1_DEFAULT), 'varpro_steps': int(VARPRO_STEPS_DEFAULT),
                'learned_init_iters': int(LEARNED_INIT_ITERS_DEFAULT),
                'auto_switch_thresh': int(AUTO_SWITCH_THRESH_DEFAULT),
                'init_method': 'auto', 'tracking_mode': 'w_phi',
                'analytic_ensemble_k': int(DEFAULT_ANALYTIC_ENSEMBLE_K),
                'learned_ensemble_k': int(KDEP)}
    dbad = [r['arm'] for r in est
            if any((r.get('deploy_consts') or {}).get(k) != v for k, v in want_dep.items())]
    A.append(('★G-DEPLOY 배포 상수 8종을 cfg 되읽기로 확인 (⛔손복사 0 — 전부 임포트)',
              not dbad, f'위반 {len(dbad)}'))

    # ── 6.3 T11 계승 ──────────────────────────────────────────────────────
    sz = os.path.getsize(G.ZIP) if os.path.exists(G.ZIP) else -1
    A.append((f'★G-SIZE 분석기가 직접 잰 크기 == {G.WANT_BYTES}', sz == G.WANT_BYTES, str(sz)))
    got1 = set(by.get(('T1', 'SHIP'), {}))
    t11set = set(json.load(open(G.SAMPLE_T11, encoding='utf-8')))
    mats = G.archive_mats()
    A.append(('★G-SAMPLE Tier-1 == 집합 차 · T11 24 / 감사 스모크 8 과 교집합 0 [namelist 직접]',
              got1 == set(t1) and len(t1) == G.N_T1 and not (set(t1) & t11set)
              and not (set(t1) & {mats[i] for i in G.AUDIT_SMOKE_IDX}),
              f'Tier-1 {len(got1)} · T11 교집합 {len(set(t1) & t11set)}'))
    pop_ok = (all(G.POP_RANGE[0] <= mats.index(b) < G.POP_RANGE[1] for b in t1)
              and all('outofdistribution' in b for b in t3))
    A.append(('★G-POP Tier-1 ⊆ mixed_data2 idx 60–119 · Tier-3 ⊆ outofdistribution', pop_ok, '—'))
    cnt = {k: len(v) for k, v in sorted(by.items())}
    A.append((f'★G-N 총 {G.N_ROWS}행 · Tier-1 5팔×{G.N_T1} · Tier-2 3팔×{G.N_T2} · '
              f'Tier-3 5팔×{G.N_T3} · 앵커 3팔×{G.N_ANC} (⚠문턱 20 은 n=28 전용)',
              len(rows) == G.N_ROWS and n_ap == G.N_T1 and n_bd == G.N_T1,
              f'rows={len(rows)} · n_ap={n_ap} · n_bd={n_bd}'))
    nfin = [r['block'] for r in rows if r.get('nrmse') is None or not np.isfinite(r['nrmse'])]
    A.append(('★G-FINITE 모든 행의 nRMSE 가 유한', not nfin, f'비유한 {len(nfin)}'))
    leak = [r for r in est if r.get('gt_std') is None or abs(r['gt_std']) > 1e-12
            or r.get('gt_std_after') is None or abs(r['gt_std_after']) > 1e-12
            or abs((r.get('gt_value') or -9) - G.GT_CONST) > 1e-12]
    A.append(('★★G-NOLEAK gt 상수(std 0) · 값==GT_CONST · initialize 후에도 [되읽기]',
              not leak, f'위반 {len(leak)}'))
    dupa = [k for k, d in by.items() if k[1] != 'ZERO'
            and len({r.get('proposal_md5') for r in d.values()}) != len(d)]
    A.append(('★G-ATOMS raw 제안 지문이 **팔별로** 전건 상이 (캐시 누출 탐지)',
              not dupa, f'중복 {dupa or "없음"}'))
    from wavesplat.optimizer.learned_init import M_OBS_CAP
    cb = [r['arm'] for r in est if r.get('n_gathered') is None
          or r['n_gathered'] != r.get('n_offered') or r.get('cap_proposal') != int(M_OBS_CAP)]
    A.append((f'★G-CAP 모음==제공 · 캡=={int(M_OBS_CAP)} [되읽기]', not cb, f'위반 {len(cb)}'))
    arb = [r['arm'] for r in est
           if (r.get('prov') or {}).get('resolved_init_method') != 'learned'
           or (r.get('prov') or {}).get('real_domain_gate_fired') is not False
           or (r.get('prov') or {}).get('ensemble_k') != KDEP]
    A.append((f'★G-ARM resolved==learned · 실측게이트 미발화 · ensemble_k=={KDEP}',
              not arb, f'위반 {len(arb)}'))
    A.append(('G-WRAPPER 출하 진입점 [⚠배포 상수상 **구조적으로 참**]',
              all((r.get('prov') or {}).get('is_ensemble_wrapper') for r in est), '—'))
    tb = [r for r in rows if r.get('t_origin') is None or abs(r['t_origin']) > 1e-9]
    A.append(('★G-TORIGIN 창 시작 t == 0 [변화 탐지기]', not tb, f'위반 {len(tb)}'))
    db = []
    for r in rows:
        es = r.get('e_shape') or [0]
        if (r.get('n_offered') != G.DENSITY * int(G.WIN_S) or r.get('n_frames') != int(G.WIN_S)
                or es[0] < G.J0 + int(G.WIN_S) or r.get('dt') is None or r.get('tq') is None
                or abs(r['tq'] - (int(G.WIN_S) - 1) * r['dt']) > 1e-9):
            db.append(r['block'])
    A.append((f'★G-DENSITY 제공=={G.DENSITY}x{int(G.WIN_S)} · 창이 자료 안 · tq==(N-1)dt',
              not db, f'위반 {len(db)}'))
    rec = meta.get('code_md5') or {}
    fz = _frozen(FROZEN)
    live = meta.get('code_md5_now') or rec
    dm = sorted(k for k in set(rec) | set(fz or {}) if (fz or {}).get(k) != rec.get(k))
    want_seed = {G.seed_of(b, 'T1') for b in t1} | {G.seed_of(b, 'T3') for b in t3}
    got_seed = {r['seed'] for r in rows if r['tier'] in ('T1', 'T3')}
    A.append(('★G-ENV 동결 3중 대조 · threads[되읽기]==1 · 행 시드 집합 되읽기 · ckpt sha256',
              meta.get('threads_readback') == 1 and bool(rec) and fz is not None and not dm
              and got_seed == want_seed,
              f'동결드리프트 {len(dm) if fz else "N/A"} · 시드집합 일치 {got_seed == want_seed}'))
    probe = G.blocked_by([('★★G-S-BLIND 합성', False, ''), ('★G-S-SRC 합성', False, ''),
                          ('★G-N 합성', False, ''), ('G-COV 합성', True, '')])
    A.append(('G-BLOCK blocked_by 자가진단 (긴 이름 우선)',
              probe == ['G-S-BLIND', 'G-S-SRC', 'G-N'], f'합성 → {probe}'))
    A.append(('G-COV 게이트가 자료를 봤다', len(rows) > 0, f'rows={len(rows)}'))
    # ── §6.4 기록 전용 (⛔막지 않는다 — T11 개정문 2 §9.1 선례) ───────────
    _sclip = {a: sum(1 for r in rows if r['arm'] == a and r.get('s_clipped'))
              for a in G.AUTO_ARMS}
    _sv = {}
    for a in G.AUTO_ARMS:
        vv = [r['s_rule'] for r in rows if r['arm'] == a and r.get('s_rule') is not None]
        _sv[a] = (round(float(np.std(vv)), 6), round(min(vv), 4), round(max(vv), 4)) if vv else None
    _cl = {a: sum(1 for r in rows if r['arm'] == a and (r.get('vp_clamp_hits') or 0) > 0)
           for a in ('SHIP', 'FIX2') + G.AUTO_ARMS}
    A.append(('⚠G-SCLIP [기록 전용] 안전 레일 [0.25, 4.0] 접촉 횟수', True, f'{_sclip}'))
    A.append(('⚠G-SVAR [기록 전용] 팔별 s 의 (sd, min, max) — ★sd≈0 이면 **단일값 검정**이다',
              True, f'{_sv}'))
    A.append(('⚠G-CLAMP [기록 전용] 경사 클램프 발화 팔별', True, f'{_cl}'))

    blocked = G.blocked_by(A)
    all_ok = all(bool(o) for _, o, _ in A)
    lab = G.label(w_ap, w_bd, w_fx) if (not blocked and all_ok) else None

    # ── 6.4 기록 전용 ─────────────────────────────────────────────────────
    sclip = {a: sum(1 for r in rows if r['arm'] == a and r.get('s_clipped'))
             for a in G.AUTO_ARMS}
    svar = {}
    for a in G.AUTO_ARMS:
        v = [r['s_rule'] for r in rows if r['arm'] == a and r.get('s_rule') is not None]
        svar[a] = dict(sd=float(np.std(v)) if v else None, min=min(v) if v else None,
                       max=max(v) if v else None, n=len(v),
                       single_value=(bool(v) and float(np.std(v)) < 1e-9))
    clamp = {a: sum(1 for r in rows if r['arm'] == a and (r.get('vp_clamp_hits') or 0) > 0)
             for a in ('SHIP', 'FIX2') + G.AUTO_ARMS}
    sw = {a: wins(a, 'ZERO', 'T2') for a in G.T2_ARMS}
    t3w = {a: wins(a, 'ZERO', 'T3') for a in ('SHIP', 'FIX2') + G.AUTO_ARMS}
    sig_worse = {a: (w <= G.SIG_WORSE) for a, (w, n) in
                 (('AUTO-AP', (w_ap, n_ap)), ('AUTO-BAND', (w_bd, n_bd)),
                  ('SHIP', (w_sh, n_sh)), ('FIX2', (w_fx, n_fx))) if n == G.N_T1}

    def R(arm):
        d, sh, fx = by.get(('T1', arm), {}), by.get(('T1', 'SHIP'), {}), by.get(('T1', 'FIX2'), {})
        v = [(sh[b]['nrmse'] - d[b]['nrmse']) / (sh[b]['nrmse'] - fx[b]['nrmse'])
             for b in sorted(set(d) & set(sh) & set(fx))
             if sh[b]['nrmse'] - fx[b]['nrmse'] > 1e-12]
        return dict(median=float(np.median(v)) if v else None, n=len(v))

    v = {'batch': 'T14-AUTOSCALE', 'kind': 'CONFIRMATORY',
         'grade': 'CONFIRMATORY (Tier-1 두 팔 blind · ⚠Hs 3/25 치우침)',
         'n_rows': len(rows),
         'gates': [{'name': n, 'pass_': bool(o), 'detail': d} for n, o, d in A],
         'gates_pass': f'{sum(1 for _, o, _ in A if o)}/{len(A)}', 'blocked_by': blocked,
         'primary': {'AUTO-AP': {'w': w_ap, 'n': n_ap, 'p': G.sign_p(w_ap, n_ap)},
                     'AUTO-BAND': {'w': w_bd, 'n': n_bd, 'p': G.sign_p(w_bd, n_bd)},
                     'thresh': G.THRESH_PRIMARY, 'bonferroni': 'α/2 = 0.025'},
         'positive_control': {'FIX2<ZERO': {'w': w_fx, 'n': n_fx, 'p': G.sign_p(w_fx, n_fx)},
                              'thresh': G.THRESH_CTRL},
         'ship_vs_zero': {'w': w_sh, 'n': n_sh, 'p': G.sign_p(w_sh, n_sh)},
         'recovery_R': {a: R(a) for a in G.AUTO_ARMS},
         'sweep_T2_descriptive': sw, 'tier3_SEC': t3w, 'sig_worse': sig_worse,
         'anchor_grade': grade, 's_clip': sclip, 's_var': svar, 'clamp': clamp, 'band_fragile': frag,
         'medians': {f'{k[0]}:{k[1]}': float(np.median([r['nrmse'] for r in d.values()]))
                     for k, d in sorted(by.items()) if d},
         'mde': G.MDE, 'prediction': {'q': G.PRED_Q, 'label_probs': G.label_probs(),
                                      'null_label_probs': G.label_probs(0.5, 0.5, 0.5)},
         'proposal_device': sorted({r.get('proposal_device') for r in rows
                                    if r.get('proposal_device')}),
         'label': lab, 'label_msg': lab or 'BLOCKED',
         'disclosures': [
             '⛔AUTOSCALE-WORKS-* 를 "우리가 저들보다 낫다" 로 읽지 마라 — 외부 기준선 행이 '
             '**0개**다. 이것은 **귀속**(어디에 착지하면 기술이 생기는가)이다',
             '⛔NO-AUTOSCALE-DETECTED 는 **미검출**이다 — "자동 선택은 불가능하다"·"도달 불가" '
             f'로 읽는 것을 금지한다. MDE q80={G.MDE["primary_q80"]} 병기 의무 · '
             f'기각 불가 대역 w={G.MDE["unrejectable_w"]} (19/28=67.9% 도 Bonferroni 아래 기각 불가)',
             '★★이 설계는 **두 규칙 중 어느 쪽인지 구별하지 못한다** — 사전등록 예측표가 '
             'P(BOTH)=0.7369 라고 **결과 전에** 말했다. 판별은 스윕(기술통계)의 몫이다',
             '⚠라벨 확률표는 **다리 독립 가정**이다. a·b 는 같은 28파일·같은 관측점·같은 상한 '
             '앵커를 공유해 강하게 종속이다 ⟹ 표는 BOTH 를 과소, 혼합칸을 과대 평가한다. '
             '⛔사후 검정력 주장에 쓰지 마라',
             '⚠Hs 가 균형이 아니다 — Tier-1 28 은 hs<2.0 이 **3개**뿐이다(T11 이 저-Hs 를 '
             '소진했다). ⛔T11 의 층별 수치와 직접 비교 금지 · hs_low 는 n=3 ⟹ 최소 p=0.125 = '
             '**어떤 자료로도 기각 불가** ⟹ 층별 라벨 금지',
             '⚠⚠AUTO-BAND 규칙은 **분해능 한계 근처에서 동작한다** — 2단 줌의 1단이 '
             '절대 격자(klim=2.0, n=120 ⟹ Δk=0.0336)이고 이 코퍼스의 k_p≈0.043 은 그 '
             '**1.3배**뿐이다. 등변성 시험에서 c=0.5(장면 k 를 절반으로)이면 규칙이 무너진다'
             '(편차 ~99%). ⛔BAND 결과를 인용할 때 이 사실을 반드시 병기하라 (G-BAND-FRAGILE)',
             '⚠SW-2.828 이 s_ap≈2.81 과 0.7% 차로 우연히 겹친다(√2 격자는 스케일-자유라서 '
             '골랐고 규칙에서 유도하지 않았다) ⛔스윕을 규칙의 증거로 쓰지 마라',
             '⛔규칙은 결과 0행 시점에 동결됐다 — K_TRAIN 의 중심 통계(기하평균)나 L_TRAIN 의 '
             '정의(최대 변)를 사후에 바꾸는 것은 금지. 새 규칙은 **새 사전등록으로만**',
             '⚠깊이 미재척도: 엄밀히는 d\'=d/s 여야 ω=√(g k tanh(kd)) 가 정확히 보존되나 '
             'T11 앵커를 지키려고 depth=400 을 고정했다. G-DEEP 이 k̂_p·d ≥ π 를 **판정 자료 '
             '위에서** 확인하고 G-INVERT 가 **필드 차이**로 잰다(분산관계 상대오차가 아니다)',
             '⚠실측 경로: 재라벨은 initialize() **상류의 좌표 변환**이라 _frames_have_gt 에 '
             '보이지 않는다 ⟹ 라우팅 불변(G-ARM 이 되읽기로 증명). ⛔real_domain_gate 를 끄지 '
             '않았다(W-12·D3 재발 방지). ★진짜 실측에서는 bank_mmse 가 이미 k̂_p 상대 뱅크라 '
             '자동정규화가 거의 no-op 일 것 — ⛔이 배치는 그것을 재지 않았고 결과로도 인용 금지',
             '⛔E2 와 대각선 비교 금지 · 저들 논문 수치와 비교 금지 · solver 미명시',
             '⛔wall_s 를 팔 간 비교에 쓰지 마라 · proposal_device 는 config 가 정하지 않는다',
             '⚠Tier-3(OOD 6)은 SEC 이고 **만장일치 전용**이다(n=6 ⟹ 6/6 만 p=0.0156)',
             '⚠형제 배치 충돌: T12·T13 이 같은 FRESH-28 을 이미 썼다 ⟹ Tier-1 의 SHIP·FIX2 '
             '앵커는 **노출된 값**이다. AUTO-* 두 팔은 미노출(blind)',
         ]}
    json.dump(v, open(os.path.join(OUTDIR, 'verdict_t14.json'), 'w'),
              ensure_ascii=False, indent=1)
    def _p(k, n):                       # ★n=0 에서도 죽지 않는다(층이 비면 게이트가 잡는다)
        q = G.sign_p(k, n)
        return 'n/a' if q is None else f'{q:.4g}'
    print(f"T14-AUTOSCALE :: {v['label_msg']}")
    print(f"  PRIMARY (문턱 {G.THRESH_PRIMARY}, Bonferroni α/2)")
    print(f"     AUTO-AP   < ZERO {w_ap}/{n_ap}  p={_p(w_ap, n_ap)}")
    print(f"     AUTO-BAND < ZERO {w_bd}/{n_bd}  p={_p(w_bd, n_bd)}")
    print(f"  양성대조 FIX2<ZERO {w_fx}/{n_fx} p={_p(w_fx, n_fx)} (문턱 {G.THRESH_CTRL})")
    print(f"  하한   SHIP<ZERO {w_sh}/{n_sh} p={_p(w_sh, n_sh)}")
    print(f"  회수율 R {v['recovery_R']}")
    print(f"  s 분포 {svar} · 레일접촉 {sclip}")
    print(f"  스윕(기술) {sw}")
    print(f"  Tier-3 SEC {t3w}")
    print(f"  앵커 등급 {grade} · 클램프 {clamp} · device {v['proposal_device']}")
    print(f"  gates {v['gates_pass']}")
    for n, o, d in A:
        print(f"   [{'OK ' if o else 'FAIL'}] {n:76s} {d}")


if __name__ == '__main__':
    main()
