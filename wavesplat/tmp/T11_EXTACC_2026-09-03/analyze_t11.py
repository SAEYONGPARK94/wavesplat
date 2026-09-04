"""`T11-EXTACC` **v2 + 개정문 2** 분석기 — ⛔러너보다 먼저 쓰인다. 이 파일이 **행 스키마를 정의**한다.

## 러너가 채워야 할 행 (여기서 정의한다 — ⛔초판 독스트링은 자신이 읽는 키 4개를 빠뜨렸다)
  arm, block(★아카이브 **전체 경로** — basename 은 두 디렉터리에서 충돌한다)
  angle(방향 퍼짐 폭), hs, nrmse, sigma4(정규화에 쓴 4σ), frame_index, seed, wall_s
  n_offered, n_gathered(★래핑해 **실제로 센** 값), n_consumed_formula(⛔산술 — 게이트 아님)
  cap_proposal, t_origin, gt_std, gt_value, gt_std_after(★initialize **뒤** 되읽기)
  proposal_md5(★raw 제안 원자 지문 — `G-ATOMS` 가 읽는다), atoms_md5(설치 원자, 참고용)
  prov(arm_provenance 통째), n_atoms_installed, vp_clamp_hits(★`G-CLAMP`)
  rescale_s, x_scale, t_scale, disp_relerr, proposal_device(★되읽기)
  dt, tq, e_shape, n_frames, density, mean_convention, axis_convention
"""
import json, os, sys, zipfile
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_t11 as G                                                # noqa: E402

ROWS = os.path.join(HERE, 'results_t11.jsonl'); META = os.path.join(HERE, 'meta_t11.json')
FROZEN = os.path.join(HERE, 'FROZEN_md5.txt')          # 결과 0행 동결 (측정면 증거물)
FROZEN_A3 = os.path.join(HERE, 'FROZEN_md5_A3.txt')    # 개정문 3 동결 (분석면)
OUTDIR = HERE          # ★verdict 출력만 분리(스모크 하네스가 HERE 를 안 흔들게)


def sign_p(k, n):
    return sum(comb(n, i) * 0.5 ** n for i in range(k, n + 1)) if n else None


def p_ge(k, n, q):
    return sum(comb(n, i) * q ** i * (1 - q) ** (n - i) for i in range(k, n + 1))


def label_probs(qr, qs):
    """★사전등록 예측 하의 4라벨 확률 — AMENDMENT_2 §2 (⛔초판은 계산하지 않았다)."""
    a, b = p_ge(G.THRESH_WIN, G.N_FILES, qr), p_ge(G.THRESH_WIN, G.N_FILES, qs)
    return {'BAND-MISMATCH': a * (1 - b), 'BOTH-WORK': a * b,
            'NEITHER-DETECTED': (1 - a) * (1 - b), 'RESCALE-HURTS': (1 - a) * b}


def main():
    rows = [json.loads(x) for x in open(ROWS, encoding='utf-8') if x.strip()]
    meta = json.load(open(META, encoding='utf-8'))
    by = {a: {r['block']: r for r in rows if r['arm'] == a} for a in G.ARMS}
    zero = by['ZERO']

    def wins(arm):
        """★팔별로 `arm ∩ ZERO` — ⛔초판은 PRIMARY(RESCALE) 분모가 SHIP 에 물려 있었다."""
        d = by[arm]
        cm = sorted(set(d) & set(zero))
        return sum(1 for b in cm if d[b]['nrmse'] < zero[b]['nrmse']), len(cm)

    w_ship, n_ship = wins('SHIP')
    w_res, n_res = wins('RESCALE')
    w_ora, n_ora = wins('ORACLE')

    A = []
    # ★분석기가 **직접** 계산한다 (⛔러너 리터럴 금지)
    sz = os.path.getsize(G.ZIP) if os.path.exists(G.ZIP) else -1
    A.append((f'★G-SIZE 분석기가 직접 잰 크기 == {G.WANT_BYTES}',
              sz == G.WANT_BYTES, str(sz)))

    # ── G-SAMPLE : 표본 == 행 · 아카이브 실존 · 선행배치/사전노출과 교집합 0 ──────
    want = set(G.sample())
    got = {r['block'] for r in rows}
    prev, missing_ref = set(), []
    for p, k in (('T1_THIRDPARTY_2026-09-03/meta_t1.json', 'picked'),
                 ('T10_EXTDISP_2026-09-03/meta_t10.json', 'picked')):
        f = os.path.join(HERE, '..', p)
        if os.path.exists(f):
            prev |= set(json.load(open(f, encoding='utf-8')).get(k, []))
        else:
            missing_ref.append(p)                    # ⛔초판은 부재 시 **통과**했다
    f = os.path.join(HERE, '..', 'T1B_THIRDPARTY_2026-09-03', 'results_t1b.jsonl')
    if os.path.exists(f):
        prev |= {json.loads(x)['block'] for x in open(f, encoding='utf-8')
                 if x.strip() and 'block' in x}
    else:
        missing_ref.append('T1B_THIRDPARTY_2026-09-03/results_t1b.jsonl')
    try:
        exposed, mats = G.exposed_blocks()          # ★zip namelist 를 직접 연다
        not_in_archive = sorted(want - set(mats))
    except Exception as e:                                            # noqa: BLE001
        exposed, mats, not_in_archive = [], [], [f'ERROR:{type(e).__name__}']
    A.append(('★G-SAMPLE 표본==행 · 아카이브 실존 · 선행배치/사전노출 교집합 0 [namelist 직접]',
              got == want and len(want) == G.N_FILES and not (want & prev)
              and not (want & set(exposed)) and not not_in_archive and not missing_ref,
              f'표본 {len(want)} · 행 {len(got)} · 선행 {len(want & prev)} · '
              f'노출 {len(want & set(exposed))}(후보 {len(exposed)}) · '
              f'미실존 {len(not_in_archive)} · 참조부재 {len(missing_ref)}'))

    # ── G-N : 문턱 17 은 n=24 **전용**이다 (v1 에 있다가 v2 에서 소실됐던 게이트) ──
    per_arm = {a: len(by[a]) for a in G.ARMS}
    A.append((f'★G-N 팔별 {G.N_FILES}행 · 총 {4 * G.N_FILES}행 (문턱 17 은 n=24 전용)',
              all(v == G.N_FILES for v in per_arm.values())
              and len(rows) == 4 * G.N_FILES
              and n_res == n_ship == n_ora == G.N_FILES,
              f'{per_arm} · rows={len(rows)} · n=({n_res},{n_ship},{n_ora})'))

    # ── G-FINITE : NaN 은 `<` 에서 False 가 되어 조용히 「패배」로 집계된다 ────────
    nonfin = [r['block'] for r in rows
              if r.get('nrmse') is None or not np.isfinite(r['nrmse'])]
    A.append(('★G-FINITE 모든 행의 nRMSE 가 유한 (⛔NaN 은 조용히 패배가 된다)',
              not nonfin, f'비유한 {len(nonfin)}'))

    # ★되읽기: 러너가 **만들어진 프레임에서 잰** std/mean 이다(⛔리터럴 0.0 이 아니다).
    #   ★개정문 2: `initialize()` **뒤에도** 한 번 더 되읽는다.
    leak = [r for r in rows if r['arm'] in ('SHIP', 'RESCALE')
            and (r.get('gt_std') is None or r.get('gt_value') is None
                 or r.get('gt_std_after') is None
                 or abs(r['gt_std']) > 1e-12 or abs(r['gt_std_after']) > 1e-12
                 or abs(r['gt_value'] - G.GT_CONST) > 1e-12)]
    A.append(('★★G-NOLEAK 붙인 gt 가 상수(std 0) · 값 == GT_CONST · initialize 후에도 [되읽기]',
              not leak, f'위반 {len(leak)}'))

    # ── G-ATOMS : ⛔초판은 **설치** 원자를 해싱해 둔감했다 ─────────────────────
    #   VARPRO 400스텝이 kx/ky 를 파일별 관측으로 재적합해 덮어쓰므로, 제안 캐시가
    #   **완전히 새더라도** 설치 지문은 전부 고유해진다 ⟹ 항상 통과.
    #   이제 `cfg['learned_atoms']`(raw 제안)을 해싱한 `proposal_md5` 를 본다.
    # ★A3 §7: 초판은 SHIP 만 봤다 — RESCALE 지문이 중복이어도 통과했다.
    fpd = {a: [r.get('proposal_md5') for r in rows if r['arm'] == a]
           for a in ('SHIP', 'RESCALE')}
    fpok = all(len(v) == len(set(v)) and all(v) for v in fpd.values())
    A.append(('★★G-ATOMS SHIP·RESCALE **제안** 원자 지문이 각각 전건 상이 (캐시 누출 탐지)',
              fpok, ' · '.join(f'{a} {len(set(v))}/{len(v)}' for a, v in fpd.items())))

    # ── G-CAP : ⛔「소비」 항은 **산술 항등식**이라 발화 불가였다 — 뺐다 ─────────
    #   살아있는 내용은 둘이다: (1) `n_gathered` 는 `_gather_init_obs` 를 래핑해 실제로
    #   센 값이고 그것이 `None` 이면 **제안 단계가 안 돌았다는 뜻**(체크포인트 부재 시
    #   `v2_konly` 폴백), (2) 행의 캡 == 지금의 배포 캡(드리프트 탐지).
    cap_now = G.deployed_cap()
    capbad, capdrift = [], []
    for r in rows:
        if r['arm'] not in ('SHIP', 'RESCALE'):
            continue
        if r.get('cap_proposal') != cap_now:
            capdrift.append(r['block'])
        g = r.get('n_gathered')
        if g is None or g != r.get('n_offered'):
            capbad.append(r['block'])
    A.append((f'★★G-CAP 제안 단계가 실제로 돌았다(모음==제공) · 캡=={cap_now} [되읽기]',
              not capbad and not capdrift,
              f'위반 {len(capbad)} · 캡드리프트 {len(capdrift)}'))

    # ── G-ARM : 팔 정체 (v1 에 있다가 소실됐다) ───────────────────────────────
    #   ⛔`is_ensemble_wrapper` 는 `auto` 면 **무조건 True** 라 그것만으로는 아무것도 안 잰다.
    k_dep = G.deployed_ensemble_k()
    armbad = []
    for r in rows:
        if r['arm'] not in ('SHIP', 'RESCALE'):
            continue
        pv = r.get('prov') or {}
        if (pv.get('resolved_init_method') != 'learned'
                or pv.get('real_domain_gate_fired') is not False
                or pv.get('ensemble_k') != k_dep):
            armbad.append(r['block'])
    A.append((f'★★G-ARM resolved==learned · 실측게이트 미발화 · ensemble_k=={k_dep} [되읽기]',
              not armbad, f'위반 {len(armbad)}'))

    wrap = [r for r in rows if r['arm'] in ('SHIP', 'RESCALE')
            and not (r.get('prov') or {}).get('is_ensemble_wrapper')]
    A.append(('G-WRAPPER 출하 진입점(make_updater) [⚠A3 §9: 배포 상수상 **구조적으로 참**]',
              not wrap,
              f'위반 {len(wrap)}'))

    # ── G-CLAMP : z 를 재척도하지 않아 파고경사가 2배 ⟹ RESCALE 만 클램프에 닿을 수 있다 ──
    # ★★스모크가 `AMENDMENT_2 §6c` 의 **예측을 반증했다**(개정문 §9 에 공시):
    #   축⑤는 *"z 미재척도로 경사가 2배 ⟹ RESCALE 만 클램프에 닿는다"* 라 예측했는데
    #   실측은 **반대**다 — `SHIP` 이 1·2회 발화하고 `RESCALE` 은 0·0 이다. 팔이 서로
    #   다른 원자를 적합하므로 「자료의 경사」로 한 예측이 뒤집혔다(저장소 상시 교훈:
    #   ***효과를 그것이 들어가는 양에서 재지 말고 판정에서 재라***).
    #   ⟹ 클램프는 **출하 추정기의 정상 동작**이다. 발화로 배치를 막지 않는다.
    #   블록 조건은 **되읽기 부재**뿐이고, 비대칭 자체는 verdict + 공시로 기록한다.
    clamp_by_arm = {}
    for r in rows:
        if r['arm'] in ('SHIP', 'RESCALE'):
            clamp_by_arm.setdefault(r['arm'], []).append(r.get('vp_clamp_hits'))
    nocnt = [r['block'] for r in rows
             if r['arm'] in ('SHIP', 'RESCALE') and r.get('vp_clamp_hits') is None]
    fired = {a: sum(1 for v in vs if (v or 0) > 0) for a, vs in clamp_by_arm.items()}
    A.append(('★G-CLAMP 경사 클램프 발화 **기록**됨 (⛔발화로 막지 않는다 — 개정문 §9)',
              not nocnt, f'발화 팔별 {fired} · 미기록 {len(nocnt)}'))

    tbad = [r for r in rows
            if r.get('t_origin') is None or abs(r['t_origin']) > 1e-9]
    A.append(('★G-TORIGIN 창 시작 t == 0 (원점 자유도 봉쇄)', not tbad,
              f'위반 {len(tbad)}'))

    # ★`disp_relerr` 는 **실제로 적용된 배율을 나눠서** 잰 것이다(x_scale·t_scale).
    rbad = []
    for r in rows:
        if r['arm'] != 'RESCALE':
            continue
        a, b = r.get('x_scale'), r.get('t_scale')
        if a is None or b is None or r.get('disp_relerr') is None \
                or abs((r.get('rescale_s') or 0.0) - G.RESCALE_S) > 1e-12 \
                or abs(a - 1.0 / G.RESCALE_S) > 1e-9 \
                or abs(b - 1.0 / np.sqrt(G.RESCALE_S)) > 1e-9 \
                or r['disp_relerr'] > 1e-9:
            rbad.append(r['block'])
    xs = [r['x_scale'] for r in rows if r['arm'] == 'RESCALE' and r.get('x_scale')]
    A.append((f'★G-RESCALE 측정 배율 x·t 가 s={G.RESCALE_S} 와 일치 · ω=√(gk) 보존 [되읽기]',
              not rbad, f'위반 {len(rbad)} · x_scale 중앙 '
                        f'{(float(np.median(xs)) if xs else float("nan")):.6f}'))

    # ── G-ENV : ⛔초판은 3/3 항진이었다 (러너 리터럴 == 분석기 리터럴, md5 는 bool()) ──
    #   ★A3 §2: 개정문 2 는 **셋 중 둘만** 고쳤다 — 시드는 여전히 `G.SEED` 상수 왕복이었다.
    #   이제 (a) 행의 `seed` 열을 **집합으로** 되읽고(러너가 블록마다 다르게 쓴 값이다),
    #   (b) 동결을 **측정면/분석면**으로 갈라 대조한다. 측정면 3중 일치가 곧
    #   *"개정문 3 은 측정을 한 비트도 안 건드렸다"* 의 기계 증명이다.
    def _read_frozen(path):
        out = {}
        if not os.path.exists(path):
            return None
        for line in open(path, encoding='utf-8'):
            if line.strip() and not line.startswith('#'):
                h, k = line.split(None, 1)
                out[k.strip()] = h.strip()
        return out

    now = G.freeze_map()
    rec = meta.get('code_md5') or {}
    fz0, fz3 = _read_frozen(FROZEN), _read_frozen(FROZEN_A3)
    run_keys = sorted(k for k in now if k not in G.ANALYSIS_SIDE)
    ana_keys = sorted(k for k in now if k in G.ANALYSIS_SIDE)
    run_vs_meta = [k for k in run_keys if now[k] != rec.get(k)]
    run_vs_fz0 = None if fz0 is None else [k for k in run_keys if now[k] != fz0.get(k)]
    ana_vs_fz3 = None if fz3 is None else sorted(
        k for k in set(ana_keys) | set(fz3) if now.get(k) != fz3.get(k))
    want_seeds = {G.SEED + i for i in range(1, G.N_FILES + 1)}
    seed_bad = sorted(a for a in G.ARMS
                      if {r.get('seed') for r in rows if r['arm'] == a} != want_seeds)
    A.append(('G-ENV 스레드[되읽기] · **행 시드 집합 되읽기** · 측정면 3중 동결 · 분석면 동결',
              meta.get('threads') == 1 and meta.get('threads_readback') == 1
              and not seed_bad and bool(rec) and not run_vs_meta
              and run_vs_fz0 == [] and ana_vs_fz3 == [],
              f"threads={meta.get('threads_readback')} 시드위반팔 {len(seed_bad)} · "
              f"측정면(런대비 {len(run_vs_meta)} · 0행동결대비 "
              f"{'N/A' if run_vs_fz0 is None else len(run_vs_fz0)}) · 분석면 "
              f"{'N/A' if ana_vs_fz3 is None else len(ana_vs_fz3)}"))

    # ── 사후 점검 (⛔A3 §0: 결과를 본 뒤 추가했다. 막을 수만 있고 열 수는 없다) ──
    dens = []
    for r in rows:
        if r.get('n_offered') != G.DENSITY * int(G.WIN_S):
            dens.append(r['block']); continue
        es = r.get('e_shape') or [0]
        if es[0] < G.J0 + int(G.WIN_S):
            dens.append(r['block']); continue
        dt, tq = r.get('dt'), r.get('tq')
        if dt is None or tq is None or abs(tq - (int(G.WIN_S) - 1) * dt) > 1e-9:
            dens.append(r['block']); continue
        if r['arm'] in ('SHIP', 'RESCALE') and r.get('n_frames') != int(G.WIN_S):
            dens.append(r['block'])
    A.append((f'★G-DENSITY [사후] 제공=={G.DENSITY}x{int(G.WIN_S)} · 창이 자료 안 · '
              f'tq==(N-1)dt [되읽기]', not dens, f'위반 {len(dens)}'))
    try:
        mats = G.archive_mats()
        pop = set(mats[G.POP_RANGE[0]:G.POP_RANGE[1]])
        outside = sorted(set(G.sample()) - pop)
    except Exception as e:                                            # noqa: BLE001
        outside = [f'ERROR:{type(e).__name__}']
    A.append((f'★G-POP [사후] 표본이 PREREG §4 모집단(mixed_data2 = idx '
              f'{G.POP_RANGE[0]}-{G.POP_RANGE[1] - 1}) 안', not outside,
              f'범위 밖 {len(outside)}'))

    probe = G.blocked_by([('★★G-NOLEAK 합성', False, ''), ('G-N 합성', False, ''),
                          ('G-ENV 합성', True, '')])
    A.append(('G-BLOCK blocked_by 자가진단 (긴 이름 우선)',
              probe == ['G-NOLEAK', 'G-N'], f'합성 → {probe}'))
    A.append(('G-COV 게이트가 자료를 봤다', len(rows) > 0, f'rows={len(rows)}'))

    blocked = G.blocked_by(A)
    # ★A3 §6 벨트: `blocked_by()` **하나**에만 매달리지 않는다. V3 가 합성으로 그 함수를
    #   무력화하니 `G-BLOCK` 이 FAIL 인데도 라벨이 인쇄됐다. 막을 수만 있는 변경이다.
    all_ok = all(bool(o) for _, o, _ in A)
    lab = G.label(w_res, w_ship, w_ora) if (not blocked and all_ok) else None

    # ── 기술통계 (⛔라벨을 만들지 않는다 — 다중성) ─────────────────────────────
    strata = {}
    for nm, pred in (('hs_low', lambda h: h <= G.HS_SPLIT),
                     ('hs_high', lambda h: h > G.HS_SPLIT)):
        for arm in ('SHIP', 'RESCALE', 'ORACLE'):
            d = by[arm]
            cm = [b for b in sorted(set(d) & set(zero)) if pred(d[b]['hs'])]
            strata[f'{arm}_{nm}'] = {
                'w': sum(1 for b in cm if d[b]['nrmse'] < zero[b]['nrmse']), 'n': len(cm)}
    clusters = {}
    for arm in ('SHIP', 'RESCALE'):
        d = by[arm]
        agg = {}
        for b in sorted(set(d) & set(zero)):
            agg.setdefault(d[b]['angle'], [0, 0])
            agg[d[b]['angle']][1] += 1
            if d[b]['nrmse'] < zero[b]['nrmse']:
                agg[d[b]['angle']][0] += 1
        clusters[arm] = {str(k): {'w': v[0], 'n': v[1]} for k, v in sorted(agg.items())}

    # ⛔n != 24 이면 판정 불가로 둔다 — 작은 n 에서 `w<=7` 은 자동 True 가 되어 항진이다
    sig_worse = {a: ((w <= G.SIG_WORSE) if n == G.N_FILES else None) for a, w, n in
                 (('SHIP', w_ship, n_ship), ('RESCALE', w_res, n_res),
                  ('ORACLE', w_ora, n_ora))}

    v = {'batch': 'T11-EXTACC', 'kind': 'CONFIRMATORY', 'grade': 'CONFIRMATORY',
         'prereg': 'v2 + AMENDMENT_2_PREDATA.md (결과 0행 시점 개정)',
         'new_runs': meta.get('n_runs'), 'n_rows': len(rows),
         'gates': [{'name': nm, 'pass_': bool(o), 'detail': d} for nm, o, d in A],
         'gates_pass': f'{sum(1 for _, o, _ in A if o)}/{len(A)}', 'blocked_by': blocked,
         'primary': {'stat': 'RESCALE < ZERO', 'w': w_res, 'n': n_res,
                     'p_one_sided': sign_p(w_res, n_res)},
         'sec_a_ship': {'w': w_ship, 'n': n_ship, 'p_one_sided': sign_p(w_ship, n_ship)},
         'sec_b_oracle': {'w': w_ora, 'n': n_ora, 'p_one_sided': sign_p(w_ora, n_ora)},
         'sig_worse_w_le_7': sig_worse,
         'strata': strata, 'angle_clusters': clusters,
         'clamp_hits_by_arm': clamp_by_arm,
         'medians': {a: float(np.median([r['nrmse'] for r in by[a].values()]))
                     for a in G.ARMS if by[a]},
         'mde': {'alpha': 0.05, 'one_sided': True, 'n': G.N_FILES,
                 'crit_w': G.THRESH_WIN, 'q50': G.MDE_Q50, 'q80': G.MDE_Q80,
                 'q90': G.MDE_Q90, 'unrejectable_w': list(G.UNREJECTABLE)},
         'prediction': {'q_rescale': G.PRED_Q[0], 'q_ship': G.PRED_Q[1],
                        'label_probs': label_probs(*G.PRED_Q),
                        'null_label_probs': label_probs(0.5, 0.5)},
         'proposal_device': sorted({r.get('proposal_device') for r in rows
                                    if r.get('proposal_device')}),
         'post_result_checks': ['G-DENSITY', 'G-POP', 'G-ENV(시드·분석면 동결)',
                                'G-ATOMS(RESCALE 확장)', '라벨 인쇄 벨트 all(ok)'],
         'amendments': ['AMENDMENT_1_PREDATA.md', 'AMENDMENT_2_PREDATA.md',
                        'AMENDMENT_3_POSTRESULT.md (결과 이후 · 분석기 전용 · 새 런 0회)'],
         'label': lab, 'label_msg': lab or 'BLOCKED',
         'disclosures': [
             '⛔BAND-MISMATCH 는 **귀속**이지 "우리가 낫다/못하다" 가 아니다',
             '⛔E2 와 대각선 비교 금지 — 읽는 것은 "각자 남의 자료에서 어느 밀도부터" 다 '
             '(PREREG §6 의무 공시)',
             '⛔저들 논문의 수치와 비교 금지(채점·격자·기준이 다르다) (PREREG §6 의무 공시)',
             '⛔solver 미명시 — "HOS"·"완전 비선형" 이라 쓰지 않는다 (PREREG §6 의무 공시)',
             '⚠예산 비대칭: 제안 단계는 프레임당 8,000 관측 중 캡 400 만 쓴다 — '
             '"고른 400" 과 "있는 것 전부" 는 다르다 (AMENDMENT_1 §6.4)',
             '⛔RESCALE 을 실제 성능으로 인용 금지 — 단위를 바꾼 사고실험이다',
             f'⛔음성 라벨(NEITHER-DETECTED / NO-INFORMATION-IN-OBS)은 **검출 불가**이지 '
             f'폐쇄가 아니다 — 인용 시 MDE q80={G.MDE_Q80} 병기 의무 '
             f'(w={G.UNREJECTABLE[0]}..{G.UNREJECTABLE[1]} 은 어떤 자료로도 기각 불가)',
             '⛔SEC-A·SEC-B 의 p 는 **무보정**이다 · 층(SEC-C)과 각도 6클러스터는 '
             '**기술통계**이고 라벨을 만들지 않는다 (각도 클러스터는 6/6 만장일치일 때만 주장 가능, P=0.0156)',
             '⛔ORACLE 은 추정기가 아니다(참 파수를 받는다) — AMENDMENT_2 §5 로 '
             '±ω 4열 + 켤레쌍 정준키로 재정의됐다. ⛔초판(±k 임의 대표, +ω 만)은 상한이 '
             '아니라 **하한**이었다',
             '⚠RESCALE 은 **단일요인이 아니다**(AMENDMENT_2 §6): 배포 경로의 절대 파수 '
             '상수 전부(사전 밴드·learned_dedup_dk 0.03·MIN_DK 0.15·성장 DC블록 0.133·'
             'VARPRO lr 2e-3)와 조건화 맵의 개구/픽셀 비(PSF 0.53→1.06칸, 학습 1.37–1.65)가 '
             's 와 **공선**으로 함께 움직인다 ⟹ 「학습 사전의 파장 밴드」 단독 귀속 불가. '
             '정본 읽기 = **"배포 추정기의 절대 스케일 설계 범위"**. k_p 기준 장면은 이미 '
             '학습 밴드 안이다(Tp 10 s ∈ U(5,11), 83백분위)',
             '⚠scripts/d3_beamform_net.py:**30** 이 DEPTH=50 하드코딩 ⟹ config depth 가 '
             '조건화 맵에 **한 번도 닿지 않는다**. 학습 코퍼스도 d=50 생성 ⟹ **SHIP 만** '
             '디포커스된다(k=0.03 에서 창 위상 1.03 rad, RESCALE 0.05 rad) = **비대칭 교락**. '
             '⛔이 배치에서 고치지 않는다(출하 팔이 아니게 된다)',
             '⚠완전 blind 아님 — 사전감사 스모크 8 + 러너 스모크 2 파일이 설계에 영향'
             '(표본에서 배제하고 G-SAMPLE 이 기계로 확인한다)',
             '⚠Tp 가 전 파일 10 s ⟹ 한 주기 조건의 진술이다',
             '⚠torch.set_num_threads(1) 은 numpy/BLAS 를 제약하지 않는다 ⟹ '
             '⛔wall_s 를 팔 간 비교에 쓰지 마라',
             '⚠제안 단계의 device 는 config 가 정하지 않는다(propose_atoms 가 '
             'cuda-if-available 로 자기가 고른다) — 행의 proposal_device 를 볼 것',
             '⚠사전등록 예측 label_probs 의 NEITHER-DETECTED 0.2136 은 실제로 '
             '**NEITHER-DETECTED + NO-INFORMATION-IN-OBS 의 합**이다 — D-3 가 ORACLE 을 '
             '사다리에 넣으면서 §2 예측에는 넣지 않았다. ⛔둘의 분해는 **미등록**이고 '
             'q_oracle 을 사후에 정하지 않는다 (AMENDMENT_3 §5)',
             '⚠AMENDMENT_3 은 **결과 이후** 개정이다 — post_result_checks 의 게이트들은 '
             '결과를 본 뒤 추가됐다. ⛔사전등록 차단 게이트로 세지 마라. '
             '(막을 수만 있고 열 수는 없으며, run_t11.py·임포트·체크포인트는 무수정)',
             '⚠경사 클램프 비대칭: AMENDMENT_2 §6c 는 RESCALE 이 닿을 것이라 **예측했으나 '
             '실측은 반대**다(SHIP 발화, RESCALE 0) — clamp_hits_by_arm 을 볼 것. '
             '⛔이 비대칭을 성능 차의 원인으로 지목하지 마라(미측정)',
         ]}
    json.dump(v, open(os.path.join(OUTDIR, 'verdict_t11.json'), 'w'),
              ensure_ascii=False, indent=1)
    print(f"T11-EXTACC :: {v['label_msg']}")
    print(f"  PRIMARY RESCALE<ZERO {w_res}/{n_res}  p={v['primary']['p_one_sided']}")
    print(f"  SEC-A   SHIP   <ZERO {w_ship}/{n_ship}  p={v['sec_a_ship']['p_one_sided']}")
    print(f"  SEC-B   ORACLE <ZERO {w_ora}/{n_ora}  p={v['sec_b_oracle']['p_one_sided']}")
    print(f"  유의 열세(w<=7) {sig_worse}")
    print(f"  중앙 nRMSE {v['medians']}")
    print(f"  층 {strata}")
    print(f"  각도 {clusters}")
    print(f"  MDE q80={G.MDE_Q80} · 기각불가 w={G.UNREJECTABLE}")
    print(f"  제안 device {v['proposal_device']}")
    print(f"  gates {v['gates_pass']}")
    for nm, o, d in A:
        print(f"   [{'OK ' if o else 'FAIL'}] {nm:64s} {d}")


if __name__ == '__main__':
    main()
