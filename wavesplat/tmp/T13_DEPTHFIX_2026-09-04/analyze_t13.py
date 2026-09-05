"""`T13-DEPTHFIX` 분석기 — ⛔러너보다 **먼저** 쓰인다. 이 파일이 **행 스키마를 정의**한다.

## 러너가 채워야 할 행
  block · arm · layer('T11-24'|'FRESH-28') · angle · hs · nrmse · sigma4 · seed · wall_s
  n_offered · n_frames · dt · tq · e_shape · t_origin · gt_std · gt_value · gt_std_after
  ★map_om_md5 · map_om_row(16셀) · map_depth_seen · bmap_calls   ← G-MAP/G-BMAPCALLS
  depth_config(upd.depth 되읽기) · proposal_md5 · proposal_device · n_gathered · cap_proposal
  n_atoms_installed · vp_clamp_hits · prov · rescale_s · x_scale · t_scale · disp_relerr
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_t13 as G                                             # noqa: E402

ROWS = os.path.join(HERE, 'results_t13.jsonl'); META = os.path.join(HERE, 'meta_t13.json')
FROZEN = os.path.join(HERE, 'FROZEN_md5.txt')        # 결과 0행 동결 (측정면 증거물)
FROZEN_A2 = os.path.join(HERE, 'FROZEN_md5_A2.txt')  # 개정문 2 동결 (분석면)
SMOKE_META = os.path.join(HERE, 'meta_t13_SMOKE.json')
ANALYSIS_SIDE = ('gates_t13.py', 'analyze_t13.py')
OUTDIR = HERE


def om_row(depth, klim=None, nk=None):
    """★`OM_G` 를 **공식으로 독립 재계산** — 러너가 적은 값을 되읽지 않는다 (G-MAP).

    `√(g·k·tanh(k·d))`, `g` 는 `model/dispersion.G` 를 **임포트**한다(⛔손복사 금지).
    """
    import d3_beamform_net as bn
    from wavesplat.model.dispersion import G as g
    klim = bn.KLIM if klim is None else klim
    nk = bn.NK if nk is None else nk
    ax = np.linspace(-klim, klim, nk)
    KX, KY = np.meshgrid(ax, ax, indexing='xy')
    k = np.sqrt(KX ** 2 + KY ** 2) + 1e-9
    om = np.sqrt(g * k * np.tanh(k * depth))
    return om[nk // 2, ::4]


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
    by = {}
    for r in rows:
        by.setdefault(r['arm'], {})[r['block']] = r
    t11b, fresh = G.layers()

    def paired(a, b, blocks=None):
        da, db = by.get(a, {}), by.get(b, {})
        cm = sorted(set(da) & set(db) & (set(blocks) if blocks else set(da)))
        return sum(1 for x in cm if da[x]['nrmse'] < db[x]['nrmse']), len(cm)

    w_fs, n_fs = paired('FIX', 'SHIP')
    w_fz, n_fz = paired('FIX', 'ZERO')
    w_sz, n_sz = paired('SHIP', 'ZERO')
    w_ctrl, n_ctrl = paired('RESCALE-FIX', 'RESCALE')
    A = []

    # ── ★★G-MAP : 처치가 **소비자에게** 도달했다 ───────────────────────────
    mbad, seen_d = [], {}
    for r in rows:
        if r['arm'] not in ('SHIP', 'FIX', 'RESCALE', 'RESCALE-FIX'):
            continue
        got = r.get('map_om_row')
        dcfg = r.get('depth_config')
        if got is None or dcfg is None:
            mbad.append(f"{r['arm']}:missing"); continue
        got = np.asarray(got, float)
        ok50 = np.allclose(got, om_row(50.0), atol=1e-5)
        okcfg = np.allclose(got, om_row(float(dcfg)), atol=1e-5)
        want = (okcfg and not ok50) if r['arm'] in G.FIX_ARMS else (ok50 and not okcfg)
        seen_d.setdefault(r['arm'], set()).add(round(float(r.get('map_depth_seen', -1)), 3))
        if not want:
            mbad.append(f"{r['arm']}:{r['block'][-12:]}")
    A.append(('★★G-MAP 소비자가 본 OM_G 를 **공식으로 독립 재계산** — FIX=d_config · SHIP=d50',
              not mbad, f'위반 {len(mbad)} · 팔별 본 수심 {seen_d}'))
    cbad = [r['arm'] for r in rows if r['arm'] in ('SHIP', 'FIX', 'RESCALE', 'RESCALE-FIX')
            and r.get('bmap_calls') != 1]
    A.append(('★★G-BMAPCALLS 런당 맵 계산 정확히 1회 (0=제안 미실행 · ≥2=예상 밖 경로)',
              not cbad, f'위반 {len(cbad)}'))

    # ── ★★앵커 (T11 비트 재현) ────────────────────────────────────────────
    t11r = {}
    if os.path.exists(G.RESULTS_T11):
        for line in open(G.RESULTS_T11, encoding='utf-8'):
            if line.strip():
                d = json.loads(line)
                t11r[(d['arm'], d['block'])] = d
    for arm in ('SHIP', 'RESCALE'):
        bad, npair = [], 0
        for b, r in by.get(arm, {}).items():
            ref = t11r.get((arm, b))
            if ref is None:
                continue
            npair += 1
            if r['nrmse'] != ref['nrmse'] or r.get('proposal_md5') != ref.get('proposal_md5'):
                bad.append(b)
        A.append((f'★★G-ANCHOR-{arm} T11 24행 비트 재현 (nrmse 등식 + proposal_md5 등식)',
                  npair == G.N_T11 and not bad, f'짝 {npair}/{G.N_T11} · 불일치 {len(bad)}'))

    dev = sorted({r.get('proposal_device') for r in rows if r.get('proposal_device')})
    A.append(('★★G-DEVICE proposal_device 가 전 팔·전 행 동일 [되읽기]',
              len(dev) == 1, f'{dev}'))
    ood = [r['block'] for r in rows if 'outofdistribution' in r['block']]
    A.append(('★G-OOD-UNTOUCHED held-out 층(OOD 6) 미접촉', not ood, f'접촉 {len(ood)}'))

    # ── 표본·층 ────────────────────────────────────────────────────────────
    want = set(G.sample()); got = {r['block'] for r in rows}
    mats = G.archive_mats()
    A.append(('★G-SAMPLE 행 == (모집단 − 감사스모크) **집합 차** · 52 · T11-24 부분집합 일치',
              got == want and len(want) == G.N_TOTAL and set(t11b) <= want
              and not (want & set(G.SMOKE_FILES)),
              f'표본 {len(want)} · 행 {len(got)} · 스모크교집합 {len(want & set(G.SMOKE_FILES))}'))
    idx = [mats.index(b) for b in got if b in mats]
    A.append((f'★G-POP 전 행 인덱스 ∈ [{G.POP_RANGE[0]},{G.POP_RANGE[1]})',
              all(G.POP_RANGE[0] <= i < G.POP_RANGE[1] for i in idx) and len(idx) == len(got),
              f'범위 {min(idx) if idx else "-"}–{max(idx) if idx else "-"}'))
    per = {a: len(by.get(a, {})) for a in G.ARMS_ALL + G.ARMS_T11ONLY}
    A.append((f'★G-N SHIP·FIX·ZERO·ORACLE {G.N_TOTAL}행 · RESCALE* {G.N_T11}행 · 총 {G.N_ROWS} '
              f'(⚠임계 33/19 는 n=52 전용)',
              all(per[a] == G.N_TOTAL for a in G.ARMS_ALL)
              and all(per[a] == G.N_T11 for a in G.ARMS_T11ONLY)
              and len(rows) == G.N_ROWS and n_fs == G.N_TOTAL,
              f'{per} · rows={len(rows)} · n_fs={n_fs}'))
    nf = [r['block'] for r in rows if r.get('nrmse') is None or not np.isfinite(r['nrmse'])]
    A.append(('★G-FINITE 모든 행의 nRMSE 가 유한', not nf, f'비유한 {len(nf)}'))

    # ── 팔 정체·누출·되읽기 ────────────────────────────────────────────────
    est = [r for r in rows if r['arm'] not in ('ZERO', 'ORACLE')]
    leak = [r for r in est if r.get('gt_std') is None or abs(r['gt_std']) > 1e-12
            or r.get('gt_std_after') is None or abs(r['gt_std_after']) > 1e-12
            or abs((r.get('gt_value') or -9) - G.GT_CONST) > 1e-12]
    A.append(('★★G-NOLEAK gt 상수(std 0) · 값==GT_CONST · initialize 후에도 [되읽기]',
              not leak, f'위반 {len(leak)}'))
    from wavesplat.optimizer.sequential_update import DEFAULT_LEARNED_ENSEMBLE_K as KDEP
    from wavesplat.optimizer.learned_init import M_OBS_CAP
    ab = [r['arm'] for r in est if (r.get('prov') or {}).get('resolved_init_method') != 'learned'
          or (r.get('prov') or {}).get('real_domain_gate_fired') is not False
          or (r.get('prov') or {}).get('ensemble_k') != KDEP]
    A.append((f'★G-ARM resolved==learned · 실측게이트 미발화 · ensemble_k=={KDEP} [되읽기]',
              not ab, f'위반 {len(ab)}'))
    dup = [a for a in ('SHIP', 'FIX', 'RESCALE', 'RESCALE-FIX')
           if len({r.get('proposal_md5') for r in by.get(a, {}).values()}) != len(by.get(a, {}))]
    same_fix = [b for b in by.get('FIX', {})
                if b in by.get('SHIP', {})
                and by['FIX'][b].get('proposal_md5') == by['SHIP'][b].get('proposal_md5')]
    ctrl_same = [b for b in by.get('RESCALE-FIX', {})
                 if b in by.get('RESCALE', {})
                 and by['RESCALE-FIX'][b].get('proposal_md5') == by['RESCALE'][b].get('proposal_md5')]
    A.append(('★G-ATOMS 팔별 제안 지문 전건 상이 **그리고** 같은 파일 FIX≠SHIP (처치가 제안을 바꿨다)',
              not dup and not same_fix,
              f'중복 팔 {dup or "없음"} · FIX==SHIP {len(same_fix)} '
              f'(⚠CTRL 동일 {len(ctrl_same)} — 공시)'))
    cb = [r['block'] for r in est if r.get('n_gathered') is None
          or r['n_gathered'] != r.get('n_offered') or r.get('cap_proposal') != int(M_OBS_CAP)]
    A.append((f'★G-CAP 제안 단계가 실제로 돌았다(모음==제공) · 캡=={int(M_OBS_CAP)} [되읽기]',
              not cb, f'위반 {len(cb)}'))
    tb = [r for r in rows if r.get('t_origin') is None or abs(r['t_origin']) > 1e-9]
    A.append(('★G-TORIGIN 창 시작 t == 0 [변화 탐지기]', not tb, f'위반 {len(tb)}'))
    db = []
    for r in rows:
        es = r.get('e_shape') or [0]
        if (r.get('n_offered') != G.DENSITY * int(G.WIN_S) or r.get('n_frames') != int(G.WIN_S)
                or es[0] < G.J0 + int(G.WIN_S) or r.get('dt') is None or r.get('tq') is None
                or abs(r['tq'] - (int(G.WIN_S) - 1) * r['dt']) > 1e-9):
            db.append(r['block'])
    A.append((f'★G-DENSITY 제공=={G.DENSITY}x{int(G.WIN_S)} · 창이 자료 안 · tq==(N-1)dt [되읽기]',
              not db, f'위반 {len(db)}'))
    rb = [r['block'] for r in rows if r['arm'] in ('RESCALE', 'RESCALE-FIX')
          and (r.get('x_scale') is None or abs(r['x_scale'] - 1.0 / G.RESCALE_S) > 1e-9
               or (r.get('disp_relerr') or 1) > 1e-6)]
    A.append((f'★G-RESCALE 배율이 s={G.RESCALE_S} 와 일치 · ω=√(gk) 보존 [변화 탐지기]',
              not rb, f'위반 {len(rb)}'))
    clamp = {a: sum(1 for r in by.get(a, {}).values() if (r.get('vp_clamp_hits') or 0) > 0)
             for a in ('SHIP', 'FIX', 'RESCALE', 'RESCALE-FIX')}
    A.append(('★G-CLAMP 경사 클램프 발화 **기록**됨 (⛔발화로 막지 않는다)',
              all(r.get('vp_clamp_hits') is not None for r in est), f'팔별 {clamp}'))
    A.append(('G-WRAPPER 출하 진입점 [⚠배포 상수상 **구조적으로 참**]',
              all((r.get('prov') or {}).get('is_ensemble_wrapper') for r in est), '—'))

    # ── ★★G-REPEAT (스모크 전용, AMENDMENT_2 §1) ──────────────────────────
    #   ⛔사전등록 §8 이 차단 게이트로 넣었는데 초판 분석기가 **구현하지 않았다**(방벽이 잡음).
    #   러너가 스모크에서 실제로 재 `meta_t13_SMOKE.json['g_repeat']` 에 기록한 것을 단언한다.
    #   ⛔스모크 meta 가 없으면 FAIL — 부재 시 통과 금지.
    rep = None
    if os.path.exists(SMOKE_META):
        rep = json.load(open(SMOKE_META, encoding='utf-8')).get('g_repeat')
    A.append(('★★G-REPEAT 스모크에서 같은 파일 SHIP 2회가 **비트 동일** (⛔부재 시 FAIL)',
              bool(rep) and all(r.get('bit') for r in rep),
              f'{"없음" if rep is None else [r["bit"] for r in rep]}'))

    # ── 환경·동결 (AMENDMENT_2 §2: 측정면 / 분석면 분리) ───────────────────
    rec = meta.get('code_md5') or {}
    now = meta.get('code_md5_now') or {}
    import run_t13 as _R
    live = _R.freeze_map()
    fz, fz2 = _frozen(FROZEN), _frozen(FROZEN_A2)
    run_keys = sorted(k for k in live if k not in ANALYSIS_SIDE)
    ana_keys = sorted(k for k in live if k in ANALYSIS_SIDE)
    d_meta = [k for k in run_keys if live[k] != rec.get(k)]
    d_fz = None if fz is None else [k for k in run_keys if live[k] != fz.get(k)]
    d_fz2 = None if fz2 is None else sorted(k for k in set(ana_keys) | set(fz2)
                                            if live.get(k) != fz2.get(k))
    want_seeds = {G.seed_of(b) for b in want}
    sb = sorted(a for a in G.ARMS_ALL if {r.get('seed') for r in by.get(a, {}).values()}
                != want_seeds)
    A.append(('★G-ENV 측정면 3중 동결(현재==meta==0행동결) · 분석면 동결 · threads · 시드 집합',
              meta.get('threads_readback') == 1 and bool(rec) and not d_meta
              and d_fz == [] and d_fz2 == [] and not sb,
              f'측정면 {len(run_keys)}키 런대비 {len(d_meta)} · 0행동결대비 '
              f'{"N/A" if d_fz is None else len(d_fz)} · 분석면 '
              f'{"N/A" if d_fz2 is None else len(d_fz2)} · 시드위반팔 {len(sb)}'))
    # ── ★AMENDMENT_3 §2: 동결 목록이 판정 경로 파일 셋을 빠뜨렸다 ────────────
    #   `initializer.py`(성장 경로가 **호출 시점에** MIN_DK·_beamform_coarse 를 임포트) ·
    #   `d3_setdiff_prior.py`(제안 경로가 denorm_atoms 임포트) ·
    #   `setdiff_prior.pt`(★**이 배치의 BLOCKER 를 만든 바로 그 파일** — 러너 주석이 축자로
    #   적어놓고 동결은 안 했다).
    #   ⛔러너(측정면)를 고치지 않는다 ⟹ **T12 의 결과 0행 동결값**과 대조한다.
    #   T12 동결(2026-09-04 23:xx)은 T13 본실행(09-05 00:13)보다 **앞선다** ⟹ 진짜 앵커다.
    import hashlib as _h
    T12D = os.path.abspath(os.path.join(HERE, '..', 'T12_ISOLATE_2026-09-04'))
    fz12 = _frozen(os.path.join(T12D, 'FROZEN_md5.txt')) or {}
    EXT = ('wavesplat/optimizer/initializer.py', 'scripts/d3_setdiff_prior.py',
           'sha256:results/direction3/setdiff_prior.pt')
    ext_bad, ext_det = [], {}
    for k in EXT:
        if k.startswith('sha256:'):
            path = os.path.join(G.ROOT, k[len('sha256:'):])
            hh = _h.sha256()
            if os.path.exists(path):
                with open(path, 'rb') as fh:
                    for c in iter(lambda: fh.read(1 << 20), b''):
                        hh.update(c)
                cur = hh.hexdigest()
            else:
                cur = None
        else:
            path = os.path.join(G.ROOT, k)
            cur = (_h.md5(open(path, 'rb').read()).hexdigest()
                   if os.path.exists(path) else None)
        ext_det[k.split('/')[-1]] = (cur == fz12.get(k))
        if cur is None or fz12.get(k) is None or cur != fz12.get(k):
            ext_bad.append(k)
    A.append(('★G-FREEZE-EXT [사후] 러너가 빠뜨린 판정경로 3파일을 **T12 의 결과 0행 동결값**과 대조',
              not ext_bad, f'{ext_det} · 불일치 {len(ext_bad)}'))

    probe = G.blocked_by([('★★G-MAP 합성', False, ''), ('★G-N 합성', False, ''),
                          ('★★G-ANCHOR-SHIP 합성', False, ''), ('G-COV 합성', True, '')])
    A.append(('G-BLOCK blocked_by 자가진단 (긴 이름 우선)',
              probe == ['G-MAP', 'G-N', 'G-ANCHOR-SHIP'], f'합성 → {probe}'))
    A.append(('G-COV 게이트가 자료를 봤다', len(rows) > 0, f'rows={len(rows)}'))

    blocked = G.blocked_by(A)
    all_ok = all(bool(o) for _, o, _ in A)
    lab = G.label(w_fs, w_fz) if (not blocked and all_ok) else None

    # ── CTRL 잡음 바닥 (결과 전 고정된 읽기) ───────────────────────────────
    def rel(a, b, blocks=None):
        da, db = by.get(a, {}), by.get(b, {})
        cm = sorted(set(da) & set(db) & (set(blocks) if blocks else set(da)))
        return [(da[x]['nrmse'] - db[x]['nrmse']) / db[x]['nrmse'] for x in cm]
    ctrl_rel = rel('RESCALE-FIX', 'RESCALE')
    eff_rel = rel('FIX', 'SHIP')
    floor = float(np.median(np.abs(ctrl_rel))) if ctrl_rel else None
    effect = float(np.median(eff_rel)) if eff_rel else None
    within = (floor is not None and effect is not None and abs(effect) <= floor)
    if lab and within:
        lab = lab + ' (WITHIN-PERTURBATION-FLOOR)'

    strata = {'T11-24': paired('FIX', 'SHIP', t11b), 'FRESH-28': paired('FIX', 'SHIP', fresh)}
    hs = {}
    for nm, pr in (('hs_low', lambda h: h <= G.HS_SPLIT), ('hs_high', lambda h: h > G.HS_SPLIT)):
        blk = [b for b, r in by.get('FIX', {}).items() if pr(r['hs'])]
        hs[nm] = paired('FIX', 'SHIP', blk)
    v = {'batch': 'T13-DEPTHFIX', 'kind': 'CONFIRMATORY',
         'amendments': ['AMENDMENT_1_PREDATA.md (결과 0행)',
                        'AMENDMENT_2_POSTRESULT.md (결과 이후 · 분석기 전용 · 새 런 0회)',
                        'AMENDMENT_3_POSTRESULT.md (결과 이후 · 공시 + 동결 확장 · 새 런 0회)',
                        'AMENDMENT_4_POSTRESULT.md (결과 이후 · **공시만** 가산 · 새 런 0회 · '
                        'A3 §3 이 부채로 남긴 PREREG §11 누락 공시를 갚는다)'],
         'post_result_checks': ['G-REPEAT (사전등록에 있었으나 초판 분석기에 없었다)',
                                'G-ENV 측정면/분석면 분리',
                                'G-FREEZE-EXT (러너 동결 목록 누락 3파일, AMENDMENT_3)'],
         'grade': 'CONFIRMATORY (⚠T11-24 층은 노출 · FRESH-28 은 두 팔 blind)',
         'n_rows': len(rows),
         'gates': [{'name': n, 'pass_': bool(o), 'detail': d} for n, o, d in A],
         'gates_pass': f'{sum(1 for _, o, _ in A if o)}/{len(A)}', 'blocked_by': blocked,
         'primary': {'stat': 'FIX < SHIP', 'w': w_fs, 'n': n_fs,
                     'p_one_sided': G.sign_p(w_fs, n_fs), 'thresh_win': G.THRESH_WIN,
                     'thresh_hurt': G.THRESH_HURT},
         'sec_a': {'stat': 'FIX < ZERO', 'w': w_fz, 'n': n_fz,
                   'p_one_sided': G.sign_p(w_fz, n_fz)},
         'ship_vs_zero': {'w': w_sz, 'n': n_sz, 'p_one_sided': G.sign_p(w_sz, n_sz)},
         'ctrl': {'w': w_ctrl, 'n': n_ctrl, 'band': list(G.CTRL_BAND),
                  'moved': not (G.CTRL_BAND[0] <= w_ctrl <= G.CTRL_BAND[1]),
                  'floor_median_abs_rel': floor, 'effect_median_rel': effect,
                  'within_floor': within,
                  'identical_proposals': len(ctrl_same)},
         'strata_descriptive': strata, 'hs_descriptive': hs,
         'clamp_by_arm': clamp,
         'medians': {a: float(np.median([r['nrmse'] for r in d.values()]))
                     for a, d in sorted(by.items()) if d},
         'mde': G.MDE, 'prediction': {'q': G.PRED_Q, 'label_probs': G.label_probs(),
                                      'null_label_probs': G.label_probs(0.5, 0.5)},
         'proposal_device': dev, 'label': lab, 'label_msg': lab or 'BLOCKED',
         'disclosures': [
             # ★AMENDMENT_4: PREREG §11 의 (i)(ii)(iii) 과 「비트 무연산」 사실이 떨어져 있었다.
             '⛔FIX 는 **하네스 처치**다 — 패키지에 없다. 이겨도 **배포 권고가 아니다**: '
             '배포는 (i) 패키지 수리(REACHABILITY §5 형태) (ii) test_shipping_defaults 핀 갱신 '
             '(iii) 배포-후 게이트 출력 첨부(08-23 교훈 "측정한 구성 ≠ 배포한 구성") 가 있는 '
             '**별도 결정**이다. ★단 그 수리는 자국 합성 코퍼스(110 에피소드 전부 d=50)에서 '
             '**비트 무연산**이므로 기존 판정의 재확증은 요구하지 않는다 — 그 사실도 배포 '
             '결정문이 다시 실측해야 한다 (PREREG §11)',
             '⛔⛔이 배치의 절반은 새 자료가 아니다 — T11-24 층의 FIX 24행이 T12 의 '
             'E-MAPDEPTH 24행과 **비트 동일**이다(nrmse·proposal_md5·seed 0/24 불일치, '
             'max|Δ|=0.0). 같은 처치를 같은 시드로 두 번 돌린 것이다. '
             '⟹ "T12 와 T13 이 **독립적으로** 같은 결론에 왔다" 는 **거짓**이며 철회한다. '
             'PRIMARY 26/52 = 10/24(T12 행의 재판독) + 16/28(신규). 신규 절반만 보면 '
             'FIX<SHIP 16/28(문턱 19) · FIX<ZERO 13/28 · SHIP<ZERO 6/28 — 라벨은 불변. '
             '⚠PREREG §3 표가 T11-24 층을 "FIX 미노출" 이라 적은 것은 작성 시점엔 참이었으나 '
             'T12 커밋 뒤 본실행에서는 거짓이다 (AMENDMENT_3 §1)',
             '⚠FRESH-28 의 SHIP 도 형제 노출이다 — T12 SEC-R 이 같은 28파일을 (다른 시드로) '
             '먼저 돌렸다. FIX 는 blind 가 맞다',
             '⛔FIX-UNDETECTED 를 "수심은 원인이 아니다"·"교락은 무해하다" 로 읽지 마라 — '
             f'검정력의 진술이다 (MDE q80={G.MDE["q80"]} 병기 의무 · 기각 불가 대역 '
             f'w={G.MDE["unrejectable_w"]}, 32/52=61.5% 도 p=0.0632)',
             '⛔DEPTH-WAS-MASKING-SKILL 을 "T11 BAND-MISMATCH 철회" 로 읽지 마라 — '
             '이 배치는 RESCALE 대 FIX 를 검정하지 않는다',
             '⚠두 다리(HELPS·HURTS) 가족 α = 0.0704 — 공시하고 보정하지 않는다',
             '⚠학습/추론 불일치: 체크포인트는 d=50 맵으로 학습됐다 ⟹ 처치가 자료-맵 정합을 '
             '고치면서 학습-추론 불일치를 만든다(군속도 비 0.74–0.89 ⟹ 방사 PSF 폭 ×1.13–1.35). '
             '⛔크기는 **추정**이지 측정이 아니다 — 그래서 HURTS 다리와 음성대조를 뒀다. '
             # ★AMENDMENT_4: PREREG §11 의 「지목 금지」와 허용 자구가 떨어져 있었다.
             '⛔따라서 `FIX-HURTS` 를 "학습/추론 불일치가 원인" 으로 **지목** 금지 — 이 배치는 '
             '불일치의 크기를 재지 않는다. 허용 자구 = "맵 수심을 자료에 맞추는 것만으로는 '
             '배포 체크포인트에서 해가 된다" (PREREG §11). '
             f'⚠단 이 배치의 라벨은 `FIX-HURTS` 가 **아니다**(FIX<SHIP w={w_fs} > 해침 문턱 '
             f'{G.THRESH_HURT}) ⟹ 그 허용 자구조차 이 자료로는 쓸 수 없다',
             '⚠이미 존재하고 이 배치가 건드리지 않는 불일치가 더 크다: 개구/픽셀 비 ×2.6–3.1',
             '⚠T11-24 층은 SHIP·RESCALE·ZERO·ORACLE 값이 **공개된 상태**로 설계됐다 ⟹ '
             '⛔"완전 blind" 금지. FRESH-28 은 두 팔 다 미노출이나 ⚠Hs 가 3/25 로 치우쳤다',
             '⛔E2 와 대각선 비교 금지 · 저들 논문 수치와 비교 금지 · solver 미명시',
             '⚠Tp 가 전 파일 10.0 s ⟹ 한 주기 조건의 진술이다',
             # ★AMENDMENT_4: PREREG §11 의 사유(BLAS 스레드)가 떨어져 있었다 — C9 가 이 토큰으로 FAIL 했다.
             '⛔wall_s 를 팔 간 비교에 쓰지 마라(사유 = **BLAS 스레드**, T11 공시) · '
             'proposal_device 는 config 가 정하지 않는다',
             '⚠층별·Hs별 진술은 **기술통계**이고 라벨을 만들지 않는다 '
             f'(T11-24 q80={G.MDE["q80_t11"]} · FRESH-28 q80={G.MDE["q80_fresh"]})',
             '⚠CTRL(RESCALE-FIX vs RESCALE)은 검정이 아니라 **섭동 잡음 바닥**이다. '
             'effect 가 floor 이하이면 라벨에 (WITHIN-PERTURBATION-FLOOR) 가 붙는다',
             # ────────────────────────────────────────────────────────────────
             # ★AMENDMENT_4 (결과 이후 · 순수 가산) — PREREG §11 의 의무 공시 중
             #   verdict 에 **한 번도 실리지 않은** 셋. ⛔공시는 읽기를 좁힐 뿐
             #   라벨을 만들지 못하므로 이 추가는 한 방향이다(막을 수만 있다).
             # ────────────────────────────────────────────────────────────────
             '⛔`FIX` 의 **절대 nRMSE 를 성능으로 인용 금지** — 읽는 것은 **짝 부호와 상대 '
             '크기**뿐이다 (PREREG §11). ⟹ 이 verdict 의 `medians` 는 짝 비교의 재료이지 '
             '성능값이 아니다',
             f'⛔조건부 공시 — **발화하지 않았다(N/A)**: PREREG §11 은 `SKILL-FIX-UNDETECTED` '
             f'를 **약한 칸**으로 등록하고, 그 칸이 나오면(FIX<ZERO 유의인데 FIX<SHIP 미검출) '
             f'`SHIP<ZERO` 도 함께 인쇄해 **두 팔 다 기술 문턱 근처**임을 보이라고 요구했다 '
             f'(F1 부류 경고). 실측은 SEC-A `FIX<ZERO` {w_fz}/{n_fz} 로 이김 문턱 '
             f'{G.THRESH_WIN} 에 미달 ⟹ 라벨이 `FIX-UNDETECTED` 이고 **조건은 발화하지 '
             f'않았다**. 그래도 `SHIP<ZERO` {w_sz}/{n_sz} 를 함께 싣는다 — ⚠`FIX`·`SHIP` '
             f'어느 쪽도 `ZERO` 를 문턱 위에서 넘지 못한다',
             '⚠**재학습(처치 3)은 미측정으로 남는다** — PREREG §11 은 이 배치가 '
             '`FIX-HELPS-*` 를 내면 재학습의 사전등록 동기가 되고 `FIX-HURTS` 를 내면 '
             '"맵만 고치면 된다" 가 죽는다고 적었다. 라벨이 `FIX-UNDETECTED` 이므로 '
             '**어느 쪽도 아니다** ⟹ ⛔이 배치는 재학습의 결과를 예단하지 않으며, '
             '재학습의 동기도 사망 선고도 만들지 않는다',
         ]}
    json.dump(v, open(os.path.join(OUTDIR, 'verdict_t13.json'), 'w'),
              ensure_ascii=False, indent=1)
    print(f"T13-DEPTHFIX :: {v['label_msg']}")
    print(f"  PRIMARY FIX<SHIP {w_fs}/{n_fs} p={v['primary']['p_one_sided']:.4g} "
          f"(이김 {G.THRESH_WIN} · 해침 {G.THRESH_HURT})")
    print(f"  SEC-A   FIX<ZERO {w_fz}/{n_fz} p={v['sec_a']['p_one_sided']:.4g}")
    print(f"  기준선  SHIP<ZERO {w_sz}/{n_sz} p={v['ship_vs_zero']['p_one_sided']:.4g}")
    print(f"  CTRL    RESCALE-FIX<RESCALE {w_ctrl}/{n_ctrl} · 밴드 {G.CTRL_BAND} · "
          f"이탈 {v['ctrl']['moved']}")
    print(f"  잡음바닥 {floor} vs 효과 {effect} ⟹ within {within}")
    print(f"  층(기술) T11-24 {strata['T11-24']} · FRESH-28 {strata['FRESH-28']} · Hs {hs}")
    print(f"  중앙 nRMSE {v['medians']}")
    print(f"  클램프 {clamp} · device {dev}")
    print(f"  gates {v['gates_pass']}")
    for n, o, d in A:
        print(f"   [{'OK ' if o else 'FAIL'}] {n:70s} {d}")


if __name__ == '__main__':
    main()
