"""M7 단일출처 — **감독 신호 축**(학습 분포 × 학습 손실 2×2).

★이 파일이 팔·코퍼스·문턱·검정·**결과 전 해석 규칙**을 유일하게 정의한다.
`PREREG.md` 는 이 파일의 산문 번역이고, **두 곳이 어긋나면 코드가 권위**다.
모든 수는 `premise_m7.json`(동결 자료에서 측정) 또는 저장소의 동결 파일에서 온다 — **손복사 0건**.

질문: ***"넷이 실측에서 자신 있게 틀리는 것은 학습 *분포* 때문인가, 학습 *손실* 때문인가?"***
"""
import json
import math
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
PREMISE = os.path.join(HERE, 'premise_m7.json')


def premise():
    return json.load(open(PREMISE))


# ── 0. 팔 정의 ───────────────────────────────────────────────────────────────
# 학습 절차는 네 칸에서 **동일**하다. 다른 것은 두 요인뿐이다.
#   D(분포) ∈ {BASE, HARD} : Phase-A 사전학습 장면 분포
#   L(손실) ∈ {0.0, 1.0}   : Phase-B 미세조정의 λ_field (0.0 도 4,000 스텝을 돈다 = 스텝 수 대조)
CELLS = {
    'BASE':   dict(data='BASE', lam_field=0.0, pretrain='PRE-BASE-1'),
    'LOSS':   dict(data='BASE', lam_field=1.0, pretrain='PRE-BASE-1'),
    'DATA':   dict(data='HARD', lam_field=0.0, pretrain='PRE-HARD-1'),
    'BOTH':   dict(data='HARD', lam_field=1.0, pretrain='PRE-HARD-1'),
    # ★음성대조 = 학습 시드만 다른 두 번째 BASE. 저장소에 이 넷의 **사전학습 시드 잡음 추정치가
    #   없다**(v1/v2/v3 는 시드가 아니라 `target_mode` 가 다르다). 이 칸이 그것을 측정한다.
    'BASE-B': dict(data='BASE', lam_field=0.0, pretrain='PRE-BASE-2'),
}
CELL_ORDER = ('BASE', 'LOSS', 'DATA', 'BOTH', 'BASE-B')
PRETRAINS = {'PRE-BASE-1': dict(data='BASE', seed=20260828),
             'PRE-BASE-2': dict(data='BASE', seed=20260829),
             'PRE-HARD-1': dict(data='HARD', seed=20260828)}
# 참조 팔(학습 0회)
REF_SHIPPED = 'SHIPPED'      # 배포 체크포인트 beamform_cond_diffusion_mixed_v1.pt
REF_BANK = 'BANK'            # 실측 배포 경로(bank_mmse) — 체크포인트 무관, 맥락용
ARMS_REAL = CELL_ORDER + (REF_SHIPPED, REF_BANK)

# ── 1. 학습 상수 (전부 기존 파일에서 인용, 새 탐색 0건) ──────────────────────
PRETRAIN_STEPS = 40000       # tmp/mixed_retrain_2026-07-26/PREREG.md §4 (동결)
PRETRAIN_LR = 2e-4
FINETUNE_STEPS = 4000        # tmp/wp_e2e_2026-07-25/e2e_finetune.py CONFIGS['A']
FINETUNE_LR = 5e-5
LAMBDA_FIELD_ON = 1.0        # 〃 (config A). ⛔λ=4.0(config B)는 wp_e2e 에서 **앵커를 망가뜨렸다**
T_LO_FRAC = 0.25             # 〃 t_lo = T//4
BSZ = 32
GRAD_CLIP = 5.0
NGRID = 20                   # 〃 장 손실 렌더 격자
# HARD 앵커 = B-26 동결 premise 의 레코드별 (Hs, Tp_disp) 중앙값.
# ★물리 스크리닝(결과 무관, 결과 전 선언): k_p·Hs > BREAKING_KHS 인 앵커는 **학습 앵커에서만**
#   제외한다(평가 레코드에서는 제외하지 않는다 — 사후배제 금지).
BREAKING_KHS = 0.44          # 심해 한계 파형경사(Stokes/Michell, ka≈0.443)
TRAIN_SEED_LO, TRAIN_SEED_HI = 2_000_000_000, 2_147_483_647   # mixed_retrain §3.1 규약

# ── 2. 코퍼스 ────────────────────────────────────────────────────────────────
REAL_RECORDS = ('AA14', 'AA15a', 'AA15b', 'BS11a', 'BS11b', 'BS11c', 'BS11d',
                'BS13a', 'BS13b', 'BS13c', 'LJ18a', 'YS17')     # scene_station.RECORDS_12
REAL_WIN_PRIMARY = 3          # tmp/B2_REAL_2026-08-23/gates_b2.py:19
REAL_WINS_SECONDARY = (0, 1, 2, 3, 4, 5)
DIFFUSION_SEEDS = (0, 1, 2)   # 레코드별 median-of-3 (전역 RNG 결합 완화)
EST_SEEDS = (42, 43, 44)      # 합성: gates_b11.EST_SEEDS 규약
E2_CORPUS = 'B12'             # 헤드라인 회귀: tmp/B12_HOS_EXTEND_2026-08-23 48 에피소드
E2_N_EP = 48
E3_CORPUS = 'M1'              # 가파른·비선형: tmp/M1_NONLIN_REAL_2026-08-28/data_m1 (LIN 24 + HOS 24)
E3_N_EP = 48
# ⛔E3 는 **M1 의 판정이 동결된 뒤에만** 돌린다(M1 다중성 보호). 그리고 HARD 팔에게는 in-family 다.

# ── 3. 검정 ──────────────────────────────────────────────────────────────────
N_REAL = len(REAL_RECORDS)
FAMILY = ('T-1_loss', 'T-2_data', 'T-3_interaction')
FAMILY_ALPHA = 0.05
NEG_CONTROL_ABORT = 10        # ★BASE vs BASE-B 가 어느 방향으로든 이 수 이상이면 E1 전체 DESCRIPTIVE


def sign_test_one_sided(k, n):
    """P(X >= k), X~Bin(n, 0.5). 단측."""
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2.0 ** n


def sign_test_two_sided(k, n):
    return min(1.0, 2.0 * min(sign_test_one_sided(k, n), sign_test_one_sided(n - k, n)))


def mde():
    """★결과 전 MDE — Holm 첫 문턱(α/3)을 넘는 최소 승수."""
    a1 = FAMILY_ALPHA / len(FAMILY)
    need = min(k for k in range(N_REAL + 1) if sign_test_one_sided(k, N_REAL) <= a1)
    return dict(n=N_REAL, holm_first_alpha=a1, min_wins=need,
                p_at_min=sign_test_one_sided(need, N_REAL),
                p_at_min_minus_1=sign_test_one_sided(need - 1, N_REAL))


# ── 4. 1차 지표: 실측 nowcast (σ 정규화) ─────────────────────────────────────
def _cell_value(rows, cell, record, field='nowcast'):
    """레코드별 = 확산 시드 3개의 **중앙값**(사전등록)."""
    v = [r[field] for r in rows
         if r.get('cell') == cell and r.get('record') == record
         and r.get('win') == REAL_WIN_PRIMARY and r.get(field) is not None]
    return st.median(v) if v else None


def contrasts(rows, field='nowcast'):
    """T-1/T-2/T-3 을 레코드 단위(n=12)로 만든다. **레코드가 독립 단위** — 창은 부차."""
    g = lambda c, r: _cell_value(rows, c, r, field)
    out = {k: {} for k in FAMILY}
    for r in REAL_RECORDS:
        b, l, d, o = g('BASE', r), g('LOSS', r), g('DATA', r), g('BOTH', r)
        if None in (b, l, d, o):
            continue
        out['T-1_loss'][r] = st.median([l - b, o - d])          # 손실 주효과 (H1: < 0)
        out['T-2_data'][r] = st.median([d - b, o - l])          # 분포 주효과 (H1: < 0)
        out['T-3_interaction'][r] = (o - d) - (l - b)           # 교호 (양측)
    res = {}
    for k, dd in out.items():
        n = len(dd)
        wins = sum(1 for v in dd.values() if v < 0)
        p = (sign_test_two_sided(wins, n) if k == 'T-3_interaction'
             else sign_test_one_sided(wins, n))
        res[k] = dict(n=n, wins_negative=wins, p=p,
                      median_delta=st.median(dd.values()) if dd else None,
                      per_record=dd)
    return res


def holm(res):
    """Holm — FAMILY 3개."""
    order = sorted(FAMILY, key=lambda k: res[k]['p'])
    out, m = {}, len(FAMILY)
    for i, k in enumerate(order):
        thr = FAMILY_ALPHA / (m - i)
        out[k] = dict(p=res[k]['p'], threshold=thr, reject=res[k]['p'] <= thr)
        if not out[k]['reject']:
            for k2 in order[i + 1:]:
                out[k2] = dict(p=res[k2]['p'], threshold=FAMILY_ALPHA / (m - order.index(k2)),
                               reject=False)
            break
    return out


def neg_control(rows, field='nowcast'):
    """★음성대조 — 학습 시드만 다른 BASE vs BASE-B."""
    d = {}
    for r in REAL_RECORDS:
        a, b = _cell_value(rows, 'BASE', r, field), _cell_value(rows, 'BASE-B', r, field)
        if a is not None and b is not None:
            d[r] = b - a
    n = len(d)
    w = sum(1 for v in d.values() if v < 0)
    return dict(n=n, wins=w, max_side=max(w, n - w),
                p=sign_test_two_sided(w, n) if n else None,
                aborts_inference=bool(n and max(w, n - w) >= NEG_CONTROL_ABORT),
                median_delta=st.median(d.values()) if d else None, per_record=d)


def skill_vs_true_zero(rows, zero_ref, field='nowcast'):
    """★기술 막대 — **참 제로필드**(tmp/B2_REAL_2026-08-23/zero_ref.py) 대비.
    ⛔`ZERO_REF=1.0` 은 정의상 참이 아니다(레코드별 0.3375–1.1559) ⟹ 절대 쓰지 않는다.
    ⛔이 계수는 **기술(DESCRIPTIVE)** 이다 — 부호검정의 귀무가 0.5 라는 근거가 없다."""
    out = {}
    for c in ARMS_REAL:
        w = [r for r in REAL_RECORDS
             if _cell_value(rows, c, r, field) is not None
             and _cell_value(rows, c, r, field) < zero_ref[r]]
        out[c] = dict(beats_true_zero=len(w), n=N_REAL, records=w)
    return out


# ── 5. 합성 회귀 (R1, 하드 바) ───────────────────────────────────────────────
# ★막대의 출처: mixed_retrain PREREG §5 G1 의 동결 여유 +0.010 (같은 부류의 do-not-forget 바).
R1_MARGIN = 0.010


def r1_regression(rows_e2):
    """각 칸의 배포-경로 평균 avg_nRMSE 가 SHIPPED 평균 + 0.010 이내인가.
    ⚠SHIPPED 평균은 **이 배치에서 다시 측정**한다(인용 금지 규율)."""
    def mean_of(cell):
        per = {}
        for r in rows_e2:
            if r.get('cell') != cell:
                continue
            per.setdefault(r['episode'], []).append(r['avg_nRMSE'])
        return (st.mean([st.median(v) for v in per.values()]) if per else None, len(per))
    ship, n_ship = mean_of(REF_SHIPPED)
    out = {}
    for c in CELL_ORDER:
        m, n = mean_of(c)
        out[c] = dict(mean=m, n_ep=n, bar=(ship + R1_MARGIN) if ship is not None else None,
                      passed=(m is not None and ship is not None and m <= ship + R1_MARGIN))
    return dict(shipped_mean=ship, shipped_n_ep=n_ship, per_cell=out,
                all_pass=all(v['passed'] for v in out.values()))


# ── 5b. ★S-1 부차 판정: **분포 이동 재학습이 헤드라인을 유지하는가** ─────────
# 심사 단서 ③(*"체크포인트가 평가와 같은 분포"*)은 **학습-측 개입으로만** 닫힌다.
# HARD 로 학습한 넷이 **동결 B-12 코퍼스**에서 헤드라인을 유지하면 그 단서가 약해진다.
# ★문턱은 새로 고르지 않는다 — `tmp/B12_HOS_EXTEND_2026-08-23/gates_b12.py:52-53` 동결값.
S1_N_EP, S1_NOMINAL, S1_CONFIRM, S1_NEG_MIN = 48, 31, 34, 31


def _b12_threshold_parity():
    """동결값 손복사 방지 — B-12 파일에서 되읽어 대조한다(불가하면 None)."""
    import importlib.util
    f = os.path.join(ROOT, 'tmp/B12_HOS_EXTEND_2026-08-23/gates_b12.py')
    if not os.path.exists(f):
        return None
    src = open(f).read()
    for line in src.splitlines():
        if line.startswith('NOMINAL, CONFIRM, WEAK'):
            vals = line.split('=')[1].split('#')[0]
            n, c, _w = [int(x) for x in vals.split(',')]
            return (n, c) == (S1_NOMINAL, S1_CONFIRM)
    return None


def s1_shift_survives(rows_e2):
    """칸별 `SHIP(ckpt) < SPEC25` 승수 + 음성대조 `SPEC25 < ZERO`. B-12 규약."""
    def med(cell):
        per = {}
        for r in rows_e2:
            if r.get('cell') == cell:
                per.setdefault(r['episode'], []).append(r['avg_nRMSE'])
        return {k: st.median(v) for k, v in per.items()}
    spec, zero = med('SPEC25'), med('ZERO')
    neg = sum(1 for k in spec if k in zero and spec[k] < zero[k])
    out = {}
    for c in list(CELL_ORDER) + [REF_SHIPPED]:
        m = med(c)
        ks = sorted(set(m) & set(spec))
        w = sum(1 for k in ks if m[k] < spec[k])
        out[c] = dict(n=len(ks), better=w, p=sign_test_one_sided(w, len(ks)) if ks else None,
                      confirmatory=w >= S1_CONFIRM, nominal=w >= S1_NOMINAL)
    return dict(neg_control=dict(better=neg, n=len(spec), ok=neg >= S1_NEG_MIN),
                per_cell=out, threshold_parity_with_B12=_b12_threshold_parity())


def label_s1(s1):
    """⛔결과 전 고정. HARD 로 학습한 두 칸(DATA·BOTH)만 이 판정의 주어다."""
    if not s1['neg_control']['ok']:
        return ('BASELINE-NO-SKILL-HERE',
                '⛔기준선이 무기술 ⟹ S-1 무효(B-12 규약).')
    hard = [s1['per_cell'][c] for c in ('DATA', 'BOTH') if c in s1['per_cell']]
    if hard and all(h['confirmatory'] for h in hard):
        return ('DISTRIBUTION-SHIFT-SURVIVES',
                '★의도적으로 이동시킨 학습 분포에서도 동결 코퍼스의 헤드라인이 유지된다 ⟹ '
                '*"체크포인트가 평가와 같은 분포"* 단서가 **약해진다**. '
                '⛔단 시드 축은 여전히 신선하지 않고, 크기 비교는 금지.')
    if hard and all(h['nominal'] for h in hard):
        return ('DISTRIBUTION-SHIFT-SURVIVES (EXPLORATORY)', '명목 통과, 가족단위 미달.')
    return ('SHIFT-COSTS-HEADLINE',
            '⛔분포를 이동시키면 헤드라인이 약해진다 ⟹ 단서 ③ 은 닫히지 않고, '
            '오히려 *"헤드라인이 학습 분포에 의존한다"* 는 방향의 증거다.')


# ── 5c. 기제 부차 — ★W12 가 지목한 **시험 시점** 자리에 닿는가 ──────────────
# W12: VARPRO 가 관측 목적함수를 63.2% 낮추는데 장 오차는 제로필드보다 나빴다.
# ⟹ 넷의 학습 손실을 바꿔도 **시험 시점 적합은 그대로**일 수 있다. 행마다 둘을 같이 기록해
#    "관측을 더 잘 맞출수록 장이 나빠지는" 관계가 칸별로 달라지는지 **기술**한다(바 없음).
MECH_FIELDS = ('varpro_obj_before', 'varpro_obj_after', 'nowcast', 'n_atoms', 'add_calls')


def mechanism(rows):
    out = {}
    for c in ARMS_REAL:
        rs = [r for r in rows if r.get('cell') == c and r.get('varpro_obj_before')]
        if not rs:
            continue
        drop = [1.0 - r['varpro_obj_after'] / r['varpro_obj_before'] for r in rs]
        out[c] = dict(n=len(rs), obs_obj_drop_median=st.median(drop),
                      nowcast_median=st.median([r['nowcast'] for r in rs]))
    return out


# ── 6. 게이트 ────────────────────────────────────────────────────────────────
def gates(man):
    """man = 실행 매니페스트(dict). 결과 0행에서도 형태가 검사되도록 전부 명시적."""
    o = []
    a = o.append
    a(('G-1 학습 3회·미세조정 5회만', man.get('n_pretrain') == 3 and man.get('n_finetune') == 5,
       f"{man.get('n_pretrain')}/{man.get('n_finetune')}"))
    a(('G-2 체크포인트 5개 sha256 상이', len(set(man.get('ckpt_sha256', {}).values())) == 5,
       str(sorted(man.get('ckpt_sha256', {})))))
    a(('★G-3 배포 체크포인트 무손상', man.get('shipped_sha256') ==
       '636c72a797ee6d0b725449293d5cd230dbd64c8eb03e345da4561b0c265239d9',
       str(man.get('shipped_sha256'))[:16]))
    a(('★G-4 팔이 실제로 그 체크포인트를 실었나(행별 되읽기)',
       man.get('rows_ckpt_mismatch') == 0, str(man.get('rows_ckpt_mismatch'))))
    a(('★G-5 실측 팔이 실제로 learned 로 갔나', man.get('rows_route_not_learned') == 0,
       str(man.get('rows_route_not_learned'))))
    a(('G-5b BANK 팔은 bank_mmse 로 갔나', man.get('bank_rows_wrong_route') == 0,
       str(man.get('bank_rows_wrong_route'))))
    a(('★G-6 표현 필터 항진 없음(HARD 학습 스트림)',
       man.get('hard_scenes_n', 0) > 0 and man.get('hard_scenes_below_repr', None) == 0,
       f"{man.get('hard_scenes_below_repr')}/{man.get('hard_scenes_n')}"))
    a(('★G-7 HARD 가 실제로 더 가파른가(학습 스트림 실측)',
       (man.get('hard_kHs_median') or 0) > (man.get('base_kHs_median') or 1),
       f"{man.get('hard_kHs_median')} vs {man.get('base_kHs_median')}"))
    a(('★G-8 앵커 물리 스크린 적용(k_p·Hs>0.44 제외, 평가에서는 미제외)',
       man.get('anchors_excluded') == ['LJ18a'] and
       sorted(man.get('eval_records', [])) == sorted(REAL_RECORDS),
       str(man.get('anchors_excluded'))))
    a(('G-9 학습 시드가 평가 시드와 분리', all(TRAIN_SEED_LO <= s < TRAIN_SEED_HI
                                          for s in man.get('train_scene_seeds_sample', [])),
       str(man.get('train_scene_seeds_sample', [])[:3])))
    a(('★G-10 복제 하네스 = 원본 장면 생성기 비트 동일',
       man.get('gen_parity_maxabs') == 0.0, str(man.get('gen_parity_maxabs'))))
    a(('★G-11 λ_field=0 팔이 진짜 no-op 아님(스텝 수 대조 확인)',
       man.get('lam0_finetune_steps') == FINETUNE_STEPS, str(man.get('lam0_finetune_steps'))))
    a(('G-12 전역 RNG 결합 계측(성장 발화 수 기록)', man.get('rows_missing_add_calls') == 0,
       str(man.get('rows_missing_add_calls'))))
    a(('G-13 참 제로필드 기준 사용(ZERO_REF=1.0 아님)',
       man.get('zero_ref_source') == 'tmp/B2_REAL_2026-08-23/zero_ref.py',
       str(man.get('zero_ref_source'))))
    a(('G-14 E3 는 M1 판정 동결 후', bool(man.get('m1_verdict_frozen')),
       str(man.get('m1_verdict_frozen'))))
    a(('G-15 동결 md5 대조', man.get('frozen_drift') == 0, str(man.get('frozen_drift'))))
    return o


# ── 7. ★결과 전 해석 규칙 (4갈래 + 중단 규칙) ───────────────────────────────
def label(res, hol, negc, r1):
    """⛔결과를 보기 전에 고정. 어떤 가지도 배포 기본값을 바꾸지 않는다(별도 사전등록 필요)."""
    if negc['aborts_inference']:
        return ('SEED-NOISE-DOMINATES',
                '음성대조(학습 시드만 다른 두 BASE)가 %d/%d 로 갈렸다 ⟹ E1 의 모든 대비는 '
                '**DESCRIPTIVE** 이고 요인 귀속을 하지 않는다.' % (negc['max_side'], negc['n']))
    L = hol['T-1_loss']['reject'] and res['T-1_loss']['wins_negative'] > res['T-1_loss']['n'] / 2
    D = hol['T-2_data']['reject'] and res['T-2_data']['wins_negative'] > res['T-2_data']['n'] / 2
    I = hol['T-3_interaction']['reject']
    if not r1['all_pass'] and (L or D):
        return ('TRADE',
                '실측에서 이득이 보이지만 합성 do-not-forget 바(+%.3f)를 깬 칸이 있다 ⟹ '
                '거래이지 개선이 아니다. 배포 권고 없음.' % R1_MARGIN)
    if L and not D:
        return ('SUPERVISION-EARNS',
                '★분포가 아니라 **감독**이 실측 기술을 만든다. wp_e2e 의 널은 옛 기저 조건이었다.')
    if D and not L:
        return ('DISTRIBUTION-EARNS',
                '★사용자 제안이 맞다 — 학습 분포가 실측 경계를 움직인다. 다음 = 분포 묶음 분해(M7b).')
    if L and D and not I:
        return ('BOTH-ADDITIVE', '두 축이 각각, 더하기로 작동한다.')
    if I:
        return ('INTERACTION', '교호가 유의 ⟹ 감독의 값은 학습 분포에 의존한다.')
    if not r1['all_pass']:
        return ('UNINTERPRETABLE-NULL',
                '⛔널이지만 합성 바를 깬 칸이 있다 ⟹ "넷을 망가뜨렸다"와 구별되지 않는다.')
    return ('NEITHER-MOVES-REAL',
            '⛔*"재학습이 무익"* 으로 쓰지 마라. 정확한 문장 = **"이 두 개입(분포 %s, 감독 %s)으로는 '
            '실측 경계가 움직이지 않았고, 이 설계의 MDE 는 12개 중 %d개 일치다"**. '
            '소거표의 한 줄이며, 합성 바를 통과했으므로 *넷이 망가져서 생긴 널이 아니다*.'
            % ('실측 앵커', 'λ_field=1.0', mde()['min_wins']))


if __name__ == '__main__':
    print('MDE', json.dumps(mde(), indent=1))
    p = premise()
    print('HARD k·Hs median', p['P1_capacity']['HARD']['k_Hs']['median'],
          'vs CURRENT', p['P1_capacity']['CURRENT']['k_Hs']['median'])
    print('HARD repr reject', p['P1_capacity']['HARD']['repr_frac']['frac_below_min'])
    print('real spread median', p['P2_real']['median_over_records']['spread_deg'])
    print('p(Tp<=real)', p['P3_support']['p_Tp_le_real'],
          'p(kHs>=real)', p['P3_support']['p_kHs_ge_real'])
