#!/usr/bin/env python3
"""`T5-ENS` 게이트·문턱·라벨 — **단일출처**. 러너·분석기는 여기를 호출만 한다.

사전등록 = `PREREG.md`(v2). 개정 사유 = `AMENDMENT_0_PREFREEZE.md`.
⛔이 파일의 수는 손복사가 아니라 **재도출**이다. 문턱을 문서에서 베끼지 마라.

★질문(§0): *"나우캐스트에서 측정된 앙상블 이득이 **예보 지평으로 전이되는가**?"*
"""
import ast
import json
import os
import statistics as st
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
B2 = os.path.abspath(os.path.join(HERE, '..', 'B2_REAL_2026-08-23'))
B3 = os.path.abspath(os.path.join(HERE, '..', 'B3_REFIT_2026-08-23'))
B7 = os.path.abspath(os.path.join(HERE, '..', 'B7_PROP_TRACK_2026-08-23'))
T3 = os.path.abspath(os.path.join(HERE, '..', 'T3_NOWCAST_2026-08-30'))
for _p in (HERE, B2, B3, B7):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ★셀 정의는 **임포트한다**. ⛔이 배치에서 재선언하지 않는다(PREREG §4.4).
from gates_b3 import RECORDS, WINDOWS, SEED, NOISE, N_PAIRS  # noqa: E402

BATCH = 'T5-ENS'
LEADS = [float(x) for x in range(15)]          # 0..14 s — B-7 규약
BAND_NEAR = (1.0, 3.0)
BAND_FAR = (4.0, 14.0)

# ─────────────────────────────────────────────────────────────────────────────
# 1. 팔 — ⛔전 키 명시. 빈 dict 로 팔을 정의하지 않는다(PREREG §4.2).
#    사고 서사(`B-7` 의 팔이 비어 있었다)는 PREREG 에만 둔다 —
#    `check_batch_artifacts` 의 `no-{}-arm` 열이 **주석까지** 정규식으로 훑기 때문이다.
# ─────────────────────────────────────────────────────────────────────────────
_BASE = dict(
    # 배포 삼종 + 표현
    varpro_steps=400, learned_init_iters=0, auto_switch_thresh=0,
    lambda_l1=1e-4, k_snap_dk=None, adaptive_n=True, max_n=30,
    # ★추적 = 이 배치의 처치 기계 (v1 이 빠뜨린 셋)
    tracking_mode='w_phi', tracking_steps=10, tracking_lr=0.005, tracking_window=3,
    # bank
    real_estimator='bank_mmse', bank_refit_every_s=None, bank_refit_async=False,
    bank_memory_s=40.0, bank_max_obs=600,
    # 기타
    use_confidence=True, seed=SEED, device='cpu',
    analytic_ensemble_k=3, real_bank_ensemble_k=3,
)

# ★`ROUTE_EXPECT` = 팔이 **실제로 도달해야 하는 경로**(요청한 `init_method` 가 아니다).
#   `BANK-K3` 는 `init_method='auto'` 를 요청하지만 실측 게이트가 `bank_mmse` 로 해소한다 —
#   그 해소가 곧 이 팔의 정의다. `PROP-M3` 는 `learned` 에 그대로 도달해야 한다
#   (⛔`v2_konly` 로 새면 그것이 이 게이트가 잡으려는 사고다).
#   ⚠스모크가 이 구분을 잡았다(동결 전) — 상세 = `AMENDMENT_0_PREFREEZE.md §7`.
ARMS = {
    'PROP-M3': dict(CFG=dict(_BASE, real_domain_gate=False, init_method='learned'),
                    MEM=(0, 1000, 2000), MEMKEY='learned_diffusion_seed', PATH='learned',
                    ROUTE_EXPECT='learned'),
    'BANK-K3': dict(CFG=dict(_BASE, real_domain_gate=True, init_method='auto'),
                    MEM=(0,), MEMKEY=None, PATH='bank',
                    ROUTE_EXPECT='bank_mmse'),
}
ARM_ORDER = ['PROP-M3', 'BANK-K3']


def arm_config(arm):
    return dict(ARMS[arm]['CFG'])


def member_overrides(arm, off):
    """멤버를 가르는 **유일한** 키. `T3 gates_t3.py:218` 축자 상속."""
    key = ARMS[arm]['MEMKEY']
    out = dict(seed=SEED)
    if key is not None:
        out[key] = SEED + int(off)
    return out


def expected_rows():
    return len(ARM_ORDER) * N_PAIRS


def expected_runs():
    return sum(len(ARMS[a]['MEM']) for a in ARM_ORDER) * N_PAIRS


def cells():
    return [(r, w) for r in RECORDS for w in WINDOWS]


# ★배포 트립와이어 — 매 셀 실행 시 러너가 부른다(단일출처).
#   ⛔`tracking_*` 셋을 포함한다: 나우캐스트 배치에선 죽은 키였지만
#   **예보 배치에선 처치 기계**다(심사 B·C).
DEPLOY_PIN = dict(
    varpro_steps=400, learned_init_iters=0, auto_switch_thresh=0, lambda_l1=1e-4,
    tracking_mode='w_phi', tracking_steps=10, tracking_lr=0.005, tracking_window=3,
    adaptive_n=True, max_n=30, use_confidence=True, k_snap_dk=None,
    real_bank_ensemble_k=3,
)

# ★출처가 **다른** 두 키 — `DEFAULT_CONFIG` 에 없고 `sequential_update.py` 안의
#   **인라인 폴백 리터럴**이다(`_bank_fit_window` 계열). 핀이 이것을 처음 드러냈다.
#   ⛔한 표에 섞으면 "출하 기본값" 이라는 말이 두 뜻을 갖는다 — 저장소의 반복 결함 부류다.
DEPLOY_PIN_FALLBACK = {
    'bank_memory_s': ("self.config.get('bank_memory_s', 40.0)", 40.0),
    'bank_max_obs': ("self.config.get('bank_max_obs', 600)", 600),
}


def assert_shipping_defaults():
    """출하 기본값 트립와이어. 어긋나면 `AssertionError`.

    두 갈래를 **따로** 검사한다:
      ① `DEFAULT_CONFIG` 에 있는 키 — 값 대조
      ② `sequential_update.py` 의 **인라인 폴백** — 소스 문자열 대조
    """
    import os as _os
    from wavesplat.run import DEFAULT_CONFIG
    bad = {k: (DEFAULT_CONFIG.get(k), v) for k, v in DEPLOY_PIN.items()
           if DEFAULT_CONFIG.get(k) != v}
    if bad:
        raise AssertionError(f'배포 기본값 드리프트(DEFAULT_CONFIG): {bad}')
    su = _os.path.join(_os.path.abspath(_os.path.join(HERE, '..', '..')),
                       'wavesplat', 'optimizer', 'sequential_update.py')
    src = open(su, encoding='utf-8').read()
    miss = [k for k, (needle, _) in DEPLOY_PIN_FALLBACK.items() if needle not in src]
    if miss:
        raise AssertionError(f'배포 기본값 드리프트(인라인 폴백): {miss}')
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 2. 문턱 · MDE · Holm — ⛔전부 재도출 (PREREG §4.5)
# ─────────────────────────────────────────────────────────────────────────────
ALPHA = 0.05
N_FAMILY = 2                      # PRIMARY(NEAR 전이) · SEC-A(FAR 전이)


def sign_test_one_sided(w, n):
    return sum(comb(n, k) for k in range(w, n + 1)) / 2 ** n


def _min_w(n, alpha):
    return next(w for w in range(n + 1) if sign_test_one_sided(w, n) <= alpha)


def thresholds(n=N_PAIRS):
    """단일검정 문턱 + Holm 단계별 최소 승수.

    ⚠**`NOT_EARN` 은 `n − HOLM_FIRST` 다** — `T3` 의 `n − NOMINAL` 보다 **엄격**하다.
    사유: 심사 A 가 v1 의 비대칭을 잡았다(양성엔 가족 보정, 음성엔 단일 α).
    역방향 꼬리를 주장하려면 양성과 **같은 α** 를 넘어야 한다.
    """
    nominal = _min_w(n, ALPHA)
    holm = [_min_w(n, ALPHA / (N_FAMILY - r)) for r in range(N_FAMILY)]
    return dict(n=n, NOMINAL=nominal, HOLM_MIN_WINS=holm, HOLM_FIRST=holm[0],
                NOT_EARN=n - holm[0],
                p_nominal=sign_test_one_sided(nominal, n),
                p_holm_first=sign_test_one_sided(holm[0], n),
                alpha=ALPHA, n_family=N_FAMILY,
                family_alpha_first=ALPHA / N_FAMILY)


def mde(n=N_PAIRS, power=0.80, thresh=None):
    """★문턱에서 `power` 를 내는 최소 참 승률. ⛔판정문에 **반드시 병기**한다."""
    t = thresholds(n)['HOLM_FIRST'] if thresh is None else thresh
    q = 0.5
    while sum(comb(n, k) * q ** k * (1 - q) ** (n - k) for k in range(t, n + 1)) < power:
        q += 0.001
    return dict(n=n, thresh=t, power=power, q80=round(q, 3))


def power_at(q, n=N_PAIRS, thresh=None):
    """★결과 전 예측 하의 검정력 (PREREG §3). v1 이 이것을 계산하지 않아 죽었다."""
    t = thresholds(n)['HOLM_FIRST'] if thresh is None else thresh
    return sum(comb(n, k) * q ** k * (1 - q) ** (n - k) for k in range(t, n + 1))


def holm_survivors(pvals):
    """Holm 단계별 생존 — 입력 {이름: p}. 반환 {이름: bool}."""
    order = sorted(pvals.items(), key=lambda kv: (kv[1] is None, kv[1]))
    out, still = {}, True
    for r, (name, p) in enumerate(order):
        a = ALPHA / (len(order) - r)
        still = still and (p is not None and p <= a)
        out[name] = still
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. 집계 — ⛔규약은 `gates_t3.paired:375` 의 `st.median` 하나뿐이다.
# ─────────────────────────────────────────────────────────────────────────────
def key(r):
    return (r.get('record'), r.get('win'))


def band_mean(per_lead, band):
    lo, hi = band
    v = [x for k, x in (per_lead or {}).items() if lo <= float(k) <= hi and x is not None]
    return st.mean(v) if v else None


def _empty(n, arm, ref):
    return dict(arm=arm, ref=ref, n=n, better=None, ties=None,
                rel_median_pct=None, p_one_sided=None)


def _map(rows, arm, field):
    return {key(r): r.get(field) for r in rows
            if r.get('arm') == arm and r.get(field) is not None}


def paired(rows, arm_a, field_a, arm_b, field_b):
    """짝 부호검정 `a < b`. ⛔0행 가드: 겹치는 셀이 없으면 카운터가 **None**."""
    a, b = _map(rows, arm_a, field_a), _map(rows, arm_b, field_b)
    ks = sorted(set(a) & set(b))
    n = len(ks)
    if n == 0:
        return _empty(0, f'{arm_a}.{field_a}', f'{arm_b}.{field_b}')
    w = sum(1 for k in ks if a[k] < b[k])
    return dict(arm=f'{arm_a}.{field_a}', ref=f'{arm_b}.{field_b}', n=n, better=w,
                ties=sum(1 for k in ks if a[k] == b[k]),
                rel_median_pct=st.median([(a[k] - b[k]) / b[k] * 100 for k in ks]),
                p_one_sided=sign_test_one_sided(w, n))


def vs_zero(rows, arm, field, zfield):
    """끝점: 팔이 **참 제로필드**를 이기는 셀 수. ⛔고정 상수 기준 금지."""
    R = [r for r in rows if r.get('arm') == arm
         and r.get(field) is not None and r.get(zfield) is not None]
    n = len(R)
    if n == 0:
        return _empty(0, f'{arm}.{field}', 'TRUE-ZERO')
    w = sum(1 for r in R if r[field] < r[zfield])
    return dict(arm=f'{arm}.{field}', ref='TRUE-ZERO', n=n, better=w,
                ties=sum(1 for r in R if r[field] == r[zfield]),
                rel_median_pct=st.median([(r[field] - r[zfield]) / r[zfield] * 100
                                          for r in R]),
                p_one_sided=sign_test_one_sided(w, n))


# ─────────────────────────────────────────────────────────────────────────────
# 4. 라벨 — 결과 전 확정. ★공역 전건 정의(v1 은 w=25 에 구멍이 있었다).
# ─────────────────────────────────────────────────────────────────────────────
CLAIMS = {
    'PRIMARY': dict(subject='PROP-M3.band_near_med3', ref='PROP-M3.band_near_m0',
                    pos='ENSEMBLE-TRANSFERS', neg='TRANSFER-REVERSED',
                    und='TRANSFER-UNDECIDED'),
    'SEC-A': dict(subject='PROP-M3.band_far_med3', ref='PROP-M3.band_far_m0',
                  pos='ENSEMBLE-TRANSFERS-FAR', neg='TRANSFER-REVERSED-FAR',
                  und='TRANSFER-UNDECIDED-FAR'),
}


def grade(v, surv):
    """(등급, 문턱표). ⛔`CONFIRMATORY` 는 **Holm 생존**일 때만."""
    t = thresholds(v['n'] or N_PAIRS)
    w = v['better']
    if w is None:
        return 'NOROWS', t
    if surv:
        return 'CONFIRMATORY', t
    if w >= t['NOMINAL']:
        return 'EXPLORATORY', t
    if w <= t['NOT_EARN']:
        return 'REVERSE', t
    return 'UNDECIDED', t


def label_for(claim, v, surv):
    """★`w ∈ [0,n] × {surv, ¬surv}` 전 조합에서 라벨이 정의된다(selftest 가 단언)."""
    c = CLAIMS[claim]
    g, t = grade(v, surv)
    if g == 'NOROWS':
        return None, g
    if g == 'CONFIRMATORY':
        return f'{BATCH} :: {c["pos"]} (CONFIRMATORY)', g
    if g == 'EXPLORATORY':
        return f'{BATCH} :: {c["pos"]} [EXPLORATORY]', g
    if g == 'REVERSE':
        # ⚠역방향 꼬리 — 가족 α 는 단측 사전등록 방향에만 썼다 ⟹ 확증 승격 금지.
        return f'{BATCH} :: {c["neg"]} [EXPLORATORY]', g
    return f'{BATCH} :: {c["und"]}', g


READING = {
    'UNDECIDED': ('나우캐스트에서 측정된 앙상블 이득이 예보 지평으로 전이되는 것은 '
                  '**검출되지 않았다**. ⛔"차이 없다"·"전이되지 않는다" 금지. '
                  '★귀결: 처치가 예보에서 힘이 없다면 앙상블 교락은 B-7·B-7b·W-12 의 '
                  '부호를 뒤집을 수 없다 ⟹ 교락이 「측정된 공시 한 줄」로 내려간다.'),
    'CONFIRMATORY': ('앙상블 이득이 예보로 전이된다 ⟹ 교락에 이빨이 있다. '
                     'EX-1/EX-2 는 **탐색적으로만** 읽고 확증은 새 사전등록으로.'),
    'EXPLORATORY': '명목 문턱은 넘었으나 가족 보정에서 탈락 ⟹ 확증 아님.',
    'REVERSE': ('⚠앙상블이 예보에서 **해친다**(T3 나우캐스트와 방향 반대). '
                '⛔확증 승격 금지 · 미해결로 기록.'),
}

DISCLOSURE = [
    '연구 팔 — 패키지에 학습 앙상블 경로 없음(PREREG §1 P1)',
    'blind 아님 — 결과 전 예측 + 검정력 표를 PREREG §3 에 실었다',
    '노출 — 나우캐스트 답을 사전등록 전에 봤고, 그것이 NEAR 밴드를 81~89% 결정한다(§2.3)',
    'BANK-K3 는 출하 기본값이고 B-7 의 bank 팔과 다른 팔이다',
    f'MDE q80 = {mde()["q80"]} 병기 의무',
    'config["device"] 는 제안 경로에 도달하지 않는다(§1 P7) ⟹ 앵커가 장치를 정한다',
    '사이트 4곳 · 레코드 12개 ⟹ 사이트 수준 일반화 금지',
]

# ─────────────────────────────────────────────────────────────────────────────
# 5. `G-FLAT` — ★토큰은 **조립**한다(리터럴 금지 = T2 G-7b 재연 방지).
#    ⛔`.py` 만 · `ast.Name`/`Attribute` 만 · 산문 통과 ·
#    ⛔`os.listdir` 금지 — **동결 목록에 적힌 파일만** 훑는다(소급 FAIL 방지).
# ─────────────────────────────────────────────────────────────────────────────
_FLAT_NAME = 'ZERO' + '_REF'


def _frozen_py_files():
    """`FROZEN_md5.txt` 에 적힌 배치-내 `.py` 만. 동결 전이면 배치 내 판정코드."""
    fz = os.path.join(HERE, 'FROZEN_md5.txt')
    names = []
    if os.path.exists(fz):
        for ln in open(fz):
            ln = ln.strip()
            if not ln or ln.startswith('#'):
                continue
            parts = ln.split(None, 1)
            if len(parts) == 2 and parts[1].endswith('.py') and '/' not in parts[1]:
                names.append(parts[1])
    else:
        names = [f for f in sorted(os.listdir(HERE))
                 if f.endswith('.py') and f.split('_')[0] in ('gates', 'run', 'analyze',
                                                              'premise', 'stageA')]
    return [os.path.join(HERE, n) for n in dict.fromkeys(names)]


def flat_ref_hits(paths=None):
    """고정 기준상수 사용 지점. **이름 노드만** 본다 — 문자열·주석·산문은 통과."""
    hits = []
    for p in (paths if paths is not None else _frozen_py_files()):
        if not os.path.exists(p):
            continue
        try:
            tree = ast.parse(open(p, encoding='utf-8').read())
        except SyntaxError:
            hits.append((os.path.basename(p), -1, 'SYNTAX'))
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == _FLAT_NAME:
                hits.append((os.path.basename(p), node.lineno, node.id))
            elif isinstance(node, ast.Attribute) and node.attr == _FLAT_NAME:
                hits.append((os.path.basename(p), node.lineno, node.attr))
    return hits


# ─────────────────────────────────────────────────────────────────────────────
# 6. 게이트
# ─────────────────────────────────────────────────────────────────────────────
BLOCKING = ['G-APPROVAL', 'G-ARM', 'G-DEPLOY-PIN', 'G-MEMBER', 'G-MEDIAN-RULE',
            'G-ANCHOR-CURVE', 'G-ANCHOR-BANK', 'G-SCENE', 'G-ROUTE', 'G-DEVICE',
            'G-REFIT0', 'G-INTERVENTION', 'G-FLAT', 'G-ROWS']


def _anchor_b7():
    """동결 `B-7 PROP-TRACK` 의 per_lead 곡선 (멤버 0 앵커)."""
    out = {}
    p = os.path.join(B7, 'results_b7.jsonl')
    if os.path.exists(p):
        for ln in open(p):
            if ln.strip():
                r = json.loads(ln)
                if r.get('cell') == 'PROP-TRACK':
                    out[(r['record'], r['win'])] = r.get('per_lead') or {}
    return out


def _anchor_t3():
    """동결 `T3 SHIP-K3.e_m0` (bank lead0 앵커) + `sigma`/`npx` 장면 대조용."""
    out = {}
    import glob
    for f in sorted(glob.glob(os.path.join(T3, 'results_t3*.jsonl'))):
        for ln in open(f):
            if ln.strip():
                r = json.loads(ln)
                if r.get('arm') == 'SHIP-K3':
                    out[(r['record'], r['win'])] = r
    return out


def gates(rows, premise=None):
    """차단·범위 게이트 전건. 반환 [(이름, bool, 상세)].

    ⚠시그니처를 `gates(rows, premise=None)` 로 고정한다 —
    `check_gate_blindspots` 가 부를 수 없으면 그 검사를 **조용히 건너뛴다**.
    """
    G = []

    def A(name, ok, detail=''):
        G.append((name, bool(ok), detail))

    ok = [r for r in rows if 'error' not in r]
    have = bool(ok)
    t = thresholds()

    # ── G-APPROVAL
    A('G-APPROVAL 승인 스탬프 전건', have and all(r.get('approved') is True for r in ok),
      f'rows={len(ok)}')

    # ── G-ARM : ⛔의도표 자기대조 금지 — **되읽기 == 의도**
    arm_ok, arm_d = have, []
    for r in ok:
        cfg = ARMS.get(r.get('arm'), {}).get('CFG')
        if cfg is None:
            arm_ok = False
            arm_d.append(f'unknown arm {r.get("arm")}')
            continue
        if not r.get('arm_cfg_nonempty'):
            arm_ok = False
            arm_d.append(f'{r["arm"]} empty cfg')
    A('G-ARM 팔 config 비어있지 않음 · 되읽기 일치', arm_ok, ';'.join(arm_d[:3]))

    # ── G-DEPLOY-PIN (추적 3키 포함)
    A('G-DEPLOY-PIN 출하 기본값 트립와이어(추적 3키 포함)',
      have and all(r.get('deploy_pin_ok') is True for r in ok))

    # ── G-MEMBER
    m_ok, m_d = have, []
    for r in ok:
        a = r.get('arm')
        want = [member_overrides(a, o) for o in ARMS[a]['MEM']] if a in ARMS else []
        ws = [w['seed'] for w in want]
        wd = [w.get(ARMS[a]['MEMKEY']) if ARMS[a]['MEMKEY'] else None for w in want]
        if list(r.get('seed_readback') or []) != ws:
            m_ok = False
            m_d.append(f'{a} seed {r.get("seed_readback")}!={ws}')
        if list(r.get('diffusion_seed_readback') or []) != wd:
            m_ok = False
            m_d.append(f'{a} dseed {r.get("diffusion_seed_readback")}!={wd}')
    A('G-MEMBER 멤버 시드 규약 == member_overrides()', m_ok, ';'.join(m_d[:3]))

    # ── G-MEDIAN-RULE : 합성 연산자가 「장 median → 채점」인가
    #    ⛔「점수 median」이면 per_lead_med 가 멤버 점수들의 median 과 **항상** 같다.
    P = [r for r in ok if r.get('arm') == 'PROP-M3']
    differs = 0
    for r in P:
        pm, pmem = r.get('per_lead_med') or {}, r.get('per_lead_members') or {}
        for l_, v in pm.items():
            mem = pmem.get(l_)
            if mem and len(mem) >= 3 and v is not None and st.median(mem) != v:
                differs += 1
                break
    A('G-MEDIAN-RULE 장 median 후 채점 (≠ 점수 median)',
      bool(P) and differs > 0 and all((r.get('field_md5_med_by_lead') or {}) for r in P),
      f'differs={differs}/{len(P)}')

    # ── G-ANCHOR-CURVE : 멤버 0 의 15-lead 곡선 전건 비트 동일
    anc = _anchor_b7()
    c_ok, c_n, c_bad = have, 0, []
    for r in P:
        ref = anc.get(key(r))
        cur = r.get('per_lead_m0') or {}
        if not ref or not cur:
            c_ok = False
            c_bad.append(f'{key(r)} no-ref')
            continue
        for l_ in LEADS:
            k = str(l_)
            if k in ref and k in cur:
                c_n += 1
                if ref[k] != cur[k]:
                    c_ok = False
                    c_bad.append(f'{key(r)}@{k}')
    A('G-ANCHOR-CURVE 멤버0 15-lead 곡선 == 동결 B-7 (비트)', c_ok and c_n > 0,
      f'compared={c_n} bad={c_bad[:3]}')

    # ── G-ANCHOR-BANK : BANK lead0 == 동결 T3 SHIP-K3.e_m0
    t3 = _anchor_t3()
    B = [r for r in ok if r.get('arm') == 'BANK-K3']
    b_ok, b_n = bool(B), 0
    for r in B:
        ref = t3.get(key(r))
        cur = (r.get('per_lead_med') or {}).get('0.0')
        if ref is None or cur is None or ref.get('e_m0') != cur:
            b_ok = False
        else:
            b_n += 1
    A('G-ANCHOR-BANK BANK lead0 == 동결 T3 SHIP-K3 (비트)', b_ok and b_n == len(B),
      f'{b_n}/{len(B)}')

    # ── G-SCENE : (a) 행 대조는 T3 에 **실재하는** sigma/npx 로 한정
    #              (b) depth/n_obs 는 **원천 항등**(premise 가 재호출로 확인)
    s_ok, s_bad = have, []
    for r in ok:
        ref = t3.get(key(r))
        if ref is None:
            continue
        for f in ('sigma', 'npx'):
            if ref.get(f) is not None and r.get(f) is not None and ref[f] != r[f]:
                s_ok = False
                s_bad.append(f'{key(r)}.{f}')
    src = (premise or {}).get('scene_source_identity')
    A('G-SCENE 장면 상수 (행: sigma·npx / 원천: depth·n_obs)',
      s_ok and src is True, f'row_bad={s_bad[:3]} source_identity={src}')

    # ── G-ROUTE : 팔이 의도한 경로로 갔는가 (v2_konly 누출 탐지)
    r_ok, r_bad = have, []
    for r in ok:
        a = r.get('arm')
        cfg = ARMS.get(a, {}).get('CFG', {})
        want_route = ARMS.get(a, {}).get('ROUTE_EXPECT')
        if r.get('init_method_readback') != want_route:
            r_ok = False
            r_bad.append(f'{a} route={r.get("init_method_readback")}!={want_route}')
        if r.get('real_gate') != cfg.get('real_domain_gate'):
            r_ok = False
            r_bad.append(f'{a} gate={r.get("real_gate")}')
        # ★bank 팔은 **bank 자신의** median-of-k 가 살아 있어야 한다.
        #   ⛔`ens_k_readback` 은 해석적 래퍼의 k 라 실측 게이트에서 1 로 접힌다 — 그것을 보면 안 된다.
        if ARMS.get(a, {}).get('PATH') == 'bank':
            want_k = cfg.get('real_bank_ensemble_k')
            if r.get('bank_ens_k_readback') != want_k:
                r_ok = False
                r_bad.append(f'{a} bank_k={r.get("bank_ens_k_readback")}!={want_k}')
    A('G-ROUTE 라우팅 되읽기 == 의도 (v2_konly 누출 0)', r_ok, ';'.join(r_bad[:3]))

    # ── G-DEVICE : 전 행 단일 스탬프
    def uniq(f):
        return {json.dumps(r.get(f), sort_keys=True) for r in ok}
    d_ok = have and all(len(uniq(f)) == 1 for f in
                        ('device', 'cuda_available', 'omp_threads', 'prior_device',
                         'ckpt_sha'))
    A('G-DEVICE 장치·스레드·체크포인트 단일 스탬프', d_ok,
      f'device={uniq("device")} cuda={uniq("cuda_available")} omp={uniq("omp_threads")}')

    # ── G-REFIT0 : bank_refit_every_s=None 이 실제로 먹었는가
    A('G-REFIT0 재적합 호출 0 전건',
      have and all((r.get('refit_calls') or 0) == 0 for r in ok),
      f'max={max([(r.get("refit_calls") or 0) for r in ok], default=None)}')

    # ── G-INTERVENTION : 멤버별 추적 스텝 > 0
    i_ok = bool(P) and all(bool(r.get('track_steps_members'))
                           and all(x and x > 0 for x in r['track_steps_members'])
                           for r in P)
    A('G-INTERVENTION PROP 추적 스텝 > 0 (멤버별 전건)', i_ok,
      f'n_prop={len(P)}')

    # ── G-FLAT
    hits = flat_ref_hits()
    A('G-FLAT 고정 기준상수 사용 0건 (ast · .py · 동결목록만)', not hits, f'{hits[:3]}')

    # ── G-ROWS
    cov = {a: len({key(r) for r in ok if r.get('arm') == a}) for a in ARM_ORDER}
    A('G-ROWS 팔별 셀 커버리지 전건',
      have and all(cov.get(a) == N_PAIRS for a in ARM_ORDER) and len(ok) == expected_rows(),
      f'{cov} rows={len(ok)}/{expected_rows()}')

    # ── 범위 축소 (⛔blocked_by 에 넣지 않는다)
    deg = sum(1 for r in P if len(set(map(str, (r.get('field_md5_members_by_lead') or {})
                                          .get('0.0', [])))) < 3)
    A('H-DEGEN 멤버 퇴화 셀 공시', True, f'degenerate={deg}/{len(P)}')
    A('H-LEADCOV 밴드 lead 커버리지',
      all(len(r.get('per_lead_med') or {}) == len(LEADS) for r in ok) if have else False,
      f'want={len(LEADS)}')
    A('H-SEED 새 에피소드 시드 청구 0 (설계상)', True, 'SEED=42 고정 · 실측 레코드')
    return G


def gate_id(name):
    """게이트 이름의 **식별자 부분**. ⛔`G` 전체 문자열을 `BLOCKING` 과 비교하지 마라."""
    return name.split()[0]


def blocked_by(G):
    """차단 게이트 중 FAIL 한 것.

    ⛔초판은 `n in BLOCKING` 이었다 — `n` 은 `'G-ROWS 팔별 셀 커버리지 전건'` 이고
    `BLOCKING` 은 `'G-ROWS'` 라서 **항상 빈 리스트**를 냈다(= 어떤 FAIL 도 판정을 막지 못한다).
    ★자가진단 4·5항(팔 삭제 민감도 · 0행 가드)이 동결 전에 잡았다.
    """
    return [gate_id(n) for n, ok, _ in G if not ok and gate_id(n) in BLOCKING]


# ─────────────────────────────────────────────────────────────────────────────
# 7. 분석 — 러너·분석기가 부르는 단일 진입점
# ─────────────────────────────────────────────────────────────────────────────
def analyze(rows):
    ok = [r for r in rows if 'error' not in r]
    v = {
        'PRIMARY': paired(ok, 'PROP-M3', 'band_near_med3', 'PROP-M3', 'band_near_m0'),
        'SEC-A': paired(ok, 'PROP-M3', 'band_far_med3', 'PROP-M3', 'band_far_m0'),
    }
    surv = holm_survivors({k: x['p_one_sided'] for k, x in v.items()})
    labels, grades = {}, {}
    for k in CLAIMS:
        labels[k], grades[k] = label_for(k, v[k], surv.get(k, False))
    ex = {
        'EX-1-NEAR': paired(ok, 'BANK-K3', 'band_near_med3', 'PROP-M3', 'band_near_med3'),
        'EX-1-FAR': paired(ok, 'BANK-K3', 'band_far_med3', 'PROP-M3', 'band_far_med3'),
        'EX-2-NEAR': vs_zero(ok, 'PROP-M3', 'band_near_med3', 'zero_near'),
        'EX-2-FAR': vs_zero(ok, 'PROP-M3', 'band_far_med3', 'zero_far'),
        'EX-3-NEAR': vs_zero(ok, 'BANK-K3', 'band_near_med3', 'zero_near'),
        'EX-3-FAR': vs_zero(ok, 'BANK-K3', 'band_far_med3', 'zero_far'),
    }
    return dict(batch=BATCH, contrasts=v, holm=surv, labels=labels, grades=grades,
                exploratory=ex, thresholds=thresholds(), mde=mde(),
                reading={k: READING.get(g, '') for k, g in grades.items()},
                disclosure=DISCLOSURE, n_rows=len(ok))


# ─────────────────────────────────────────────────────────────────────────────
# 8. ★자가진단 (동결 **전** 실행) — PREREG §7.3
#    `T2 G-7b` 교훈: 게이트는 **통과 가능해야 한다**. 그리고 **항진이어서도 안 된다**.
# ─────────────────────────────────────────────────────────────────────────────
_PREMISE_OK = {'scene_source_identity': True}


def _fixture():
    """동결 앵커에서 **합성 픽스처**를 만든다 — 게이트가 통과 가능함을 먼저 증명한다.

    ★앵커 유래이므로 `G-ANCHOR-*` 는 구성상 통과한다. 나머지 게이트는 값을 채워 맞춘다.
    ⛔이 픽스처는 시험 전용이고 결과 파일이 아니다.
    """
    anc, t3 = _anchor_b7(), _anchor_t3()
    rows = []
    for (rec, win) in cells():
        ref = anc.get((rec, win)) or {}
        r3 = t3.get((rec, win)) or {}
        if not ref or not r3:
            continue
        med = {k: v * 0.97 for k, v in ref.items()}          # 앙상블이 조금 낫다
        members = {k: [v, v * 1.02, v * 0.99] for k, v in ref.items()}
        z = {k: v * 1.6 for k, v in ref.items()}
        base = dict(batch=BATCH, record=rec, win=win, approved=True,
                    arm_cfg_nonempty=True, deploy_pin_ok=True,
                    per_lead_members=members,
                    zero_per_lead=z,
                    zero_near=band_mean(z, BAND_NEAR), zero_far=band_mean(z, BAND_FAR),
                    refit_calls=0, device='cpu', cuda_available=True, omp_threads=1,
                    prior_device='cpu', ckpt_sha='deadbeef',
                    sigma=r3.get('sigma'), npx=r3.get('npx'),
                    depth=17.0, n_obs=600,
                    field_md5_med_by_lead={k: f'm{k}' for k in ref},
                    field_md5_members_by_lead={k: [f'a{k}', f'b{k}', f'c{k}'] for k in ref})
        rows.append(dict(base, arm='PROP-M3', n_members=3, member_offsets=[0, 1000, 2000],
                         per_lead_m0=dict(ref), per_lead_med=med,
                         band_near_m0=band_mean(ref, BAND_NEAR),
                         band_near_med3=band_mean(med, BAND_NEAR),
                         band_far_m0=band_mean(ref, BAND_FAR),
                         band_far_med3=band_mean(med, BAND_FAR),
                         seed_readback=[SEED, SEED, SEED],
                         diffusion_seed_readback=[SEED, SEED + 1000, SEED + 2000],
                         init_method_readback='learned', real_gate=False,
                         real_estimator_readback='bank_mmse',
                         bank_ens_k_readback=None, ens_k_readback=None,
                         track_steps_members=[14, 14, 14]))
        bl0 = r3.get('e_m0')
        bmed = {k: (bl0 if k == '0.0' else v * 0.8) for k, v in ref.items()}
        rows.append(dict(base, arm='BANK-K3', n_members=1, member_offsets=[0],
                         per_lead_m0=bmed, per_lead_med=bmed,
                         per_lead_members={k: [v] for k, v in bmed.items()},
                         band_near_m0=band_mean(bmed, BAND_NEAR),
                         band_near_med3=band_mean(bmed, BAND_NEAR),
                         band_far_m0=band_mean(bmed, BAND_FAR),
                         band_far_med3=band_mean(bmed, BAND_FAR),
                         seed_readback=[SEED], diffusion_seed_readback=[None],
                         init_method_readback='bank_mmse', real_gate=True,
                         real_estimator_readback='bank_mmse',
                         bank_ens_k_readback=3, ens_k_readback=1,
                         track_steps_members=[14],
                         field_md5_members_by_lead={k: [f'a{k}'] for k in ref}))
    return rows


# ★게이트마다 **위조 1건**. 위조를 넣으면 그 게이트가 반드시 FAIL 해야 한다(항진 금지).
def _forge(rows, what):
    R = [dict(r) for r in rows]
    if what == 'G-APPROVAL':
        R[0]['approved'] = False
    elif what == 'G-ARM':
        R[0]['arm_cfg_nonempty'] = False
    elif what == 'G-DEPLOY-PIN':
        R[0]['deploy_pin_ok'] = False
    elif what == 'G-MEMBER':
        R[0]['diffusion_seed_readback'] = [SEED, SEED + 999, SEED + 2000]
    elif what == 'G-MEDIAN-RULE':
        for r in R:                                  # 「점수 median」으로 바꾼다
            if r['arm'] == 'PROP-M3':
                r['per_lead_med'] = {k: st.median(v)
                                     for k, v in r['per_lead_members'].items()}
    elif what == 'G-ANCHOR-CURVE':
        for r in R:
            if r['arm'] == 'PROP-M3':
                r['per_lead_m0'] = {k: v * 1.000001 for k, v in r['per_lead_m0'].items()}
                break
    elif what == 'G-ANCHOR-BANK':
        for r in R:
            if r['arm'] == 'BANK-K3':
                r['per_lead_med'] = dict(r['per_lead_med'], **{'0.0': -1.0})
                break
    elif what == 'G-SCENE':
        R[0]['sigma'] = (R[0].get('sigma') or 1.0) + 1.0
    elif what == 'G-ROUTE':
        for r in R:
            if r['arm'] == 'PROP-M3':
                r['init_method_readback'] = 'v2_konly'   # ★누출 시나리오
                break
    elif what == 'G-DEVICE':
        R[0]['cuda_available'] = False
    elif what == 'G-REFIT0':
        R[0]['refit_calls'] = 3
    elif what == 'G-INTERVENTION':
        for r in R:
            if r['arm'] == 'PROP-M3':
                r['track_steps_members'] = [14, 0, 14]
                break
    elif what == 'G-ROWS':
        R = R[:-1]
    else:
        return None
    return R


# ★행 위조로 시험할 수 없는 게이트 + **그 대신 무엇이 덮는가**.
#   ⛔사문화 방지: `selftest` 가 「대체 시험이 실제로 존재하고 통과했는가」를 단언한다.
FORGERY_EXEMPT = {
    'G-FLAT': '파일 수준 게이트 — 행 위조로 못 건드린다. 대체 시험 = selftest 7항'
              '(`_g_flat_probe.py` 를 실제로 써넣어 탐지되는지 확인).',
}


def selftest(verbose=True):
    """PREREG §7.3 의 7항. ⛔실패하면 동결하지 마라."""
    fails = []

    def chk(name, ok, detail=''):
        if verbose:
            print(f'  [{"OK " if ok else "FAIL"}] {name}  {detail}')
        if not ok:
            fails.append(name)

    print('── §7.3 게이트 자가진단 ──────────────────────────────────────────')

    # 1. 합성 픽스처로 현 상태 전건 PASS  ← ★게이트가 통과 **가능**한가
    rows = _fixture()
    chk('1 픽스처 생성', len(rows) == expected_rows(), f'{len(rows)}/{expected_rows()}')
    G = gates(rows, premise=_PREMISE_OK)
    bad = [n for n, ok, d in G if not ok]
    chk('1 픽스처에서 전 게이트 PASS', not bad, f'fail={bad}')

    # 2·3. 게이트별 위조 → 반드시 FAIL · uncovered == []
    uncovered = []
    for g in BLOCKING:
        F = _forge(rows, g)
        if F is None:
            if g not in FORGERY_EXEMPT:
                uncovered.append(g)
            continue
        got = dict((gate_id(n), ok) for n, ok, _ in gates(F, premise=_PREMISE_OK))
        chk(f'2 위조탐침 {g} → FAIL', got.get(g) is False, f'got={got.get(g)}')
    chk('3 위조 없는 차단 게이트 0개 (uncovered)', not uncovered, f'{uncovered}')
    # ★면제의 사문화 방지 — 면제된 게이트는 (a) 실재하고 (b) 사유가 적혀 있어야 한다.
    chk('3b 면제 게이트가 전부 차단 목록에 실재 · 사유 기재',
        all(g in BLOCKING and FORGERY_EXEMPT[g].strip() for g in FORGERY_EXEMPT),
        f'{list(FORGERY_EXEMPT)}')

    # 4. 팔 삭제 민감도 — 어느 팔을 지워도 반드시 어떤 차단 게이트가 FAIL (항등 게이트 0)
    for a in ARM_ORDER:
        D = [r for r in rows if r['arm'] != a]
        chk(f'4 팔 삭제 민감도 ({a})', bool(blocked_by(gates(D, premise=_PREMISE_OK))))

    # 5. 0행 가드
    G0 = gates([], premise=_PREMISE_OK)
    A0 = analyze([])
    chk('5 0행: 차단 게이트가 FAIL', bool(blocked_by(G0)))
    chk('5 0행: 라벨 None', all(v is None for v in A0['labels'].values()),
        f'{A0["labels"]}')
    chk('5 0행: 카운터 None', all(A0['contrasts'][k]['better'] is None for k in CLAIMS))

    # 6. ★라벨 공역 — w ∈ [0,n] × {surv, ¬surv} 전 조합에서 정의됨 (v1 의 w=25 구멍)
    holes = []
    for w in range(N_PAIRS + 1):
        for surv in (True, False):
            v = dict(n=N_PAIRS, better=w, p_one_sided=sign_test_one_sided(w, N_PAIRS))
            lab, g = label_for('PRIMARY', v, surv)
            if lab is None or g not in READING:
                holes.append((w, surv, lab, g))
    chk('6 라벨 공역 전건 정의 (74 조합)', not holes, f'holes={holes[:3]}')

    # 7. G-FLAT 항진 검사 — 위조 파일을 넣으면 반드시 잡혀야 한다
    probe = os.path.join(HERE, '_g_flat_probe.py')
    hits0 = flat_ref_hits()
    try:
        open(probe, 'w').write(f'{_FLAT_NAME} = 1.0\nx = {_FLAT_NAME}\n')
        hits1 = flat_ref_hits(paths=[probe])
        chk('7 G-FLAT 현 상태 0건', not hits0, f'{hits0[:3]}')
        chk('7 G-FLAT 위조 탐지', bool(hits1), f'{hits1[:2]}')
    finally:
        if os.path.exists(probe):
            os.remove(probe)

    print('──────────────────────────────────────────────────────────────────')
    if fails:
        print(f'⛔자가진단 실패 {len(fails)}건: {fails}')
        return 1
    print(f'✅자가진단 전건 통과 · 차단 게이트 {len(BLOCKING)} · 위조탐침 {len(BLOCKING)}')
    print(f'   문턱 {thresholds()} ')
    print(f'   MDE {mde()}  |  검정력 q=0.70 → {power_at(0.70):.4f}')
    return 0


if __name__ == '__main__':
    sys.exit(selftest())
