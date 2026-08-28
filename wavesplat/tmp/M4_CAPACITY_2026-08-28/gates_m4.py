"""M4 단일출처 — **표현의 *용량* 이 실측 경계인가** (2026-08-28, 결과 0행에서 동결).

    python3 tmp/M4_CAPACITY_2026-08-28/gates_m4.py --selftest    # 런 0회·결과 0개

이 파일이 **유일하게** 정의한다: 용량의 정의 · 팔 · 표본 · 문턱 · 라벨 · 결과 전 해석 규칙 ·
게이트. `PREREG.md` 는 이 파일의 산문 번역이고 **둘이 어긋나면 코드가 권위**다.

## 용량(capacity) 의 정의 — 코드에서 도출한다, 손으로 고르지 않는다

제안 표현이 설치하는 원자 수 `K` 는 **한 함수**가 정한다:

    learned_init.pool_and_select(kx, ky, energy,
                                 energy_thresh, dedup_dk, max_n)   # learned_init.py:168

    ① `en > energy_thresh * en.max()` 로 약한 원자를 버린다
    ② 강한 순으로 `dedup_dk` 최소분리를 지키며 탐욕 선택한다
    ③ `max_n` 에서 자른다

호출 사슬(전부 실측 확인): `learned_init_params`(:342-344,:359) ← `_initialize_learned`
(`sequential_update.py:876`) ← 배포 라우터(`auto_switch_thresh=0` ⟹ **합성 100%가 학습 경로**).
⟹ ★**용량은 노브 하나가 아니라 이 세 개의 곱이다.** `max_n` 만 보는 것이 `B-2d` 의 결함이었다.

## ⛔이 배치가 재지 **않는** 용량

`bank_mmse` 의 뱅크 크기(`dk_c=kp/10`, `band=(0.35,2.5)·kp` ⟹ **B=1908 원자·3816 계수**,
척도 무관 — `premise_m4.json:P4`). 그것은 **다른 표현**이고 배포 실측 경로가 쓰는 것이다.
⟹ 이 배치는 **제안 표현의 용량**만 움직인다. 결론을 배포 실측 거동으로 옮기지 마라.
"""
import inspect
import json
import os
import statistics as st
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for _p in (ROOT,
           os.path.join(ROOT, 'tmp', 'B22_PRIOR_ABLATION_2026-08-27'),
           os.path.join(ROOT, 'tmp', 'B19_KGRID_SPARSE_2026-08-25'),
           os.path.join(ROOT, 'tmp', 'B12_HOS_EXTEND_2026-08-23'),
           os.path.join(ROOT, 'tmp', 'B11_HOS_BASELINE_2026-08-23'),
           os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23'),
           os.path.join(ROOT, 'scripts')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gates_b22 as B22            # noqa: E402  배포 상수·합성 표본·시드의 단일출처
import gates_b2 as B2              # noqa: E402  실측 12 레코드·부호검정의 단일출처

# ─────────────────────────────────────────────────────────────────────────────
# 0. 방향 불변식 — 결과 전에 박는다
# ─────────────────────────────────────────────────────────────────────────────
DIRECTION = (
    'ELIMINATION-ONLY. 이 배치는 "경계가 용량 축을 따라 놓여 있는가" 만 묻는다. '
    '어느 분기로 가도 산출은 **소거표의 한 칸**이고, ⛔"경계의 원인은 X 다" 라는 지목으로 '
    '올리지 않는다. PRIMARY 처치(K-MAX)는 배포 기본값이 **아니며**, 이 배치는 '
    '⛔배포 변경을 제안하지 않는다(별도 사전등록 사항).')

# ─────────────────────────────────────────────────────────────────────────────
# 1. 배포 상수 — 패키지 단일출처 + 트립와이어
# ─────────────────────────────────────────────────────────────────────────────
DEPLOY = B22.deploy_constants          # 함수 그대로 재사용(사본 아님)
DEPLOY_EXPECTED = dict(B22.DEPLOY_EXPECTED)   # ★핀. 출하 기본값이 바뀌면 G-01 이 멈춘다


def shipped_knobs():
    """(e_th, ddk, max_n) — `pool_and_select` **서명 기본값**. ⛔손복사 금지."""
    from wavesplat.optimizer.learned_init import pool_and_select
    s = inspect.signature(pool_and_select).parameters
    return (float(s['energy_thresh'].default), float(s['dedup_dk'].default),
            int(s['max_n'].default))


RAW_ATOMS_LEN = 128        # nsamp(4) × M(32). ⚠상수가 아니라 **자료에서 확인**한다(G-05b).

# ─────────────────────────────────────────────────────────────────────────────
# 2. 팔 — ⛔빈 dict 금지. 값은 전부 명시하거나 배포 단일출처에서 채운다.
#    'SHIP' 자리표시자는 `arm_knobs()` 가 패키지 값으로 채운다.
# ─────────────────────────────────────────────────────────────────────────────
#   max_n / e_th / ddk : 용량 노브 셋
#   si_l1              : SCALE_INVARIANT_L1 모듈 플래그(= B-2c 의 `LAM-SCALE`)
#   grow_off           : grow_residual_threshold 를 실질 무한대로 (성장 가지 계수기)
#   inline             : 원자를 주입하지 않는다(캐시 양성대조)
def _A(max_n, e_th='SHIP', ddk='SHIP', si_l1=False, grow_off=False, inline=False):
    return dict(max_n=max_n, e_th=e_th, ddk=ddk, si_l1=si_l1,
                grow_off=grow_off, inline=inline)


ARMS = {
    # ── 용량 사다리 (K 단조, 매 스텝 **한 노브**) ────────────────────────────
    'K-8':        _A(8),
    'K-15':       _A(15),
    'SHIP-K':     _A('SHIP'),                      # ★배포값 = 기준선
    'K-60':       _A(60),
    'K-128':      _A(128),
    'K-128-D15':  _A(128, ddk='HALF'),
    'K-128-D0':   _A(128, ddk=0.0),
    'K-MAX':      _A(128, e_th=0.0, ddk=0.0),      # ★PRIMARY 처치 = 제안 원자 **전부** 설치
    # ── 2×2 (용량 × 척도맞춤 벌점) — `B-2c` 는 λ 를 **내리기만** 했다 ────────
    'SHIP-K-LSC': _A('SHIP', si_l1=True),          # = B-2c `LAM-SCALE` 을 정정 수심에서 재측정
    'K-MAX-LSC':  _A(128, e_th=0.0, ddk=0.0, si_l1=True),
    # ── 대조 ────────────────────────────────────────────────────────────────
    'K-8-NOGROW': _A(8, grow_off=True),            # 성장 **가지** 계수기 (B-2f 교훈)
    'SHIP-K-INL': _A('SHIP', inline=True),         # 원자 캐시 양성대조 (비트동일 요구)
}

LADDER = ['K-8', 'K-15', 'SHIP-K', 'K-60', 'K-128', 'K-128-D15', 'K-128-D0', 'K-MAX']
BASE = 'SHIP-K'
PRIMARY_ARM = 'K-MAX'
TWOBYTWO = [('SHIP-K', 'SHIP-K-LSC'), ('K-MAX', 'K-MAX-LSC')]
REAL_ARMS = LADDER + ['SHIP-K-LSC', 'K-MAX-LSC', 'K-8-NOGROW', 'SHIP-K-INL']
SYN_ARMS = ['SHIP-K', 'K-60', 'K-128-D0', 'K-MAX']      # 합성은 4점 용량-반응만
SYN_IMPORTED_BASE = 'SHIP-K'      # ★B-22 `LEARN` 동결 행을 수입한다(재실행 0)
INLINE_SEEDS = (42,)              # 양성대조는 시드 42 한 개(예산)

# 매 스텝 정확히 한 노브만 움직이는 사슬 (G-03 이 강제)
CHAIN = [('K-8', 'K-15'), ('K-15', 'SHIP-K'), ('SHIP-K', 'K-60'),
         ('K-60', 'K-128'), ('K-128', 'K-128-D15'), ('K-128-D15', 'K-128-D0'),
         ('K-128-D0', 'K-MAX')]


def arm_knobs(arm):
    """팔의 **완성된** (max_n, e_th, ddk, si_l1, grow_off, inline). 자리표시자를 채운다."""
    e_th, ddk, max_n = shipped_knobs()
    a = dict(ARMS[arm])
    sub = {'SHIP': None, 'HALF': None}
    a['max_n'] = max_n if a['max_n'] == 'SHIP' else int(a['max_n'])
    a['e_th'] = e_th if a['e_th'] == 'SHIP' else float(a['e_th'])
    a['ddk'] = (ddk if a['ddk'] == 'SHIP'
                else (ddk / 2.0 if a['ddk'] == 'HALF' else float(a['ddk'])))
    del sub
    return a


def arm_config(arm, atoms, depth=None, seed=None, real=False):
    """★팔의 **완성된 estimator config**. 러너·자가진단이 같은 함수를 부른다.

    배포값은 `DEPLOY()` 에서 채우고(⛔리터럴 금지), 이 팔이 바꾸는 것만 위에 얹는다.
    `real=True` 면 실측 게이트를 우회하고 제안 표현을 **강제**한다(`B-2` 의 `PROP` 규약).
    """
    dep = DEPLOY()
    k = arm_knobs(arm)
    cfg = {'varpro_steps': dep['varpro_steps'],
           'learned_init_iters': dep['learned_init_iters'],
           'auto_switch_thresh': dep['auto_switch_thresh'],
           'lambda_l1': dep['lambda_l1'],
           'analytic_ensemble_k': dep['analytic_ensemble_k'],
           'learned_energy_thresh': k['e_th'],
           'learned_dedup_dk': k['ddk'],
           'max_n': k['max_n'],
           'learned_atoms': None if k['inline'] else atoms}   # ⛔None 도 **명시**
    if k['grow_off']:
        cfg['grow_residual_threshold'] = 1e9        # 성장 가지를 구조적으로 막는다
    if real:
        cfg['init_method'] = 'learned'              # ⛔배포 실측 게이트 우회(연구용 강제)
        cfg['real_domain_gate'] = False
    else:
        cfg['init_method'] = dep['init_method']     # 'auto' — 합성은 라우터가 learned 로 간다
    if depth is not None:
        cfg['depth'] = float(depth)
    if seed is not None:
        cfg['seed'] = int(seed)
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# 3. 표본
# ─────────────────────────────────────────────────────────────────────────────
RECORDS = list(B2.RECORDS)              # 실측 12 (단일출처 = gates_b2)
SMALL_BOX = list(B2.SMALL_BOX)          # 결과 전 선언된 층(박스 13–19 m)
LARGE_BOX = list(B2.LARGE_BOX)
WIN = B2.WIN                            # 3
EST_SEEDS = tuple(B22.EST_SEEDS)        # (42, 43, 44) — median-of-3 (S13b 교훈)
FIELD_REAL = B2.FIELD                   # 'nowcast'
FIELD_SYN = 'avg_nRMSE'


def syn_episodes():
    """합성 = `B-12`/`B-22` 의 혼잡 HOS 48 (⛔생성 0회, 디스크에 있다)."""
    return B22.episodes()


def real_zero_ref(record):
    """실측 채점 기준 = **참 제로필드**(레코드별). ⛔`1.0` 상수 금지(08-23 사고)."""
    from zero_ref import zero_ref_nowcast
    return float(zero_ref_nowcast(record, win=WIN))


# ─────────────────────────────────────────────────────────────────────────────
# 4. 문턱 — 산술로 도출한다
# ─────────────────────────────────────────────────────────────────────────────
ALPHA = B22.ALPHA                       # 0.05
FAMILY_ALPHA = B22.FAMILY_ALPHA         # 0.025 (B-22 승계)


def sign_p(w, n):
    return sum(comb(n, k) for k in range(w, n + 1)) / 2 ** n


def thresholds(n):
    """`w >= T` 가 단측 p<=ALPHA 를 주는 최소 T, 그리고 가족단위 T."""
    t = min(w for w in range(n + 1) if sign_p(w, n) <= ALPHA)
    tf = min(w for w in range(n + 1) if sign_p(w, n) <= FAMILY_ALPHA)
    return dict(n=n, T_alpha=t, p_at_T=sign_p(t, n),
                T_family=tf, p_at_Tf=sign_p(tf, n))


REAL_T = thresholds(12)                 # T_alpha = 10 (p=0.019287)
SYN_T = thresholds(48)                  # T_alpha = 31 (p=0.029...)

# ─────────────────────────────────────────────────────────────────────────────
# 5. ★결과 전 해석 규칙 — 어느 결과가 나와도 무엇을 쓸지 **지금** 정한다
# ─────────────────────────────────────────────────────────────────────────────
VERDICTS = ('CAPACITY-EARNS', 'CAPACITY-HURTS', 'CAPACITY-NULL')

READING = {
 'CAPACITY-EARNS':
   '실측에서 용량을 늘리면 제안 표현이 좋아진다(K-MAX < SHIP-K, >=T/12). ⟹ 소거표의 '
   '★미결 칸이 ✅"용량이 실측 경계의 한 요인"으로 바뀐다. 처방 = 자료에 맞춘 적응 차수'
   '(B-2d §3 이 예약해 둔 처방). ⛔단 배포 실측 경로는 bank_mmse 이므로 **배포 변경이 '
   '아니다**. ⛔"경계의 원인" 지목 금지 — 여전히 한 축의 양성일 뿐이다.',
 'CAPACITY-HURTS':
   '실측에서 용량을 늘리면 **나빠진다**(SHIP-K < K-MAX, >=T/12). ⟹ 소거표의 ★미결 칸이 '
   '⛔"아니다 — 방향이 반대"로 닫힌다. 정직한 문장 = *"제안 표현은 실측에서 용량이 모자라서 '
   '지는 것이 아니다"*. 2×2(척도맞춤 벌점)가 회복시키면 결핍은 **용량이 아니라 정칙화**이고, '
   '회복시키지 못하면 **용량 축 자체가 지렛대가 아니다**.',
 'CAPACITY-NULL':
   '부호로 갈리지 않는다. ⟹ 소거표의 ★미결 칸이 ⛔"아니다 — 무영향"으로 닫힌다. '
   '★그러면 소거가 (자료·관측 다수)+(표현 1건) 에서 **자료·관측 쪽으로 단일화**된다 — '
   '이 배치가 노린 산출이 바로 이것이고, **널이어도 논문의 한 절이 바뀐다**. '
   '병기 의무: 용량-반응표(사다리 8점)와 비트동일 셀 수.',
}

# ⚠어느 분기에서도 반드시 함께 적는 공시
DISCLOSURE = {
 'forced_arm':
   '⛔이 배치의 실측 팔은 배포 거동이 아니다. 배포 실측 게이트는 bank_mmse 로 보내며 '
   '(sequential_update.py:715), 제안 표현은 real_domain_gate=False 로 **강제**했다.',
 'other_representation':
   '⚠배포 실측 팔의 용량은 B=1908 원자(3816 계수, 척도 무관; premise_m4.json:P4)이고 '
   '제안 표현은 최대 128 이다. 두 팔의 용량 비는 **교락**이며 이 배치는 그것을 풀지 않는다.',
 'sparse_untouched':
   '⛔희소 합성층에 대해 아무 말도 하지 않는다(합성 표본은 혼잡 HOS 48 고정).',
 'nsamp_untouched':
   '⛔제안 **자체**의 용량(learned_nsamp=4, M=32 ⟹ 원자 128)은 고정이다. 풀을 키우는 '
   '축은 미측정이고, 이 배치의 상한은 그 128 이다.',
 'window_site':
   '⚠실측은 고정 관측소 4 사이트·12 레코드·창 1개(win=3). 사이트 수준 일반화 금지.',
 'b2d_rerun':
   '✅부수효과: `B-2d`(max_n 30→10→4)의 처치 팔은 **오기 수심 17 에서만** 돌았고 '
   '정정 수심 30 에서 재실행된 적이 없다. K-8/K-15 가 그 칸을 정정 수심에서 채운다. '
   '⛔단 값이 다르므로 B-2d 수치의 "재현"이 아니다.',
}

# ★예측을 결과 전에 적는다(사후 해석 금지용 기록이지 가설이 아니다)
PREDICTION = (
 '런0 전제(premise_m4.json)가 이미 말해 주는 것: 헤드라인 합성 코퍼스에서 배포 K 는 '
 '중앙 19(상한 30 이 무는 비율 20/144=13.9%)이고, 실측 PROP 은 12 중 5 가 상한 30 에 '
 '붙어 있다(B-2). ⟹ **상한이 무는 비율이 합성보다 실측에서 높다**. 그러나 그것은 '
 '"용량이 부족하다"의 증거가 아니라 **가설의 근거**일 뿐이다. 나는 실측에서 '
 'CAPACITY-HURTS 또는 CAPACITY-NULL 이 나올 확률이 더 높다고 본다 — 사유: (i) B-2d 가 '
 '차수를 3배 낮춰도 짝 널이었고(p=0.77), (ii) 실측 소형 박스는 질의점이 16–25 개인데 '
 '600 관측에 128 원자(256 자유도)는 과하다. ⛔이 문단은 판정에 아무 힘이 없다.')


def label_real(res):
    """PRIMARY 라벨. `res` = paired(K-MAX vs SHIP-K) 결과 dict."""
    n, w = res['n'], res['better']          # better = #(K-MAX < SHIP-K)
    T = thresholds(n)['T_alpha']
    if w >= T:
        return ('CAPACITY-EARNS',
                f'K-MAX 가 배포 용량을 {w}/{n} 에서 이긴다 (p={sign_p(w, n):.4g}).')
    if (n - w) >= T:
        return ('CAPACITY-HURTS',
                f'배포 용량이 K-MAX 를 {n - w}/{n} 에서 이긴다 (p={sign_p(n - w, n):.4g}).')
    return ('CAPACITY-NULL', f'부호로 갈리지 않는다 ({w}/{n}).')


def label_syn(res):
    n, w = res['n'], res['better']
    T = thresholds(n)['T_alpha']
    if w >= T:
        return ('SYN-CAPACITY-EARNS', f'{w}/{n}')
    if (n - w) >= T:
        return ('SYN-CAPACITY-HURTS', f'{n - w}/{n}')
    return ('SYN-CAPACITY-NULL', f'{w}/{n}')


# ─────────────────────────────────────────────────────────────────────────────
# 6. 집계
# ─────────────────────────────────────────────────────────────────────────────
def median_of_seeds(rows, arm, key, field):
    """(단위 -> median-of-EST_SEEDS 값). `key` = 'record' 또는 'episode'."""
    acc = {}
    for r in rows:
        if r.get('arm') != arm or r.get('pc') or r.get(field) is None:
            continue
        acc.setdefault(r[key], []).append(float(r[field]))
    return {k: st.median(v) for k, v in acc.items()}


def paired(rows, arm, ref, key, field, subset=None):
    a = median_of_seeds(rows, arm, key, field)
    b = median_of_seeds(rows, ref, key, field)
    ks = sorted(set(a) & set(b))
    if subset:
        ks = [k for k in ks if k in subset]
    n = len(ks)
    w = sum(1 for k in ks if a[k] < b[k])
    tie = sum(1 for k in ks if a[k] == b[k])
    rel = [(a[k] - b[k]) / b[k] * 100 for k in ks if b[k]]
    return dict(arm=arm, ref=ref, n=n, better=w, tie=tie,
                rel_median_pct=(st.median(rel) if rel else None),
                p_one_sided=(sign_p(w, n) if n else None),
                p_other_tail=(sign_p(n - w, n) if n else None))


def skill(rows, arm, key, field, zero):
    """무기술 기준(참 제로필드) 대비 — 부차. ⛔PRIMARY 아님."""
    a = median_of_seeds(rows, arm, key, field)
    ks = sorted(set(a) & set(zero))
    n = len(ks)
    w = sum(1 for k in ks if a[k] < zero[k])
    return dict(arm=arm, n=n, beats_zero=w,
                p_one_sided=(sign_p(w, n) if n else None),
                median=(st.median([a[k] for k in ks]) if ks else None))


def holm(pairs, alpha=None):
    """Holm 단계하강. `pairs` = [(name, p), ...]."""
    alpha = ALPHA if alpha is None else alpha
    s = sorted(pairs, key=lambda x: x[1])
    m = len(s)
    out, ok = [], True
    for i, (nm, p) in enumerate(s):
        thr = alpha / (m - i)
        ok = ok and (p <= thr)
        out.append(dict(name=nm, p=p, thr=thr, reject=bool(ok)))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 7. 게이트 — 런 0회 (G-0x) / 런 후 (G-1x)
# ─────────────────────────────────────────────────────────────────────────────
B22_DIR = os.path.join(ROOT, 'tmp/B22_PRIOR_ABLATION_2026-08-27')
B2_DIR = os.path.join(ROOT, 'tmp/B2_REAL_2026-08-23')
ATOMS_REAL = os.path.join(HERE, 'atoms_real')
PREMISE = os.path.join(HERE, 'premise_m4.json')


def _b22_learned_atom_files():
    import glob
    return sorted(glob.glob(os.path.join(B22_DIR, 'atoms', '*__learned__*.json')))


def _b22_learn_rows():
    import glob
    out = []
    for f in sorted(glob.glob(os.path.join(B22_DIR, 'results_b22*.jsonl'))):
        for line in open(f):
            r = json.loads(line)
            if 'error' not in r and r.get('arm') == 'LEARN':
                out.append(r)
    return out


def pooled_K(atoms, arm):
    """★러너/게이트가 **패키지 함수로** 도출하는 기대 설치 수. ⛔재구현 금지."""
    import numpy as np
    from wavesplat.optimizer.learned_init import pool_and_select
    k = arm_knobs(arm)
    sk, _, _ = pool_and_select(np.asarray(atoms['kx'], float),
                               np.asarray(atoms['ky'], float),
                               np.asarray(atoms['energy'], float),
                               energy_thresh=k['e_th'], dedup_dk=k['ddk'],
                               max_n=k['max_n'])
    return int(len(sk))


def gates_runfree():
    out = []
    # G-01 배포 상수 트립와이어
    dep = DEPLOY()
    bad = {k: (dep.get(k), v) for k, v in DEPLOY_EXPECTED.items() if dep.get(k) != v}
    out.append(('G-01 배포 상수 핀', not bad, str(bad or dep)[:120]))
    # G-02 기준선 팔 == 배포값 (손복사 0)
    e_th, ddk, max_n = shipped_knobs()
    b = arm_knobs(BASE)
    out.append(('G-02 SHIP-K == 배포 선택 노브',
                (b['e_th'], b['ddk'], b['max_n']) == (e_th, ddk, max_n),
                f"{(b['e_th'], b['ddk'], b['max_n'])} vs {(e_th, ddk, max_n)}"))
    # G-03 사슬이 매 스텝 한 노브만 움직인다
    diffs = []
    for a, bnm in CHAIN:
        ka, kb = arm_knobs(a), arm_knobs(bnm)
        d = [k for k in ('max_n', 'e_th', 'ddk', 'si_l1', 'grow_off', 'inline')
             if ka[k] != kb[k]]
        diffs.append((a, bnm, d))
    out.append(('G-03 사슬 단일요인', all(len(d) == 1 for _, _, d in diffs),
                str([(a, b, d) for a, b, d in diffs if len(d) != 1])[:120]))
    # G-03b 2x2 의 세로 변은 si_l1 하나만 움직인다
    v = []
    for a, bnm in TWOBYTWO:
        ka, kb = arm_knobs(a), arm_knobs(bnm)
        v.append([k for k in ka if ka[k] != kb[k]])
    out.append(('G-03b 2x2 세로변 단일요인', all(d == ['si_l1'] for d in v), str(v)))
    # G-04 표본
    out.append(('G-04 실측 12 레코드 (gates_b2 단일출처)', len(RECORDS) == 12,
                f'{len(RECORDS)}'))
    out.append(('G-04b 합성 48 에피소드 (gates_b22 단일출처)',
                len(syn_episodes()) == 48, f'{len(syn_episodes())}'))
    # ★G-05 외부 동결 자료 참조가 **비어 있지 않은가** (M2 G-13 교훈)
    try:
        zr = json.load(open(os.path.join(B2_DIR, 'true_zero_field.json')))
    except Exception as e:                                   # noqa: BLE001
        zr = {}
        out.append(('G-05z 참 제로필드 파일', False, repr(e)[:80]))
    out.append(('G-05a 참 제로필드 키 == RECORDS',
                set(zr) == set(RECORDS) and len(zr) == 12,
                f'{len(zr)} keys, 차집합 {sorted(set(RECORDS) - set(zr))}'))
    fs = _b22_learned_atom_files()
    out.append(('G-05b B-22 학습 원자 파일 144개',
                len(fs) == len(syn_episodes()) * len(EST_SEEDS), f'{len(fs)}'))
    if fs:
        lens = {len(json.load(open(f))['atoms']['kx']) for f in fs}
        out.append(('G-05c 원자 배열 길이 == 128 (nsamp4 × M32)',
                    lens == {RAW_ATOMS_LEN}, str(sorted(lens))))
    rws = _b22_learn_rows()
    out.append(('G-05d B-22 LEARN 동결 행 144개 (합성 기준선 수입원)',
                len(rws) == len(syn_episodes()) * len(EST_SEEDS), f'{len(rws)}'))
    # ★G-06 처치가 **구조적 no-op 이 아닌가** — 자료에서 확인(B-2f 교훈)
    if fs:
        gt = [(pooled_K(json.load(open(f))['atoms'], PRIMARY_ARM),
               pooled_K(json.load(open(f))['atoms'], BASE)) for f in fs]
        strict = sum(1 for a, b in gt if a > b)
        out.append((f'G-06 합성: {PRIMARY_ARM} K > {BASE} K 가 전건',
                    strict == len(gt), f'{strict}/{len(gt)}'))
        out.append(('G-06b 합성: K-MAX 는 항상 원자 전부(128)',
                    all(a == RAW_ATOMS_LEN for a, _ in gt),
                    str(sorted({a for a, _ in gt}))))
        cap = sum(1 for _, b in gt if b >= shipped_knobs()[2])
        out.append(('G-06c 합성 상한 포화율(공시용, 판정 아님)', True,
                    f'{cap}/{len(gt)}'))
    # G-07 문턱 산술
    out.append(('G-07 문턱 산술 (n=12)', REAL_T['T_alpha'] == 10 and
                abs(REAL_T['p_at_T'] - 0.019287109375) < 1e-12, str(REAL_T)))
    out.append(('G-07b 문턱 산술 (n=48)', SYN_T['T_alpha'] >= 1, str(SYN_T)))
    # G-08 전제 파일이 존재하고 이 게이트와 같은 노브를 봤는가
    if os.path.exists(PREMISE):
        pm = json.load(open(PREMISE))
        out.append(('G-08 premise 의 배포 노브 == 패키지',
                    (pm['shipped_selection_knobs']['energy_thresh'],
                     pm['shipped_selection_knobs']['dedup_dk'],
                     pm['shipped_selection_knobs']['max_n']) == shipped_knobs(),
                    str(pm['shipped_selection_knobs'])))
    else:
        out.append(('G-08 premise_m4.json 존재', False, '없음'))
    # G-09 실측 수심 — ⛔B-2 사고 재발 방지
    try:
        from scene_station import RECORDS_12
        bs = {k: v['depth'] for k, v in RECORDS_12.items() if k.startswith('BS')}
        out.append(('G-09 BS 7 레코드 depth == 30.0 (08-23 정정)',
                    set(bs.values()) == {30.0} and len(bs) == 7, str(bs)))
    except Exception as e:                                   # noqa: BLE001
        out.append(('G-09 scene_station 수심', False, repr(e)[:80]))
    return out


ROWLESS_OK = ('G-01', 'G-02', 'G-03', 'G-03b', 'G-04', 'G-04b', 'G-05a', 'G-05b',
              'G-05c', 'G-05d', 'G-06', 'G-06b', 'G-06c', 'G-07', 'G-07b', 'G-08',
              'G-09')


def gates_run(rows):
    """런 후 게이트. `rows` = 이 배치의 전 결과 행."""
    ok = [r for r in rows if 'error' not in r]
    out = [('G-11 에러 0행', len(rows) == len(ok), f'{len(rows) - len(ok)}')]
    out.append(('G-12 0행 가드', len(ok) > 0, f'{len(ok)}'))
    real = [r for r in ok if r.get('stratum') == 'real' and not r.get('pc')]
    syn = [r for r in ok if r.get('stratum') == 'syn' and not r.get('pc')]
    # G-13 팔별 행 수
    for a in REAL_ARMS:
        want = len(RECORDS) * (len(INLINE_SEEDS) if arm_knobs(a)['inline']
                               else len(EST_SEEDS))
        n = sum(1 for r in real if r['arm'] == a)
        out.append((f'G-13 real {a}', n == want, f'{n}/{want}'))
    for a in SYN_ARMS:
        want = len(syn_episodes()) * len(EST_SEEDS)
        n = sum(1 for r in syn if r['arm'] == a)
        out.append((f'G-13s syn {a}', n == want, f'{n}/{want}'))
    # ★G-14 개입 되읽기 — **세 출처의 항등식** (⛔한 출처의 항등식이 아니다)
    #   `n_gaussians` = 추정기 **객체**의 최종 성분 수
    #   `K_expected`  = 러너가 **주입 배열**에 패키지 `pool_and_select` 를 돌려 낸 설치 기대값
    #   `apg_dn`      = **가지 계수기**가 기록한 prune/grow 의 성분 수 변화 (B-2f 교훈)
    #   ⟹ n_gaussians == K_expected + Σ apg_dn 이어야 한다. 배열이 안 닿았으면 어긋난다.
    bad = [(r['arm'], r.get('unit'), r.get('n_gaussians'), r.get('K_expected'),
            r.get('apg_dn'))
           for r in ok if r.get('K_expected') is not None
           and r.get('n_gaussians') is not None and r.get('apg_dn') is not None
           and r['n_gaussians'] != r['K_expected'] + sum(r['apg_dn'])]
    out.append(('G-14 n_gaussians == K_expected + Σapg_dn (세 출처 항등)',
                not bad, str(bad[:3])[:140]))
    # G-14a 계수기가 없는 행(수입·합성)에는 구조 불변식만 요구한다
    bad2 = [(r['arm'], r.get('unit'), r.get('n_gaussians'))
            for r in ok if r.get('apg_dn') is None and r.get('n_gaussians') is not None
            and not (0 < r['n_gaussians'] <= arm_knobs(r['arm'])['max_n'])]
    out.append(('G-14a 0 < n_gaussians <= max_n (계수기 없는 행)', not bad2,
                str(bad2[:3])[:120]))
    # G-14b 처치가 실제로 K 를 늘렸는가 (셀 단위, 비트 no-op 를 드러낸다)
    kb = {(r.get('unit'), r.get('est_seed')): r.get('K_expected')
          for r in ok if r['arm'] == BASE and not r.get('pc')}
    grew = [(r.get('unit'), r.get('est_seed')) for r in ok
            if r['arm'] == PRIMARY_ARM and not r.get('pc')
            and kb.get((r.get('unit'), r.get('est_seed'))) is not None
            and r.get('K_expected', 0) > kb[(r.get('unit'), r.get('est_seed'))]]
    nprim = sum(1 for r in ok if r['arm'] == PRIMARY_ARM and not r.get('pc'))
    out.append((f'G-14b {PRIMARY_ARM} 이 전 셀에서 K 를 늘렸다',
                nprim > 0 and len(grew) == nprim, f'{len(grew)}/{nprim}'))
    # ★G-15 캐시 양성대조 — 비트동일
    inl = {(r['unit'], r['est_seed']): r for r in ok if r['arm'] == 'SHIP-K-INL'}
    base = {(r['unit'], r['est_seed']): r for r in ok if r['arm'] == BASE}
    same = [k for k in inl if k in base
            and repr(inl[k].get(FIELD_REAL)) == repr(base[k].get(FIELD_REAL))]
    out.append(('G-15 원자 캐시 = 인라인 제안 (비트동일)',
                len(inl) > 0 and len(same) == len(inl), f'{len(same)}/{len(inl)}'))
    # ★G-16 성장 **가지** 계수기 (B-2f 교훈: 함수가 아니라 가지를 센다)
    ng = {(r['unit'], r['est_seed']): r for r in ok if r['arm'] == 'K-8-NOGROW'}
    k8 = {(r['unit'], r['est_seed']): r for r in ok if r['arm'] == 'K-8'}
    ident = [k for k in ng if k in k8
             and repr(ng[k].get(FIELD_REAL)) == repr(k8[k].get(FIELD_REAL))]
    out.append(('G-16 성장 가지 불활성? (비트동일 셀 수 — 공시용, 판정 아님)', True,
                f'{len(ident)}/{len(ng)}'))
    # G-17 확산 호출 0 — 캐시만 쓴다(INLINE 제외)
    nz = [(r['arm'], r.get('unit')) for r in ok
          if not arm_knobs(r['arm'])['inline'] and r.get('propose_calls')]
    out.append(('G-17 캐시 팔에서 확산 호출 0', not nz, str(nz[:4])[:120]))
    # G-18 GPU 미사용
    gpu = [r.get('unit') for r in ok if r.get('cuda_available') is not False]
    out.append(('G-18 GPU 미사용 (cuda_available False 전건)', not gpu,
                f'{len(gpu)} 행'))
    # G-19 같은 (단위, 시드) 는 팔 전체가 **같은 원자 풀**을 나눠 쓴다
    grp = {}
    for r in ok:
        if arm_knobs(r['arm'])['inline'] or r.get('atoms_md5') is None:
            continue
        grp.setdefault((r.get('unit'), r.get('est_seed')), set()).add(r['atoms_md5'])
    multi = [k for k, v in grp.items() if len(v) != 1]
    out.append(('G-19 셀당 원자 풀 1개', not multi, str(multi[:4])[:120]))
    # G-20 실측: 팔 간 관측·질의 해시 일치 (B-2 G-3 승계)
    h = {}
    for r in real:
        h.setdefault(r['unit'], set()).add((r.get('obs_hash'), r.get('q_hash')))
    badh = [k for k, v in h.items() if len(v) != 1]
    out.append(('G-20 실측 팔 간 관측·질의 해시 일치', not badh, str(badh)[:100]))
    # G-21 실측 게이트 우회 되읽기
    br = [r['unit'] for r in real
          if r.get('real_gate') is not False and r.get('real_gate') is not None]
    out.append(('G-21 실측 팔이 bank 경로로 새지 않았다', not br, str(br[:4])))
    out.append(('G-21b 실측 팔 전건 n_gaussians>0 (bank 는 0개다)',
                all((r.get('n_gaussians') or 0) > 0 for r in real),
                str(sorted({r.get('n_gaussians') for r in real}))[:100]))
    # G-22 수심 되읽기
    dep_bad = [(r['unit'], r.get('depth')) for r in real
               if r['unit'] in SMALL_BOX and r.get('depth') != 30.0]
    out.append(('G-22 소형 박스 수심 30.0 되읽기', not dep_bad, str(dep_bad[:4])))
    # G-23 si_l1 되읽기 (모듈 플래그가 실제로 켜졌나)
    sib = [(r['arm'], r.get('si_l1_readback')) for r in ok
           if r.get('si_l1_readback') is not None
           and bool(r['si_l1_readback']) != bool(arm_knobs(r['arm'])['si_l1'])]
    out.append(('G-23 SCALE_INVARIANT_L1 되읽기 == 팔 정의', not sib, str(sib[:4])))
    return out


def _gid(name):
    return name.split()[0]


# ─────────────────────────────────────────────────────────────────────────────
# 8. 자가진단 — 런 0회·결과 0개
# ─────────────────────────────────────────────────────────────────────────────
def selftest():
    fails = []

    def chk(cond, msg):
        if not cond:
            fails.append(msg)

    # [1] 런0 게이트가 실제로 통과하는가 (⛔"구조적으로 통과 불가" 방지 = M2 G-13 교훈)
    rf = gates_runfree()
    for gid, okv, msg in rf:
        chk(okv, f'runfree {gid}: {msg}')
    # [2] 라벨 기계가 세 분기를 **전부** 낼 수 있는가 (죽은 분기 금지)
    got = {label_real(dict(n=12, better=w))[0] for w in (0, 6, 12)}
    chk(got == set(VERDICTS), f'라벨 분기 {got}')
    # [3] 문턱이 산술과 일치
    chk(sign_p(10, 12) < ALPHA <= sign_p(9, 12), '문턱 10/12')
    # [4] 집계기 자가시험 — 알려진 답
    rows = [dict(arm='A', unit='u1', est_seed=s, stratum='real', nowcast=0.5)
            for s in EST_SEEDS] + \
           [dict(arm='B', unit='u1', est_seed=s, stratum='real', nowcast=0.6)
            for s in EST_SEEDS]
    pr = paired(rows, 'A', 'B', 'unit', 'nowcast')
    chk(pr['better'] == 1 and abs(pr['rel_median_pct'] + 16.6666666667) < 1e-6,
        f'paired {pr}')
    # [5] median-of-3 가 실제로 중앙값인가
    rows = [dict(arm='A', unit='u', est_seed=s, stratum='real', nowcast=v)
            for s, v in zip(EST_SEEDS, (0.1, 0.9, 0.5))]
    chk(median_of_seeds(rows, 'A', 'unit', 'nowcast')['u'] == 0.5, 'median-of-3')
    # [6] ★변형 하네스 — 결과를 흔들면 라벨이 **바뀌어야** 한다(시험이 공허하지 않음)
    chk(label_real(dict(n=12, better=11))[0] == 'CAPACITY-EARNS', '변형 11/12')
    chk(label_real(dict(n=12, better=1))[0] == 'CAPACITY-HURTS', '변형 1/12')
    chk(label_real(dict(n=12, better=7))[0] == 'CAPACITY-NULL', '변형 7/12')
    # [7] ★0행 가드가 실제로 FAIL 을 내는가
    g0 = dict(gates_run([]) and {})
    grun = gates_run([])
    chk(any(_gid(n) == 'G-12' and not okv for n, okv, _ in grun), '0행 가드 미작동')
    del g0
    # [8] ★처치가 no-op 이 아님을 **자료로** 확인 (G-06 이 그것이고 [1] 이 이미 돌렸다)
    chk(any(_gid(n) == 'G-06' for n, _, _ in rf), 'G-06 부재')
    # [9] Holm 이 **단계하강에서 멈추는가** (한 번 실패하면 이후 전부 기각 안 함)
    h = holm([('a', 0.001), ('b', 0.03), ('c', 0.04)])   # thr = .0167 / .025 / .05
    chk([x['reject'] for x in h] == [True, False, False], f'holm stop {h}')
    h2 = holm([('a', 0.001), ('b', 0.02), ('c', 0.4)])   # b 는 .02 <= .025 라 기각된다
    chk([x['reject'] for x in h2] == [True, True, False], f'holm pass {h2}')
    # [10] 실측 채점 기준이 상수 1.0 이 **아님**을 확인
    from zero_ref import selftest as zr_selftest
    z = zr_selftest()
    chk(z['n'] == 12 and z['n_far_from_one'] >= 1, f'zero_ref {z}')
    return fails


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        f = selftest()
        print('=== 런0 게이트 ===')
        for gid, okv, msg in gates_runfree():
            print(f'  [{"PASS" if okv else "FAIL"}] {gid}: {msg[:110]}')
        print('=== 자가진단 ===')
        print('  PASS' if not f else '\n'.join('  FAIL ' + x for x in f))
        sys.exit(1 if f else 0)
    print(json.dumps(dict(arms=sorted(ARMS), ladder=LADDER, base=BASE,
                          primary=PRIMARY_ARM, real_T=REAL_T, syn_T=SYN_T,
                          verdicts=list(VERDICTS)), indent=1, ensure_ascii=False))
