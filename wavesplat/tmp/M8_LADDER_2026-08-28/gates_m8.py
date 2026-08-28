"""M-8 단일출처 — ★**bank → 제안 표현 사다리**: 실측에서 기술은 어디서 죽는가.

⛔**결과 0행에서 작성했다.** 이 파일이 **상수·팔·작업목록·행 스탬프·문턱·게이트·라벨·
결과 전 해석 규칙**의 **유일 정의**다. `PREREG.md` 는 산문 번역이고 어긋나면 **코드가 이긴다**.

    python3 tmp/M8_LADDER_2026-08-28/gates_m8.py              # 자가진단 (변형 하네스 포함)
    python3 tmp/M8_LADDER_2026-08-28/gates_m8.py --refcheck   # 외부 동결 참조를 실제 파일에 대고

────────────────────────────────────────────────────────────────────────────
★**이 배치가 결과 전에 고정한 것** (`PREREG.md` §1–§13 의 기계화)

  · 사다리 9계단 6고리 + 부차 3팔 = **12 팔** × 12 레코드 × 3 시드 = **432 런**
  · PRIMARY = **L3** (`MMSE-PROP` vs `BANK-B30`), `nowcast`, n=12(median-of-3),
    단측 부호검정, 문턱 **10/12** — 라벨은 **양방향 대칭**으로 적혀 있다(§8.1)
  · 사슬 = 6 고리 Holm(α=0.05) → 라벨 네 갈래(`SKILL-DIES-AT-*` / `-DISTRIBUTED` /
    `GAP-NOT-FACTORIZABLE` / `LADDER-VOID`)
  · `DIRECTION = LOCATION-OR-DECOMPOSITION` — ⛔*"원인은 X 다"* 로 올리지 않는다

★**이 배치가 결과 전에 짊어진 세 위험, 그리고 그 기계화**

 1. ⛔⛔**디바이스 누출** (`PREREG §5.1` BLOCKER-A) — `learned_init.propose_atoms:130-132`
    가 `config['device']` 를 **받지도 않고** 스스로 `cuda if available` 을 고른다.
    ⟹ 러너가 `CUDA_VISIBLE_DEVICES=''` 를 **torch 임포트 전에** 세우고 행마다
    `cuda_available` 을 싣는다. **`G-DEV` 가 전 행 `False` 를 요구한다.**
    ⛔`B-2 PROP` 열은 앵커로 **쓰지 않는다**(CPU 에서 12/12 재현 불가).
 2. ★**전역 torch RNG 결합** (`PREREG §6`) — 성장 위상이 `sequential_update.py:1617`
    `torch.rand` 에서 나오고 실측에서 **실제로 발화한다**. ⟹ 계단마다 제안을 **인라인**으로
    돌리고(원자 캐시 금지), 행에 **소모 draw**·`n_sel`·`apg_dn` 을 싣고, **시드 3개**를 쓴다.
    ⛔그 계측값은 **판정에 쓰지 않는다**(공시·게이트 전용).
 3. ★**앙상블 교락** — 지금까지 모든 *"bank > 제안"* 비교가 **median-of-3 vs 단일 런**이었다.
    ⟹ 사다리 **0단**(`L1`)에서 `real_bank_ensemble_k=1` 로 그 칸을 닫는다.

★**게이트 규율** (오늘 배운 것을 그대로 기계화한다)
  · **커버리지 메타게이트 `G-COV`** — 모든 게이트가 실제로 본 행/짝 수를 세고 0 이면 FAIL
    (`M5b G-9`). ⚠자가진단은 **러너의 작업목록·스탬프에서 만든 합성 행**으로 시험한다
    (`M4 G-13` 의 **필터 항진**은 게이트를 모형하면 안 잡힌다).
  · **팔-존재 가드** — 팔을 세는 게이트는 **분모를 조건에 넣는다**(`M4 G-15` 규약).
    ⟹ 팔 하나를 지우면 **반드시 누군가 FAIL** 한다(자가진단 ④가 12팔 전건을 시험).
  · **개입 되읽기는 항등식이 아니다** — 설정 dict 이 아니라 **객체**(`BankMMSEFit.n_candidates`,
    `BankMMSEFit.kxb`, `model.n_gaussians`, VARPRO `info['steps_executed']`, 성장 **가지** 계수기)
    에서 되읽는다. ⛔`effective_config` 를 입력 dict 복사로 만들지 않는다(`S13b` 교훈).
  · ⛔**빈 dict `{}` 로 팔을 정의하지 않는다**(`gates_b7.py:46` BLOCKER, 2026-08-23).
  · **문턱은 `comb` 로 산술 도출**한다. ⛔손복사 금지(전제 `P4` 와는 **대조**만 한다).

⛔**이 배치가 주장하지 않는 것**은 `READING` 상수에 축자로 들어 있다(§11).
"""
import glob as _glob
import hashlib
import inspect
import json
import os
import statistics as st
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for _p in (HERE, ROOT,
           os.path.join(ROOT, 'scripts'),
           os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23'),
           os.path.join(ROOT, 'tmp', 'B3_REFIT_2026-08-23'),
           os.path.join(ROOT, 'tmp', 'B22_PRIOR_ABLATION_2026-08-27')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── 표본·창·시드·잡음·검정: 기존 배치의 단일출처를 **임포트**한다 (손복사 0) ────
from gates_b2 import (RECORDS, WIN, SEED as ANCHOR_SEED,          # noqa: E402
                      LARGE_BOX, SMALL_BOX, sign_test_one_sided)
from gates_b3 import NOISE                                        # noqa: E402
import gates_b22 as B22                                           # noqa: E402

BATCH = 'M8_LADDER_2026-08-28'
DIRECTION = 'LOCATION-OR-DECOMPOSITION'
FIELD = 'nowcast'                 # lead 0 = PRIMARY. `leadmean` 은 사전등록된 부차.
FIELD_SECOND = 'leadmean'
EST_SEEDS = tuple(B22.EST_SEEDS)  # (42, 43, 44) — median-of-3 (S13b·B-22·M4 선례)
PREMISE_JSON = os.path.join(HERE, 'premise_m8.json')

# ── 외부 동결 참조 (⚠`--refcheck` 가 **실제 파일에 대고** 비지 않음을 확인한다) ──
FROZEN_B2 = os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23', 'results.jsonl')
FROZEN_M4_GLOB = os.path.join(ROOT, 'tmp', 'M4_CAPACITY_2026-08-28',
                              'results_m4_*.jsonl')
M4_ANCHOR_ARM = 'SHIP-K-INL'      # ★CPU 고정 인라인 제안 팔 (⛔`B-2 PROP` 이 아니다)
REGRESS_JSON = os.path.join(HERE, 'regress_bank_mmse.json')

# ── 배포 상수: 패키지 단일출처 + **핀**(트립와이어) ──────────────────────────
DEPLOY = B22.deploy_constants                 # 함수 그대로 재사용 (사본 아님)
DEPLOY_EXPECTED = dict(B22.DEPLOY_EXPECTED)   # 출하 기본값이 바뀌면 여기서 멈춘다
# `deploy_constants()` 가 덮지 않는 나머지 배포 키의 **핀**. `assert_shipping_defaults()`
# 가 `wavesplat.run.DEFAULT_CONFIG` 와 대조한다. ⛔손복사가 아니라 **트립와이어**다.
DEPLOY_EXTRA_EXPECTED = {
    'adaptive_n': True, 'prune_threshold': 0.10, 'grow_residual_threshold': 0.05,
    'real_estimator': 'bank_mmse', 'real_bank_ensemble_k': 3, 'k_snap_dk': None,
    'use_confidence': True, 'tracking_mode': 'w_phi', 'device': 'cpu',
    'bank_refit_every_s': None, 'bank_refit_async': False,
}
# ★이 하네스는 `initialize` + `predict_field` 만 부른다(`update()` **0회**) ⟹
#   `bank_refit_*`·`tracking_*` 는 **구조적으로 무관**하다(`PREREG §1.2` A10).
#   그래도 핀에 넣는 이유 = *"측정한 구성과 배포한 구성이 다르면 그 측정은 배포를
#   지탱하지 못한다"*(2026-08-23) 를 **행에 남기기 위해서**다.

# ── ★이 배치가 요구하는 패키지 패치 (`PATCH.md` 정본, 가산 2키·기본값 보존형) ──
BANK_PATCH_KEYS = ('dk_frac', 'k_set')        # `bank_mmse.fit_bank_mmse` 의 새 인자
BANK_CFG_KEYS = ('bank_dk_frac', 'bank_k_set')  # `_bank_fit_window` 가 읽는 config 키

REGRESS_OUT = os.path.join(HERE, 'regress_bank_mmse.out')
REGRESS_SRC = os.path.join(HERE, 'regress_bank_mmse.py')
# ★`G-PATCH-REG` 가 받는 증거는 **둘 중 하나**다 (⛔손으로 붙인 라벨은 아니다):
#   (a) `regress_bank_mmse.json` — 최상위 `all_pass`(또는 `ok`/`pass`) 가 참이고
#       `layers`(없으면 최상위)에 `L1`–`L4` 네 겹이 전부 참.
#   (b) `regress_bank_mmse.out` — 회귀 **자신이 찍은** verdict 줄
#       `regress_bank_mmse: PASS (0 실패)` + 네 겹 태그가 전부 등장.
#   ★어느 쪽이든 **신선도**를 요구한다: 산출물이 `regress_bank_mmse.py`·`bank_mmse.py`·
#     `sequential_update.py` 보다 **나중**이어야 한다. ⛔낡은 통과 기록을 통과로 세지 않는다
#     (`check_batch_artifacts` 의 mtime 규약과 같은 부류).
PATCH_CONTRACT = ('regress_bank_mmse.json {"all_pass":true,"layers":{L1..L4}} 또는 '
                  'regress_bank_mmse.out 에 "regress_bank_mmse: PASS (0 실패)" '
                  '+ L1–L4 태그, 그리고 소스보다 새로울 것')
REGRESS_VERDICT_RE = 'regress_bank_mmse: PASS'
REGRESS_LAYERS = ('L1', 'L2', 'L3', 'L4')

# ─────────────────────────────────────────────────────────────────────────────
# 1. 사다리 — 9 계단 · 6 고리 · **한 번에 한 인자** (`PREREG §2`)
# ─────────────────────────────────────────────────────────────────────────────
# 자리표시자(런타임 주입, `arm_config()` 가 인자로 채운다. ⛔조용한 통과 금지 =
# 자가진단 ⑦ 이 최종 config 에 자리표시자가 남지 않음을 확인한다):
ATOMS = '<PROPOSAL_ATOMS>'          # 제안이 고른 원자 (kx, ky)
ATOMS_BAND = '<PROPOSAL_ATOMS_IN_BAND>'   # 그 원자를 bank 의 환대 (0.35,2.5)kp 로 필터
KP10 = '<KP_BANK_OVER_10>'          # bank 자신의 2단 beamform zoom kp / 10

_BANK = dict(init_method='auto', real_domain_gate=True, real_estimator='bank_mmse')
_LEARN = dict(init_method='learned', real_domain_gate=False)

ARMS = {
    # ── 사다리 9 계단 (`RUNGS`) ─────────────────────────────────────────────
    'BANK':        dict(_BANK, real_bank_ensemble_k=3, bank_dk_frac=None,
                        bank_k_set=None, k_snap_dk=None),
    'BANK-K1':     dict(_BANK, real_bank_ensemble_k=1, bank_dk_frac=None,
                        bank_k_set=None, k_snap_dk=None),
    'BANK-B472':   dict(_BANK, real_bank_ensemble_k=1, bank_dk_frac=0.2,
                        bank_k_set=None, k_snap_dk=None),
    'BANK-B118':   dict(_BANK, real_bank_ensemble_k=1, bank_dk_frac=0.4,
                        bank_k_set=None, k_snap_dk=None),
    'BANK-B30':    dict(_BANK, real_bank_ensemble_k=1, bank_dk_frac=0.8,
                        bank_k_set=None, k_snap_dk=None),
    'MMSE-PROP':   dict(_BANK, real_bank_ensemble_k=1, bank_dk_frac=None,
                        bank_k_set=ATOMS, k_snap_dk=None),
    'PROP-LS':     dict(_LEARN, varpro_steps=0, adaptive_n=False, k_snap_dk=None),
    'PROP-LS-APG': dict(_LEARN, varpro_steps=0, adaptive_n=True, k_snap_dk=None),
    'PROP':        dict(_LEARN, varpro_steps=400, adaptive_n=True, k_snap_dk=None),
    # ── 부차 팔 3 (Holm 가족 **밖**. `PREREG §2.2`) ─────────────────────────
    'MMSE-PROP-BAND': dict(_BANK, real_bank_ensemble_k=1, bank_dk_frac=None,
                           bank_k_set=ATOMS_BAND, k_snap_dk=None),
    'PROP-SNAP':      dict(_LEARN, varpro_steps=400, adaptive_n=True, k_snap_dk=KP10),
    'PROP-LS-SNAP':   dict(_LEARN, varpro_steps=0, adaptive_n=False, k_snap_dk=KP10),
}

RUNGS = ['BANK', 'BANK-K1', 'BANK-B472', 'BANK-B118', 'BANK-B30',
         'MMSE-PROP', 'PROP-LS', 'PROP-LS-APG', 'PROP']
SECONDARY = ['MMSE-PROP-BAND', 'PROP-SNAP', 'PROP-LS-SNAP']
ARM_ORDER = RUNGS + SECONDARY
BANK_ARMS = [a for a in ARM_ORDER if ARMS[a]['init_method'] == 'auto']
LEARN_ARMS = [a for a in ARM_ORDER if ARMS[a]['init_method'] == 'learned']
KSET_ARMS = [a for a in ARM_ORDER if ARMS[a].get('bank_k_set') in (ATOMS, ATOMS_BAND)]
GRID_ARMS = [a for a in BANK_ARMS if a not in KSET_ARMS]
SNAP_ARMS = [a for a in ARM_ORDER if ARMS[a].get('k_snap_dk') == KP10]
# ★`atoms_md5` 가 (record, seed) 안에서 **전부 일치**해야 하는 팔 (`G-ATOM`)
ATOM_ARMS = KSET_ARMS + LEARN_ARMS
DK_FRAC_ARMS = {a: ARMS[a]['bank_dk_frac'] for a in BANK_ARMS
                if ARMS[a].get('bank_dk_frac') is not None}   # B472/B118/B30
ENDPOINTS = ('BANK', 'PROP')     # `G-END` — 이 배치가 **직접** 내는 끝점 대비

# 고리 = 인자. `(고리, bank 쪽 계단, 제안 쪽 계단, 무엇이 바뀌는가)`
LINKS = [
    ('L1', 'BANK', 'BANK-K1', '앙상블 (median-of-3 → 단일)'),
    ('L2', 'BANK-K1', 'BANK-B30', '후보 기수 1904–1918 → 30'),
    ('L3', 'BANK-B30', 'MMSE-PROP', '★파수 배치 — 규칙 격자 vs 학습된 연속'),
    ('L4', 'MMSE-PROP', 'PROP-LS', '결합기 — MMSE(사전γ+val σ+EM) vs LS ridge 0.5'),
    ('L5', 'PROP-LS', 'PROP-LS-APG', '적응 N (prune+grow)'),
    ('L6', 'PROP-LS-APG', 'PROP', 'VARPRO 400 스텝 (비볼록 k 정련)'),
]
LINK_IDS = [l[0] for l in LINKS]
# ★고리별로 **결과 전에 선언한** config 차이. 자가진단이 `arm_config()` 의 실제 차이와
#   대조한다 ⟹ *"한 번에 한 인자"* 가 산문이 아니라 **검사되는 성질**이 된다.
#   ⚠**L4 는 완전한 단일요인이 아니다**(`PREREG §2.1` 정직 공시): MMSE 는 *"사전 γ +
#   val 선택 σ + EM"* 세 가지를 한 덩어리로 갖고, 두 계단이 **다른 코드 경로**라서
#   경로 스위치 키들이 함께 움직인다. 이 배치는 그 덩어리를 **하나의 인자**로 취급하고
#   분해는 별도 사전등록 몫이다. ⛔그래서 여기 적힌 키 목록이 넓다 — 숨기지 않는다.
LINK_DIFF_KEYS = {
    'L1': {'real_bank_ensemble_k'},
    'L2': {'bank_dk_frac'},
    'L3': {'bank_dk_frac', 'bank_k_set'},      # 후보집합 하나 (격자 vs 주입)
    'L4': {'init_method', 'real_domain_gate', 'real_estimator',
           'real_bank_ensemble_k', 'bank_k_set', 'varpro_steps',
           'adaptive_n'},                      # ⚠경로 스위치 + 결합기 덩어리
    'L5': {'adaptive_n'},
    'L6': {'varpro_steps'},
}
PRIMARY_LINK = 'L3'
DOSE_ARMS = ['BANK-K1', 'BANK-B472', 'BANK-B118', 'BANK-B30']   # L2 용량-반응 4점

N_PRIMARY = len(RECORDS)                                        # 12
N_RUNS = len(ARM_ORDER) * len(RECORDS) * len(EST_SEEDS)         # 432

# ─────────────────────────────────────────────────────────────────────────────
# 2. ★작업 목록·행 스탬프의 **단일출처** (`M4 AMENDMENT_1 A-3` 교훈)
#    ⛔러너와 게이트가 각자 알고 있으면 그것이 `G-13` 항등 실패를 낳은 결함이다.
# ─────────────────────────────────────────────────────────────────────────────
def expected_jobs():
    """★러너의 작업 목록 **단일출처**. `run_m8.jobs()` 가 이 함수를 부른다.

    순서 = (record, seed) 바깥 · arm 안쪽. ⟹ 장면(레코드당 ~15 s)과 원자 제안이
    셀 안에서 재사용되고, **샤드를 계단 축으로 쪼갤 수 있다**(`PREREG §15`).
    """
    return [(arm, rec, s) for rec in RECORDS for s in EST_SEEDS for arm in ARM_ORDER]


def row_identity(arm, record, seed):
    """★행의 **식별 스탬프** 단일출처. 러너가 모든 행에 이것을 얹는다."""
    return dict(batch=BATCH, arm=arm, record=record, est_seed=int(seed), win=WIN,
                rung=(RUNGS.index(arm) if arm in RUNGS else None),
                family=('ladder' if arm in RUNGS else 'secondary'))


def key(r):
    return (r.get('arm'), r.get('record'), r.get('est_seed'))


# ─────────────────────────────────────────────────────────────────────────────
# 3. 팔의 완성된 config — ★러너·자가진단이 **같은 함수**를 부른다
# ─────────────────────────────────────────────────────────────────────────────
def shipping_base():
    """배포 기본값 dict. ⚠**스냅샷이 아니다** — `assert_shipping_defaults()` 가
    `DEPLOY_EXPECTED` + `DEPLOY_EXTRA_EXPECTED` 핀으로 먼저 검사하고 죽는다."""
    from wavesplat.run import DEFAULT_CONFIG
    return dict(DEFAULT_CONFIG)


def assert_shipping_defaults():
    """★런 0회에서 죽는다 — 배포 기본값이 이 사전등록의 핀과 다르면 예외.

    ⛔조용한 흡수 금지(`gates_b7.py:46` 의 `{}` BLOCKER). 배포값이 바뀌었다면
    이 배치를 돌리기 전에 **사전등록을 개정**해야 한다."""
    from wavesplat.run import DEFAULT_CONFIG
    dep = DEPLOY()
    diff = {k: (dep.get(k), v) for k, v in DEPLOY_EXPECTED.items() if dep.get(k) != v}
    diff.update({k: (DEFAULT_CONFIG.get(k), v)
                 for k, v in DEPLOY_EXTRA_EXPECTED.items()
                 if DEFAULT_CONFIG.get(k) != v})
    if diff:
        raise SystemExit('⛔배포 기본값이 사전등록 핀과 다르다 (shipped, pinned) = '
                         f'{diff}\n   ⟹ 돌리기 전에 AMENDMENT 를 쓰라.')
    return True


def assert_bank_patch():
    """★런 0회 — `PATCH.md` 의 가산 2키가 **실제로 배선됐는지** 되읽는다.

    셋을 본다: ①`fit_bank_mmse` 시그니처에 `dk_frac`·`k_set` 이 있고 기본값이 `None`
    ②`_bank_fit_window` 가 `config.get('bank_dk_frac'/'bank_k_set')` 를 넘긴다
    ③**조용한 무시 방지 가드**(`real_estimator != 'bank_mmse'` 이면 예외)가 있다.
    ⛔*"설정을 넣었다 ≠ 설정이 먹었다"* — 이 저장소가 반복해서 대가를 치른 부류다."""
    from wavesplat.optimizer.bank_mmse import fit_bank_mmse
    from wavesplat.optimizer.sequential_update import SequentialUpdater
    ps = inspect.signature(fit_bank_mmse).parameters
    miss = [k for k in BANK_PATCH_KEYS if k not in ps]
    if miss:
        raise SystemExit(f'⛔bank_mmse 패치 미적용: fit_bank_mmse 에 {miss} 가 없다. '
                         '(PATCH.md §1 — 임무 A)')
    bad = [k for k in BANK_PATCH_KEYS if ps[k].default is not None]
    if bad:
        raise SystemExit(f'⛔패치가 기본값 보존형이 아니다: {bad} 의 기본값이 None 이 아니다.')
    src = inspect.getsource(SequentialUpdater._bank_fit_window)
    miss = [k for k in BANK_CFG_KEYS if k not in src]
    if miss:
        raise SystemExit(f'⛔_bank_fit_window 가 {miss} 를 넘기지 않는다. (PATCH.md §2)')
    whole = inspect.getsource(SequentialUpdater)
    if not all(k in whole for k in BANK_CFG_KEYS) or 'real_estimator' not in whole:
        raise SystemExit('⛔조용한 무시 방지 가드가 없다. (PATCH.md §2 후반)')
    return True


def arm_config(arm, *, depth, seed, atoms=None, atoms_band=None, kp_bank=None):
    """팔의 **완성된 estimator config**. ⛔자리표시자를 남기지 않는다.

    `atoms`/`atoms_band` = `(kx, ky)` 리스트 짝, `kp_bank` = bank 의 beamform kp.
    필요한 자리표시자에 값이 안 들어오면 **예외로 죽는다**(조용한 no-op 금지)."""
    cfg = shipping_base()
    cfg.update(depth=float(depth), seed=int(seed), device='cpu')
    over = dict(ARMS[arm])
    for k, v in list(over.items()):
        if v == ATOMS:
            if atoms is None:
                raise ValueError(f'{arm}: 제안 원자가 필요한데 주어지지 않았다')
            over[k] = (list(map(float, atoms[0])), list(map(float, atoms[1])))
        elif v == ATOMS_BAND:
            if atoms_band is None:
                raise ValueError(f'{arm}: 환대 필터 원자가 필요한데 주어지지 않았다')
            over[k] = (list(map(float, atoms_band[0])), list(map(float, atoms_band[1])))
        elif v == KP10:
            if kp_bank is None:
                raise ValueError(f'{arm}: kp_bank 가 필요한데 주어지지 않았다')
            over[k] = float(kp_bank) / 10.0
    cfg.update(over)
    leftover = [k for k, v in cfg.items()
                if isinstance(v, str) and v.startswith('<') and v.endswith('>')]
    if leftover:
        raise ValueError(f'{arm}: 자리표시자가 남았다 {leftover}')
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# 4. ★계단별 **개입 되읽기 기대값** — `G-INT-*` 가 이 표에 대고 검사한다
#    ⛔설정 dict 이 아니라 **객체**에서 되읽은 값이 여기에 온다.
# ─────────────────────────────────────────────────────────────────────────────
def expect(arm):
    a = ARMS[arm]
    bank = a['init_method'] == 'auto'
    return dict(
        path='bank' if bank else 'learned',
        real_gate=True if bank else None,
        # ★실측 게이트가 발화하면 `_initialize_inner:749` 가 `self.init_method` 를
        #   **'bank_mmse'** 로 바꾼다 ⟹ 되읽기 기대값은 'auto' 가 아니라 라우팅 **결과**다.
        #   (그래서 이 되읽기가 *"게이트가 실제로 발화했다"* 의 직접 증거가 된다.)
        init_method='bank_mmse' if bank else 'learned',
        auto_selected=('bank_mmse' if bank else None),
        n_gaussians=0 if bank else None,          # None = 자유 (n_sel 로 따로 검사)
        bank_members=(a.get('real_bank_ensemble_k') if bank else None),
        bank_dk_frac=(a.get('bank_dk_frac') if bank else None),
        kset=(arm in KSET_ARMS),
        vp_steps=(None if bank else int(a.get('varpro_steps', 400))),
        adaptive_n=(None if bank else bool(a.get('adaptive_n', True))),
        snap=(arm in SNAP_ARMS),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. 문턱 — **산술로 도출한다** (⛔손복사 금지. 전제 `P4` 와는 대조만)
# ─────────────────────────────────────────────────────────────────────────────
ALPHA = float(B22.ALPHA)                 # 0.05
N_FAMILY = len(LINKS)                    # 6 — Holm 가족 크기


def sign_p(w, n):
    return sum(comb(n, k) for k in range(w, n + 1)) / 2 ** n


def min_wins(n, alpha):
    """단측 부호검정에서 `p <= alpha` 를 주는 최소 승수. 없으면 `n + 1`."""
    for w in range(n + 1):
        if sign_p(w, n) <= alpha:
            return w
    return n + 1


def power_at(q, n, thr):
    return sum(comb(n, k) * q ** k * (1 - q) ** (n - k) for k in range(thr, n + 1))


def q80(n, thr):
    lo, hi = 0.5, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if power_at(mid, n, thr) < 0.80:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 4)


THRESHOLD_PRIMARY = min_wins(N_PRIMARY, ALPHA)                  # 10 (p=0.019287)
ALPHA_PRIMARY = sign_p(THRESHOLD_PRIMARY, N_PRIMARY)
Q80_PRIMARY = q80(N_PRIMARY, THRESHOLD_PRIMARY)                 # 0.8693
HOLM_ALPHAS = [ALPHA / (N_FAMILY - i) for i in range(N_FAMILY)]
HOLM_MIN_WINS = [min_wins(N_PRIMARY, a) for a in HOLM_ALPHAS]   # [11,11,11,11,10,10]
FLOOR_P = sign_p(N_PRIMARY, N_PRIMARY)                          # 2.44e-04


def holm(pvals):
    """Holm 단계적 하강. `pvals` = {id: p}. 반환 = (생존집합, 정렬된 (id, p, alpha, 기각))."""
    order = sorted(pvals.items(), key=lambda kv: (kv[1], kv[0]))
    surv, rows, stopped = set(), [], False
    for i, (k, p) in enumerate(order):
        a = ALPHA / (len(order) - i)
        rej = (not stopped) and (p is not None) and (p <= a)
        if not rej:
            stopped = True
        if rej:
            surv.add(k)
        rows.append(dict(link=k, p=p, alpha=a, rejected=bool(rej)))
    return surv, rows


# ─────────────────────────────────────────────────────────────────────────────
# 6. ★결과 전 해석 규칙 — **코드 안에** 둔다 (⛔결과를 보고 바꾸지 않는다)
# ─────────────────────────────────────────────────────────────────────────────
READING = {
    'DIRECTION': DIRECTION,
    'PRIMARY': (f'L3 = MMSE-PROP vs BANK-B30, {FIELD}, n={N_PRIMARY}(median-of-3), '
                f'단측 부호검정, 문턱 {THRESHOLD_PRIMARY}/{N_PRIMARY} '
                f'(α={ALPHA_PRIMARY:.6f}, 80% 검정력 요구 승률 q={Q80_PRIMARY}). '
                '★라벨은 **양방향 대칭**이다 — 본 방향이 라벨을 만들지 않는다.'),
    'CHAIN': (f'6 고리 Holm(α={ALPHA}). 생존집합 |S|=1 → SKILL-DIES-AT-<Lk> '
              '(Lk==L3 이면 CONFIRMATORY, 아니면 EXPLORATORY) · |S|>=2 → '
              'SKILL-DIES-DISTRIBUTED · S=∅ 이고 G-END 통과 → GAP-NOT-FACTORIZABLE · '
              'G-END 실패 → LADDER-VOID (판정 없음).'),
    'S_EMPTY_ORDER': ('⛔S=∅ 이면 판정 전에 **개입 되읽기**(G-INT-*)가 통과했는지부터 본다. '
                      '하나라도 실패하면 라벨은 LADDER-VOID 다. ⛔사후에 이 순서를 바꾸지 않는다.'),
    'MAGNITUDE': ('★크기 서술은 **로그비 분해로만**(서술 전용, 검정 아님): '
                  'Δ_k = median_records log(x_{k+1}/x_k), ΣΔ_k vs log(R8/R0). '
                  '⛔`F1` 교훈대로 **척도를 셋 병기**한다 — 짝 승수 · 절대 nRMSE · '
                  '참 제로필드 대비.'),
    'NOT_CLAIMED': [
        '⛔기제 지목 금지 — DIRECTION = LOCATION-OR-DECOMPOSITION.',
        '⛔배포 권고 금지 (어느 고리가 이겨도 "bank 를 배포에" · "max_n 을 올리자" 로 '
        '올리지 않는다. 별도 사전등록 몫. ⚠M4 가 제안 팔의 용량 증가를 12/12 로 죽였다).',
        '⛔크기·순위 주장 금지(F1) — 고리 간 크기 비교는 이 검정력으로 불가. 척도 셋 병기 의무.',
        '⛔`B-16` 합성 결론의 실측 전이 금지 — PROP-SNAP 은 **사후 양자화**이지 격자 적합이 아니다.',
        '⛔사이트 수준 일반화 금지(사이트 4개) · ⛔외부 기준선 추가 금지(S17+B-5+B2_WANGFIX) · '
        '⛔세 번째 LETKF 수리 금지.',
        '⛔ZERO_REF = 1.0 금지. 짝 비교만 판정에 쓴다.',
        '⛔M5b·M4 의 계수와 이 배치의 계수를 직접 비교 금지(장면 규약·팔이 다르다).',
        '⛔A7(신뢰 가중)·A1 제안쪽 앙상블을 "측정했다" 고 쓰지 않는다 — **도달 불가**로 공시한다.',
        '⛔L4 를 "MMSE 가 낫다/못하다" 로 분해하지 않는다 — 사전 γ·val σ·EM 이 한 덩어리다.',
        '⚠이 설계는 blind 가 아니다 — `PREREG §13` 의 노출 공시를 결과 문서에서 지우지 마라.',
    ],
    'RNG_DISCLOSURE': ('⛔행의 `rng_*`·`apg_*`·`n_sel` 계측은 **판정에 쓰지 않는다** — '
                       '공시·게이트 전용이다(`PREREG §6.1.4`).'),
}

# ★결과 전 선언: BLOCKING / SCOPING. ⛔결과를 보고 이 분류를 바꾸지 않는다.
SCOPING = {
    'G-BAND': 'MMSE-PROP-BAND 의 해석만 빠진다 (환대 제약을 배치와 분리하는 부차 팔).',
    'G-LEAD': 'leadmean(부차 지표) 해석만 빠진다. PRIMARY 는 nowcast 라 무관.',
    'G-0b2': '부차 팔 3개의 해석만 빠진다 (Holm 가족 **밖**이므로 사다리 판정 무관).',
    'G-0d2': '부차 팔 3개의 해석만 빠진다 (같은 사유).',
}

# 각 `gates()` 호출이 남기는 게이트별 커버리지(= 실제로 들여다본 행/짝 수).
COVERAGE = {}


# ─────────────────────────────────────────────────────────────────────────────
# 7. 외부 참조 로더 — ⛔비면 **빈 dict**(공허 PASS 금지). `bool(ref) and …` 로만 쓴다.
# ─────────────────────────────────────────────────────────────────────────────
def _load_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def b2_ship_ref():
    """`B-2` 동결 `SHIP` 행 → {record: {nowcast, leadmean, obs_hash, q_hash, n_obs}}.

    ★bank 끝단(`R0`)의 **비트 앵커**(`G-ANCHOR-B`). 시드 42 · win 3 · cap_mode uniform."""
    out = {}
    for r in _load_jsonl(FROZEN_B2):
        if r.get('cell') != 'SHIP' or r.get('win') != WIN or 'error' in r:
            continue
        out.setdefault(r['record'], dict(nowcast=r.get('nowcast'),
                                         leadmean=r.get('leadmean'),
                                         obs_hash=r.get('obs_hash'),
                                         q_hash=r.get('q_hash'),
                                         n_obs=r.get('n_obs'),
                                         n_atoms=r.get('n_atoms')))
    return out


def m4_prop_ref():
    """`M4` 동결 `SHIP-K-INL` 행 → {record: {...}}. ★제안 끝단(`R8`)의 **비트 앵커**.

    ⛔`B-2` 의 `PROP` 열은 쓰지 않는다 — CPU 에서 12/12 재현되지 않는다
    (`PREREG §5.1` BLOCKER-A: `propose_atoms` 가 `config['device']` 를 안 본다)."""
    out = {}
    for p in sorted(_glob.glob(FROZEN_M4_GLOB)):
        for r in _load_jsonl(p):
            if (r.get('arm') != M4_ANCHOR_ARM or r.get('stratum') != 'real'
                    or 'error' in r or int(r.get('est_seed', -1)) != ANCHOR_SEED):
                continue
            out.setdefault(r['unit'], dict(nowcast=r.get('nowcast'),
                                           leadmean=r.get('leadmean'),
                                           obs_hash=r.get('obs_hash'),
                                           q_hash=r.get('q_hash'),
                                           n_gaussians=r.get('n_gaussians'),
                                           apg_dn=r.get('apg_dn'),
                                           raw_atoms_md5=r.get('atoms_md5_inline'),
                                           cuda_available=r.get('cuda_available')))
    return out


def premise(path=None):
    p = PREMISE_JSON if path is None else path
    return json.load(open(p)) if os.path.exists(p) else {}


def premise_B(pre=None):
    """(record, frac) → 후보 기수 B. **비면 빈 dict**(공허 PASS 금지)."""
    d = ((pre if pre is not None else premise()).get('P2') or {}).get('B_of_frac') or {}
    return {(rec, float(f)): int(v) for rec, per in d.items() for f, v in per.items()}


def premise_kp(pre=None):
    return ((pre if pre is not None else premise()).get('P2') or {}).get('kp') or {}


def premise_zero(pre=None):
    return (pre if pre is not None else premise()).get('P6') or {}


def zero_ref_map():
    """참 제로필드 — ⛔`1.0` 상수 금지. `zero_ref.py` 가 유일 경로다(08-23 사고)."""
    from zero_ref import zero_ref_nowcast
    return {rec: float(zero_ref_nowcast(rec, win=WIN)) for rec in RECORDS}


def regress_report():
    return json.load(open(REGRESS_JSON)) if os.path.exists(REGRESS_JSON) else {}


def _fresher_than_sources(path):
    """산출물이 **패치 소스보다 새로운가**. ⛔낡은 통과 기록을 통과로 세지 않는다."""
    if not os.path.exists(path):
        return False, 'missing'
    t = os.path.getmtime(path)
    srcs = [REGRESS_SRC,
            os.path.join(ROOT, 'wavesplat', 'optimizer', 'bank_mmse.py'),
            os.path.join(ROOT, 'wavesplat', 'optimizer', 'sequential_update.py')]
    stale = [os.path.basename(s) for s in srcs
             if os.path.exists(s) and os.path.getmtime(s) > t]
    return (not stale), (f'stale vs {stale}' if stale else 'fresh')


def regress_pass(rep=None):
    """`PATCH.md §4` 의 네 겹 회귀가 통과했는가. ⛔없거나 낡으면 **FAIL**(공허 PASS 금지)."""
    rep = regress_report() if rep is None else rep
    if rep:
        top = rep.get('all_pass', rep.get('ok', rep.get('pass')))
        layers = rep.get('layers') or rep

        def truthy(v):
            if isinstance(v, dict):
                return bool(v.get('ok', v.get('pass',
                                              v.get('equal', v.get('n_bad') == 0))))
            return bool(v)

        got = {k: layers.get(k) for k in REGRESS_LAYERS}
        fresh, fmsg = _fresher_than_sources(REGRESS_JSON)
        ok = (top is not False and all(k in layers for k in REGRESS_LAYERS)
              and all(truthy(v) for v in got.values()) and fresh)
        return ok, (f'json all_pass={top} '
                    f'layers={ {k: truthy(v) for k, v in got.items()} } {fmsg}')
    if os.path.exists(REGRESS_OUT):
        txt = open(REGRESS_OUT, errors='replace').read()
        verdict_ok = REGRESS_VERDICT_RE in txt and '(0 실패)' in txt
        layers_ok = all(lay in txt for lay in REGRESS_LAYERS)
        fresh, fmsg = _fresher_than_sources(REGRESS_OUT)
        return (verdict_ok and layers_ok and fresh), (
            f'out verdict={verdict_ok} layers={layers_ok} {fmsg}')
    return False, (f'{os.path.basename(REGRESS_JSON)}/'
                   f'{os.path.basename(REGRESS_OUT)} 없음 — 계약: {PATCH_CONTRACT}')


def hash_obs(a, z=None):
    """★러너·게이트가 **같은 함수**로 해시한다(⛔두 곳에 복사하면 그게 결함 부류다).

    `run_b2.py:50` 규약 축자 ⟹ `B-2` 동결 `obs_hash` 와 **직접 비교 가능**하다."""
    import numpy as np
    arr = (np.column_stack([np.asarray(a, float), np.asarray(z, float)])
           if z is not None else np.asarray(a, float))
    return hashlib.md5(np.ascontiguousarray(arr, dtype=np.float64)).hexdigest()[:12]


def atoms_md5(kx, ky):
    """선택된 원자 집합 (kx, ky) 의 md5. ★러너와 게이트의 **유일 정의**."""
    import numpy as np
    if kx is None or ky is None:
        return None
    arr = np.column_stack([np.asarray(kx, float).ravel(),
                           np.asarray(ky, float).ravel()])
    return hashlib.md5(np.ascontiguousarray(arr, dtype=np.float64)).hexdigest()


def bits(x):
    import numpy as np
    return np.float64(x).tobytes().hex()


# ─────────────────────────────────────────────────────────────────────────────
# 8. 집계 — median-of-3 시드 → 레코드당 한 값 → 짝 부호검정
# ─────────────────────────────────────────────────────────────────────────────
def _int(v):
    """⛔`int(x or -1)` 금지 — **0 이 falsy** 라 정당한 0 을 결측으로 만든다.

    이 파일에서 같은 결함이 자가진단에 세 번 잡혔다(`G-INT-N`·`G-DEFAULT2`). 유일 경로."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _finite(x):
    try:
        return x is not None and float(x) == float(x) and abs(float(x)) != float('inf')
    except (TypeError, ValueError):
        return False


def med_by_record(rows, arm, field=FIELD):
    """(record → median-of-seeds). ⚠시드가 3개 다 있어야 그 레코드를 낸다."""
    per = {}
    for r in rows:
        if 'error' in r or r.get('arm') != arm or not _finite(r.get(field)):
            continue
        per.setdefault(r['record'], {})[int(r['est_seed'])] = float(r[field])
    return {rec: st.median([v[s] for s in sorted(v)])
            for rec, v in per.items() if len(v) == len(EST_SEEDS)}


def seed_spread(rows, arm, field=FIELD):
    """★시드 산포 — 판정문에 **병기**한다(`PREREG §4`). 판정에는 쓰지 않는다."""
    per = {}
    for r in rows:
        if 'error' in r or r.get('arm') != arm or not _finite(r.get(field)):
            continue
        per.setdefault(r['record'], []).append(float(r[field]))
    cv = []
    for rec, vs in per.items():
        if len(vs) > 1 and st.mean(vs) != 0:
            cv.append(abs(st.pstdev(vs) / st.mean(vs)))
    return dict(arm=arm, field=field, n_records=len(per),
                cv_median_pct=(round(100 * st.median(cv), 3) if cv else None),
                cv_max_pct=(round(100 * max(cv), 3) if cv else None))


def paired(rows, arm_a, arm_b, field=FIELD):
    """`arm_b` 가 `arm_a` 보다 **나쁜**(오차가 큰) 레코드를 센다 = 사슬의 검정 방향.

    `better` 는 반대 방향(작아진 레코드 수) — PRIMARY 의 양방향 라벨이 이것을 쓴다."""
    a = med_by_record(rows, arm_a, field)
    b = med_by_record(rows, arm_b, field)
    ks = sorted(set(a) & set(b))
    n = len(ks)
    worse = sum(1 for k in ks if b[k] > a[k])
    better = sum(1 for k in ks if b[k] < a[k])
    return {'a': arm_a, 'b': arm_b, 'field': field, 'n': n,
            'worse': worse, 'better': better, 'ties': n - worse - better,
            'rel_median_pct': (st.median([(b[k] - a[k]) / a[k] * 100 for k in ks])
                               if ks else None),
            'log_ratio_median': (st.median([_log(b[k] / a[k]) for k in ks])
                                 if ks else None),
            'p_worse': sign_test_one_sided(worse, n) if n else None,
            'p_better': sign_test_one_sided(better, n) if n else None,
            'records': ks}


def _log(x):
    import math
    return math.log(x) if x > 0 else float('nan')


def chain(rows, field=FIELD):
    """6 고리 전부. 각 고리의 검정 = *"제안 쪽 계단이 bank 쪽 계단보다 나쁘다"*."""
    out = []
    for lid, a, b, why in LINKS:
        d = paired(rows, a, b, field)
        d.update(link=lid, why=why)
        out.append(d)
    return out


def magnitude_table(rows, field=FIELD):
    """★척도 **셋** 병기 (`F1` 교훈). ⛔검정이 아니라 서술이다."""
    zr = premise_zero() or {}
    out = []
    for arm in RUNGS:
        m = med_by_record(rows, arm, field)
        if not m:
            out.append(dict(arm=arm, n=0))
            continue
        rel = [m[r] / float(zr[r]) for r in m if r in zr and float(zr.get(r, 0)) > 0]
        out.append(dict(arm=arm, rung=RUNGS.index(arm), n=len(m),
                        abs_nrmse_median=round(st.median(list(m.values())), 6),
                        vs_true_zero_median=(round(st.median(rel), 6) if rel else None)))
    links = []
    tot = 0.0
    for d in chain(rows, field):
        lr = d['log_ratio_median']
        if lr == lr:
            tot += lr
        links.append(dict(link=d['link'], a=d['a'], b=d['b'],
                          delta_log=(round(lr, 6) if lr == lr else None),
                          mult_median=(round(2.718281828459045 ** lr, 6)
                                       if lr == lr else None),
                          rel_median_pct=(round(d['rel_median_pct'], 4)
                                          if d['rel_median_pct'] is not None else None)))
    end = paired(rows, ENDPOINTS[0], ENDPOINTS[1], field)
    return dict(per_rung=out, per_link=links,
                sum_delta_log=round(tot, 6),
                endpoint_delta_log=(round(end['log_ratio_median'], 6)
                                    if end['log_ratio_median'] == end['log_ratio_median']
                                    else None),
                caveat=READING['MAGNITUDE'])


# ─────────────────────────────────────────────────────────────────────────────
# 9. 게이트
# ─────────────────────────────────────────────────────────────────────────────
def _sel(ok, arms):
    """팔 부분집합의 행 + **존재 가드**(분모를 조건에 넣는다 — `M4 G-15` 규약)."""
    arms = list(arms)
    sel = [r for r in ok if r.get('arm') in arms]
    full = len(sel) == len(arms) * len(RECORDS) * len(EST_SEEDS)
    return sel, full


def gates(rows):
    """★모든 게이트가 **커버리지**(실제로 본 행/짝 수)를 남긴다. `G-COV` 가 0 을 막는다."""
    ok = [r for r in rows if 'error' not in r]
    lad = [r for r in ok if r.get('arm') in RUNGS]
    sec = [r for r in ok if r.get('arm') in SECONDARY]
    out, cov = [], {}

    def A(gid, good, msg, n):
        out.append((gid, bool(good), str(msg)))
        cov[gid.split()[0]] = int(n)

    pre = premise()

    # ── G-0 계열 — 런 수·에러·중복·유한성 ────────────────────────────────────
    A('G-0 런 수', len(rows) == N_RUNS, f'{len(rows)}/{N_RUNS}', len(rows))
    err_l = [key(r) for r in rows if 'error' in r and r.get('arm') in RUNGS]
    A('G-0b 사다리 9계단 에러 0', not err_l, str(err_l[:3]),
      len([r for r in rows if r.get('arm') in RUNGS]))
    err_s = [key(r) for r in rows if 'error' in r and r.get('arm') in SECONDARY]
    A('G-0b2 부차 3팔 에러 0 (SCOPING)', not err_s, str(err_s[:3]),
      len([r for r in rows if r.get('arm') in SECONDARY]))
    A('G-0c 셀 중복 0', len({key(r) for r in rows}) == len(rows),
      f'{len(rows) - len({key(r) for r in rows})} 중복', len(rows))
    exp = set(expected_jobs())
    got = {(r.get('arm'), r.get('record'), int(r.get('est_seed', -1))) for r in rows}
    A('G-0e 작업목록 == expected_jobs() 전건', got == exp,
      f'누락 {len(exp - got)} 잉여 {len(got - exp)}', len(exp))
    # NaN 은 `a < b` 에서 조용히 **패배**가 된다 ⟹ 유한성을 명시적으로 요구한다.
    nf = [key(r) for r in lad
          if not (_finite(r.get(FIELD)) and _finite(r.get(FIELD_SECOND)))]
    A(f'G-0d 사다리 {FIELD}·{FIELD_SECOND} 전 행 유한', not nf and bool(lad),
      str(nf[:3]), len(lad))
    nf2 = [key(r) for r in sec
           if not (_finite(r.get(FIELD)) and _finite(r.get(FIELD_SECOND)))]
    A('G-0d2 부차 유한 (SCOPING)', not nf2 and bool(sec), str(nf2[:3]), len(sec))

    # ── ★G-DEV — BLOCKER-A 의 기계화 ────────────────────────────────────────
    dev = [r for r in ok if r.get('cuda_available') is not None]
    A('G-DEV ★전 행 cuda_available == False (디바이스 누출 방지)',
      bool(ok) and len(dev) == len(ok) and all(r['cuda_available'] is False for r in dev),
      f'{len(dev)}/{len(ok)} 기록 · True={sum(1 for r in dev if r["cuda_available"])}',
      len(dev))

    # ── G-INT-* 개입 되읽기 (⛔항등식 금지 — **객체** 되읽기·파생량·가지 계수기) ──
    bad, seen = [], 0
    for r in ok:
        e = expect(r['arm'])
        seen += 1
        if r.get('path_used') != e['path']:
            bad.append((r['arm'], r['record'], 'path', r.get('path_used')))
        elif r.get('init_method_readback') != e['init_method']:
            bad.append((r['arm'], r['record'], 'init', r.get('init_method_readback')))
        elif r.get('auto_selected') != e['auto_selected']:
            bad.append((r['arm'], r['record'], 'auto_selected', r.get('auto_selected')))
        elif e['path'] == 'bank' and r.get('real_gate') is not True:
            bad.append((r['arm'], r['record'], 'real_gate', r.get('real_gate')))
        elif e['path'] == 'learned' and r.get('real_gate') is not None:
            bad.append((r['arm'], r['record'], 'real_gate', r.get('real_gate')))
    A('G-INT ★경로·라우팅 되읽기 (객체에서)', not bad and bool(ok), str(bad[:3]), seen)

    # G-INT-N: 성분 수 — bank 는 0, LS 계열은 n_sel, 나머지는 > 0
    bad, seen = [], 0
    bs, bfull = _sel(ok, BANK_ARMS)
    for r in bs:
        seen += 1
        if _int(r.get('n_gaussians')) != 0:
            bad.append((r['arm'], r['record'], r.get('n_gaussians')))
    ls_arms = [a for a in LEARN_ARMS if not ARMS[a].get('adaptive_n', True)]
    lsr, lsfull = _sel(ok, ls_arms)
    for r in lsr:
        seen += 1
        if r.get('n_gaussians') != r.get('n_sel') or not r.get('n_sel'):
            bad.append((r['arm'], r['record'], r.get('n_gaussians'), r.get('n_sel')))
    ap_arms = [a for a in LEARN_ARMS if ARMS[a].get('adaptive_n', True)]
    apr, apfull = _sel(ok, ap_arms)
    for r in apr:
        seen += 1
        if (_int(r.get('n_gaussians')) or 0) <= 0:
            bad.append((r['arm'], r['record'], r.get('n_gaussians')))
    A('G-INT-N ★성분 수 되읽기: bank 0 · PROP-LS/PROP-LS-SNAP == n_sel · 적응 팔 > 0',
      not bad and bfull and lsfull and apfull, str(bad[:3]), seen)

    # G-INT-B: 후보 기수 — 전제 P2 와 대조 (BANK-B472/B118/B30)
    pB = premise_B(pre)
    bad, seen = [], 0
    dose_rows, dose_full = _sel(ok, list(DK_FRAC_ARMS))
    for r in dose_rows:
        frac = DK_FRAC_ARMS[r['arm']]
        want = pB.get((r['record'], float(frac)))
        seen += 1
        if want is None or _int(r.get('bank_B')) != int(want):
            bad.append((r['arm'], r['record'], r.get('bank_B'), want))
    A('G-INT-B ★BANK-B472/BANK-B118/BANK-B30 의 bank_B 가 전제 P2 와 12/12 일치',
      bool(pB) and dose_full and not bad, f'{seen} 행 · {str(bad[:3])}', seen)

    # G-INT-B2: 기수가 계단을 따라 **단조 감소** (같은 (record, seed) 안에서)
    ladder_b = ['BANK-K1'] + list(DK_FRAC_ARMS)
    lb_rows, lb_full = _sel(ok, ladder_b)
    idx = {(r['arm'], r['record'], r['est_seed']): r.get('bank_B') for r in lb_rows}
    bad, seen = [], 0
    for rec in RECORDS:
        for s in EST_SEEDS:
            vs = [idx.get((a, rec, s)) for a in ladder_b]
            if any(v is None for v in vs):
                bad.append((rec, s, 'missing')); continue
            seen += 1
            if not all(x > y for x, y in zip(vs, vs[1:])):
                bad.append((rec, s, vs))
    A('G-INT-B2 ★후보 기수가 L2 사다리에서 강한 단조 감소 (1904–1918 > 472 > 118 > 30)',
      lb_full and not bad, str(bad[:3]), seen)

    # G-INT-KSET: 주입한 원자가 **적합 객체 안에서** 후보가 됐는가
    ks_rows, ks_full = _sel(ok, KSET_ARMS)
    bad, seen = [], 0
    for r in ks_rows:
        seen += 1
        want = r.get('kset_md5') if r['arm'] == 'MMSE-PROP-BAND' else r.get('atoms_md5')
        if not r.get('bank_kset_md5') or r.get('bank_kset_md5') != want:
            bad.append((r['arm'], r['record'], 'md5'))
        elif _int(r.get('bank_B')) is None or _int(r.get('bank_B')) != _int(r.get('kset_n')):
            bad.append((r['arm'], r['record'], 'B!=kset_n',
                        r.get('bank_B'), r.get('kset_n')))
        elif r.get('bank_dk_c') is not None or r.get('bank_dk_c_nan') is not True:
            bad.append((r['arm'], r['record'], 'dk_c 가 NaN 이 아니다',
                        r.get('bank_dk_c'), r.get('bank_dk_c_nan')))
    A('G-INT-KSET ★주입 원자가 BankMMSEFit.kxb/kyb 로 되읽힌다 (dk_c 는 NaN)',
      ks_full and not bad, str(bad[:3]), seen)

    # G-INT-KSET2: 격자 팔은 그 반대 — 후보가 원자와 **다르다**(처치가 실제로 갈렸다)
    gr_rows, gr_full = _sel(ok, GRID_ARMS)
    bad, seen = [], 0
    for r in gr_rows:
        seen += 1
        if r.get('bank_kset_md5') and r.get('atoms_md5') \
                and r['bank_kset_md5'] == r['atoms_md5']:
            bad.append((r['arm'], r['record'], '격자 팔인데 후보가 원자와 같다'))
        if not _finite(r.get('bank_dk_c')) or r.get('bank_dk_c_nan') is not False:
            bad.append((r['arm'], r['record'], 'dk_c 가 유한하지 않다',
                        r.get('bank_dk_c')))
    A('G-INT-KSET2 격자 팔은 후보 != 제안 원자이고 dk_c 가 유한',
      gr_full and not bad, str(bad[:3]), seen)

    # G-INT-ENS: bank 앙상블 멤버 수 (L1 의 개입 되읽기)
    bad, seen = [], 0
    for r in bs:
        seen += 1
        if _int(r.get('bank_members')) != int(expect(r['arm'])['bank_members']):
            bad.append((r['arm'], r['record'], r.get('bank_members')))
        elif len(r.get('bank_seeds') or []) != int(expect(r['arm'])['bank_members']):
            bad.append((r['arm'], r['record'], 'seeds', r.get('bank_seeds')))
    A('G-INT-ENS ★BANK 는 median-of-3 · 나머지 bank 계단은 단일 (L1 개입 되읽기)',
      bfull and not bad, str(bad[:3]), seen)

    # G-INT-VP: VARPRO 실행 스텝 (L6 의 개입 되읽기) — info['steps_executed'] 되읽기
    lr_rows, lr_full = _sel(ok, LEARN_ARMS)
    bad, seen = [], 0
    for r in lr_rows:
        want = expect(r['arm'])['vp_steps']
        seen += 1
        if want == 0:
            if r.get('vp_fired') or _int(r.get('vp_steps_executed')) != 0:
                bad.append((r['arm'], r['record'], r.get('vp_steps_executed')))
        else:
            if not r.get('vp_fired') or _int(r.get('vp_steps_executed')) != want:
                bad.append((r['arm'], r['record'], r.get('vp_steps_executed'), want))
    A('G-INT-VP ★VARPRO 실행 스텝을 varpro info 에서 되읽는다 (0 또는 400)',
      lr_full and not bad, str(bad[:3]), seen)

    # G-INT-APG: 적응 N **가지** 계수기 (L5). ⛔함수 호출 수가 아니라 가지다(B-2f 교훈)
    bad, seen = [], 0
    for r in lr_rows:
        want = expect(r['arm'])['adaptive_n']
        seen += 1
        if r.get('adaptive_n_readback') is not want:
            bad.append((r['arm'], r['record'], 'flag', r.get('adaptive_n_readback')))
        elif want is False and (_int(r.get('apg_calls')) != 0
                                or (r.get('apg_dn') or []) != []):
            bad.append((r['arm'], r['record'], 'apg', r.get('apg_calls')))
        elif want is True and (_int(r.get('apg_calls')) or 0) < 1:
            bad.append((r['arm'], r['record'], 'apg 미발화', r.get('apg_calls')))
    A('G-INT-APG ★적응 N 이 가지 단위로 되읽힌다 (off 팔은 정확히 0회)',
      lr_full and not bad, str(bad[:3]), seen)

    # G-INT-SNAP: 파수 격자 스냅이 정확히 부차 두 팔에서만 발화
    bad, seen = [], 0
    for r in ok:
        want = expect(r['arm'])['snap']
        if r['arm'] not in LEARN_ARMS:
            continue
        seen += 1
        if bool(r.get('snap_fired')) != want:
            bad.append((r['arm'], r['record'], r.get('snap_fired')))
        elif want and not (_finite(r.get('snap_dk')) and float(r['snap_dk']) > 0):
            bad.append((r['arm'], r['record'], 'snap_dk', r.get('snap_dk')))
    sn_rows, sn_full = _sel(ok, SNAP_ARMS)
    A('G-INT-SNAP ★k_snap 이 PROP-SNAP·PROP-LS-SNAP 에서만 발화 (k_snap.fired 되읽기)',
      lr_full and sn_full and not bad, str(bad[:3]), seen)

    # ── G-ATOM ★원자 동일성: R5–R8 · 부차 제안 팔이 같은 원자를 본다 ──────────
    at_rows, at_full = _sel(ok, ATOM_ARMS)
    bad, seen = [], 0
    for rec in RECORDS:
        for s in EST_SEEDS:
            ms = {r['arm']: r.get('atoms_md5') for r in at_rows
                  if r['record'] == rec and r['est_seed'] == s}
            if len(ms) != len(ATOM_ARMS) or any(v is None for v in ms.values()):
                bad.append((rec, s, 'missing', sorted(ms))); continue
            seen += 1
            if len(set(ms.values())) != 1:
                bad.append((rec, s, sorted(set(ms.values()))[:2]))
    A('G-ATOM ★제안 원자 md5 가 (record, seed) 안에서 전 팔 일치 (인라인 제안의 정합)',
      at_full and not bad, f'{seen} 셀 · {str(bad[:3])}', seen)

    # ── G-RNG / G-RNG2 (계측 shim 사망 = 항등 통과 금지) ─────────────────────
    miss = [key(r) for r in ok if not (r.get('rng_hash_after_seed')
                                       and r.get('rng_hash_after_init'))]
    A('G-RNG RNG 상태 해시가 전 행에 비어있지 않게 기록됐다 (⛔공시 전용)',
      bool(ok) and not miss, str(miss[:3]), len(ok))
    rng2_arms = ['PROP-LS', 'PROP-LS-APG', 'PROP']
    r2_rows, r2_full = _sel(ok, rng2_arms)
    bad, seen = [], 0
    for rec in RECORDS:
        for s in EST_SEEDS:
            ns = {r['arm']: r.get('n_sel') for r in r2_rows
                  if r['record'] == rec and r['est_seed'] == s}
            if len(ns) != len(rng2_arms) or any(v is None for v in ns.values()):
                bad.append((rec, s, 'missing')); continue
            seen += 1
            if len(set(ns.values())) != 1:
                bad.append((rec, s, ns))
    A('G-RNG2 ★PROP-LS·PROP-LS-APG·PROP 의 n_sel 이 (record, seed) 안에서 일치 '
      '(전역 스트림 정렬의 필요조건)', r2_full and not bad, str(bad[:3]), seen)

    # ── G-SCENE 단일요인: 장면이 팔에 걸쳐 비트동일 ──────────────────────────
    bad, seen = [], 0
    for rec in RECORDS:
        grp = [r for r in ok if r['record'] == rec]
        if not grp:
            bad.append((rec, 'no rows')); continue
        seen += 1
        for k in ('obs_hash', 'q_hash', 'gt_hash', 'sigma_bits', 'kp_bits', 'n_obs'):
            if len({r.get(k) for r in grp}) != 1:
                bad.append((rec, k))
    A('G-SCENE 관측·질의·참값·σ·kp 가 12팔 × 3시드에 걸쳐 비트동일 (단일요인)',
      not bad and seen == len(RECORDS), str(bad[:4]), seen)

    # G-SCENE-B2: 그 장면이 `B-2` 동결 장면과 **비트동일** (외부 앵커)
    ref2 = b2_ship_ref()
    bad, seen = [], 0
    for r in ok:
        c = ref2.get(r['record'])
        if not c:
            continue
        seen += 1
        if r.get('obs_hash') != c['obs_hash'] or r.get('q_hash') != c['q_hash'] \
                or _int(r.get('n_obs')) != int(c['n_obs']):
            bad.append((r['arm'], r['record'], r.get('obs_hash'), c['obs_hash']))
    A('G-SCENE-B2 ★장면이 B-2 동결 장면과 비트동일 (obs_hash·q_hash·n_obs)',
      bool(ref2) and len(ref2) == len(RECORDS) and not bad and seen == len(ok),
      f'ref={len(ref2)}/{len(RECORDS)} · {str(bad[:2])}', seen)

    # ── G-ANCHOR-B / G-ANCHOR-P ★양성대조(비트) ─────────────────────────────
    bad, seen = [], 0
    for rec in RECORDS:
        c = ref2.get(rec)
        rr = [r for r in ok if r['arm'] == 'BANK' and r['record'] == rec
              and int(r['est_seed']) == ANCHOR_SEED]
        if not c or len(rr) != 1:
            bad.append((rec, 'missing')); continue
        seen += 1
        if rr[0].get(FIELD) != c['nowcast'] or rr[0].get(FIELD_SECOND) != c['leadmean']:
            bad.append((rec, rr[0].get(FIELD), c['nowcast']))
    A('G-ANCHOR-B ★R0(BANK, seed 42) == B-2 동결 SHIP **비트** 12/12',
      bool(ref2) and not bad and seen == len(RECORDS), str(bad[:3]), seen)

    ref4 = m4_prop_ref()
    bad, seen = [], 0
    for rec in RECORDS:
        c = ref4.get(rec)
        rr = [r for r in ok if r['arm'] == 'PROP' and r['record'] == rec
              and int(r['est_seed']) == ANCHOR_SEED]
        if not c or len(rr) != 1:
            bad.append((rec, 'missing')); continue
        seen += 1
        if rr[0].get(FIELD) != c['nowcast'] or rr[0].get(FIELD_SECOND) != c['leadmean']:
            bad.append((rec, rr[0].get(FIELD), c['nowcast']))
        elif _int(rr[0].get('n_gaussians')) != int(c['n_gaussians']):
            bad.append((rec, 'N', rr[0].get('n_gaussians'), c['n_gaussians']))
        elif list(rr[0].get('apg_dn') or []) != list(c['apg_dn'] or []):
            bad.append((rec, 'apg_dn', rr[0].get('apg_dn'), c['apg_dn']))
    A('G-ANCHOR-P ★R8(PROP, seed 42) == M4 동결 SHIP-K-INL **비트** 12/12 '
      '(⛔B-2 PROP 은 앵커가 아니다)',
      bool(ref4) and len(ref4) == len(RECORDS) and not bad and seen == len(RECORDS),
      f'ref={len(ref4)}/{len(RECORDS)} · {str(bad[:3])}', seen)

    # ── ★G-END 끝점 대비를 이 배치가 **직접** 낸다 (수입 금지) ───────────────
    end = paired(rows, ENDPOINTS[0], ENDPOINTS[1], FIELD)
    A(f'G-END ★{ENDPOINTS[0]} < {ENDPOINTS[1]} 이 >= {THRESHOLD_PRIMARY}/{N_PRIMARY} '
      '(끝점이 알려진 대비를 재현) — 실패 시 LADDER-VOID',
      end['n'] == N_PRIMARY and end['worse'] >= THRESHOLD_PRIMARY,
      f'{end["worse"]}/{end["n"]} p={end["p_worse"]}', end['n'])

    # ── G-PATCH ★패키지 패치가 배선됐고 기본값 경로가 비트동일하다 ───────────
    sig_ok, sig_msg = _patch_sig_report()
    A('G-PATCH-SIG ★가산 2키가 실제로 배선됐다 (시그니처·배선·무시방지 가드 되읽기)',
      sig_ok, sig_msg, 3)
    reg_ok, reg_msg = regress_pass()
    A('G-PATCH-REG ★기본값 경로 비트동일 회귀 4겹이 통과했다 (regress_bank_mmse.json)',
      reg_ok, reg_msg, 4)
    bad = [key(r) for r in bs
           if (r.get('cap_bank_dk_frac') is not None) != (r['arm'] in DK_FRAC_ARMS)
           or (r.get('cap_bank_k_set_n') is not None) != (r['arm'] in KSET_ARMS)]
    A('G-PATCH-OFF 두 새 키가 R0·R1 에서 **꺼져 있다**(None) 그리고 처치 팔에서만 켜진다',
      bfull and not bad, str(bad[:3]), len(bs))

    # ── G-DEFAULT 배포 상수 되읽기 (⛔빈 dict `{}` 로 팔을 정의하지 않았다) ──
    drift = {}
    for k in ('lambda_l1', 'max_n', 'prune_threshold', 'grow_residual_threshold',
              'use_confidence', 'tracking_mode', 'real_estimator', 'device'):
        seenv = {r.get('cap_' + k) for r in ok}
        if len(seenv) != 1:
            drift[k] = sorted(str(x) for x in seenv)[:4]
    A('G-DEFAULT 배포 상수(객체 되읽기)가 전 런 고정 — 조용한 흡수 없음',
      bool(ok) and not drift, str(drift), len(ok))
    li = [r for r in lr_rows if r.get('init_iters_readback') != 0]
    A('G-DEFAULT2 learned_init_iters=0 을 **결과 객체**에서 되읽는다 (n_iters == 0)',
      lr_full and not li, f'{len(li)} 위반', len(lr_rows))
    empty = [a for a, v in ARMS.items() if not v]
    A('G-DEFAULT3 ⛔빈 dict `{}` 로 정의된 팔이 없다', not empty, str(empty), len(ARMS))

    # ── G-ZERO 채점 기준 (⛔`ZERO_REF=1.0` 금지) ────────────────────────────
    pz = premise_zero(pre)
    bad, seen = [], 0
    for r in ok:
        z = r.get('zero_ref')
        seen += 1
        if not _finite(z) or float(z) == 1.0:
            bad.append((r['record'], z))
        elif r['record'] in pz and abs(float(z) - float(pz[r['record']])) > 1e-6:
            bad.append((r['record'], z, pz[r['record']]))
    A('G-ZERO ★참 제로필드가 zero_ref.py 에서 오고 어느 레코드도 정확히 1.0 이 아니다',
      bool(pz) and not bad and seen == len(ok), str(bad[:3]), seen)

    # ── G-KP bank 의 beamform kp 가 전제 P2 와 일치 (기수 표의 전제) ─────────
    pkp = premise_kp(pre)
    bad, seen = [], 0
    for r in bs:
        w = pkp.get(r['record'])
        seen += 1
        if w is None or not _finite(r.get('bank_kp')) \
                or abs(float(r['bank_kp']) - float(w)) > 1e-5:
            bad.append((r['arm'], r['record'], r.get('bank_kp'), w))
        elif r.get('kp_bank_calc') is not None \
                and float(r['kp_bank_calc']) != float(r['bank_kp']):
            # ★하네스가 재현한 zoom 과 **적합 객체의 kp** 가 비트 동일해야 한다
            #   (환대 필터·k_snap_dk 가 그 재현값을 쓰기 때문이다)
            bad.append((r['arm'], r['record'], 'calc!=obj',
                        r.get('kp_bank_calc'), r.get('bank_kp')))
    A('G-KP bank 의 2단 beamform kp 가 전제 P2 와 일치 + 하네스 재현과 비트 동일',
      bool(pkp) and bfull and not bad, str(bad[:3]), seen)

    # ── G-SEED 시드 3개가 전건 있다 ────────────────────────────────────────
    bad = [(a, rec) for a in ARM_ORDER for rec in RECORDS
           if {int(r['est_seed']) for r in ok
               if r['arm'] == a and r['record'] == rec} != set(EST_SEEDS)]
    A(f'G-SEED 팔×레코드마다 시드 {EST_SEEDS} 전건 (median-of-3 의 전제)',
      not bad, str(bad[:3]), len(ARM_ORDER) * len(RECORDS))

    # ── SCOPING 게이트 둘 ──────────────────────────────────────────────────
    bd_rows, bd_full = _sel(ok, ['MMSE-PROP-BAND'])
    bad, seen = [], 0
    for r in bd_rows:
        seen += 1
        f = r.get('frac_in_band')
        if not _finite(f) or int(r.get('kset_n') or 0) < 1:
            bad.append((r['record'], f, r.get('kset_n')))
    A('G-BAND (SCOPING) MMSE-PROP-BAND 가 frac_in_band 를 싣고 원자가 남았다',
      bd_full and not bad, str(bad[:3]), seen)

    bad, seen = [], 0
    for rec in RECORDS:
        ls = {len(r.get('curve') or []) for r in ok if r['record'] == rec}
        nl = {_int(r.get('n_leads')) for r in ok if r['record'] == rec}
        if not ls:
            bad.append((rec, 'no rows')); continue
        seen += 1
        if len(ls) != 1 or ls != nl or 0 in ls:
            bad.append((rec, sorted(ls), sorted(nl)))
    A('G-LEAD (SCOPING) curve 길이가 레코드별 lead 수와 일치하고 팔에 걸쳐 같다',
      not bad and seen == len(RECORDS), str(bad[:3]), seen)

    # ── ★G-COV 메타: 커버리지 0 인 게이트가 없다 (필터 항진 금지) ────────────
    zero = sorted(k for k, v in cov.items() if v <= 0)
    A('G-COV ★커버리지 0 게이트 없음 (항등적으로 참/거짓인 검사 금지)',
      not zero, str(zero), len(cov))

    COVERAGE.clear()
    COVERAGE.update(cov)
    return out


def _patch_sig_report():
    try:
        assert_bank_patch()
        return True, 'fit_bank_mmse(dk_frac=None, k_set=None) + _bank_fit_window 배선 + 가드'
    except SystemExit as e:
        return False, str(e)[:150]
    except Exception as e:                                        # noqa: BLE001
        return False, repr(e)[:150]


def blocking_failures(gate_rows):
    """SCOPING 으로 **결과 전에** 선언한 것을 뺀 FAIL 목록."""
    return [gid for gid, ok_, _ in gate_rows
            if not ok_ and gid.split()[0] not in SCOPING]


# ─────────────────────────────────────────────────────────────────────────────
# 10. 라벨 — ★네 갈래 전부 결과 전에 적는다
# ─────────────────────────────────────────────────────────────────────────────
def primary_label(prim):
    """PRIMARY = L3, `MMSE-PROP` vs `BANK-B30`. ★양방향 대칭."""
    b, w, n = prim['better'], prim['worse'], prim['n']
    if b >= THRESHOLD_PRIMARY:
        return ('PRIOR-ATOMS-EARN',
                f'`MMSE-PROP < BANK-B30` {b}/{n} (p={prim["p_better"]:.6f}). '
                '★**기여 ① 의 실측 최초 양성** — *"같은 기수·같은 결합기·같은 관측에서 '
                '학습된 원자가 규칙 격자를 이긴다."* ⛔*"사전이 실측에서 작동한다"* 로 '
                '올리지 않는다(배포 팔은 여전히 진다).')
    if w >= THRESHOLD_PRIMARY:
        return ('PRIOR-ATOMS-LOSE',
                f'`MMSE-PROP > BANK-B30` {w}/{n} (p={prim["p_worse"]:.6f}). '
                '*"원자 자체가 규칙 격자보다 나쁘다"* ⟹ 경계가 **사전 자신** 위에 있다.')
    return ('PRIOR-ATOMS-UNDECIDED',
            f'{b} better / {w} worse / {n} (문턱 {THRESHOLD_PRIMARY}). '
            f'⛔*"차이 없다"* 금지(등가검정 안 했다). MDE: 80% 검정력이 요구하는 승률 '
            f'q={Q80_PRIMARY}, 부호검정 바닥 p={FLOOR_P:.2e}.')


def chain_label(links, surv, end_ok):
    """사슬 라벨 — Holm 생존집합 `surv` 로 네 갈래."""
    if not end_ok:
        return ('LADDER-VOID',
                '⛔`G-END` 실패 — 끝점이 알려진 대비를 재현하지 못했다. '
                '사다리는 아무것도 국소화할 수 없다. **판정 없음.**')
    if len(surv) == 1:
        lk = sorted(surv)[0]
        why = dict((l[0], l[3]) for l in LINKS)[lk]
        tag = 'CONFIRMATORY' if lk == PRIMARY_LINK else 'EXPLORATORY'
        return (f'SKILL-DIES-AT-{lk}',
                f'고리 {lk}({why}) 하나가 Holm 문턱을 넘었다 — **{tag}** '
                f'({"PRIMARY 그 자체" if lk == PRIMARY_LINK else "PRIMARY 가 아니다"}). '
                '⛔*"원인은 X 다"* 로 올리지 않는다(DIRECTION 은 국소화까지다).')
    if len(surv) >= 2:
        return ('SKILL-DIES-DISTRIBUTED',
                f'Holm 생존 {sorted(surv)} — ⟹ ***"단일 원인이 아니다"***. '
                '논문 §sec:boundary 에 **분해표**를 싣고 ⛔*"가장 큰 고리"* 를 '
                '**지목하지 않는다**(크기 순위는 이 검정력으로 불가, `F1`).')
    return ('GAP-NOT-FACTORIZABLE',
            '끝점은 갈리는데(G-END 통과) 어느 단일 고리도 Holm 문턱을 못 넘었다 '
            '⟹ **상호작용**. 다음 설계는 **2요인 팔**이거나 **레코드 증설**이다. '
            '⛔*"차이 없다"* 금지.')


def verdict(rows, gate_rows=None, field=FIELD):
    """★판정의 유일 정의. `analyze_m8.py` 는 이것을 부르기만 한다.

    ★**`G-END` 가 실패하면 `LADDER-VOID` 이고 그때는 PRIMARY·사슬·크기표를 내지 않는다**
    (`PREREG §10.3` 축자: *"개입이 안 걸렸다는 뜻이므로 **결과를 열지 않고** 진단서를
    쓴다"* — `M5`→`M5b` 선례). ⛔사후에 이 순서를 바꾸지 않는다.
    """
    gate_rows = gates(rows) if gate_rows is None else gate_rows
    gd = dict((g.split()[0], ok_) for g, ok_, _ in gate_rows)
    end_ok = bool(gd.get('G-END'))
    if not end_ok:
        clab, cwhy = chain_label(None, set(), False)
        return dict(batch=BATCH, direction=DIRECTION, field=field,
                    chain_label=clab, chain_why=cwhy,
                    primary_label=None, primary_why=(
                        '⛔`LADDER-VOID` — 결과를 열지 않는다. PRIMARY·사슬·크기표를 '
                        '내지 않고 진단서를 쓴다(`PREREG §10.3`).'),
                    primary=None, chain=None, holm=None, holm_survivors=None,
                    magnitude=None, seed_spread=None,
                    endpoint=paired(rows, ENDPOINTS[0], ENDPOINTS[1], field),
                    thresholds=dict(primary=THRESHOLD_PRIMARY,
                                    alpha_primary=ALPHA_PRIMARY,
                                    q80_primary=Q80_PRIMARY,
                                    holm_alphas=HOLM_ALPHAS,
                                    holm_min_wins=HOLM_MIN_WINS, floor_p=FLOOR_P),
                    reading=READING)
    links = chain(rows, field)
    pv = {d['link']: d['p_worse'] for d in links}
    surv, holm_rows = holm(pv)
    prim = paired(rows, 'BANK-B30', 'MMSE-PROP', field)
    prim['link'] = PRIMARY_LINK
    plab, pwhy = primary_label(prim)
    clab, cwhy = chain_label(links, surv, end_ok)
    return dict(
        batch=BATCH, direction=DIRECTION, field=field,
        primary_label=plab, primary_why=pwhy, primary=prim,
        chain_label=clab, chain_why=cwhy,
        chain=links, holm=holm_rows, holm_survivors=sorted(surv),
        endpoint=paired(rows, ENDPOINTS[0], ENDPOINTS[1], field),
        thresholds=dict(primary=THRESHOLD_PRIMARY, alpha_primary=ALPHA_PRIMARY,
                        q80_primary=Q80_PRIMARY, holm_alphas=HOLM_ALPHAS,
                        holm_min_wins=HOLM_MIN_WINS, floor_p=FLOOR_P),
        magnitude=magnitude_table(rows, field),
        seed_spread=[seed_spread(rows, a, field) for a in ARM_ORDER],
        reading=READING)


# ─────────────────────────────────────────────────────────────────────────────
# 11. 자가진단 — ★변형 하네스는 **러너의 작업목록·스탬프**에서 행을 만든다
#     (`M4 G-13` 의 **필터 항진**은 게이트를 모형하면 잡히지 않는다)
# ─────────────────────────────────────────────────────────────────────────────
def _fixture(anchored=True, link_wins=None, premise_path=None):
    """정상 세계 합성 행. ⛔판정 자료가 아니다.

    `anchored=True` 면 **실제 동결 파일과 전제 파일**의 값을 싣는다 — 그래야
    `G-ANCHOR-*`·`G-SCENE-B2`·`G-INT-B`·`G-ZERO`·`G-KP` 가 **자료에 대고 만족 가능**한지
    사전등록 시점에 시험된다.
    """
    pre = premise(premise_path)
    b2 = b2_ship_ref() if anchored else {}
    m4 = m4_prop_ref() if anchored else {}
    pB = premise_B(pre)
    pkp = premise_kp(pre)
    pz = premise_zero(pre)
    wins = dict(link_wins or {})
    rows = []
    for (arm, rec, s) in expected_jobs():
        ri = RECORDS.index(rec)
        c2 = b2.get(rec) or {}
        c4 = m4.get(rec) or {}
        lo = float(c2.get('nowcast') or (0.5 + 0.01 * ri))
        hi = float(c4.get('nowcast') or (0.9 + 0.01 * ri))
        if arm in RUNGS:
            i = RUNGS.index(arm)
            v = lo + (hi - lo) * i / (len(RUNGS) - 1)
            if link_wins:            # 라벨 분기 시험용 — 앵커를 쓰지 않는다
                v = 1.0
                for k, (lid, a, b, _w) in enumerate(LINKS):
                    if RUNGS.index(b) <= i:
                        v += (0.05 if ri < wins.get(lid, 0) else -0.05)
            if not link_wins:
                if arm == 'BANK':
                    v = lo
                if arm == 'PROP':
                    v = hi
        else:
            v = hi + 0.02 * (SECONDARY.index(arm) + 1)
        if int(s) != ANCHOR_SEED:
            v = v * (1.0 + 1e-6 * (int(s) - ANCHOR_SEED))
        e = expect(arm)
        n_sel = 26 + (ri % 5)
        frac = ARMS[arm].get('bank_dk_frac')
        kset_n = (n_sel if arm == 'MMSE-PROP'
                  else (max(1, n_sel - 2) if arm == 'MMSE-PROP-BAND' else None))
        amd5 = f'atoms{rec}{s}'
        kmd5 = (amd5 if arm == 'MMSE-PROP'
                else (f'band{rec}{s}' if arm == 'MMSE-PROP-BAND' else None))
        if e['path'] == 'bank':
            bank_B = (kset_n if arm in KSET_ARMS
                      else int(pB.get((rec, float(frac)), 1908) if frac
                               else pB.get((rec, 0.1), 1908)))
            bank_dk_c = None if arm in KSET_ARMS else 0.02
            bank_dk_c_nan = (arm in KSET_ARMS)
            n_g = 0
        else:
            bank_B = bank_dk_c = bank_dk_c_nan = None
            n_g = n_sel if e['adaptive_n'] is False else max(3, n_sel - 7)
        row = dict(
            cuda_available=False,
            path_used=e['path'], init_method_readback=e['init_method'],
            auto_selected=e['auto_selected'],
            real_gate=(True if e['path'] == 'bank' else None),
            n_gaussians=n_g, n_sel=n_sel,
            bank_B=bank_B, bank_dk_c=bank_dk_c, bank_dk_c_nan=bank_dk_c_nan,
            bank_kp=(float(pkp.get(rec, 0.2)) if e['path'] == 'bank' else None),
            kp_bank_calc=(float(pkp.get(rec, 0.2))
                          if (e['path'] == 'bank' or e['snap']) else None),
            bank_members=e['bank_members'],
            bank_seeds=(list(range(int(e['bank_members'] or 0)))
                        if e['path'] == 'bank' else None),
            bank_kset_md5=(kmd5 if arm in KSET_ARMS
                           else (f'grid{rec}{arm}' if e['path'] == 'bank' else None)),
            atoms_md5=(amd5 if arm in ATOM_ARMS else None),
            raw_atoms_md5=(c4.get('raw_atoms_md5') if arm in LEARN_ARMS else None),
            kset_n=kset_n, kset_md5=kmd5,
            frac_in_band=(0.9 if arm == 'MMSE-PROP-BAND' else None),
            vp_fired=(e['vp_steps'] not in (None, 0)),
            vp_steps_executed=(0 if e['vp_steps'] is None else int(e['vp_steps'])),
            adaptive_n_readback=e['adaptive_n'],
            apg_calls=(1 if e['adaptive_n'] else 0),
            apg_dn=([n_g - n_sel] if e['adaptive_n'] else []),
            init_iters_readback=(0 if e['path'] == 'learned' else None),
            snap_fired=e['snap'], snap_dk=(float(pkp.get(rec, 0.2)) / 10
                                           if e['snap'] else None),
            rng_hash_after_seed=f'rs{rec}{s}', rng_hash_after_init=f'ri{rec}{s}{arm}',
            rng_torch_calls=7, rng_torch_elems=99,
            obs_hash=c2.get('obs_hash') or f'o{rec}',
            q_hash=c2.get('q_hash') or f'q{rec}',
            gt_hash=f'g{rec}', sigma_bits=f's{rec}', kp_bits=f'k{rec}',
            n_obs=int(c2.get('n_obs') or 600), npx=396,
            n_leads=15, curve=[v] * 15,
            zero_ref=float(pz.get(rec, 0.7)),
            nowcast=v, leadmean=v * 1.1,
            nowcast_bits=bits(v), leadmean_bits=bits(v * 1.1),
            depth=17.0, Hs=1.5, sigma=0.39, dt=0.1, wall_s=1.0,
            **{'cap_' + k: v2 for k, v2 in dict(
                lambda_l1=1e-4, max_n=30, prune_threshold=0.10,
                grow_residual_threshold=0.05, use_confidence=True,
                tracking_mode='w_phi', real_estimator='bank_mmse',
                device='cpu').items()},
            cap_bank_dk_frac=frac,
            cap_bank_k_set_n=(kset_n if arm in KSET_ARMS else None),
        )
        row.update(row_identity(arm, rec, s))
        # ★앵커 팔은 동결 파일의 **그 값**을 그대로 싣는다
        if anchored and int(s) == ANCHOR_SEED and arm == 'BANK' and c2:
            row.update(nowcast=c2['nowcast'], leadmean=c2['leadmean'],
                       nowcast_bits=bits(c2['nowcast']),
                       leadmean_bits=bits(c2['leadmean']))
        if anchored and int(s) == ANCHOR_SEED and arm == 'PROP' and c4:
            row.update(nowcast=c4['nowcast'], leadmean=c4['leadmean'],
                       n_gaussians=int(c4['n_gaussians']),
                       apg_dn=list(c4['apg_dn'] or []),
                       n_sel=int(c4['n_gaussians']) - sum(c4['apg_dn'] or [0]),
                       nowcast_bits=bits(c4['nowcast']),
                       leadmean_bits=bits(c4['leadmean']))
        rows.append(row)
    # ★`n_sel` 은 (record, seed) 안에서 R6·R7·R8 이 같아야 한다 (`G-RNG2`)
    ns = {(r['record'], r['est_seed']): r['n_sel'] for r in rows if r['arm'] == 'PROP'}
    for r in rows:
        if r['arm'] in LEARN_ARMS or r['arm'] in KSET_ARMS:
            v = ns[(r['record'], r['est_seed'])]
            r['n_sel'] = v
            if r['arm'] in ('PROP-LS', 'PROP-LS-SNAP'):
                r['n_gaussians'] = v
            elif r['arm'] in LEARN_ARMS:
                r['n_gaussians'] = max(3, v - 7)
                r['apg_dn'] = [r['n_gaussians'] - v]
            if r['arm'] == 'MMSE-PROP':
                r['kset_n'] = v
                r['bank_B'] = v
            elif r['arm'] == 'MMSE-PROP-BAND':
                r['kset_n'] = max(1, v - 2)
                r['bank_B'] = r['kset_n']
    for r in rows:
        if anchored and int(r['est_seed']) == ANCHOR_SEED and r['arm'] == 'PROP':
            c4 = m4.get(r['record']) or {}
            if c4:
                r['n_gaussians'] = int(c4['n_gaussians'])
                r['apg_dn'] = list(c4['apg_dn'] or [])
    return rows


def synthetic_rows():
    """★`tmp/_audit/check_gate_blindspots.py` 의 사전등록 시점 훅 (결과 0행)."""
    return _fixture(anchored=True)


def _label_rows(wins):
    return _fixture(anchored=False, link_wins=wins)


def selftest():
    fails = []

    def chk(cond, msg):
        if not cond:
            fails.append(msg)

    # ① 문턱이 산술에서 나왔고 전제 P4 와 **대조**된다 (⛔손복사 금지)
    p4 = (premise().get('P4') or {})
    chk(THRESHOLD_PRIMARY == 10, f'THRESHOLD_PRIMARY={THRESHOLD_PRIMARY}')
    chk(HOLM_MIN_WINS == [11, 11, 11, 11, 10, 10], str(HOLM_MIN_WINS))
    chk(abs(ALPHA_PRIMARY - 0.01928710937) < 1e-9, f'α={ALPHA_PRIMARY}')
    if p4:
        chk(p4.get('primary_threshold') == THRESHOLD_PRIMARY, '전제 P4 문턱 불일치')
        chk(p4.get('holm_min_wins') == HOLM_MIN_WINS, '전제 P4 Holm 불일치')
        chk(abs(p4.get('primary_q80', 0) - Q80_PRIMARY) < 1e-4, '전제 P4 q80 불일치')
        chk(abs(p4.get('floor_p', 0) - FLOOR_P) < 1e-9, '전제 P4 floor_p 불일치')
    chk(N_RUNS == 432, f'N_RUNS={N_RUNS}')
    chk(len(expected_jobs()) == N_RUNS, 'expected_jobs 길이')
    chk(len(set(expected_jobs())) == N_RUNS, 'expected_jobs 중복')

    # ② 사다리·고리의 구조 (한 번에 한 인자)
    chk(len(RUNGS) == 9 and len(LINKS) == 6 and len(ARM_ORDER) == 12, '사다리 구조')
    for lid, a, b, _w in LINKS:
        chk(a in ARM_ORDER and b in ARM_ORDER, f'{lid} 팔 이름')
    chk(PRIMARY_LINK == 'L3' and LINKS[2][0] == 'L3', 'PRIMARY 는 L3')
    chk(set(LINK_DIFF_KEYS) == set(LINK_IDS), 'LINK_DIFF_KEYS 가 고리를 다 안 덮는다')
    for lid, a, b, _w in LINKS:
        d = {k for k in set(ARMS[a]) | set(ARMS[b]) if ARMS[a].get(k) != ARMS[b].get(k)}
        chk(d <= LINK_DIFF_KEYS[lid],
            f'{lid}: 팔 선언 차이 {sorted(d)} 가 선언 {sorted(LINK_DIFF_KEYS[lid])} 밖이다')
    chk(LINK_DIFF_KEYS['L1'] == {'real_bank_ensemble_k'}, 'L1 = 앙상블 한 인자')
    chk(LINK_DIFF_KEYS['L2'] == {'bank_dk_frac'}, 'L2 = 기수 한 인자')
    chk(LINK_DIFF_KEYS['L5'] == {'adaptive_n'}, 'L5 = 적응 N 한 인자')
    chk(LINK_DIFF_KEYS['L6'] == {'varpro_steps'}, 'L6 = VARPRO 한 인자')
    chk(not [a for a, v in ARMS.items() if not v], '⛔빈 dict 팔')

    # ③ 라벨 네 갈래가 전부 도달 가능하고 문턱이 정확히 그 자리에 있다
    def lab(wins):
        rows = _label_rows(wins)
        v = verdict(rows, gate_rows=[('G-END x', True, '')])
        return v['chain_label'], v['primary_label'], v

    zero = {k: 0 for k in LINK_IDS}
    chk(lab(dict(zero, L3=12))[0] == 'SKILL-DIES-AT-L3', 'branch AT-L3')
    chk(lab(dict(zero, L1=12))[0] == 'SKILL-DIES-AT-L1', 'branch AT-L1')
    chk(lab(dict(zero, L3=12, L5=12))[0] == 'SKILL-DIES-DISTRIBUTED', 'branch DIST')
    chk(lab(zero)[0] == 'GAP-NOT-FACTORIZABLE', 'branch GAP')
    v = verdict(_label_rows(zero), gate_rows=[('G-END x', False, '')])
    chk(v['chain_label'] == 'LADDER-VOID', 'branch VOID')
    # Holm 문턱: 단일 고리는 11/12 이 필요하다 (10 이면 살아남지 않는다)
    chk(lab(dict(zero, L3=11))[0] == 'SKILL-DIES-AT-L3', 'Holm 11 통과')
    chk(lab(dict(zero, L3=10))[0] == 'GAP-NOT-FACTORIZABLE', 'Holm 10 은 불통과')
    # PRIMARY 는 단일 검정이라 10/12 이면 선다 (양방향 대칭)
    chk(lab(dict(zero, L3=10))[1] == 'PRIOR-ATOMS-LOSE', 'PRIMARY LOSE @10')
    chk(lab(dict(zero, L3=2))[1] == 'PRIOR-ATOMS-EARN', 'PRIMARY EARN @10')
    chk(lab(dict(zero, L3=9))[1] == 'PRIOR-ATOMS-UNDECIDED', 'PRIMARY UNDECIDED @9')
    chk(lab(dict(zero, L3=3))[1] == 'PRIOR-ATOMS-UNDECIDED', 'PRIMARY UNDECIDED @3')

    # ④ 깨끗한 픽스처가 **전 게이트**를 통과한다 (⚠전제·동결 파일이 있어야 한다)
    good = _fixture(anchored=True)
    g = dict((k.split()[0], ok_) for k, ok_, _ in gates(good))
    # ★`G-PATCH-*` 는 **임무 A 의 산출물**(패키지 패치 + 회귀 JSON)에 의존한다.
    #   게이트 **논리**와 **환경 준비**를 섞지 않는다 — 논리는 여기서, 환경은 `--refcheck`
    #   와 `run_m8.assert_bank_patch()` 와 `G-PATCH-*`(차단) 셋이 각각 막는다.
    env_gates = {k: v for k, v in g.items() if k.startswith('G-PATCH')}
    chk(all(v for k, v in g.items() if k not in env_gates),
        '깨끗한 픽스처 FAIL: ' + str([k for k, v in g.items()
                                    if not v and k not in env_gates]))
    clean_cov = dict(COVERAGE)
    zerocov = sorted(k for k, v in clean_cov.items() if v <= 0)
    chk(not zerocov, f'커버리지 0 게이트: {zerocov}')
    chk(g.get('G-COV') is True, 'G-COV 메타게이트가 깨끗한 픽스처에서 PASS 해야 한다')

    # ⑤ ★★**공허성 방지** — 팔 하나를 지우면 **반드시** 누가 FAIL 한다.
    #    그리고 **런 수 게이트 말고도** 누가 FAIL 해야 한다(= 팔-인지 게이트가 있다).
    for arm in ARM_ORDER:
        rows = [r for r in good if r['arm'] != arm]
        gg = gates(rows)
        bad = [gid.split()[0] for gid, ok_, _ in gg if not ok_
               and not gid.startswith('G-PATCH')]
        chk(bad, f'{arm} 삭제가 어느 게이트에도 안 걸린다')
        chk([x for x in bad if x not in ('G-0', 'G-0e', 'G-SEED')],
            f'{arm} 삭제가 런 수/표본 게이트에만 걸린다 (팔-인지 게이트 없음): {bad}')

    # ⑥ ★변형 하네스 — 결함을 하나씩 심으면 **그 게이트가** FAIL 해야 한다
    def mutate(fn, gate_id, base=None):
        rows = [dict(r) for r in (base or good)]
        fn(rows)
        gg = dict((k.split()[0], ok_) for k, ok_, _ in gates(rows))
        chk(gg.get(gate_id) is False, f'변형이 {gate_id} 에 안 걸린다')

    mutate(lambda rs: rs.pop(), 'G-0')
    mutate(lambda rs: rs.append(dict(rs[0])), 'G-0c')
    mutate(lambda rs: rs[0].__setitem__('nowcast', float('nan')), 'G-0d')
    mutate(lambda rs: rs[0].__setitem__('cuda_available', True), 'G-DEV')
    mutate(lambda rs: [r.__setitem__('cuda_available', None) for r in rs], 'G-DEV')
    mutate(lambda rs: rs[0].__setitem__('path_used', 'learned'), 'G-INT')
    mutate(lambda rs: [r.__setitem__('auto_selected', None) for r in rs
                       if r['arm'] == 'BANK'], 'G-INT')
    mutate(lambda rs: [r.__setitem__('real_gate', True) for r in rs
                       if r['arm'] == 'PROP'], 'G-INT')
    mutate(lambda rs: [r.__setitem__('kp_bank_calc', 9.9) for r in rs
                       if r['arm'] == 'BANK'], 'G-KP')
    mutate(lambda rs: [r.__setitem__('n_gaussians', 5) for r in rs
                       if r['arm'] == 'BANK-B30'], 'G-INT-N')
    mutate(lambda rs: [r.__setitem__('bank_B', 471) for r in rs
                       if r['arm'] == 'BANK-B472'], 'G-INT-B')
    mutate(lambda rs: [r.__setitem__('bank_B', 30) for r in rs
                       if r['arm'] == 'BANK-B118'], 'G-INT-B')
    mutate(lambda rs: [r.__setitem__('bank_kset_md5', 'X') for r in rs
                       if r['arm'] == 'MMSE-PROP'], 'G-INT-KSET')
    mutate(lambda rs: [r.update(dict(bank_dk_c=0.01, bank_dk_c_nan=False))
                       for r in rs if r['arm'] == 'MMSE-PROP'], 'G-INT-KSET')
    mutate(lambda rs: [r.update(dict(bank_kset_md5='SAME', atoms_md5='SAME'))
                       for r in rs if r['arm'] == 'BANK-B30'], 'G-INT-KSET2')
    mutate(lambda rs: [r.__setitem__('bank_members', 3) for r in rs
                       if r['arm'] == 'BANK-K1'], 'G-INT-ENS')
    mutate(lambda rs: [r.__setitem__('vp_steps_executed', 400) for r in rs
                       if r['arm'] == 'PROP-LS'], 'G-INT-VP')
    mutate(lambda rs: [r.__setitem__('vp_steps_executed', 12) for r in rs
                       if r['arm'] == 'PROP'], 'G-INT-VP')
    mutate(lambda rs: [r.__setitem__('apg_calls', 1) for r in rs
                       if r['arm'] == 'PROP-LS'], 'G-INT-APG')
    mutate(lambda rs: [r.__setitem__('apg_calls', 0) for r in rs
                       if r['arm'] == 'PROP'], 'G-INT-APG')
    mutate(lambda rs: [r.__setitem__('snap_fired', True) for r in rs
                       if r['arm'] == 'PROP'], 'G-INT-SNAP')
    mutate(lambda rs: [r.__setitem__('snap_fired', False) for r in rs
                       if r['arm'] == 'PROP-SNAP'], 'G-INT-SNAP')
    mutate(lambda rs: [r.__setitem__('atoms_md5', 'X') for r in rs
                       if r['arm'] == 'PROP-LS'], 'G-ATOM')
    mutate(lambda rs: rs[0].__setitem__('rng_hash_after_init', None), 'G-RNG')
    mutate(lambda rs: [r.__setitem__('n_sel', 99) for r in rs
                       if r['arm'] == 'PROP-LS-APG'], 'G-RNG2')
    mutate(lambda rs: rs[0].__setitem__('q_hash', 'X'), 'G-SCENE')
    mutate(lambda rs: [r.__setitem__('obs_hash', 'X') for r in rs], 'G-SCENE-B2')
    mutate(lambda rs: [r.__setitem__('nowcast', 0.123) for r in rs
                       if r['arm'] == 'BANK' and r['est_seed'] == ANCHOR_SEED],
           'G-ANCHOR-B')
    mutate(lambda rs: [r.__setitem__('apg_dn', [0]) for r in rs
                       if r['arm'] == 'PROP' and r['est_seed'] == ANCHOR_SEED],
           'G-ANCHOR-P')
    mutate(lambda rs: [r.__setitem__('nowcast', 0.01) for r in rs
                       if r['arm'] == 'PROP'], 'G-END')
    mutate(lambda rs: rs[0].__setitem__('cap_lambda_l1', 1e-3), 'G-DEFAULT')
    mutate(lambda rs: [r.__setitem__('init_iters_readback', 500) for r in rs
                       if r['arm'] == 'PROP'], 'G-DEFAULT2')
    mutate(lambda rs: rs[0].__setitem__('zero_ref', 1.0), 'G-ZERO')
    mutate(lambda rs: [r.__setitem__('bank_kp', 9.9) for r in rs
                       if r['arm'] == 'BANK'], 'G-KP')
    mutate(lambda rs: [r.__setitem__('cap_bank_dk_frac', 0.2) for r in rs
                       if r['arm'] == 'BANK'], 'G-PATCH-OFF')
    mutate(lambda rs: [r.__setitem__('frac_in_band', None) for r in rs
                       if r['arm'] == 'MMSE-PROP-BAND'], 'G-BAND')
    mutate(lambda rs: rs[0].__setitem__('curve', [0.1]), 'G-LEAD')
    mutate(lambda rs: rs[0].update(dict(est_seed=99)), 'G-0e')

    # ⑦ 자리표시자가 최종 config 에 남지 않는다 + 팔이 실제로 다르다
    try:
        assert_shipping_defaults()
        base_ok = True
    except SystemExit as e:                                       # noqa: BLE001
        base_ok, _ = False, chk(False, f'배포 핀: {e}')
    if base_ok:
        atoms = ([0.1, 0.2], [0.3, 0.4])
        cfgs = {}
        for a in ARM_ORDER:
            cfgs[a] = arm_config(a, depth=17.0, seed=42, atoms=atoms,
                                 atoms_band=atoms, kp_bank=0.21)
            chk(not [k for k, v in cfgs[a].items()
                     if isinstance(v, str) and v.startswith('<')],
                f'{a}: 자리표시자 잔존')
        for lid, a, b, _w in LINKS:
            d = {k for k in set(cfgs[a]) | set(cfgs[b])
                 if repr(cfgs[a].get(k)) != repr(cfgs[b].get(k))}
            chk(d, f'{lid}: 두 계단의 config 가 같다 (처치 없음)')
            chk(d <= LINK_DIFF_KEYS[lid],
                f'{lid}: 실제 config 차이 {sorted(d)} 가 선언 '
                f'{sorted(LINK_DIFF_KEYS[lid])} 밖이다')
        # ⛔자리표시자가 안 채워지면 **예외로 죽는다**(조용한 no-op 금지)
        for a in KSET_ARMS:
            try:
                arm_config(a, depth=17.0, seed=42)
                chk(False, f'{a}: 원자 없이도 config 가 만들어진다')
            except ValueError:
                pass
        for a in SNAP_ARMS:
            try:
                arm_config(a, depth=17.0, seed=42, atoms=atoms, atoms_band=atoms)
                chk(False, f'{a}: kp_bank 없이도 config 가 만들어진다')
            except ValueError:
                pass

    # ⑧ ★공허 참조 방어 — 참조가 비면 **FAIL** 이어야 한다(PASS 아님)
    global FROZEN_B2, FROZEN_M4_GLOB, PREMISE_JSON, REGRESS_JSON, REGRESS_OUT
    sv = (FROZEN_B2, FROZEN_M4_GLOB, PREMISE_JSON, REGRESS_JSON, REGRESS_OUT)
    FROZEN_B2 = os.path.join(HERE, '__none__.jsonl')
    FROZEN_M4_GLOB = os.path.join(HERE, '__none__*.jsonl')
    PREMISE_JSON = os.path.join(HERE, '__none__.json')
    REGRESS_JSON = os.path.join(HERE, '__none__.json')
    REGRESS_OUT = os.path.join(HERE, '__none__.out')
    gg = dict((k.split()[0], ok_) for k, ok_, _ in gates(good))
    FROZEN_B2, FROZEN_M4_GLOB, PREMISE_JSON, REGRESS_JSON, REGRESS_OUT = sv
    for k in ('G-ANCHOR-B', 'G-ANCHOR-P', 'G-SCENE-B2', 'G-INT-B', 'G-ZERO',
              'G-KP', 'G-PATCH-REG'):
        chk(gg.get(k) is False, f'참조가 비면 {k} 는 FAIL 이어야 한다(공허 PASS 금지)')

    # ⑨ 외부 참조가 **실제로** 있고 12 레코드를 덮는다
    chk(len(b2_ship_ref()) == len(RECORDS),
        f'B-2 SHIP 앵커 {len(b2_ship_ref())}/{len(RECORDS)}')
    chk(len(m4_prop_ref()) == len(RECORDS),
        f'M4 {M4_ANCHOR_ARM} 앵커 {len(m4_prop_ref())}/{len(RECORDS)}')
    chk(all(v.get('cuda_available') is False for v in m4_prop_ref().values()),
        '★M4 앵커가 CPU 고정 행이 아니다')
    chk(len(premise_B()) >= len(RECORDS), '전제 P2 기수표 부재')
    chk(len(premise_zero()) == len(RECORDS), '전제 P6 제로필드 부재')
    chk(all(abs(v - 1.0) > 1e-9 for v in premise_zero().values()),
        '⛔전제 P6 에 정확히 1.0 인 레코드가 있다')

    # ⑩ 규약 상수가 문서의 주장을 지탱한다
    chk(len(RECORDS) == 12 and WIN == 3 and ANCHOR_SEED == 42, '표본 규약')
    chk(abs(NOISE - 0.02) < 1e-12, f'NOISE={NOISE}')
    chk(EST_SEEDS == (42, 43, 44), f'EST_SEEDS={EST_SEEDS}')
    chk(sorted(LARGE_BOX + SMALL_BOX) == sorted(RECORDS), '층 분해 표본')
    chk(set(SCOPING) == {'G-BAND', 'G-LEAD', 'G-0b2', 'G-0d2'}, 'SCOPING 선언')

    gates(good)      # ⚠COVERAGE 를 깨끗한 픽스처의 것으로 되돌린다
    print(f'selftest: {"PASS" if not fails else "FAIL"}  ({len(fails)} 실패)')
    for f in fails:
        print('  -', f)
    notready = sorted(k for k, v in env_gates.items() if not v)
    if notready:
        print(f'  ⚠**환경 미비**(게이트 논리 아님): {notready} — 임무 A 의 '
              f'bank_mmse 패치 + 회귀가 아직 없다. 계약 = {PATCH_CONTRACT}')
        print('   ⟹ `run_m8.py` 는 시작 즉시 죽고, `analyze_m8.py` 는 차단 FAIL 로 '
              '판정하지 않는다 (막는 곳이 셋이다).')
    if not fails:
        print(f'  arms={len(ARM_ORDER)} (rungs {len(RUNGS)} + 부차 {len(SECONDARY)}) '
              f'· links={LINK_IDS} · PRIMARY={PRIMARY_LINK} '
              f'· N_RUNS={N_RUNS} · 문턱 {THRESHOLD_PRIMARY}/{N_PRIMARY}')
        print(f'  Holm α={[round(a, 6) for a in HOLM_ALPHAS]} → 최소승수 {HOLM_MIN_WINS} '
              f'· q80={Q80_PRIMARY} · 바닥 p={FLOOR_P:.2e}')
        print('  커버리지(깨끗한 픽스처): '
              + ', '.join(f'{k}={v}' for k, v in sorted(clean_cov.items())))
    return 0 if not fails else 1


def refcheck():
    """★외부 참조를 **실제 파일에 대고** 돌린다 + 게이트가 자료에 대고 만족 가능한가."""
    b2, m4 = b2_ship_ref(), m4_prop_ref()
    pre = premise()
    print(f'FROZEN_B2      = {FROZEN_B2}')
    print(f'  존재={os.path.exists(FROZEN_B2)}  SHIP ref={len(b2)}/{len(RECORDS)}  '
          f'누락={[r for r in RECORDS if r not in b2]}')
    print(f'FROZEN_M4      = {FROZEN_M4_GLOB}')
    print(f'  파일={len(_glob.glob(FROZEN_M4_GLOB))}  {M4_ANCHOR_ARM} ref='
          f'{len(m4)}/{len(RECORDS)}  누락={[r for r in RECORDS if r not in m4]}  '
          f'cuda={sorted({v.get("cuda_available") for v in m4.values()})}')
    print(f'PREMISE        = {PREMISE_JSON}  존재={os.path.exists(PREMISE_JSON)}')
    print(f'  P2 기수={len(premise_B(pre))} 항목 · P2 kp={len(premise_kp(pre))} · '
          f'P6 제로필드={len(premise_zero(pre))} '
          f'(정확히 1.0 인 레코드={[k for k, v in premise_zero(pre).items() if v == 1.0]})')
    try:
        zr = zero_ref_map()
        print(f'  zero_ref.py 재계산 {len(zr)}/{len(RECORDS)} · '
              f'최대 |전제−재계산|='
              f'{max(abs(zr[k] - float(premise_zero(pre)[k])) for k in zr):.2e}')
    except Exception as e:                                        # noqa: BLE001
        print(f'  ⚠zero_ref 재계산 실패: {e!r}')
    sig_ok, sig_msg = _patch_sig_report()
    print(f'PATCH(bank)    = {"OK" if sig_ok else "MISSING"}  {sig_msg[:110]}')
    reg_ok, reg_msg = regress_pass()
    print(f'REGRESS        = {REGRESS_JSON} | {REGRESS_OUT}  '
          f'{"OK" if reg_ok else "MISSING/FAIL"}  {reg_msg[:130]}')
    try:
        assert_shipping_defaults()
        print('DEPLOY 핀      = OK')
        dep_ok = True
    except SystemExit as e:
        print(f'DEPLOY 핀      = ⛔{e}')
        dep_ok = False
    gates(_fixture(anchored=True))
    zero = sorted(k for k, v in COVERAGE.items() if v <= 0)
    print('커버리지: ' + ', '.join(f'{k}={v}' for k, v in sorted(COVERAGE.items())))
    print(f'커버리지 0 게이트 = {zero}')
    okc = (len(b2) == len(RECORDS) and len(m4) == len(RECORDS) and bool(pre)
           and not zero and dep_ok and sig_ok and reg_ok)
    if not okc:
        print('\n⛔refcheck 실패 — 위에서 MISSING 인 것을 먼저 만들어야 한다.')
        if not sig_ok:
            print('   · bank_mmse 패치 (PATCH.md §1·§2, 임무 A)')
        if not reg_ok:
            print(f'   · {PATCH_CONTRACT}')
    return 0 if okc else 1


if __name__ == '__main__':
    if '--refcheck' in sys.argv:
        sys.exit(refcheck())
    sys.exit(selftest())
