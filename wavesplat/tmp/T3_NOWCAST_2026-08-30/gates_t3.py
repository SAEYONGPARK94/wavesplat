"""T3-NOW 단일출처 — ★**실측 나우캐스트의 결손을 두 개의 단일요인 노브로 가른다.**

⛔**이 파일이 판정의 유일한 권위다.** `PREREG.md` 는 이 파일의 산문 번역이고
두 곳이 어긋나면 **코드가 이긴다**. 문턱·팔·표본·라벨·해석 규칙·게이트 전부 여기 있다.
⛔**결과 0행에서 동결한다.**

## 0. 질문 (한 문장)

***"실측 나우캐스트(lead 0)에서 제안 표현이 배포 경로에 지는 것은 — **제안 추첨의 분산**
때문인가, **배포된 마지막 비볼록 정련(VARPRO 400)** 때문인가, 아니면 둘 다 아닌가?"***

## 1. 왜 두 노브인가 — ⛔**전체 대비를 또 재는 배치가 아니다**

`SHIP` vs `PROP` **전체** 대비는 이미 여러 번 쟀다(`B-2`·`B-7`·`M5b`·`M8`·`T2`). 이 배치는
그 간격을 **한 번에 한 인자씩** 움직인다. 두 노브는 **배포된 실측/합성 경로에서 실제로 실행되는
코드**이고 각각 파일:줄이 있다:

| 노브 | 배포 코드에서 읽히는 지점 | 실행 증거 |
|---|---|---|
| **앙상블** `real_bank_ensemble_k` (출하 3) | `sequential_update.py:1134-1139` (`_bank_seeds = [base+off for off in (0,1000,2000,3000,4000)[:k]]`), 장 median = `:1337-1338` `np.median(preds, axis=0)` | `M8` 동결 행 `bank_members=3`·`bank_seeds=(42,1042,2042)` 36/36 |
| **VARPRO** `varpro_steps` (출하 400) | `sequential_update.py:921-925` (`vp_steps>0` 이면 `varpro.apply_varpro`) | `M8` 동결 행 `vp_fired=True`·`vp_steps_executed=400` **36/36** |

★**앙상블 노브는 제안 표현에 한 번도 이식된 적이 없다.** 저장소의 모든 실측 `SHIP` 수치는
**median-of-3 장**이고 모든 실측 `PROP` 수치는 **단일 런**이다 —
`M8_LADDER_2026-08-28/DIFF_INVENTORY.md` §3 `D7` 행 축자: *"**미측정** … 개입 0건 …
이것은 **모든 실측 SHIP<PROP 비교 안에 이미 들어앉은 교락**이다"*.

⚠★**그리고 이 배치는 `T2` 와 논증 구조가 다르다.** `T2` 는 *"배포 경로에서 **측정된** 기술
담지자를 이식"* 했다(재적합 32/36 vs 16/36). 앙상블은 **bank 쪽에서 거의 아무것도 아니다**
(`M8` L1 = 5/12, rel 0.00%). ⟹ 정당화는 *이식* 이 아니라 **기제**다:
앙상블은 **분산 축소 장치**이고 그 값은 추정기의 분산에 비례하는데,
시드 CV 중앙이 **bank 1.04% vs PROP 13.57%**(최대 67.3%)로 **약 13배** 갈린다
(`M8b_LADDER_2026-08-28/RESULT.md §1`, `premise_t3.json:seed_cv` 가 재도출한다).

## 2. ★검정력 — `M8b` 가 남긴 정확한 처방

`M8` 은 같은 질문(나우캐스트 사다리)을 **n=12** 로 물었고 6고리 Holm 첫 문턱이 **11/12** 라
***10/12 는 어떤 자료에서도 기각 불가***였다(`M8b/AMENDMENT_2.md`). 그 사전등록이 결과 전에
적은 다음 설계가 **"2요인 팔이거나 레코드 증설"** 이다.
⟹ 이 배치는 **창 축을 붙여 n=12 → n=36** 으로 간다(`gates_b3.WINDOWS = [1,3,5]`).
n=36·가족 3 Holm 의 첫 문턱은 **25/36**(p=0.014408)이고 **가족 MDE q80 = 0.742** 다.
`M8` 이 잰 효과 크기(9/12=0.75 · 10/12=0.83)는 **이 설계가 판정할 수 있는 범위 안**이다.

## 3. ⛔이 배치가 주장하지 않는 것

`DIRECTION = LOCATE-OR-ELIMINATE`. ⛔*"경계의 원인은 X 다"* 로 넘겨 읽지 않는다.
⛔배포 권고 아님 · ⛔예산 주장 아님 · ⛔사이트/시드 일반화 금지 · ⛔고정 기준상수 채점 금지.
전문 = `READING['NOT_CLAIMED']`.
"""
import ast
import inspect
import json
import math
import os
import statistics as st
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
B2 = os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23')
B3 = os.path.join(ROOT, 'tmp', 'B3_REFIT_2026-08-23')
T2 = os.path.join(ROOT, 'tmp', 'T2_STATIONARITY_2026-08-30')
M8 = os.path.join(ROOT, 'tmp', 'M8_LADDER_2026-08-28')
for _p in (HERE, ROOT, B2, B3, T2, M8):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ★표본·잡음·시드·창은 `gates_b3` 가 단일출처다(⛔손복사 금지 — 규율 9).
from gates_b3 import RECORDS, WINDOWS, SEED, NOISE, N_PAIRS      # noqa: E402

BATCH = 'T3_NOWCAST_2026-08-30'
NAME = 'T3-NOW'
DIRECTION = ('LOCATE-OR-ELIMINATE. 이 배치는 "실측 나우캐스트 결손의 담지자가 **제안 추첨의 '
             '분산**인가 **배포된 VARPRO 정련**인가" 만 묻는다. ⛔어느 분기에서도 '
             '"경계의 원인은 X 다" 로 승격하지 않는다 — 양성이어도 주장은 *"이 노브가 '
             '결손의 일부를 나른다"* 까지다.')

# ─────────────────────────────────────────────────────────────────────────────
# 1. 채점 — ⛔고정 기준상수 금지. 기준은 **레코드·창별 참 제로필드**(lead 0).
# ─────────────────────────────────────────────────────────────────────────────
LEAD = 0.0                # ★나우캐스트 = lead 0 **단독**. `update()` 0회 (규약은 §4).
FIELD_SINGLE = 'e_m0'     # 멤버 0(=출하 단일 런)의 σ정규화 RMSE
FIELD_MED = 'e_med3'      # 하네스 **점별 median 장**의 σ정규화 RMSE

# ⚠`T2 RESULT §5` 의 나우캐스트 표는 `LEAD_NOW=(0.0,1.0)` **평균**이라 이 배치와 다른 양이다
#   (거기 15/36, 여기 17/36). ⛔두 수를 섞지 마라 — `premise_t3.py` 가 둘 다 재도출한다.

_FLAT_NAME = 'ZERO' + '_REF'     # ⛔소스에 리터럴로 두지 않는다(사유는 아래 함수 독스트링)


def _no_flat_ref_constant(dirpath=None):
    """이 배치의 어느 `.py` 도 **고정 기준상수를 코드로** 쓰지 않는가.

    ⛔**문자열 검색으로 짜면 안 된다** — 금지문 자신이 그 토큰을 담으므로 `'<토큰>' in src`
    는 **구조적으로 항상 실패**한다(= 라벨이 영원히 `None`). 2026-08-30 `T2` 자가진단이
    동결 **전에** 이 결함을 잡았다(`T2_STATIONARITY_2026-08-30/AMENDMENT_0_PREFREEZE.md`).
    ⟹ **`ast` 로 「코드에서 그 이름을 쓰는가」** 만 본다: 독스트링·주석의 산문은 통과.
    """
    d = HERE if dirpath is None else dirpath
    hits = []
    for f in sorted(os.listdir(d)):
        if not f.endswith('.py'):
            continue
        try:
            tree = ast.parse(open(os.path.join(d, f), encoding='utf-8').read())
        except SyntaxError as e:
            return False, f'{f}: 파싱 실패 {e}'
        for node in ast.walk(tree):
            if (isinstance(node, ast.Name) and node.id == _FLAT_NAME) or \
                    (isinstance(node, ast.Attribute) and node.attr == _FLAT_NAME):
                hits.append(f'{f}:{getattr(node, "lineno", "?")}')
    return not hits, str(hits)[:90]


def zero_lead0(record, win):
    """참 제로필드(lead 0). ⛔상수가 아니라 **함수** — `zero_ref.py` 가 단일출처다."""
    from zero_ref import zero_ref_perlead
    return zero_ref_perlead(record, win=win, noise=NOISE)[LEAD]


# ─────────────────────────────────────────────────────────────────────────────
# 2. 앙상블 규약 — ★**패키지 소스에서 읽는다**(⛔오프셋 손복사 금지)
# ─────────────────────────────────────────────────────────────────────────────
def package_bank_offsets():
    """`_initialize_bank` 가 쓰는 멤버 오프셋 튜플을 **패키지 소스에서 파싱**한다.

    근거 = `sequential_update.py:1139`
        `self._bank_seeds = [base + off for off in (0, 1000, 2000, 3000, 4000)[:k]]`
    ⛔찾지 못하면 **조용히 기본값으로 넘어가지 않고 죽는다**(그것이 손복사보다 나쁘다).
    ★그리고 이 값의 **구속력 있는 증명은 `G-MIRROR`** 다 — 오프셋이 틀리면 하네스
    median 이 패키지 k=3 장을 비트로 재현하지 못한다.
    """
    from wavesplat.optimizer.sequential_update import SequentialUpdater
    src = inspect.getsource(SequentialUpdater._initialize_bank)
    tree = ast.parse(src.lstrip() if src.startswith(' ') else src)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Attribute) and t.attr == '_bank_seeds'
                for t in node.targets):
            for sub in ast.walk(node.value):
                if isinstance(sub, ast.Tuple) and sub.elts and all(
                        isinstance(e, ast.Constant) and isinstance(e.value, int)
                        for e in sub.elts):
                    found.append(tuple(e.value for e in sub.elts))
    if len(found) != 1:
        raise AssertionError(f'⛔`_bank_seeds` 오프셋 튜플을 유일하게 찾지 못했다: {found}')
    return found[0]


def package_ens_k():
    from wavesplat.optimizer.sequential_update import DEFAULT_REAL_BANK_ENSEMBLE_K
    return int(DEFAULT_REAL_BANK_ENSEMBLE_K)


ALL_OFFSETS = package_bank_offsets()          # (0, 1000, 2000, 3000, 4000)
ENS_K = package_ens_k()                       # 3 = 출하 `real_bank_ensemble_k`
OFFSETS = tuple(ALL_OFFSETS[:ENS_K])          # (0, 1000, 2000) — 두 표현이 **같은 오프셋**

# ─────────────────────────────────────────────────────────────────────────────
# 3. 팔 — ⛔**빈 dict 금지**(`gates_b7.py:46` BLOCKER). 모든 키를 명시한다.
#    `EST` = 추정기 config 에 얹는 것 · `MEM` = 멤버 오프셋 · `MEMKEY` = 오프셋이 들어가는 키
# ─────────────────────────────────────────────────────────────────────────────
_BANK = dict(real_domain_gate=True, init_method='auto', real_estimator='bank_mmse',
             bank_refit_every_s=None, bank_refit_async=False, bank_memory_s=40.0,
             bank_max_obs=600, k_snap_dk=None, adaptive_n=True, varpro_steps=400,
             learned_init_iters=0, lambda_l1=1e-4, tracking_mode='w_phi',
             use_confidence=True, max_n=30, analytic_ensemble_k=3)
_PROP = dict(real_domain_gate=False, init_method='learned', real_estimator='bank_mmse',
             bank_refit_every_s=None, bank_refit_async=False, bank_memory_s=40.0,
             bank_max_obs=600, k_snap_dk=None, adaptive_n=True,
             learned_init_iters=0, lambda_l1=1e-4, tracking_mode='w_phi',
             use_confidence=True, max_n=30, analytic_ensemble_k=3)

ARMS = {
    # ── 배포 실측 경로 (양성대조 + `T2` 앵커) ────────────────────────────────
    'SHIP-K3':     dict(EST=dict(_BANK, real_bank_ensemble_k=ENS_K),
                        MEM=(0,), MEMKEY=None, PATH='bank'),
    # ── ★`G-MIRROR` 전용 + bank 쪽 앙상블 크기 눈금 (E2) ─────────────────────
    'SHIP-K1M3':   dict(EST=dict(_BANK, real_bank_ensemble_k=1),
                        MEM=OFFSETS, MEMKEY='seed', PATH='bank'),
    # ── ★신규 칸: 제안 표현 + 이식된 앙상블 (출하 VARPRO 400) ────────────────
    'PROP-V400M3': dict(EST=dict(_PROP, varpro_steps=400),
                        MEM=OFFSETS, MEMKEY='learned_diffusion_seed', PATH='learned'),
    # ── ★신규 칸: 같은 것에서 VARPRO 만 끈다 (= `M8` L6 를 n=36 에서) ─────────
    'PROP-V0M3':   dict(EST=dict(_PROP, varpro_steps=0),
                        MEM=OFFSETS, MEMKEY='learned_diffusion_seed', PATH='learned'),
}
ARM_ORDER = ['SHIP-K3', 'SHIP-K1M3', 'PROP-V400M3', 'PROP-V0M3']
BANK_ARMS = [a for a in ARM_ORDER if ARMS[a]['PATH'] == 'bank']
PROP_ARMS = [a for a in ARM_ORDER if ARMS[a]['PATH'] == 'learned']
ENS_ARMS = [a for a in ARM_ORDER if len(ARMS[a]['MEM']) > 1]

# ★결과 전에 선언하는 「한 번에 한 인자」 — 자가진단이 `arm_config()` 의 **실제 차이**와
#   대조한다 ⟹ *"단일요인"* 이 산문이 아니라 **검사되는 성질**이 된다(선례 `gates_m8.LINK_DIFF_KEYS`).
CONTRASTS = {
    'ENS-BANK':  ('SHIP-K1M3', 'SHIP-K3', {'real_bank_ensemble_k'}),
    'VARPRO':    ('PROP-V0M3', 'PROP-V400M3', {'varpro_steps'}),
}

# ★`PROP-V*M3` 의 멤버 규약 (하네스가 소유, 패키지 무수정):
#   `cfg['seed'] = SEED` 고정 + `cfg['learned_diffusion_seed'] = SEED + off`.
#   근거 = `learned_init.py:355-357` — `dseed = int(config.get('learned_diffusion_seed',
#   0 if seed is None else seed))` ⟹ **오프셋 0 은 출하 기본값과 항등**이고
#   그것이 `G-ANCHOR-PROP`(= `T2 PROP-FROZEN` lead0 비트 재현)이 증명하는 것이다.
# ★`SHIP-K1M3` 의 멤버 규약: `cfg['seed'] = SEED + off`, `real_bank_ensemble_k=1`
#   ⟹ `_bank_seeds=[SEED+off]` ⟹ 세 멤버의 장을 점별 median 하면 **k=3 과 같아야 한다**.
MEDIAN_RULE = ('np.median(np.stack(member_fields, axis=0), axis=0) — '
               '`sequential_update.py:1337-1338` 의 축자 거울')


def arm_config(arm):
    """팔의 완성된 추정기 config **오버레이**. ⛔러너·게이트·자가진단이 같은 함수를 부른다."""
    return dict(ARMS[arm]['EST'])


def member_overrides(arm, off):
    """멤버별 오버레이. ⛔러너가 여기 말고 다른 데서 시드를 만들면 `G-MEM` 이 잡는다."""
    key = ARMS[arm]['MEMKEY']
    out = dict(seed=SEED)
    if key is not None:
        out[key] = SEED + int(off)
    return out


def expected_members(arm):
    return len(ARMS[arm]['MEM'])


def expected_runs():
    """★총 추정기 런 수를 **프로토콜에서 도출**한다(⛔"360" 을 손으로 적지 않는다)."""
    return sum(expected_members(a) for a in ARM_ORDER) * N_PAIRS


def expected_rows():
    return len(ARM_ORDER) * N_PAIRS


def expected_vp_steps(arm):
    """실행되는 VARPRO 스텝. bank 경로는 `_initialize_learned` 를 안 타므로 **구조적 0**."""
    return 0 if ARMS[arm]['PATH'] == 'bank' else int(arm_config(arm)['varpro_steps'])


def expected_bank_members(arm):
    return int(arm_config(arm)['real_bank_ensemble_k']) if ARMS[arm]['PATH'] == 'bank' else None


# ─────────────────────────────────────────────────────────────────────────────
# 4. 프로토콜 계약 (러너가 지켜야 하는 것 — ★앵커가 이것을 비트로 강제한다)
# ─────────────────────────────────────────────────────────────────────────────
PROTOCOL = """\
① 장면 = `scene_station(record, win=win, noise=NOISE)` (⛔`with_future_obs` 불필요 —
   그 인자는 `P/z/Pq/maps_gt` 를 **비트로 바꾸지 않는다**, `scene_station.py:74` 축자).
② `t_shift = P[:,2].min()` · `t0 = Pq[0,2] - t_shift` ·
   `frames = make_gif_bank_deployed.frames_from_scene(sc, t_shift, dtype=torch.float32)`.
③ 멤버마다: `set_seed(SEED)` → `cfg = dict(DEFAULT_CONFIG)` +
   {depth, device} + `arm_config(arm)` + `member_overrides(arm, off)` →
   `make_updater(cfg)` → `initialize(frames, x_range, y_range, verbose=False)` →
   `tx = torch.tensor(sc['gxq'], dtype=torch.float32)` · `ty = 같은 식으로 gyq` →
   `Z = upd.predict_field(tx, ty, t0).cpu().numpy().reshape(len(gyq), len(gxq))`.
   `x_range=(float(gxq.min()), float(gxq.max()))`, `y_range=(gyq 로 같은 식)`.
   ⛔**여기 세 줄은 `T2 run_t2.py:run()` 의 축자다** — 인자 형(torch float32)을 바꾸면
   앵커가 깨진다.
④ 채점: `mm = isfinite(maps_gt[0])`; `e = sqrt(mean((Z[mm]-g[mm])**2)) / sc['sg']`.
⑤ 멤버 장을 쌓아 **점별 median** → 같은 식으로 `e_med3`.  ⛔`update()` 는 **0회**다
   (lead 0 은 초기화 창 이후 관측이 들어오기 **전** 이다).
⛔②–④ 는 `T2 run_t2.py` 의 lead-0 경로와 **글자 그대로 같다** — 그래서 앵커가 비트로 성립한다.
"""

ROW_CONTRACT = (
    'batch', 'arm', 'record', 'win', 'approved', 'n_members', 'member_offsets',
    'member_seed_used', 'e_members', 'e_m0', 'e_med3', 'field_md5_members',
    'field_md5_med', 'spread_rms', 'zero_lead0', 'ens_k_readback', 'bank_seeds',
    'analytic_ensemble_k_readback', 'vp_fired', 'vp_steps_executed', 'vp_k_shift',
    'n_atoms', 'init_method_readback', 'real_gate', 'real_estimator_readback',
    'seed_readback', 'diffusion_seed_readback', 'deploy_pin_ok', 'anchor_bitmatch',
    'device', 'cuda_available', 'omp_threads', 'sigma', 'npx', 'wall_s',
)

# ─────────────────────────────────────────────────────────────────────────────
# 5. 문턱 — `math.comb` 이 직접 낸다. ⛔손복사 0건.
# ─────────────────────────────────────────────────────────────────────────────
ALPHA = 0.05
N_FAMILY = 3                       # C1(PRIMARY) · C2 · C3 — **확증 주장 셋**, Holm
CLAIMS = ('C1', 'C2', 'C3')


def sign_test_one_sided(w, n):
    return sum(comb(n, k) for k in range(w, n + 1)) / 2 ** n


def _min_w(n, alpha):
    return next(w for w in range(n + 1) if sign_test_one_sided(w, n) <= alpha)


def thresholds(n=N_PAIRS):
    """단일검정 문턱 + **Holm 단계별 최소 승수**. ⛔가족 주장에는 첫 단계가 권위다."""
    nominal = _min_w(n, ALPHA)
    holm = [_min_w(n, ALPHA / (N_FAMILY - r)) for r in range(N_FAMILY)]
    return dict(n=n, NOMINAL=nominal, NOT_EARN=n - nominal,
                HOLM_MIN_WINS=holm, HOLM_FIRST=holm[0],
                p_nominal=sign_test_one_sided(nominal, n),
                p_holm_first=sign_test_one_sided(holm[0], n),
                alpha=ALPHA, n_family=N_FAMILY,
                family_alpha_first=ALPHA / N_FAMILY)


def mde(n=N_PAIRS, power=0.80, thresh=None):
    """★문턱에서 `power` 를 내는 최소 참 승률. ⛔판정문에 **반드시 병기**한다.

    ⚠기본은 **가족(Holm 첫 단계) MDE** 다. 단일검정 MDE 를 붙이는 것은 **과소 보고**다
    (`M8b/AMENDMENT_2.md §5` 가 정확히 그 사유로 개명됐다).
    """
    t = thresholds(n)['HOLM_FIRST'] if thresh is None else int(thresh)
    q = 0.5
    while sum(comb(n, k) * q ** k * (1 - q) ** (n - k) for k in range(t, n + 1)) < power:
        q += 0.001
    return dict(n=n, thresh=t, power=power, q80=round(q, 3), grade='family(Holm-1)')


def mde_single(n=N_PAIRS, power=0.80):
    m = mde(n, power, thresh=thresholds(n)['NOMINAL'])
    m['grade'] = 'single-test(⛔가족 주장에 쓰지 마라)'
    return m


def holm(pvals, alpha=ALPHA):
    """Holm-Bonferroni 생존집합(단측 p 목록 순서 그대로 bool 목록)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: (pvals[i] if pvals[i] is not None else 1.0))
    surv = [False] * m
    for rank, i in enumerate(order):
        p = pvals[i]
        if p is not None and p <= alpha / (m - rank):
            surv[i] = True
        else:
            break
    return surv


# ─────────────────────────────────────────────────────────────────────────────
# 6. 집계 — ⛔**0행이면 카운터는 0 이 아니라 `None`**
# ─────────────────────────────────────────────────────────────────────────────
def key(r):
    return (r.get('record'), r.get('win'))


def _ok(rows):
    return [r for r in rows if 'error' not in r]


def _map(rows, arm, field):
    return {key(r): r[field] for r in rows
            if r.get('arm') == arm and 'error' not in r and r.get(field) is not None}


def _empty(n, **extra):
    out = dict(n=n, better=None, ties=None, p_one_sided=None, rel_median_pct=None,
               median_arm=None, median_ref=None)
    out.update(extra)
    return out


def paired(rows, arm_a, field_a, arm_b, field_b):
    """짝 부호검정 `a < b`. ⛔0행 가드: 겹치는 셀이 없으면 카운터가 **None** 이다."""
    a, b = _map(rows, arm_a, field_a), _map(rows, arm_b, field_b)
    ks = sorted(set(a) & set(b))
    n = len(ks)
    if n == 0:
        return _empty(0, arm=f'{arm_a}.{field_a}', ref=f'{arm_b}.{field_b}')
    w = sum(1 for k in ks if a[k] < b[k])
    return dict(arm=f'{arm_a}.{field_a}', ref=f'{arm_b}.{field_b}', n=n, better=w,
                ties=sum(1 for k in ks if a[k] == b[k]),
                rel_median_pct=st.median([(a[k] - b[k]) / b[k] * 100 for k in ks]),
                median_arm=st.median([a[k] for k in ks]),
                median_ref=st.median([b[k] for k in ks]),
                p_one_sided=sign_test_one_sided(w, n))


def vs_zero(rows, arm, field):
    """★참 제로필드 대비. ⛔고정 상수를 쓰지 않는다 — 행의 `zero_lead0` 가 그 값이고
    `G-Z` 가 그것을 `zero_ref.py` 와 대조한다(⟹ 행을 위조해도 게이트가 잡는다)."""
    ks, rel, w = [], [], 0
    for r in _ok(rows):
        if r.get('arm') != arm:
            continue
        v, z = r.get(field), r.get('zero_lead0')
        if v is None or not z:
            continue
        ks.append(key(r))
        rel.append(v / z)
        w += (v < z)
    n = len(ks)
    if n == 0:
        return _empty(0, arm=f'{arm}.{field}', ref='true-zero-field', ratio_median=None)
    return dict(arm=f'{arm}.{field}', ref='true-zero-field', n=n, better=w,
                ties=None, rel_median_pct=None, median_arm=st.median(
                    [r[field] for r in _ok(rows) if r.get('arm') == arm
                     and r.get(field) is not None]),
                median_ref=None, ratio_median=st.median(rel),
                p_one_sided=sign_test_one_sided(w, n))


# ── ★E1 상호작용 (⚠EXPLORATORY · 가족 밖) ──────────────────────────────────
def interaction(rows):
    """`g = log(e_med3 / e_m0)` 를 팔별로 내고 `g_V400 < g_V0` 인 셀을 센다.

    = *"VARPRO 가 켜져 있을 때 앙상블이 **더** 번다"*. ⛔확증 금지 · MDE 병기 의무.
    """
    gs = {}
    for arm in ('PROP-V400M3', 'PROP-V0M3'):
        m3, m0 = _map(rows, arm, FIELD_MED), _map(rows, arm, FIELD_SINGLE)
        gs[arm] = {k: math.log(m3[k] / m0[k]) for k in set(m3) & set(m0) if m0[k] > 0}
    ks = sorted(set(gs['PROP-V400M3']) & set(gs['PROP-V0M3']))
    if not ks:
        return dict(n=0, v400_gains_more=None, p_one_sided=None, grade='EXPLORATORY',
                    g_v400_median=None, g_v0_median=None, mde=None)
    w = sum(1 for k in ks if gs['PROP-V400M3'][k] < gs['PROP-V0M3'][k])
    return dict(n=len(ks), v400_gains_more=w,
                p_one_sided=sign_test_one_sided(w, len(ks)),
                g_v400_median=st.median([gs['PROP-V400M3'][k] for k in ks]),
                g_v0_median=st.median([gs['PROP-V0M3'][k] for k in ks]),
                grade='EXPLORATORY', mde=mde(len(ks)))


# ── ★E3 분산/공통 분해 (⛔서술 전용 · 검정 없음 · 라벨 없음) ────────────────
def spread_report(rows):
    """멤버 간 점별 표준편차의 RMS(σ정규화) `spread_rms` 를 오차와 비교한다.

    `R = spread_rms / e_m0` — ★결과 전 읽기: **R 이 작으면** 결손은 추첨을 가로질러
    **공통**이고(=앙상블로 걷히지 않는다), **R 이 크면** 추첨 분산이 큰 몫이다.
    ⛔이것은 **크기 귀속이 아니다** — 검정도 라벨도 붙이지 않는다.
    """
    out = {}
    for arm in ENS_ARMS:
        vals = [(r['spread_rms'], r['e_m0']) for r in _ok(rows)
                if r.get('arm') == arm and r.get('spread_rms') is not None
                and r.get('e_m0')]
        out[arm] = (dict(n=len(vals),
                         spread_median=st.median([v[0] for v in vals]),
                         e_m0_median=st.median([v[1] for v in vals]),
                         R_median=st.median([v[0] / v[1] for v in vals]))
                    if vals else dict(n=0, spread_median=None, e_m0_median=None,
                                      R_median=None))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 7. 라벨 — 결과 전 확정. ⛔`UNDECIDED` 는 **「검출 불가」** 이고 MDE 병기가 의무다.
# ─────────────────────────────────────────────────────────────────────────────
def _grade(v, surv):
    """(등급, 문턱표) — CONFIRMATORY 는 **Holm 생존**일 때만."""
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


def _mdetag(v):
    return f"MDE(가족 q80={mde(v['n'] or N_PAIRS)['q80']})"


def label_c1(v, surv):
    """C1 PRIMARY — `PROP-V400M3.e_med3 < PROP-V400M3.e_m0` (앙상블 이식)."""
    g, t = _grade(v, surv)
    w, n = v['better'], v['n']
    if g == 'NOROWS':
        return None, '⛔행 0개 — 판정 없음.'
    if g in ('CONFIRMATORY', 'EXPLORATORY'):
        return (f'{NAME} :: PROP-ENSEMBLE-EARNS ({g})',
                f'{w}/{n} (p={v["p_one_sided"]:.5f}) — ★실측 나우캐스트 결손의 일부가 '
                f'**제안 추첨의 분산**이고, 배포 경로가 이미 가진 장치(median-of-{ENS_K})로 '
                f'걷힌다. ⟹ 저장소의 모든 실측 `SHIP` vs `PROP` 수치에 앉아 있던 '
                f'**앙상블 비대칭 교락이 크기를 갖는다**. ⛔배포 권고 아님 · ⛔예산 주장 아님.'
                + ('' if g == 'CONFIRMATORY' else ' ⚠**Holm 미생존 = 확증 아님**, ' + _mdetag(v)))
    if g == 'REVERSE':
        return (f'{NAME} :: PROP-ENSEMBLE-NULL-OR-HURTS',
                f'{w}/{n} ⟹ ★결손은 추첨 분산이 아니라 **추첨을 가로질러 공통**이다. '
                f'분산 축이 닫히고, 앙상블 비대칭은 **측정으로** 교락 목록에서 내려간다.')
    return (f'{NAME} :: PROP-ENSEMBLE-UNDECIDED',
            f'부호로 갈리지 않는다({w}/{n}, p={v["p_one_sided"]:.4f}). ⛔승격 금지 · '
            f'소거표에 **「검출 불가」** 로 적는다(⛔*"차이 없다"* 금지) · ' + _mdetag(v))


def label_c2(v, surv):
    """C2 — `PROP-V0M3.e_m0 < PROP-V400M3.e_m0` (배포된 VARPRO 400 이 해가 되는가)."""
    g, t = _grade(v, surv)
    w, n = v['better'], v['n']
    if g == 'NOROWS':
        return None, '⛔행 0개 — 판정 없음.'
    if g in ('CONFIRMATORY', 'EXPLORATORY'):
        return (f'{NAME} :: VARPRO-HURTS-REAL-NOWCAST ({g})',
                f'{w}/{n} (p={v["p_one_sided"]:.5f}) — 출하 기본값 `varpro_steps=400` 의 '
                f'마지막 비볼록 정련이 **실측 나우캐스트를 해친다**. `M8` L6 가 n=12 에서 '
                f'**판정할 수 없던** 칸이 n=36 에서 갈렸다. ⛔배포 변경 권고 아님 — 실측 '
                f'게이트는 여전히 `bank_mmse` 로 보내고 이 노브는 **합성 경로**의 기본값이다.'
                + ('' if g == 'CONFIRMATORY' else ' ⚠**Holm 미생존 = 확증 아님**, ' + _mdetag(v)))
    if g == 'REVERSE':
        return (f'{NAME} :: VARPRO-HELPS-OR-NULL',
                f'{w}/{n} ⟹ ⚠**역방향 꼬리**(사전등록 방향의 반대). `M8` L6 의 방향이 '
                f'이 환경에서 재현되지 않는다 ⟹ ⛔**역방향을 확증으로 읽지 마라** — '
                f'가족 α 는 사전등록 방향에만 쓰였다(EXPLORATORY).')
    return (f'{NAME} :: VARPRO-UNDECIDED',
            f'{w}/{n} (p={v["p_one_sided"]:.4f}). ⛔승격 금지 · 「검출 불가」 · ' + _mdetag(v))


def label_c3(v, surv):
    """C3 — `PROP-V400M3.e_med3` vs **참 제로필드** (끝점: 기술이 생기는가)."""
    g, t = _grade(v, surv)
    w, n = v['better'], v['n']
    if g == 'NOROWS':
        return None, '⛔행 0개 — 판정 없음.'
    if g in ('CONFIRMATORY', 'EXPLORATORY'):
        return (f'{NAME} :: PROP-EARNS-REAL-NOWCAST ({g})',
                f'{w}/{n} (p={v["p_one_sided"]:.5f}, 비 {v["ratio_median"]:.4f}) — ★제안 '
                f'표현이 앙상블을 얻으면 실측 나우캐스트에서 **참 제로필드를 이긴다**. '
                f'⛔이것은 정확도 주장이지 배포 주장이 아니다(단가 {ENS_K}배).'
                + ('' if g == 'CONFIRMATORY' else ' ⚠**Holm 미생존 = 확증 아님**, ' + _mdetag(v)))
    if g == 'REVERSE':
        return (f'{NAME} :: PROP-NO-NOWCAST-SKILL-EVEN-WITH-ENSEMBLE',
                f'{w}/{n} (비 {v["ratio_median"]:.4f}) ⟹ 배포 경로의 앙상블 장치를 그대로 '
                f'이식해도 제안 표현은 실측 나우캐스트에서 **무기술**이다.')
    return (f'{NAME} :: PROP-NOWCAST-UNDECIDED',
            f'{w}/{n} (p={v["p_one_sided"]:.4f}). ⛔승격 금지 · 「검출 불가」 · ' + _mdetag(v))


# ★결과 전 고정된 **읽기 규칙**. ⛔*"널이면 무엇을 쓰나"* 에 답이 없으면 설계가 아니다.
READING = {
    'C1+': '★소거표에 **양성 행**: *"실측 나우캐스트 결손의 일부는 제안 추첨의 분산이고, '
           '배포 경로가 이미 가진 median-of-k 로 걷힌다."* ⟹ 저장소의 **모든** 실측 '
           'SHIP vs PROP 계수에 앙상블 비대칭 공시가 붙는다(크기 포함). ⛔배포 권고 아님.',
    'C1-': '★소거표에 **배제 행**: *"결손은 추첨을 가로질러 공통이다."* 분산 축이 닫히고 '
           '앙상블 비대칭이 교락 목록에서 **측정으로** 내려간다 ⟹ 이후 배치는 그것을 '
           '공시 한 줄로 처리한다. 남는 후보는 **적합 통계**(D5·D6)와 **지지집합**이다.',
    'C2+': '★*"배포된 마지막 정련(VARPRO 400)이 실측 나우캐스트를 해친다."* `M8` L6 가 '
           'n=12 에서 판정 불가였던 칸이 갈린다. ⛔배포 변경 권고 아님(별도 사전등록) · '
           '⛔합성 경로 판정(W4·W5)을 뒤집는다고 읽지 마라 — 층이 다르다.',
    'C2-': '★*"VARPRO 는 실측 나우캐스트 결손의 담지자가 아니다."* `M8` L6 의 9/12 는 '
           '검정력이 커진 표본에서 재현되지 않았다 ⟹ 소거표 한 줄, 그리고 `M8` 사다리의 '
           '마지막 고리가 **닫힌다**.',
    'C1+ & C2+': '⚠두 요인이 **함께** 나오면 크기 합산 금지 — 상호작용(E1)은 EXPLORATORY 이고 '
                 '로그비 가법성은 이 배치가 검정하지 않는다(`M8b §2.1` 의 교훈: 중앙값 비는 '
                 '가법이 아니다).',
    'C1- & C2-': '★가장 강한 소거: **두 노브 모두 아니다.** 남는 후보는 `DIFF_INVENTORY §4` 의 '
                 '1위 축(**적합 통계** = 명시 관측잡음 · beamform 사전 · held-out val 게이팅 · '
                 'SBL-EM)과 지지집합이다 ⟹ 조건부 후속 `T3-FIT` 의 트리거가 참이 된다.',
    'C3+': '★실측에서 제안 표현의 **첫 양성 끝점**. ⛔그래도 배포 권고 아님(단가·게이트).',
    'C3-': '기존 서술 유지: *"제안 표현은 실측 12 레코드에서 기술이 없다"* — 이번에는 '
           '**배포 경로의 앙상블을 준 뒤에도** 그렇다.',
    'UNDECIDED': '⛔승격 금지. **가족 MDE 를 병기**하고 소거표에 **「검출 불가」** 로 적는다 '
                 '(⛔*"차이 없다"* 금지 — `M8b` 개명 사건). ⛔단일검정 MDE 로 바꿔 적지 마라.',
    'NOT_CLAIMED': (
        '⛔①배포 권고 금지(실측 게이트는 `bank_mmse` 로 보낸다) · '
        '⛔②예산·실시간 주장 금지(멤버 3개 = 단가 3배, 비동기 경로는 bank 전용) · '
        '⛔③크기를 `B-1`·`M8`·`T2`·`M5b` 계수와 **직접 비교 금지**(다른 팔·다른 기저·다른 지표) · '
        '⛔④사이트(4개)·레코드(12개)·시드(기준 42) 일반화 금지 · '
        '⛔⑤고정 기준상수 채점 금지(`G-Z`·`G-FLAT`) · '
        '⛔⑥E1(상호작용)·E2(bank 앙상블)·E3(분산/공통) 를 **확증으로 승격 금지** · '
        '⛔⑦`premise` 의 장치 민감도 관찰을 *"GPU 가 나쁘다"* 로 읽지 마라 — 측정된 것은 '
        '**불변성의 부재**이지 우열이 아니다 · '
        '⛔⑧`M8` 의 사다리 계수를 이 배치의 팔에 전이 금지(장치 스탬프가 다르다) · '
        '⛔⑨*"경계의 원인은 X 다"* 금지(DIRECTION).'),
}

# ─────────────────────────────────────────────────────────────────────────────
# 8. 배포 기본값 트립와이어 (⛔손복사 상수가 아니라 **트립와이어**)
# ─────────────────────────────────────────────────────────────────────────────
DEPLOY_PIN = {
    'bank_refit_every_s': None, 'bank_refit_async': False,
    'real_estimator': 'bank_mmse', 'real_bank_ensemble_k': 3,
    'init_method': 'auto', 'varpro_steps': 400, 'learned_init_iters': 0,
    'auto_switch_thresh': 0, 'lambda_l1': 1e-4, 'tracking_mode': 'w_phi',
    'adaptive_n': True, 'use_confidence': True, 'k_snap_dk': None, 'device': 'cpu',
    'analytic_ensemble_k': 3, 'max_n': 30,
}


def assert_shipping_defaults():
    from wavesplat.run import DEFAULT_CONFIG
    bad = {k: (DEFAULT_CONFIG.get(k), v) for k, v in DEPLOY_PIN.items()
           if DEFAULT_CONFIG.get(k) != v}
    if bad:
        raise AssertionError(f'⛔출하 기본값이 이 사전등록 이후 움직였다 (shipped, pinned): '
                             f'{bad}\n   ⟹ 돌리기 전에 AMENDMENT 를 쓰라.')
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 9. 앵커 (동결 배치와의 **비트** 재현) — ⛔`M8` 은 앵커가 **아니다**(§공시)
# ─────────────────────────────────────────────────────────────────────────────
T2_RESULTS = os.path.join(T2, 'results_t2.jsonl')
ANCHORS = {                       # 이 배치의 팔 → (동결 파일, 동결 팔, 동결 필드, 이 배치 필드)
    'SHIP-K3':     (T2_RESULTS, 'SHIP-FROZEN', 'lead0', 'e_m0'),
    'PROP-V400M3': (T2_RESULTS, 'PROP-FROZEN', 'lead0', 'e_m0'),
}
# ⛔`SHIP-K1M3`·`PROP-V0M3` 에는 앵커가 **없다** — 전자는 `G-MIRROR` 라는 **더 강한**
#   내부 항등식이 대신하고, 후자가 이 배치의 신규 칸이다.
PREMISE = os.path.join(HERE, 'premise_t3.json')


def anchor_index():
    """동결 `T2` 행에서 (팔, record, win) → 값. ⛔파일이 없으면 **빈 dict 이 아니라 예외**."""
    if not os.path.exists(T2_RESULTS):
        raise FileNotFoundError(f'⛔앵커 자료 부재: {T2_RESULTS}')
    idx = {}
    for line in open(T2_RESULTS):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if 'error' in r:
            continue
        idx[(r.get('arm'), r.get('record'), r.get('win'))] = r
    return idx


# ─────────────────────────────────────────────────────────────────────────────
# 10. 게이트 (⛔0행 가드 · ⛔필터 항진 금지 · ★개입 계수는 **가지** 단위)
# ─────────────────────────────────────────────────────────────────────────────
def gates(rows, premise=None, check_flat_dir=None):
    out = []
    ok = _ok(rows)
    A = out.append

    # ── 커버리지 (⛔0행이면 여기서 전부 실패한다 = 공허 PASS 불가)
    A(('G-0  행 수', len(ok) == expected_rows(), f'{len(ok)}/{expected_rows()}'))
    A(('G-0b 에러 0', len(rows) == len(ok) and len(rows) > 0, f'{len(rows) - len(ok)}'))
    percell = {a: len(_map(ok, a, 'e_m0')) for a in ARM_ORDER}
    A(('G-0c 0행 가드(팔마다 36셀)', len(ok) > 0 and all(
        percell[a] == N_PAIRS for a in ARM_ORDER), str(percell)))
    A(('G-0d 셀 전건', len({key(r) for r in ok}) == N_PAIRS and
       {key(r) for r in ok} == {(rec, w) for rec in RECORDS for w in WINDOWS},
       f'{len({key(r) for r in ok})}/{N_PAIRS}'))
    runs = sum(int(r.get('n_members') or 0) for r in ok)
    A(('G-0e 추정기 런 수 == 프로토콜 도출값', runs == expected_runs(),
       f'{runs}/{expected_runs()}'))
    miss = sorted({k for r in ok for k in ROW_CONTRACT if k not in r})
    A(('G-0f 행 계약 전건 충족', bool(ok) and not miss, str(miss)[:90]))

    # ── ★개입 계수 = **가지** 단위 (⛔"설정을 넣었다 != 설정이 먹었다")
    for arm in ARM_ORDER:
        R = [r for r in ok if r.get('arm') == arm]
        want_m = expected_members(arm)
        A((f'G-1 {arm} 멤버 수 == {want_m}',
           len(R) == N_PAIRS and all(
               r.get('n_members') == want_m and len(r.get('e_members') or []) == want_m
               and tuple(r.get('member_offsets') or ()) == tuple(ARMS[arm]['MEM'])
               for r in R),
           f'{len(R)} rows, offsets='
           f'{sorted({tuple(r.get("member_offsets") or ()) for r in R})}'))
        want_vp = expected_vp_steps(arm)
        A((f'G-2 {arm} vp_steps_executed == {want_vp} (멤버 전건)',
           bool(R) and all(list(r.get('vp_steps_executed') or []) == [want_vp] * want_m
                           for r in R),
           str(sorted({tuple(r.get('vp_steps_executed') or ()) for r in R}))[:60]))
        A((f'G-2b {arm} vp_fired == {bool(want_vp)}',
           bool(R) and all(all(bool(x) == bool(want_vp)
                               for x in (r.get('vp_fired') or [])) for r in R),
           str(sorted({tuple(bool(x) for x in (r.get('vp_fired') or ()))
                       for r in R}))[:60]))
        want_bm = expected_bank_members(arm)
        A((f'G-3 {arm} bank_members == {want_bm}',
           bool(R) and all(list(r.get('ens_k_readback') or [None] * want_m)
                           == [want_bm] * want_m for r in R),
           str(sorted({tuple(r.get('ens_k_readback') or ()) for r in R},
                      key=str))[:60]))

    # ── ★경로 되읽기 — ⛔팔 정의가 아니라 **추정기 객체**에서 읽는다(S13b 교훈)
    bank = [r for r in ok if r.get('arm') in BANK_ARMS]
    prop = [r for r in ok if r.get('arm') in PROP_ARMS]
    A(('G-4a bank 팔이 bank 경로', bool(bank) and all(
        all(x == 0 for x in (r.get('n_atoms') or []))
        and all(x is True for x in (r.get('real_gate') or []))
        and all(x == 'bank_mmse' for x in (r.get('init_method_readback') or []))
        and all(x == 1 for x in (r.get('analytic_ensemble_k_readback') or []))
        for r in bank),
       f'n={len(bank)} ' + str(sorted({
           (tuple(r.get('n_atoms') or ()), tuple(r.get('real_gate') or ()))
           for r in bank}, key=str))[:70]))
    A(('G-4b prop 팔이 learned 경로', bool(prop) and all(
        all(x > 0 for x in (r.get('n_atoms') or []))
        and all(x is False for x in (r.get('real_gate') or []))
        and all(x == 'learned' for x in (r.get('init_method_readback') or []))
        for r in prop),
       f'n={len(prop)} ' + str(sorted({
           tuple(r.get('init_method_readback') or ()) for r in prop}))[:70]))

    # ── ★멤버 시드 규약 되읽기 (⛔러너가 딴 데서 시드를 만들면 여기서 갈린다)
    for arm in ARM_ORDER:
        R = [r for r in ok if r.get('arm') == arm]
        want = [member_overrides(arm, o) for o in ARMS[arm]['MEM']]
        keyname = ARMS[arm]['MEMKEY']
        wseed = [w['seed'] for w in want]
        wdiff = [w.get('learned_diffusion_seed') for w in want]
        A((f'G-5 {arm} 멤버 시드 규약 ({keyname})',
           bool(R) and all(list(r.get('seed_readback') or []) == wseed
                           and list(r.get('diffusion_seed_readback')
                                    or [None] * len(wseed)) == wdiff for r in R),
           f'want seed={wseed} dseed={wdiff}'))
    A(('G-5b bank 팔 `_bank_seeds` 가 오프셋 규약과 일치', bool(bank) and all(
        list(r.get('bank_seeds') or []) == [
            [SEED + o + ALL_OFFSETS[i] for i in range(expected_bank_members(r['arm']))]
            for o in ARMS[r['arm']]['MEM']]
        for r in bank), ''))

    # ── ★★G-MIRROR: 하네스 median 이 패키지 median-of-k 를 **비트로** 재현하는가
    a = {key(r): (r.get('field_md5_med'), r.get('e_med3'))
         for r in ok if r.get('arm') == 'SHIP-K1M3'}
    b = {key(r): (r.get('field_md5_members') or [None])[0]
         for r in ok if r.get('arm') == 'SHIP-K3'}
    b2 = {key(r): r.get('e_m0') for r in ok if r.get('arm') == 'SHIP-K3'}
    ks = sorted(set(a) & set(b))
    A(('G-MIRROR 하네스 median == 패키지 k=3 (해시+값 비트)',
       bool(ks) and len(ks) == N_PAIRS and all(
           a[k][0] is not None and a[k][0] == b[k] and a[k][1] == b2[k] for k in ks),
       f'{sum(1 for k in ks if a[k][0] == b[k] and a[k][1] == b2[k])}/{len(ks)}'))
    # ⛔공허한 거울 방지. ★2026-08-30 개정 = `AMENDMENT_0_PREFREEZE.md`(결과 0행 시점).
    #   원래 요구 「ENS 팔 **전 셀**에서 멤버 해시가 서로 다르다」는 **구조적으로 통과 불가**
    #   였다: bank 멤버는 `fit_bank_mmse` 의 **val-split 시드**만 다르고, 그 split 이 고르는
    #   하이퍼파라미터(`pexp`·`smult`)가 같은 격자점에 떨어지면 적합이 **비트 동일**해진다.
    #   실측(36셀 탐침, 채점 0회 · 판정자료 아님): 3-distinct **16** · 2-distinct **11** ·
    #   1-distinct **9** ⟹ 36/36 은 어떤 자료에서도 불가능. ★같은 탐침에서 `G-MIRROR`
    #   자신은 **36/36 성립**했다(= 고칠 것은 거울이 아니라 비공허 조건이다).
    #   ⟹ **기제가 다른 두 팔 부류를 나눈다**:
    #     · `PROP-*` = 멤버가 `learned_diffusion_seed` 로 갈린다(확산 재표집) ⟹ **전 셀 전건**
    #       (원래 강도 무수정. `FORGERIES['G-MIRROR-b']` 가 이 가지를 친다).
    #     · `SHIP-K1M3` = 위 사유로 **적어도 한 셀**에서 전건 distinct 면 median 연산자가
    #       실제로 시험된다(평균·오프셋 위조는 거기서 죽는다).
    #   ★degenerate 셀 수는 detail 에 **적어 보고**한다 — ⛔지우지 않는다. 그 자체가 측정이고
    #   `premise_t3.json:m8_l1_ensemble` 의 `rel_median_pct = 0.00%` 를 설명한다.
    def _nd(r):
        return len(set(r.get('field_md5_members') or []))

    degen = {a: '%d/%d' % (sum(1 for r in ok if r.get('arm') == a
                               and _nd(r) == r.get('n_members')),
                           sum(1 for r in ok if r.get('arm') == a))
             for a in ENS_ARMS}
    A(('G-MIRROR-b 거울이 공허하지 않다(PROP 전건 · bank ≥1셀)',
       bool(ok)
       and all(_nd(r) == r.get('n_members')
               for r in ok if r.get('arm') in ENS_ARMS and r.get('arm') in PROP_ARMS)
       and all(any(_nd(r) == r.get('n_members') for r in ok if r.get('arm') == a)
               for a in ENS_ARMS if a in BANK_ARMS),
       f'전-distinct 셀 {degen}'))

    # ── ★G-ANCHOR: 동결 `T2` 행과의 **비트** 재현
    for arm in ANCHORS:
        v = [r.get('anchor_bitmatch') for r in ok if r.get('arm') == arm]
        A((f'G-ANCHOR {arm} 비트 {N_PAIRS}/{N_PAIRS}',
           len(v) == N_PAIRS and all(x is True for x in v),
           f'{sum(1 for x in v if x is True)}/{len(v)}'))
    A(('G-ANCHOR-N 앵커 없는 팔은 None 을 싣는다(공허 PASS 방지)', bool(ok) and all(
        r.get('anchor_bitmatch') is None
        for r in ok if r.get('arm') not in ANCHORS), ''))

    # ── ★단일요인 — `arm_config()` 의 **실제 차이**가 선언과 같은가 (런 무관)
    for cid, (a_arm, b_arm, want) in CONTRASTS.items():
        ca, cb = arm_config(a_arm), arm_config(b_arm)
        diff = {k for k in set(ca) | set(cb) if ca.get(k) != cb.get(k)}
        A((f'G-6 {cid} 단일요인 == {sorted(want)}', diff == want, str(sorted(diff))))
    A(('G-6b 개입이 실제로 장을 움직였다(항진 아님)', all(
        sum(1 for k in set(_map(ok, x, 'e_m0')) & set(_map(ok, y, 'e_m0'))
            if _map(ok, x, 'e_m0')[k] != _map(ok, y, 'e_m0')[k]) > 0
        for x, y, _ in CONTRASTS.values()),
       str({cid: sum(1 for k in set(_map(ok, x, 'e_m0')) & set(_map(ok, y, 'e_m0'))
                     if _map(ok, x, 'e_m0')[k] != _map(ok, y, 'e_m0')[k])
            for cid, (x, y, _) in CONTRASTS.items()})))

    # ── 채점 기준 (⛔고정 상수 금지) — 행의 값을 `zero_ref.py` 와 **대조**한다
    bad_z = []
    for r in ok:
        try:
            if r.get('zero_lead0') != zero_lead0(r['record'], r['win']):
                bad_z.append(key(r))
        except Exception as e:                                       # noqa: BLE001
            bad_z.append((key(r), repr(e)[:30]))
    A(('G-Z 행의 참 제로필드 == zero_ref.py (전건)',
       bool(ok) and not bad_z, f'{len(bad_z)} 불일치'))
    A((f'G-FLAT 고정 기준상수 부재({_FLAT_NAME})',
       *_no_flat_ref_constant(check_flat_dir)))

    # ── 장치·스레드 **단일 스탬프** (⚠제안 경로는 장치에 민감하다 — premise §P2)
    stamp = sorted({(r.get('device'), r.get('cuda_available'), str(r.get('omp_threads')))
                    for r in ok})
    A(('G-7 장치/스레드 단일 스탬프', len(stamp) == 1, str(stamp)[:90]))
    A(('G-7b 출하 기본값 핀', bool(ok) and all(r.get('deploy_pin_ok') is True for r in ok), ''))
    A(('G-7c 승인 스탬프 전건', bool(ok) and all(r.get('approved') is True for r in ok), ''))

    # ── ★양성대조: 동결 자료가 낸 수를 이 배치가 재현하는가 (⛔손복사 아님)
    if premise is None and os.path.exists(PREMISE):
        premise = json.load(open(PREMISE))
    if premise:
        pc = [('SHIP-K3', 'e_m0', 'ship_lead0_vs_zero'),
              ('PROP-V400M3', 'e_m0', 'prop_lead0_vs_zero')]
        for arm, fld, kk in pc:
            got, wnt = vs_zero(ok, arm, fld)['better'], premise[kk]['better']
            A((f'G-PC {arm} vs 제로 == 동결 {wnt}', got == wnt, f'{got}'))
        got = paired(ok, 'SHIP-K3', 'e_m0', 'PROP-V400M3', 'e_m0')['better']
        A((f'G-PC 짝 SHIP<PROP == 동결 {premise["ship_lt_prop_lead0"]["better"]}',
           got == premise['ship_lt_prop_lead0']['better'], f'{got}'))
    else:
        A(('G-PC premise_t3.json 존재', False, '⛔전제 파일이 없다'))

    # ── 필터 비항진 (⛔세는 집합이 팔을 구조적으로 배제하면 항상-참이다)
    seen = {r.get('arm') for r in ok}
    A(('G-8 필터 비항진', seen == set(ARM_ORDER)
       and len(BANK_ARMS) > 0 and len(PROP_ARMS) > 0 and len(ENS_ARMS) > 0
       and len(BANK_ARMS) + len(PROP_ARMS) == len(ARMS), str(sorted(seen))))
    A(('G-9 비용 스탬프 전건', bool(ok) and all(
        isinstance(r.get('wall_s'), (int, float)) for r in ok), ''))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 11. 판정
# ─────────────────────────────────────────────────────────────────────────────
def analyze(rows, premise=None):
    ok = _ok(rows)
    c1 = paired(ok, 'PROP-V400M3', FIELD_MED, 'PROP-V400M3', FIELD_SINGLE)
    c2 = paired(ok, 'PROP-V0M3', FIELD_SINGLE, 'PROP-V400M3', FIELD_SINGLE)
    c3 = vs_zero(ok, 'PROP-V400M3', FIELD_MED)
    surv = holm([c1['p_one_sided'], c2['p_one_sided'], c3['p_one_sided']])
    g = gates(ok, premise)
    blocked = [n for n, p, _ in g if not p]
    l1, w1 = label_c1(c1, surv[0])
    l2, w2 = label_c2(c2, surv[1])
    l3, w3 = label_c3(c3, surv[2])
    return dict(
        batch=BATCH, name=NAME,
        label=None if blocked else l1, why=w1,                # PRIMARY = C1
        label_c2=None if blocked else l2, why_c2=w2,
        label_c3=None if blocked else l3, why_c3=w3,
        blocked_by=blocked,                # ⛔게이트 실패 시 **라벨 없이 닫는다**(M-8 선례)
        direction=DIRECTION, reading=READING, protocol=PROTOCOL,
        primary=c1, c2=c2, c3=c3, holm_survivors=dict(zip(CLAIMS, surv)),
        thresholds=thresholds(), mde=mde(), mde_single=mde_single(),
        exploratory=dict(interaction=interaction(ok),
                         bank_ensemble=paired(ok, 'SHIP-K3', 'e_m0',
                                              'SHIP-K1M3', 'e_m0'),
                         spread=spread_report(ok)),
        descriptive=dict(
            vs_zero={a: vs_zero(ok, a, f) for a, f in
                     (('SHIP-K3', 'e_m0'), ('SHIP-K1M3', 'e_med3'),
                      ('PROP-V400M3', 'e_m0'), ('PROP-V400M3', 'e_med3'),
                      ('PROP-V0M3', 'e_m0'), ('PROP-V0M3', 'e_med3'))},
            wall_s_median=(st.median([r['wall_s'] for r in ok]) if ok else None)),
        gates=[{'name': n, 'pass': bool(p), 'detail': d} for n, p, d in g],
        n_rows=len(rows))


# ─────────────────────────────────────────────────────────────────────────────
# 12. 픽스처 — ★`check_gate_blindspots` 가 **결과 0행에서** 이 훅을 쓴다
# ─────────────────────────────────────────────────────────────────────────────
def synthetic_rows(seed=7):
    """전 게이트를 통과하는 **깨끗한 합성 144행**. ⛔판정 자료가 아니다.

    ⚠**결정적이어야 한다** — `hash()` 는 프로세스마다 달라지므로 쓰지 않는다
    (⛔`PYTHONHASHSEED` 에 흔들리는 픽스처는 검사기를 프로세스별로 다르게 만든다).
    """
    import hashlib
    import random
    rng = random.Random(seed)
    anc = anchor_index()
    rows = []
    for ri, rec in enumerate(RECORDS):
        for wi, win in enumerate(WINDOWS):
            ci = ri * len(WINDOWS) + wi           # ⛔결정적 (⛔`hash()` 금지)
            z = zero_lead0(rec, win)
            ship = anc[('SHIP-FROZEN', rec, win)]['lead0']
            prop = anc[('PROP-FROZEN', rec, win)]['lead0']
            # ⚠`e_med3` 는 **median 장의 오차**이지 오차의 median 이 아니다 ⟹ 따로 세운다.
            emed = {'SHIP-K1M3': ship,                       # 거울 성립(= SHIP-K3)
                    'PROP-V400M3': prop * (1 - 0.05 * ((ci % 3) - 1)),
                    'PROP-V0M3': prop * 0.88}
            emem = {'SHIP-K3': [ship],
                    'SHIP-K1M3': [ship * 1.03, ship * 0.97, ship * 1.01],
                    'PROP-V400M3': [prop, prop * (1 + rng.uniform(-.3, .3)),
                                    prop * (1 + rng.uniform(-.3, .3))],
                    'PROP-V0M3': [prop * 0.90, prop * 0.95, prop * 0.85]}
            for arm in ARM_ORDER:
                nm = expected_members(arm)
                e = emem[arm]
                md5s = [hashlib.md5(f'{arm}|{rec}|{win}|{i}'.encode()).hexdigest()
                        for i in range(nm)]
                if arm == 'SHIP-K1M3':
                    md5s[0] = hashlib.md5(
                        f'SHIP-K3|{rec}|{win}|0'.encode()).hexdigest()
                med_md5 = (hashlib.md5(f'SHIP-K3|{rec}|{win}|0'.encode()).hexdigest()
                           if arm == 'SHIP-K1M3'
                           else hashlib.md5(f'{arm}|{rec}|{win}|med'.encode()).hexdigest())
                med = (emed[arm] if nm > 1 else None)
                rows.append(dict(
                    batch=BATCH, arm=arm, record=rec, win=win, approved=True,
                    n_members=nm, member_offsets=list(ARMS[arm]['MEM']),
                    member_seed_used=[member_overrides(arm, o)
                                      for o in ARMS[arm]['MEM']],
                    e_members=e, e_m0=e[0], e_med3=med,
                    field_md5_members=md5s,
                    field_md5_med=(med_md5 if nm > 1 else None),
                    spread_rms=(0.1 * e[0] if nm > 1 else None),
                    zero_lead0=z,
                    ens_k_readback=[expected_bank_members(arm)] * nm,
                    bank_seeds=([[SEED + o + ALL_OFFSETS[i]
                                  for i in range(expected_bank_members(arm))]
                                 for o in ARMS[arm]['MEM']]
                                if ARMS[arm]['PATH'] == 'bank' else [None] * nm),
                    analytic_ensemble_k_readback=(
                        [1] * nm if ARMS[arm]['PATH'] == 'bank' else [None] * nm),
                    vp_fired=[bool(expected_vp_steps(arm))] * nm,
                    vp_steps_executed=[expected_vp_steps(arm)] * nm,
                    vp_k_shift=[0.02 if expected_vp_steps(arm) else None] * nm,
                    n_atoms=([0] * nm if ARMS[arm]['PATH'] == 'bank' else [27] * nm),
                    init_method_readback=(['bank_mmse'] * nm
                                          if ARMS[arm]['PATH'] == 'bank'
                                          else ['learned'] * nm),
                    real_gate=([True] * nm if ARMS[arm]['PATH'] == 'bank'
                               else [False] * nm),
                    real_estimator_readback=['bank_mmse'] * nm,
                    seed_readback=[member_overrides(arm, o)['seed']
                                   for o in ARMS[arm]['MEM']],
                    diffusion_seed_readback=[
                        member_overrides(arm, o).get('learned_diffusion_seed')
                        for o in ARMS[arm]['MEM']],
                    deploy_pin_ok=True,
                    anchor_bitmatch=(True if arm in ANCHORS else None),
                    device='cpu', cuda_available=True, omp_threads='1',
                    sigma=0.3, npx=396, wall_s=1.0 + nm))
    return rows


def synthetic_premise(rows=None):
    """픽스처와 정합하는 전제(양성대조 `G-PC` 가 공허하지 않게)."""
    rows = synthetic_rows() if rows is None else rows
    return dict(
        ship_lead0_vs_zero=vs_zero(rows, 'SHIP-K3', 'e_m0'),
        prop_lead0_vs_zero=vs_zero(rows, 'PROP-V400M3', 'e_m0'),
        ship_lt_prop_lead0=paired(rows, 'SHIP-K3', 'e_m0', 'PROP-V400M3', 'e_m0'))


# ─────────────────────────────────────────────────────────────────────────────
# 13. 자가진단 — ★**게이트마다 「위조 입력을 넣으면 반드시 실패하는가」 항진 검사**
# ─────────────────────────────────────────────────────────────────────────────
def _names(g):
    return [n for n, _, _ in g]


def _failed(g):
    return {n for n, p, _ in g if not p}


def _mut(rows, fn):
    import copy
    r2 = copy.deepcopy(rows)
    fn(r2)
    return r2


# (게이트 이름 접두, 위조 함수) — ⛔**한 줄이라도 빠지면 그 게이트는 검사된 적이 없다**
FORGERIES = [
    ('G-0 ', lambda R: R.pop()),
    ('G-0b', lambda R: R.__setitem__(0, dict(R[0], error='boom'))),
    ('G-0c', lambda R: [R.__setitem__(i, dict(R[i], e_m0=None))
                        for i, r in enumerate(R) if r['arm'] == 'PROP-V0M3'][:1]),
    ('G-0d', lambda R: [r.__setitem__('record', 'GHOST') for r in R[:4]]),
    ('G-0e', lambda R: R[0].__setitem__('n_members', 99)),
    ('G-0f', lambda R: R[0].pop('sigma')),
    ('G-1 ', lambda R: R[1].__setitem__('member_offsets', [0, 1, 2])),
    ('G-2 ', lambda R: [r.__setitem__('vp_steps_executed', [0, 0, 0])
                        for r in R if r['arm'] == 'PROP-V400M3'][:1]),
    ('G-2b', lambda R: [r.__setitem__('vp_fired', [False, False, False])
                        for r in R if r['arm'] == 'PROP-V400M3'][:1]),
    ('G-3 ', lambda R: [r.__setitem__('ens_k_readback', [3, 3, 3])
                        for r in R if r['arm'] == 'SHIP-K1M3'][:1]),
    ('G-4a', lambda R: [r.__setitem__('real_gate', [False])
                        for r in R if r['arm'] == 'SHIP-K3'][:1]),
    ('G-4b', lambda R: [r.__setitem__('n_atoms', [0, 0, 0])
                        for r in R if r['arm'] == 'PROP-V400M3'][:1]),
    ('G-5 ', lambda R: [r.__setitem__('diffusion_seed_readback', [42, 43, 44])
                        for r in R if r['arm'] == 'PROP-V400M3'][:1]),
    ('G-5b', lambda R: [r.__setitem__('bank_seeds', [[0], [0], [0]])
                        for r in R if r['arm'] == 'SHIP-K1M3'][:1]),
    # ★거울 위조 = *"median 대신 평균"* / *"오프셋 (0,1,2)"* 이 만드는 바로 그 상태
    ('G-MIRROR ', lambda R: [r.__setitem__('field_md5_med', 'deadbeef')
                             for r in R if r['arm'] == 'SHIP-K1M3'][:1]),
    ('G-MIRROR-b', lambda R: [r.__setitem__('field_md5_members', ['x', 'x', 'x'])
                              for r in R if r['arm'] == 'PROP-V0M3'][:1]),
    ('G-ANCHOR SHIP-K3', lambda R: [r.__setitem__('anchor_bitmatch', False)
                                    for r in R if r['arm'] == 'SHIP-K3'][:1]),
    ('G-ANCHOR PROP-V400M3', lambda R: [r.__setitem__('anchor_bitmatch', False)
                                        for r in R if r['arm'] == 'PROP-V400M3'][:1]),
    ('G-ANCHOR-N', lambda R: [r.__setitem__('anchor_bitmatch', True)
                              for r in R if r['arm'] == 'PROP-V0M3'][:1]),
    ('G-6b', lambda R: [r.__setitem__('e_m0', _map(R, 'PROP-V400M3', 'e_m0')[key(r)])
                        for r in R if r['arm'] == 'PROP-V0M3']),
    ('G-Z', lambda R: R[0].__setitem__('zero_lead0', 1.0)),
    ('G-7 ', lambda R: R[0].__setitem__('cuda_available', False)),
    ('G-7b', lambda R: R[0].__setitem__('deploy_pin_ok', False)),
    ('G-7c', lambda R: R[0].__setitem__('approved', False)),
    ('G-PC SHIP-K3', lambda R: [r.__setitem__('e_m0', r['zero_lead0'] * 9)
                                for r in R if r['arm'] == 'SHIP-K3']),
    ('G-PC PROP-V400M3', lambda R: [r.__setitem__('e_m0', r['zero_lead0'] * 0.01)
                                    for r in R if r['arm'] == 'PROP-V400M3']),
    ('G-PC 짝', lambda R: [r.__setitem__('e_m0', 99.0)
                          for r in R if r['arm'] == 'SHIP-K3']),
    ('G-8', lambda R: [r.__setitem__('arm', 'GHOST-ARM') for r in R
                       if r['arm'] == 'PROP-V0M3']),
    ('G-9', lambda R: R[0].__setitem__('wall_s', None)),
]

# ⛔**항진 검사에서 면제되는 게이트와 그 사유** — 면제는 *"검사 안 함"* 이 아니라
#   *"행 위조로는 못 흔들리므로 **다른 단계가** 검사한다"* 는 뜻이다. ⛔목록을 늘리려면
#   그 자리에 대체 검사를 적어야 한다.
FORGERY_EXEMPT = {
    'G-6 ': '정적 게이트(행 무관, `arm_config()` 비교) — 자가진단 ③ 이 같은 성질을 '
            '독립으로 검사한다(선언 == 실제 차이).',
    'G-FLAT': '소스 파일을 본다(행 무관) — 자가진단 ⑧ 이 **위조 파일**을 넣어 항진을 친다.',
}


def selftest(verbose=True):
    fails = []
    t = thresholds()

    # ① 문턱·MDE 산술 (⛔손복사 0건 — `math.comb` 이 직접 낸다)
    assert t['n'] == N_PAIRS == 36 and len(RECORDS) == 12 and WINDOWS == [1, 3, 5]
    assert t['NOMINAL'] == _min_w(36, 0.05) and t['NOT_EARN'] == 36 - t['NOMINAL']
    assert t['HOLM_MIN_WINS'] == [_min_w(36, 0.05 / 3), _min_w(36, 0.05 / 2),
                                  _min_w(36, 0.05)], t
    assert mde()['q80'] >= mde_single()['q80'], (mde(), mde_single())
    assert holm([0.001, 0.02, 0.9]) == [True, True, False]
    assert holm([0.02, 0.02, 0.02]) == [False, False, False]   # 0.02 > 0.05/3

    # ② 팔 정의 — ⛔빈 dict 금지 · 오프셋은 **패키지에서** 왔다
    assert set(ARM_ORDER) == set(ARMS) and len(ARMS) == 4
    for a in ARM_ORDER:
        assert arm_config(a), f'{a}: 빈 dict 팔'
        assert 'real_domain_gate' in arm_config(a) and 'k_snap_dk' in arm_config(a)
    assert OFFSETS == tuple(package_bank_offsets()[:package_ens_k()]) == (0, 1000, 2000)
    assert expected_runs() == 360 and expected_rows() == 144, expected_runs()
    assert expected_vp_steps('SHIP-K3') == 0 and expected_vp_steps('PROP-V400M3') == 400
    assert set(READING) >= {'C1+', 'C1-', 'C2+', 'C2-', 'C3+', 'C3-',
                            'UNDECIDED', 'NOT_CLAIMED'}

    # ③ 단일요인 선언 == 실제 차이
    for cid, (x, y, want) in CONTRASTS.items():
        cx, cy = arm_config(x), arm_config(y)
        got = {k for k in set(cx) | set(cy) if cx.get(k) != cy.get(k)}
        assert got == want, f'{cid}: {got} != {want}'

    # ④ 깨끗한 픽스처가 **전 게이트** 통과 + ⛔커버리지 0 게이트 0개
    rows = synthetic_rows()
    prem = synthetic_premise(rows)
    g0 = gates(rows, prem)
    bad = _failed(g0)
    if bad:
        fails.append(f'깨끗한 픽스처가 게이트를 통과하지 못한다: {sorted(bad)}')
    if len(g0) != len(set(_names(g0))):
        fails.append('게이트 이름이 중복이다')

    # ⑤ ★★항진 검사 — 위조를 넣으면 **그 게이트가** 실패해야 한다
    for prefix, fn in FORGERIES:
        try:
            gm = gates(_mut(rows, fn), prem)
        except Exception as e:                                       # noqa: BLE001
            fails.append(f'위조 [{prefix}] 가 예외를 냈다: {e!r}')
            continue
        hit = {n for n in _failed(gm) if n.startswith(prefix)}
        if not hit:
            fails.append(f'★항진: 위조 [{prefix}] 를 넣었는데 그 게이트가 통과했다 '
                         f'(실패한 것들={sorted(_failed(gm))[:3]})')
    uncovered = [n for n in _names(g0)
                 if not any(n.startswith(p) for p, _ in FORGERIES)
                 and not any(n.startswith(p) for p in FORGERY_EXEMPT)]
    if uncovered:
        fails.append(f'항진 검사가 덮지 않는 게이트: {uncovered}')
    stale = [p for p in FORGERY_EXEMPT if not any(n.startswith(p) for n in _names(g0))]
    if stale:
        fails.append(f'면제 목록에 죽은 항목: {stale}')

    # ⑥ ★삭제 민감도 — 팔을 하나 지우면 반드시 누가 FAIL (`M4 G-13` 부류 방어)
    for arm in ARM_ORDER:
        gm = gates([r for r in rows if r['arm'] != arm], prem)
        if not _failed(gm):
            fails.append(f'★삭제 민감도 없음: {arm} 을 통째로 지웠는데 전 게이트 통과')

    # ⑦ ⛔0행 가드 — 카운터가 **0 이 아니라 None**, 라벨은 닫힌다
    z = analyze([], prem)
    if z['label'] is not None or z['primary']['better'] is not None \
            or z['c3']['better'] is not None:
        fails.append(f'0행 가드 실패: {z["label"]} / {z["primary"]["better"]}')
    if not z['blocked_by']:
        fails.append('0행인데 blocked_by 가 비었다')

    # ⑧ ⛔고정 기준상수 — 항진 검사(위조 파일을 넣으면 반드시 FAIL)
    ok0, _ = _no_flat_ref_constant()
    probe = os.path.join(HERE, '_flat_probe.py')
    try:
        open(probe, 'w').write(_FLAT_NAME + ' = 1.0\nx = ' + _FLAT_NAME + '\n')
        ok1, det = _no_flat_ref_constant()
    finally:
        os.path.exists(probe) and os.remove(probe)
    if not (ok0 and not ok1):
        fails.append(f'G-FLAT 가 항진한다: 현상태={ok0} 위조탐지={not ok1} {det}')

    # ⑨ 앵커 자료가 36셀 전건 있는가 (⛔없으면 `G-ANCHOR` 가 구조적으로 실패한다)
    anc = anchor_index()
    for arm in ('SHIP-FROZEN', 'PROP-FROZEN'):
        n = sum(1 for (a, rec, w) in anc if a == arm)
        if n != N_PAIRS:
            fails.append(f'앵커 {arm}: {n}/{N_PAIRS} 셀')

    # ⑩ 배포 기본값 트립와이어
    try:
        assert_shipping_defaults()
    except Exception as e:                                           # noqa: BLE001
        fails.append(f'출하 기본값 핀: {e}')

    # ⑪ 라벨 함수가 세 갈래를 **실제로** 낸다(도달 불가 라벨 금지 — `check_thresholds` 부류)
    for lab, hi, lo in ((label_c1, t['HOLM_FIRST'], t['NOT_EARN']),
                        (label_c2, t['HOLM_FIRST'], t['NOT_EARN']),
                        (label_c3, t['HOLM_FIRST'], t['NOT_EARN'])):
        hiv = dict(n=36, better=hi, p_one_sided=sign_test_one_sided(hi, 36),
                   ratio_median=0.8)
        lov = dict(n=36, better=lo, p_one_sided=sign_test_one_sided(lo, 36),
                   ratio_median=1.2)
        mid = dict(n=36, better=18, p_one_sided=sign_test_one_sided(18, 36),
                   ratio_median=1.0)
        names = {lab(hiv, True)[0], lab(lov, False)[0], lab(mid, False)[0]}
        if len(names) != 3:
            fails.append(f'{lab.__name__}: 세 갈래가 구분되지 않는다 {names}')

    for f in fails:
        print('  FAIL', f)
    if verbose:
        print(f'gates_t3 selftest: {"PASS" if not fails else "FAIL"}  '
              f'n={t["n"]} NOMINAL={t["NOMINAL"]} HOLM={t["HOLM_MIN_WINS"]} '
              f'NOT_EARN={t["NOT_EARN"]} q80(family)={mde()["q80"]} '
              f'q80(single)={mde_single()["q80"]} runs={expected_runs()} '
              f'rows={expected_rows()} offsets={OFFSETS} gates={len(g0)}')
    return not fails


if __name__ == '__main__':
    sys.exit(0 if selftest() else 1)
