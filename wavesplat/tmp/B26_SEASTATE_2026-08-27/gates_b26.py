"""B-26 단일출처 — **해상상태 교락을 깬다** (2026-08-27). ⛔결과 0행에서 동결.

★이 파일이 표본·팔·문턱·게이트·라벨·**결과 전 해석 규칙**의 **유일한 정의**다.
러너(`run_b26.py`)·생성기(`gen_b26.py`)·분석기(`analyze_b26.py`)는 전부 여기서만 읽는다.

## 왜 이 배치인가
논문의 서사는 ***"합성에서 벌고 실측에서 못 번다"*** = **sim-to-real 경계**다.
그런데 두 코퍼스는 해상상태가 다르다:
  · 합성 헤드라인(`B-1`·`B-12`·`B-16`·`B-22`) = **BF4(1.2, 7.0) / BF5(2.0, 9.0)**
  · 실측 12 레코드         = **Hs 중앙 0.725 m · Tp 중앙 3.49 s** (§전제, 이 배치가 재계산)
⟹ ★***"sim-to-real" 과 "BF4·BF5 대 실측 해상상태" 가 완전히 교락돼 있다.***
   실측 음성의 원인이 **자료 영역(sim-to-real)**인지 **해상상태 레짐**인지 저장소는 **한 번도
   가른 적이 없다.** 이 배치가 그 칸을 채운다: **참값이 있는 합성 위에서 해상상태만 옮긴다.**

## ⛔이 배치가 **주장하지 않는 것** (결과 전에 적는다)
- ⛔**비선형성 축을 다시 답하지 않는다.** GT 는 **선형 중첩**으로 고정이다(§생성기).
  `B-11`/`B-12` 가 HOS M=3 에서 답했고 여기서는 그 축을 **붙박아 둔다**.
- ⛔**희소성 축이 아니다.** 성분 수는 **혼잡 {15, 30} 으로 고정**이다.
- ⛔**실측을 재현하지 않는다.** 관측 모형·기하·센서는 전부 합성 그대로다.
  바뀌는 것은 **바다의 (Hs, Tp) 두 수**뿐이다.
- ⛔**`B-1`·`B-12` 의 크기와 직접 비교하지 않는다**(다른 코퍼스·다른 시드·다른 GT 클래스).
  비교는 **이 배치 안의 네 칸끼리만** 한다.

## ★설계 — 2×2 요인 (BF 사다리가 아니다)
BF2→BF5 사다리는 **Hs 와 Tp 를 함께 올린다** ⟹ 그것으로는 무엇이 매개하는지 못 가른다.
⟹ **높이 축과 주기 축을 직교시킨다.** 네 모서리:

| 코너 | Hs [m] | Tp [s] | λ_p [m] | Hs/λ_p | 뜻 |
|---|---|---|---|---|---|
| `SS-BF4`   | 1.200 | 7.00 | 76.5 | 0.0157 | **헤드라인의 해상상태**(= `wavesim` BF4, 학습 앵커) |
| `SS-LOWH`  | 0.725 | 7.00 | 76.5 | 0.0095 | 실측 **파고**, 합성 **주기** |
| `SS-SHORT` | 1.200 | 3.49 | 19.0 | 0.0631 | 합성 **파고**, 실측 **주기** |
| `SS-REAL`  | 0.725 | 3.49 | 19.0 | 0.0381 | ★**실측 해상상태**(Hs·Tp·가파름 동시 정합) |

★`SS-BF4` 는 낭비 칸이 아니다 — **추정 쪽 내부 양성대조**다(§G-16).
★`SS-REAL` 의 k·Hs = 0.2395 이고 **실측 12 레코드의 k·Hs 중앙은 0.2445** 다 ⟹ 가파름도 맞는다.

## ★전제 3건 — 전부 이 배치가 **실측**했다 (`premise_b26.py`, 산출 `premise_b26.json`)
1. ⛔**BF2/BF3 혼잡 에피소드는 저장소에 0개다.** 전수: BF2 는 1comp 만(62개), BF3 는 1/2/5comp
   만(183개). `ncomp ≥ 15` 인 BF2·BF3 는 **저장소 어디에도 없다** ⟹ **생성이 선행한다.**
2. ⛔★**`B-12` 의 HOS 생성기로는 BF2 를 만들 수 없다.** `gen_hos.snap_components` 는
   `ix = round(kx/dkx) % Nx` 로 스펙트럼 빈에 스냅하는데, HOS 격자(192×128 / 600×400 m)의
   축별 나이퀴스트는 **1.0053 rad/m** 이다. 시드 12개 실측(`premise_b26.json:P3`):
   BF2 는 15/30comp 에서 **4–6 / 9–12 개**가 `% Nx` 로 되감기고(에너지 최대 **10.0%**)
   `mesh_adequacy` M=3 판정이 **INADEQUATE 21/24** 다 ⟹ **조용한 에일리어싱**.
   ★그리고 더 강하게 — **`SS-REAL` 자체가 6–15 개 초과(에너지 최대 16.5%)이고
   M=3 에서 23/24 INADEQUATE** 다 ⟹ ***HOS 경로는 실측 해상상태를 아예 못 담는다.***
   ⟹ ★**임무 지시의 축자 설계(`gen_hos.py:48` 의 bf 튜플 확장)는 BF2 에서 실행 불가다.**
   ⟹ **선형 생성기로 간다**(§생성기). 그 대가는 §"주장하지 않는 것" 첫 줄에 적었다.
3. ⚠★**실측 Tp 는 배포 사전의 학습 분포 밖이다.**
   `tmp/mixed_retrain_2026-07-26/gen_mixed.py:78-83` 축자:
   `ANCHORS = ((0.3,4.0),(0.6,5.5),(1.2,7.0),(2.0,9.0))` (= `wavesim.pipeline.SEA_STATE`),
   `ANCHOR_FRAC=0.5`, `TP_BOX=(3.8, 11.5)`, `ANCHOR_JITTER=0.15`.
   ⟹ 학습 장면의 **Tp 최소값이 3.400 s**(= BF2 앵커 × 0.85)이고
   **P(Tp ≤ 3.49) = 0.933%** 다. 그리고 (|ΔTp|<5% ∧ |ΔHs|<10%) 근방 밀도는
   `SS-BF4` **3.16%** 대 `SS-REAL`·`SS-SHORT` **0.0000%**.
   ⟹ ★**주기 축은 곧 "사전의 학습 지지집합 안 대 그 극단 가장자리" 축이다.**
   ⚠이것이 §해석 규칙의 두 번째 행이 성립하는 근거다.

## 생성기 — `tmp/S8_deploy_confirm_2026-08-10/gen_s8.py` 의 호출을 **그대로** 쓴다
`data_s8_crowded_2026-08-10` + `data_s9_crowded_2026-08-11` 이 곧 **`B-1` 헤드라인의 혼잡 64
에피소드**이고 그것을 만든 것이 `gen_s8.py` 다. 같은 호출을 쓴다:
  `run_episode(mesh 200×167, duration 30.0, dt 0.1, depth 50.0, n_pts 50,
               raydrop 'uniform', GPU raycast, ra=rf=ds=True, cfg.hos.enabled=False)`
⛔**몽키패치 0개**(`gen_hos` 의 스냅 패치는 HOS 짝을 위한 것이고 여기서는 **해악**이다 — 전제 2).
⛔**`wavesim/**` 무수정** — 새 해상상태는 **이미 노출된 배포 노브** `cfg.sea_states` 에 넣는다
(`wavesim/pipeline.py:113` 이 `cfg.sea_states.get(...)` 로 읽는다. BF6/BF7 이 그 경로로 존재한다).
★**증명은 `G-16b` PC-1**: 동결 `data_s8_crowded_2026-08-10` 에피소드 하나를 이 배치의 생성
함수로 재생성해 **`gt_wave_field`·`lidar_points`·`lidar_valid_mask` 비트 동일**을 요구한다.

## 표본 — **결과 0행에서, 런 0회로 확정**
`mesh_adequacy.check()` 는 `jonswap_components()` 의 순함수다(**시뮬레이션 불필요**)
⟹ ★**생성 전에 96 에피소드 목록과 코너별 정확한 n 을 확정할 수 있다.**
규칙(동결): 각 (코너, ncomp) 에 대해 시드 풀 1641,1642,… 를 **순서대로** 걸으며
`mesh_verdict != 'INADEQUATE'` 이고 `check_q_constraint` 통과인 **앞의 12개**를 취한다.
실측 결과: 거부 **1건뿐**(`SS-REAL` 15comp seed 1644, var_under 0.291) ⟹ 소모 시드 **1641–1653**.
**총 96 에피소드 = 4 코너 × 2 ncomp × 12 시드, 코너당 n = 24.**
⚠이 선택은 **관측 이전·GT 무관·결과 무관**이고, 짝 비교는 팔 양쪽에 동일하게 적용된다.
⚠방향: 선택은 짧은 주기 코너에서 **가장 어려운 에피소드를 뺀다** ⟹ 거기서 음성이 나오면
그 음성은 **과소평가**된 것이다(내 결론에 불리한 쪽으로 보수적).

## 팔 — `B-12` 를 **그대로 import** (손복사 0)
`SHIP-NOW` = `run_b12.run_ship`(배포 `DEFAULT_CONFIG`, 추정기 시드 median-of-3 42·43·44)
`SPEC25`   = `run_b12.run_spec` → `run_s16.run_letkf`(jonswap · loc 25 · ens 600, 필터 시드 3개)
`ZERO`     = 기준선 행의 `zero_nRMSE`
⛔**빈 dict 로 정의된 팔 0개.** ★기준선은 **각 코너의 참 (Hs, Tp) 를 배경으로 받는다**
(`run_s16.py:212-213` 이 `loader.Hs_true`/`Tp_true` 를 넘긴다) ⟹ **비교자는 코너마다 옳은
사전을 받는다.** `G-14` 가 이것을 행에서 확인한다 ⟹ *"기준선이 새 해상상태에서 불리했다"* 반박 차단.
"""
import glob
import os
import re
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
B12 = os.path.join(ROOT, 'tmp', 'B12_HOS_EXTEND_2026-08-23')
B11 = os.path.join(ROOT, 'tmp', 'B11_HOS_BASELINE_2026-08-23')
S16 = os.path.join(ROOT, 'tmp', 'S16_BASELINE_2026-08-14')
S8 = os.path.join(ROOT, 'tmp', 'S8_deploy_confirm_2026-08-10')
for _p in (HERE, B12, B11, S16, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── 팔·집계 함수는 B-12/B-11 에서 import (중복 구현 금지) ──────────────────────
from gates_b11 import (EST_SEEDS, FILTER_SEEDS, SPEC,          # noqa: F401,E402
                       sign_test_one_sided, med_by_episode, paired)

DATA = os.path.join(HERE, 'data_b26')
SMOKE = os.path.join(HERE, 'smoke_b26')

# ── 생성 상수: `gen_s8.py` 와 같은 값. ★손복사가 아니라 원본에서 되읽는다 ──────
MESH_NX, MESH_NY = 200, 167          # gen_s8.py:40  NX, NY = 200, 167
SRC_EPISODE = os.path.join(ROOT, 'data_finemesh_2026-07-23',
                           'episode_BF4_15comp_seed0_uniform.h5')
RAYDROP = 'uniform'
SUFFIX = 'uniform_ra_rf_ds'
# ★PC-1 (생성기 비트동일 양성대조) 대상 — `B-1` 헤드라인 혼잡 코퍼스의 한 에피소드
PC1_PATH = os.path.join(ROOT, 'data_s8_crowded_2026-08-10',
                        'episode_BF4_15comp_seed32_uniform_ra_rf_ds.h5')
PC1 = dict(sea_state='BF4', n_components=15, seed=32)
PC1_DATASETS = ('gt_wave_field', 'lidar_points', 'lidar_valid_mask')


def gen_params():
    """duration·dt·depth·n_pts 를 **동결 원본 에피소드에서 읽는다**(gen_s8.py:98-103 규약).
    ⛔손으로 적지 않는다. 네 코너가 **같은 값**을 받는 것이 이 배치의 전제다."""
    import h5py
    with h5py.File(SRC_EPISODE, 'r') as f:
        depth = float(f['config'].attrs['depth'])
        n_pts = int(f['config/lidar_config'].attrs.get('n_pts', 0)) or 50
        ts = f['frames/timestamps'][:]
    return dict(duration=float(ts[-1] - ts[0]), dt=float(ts[1] - ts[0]),
                depth=depth, n_pts=n_pts, n_frames=len(ts))


# ── 2×2 요인: 코너 정의 (⛔빈 dict 금지 — 두 수를 명시한다) ────────────────────
HS_HI, HS_LO = 1.2, 0.725            # 1.2 = wavesim BF4;  0.725 = 실측 12 레코드 Hs 중앙
TP_HI, TP_LO = 7.0, 3.49             # 7.0 = wavesim BF4;  3.49 = 실측 12 레코드 Tp 중앙
CORNERS = {
    'SS-BF4':   dict(Hs=HS_HI, Tp=TP_HI, hs='hi', tp='hi'),
    'SS-LOWH':  dict(Hs=HS_LO, Tp=TP_HI, hs='lo', tp='hi'),
    'SS-SHORT': dict(Hs=HS_HI, Tp=TP_LO, hs='hi', tp='lo'),
    'SS-REAL':  dict(Hs=HS_LO, Tp=TP_LO, hs='lo', tp='lo'),
}
CORNER_ORDER = ('SS-BF4', 'SS-LOWH', 'SS-SHORT', 'SS-REAL')
ANCHOR_CORNER = 'SS-BF4'             # 내부 기준 칸 (G-16)
PRIMARY_CORNER = 'SS-REAL'           # PRIMARY
NCOMPS = (15, 30)                    # ⛔혼잡 고정 — 희소성 축이 아니다
SEED_POOL = tuple(range(1641, 1681))  # 시드 원장 [1592–1818] 중 미사용 구간
K_SEEDS = 12
N_EP = len(CORNERS) * len(NCOMPS) * K_SEEDS          # 96
N_PER_CORNER = len(NCOMPS) * K_SEEDS                 # 24

# 실측 12 레코드 재계산값 (premise_b26.py 가 낸다; 여기 두 수는 코너 정의에 쓰인 값)
REAL_HS_MEDIAN, REAL_TP_MEDIAN = 0.725, 3.49
REAL_N_RECORDS = 12

# ── 문턱: ★코드가 계산한다 (n=24, 단측 부호검정) ──────────────────────────────
N_FAMILY = 4                          # 코너 4개 = 주장 가족
ALPHA = 0.05
FAMILY_ALPHA = ALPHA / N_FAMILY       # 0.0125
NOMINAL, CONFIRM, WEAK = 17, 18, 7    # /24  (0.03196 · 0.01133 ≤ 0.0125 · n-NOMINAL)
NEG_MIN = 17                          # /24  SPEC25 < ZERO 최소 (음성 대조)
MESH_BAD = 'INADEQUATE'


def threshold_table(n=N_PER_CORNER):
    """문턱을 **재도출**해서 사전등록 숫자와 대조할 수 있게 한다(G-19)."""
    return {'n': n,
            'p_nominal': sign_test_one_sided(NOMINAL, n),
            'p_confirm': sign_test_one_sided(CONFIRM, n),
            'p_weak_le': sum(comb(n, k) for k in range(0, WEAK + 1)) / 2 ** n,
            'family_alpha': FAMILY_ALPHA,
            'confirm_within_family': sign_test_one_sided(CONFIRM, n) <= FAMILY_ALPHA,
            'weak_is_sibling': WEAK == n - NOMINAL}


# ── 표본: 런 0회로 재도출 가능 (G-1 이 동결 목록과 대조) ───────────────────────
def components(corner, ncomp, seed):
    from wavesim.wave_generator import jonswap_components
    c = CORNERS[corner]
    return jonswap_components(Hs=c['Hs'], Tp=c['Tp'], n_components=ncomp,
                              depth=gen_params()['depth'], seed=seed,
                              random_amplitudes=True, random_frequencies=True,
                              stream_tag=f'{corner}_{ncomp}')


def _cfg():
    from wavesim.config import WaveSimConfig
    cfg = WaveSimConfig()
    cfg.mesh.nx, cfg.mesh.ny = MESH_NX, MESH_NY
    cfg.hos.enabled = False                       # ⛔선형 GT (§주장하지 않는 것)
    cfg.sea_states = dict(cfg.sea_states)         # 배포 노브 — wavesim 무수정
    for name, c in CORNERS.items():
        cfg.sea_states[name] = {'Hs': c['Hs'], 'Tp': c['Tp']}
    return cfg


def screen(corner, ncomp, seed):
    """생성 전 선별 — **GT 무관·결과 무관**. (mesh_verdict, q_ok, var_under)"""
    from wavesim import mesh_adequacy
    from wavesim.wave_generator import check_q_constraint
    comp = components(corner, ncomp, seed)
    q_ok, q_val = check_q_constraint(comp)
    info = mesh_adequacy.check(comp, _cfg(), hos_order=None)
    return dict(mesh_verdict=info['mesh_verdict'],
                var_under=float(info['var_under_resolved']),
                lambda_min=float(info['lambda_min']),
                lambda_peak=float(info['lambda_peak']),
                q_ok=bool(q_ok), q_val=float(q_val))


def fname(corner, ncomp, seed):
    return f'episode_{corner}_{ncomp}comp_seed{seed}_{SUFFIX}.h5'


EP_RE = re.compile(r'episode_(SS-[A-Z0-9]+)_(\d+)comp_seed(\d+)_(.+)\.h5')


def select():
    """동결 선택 규칙. 반환: (episodes, rejected). ⛔결과를 보지 않는다."""
    eps, rej = [], []
    for corner in CORNER_ORDER:
        for ncomp in NCOMPS:
            got = 0
            for seed in SEED_POOL:
                s = screen(corner, ncomp, seed)
                if s['mesh_verdict'] == MESH_BAD or not s['q_ok']:
                    rej.append(dict(corner=corner, ncomp=ncomp, seed=seed, **s))
                    continue
                eps.append(dict(corner=corner, ncomp=ncomp, seed=seed,
                                episode=fname(corner, ncomp, seed)[len('episode_'):-3],
                                path=os.path.join(DATA, fname(corner, ncomp, seed)),
                                Hs=CORNERS[corner]['Hs'], Tp=CORNERS[corner]['Tp'],
                                hs_level=CORNERS[corner]['hs'],
                                tp_level=CORNERS[corner]['tp'], **s))
                got += 1
                if got == K_SEEDS:
                    break
            assert got == K_SEEDS, f'{corner} {ncomp}comp: 풀 부족 ({got}/{K_SEEDS})'
    return eps, rej


# ★동결 기대값 — `select()` 를 결과 0행에서 돌려 얻었다. G-1 이 매 실행 재대조한다.
FROZEN_SEEDS = {
    'SS-BF4|15': tuple(range(1641, 1653)), 'SS-BF4|30': tuple(range(1641, 1653)),
    'SS-LOWH|15': tuple(range(1641, 1653)), 'SS-LOWH|30': tuple(range(1641, 1653)),
    'SS-SHORT|15': tuple(range(1641, 1653)), 'SS-SHORT|30': tuple(range(1641, 1653)),
    'SS-REAL|15': (1641, 1642, 1643, 1645, 1646, 1647, 1648, 1649, 1650, 1651, 1652, 1653),
    'SS-REAL|30': tuple(range(1641, 1653)),
}
FROZEN_REJECTED = (('SS-REAL', 15, 1644),)
SEEDS_CONSUMED = tuple(range(1641, 1654))


# ── 배포 상수 되읽기 (패키지에서, 손복사 0) ────────────────────────────────────
def deploy_constants():
    from wavesplat.run import DEFAULT_CONFIG
    from wavesplat.optimizer import sequential_update as SU
    return {'DEFAULT_CONFIG.varpro_steps': DEFAULT_CONFIG['varpro_steps'],
            'DEFAULT_CONFIG.learned_init_iters': DEFAULT_CONFIG['learned_init_iters'],
            'DEFAULT_CONFIG.auto_switch_thresh': DEFAULT_CONFIG['auto_switch_thresh'],
            'DEFAULT_CONFIG.lambda_l1': DEFAULT_CONFIG['lambda_l1'],
            'DEFAULT_CONFIG.init_method': DEFAULT_CONFIG['init_method'],
            'SU.VARPRO_STEPS_DEFAULT': SU.VARPRO_STEPS_DEFAULT,
            'SU.LEARNED_INIT_ITERS_DEFAULT': SU.LEARNED_INIT_ITERS_DEFAULT,
            'SU.AUTO_SWITCH_THRESH_DEFAULT': SU.AUTO_SWITCH_THRESH_DEFAULT,
            'SU.LAMBDA_L1_DEFAULT': SU.LAMBDA_L1_DEFAULT}


DEPLOY_EXPECTED = {'varpro_steps': 400, 'learned_init_iters': 0,
                   'auto_switch_thresh': 0, 'lambda_l1': 1e-4}
ROUTE_EXPECTED = 'learned'            # 합성 경로 통일(임계 0) ⟹ 라우터는 항상 학습

# 러너가 `run_estimation` / `run_letkf` 의 **반환 객체**에서 잡아 행에 싣는 키
# (⛔입력 dict 복사가 아니다 — CLAUDE.md "effective_config 를 의도표로 만들지 마라")
SHIP_CAPTURE = ('sea_state', 'Hs_true', 'Tp_true', 'auto_selected', 'auto_n_selected',
                'auto_real_gate', 'ensemble_k', 'init_iters', 'varpro_steps',
                'varpro_fired', 'n_gaussians', 'avg_nRMSE')
SPEC_CAPTURE = ('Hs', 'Tp', 'depth', 'loc_radius', 'n_ensemble', 'background',
                'filter_class', 'avg_nRMSE', 'zero_nRMSE')

# ── 훈련 분포 전제 (전제 3) — 게이트가 재도출한다 ─────────────────────────────
MIXED = os.path.join(ROOT, 'tmp', 'mixed_retrain_2026-07-26', 'gen_mixed.py')
TRAIN_TP_MIN_EXPECTED = 3.40          # = BF2 앵커 4.0 × (1 - 0.15)
TRAIN_P_TP_LE_REAL_EXPECTED = 0.00933  # P(Tp <= 3.49) @ n=400k seed 0;  ±0.0015 허용
TRAIN_TOL = 0.0015


def train_seastate_facts(n=400_000, seed=0):
    """`gen_mixed._draw_HsTp` 의 상수를 **파일에서 읽어** 분포를 재도출한다."""
    import ast
    import numpy as np
    src = open(MIXED).read()
    g = {}
    for name in ('ANCHORS', 'ANCHOR_FRAC', 'HS_BOX', 'TP_BOX', 'ANCHOR_JITTER'):
        m = re.search(rf'^{name}\s*=\s*(.+?)\s*(?:#.*)?$', src, re.M)
        g[name] = ast.literal_eval(m.group(1))
    A = np.array(g['ANCHORS'], float)
    rng = np.random.default_rng(seed)
    tp = np.empty(n)
    anc = rng.uniform(size=n) < g['ANCHOR_FRAC']
    ai = rng.integers(0, len(A), size=n)
    j = g['ANCHOR_JITTER']
    tp[anc] = A[ai[anc], 1] * rng.uniform(1 - j, 1 + j, size=anc.sum())
    tp[~anc] = rng.uniform(*g['TP_BOX'], size=(~anc).sum())
    from wavesim.pipeline import SEA_STATE
    ss = tuple((v['Hs'], v['Tp']) for v in SEA_STATE.values())
    return dict(anchors=g['ANCHORS'], tp_box=g['TP_BOX'],
                anchors_equal_sea_state=tuple(map(tuple, A)) == ss,
                tp_min=float(A[:, 1].min() * (1 - j)),
                p_tp_le_real=float((tp <= REAL_TP_MEDIAN).mean()))


# ── 집계 ──────────────────────────────────────────────────────────────────────
def aggregate(rows):
    ok = [r for r in rows if 'error' not in r]
    out = {}
    for corner in CORNER_ORDER:
        sub = [r for r in ok if r.get('corner') == corner]
        ship = med_by_episode(sub, 'SHIP-NOW')
        spec = med_by_episode(sub, 'SPEC25')
        zero = med_by_episode(sub, 'SPEC25', field='zero_nRMSE')
        out[corner] = dict(ship=ship, spec=spec, zero=zero,
                           prim=paired(ship, spec), neg=paired(spec, zero),
                           sz=paired(ship, zero))
    return out


def corner_label(prim, neg):
    """코너 하나의 라벨. ⛔음성대조가 죽으면 우열을 말하지 않는다."""
    if neg['n'] == 0 or neg['better'] < NEG_MIN:
        return ('BASELINE-NO-SKILL', f'기준선 무기술 {neg["better"]}/{neg["n"]} (문턱 {NEG_MIN})')
    w, n = prim['better'], prim['n']
    if w >= CONFIRM:
        return ('HOLDS (CONFIRMATORY)', f'{w}/{n} p={prim["p_one_sided"]:.6f}')
    if w >= NOMINAL:
        return ('HOLDS (NOMINAL)', f'{w}/{n} p={prim["p_one_sided"]:.6f} — 가족단위 미달')
    if w <= WEAK:
        return ('BREAKS', f'{w}/{n} — 부호가 반대로 간다')
    return ('UNDECIDED', f'{w}/{n} — 부호로 갈리지 않는다')


def cross_test(agg, a=PRIMARY_CORNER, b=ANCHOR_CORNER):
    """★사전 지정 대비 **1개**뿐: `SS-REAL` vs `SS-BF4`. 상대차의 Mann–Whitney(양측)."""
    from scipy.stats import fisher_exact, mannwhitneyu

    def rel(c):
        d = agg[c]
        return [(d['ship'][k] - d['spec'][k]) / d['spec'][k]
                for k in sorted(set(d['ship']) & set(d['spec']))]
    xa, xb = rel(a), rel(b)
    u = mannwhitneyu(xa, xb, alternative='two-sided')
    wa, wb = agg[a]['prim']['better'], agg[b]['prim']['better']
    fe = fisher_exact([[wa, agg[a]['prim']['n'] - wa], [wb, agg[b]['prim']['n'] - wb]])
    return dict(contrast=f'{a} vs {b}', n_a=len(xa), n_b=len(xb),
                mw_U=float(u.statistic), mw_p=float(u.pvalue),
                fisher_p=float(fe[1]), wins=[wa, wb], alpha=ALPHA,
                differs=bool(u.pvalue < ALPHA))


# ─────────────────────────────────────────────────────────────────────────────
# ★결과 전 해석 규칙 — **어느 결과가 나와도 무엇을 쓸지 지금 정한다**
# ─────────────────────────────────────────────────────────────────────────────
READING = {
    'HOLDS (CONFIRMATORY)': (
        '★**경계는 해상상태가 아니다.** 실측 해상상태를 참값 있는 합성에서 재현해도 헤드라인이 '
        '산다 ⟹ 실측 음성의 원인은 **sim-to-real(관측 모형·자료 영역)** 쪽으로 좁혀지고, '
        '논문 §「작동 경계」가 ***처음으로 해상상태 교락 없이*** 선다. '
        '⚠단 이 코퍼스는 **선형 GT** 다(§주장하지 않는 것 1 — 비선형 축은 `B-11`/`B-12` 가 '
        '답했다). 코너 대비는 네 칸이 같은 GT 클래스라 **무손상**이나, '
        '***"헤드라인이 산다"*** 문장은 **선형 GT 를 명시해** 인용한다.'),
    'HOLDS (NOMINAL)': (
        '⚠명목만 통과 ⟹ **`EXPLORATORY`**. 위 문장을 쓰되 가족단위 미달을 병기한다. '
        '⚠단 이 코퍼스는 **선형 GT** 다 — 코너 대비는 무손상이나 '
        '***"헤드라인이 산다"*** 문장은 **선형 GT 를 명시해** 인용한다.'),
    'BREAKS': (
        '★★**논문의 「작동 경계」를 재정의한다.** 경계는 *"희소성"* 이 아니라 '
        '***"학습된 앵커 근방"*** 이다(전제 3: `SS-REAL` 근방 학습 밀도 0.0000%). '
        '⟹ (a) `B-18` 의 `NEITHER-SUFFICIENT` 가 **후보 밖**이라 한 그 원인이 여기다. '
        '(b) `B-23`(재학습)이 **처방**이 된다 — 지금은 근거 없는 축이지만 이때 근거가 생긴다. '
        '(c) ⛔*"sim-to-real 이 경계"* 문장을 논문에서 **뺀다**.'),
    'UNDECIDED': (
        '`SEASTATE-UNDECIDED` — 어느 쪽도 못 쓴다. ★**이 라벨도 결과다**: 교락이 안 깨졌으므로 '
        '⛔***실측 음성을 sim-to-real 로 귀속하는 문장을 논문에서 빼야 한다***(귀속의 근거가 없다).'),
    'BASELINE-NO-SKILL': (
        '⛔비교자가 무기술이므로 **우열을 말하지 않는다**(`B-9b`·`B-11` 규약). '
        '★그러나 그 자체가 강한 관찰이다: 실측 해상상태에서는 **최강 기준선조차** 제로필드를 '
        '못 이긴다 ⟹ 실측 음성이 **방법이 아니라 레짐**에 붙는다. 이 경우 `SHIP<ZERO` 를 보고한다.'),
}

# 2×2 분해 읽기 — 코너 라벨 **네 개만** 읽는다(추가 검정 0개, 08-23 층간대비 교훈).
DECOMPOSITION = {
    ('hold', 'hold', 'hold'): '해상상태 축은 매개하지 않는다 (네 칸 전부 유지).',
    ('hold', 'break', 'break'): '★매개는 **주기/파장**이다 (= 사전의 학습 지지집합 밖).',
    ('break', 'hold', 'break'): '★매개는 **파고**다 (= 절대 센서잡음 대비 SNR).',
    ('hold', 'hold', 'break'): '★**상호작용** — 둘이 함께여야 죽는다. 단일 축 귀속 금지.',
    ('break', 'break', 'break'): '두 축 모두 단독으로 죽인다.',
}


def decompose(labels):
    """labels: {corner: label}. `SS-BF4` 는 기준 칸이라 분해에 안 들어간다."""
    def st(c):
        return 'hold' if labels[c].startswith('HOLDS') else (
            'break' if labels[c] in ('BREAKS', 'BASELINE-NO-SKILL') else 'undecided')
    key = (st('SS-LOWH'), st('SS-SHORT'), st('SS-REAL'))
    return key, DECOMPOSITION.get(key, '조합이 표에 없다 ⟹ 단일 축 귀속 금지, 네 라벨을 그대로 보고한다.')


# ─────────────────────────────────────────────────────────────────────────────
# 게이트 — ★단일출처. 러너·분석기가 같은 함수를 부른다. 공허한 통과 금지.
# ─────────────────────────────────────────────────────────────────────────────
RUNFREE_GATES = ('G-0', 'G-1', 'G-2', 'G-3', 'G-4', 'G-5', 'G-19', 'G-20')
# ★행이 0개일 때 **참이 되어도 되는** 게이트만 여기 적는다(W8 `G-4` 공허한 통과 부류).
#   `selftest()` 가 `rows=[]` 로 돌려 **이 목록 밖에서 통과한 것**을 실패로 센다.
#   `G-11`(에러 0행) · `G-18`(shim 투명)은 0행에서 정의상 참이고, **행이 있는지는
#   `G-12`/`G-12b` 가 따로 요구**하므로 이 둘의 공허한 통과는 무해하다.
ROWLESS_OK = ('G-11', 'G-18')
# G-5 의 탐색 패턴 — ★자가진단이 이것을 갈아 끼워 게이트가 **실제로 발화하는지** 본다.
BF23_GLOB = 'episode_BF[23]_*.h5'
MESH_DX_EXPECTED = 1.2060301507537687      # gen_s8 mesh 200×167 의 dx


def gates_runfree():
    """생성 **전**에 전부 판정 가능한 게이트. `python3 gates_b26.py` 로 돈다."""
    out = []
    eps, rej = select()
    got = {f'{c}|{n}': tuple(e['seed'] for e in eps
                             if e['corner'] == c and e['ncomp'] == n)
           for c in CORNER_ORDER for n in NCOMPS}
    out.append(('G-1 표본 재도출 = 동결 목록', got == FROZEN_SEEDS,
                f'{len(eps)}/{N_EP} 에피소드'))
    out.append(('G-1b 코너당 24', all(len(v) == K_SEEDS for v in got.values()),
                str(sorted({len(v) for v in got.values()}))))
    out.append(('G-2 선택분 INADEQUATE 0 · 거부목록 일치',
                all(e['mesh_verdict'] != MESH_BAD for e in eps)
                and tuple((r['corner'], r['ncomp'], r['seed']) for r in rej) == FROZEN_REJECTED,
                f'거부 {[(r["corner"], r["ncomp"], r["seed"]) for r in rej]}'))
    out.append(('G-3 Q 제약 96/96', all(e['q_ok'] for e in eps),
                f'{sum(e["q_ok"] for e in eps)}/{len(eps)}'))
    # G-0 시드 원장 충돌 — *"이 배치가 소모한 시드를 **다른** 배치가 이미 썼는가"*
    # ⚠2026-08-27 자가진단이 잡은 결함: 종전에는 **이 배치 자신의 산출물**
    #   (`data_b26/`·`smoke_b26/`·`pc1/`)까지 세어, 생성이 끝나는 순간 G-0 이 **반드시
    #   FAIL** 하게 돼 있었다 ⟹ 분석기가 낼 최종 게이트표가 구조적으로 한 칸 깨진다.
    #   ⟹ **배치 디렉터리 아래는 제외**한다. 게이트의 뜻은 그대로다(밖과의 충돌만 본다).
    used = set()
    _mine = os.path.realpath(HERE) + os.sep
    for p in glob.glob(os.path.join(ROOT, '..', '**', 'episode_*seed*.h5'), recursive=True):
        if os.path.realpath(p).startswith(_mine):
            continue
        m = re.search(r'seed(\d+)_', os.path.basename(p))
        if m:
            used.add(int(m.group(1)))
    out.append(('G-0 시드 원장 충돌 0 (⛔이 배치 디렉터리 제외)',
                not (set(SEEDS_CONSUMED) & used),
                f'충돌 {sorted(set(SEEDS_CONSUMED) & used)}'))
    # G-4 훈련 분포 전제 재도출
    try:
        t = train_seastate_facts()
        ok = (t['anchors_equal_sea_state']
              and abs(t['tp_min'] - TRAIN_TP_MIN_EXPECTED) < 1e-9
              and abs(t['p_tp_le_real'] - TRAIN_P_TP_LE_REAL_EXPECTED) < TRAIN_TOL)
        out.append(('★G-4 학습 Tp 분포 전제 재도출', bool(ok),
                    f'ANCHORS≡SEA_STATE {t["anchors_equal_sea_state"]} · '
                    f'Tp_min {t["tp_min"]:.3f} · P(Tp≤{REAL_TP_MEDIAN}) {100*t["p_tp_le_real"]:.3f}%'))
    except Exception as e:                                   # noqa: BLE001
        out.append(('★G-4 학습 Tp 분포 전제 재도출', False, repr(e)[:80]))
    # G-5 BF2/BF3 혼잡 부재 (전제 1)
    bad = [p for p in glob.glob(os.path.join(ROOT, '..', '**', BF23_GLOB),
                                recursive=True)
           if int(re.search(r'_(\d+)comp', os.path.basename(p)).group(1)) >= 15]
    out.append(('G-5 BF2/BF3 혼잡 에피소드 0개 (생성이 선행한다)', not bad, f'{len(bad)}개'))
    # G-19 문턱 산술
    t = threshold_table()
    out.append(('★G-19 문턱 산술', t['confirm_within_family'] and t['weak_is_sibling']
                and t['p_nominal'] < ALPHA,
                f'NOMINAL {NOMINAL} p={t["p_nominal"]:.5f} · CONFIRM {CONFIRM} '
                f'p={t["p_confirm"]:.5f} ≤ {FAMILY_ALPHA} · WEAK {WEAK}=n-NOMINAL'))
    # G-20 배포 상수 단일출처
    try:
        d = deploy_constants()
        ok = (d['DEFAULT_CONFIG.varpro_steps'] == DEPLOY_EXPECTED['varpro_steps']
              == d['SU.VARPRO_STEPS_DEFAULT']
              and d['DEFAULT_CONFIG.learned_init_iters'] == DEPLOY_EXPECTED['learned_init_iters']
              == d['SU.LEARNED_INIT_ITERS_DEFAULT']
              and d['DEFAULT_CONFIG.auto_switch_thresh'] == DEPLOY_EXPECTED['auto_switch_thresh']
              == d['SU.AUTO_SWITCH_THRESH_DEFAULT']
              and abs(d['DEFAULT_CONFIG.lambda_l1'] - DEPLOY_EXPECTED['lambda_l1']) < 1e-12
              and d['DEFAULT_CONFIG.lambda_l1'] == d['SU.LAMBDA_L1_DEFAULT'])
        out.append(('★G-20 배포 상수 = 단일출처 = 사전등록값', bool(ok), str(d)[:150]))
    except Exception as e:                                   # noqa: BLE001
        out.append(('★G-20 배포 상수', False, repr(e)[:80]))
    return out


def _read_ep(path):
    """★`gates_data` 의 **유일한 I/O 지점**. 자가진단은 이 함수만 갈아 끼워
    게이트 논리를 **한 글자도 바꾸지 않고** 발화를 시험한다."""
    import h5py
    import numpy as np
    with h5py.File(path, 'r') as f:
        a = dict(f['config'].attrs)
        gt = f['frames/gt_wave_field'][:]
    return dict(Hs=float(a['Hs']), Tp=float(a['Tp']),
                hs4std=4 * float(np.std(gt)), n_frames=int(gt.shape[0]),
                mesh_dx=float(a['mesh_dx']), depth=float(a['depth']),
                ra=bool(a.get('random_amplitudes')),
                rf=bool(a.get('random_frequencies')),
                ds=bool(a.get('decorrelate_stream')))


def gates_data(read=None, exists=None):
    """생성 **후**·추정 **전**: 저장된 .h5 에서 되읽는다. ⛔`cfg[k]==cfg[k]` 항등식 아님.
    ⚠`read`/`exists` 는 **자가진단 전용 주입구**다 — 기본값이 실제 파일을 연다."""
    read = read or _read_ep
    exists = exists or os.path.exists
    out = []
    eps, _ = select()
    have = [e for e in eps if exists(e['path'])]
    out.append(('G-6 에피소드 96개 생성됨', len(have) == N_EP, f'{len(have)}/{N_EP}'))
    if not have:
        return out
    seen, meta_ok, nfr = set(), True, set()
    hs4, rel = {c: [] for c in CORNER_ORDER}, []
    gp = gen_params()
    for e in have:
        a = read(e['path'])
        seen.add((round(a['Hs'], 6), round(a['Tp'], 6)))
        v = a['hs4std']
        hs4[e['corner']].append(v)
        rel.append(v / e['Hs'])
        nfr.add(a['n_frames'])
        meta_ok &= (abs(a['mesh_dx'] - MESH_DX_EXPECTED) < 1e-9
                    and a['ra'] and a['rf'] and a['ds']
                    and abs(a['depth'] - gp['depth']) < 1e-9)
    out.append(('★G-7 저장 파일에서 (Hs,Tp) 되읽기 = 코너 4종', len(seen) == 4, str(sorted(seen))))
    # ★G-8 개입의 **파생 측정** — 설정 메아리가 아니라 자료에서 잰 값.
    # ⚠공칭 Hs 대비 실현 4·std 는 코너별로 흔들린다(런 0회 실측: [0.846, 1.188]) —
    #   긴 주기 코너일수록 창 안에 드는 독립 파주기가 적기 때문이다. 평균은 편향 0.
    #   ⟹ 문턱 셋을 **결과 전에** 그 실측 위에 여유를 두고 정한다.
    means = {c: (sum(v) / len(v) if v else float('nan')) for c, v in hs4.items()}
    out.append(('★G-8a 코너 평균 4·std/Hs == 1 (±5%)',
                all(abs(means[c] / CORNERS[c]['Hs'] - 1) < 0.05 for c in CORNER_ORDER if hs4[c]),
                str({c: round(means[c] / CORNERS[c]['Hs'], 4) for c in CORNER_ORDER if hs4[c]})))
    lo = [v for c in CORNER_ORDER if CORNERS[c]['hs'] == 'lo' for v in hs4[c]]
    hi = [v for c in CORNER_ORDER if CORNERS[c]['hs'] == 'hi' for v in hs4[c]]
    out.append(('★G-8b **파고 축이 실제로 갈렸다**: 저-Hs 최대 < 고-Hs 최소 (겹침 0)',
                bool(lo and hi) and max(lo) < min(hi),
                f'max(lo)={max(lo):.4f} m < min(hi)={min(hi):.4f} m' if lo and hi else ''))
    out.append(('★G-8c 에피소드별 |4·std/Hs − 1| ≤ 0.25 (런 0회 실측 0.188 + 여유)',
                bool(rel) and max(abs(r - 1) for r in rel) <= 0.25,
                f'max {max(abs(r-1) for r in rel):.4f}' if rel else ''))
    out.append(('G-9 메타 되읽기(mesh 200×167 · ra/rf/ds · depth)', meta_ok, ''))
    out.append(('G-10 프레임 수 단일', len(nfr) == 1 and nfr == {gp['n_frames']}, str(nfr)))
    return out


def gates_run(rows):
    """추정 후. `rows` = results_b26*.jsonl 전건."""
    out = []
    ok = [r for r in rows if 'error' not in r]
    out.append(('G-11 에러 0행', len(rows) == len(ok), f'{len(rows)-len(ok)}'))
    sh = [r for r in ok if r.get('arm') == 'SHIP-NOW']
    sp = [r for r in ok if r.get('arm') == 'SPEC25']
    out.append(('G-12 SHIP 288행', len(sh) == N_EP * len(EST_SEEDS), f'{len(sh)}/{N_EP*3}'))
    out.append(('G-12b SPEC25 288행', len(sp) == N_EP * len(FILTER_SEEDS), f'{len(sp)}/{N_EP*3}'))
    es, ep_ = {r['episode'] for r in sh}, {r['episode'] for r in sp}
    out.append(('★G-13 두 팔 에피소드 집합 완전 일치', es == ep_ and len(es) == N_EP,
                f'{len(es)}∩{len(ep_)}={len(es & ep_)}'))
    out.append(('★G-14 SHIP 배포값 되읽기(400·0·0·1e-4)',
                bool(sh) and all(r.get('varpro_steps') == DEPLOY_EXPECTED['varpro_steps']
                                 and r.get('learned_init_iters') == DEPLOY_EXPECTED['learned_init_iters']
                                 and r.get('auto_switch_thresh') == DEPLOY_EXPECTED['auto_switch_thresh']
                                 and abs(r.get('lambda_l1', -1) - DEPLOY_EXPECTED['lambda_l1']) < 1e-12
                                 for r in sh),
                str(sorted({(r.get('varpro_steps'), r.get('learned_init_iters'),
                             r.get('auto_switch_thresh')) for r in sh}))[:70]))
    out.append(('★G-15 라우터가 전 코너에서 학습 경로 (배포 사전이 실제로 시험됐다)',
                bool(sh) and all(r.get('cap_auto_selected') == ROUTE_EXPECTED for r in sh),
                str(sorted({(r.get('corner'), r.get('cap_auto_selected')) for r in sh}))[:120]))
    out.append(('★G-15b VARPRO 발화 전건 (처치가 아니라 배포가 돌았다는 계수기)',
                bool(sh) and all(bool(r.get('cap_varpro_fired')) for r in sh),
                f'{sum(bool(r.get("cap_varpro_fired")) for r in sh)}/{len(sh)}'))
    out.append(('★G-16 SPEC25 되읽기(loc 25 · ens 600 · jonswap)',
                bool(sp) and all(abs(r.get('loc_radius', -1) - SPEC['loc']) < 1e-9
                                 and r.get('n_ensemble') == SPEC['n_ensemble']
                                 and r.get('background') == SPEC['background'] for r in sp), ''))
    out.append(('★G-17 기준선 배경이 코너를 따라갔다 (비교자 불리 반박 차단)',
                bool(sp) and all(abs(r.get('cap_Hs', -1) - CORNERS[r['corner']]['Hs']) < 1e-6
                                 and abs(r.get('cap_Tp', -1) - CORNERS[r['corner']]['Tp']) < 1e-6
                                 for r in sp),
                str(sorted({(r.get('corner'), round(r.get('cap_Hs', -1), 3),
                             round(r.get('cap_Tp', -1), 3)) for r in sp}))[:140]))
    out.append(('★G-18 계측 shim 투명 (잡은 avg_nRMSE == 행의 avg_nRMSE, 비트)',
                all(r.get('cap_avg_nRMSE') == r.get('avg_nRMSE') for r in ok),
                f'{sum(r.get("cap_avg_nRMSE") == r.get("avg_nRMSE") for r in ok)}/{len(ok)}'))
    zero = [r for r in sp if r.get('zero_nRMSE') is not None]
    out.append(('G-18b ZERO 96 에피소드', len({r['episode'] for r in zero}) == N_EP,
                f'{len({r["episode"] for r in zero})}'))
    # ★G-21 내부 기준 칸 생존 — FAIL 이면 배치 무효
    agg = aggregate(ok)
    a = agg.get(ANCHOR_CORNER, {}).get('neg', {'better': -1, 'n': 0})
    out.append((f'★G-21 내부 기준 칸 {ANCHOR_CORNER} 음성대조 생존 (FAIL ⟹ 배치 무효)',
                a['better'] >= NEG_MIN, f'{a["better"]}/{a["n"]} (문턱 {NEG_MIN})'))
    return out


def gates_pc1(pc1):
    """PC-1 생성기 비트동일 양성대조. `pc1` = gen_b26.pc1() 의 산출 dict."""
    return [('★G-16b PC-1 생성기 비트동일 (동결 `B-1` 코퍼스 재생성)',
             bool(pc1) and all(pc1['identical'].get(k) for k in PC1_DATASETS),
             str(pc1.get('identical')) + ' ' + str(pc1.get('maxabs', ''))[:60])]


# ─────────────────────────────────────────────────────────────────────────────
# ★런 전 자가진단 — 가짜 행·가짜 파일로 게이트가 **실제로 발화하는지** 확인한다.
#
# 왜 있는가: 이 저장소가 반복해서 물린 결함은 *"게이트가 있는데 검사하지 않는다"* 부류다
# (W8 `G-4` 는 **없는 키**를 읽어 전 행 `None` 인 채 **공허하게 통과**했고,
#  W7 `G-3` 은 **다른 노브**를 넣어 통과했으며, 2026-08-23 야간 감사는 하루에 세 번
#  같은 부류를 잡았다). ⟹ ***게이트도 시험한다.***
#
# ⛔파일을 쓰지 않는다 · ⛔결과를 만들지 않는다 · 런 0회 · 실측값 0개.
# ⚠픽스처 값은 **임의 상수**(0.25·0.60·1.15)와 이름 해시의 ±5% 흔들림이다 —
#   어떤 실측도 담기지 않으므로 **결과를 미리 보는 것이 아니다**.
# ─────────────────────────────────────────────────────────────────────────────
FIX_SPEC_BASE = 0.25          # 픽스처 기준선 nRMSE (임의 — 정보 0)
FIX_SHIP_MULT = 0.60          # 픽스처 SHIP = SPEC × 0.60
FIX_ZERO_MULT = 1.15          # 픽스처 ZERO = SPEC × 1.15


def gate_id(name):
    """게이트 이름 → 식별자. `★G-12b …` → `G-12b`. ⚠`G-12` 와 `G-12b` 를 **구분**한다."""
    return name.lstrip('★ ').split()[0]


def _jit(key):
    """이름 해시 기반 ±5% 결정적 흔들림 — 픽스처가 동점 투성이가 되지 않게 한다."""
    import hashlib
    h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
    return 1.0 + (h % 2001 - 1000) / 20000.0


def _fixture_eps():
    """동결 표본에서 96 에피소드 (⛔`select()` 를 안 쓴다 — 그쪽은 따로 시험한다)."""
    return [dict(corner=c, ncomp=n, seed=s,
                 episode=fname(c, n, s)[len('episode_'):-3])
            for c in CORNER_ORDER for n in NCOMPS for s in FROZEN_SEEDS[f'{c}|{n}']]


def _fixture_rows():
    """정상 케이스 576행. 러너가 **실제로 싣는 키**를 그대로 갖춘다(run_b26.py 참조)."""
    rows = []
    for e in _fixture_eps():
        c = CORNERS[e['corner']]
        spec = FIX_SPEC_BASE * _jit(e['episode'])
        zero = spec * FIX_ZERO_MULT
        for s in EST_SEEDS:
            v = spec * FIX_SHIP_MULT * _jit(f'{e["episode"]}|ship|{s}')
            rows.append(dict(
                arm='SHIP-NOW', episode=e['episode'], corner=e['corner'],
                bf=e['corner'], ncomp=e['ncomp'], gen_seed=e['seed'], est_seed=s,
                avg_nRMSE=v, zero_nRMSE=zero, wall_s=1.0,
                varpro_steps=DEPLOY_EXPECTED['varpro_steps'],
                learned_init_iters=DEPLOY_EXPECTED['learned_init_iters'],
                auto_switch_thresh=DEPLOY_EXPECTED['auto_switch_thresh'],
                bank_refit_every_s=None,
                lambda_l1=DEPLOY_EXPECTED['lambda_l1'],              # ★객체 되읽기
                obj_lambda_l1=DEPLOY_EXPECTED['lambda_l1'],
                obj_init_method='auto', obj_tracking_mode='w_phi',
                obj_real_estimator='bank_mmse',
                cap_sea_state=e['corner'], cap_Hs_true=c['Hs'], cap_Tp_true=c['Tp'],
                cap_auto_selected=ROUTE_EXPECTED, cap_auto_n_selected=18,
                cap_auto_real_gate=None, cap_ensemble_k=1, cap_init_iters=0,
                cap_varpro_steps=DEPLOY_EXPECTED['varpro_steps'],
                cap_varpro_fired=True, cap_n_gaussians=18, cap_avg_nRMSE=v))
        for s in FILTER_SEEDS:
            v = spec * _jit(f'{e["episode"]}|spec|{s}')
            rows.append(dict(
                arm='SPEC25', episode=e['episode'], corner=e['corner'],
                bf=e['corner'], ncomp=e['ncomp'], gen_seed=e['seed'], filter_seed=s,
                avg_nRMSE=v, zero_nRMSE=zero, wall_s=1.0,
                loc_radius=SPEC['loc'], n_ensemble=SPEC['n_ensemble'],
                background=SPEC['background'],
                cap_Hs=c['Hs'], cap_Tp=c['Tp'], cap_depth=50.0,
                cap_loc_radius=SPEC['loc'], cap_n_ensemble=SPEC['n_ensemble'],
                cap_background=SPEC['background'], cap_filter_class='LETKFGridSpec',
                cap_avg_nRMSE=v, cap_zero_nRMSE=zero))
    return rows


def _fixture_data():
    """정상 케이스 자료 되읽기: `(exists, read)`. ⛔.h5 를 만들지 않는다."""
    gp = gen_params()
    base = {}
    for e in _fixture_eps():
        c = CORNERS[e['corner']]
        base[os.path.join(DATA, fname(e['corner'], e['ncomp'], e['seed']))] = dict(
            Hs=c['Hs'], Tp=c['Tp'], hs4std=c['Hs'], n_frames=gp['n_frames'],
            mesh_dx=MESH_DX_EXPECTED, depth=gp['depth'], ra=True, rf=True, ds=True)
    return base


def _fixture_pc1():
    return dict(frozen=PC1_PATH, regenerated='(fixture)',
                identical={k: True for k in PC1_DATASETS},
                maxabs={k: 0.0 for k in PC1_DATASETS})


def _all_gates(rows, data, pc1):
    """자가진단이 한 번에 보는 게이트 전체(런 0회 게이트는 §4 에서 따로)."""
    return (list(gates_data(read=lambda p: data[p], exists=lambda p: p in data))
            + list(gates_run(rows)) + list(gates_pc1(pc1)))


# ── 변형: 게이트마다 **그것을 깨는** 개입 ──────────────────────────────────────
def _perturbations():
    """`(게이트 id, 설명, 변형함수)`. 변형은 `(rows, data, pc1)` 을 받아 새 셋을 낸다."""
    import copy

    def R(fn):                                    # rows 만 바꾼다
        def apply(rows, data, pc1):
            rs = copy.deepcopy(rows)
            fn(rs)
            return rs, data, pc1
        return apply

    def D(fn):                                    # data 만 바꾼다
        def apply(rows, data, pc1):
            dd = copy.deepcopy(data)
            fn(dd)
            return rows, dd, pc1
        return apply

    def P(fn):                                    # pc1 만 바꾼다
        def apply(rows, data, pc1):
            pp = copy.deepcopy(pc1)
            fn(pp)
            return rows, data, pp
        return apply

    def first(rs, arm, corner=None):
        return next(r for r in rs if r['arm'] == arm
                    and (corner is None or r['corner'] == corner))

    def rename_one_spec(rs):
        name = first(rs, 'SPEC25')['episode']
        for r in rs:
            if r['arm'] == 'SPEC25' and r['episode'] == name:
                r['episode'] = 'GHOST'

    def drop_zero_one_episode(rs):
        name = first(rs, 'SPEC25')['episode']
        for r in rs:
            if r['arm'] == 'SPEC25' and r['episode'] == name:
                r['zero_nRMSE'] = None

    def baseline_kept_bf4(rs):
        for r in rs:
            if r['arm'] == 'SPEC25' and r['corner'] == PRIMARY_CORNER:
                r['cap_Hs'] = CORNERS[ANCHOR_CORNER]['Hs']
                r['cap_Tp'] = CORNERS[ANCHOR_CORNER]['Tp']

    def anchor_neg_dies(rs):
        for r in rs:
            if r['arm'] == 'SPEC25' and r['corner'] == ANCHOR_CORNER:
                r['zero_nRMSE'] = r['avg_nRMSE'] * 0.5

    def one_key(d, key, value):
        k = sorted(d)[0]
        d[k] = dict(d[k], **{key: value})

    def lo_overlaps_hi(d):
        k = next(p for p in sorted(d) if f'_{PRIMARY_CORNER}_' in p)
        d[k] = dict(d[k], hs4std=1.25)            # 저-Hs 인데 고-Hs 위로

    def corner_mean_off(d):
        for k in sorted(d):
            if '_SS-LOWH_' in k:
                d[k] = dict(d[k], hs4std=d[k]['hs4std'] * 1.20)

    def one_ep_ratio_off(d):
        k = sorted(d)[0]
        d[k] = dict(d[k], hs4std=d[k]['Hs'] * 1.30)

    return [
        # ── 추정 후 (gates_run) ────────────────────────────────────────────
        ('G-11', '오류 행 1개 (러너가 예외를 삼켰다)', R(lambda rs: rs.append(
            dict(arm='SHIP-NOW', episode='x', corner=PRIMARY_CORNER,
                 est_seed=42, error='boom')))),
        ('G-12', 'SHIP 한 행 유실', R(lambda rs: rs.remove(first(rs, 'SHIP-NOW')))),
        ('G-12b', 'SPEC25 한 행 유실', R(lambda rs: rs.remove(first(rs, 'SPEC25')))),
        ('G-13', '★두 팔이 다른 에피소드를 돌았다 (한 에피소드 이름만 다르다)',
         R(rename_one_spec)),
        ('G-14', '★배포값이 아니다 — varpro_steps 400→0', R(
            lambda rs: first(rs, 'SHIP-NOW').__setitem__('varpro_steps', 0))),
        ('G-14', '★λ 가 구 기본값 1e-3 (객체 되읽기가 잡는다)', R(
            lambda rs: first(rs, 'SHIP-NOW').__setitem__('lambda_l1', 1e-3))),
        ('G-15', '★라우터가 한 코너에서 v2_konly 로 갔다 (배포 사전이 안 시험됐다)', R(
            lambda rs: first(rs, 'SHIP-NOW', PRIMARY_CORNER).__setitem__(
                'cap_auto_selected', 'v2_konly'))),
        ('G-15b', '★VARPRO 가 발화하지 않았다', R(
            lambda rs: first(rs, 'SHIP-NOW').__setitem__('cap_varpro_fired', False))),
        ('G-16', '★기준선이 loc 25 가 아니다', R(
            lambda rs: first(rs, 'SPEC25').__setitem__('loc_radius', 15.0))),
        ('G-17', '★★기준선 배경이 코너를 안 따라갔다 (SS-REAL 인데 BF4 사전)',
         R(baseline_kept_bf4)),
        ('G-18', '★shim 이 값을 바꿨다 (잡은 값 ≠ 행의 값)', R(
            lambda rs: first(rs, 'SHIP-NOW').__setitem__(
                'cap_avg_nRMSE', first(rs, 'SHIP-NOW')['avg_nRMSE'] + 1e-12))),
        ('G-18b', 'ZERO 가 한 에피소드에서 비었다', R(drop_zero_one_episode)),
        ('G-21', '★★내부 기준 칸의 음성대조가 죽었다 (FAIL ⟹ 배치 무효)',
         R(anchor_neg_dies)),
        # ── 생성 후·추정 전 (gates_data) ───────────────────────────────────
        ('G-6', '에피소드 한 개가 없다', D(lambda d: d.pop(sorted(d)[0]))),
        ('G-7', '저장 파일의 (Hs,Tp) 가 코너 4종이 아니다',
         D(lambda d: one_key(d, 'Hs', 0.9))),
        ('G-8a', '★코너 평균 4·std/Hs 가 1 에서 20% 벗어난다', D(corner_mean_off)),
        ('G-8b', '★★파고 축이 안 갈렸다 (저-Hs 한 개가 고-Hs 위로)', D(lo_overlaps_hi)),
        ('G-8c', '★에피소드 하나의 4·std/Hs 가 1.30', D(one_ep_ratio_off)),
        ('G-9', '메타(mesh_dx)가 다르다', D(lambda d: one_key(d, 'mesh_dx', 1.0))),
        ('G-10', '프레임 수가 갈린다', D(lambda d: one_key(d, 'n_frames', 300))),
        # ── 생성기 양성대조 ─────────────────────────────────────────────────
        ('G-16b', '★PC-1 이 비트 동일이 아니다 (생성기가 헤드라인 코퍼스와 다르다)',
         P(lambda p: p['identical'].__setitem__('gt_wave_field', False))),
    ]


def _runfree_perturbations():
    """런 0회 게이트의 변형 — **모듈 전역을 갈아 끼웠다 되돌린다**."""
    import copy
    base_eps, base_rej = select()
    rej_like = [dict(corner=c, ncomp=n, seed=s) for c, n, s in FROZEN_REJECTED]

    def trunc():
        return base_eps[1:], rej_like

    def qbad():
        eps = copy.deepcopy(base_eps)
        eps[0]['q_ok'] = False
        return eps, rej_like

    frozen_bad = dict(FROZEN_SEEDS)
    frozen_bad[f'{ANCHOR_CORNER}|15'] = tuple(range(1642, 1654))
    return [
        ('G-0', '시드 원장이 이미 쓰인 시드를 포함한다', {'SEEDS_CONSUMED': (0,)}),
        ('G-1', '★재도출 표본이 동결 목록과 어긋난다', {'FROZEN_SEEDS': frozen_bad}),
        ('G-1b', '한 코너가 11개뿐이다 (표본 부족)', {'select': trunc}),
        ('G-2', '거부 목록이 동결과 다르다', {'FROZEN_REJECTED': ()}),
        ('G-3', 'Q 제약을 어긴 에피소드가 표본에 들어왔다', {'select': qbad}),
        ('G-4', '★학습 Tp 분포 전제가 어긋난다 (§1.4 의 근거)',
         {'TRAIN_TP_MIN_EXPECTED': 9.99}),
        ('G-5', 'BF2/BF3 혼잡 에피소드가 저장소에 있다', {'BF23_GLOB': 'episode_BF4_*.h5'}),
        ('G-19', '★CONFIRM 이 가족단위 α 를 넘는다', {'CONFIRM': NOMINAL}),
        ('G-20', '★배포 상수가 사전등록값과 다르다',
         {'DEPLOY_EXPECTED': dict(DEPLOY_EXPECTED, varpro_steps=500)}),
    ]


def _label_reachability():
    """★코너 라벨 **5종이 전부 도달 가능한지** 기계로 확인한다(결과 전).

    ⛔도달 불가 라벨은 저장소가 이미 두 번 물린 결함이다(`gates_b9` `LOSES-GAIN`,
    `gates_b10` `NOSKILL`) — **불리한 라벨만 조용히 막히는** 부류라 더 나쁘다."""
    import copy
    base = _fixture_rows()
    prim_eps = sorted({r['episode'] for r in base if r['corner'] == PRIMARY_CORNER})
    seen = {}

    def _try(name, fn):
        rs = copy.deepcopy(base)
        fn(rs)
        d = aggregate(rs)[PRIMARY_CORNER]
        lab, msg = corner_label(d['prim'], d['neg'])
        seen.setdefault(lab, f'{name} → {lab} ({msg})')
        return lab

    def worsen(k):
        bad = set(prim_eps[len(prim_eps) - k:])

        def fn(rs):
            for r in rs:
                if r['arm'] == 'SHIP-NOW' and r['episode'] in bad:
                    r['avg_nRMSE'] *= 2.0
                    r['cap_avg_nRMSE'] = r['avg_nRMSE']
        return fn

    def neg_dies(rs):
        for r in rs:
            if r['arm'] == 'SPEC25' and r['corner'] == PRIMARY_CORNER:
                r['zero_nRMSE'] = r['avg_nRMSE'] * 0.5

    _try('기저 픽스처 (24/24)', lambda rs: None)
    _try(f'{N_PER_CORNER - NOMINAL}개 악화 ({NOMINAL}/24)', worsen(N_PER_CORNER - NOMINAL))
    _try('12개 악화 (12/24)', worsen(12))
    _try('24개 전부 악화 (0/24)', worsen(24))
    _try('음성대조 사망 (SPEC25 > ZERO)', neg_dies)
    return seen


def selftest(verbose=True, runfree=True):
    """★게이트 자가진단. `(통과여부, 로그)`. ⛔파일을 쓰지 않는다."""
    log = []

    def say(s):
        log.append(s)
        if verbose:
            print(s, flush=True)

    fails = []
    rows, data, pc1 = _fixture_rows(), _fixture_data(), _fixture_pc1()

    # [1/6] 정상 케이스에서 전 게이트 PASS -------------------------------------
    res = _all_gates(rows, data, pc1)
    bad = [n for n, okk, _ in res if not okk]
    say(f'[1/6] 픽스처 {len(rows)}행 · 자료 {len(data)}건 — '
        f'게이트 {len(res) - len(bad)}/{len(res)} PASS')
    for n, okk, d in res:
        if not okk:
            say(f'      ⛔{n}: {d}')
    if bad:
        fails.append(f'픽스처에서 통과하지 못한 게이트 {len(bad)}: {bad}')

    if runfree:
        rf = gates_runfree()
        rbad = [n for n, okk, _ in rf if not okk]
        say(f'[1b/6] 런 0회 게이트 {len(rf) - len(rbad)}/{len(rf)} PASS')
        for n, okk, d in rf:
            if not okk:
                say(f'      ⛔{n}: {d}')
        if rbad:
            fails.append(f'런 0회 게이트 실패 {rbad}')

    # [2/6] 공허한 통과 (rows=[] · data={}) ------------------------------------
    empty = _all_gates([], {}, {})
    leaked = [n for n, okk, _ in empty if okk and gate_id(n) not in ROWLESS_OK]
    say(f'[2/6] rows=[] · data={{}} — 통과 {sum(1 for _, o, _ in empty if o)}/{len(empty)} '
        f'(허용 {list(ROWLESS_OK)}) · ⛔공허한 통과 {len(leaked)}')
    for n in leaked:
        say(f'      ⛔공허한 통과: {n}')
    if leaked:
        fails.append(f'0행에서 통과한 게이트: {leaked}')

    # [3/6] 변형이 그 게이트를 실제로 FAIL 로 뒤집는가 --------------------------
    perts = _perturbations()
    n_flip = 0
    for gid, desc, mut in perts:
        r2 = _all_gates(*mut(rows, data, pc1))
        hit = [(n, okk) for n, okk, _ in r2 if gate_id(n) == gid]
        if not hit:
            fails.append(f'{gid}: 그 이름의 게이트가 없다 — {desc}')
            say(f'      ⛔{gid} 게이트 없음 — {desc}')
        elif all(okk for _, okk in hit):
            fails.append(f'{gid}: 변형에도 통과 — {desc}')
            say(f'      ⛔{gid} 발화 안 함 — {desc}')
        else:
            n_flip += 1
    say(f'[3/6] 변형 {len(perts)}건 중 해당 게이트가 발화한 것 **{n_flip}**')

    # [4/6] 런 0회 게이트 변형 (전역 스왑) -------------------------------------
    if runfree:
        g = globals()
        rperts = _runfree_perturbations()
        n_rflip = 0
        for gid, desc, patch in rperts:
            old = {k: g[k] for k in patch}
            g.update(patch)
            try:
                r2 = gates_runfree()
            finally:
                g.update(old)
            hit = [(n, okk) for n, okk, _ in r2 if gate_id(n) == gid]
            if not hit:
                fails.append(f'{gid}: 런 0회 게이트가 없다 — {desc}')
                say(f'      ⛔{gid} 게이트 없음 — {desc}')
            elif all(okk for _, okk in hit):
                fails.append(f'{gid}: 변형에도 통과 — {desc}')
                say(f'      ⛔{gid} 발화 안 함 — {desc}')
            else:
                n_rflip += 1
        say(f'[4/6] 런 0회 변형 {len(rperts)}건 중 발화 **{n_rflip}**')

    # [5/6] 라벨 도달성 + READING 키 -------------------------------------------
    seen = _label_reachability()
    for v in seen.values():
        say(f'      · {v}')
    need = {'HOLDS (CONFIRMATORY)', 'HOLDS (NOMINAL)', 'UNDECIDED', 'BREAKS',
            'BASELINE-NO-SKILL'}
    say(f'[5/6] `corner_label` 이 낸 라벨 {sorted(seen)}')
    if not need <= set(seen):
        fails.append(f'도달 불가 라벨: {sorted(need - set(seen))}')
    missing = sorted(need - set(READING))
    if missing:
        fails.append(f'READING 에 항목이 없는 라벨(분석기가 KeyError 로 죽는다): {missing}')
    dec_missing = [k for k in DECOMPOSITION if len(k) != 3]
    if dec_missing:
        fails.append(f'DECOMPOSITION 키 형태 오류: {dec_missing}')

    # [6/6] 문턱 산술 ----------------------------------------------------------
    t = threshold_table()
    say(f'[6/6] 문턱(n={t["n"]}): CONFIRM ≥{CONFIRM} p={t["p_confirm"]:.6f} ≤ '
        f'가족 α {t["family_alpha"]} · NOMINAL ≥{NOMINAL} p={t["p_nominal"]:.6f} · '
        f'BREAKS ≤{WEAK} p={t["p_weak_le"]:.6f} · 음성대조 ≥{NEG_MIN}')
    if not (t['confirm_within_family'] and t['weak_is_sibling']):
        fails.append('문턱 산술 위반')

    ok_all = not fails
    say('✅자가진단 통과 — 게이트가 발화한다' if ok_all
        else f'⛔자가진단 실패 {len(fails)}건: {fails}')
    return ok_all, log


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        sys.exit(0 if selftest()[0] else 1)
    print('=== B-26 런 0회 게이트 ===')
    _rf = gates_runfree()
    n_ok = 0
    for name, ok, msg in _rf:
        print(f'  [{"PASS" if ok else "FAIL"}] {name}: {msg}')
        n_ok += bool(ok)
    print(f'\n문턱표: {threshold_table()}')
    print(f'소모 시드: {SEEDS_CONSUMED[0]}–{SEEDS_CONSUMED[-1]} ({len(SEEDS_CONSUMED)}개)')
    print(f'에피소드 {N_EP} · 코너당 {N_PER_CORNER} · 런 {N_EP*len(EST_SEEDS)*2}')
    print('\n=== ★게이트 자가진단 (가짜 행/자료로 게이트가 발화하는지) ===')
    ok_st, _ = selftest(runfree=False)          # 위에서 이미 런 0회 게이트를 돌렸다
    sys.exit(0 if (ok_st and n_ok == len(_rf)) else 1)
