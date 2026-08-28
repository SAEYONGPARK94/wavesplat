"""M3 단일출처 — **학습 지지집합 밖을 (Hs,Tp)가 아닌 좌표에서** (2026-08-28). ⛔결과 0행에서 동결.

★이 파일이 표본·팔·문턱·게이트·라벨·**결과 전 해석 규칙**의 **유일한 정의**다.
러너(`run_m3.py`)·생성기(`gen_m3.py`)·분석기(`analyze_m3.py`)는 전부 여기서만 읽는다.
설계 산문·좌표 표·"밖" 정의의 산술 도출은 `PREREG.md` — 상수는 **여기**가 원본이다.

## 질문
***"배포 사전은 (Hs,Tp) 말고 다른 좌표에서도 학습 지지집합 밖에서 버는가?"***
ML 심사의 남은 사유 ③ = *"체크포인트 = 평가 분포"*. `B-26` 은 (Hs,Tp) 두 좌표에서만
밖에 나갔고(그마저 **사후** 확인), 구조 좌표는 전부 in-distribution 이었다.
⟹ 학습이 무작위화한/고정한 좌표 중 평가가 **중심에 고정**해 둔 축 셋을 골라
**결합 밀도 0 인 값**으로 옮긴다: **평균 방향 · 총 방향폭 · 수심**.

## 셀 3개 (BF4 고정 — (Hs,Tp) 축은 B-26 이 답했다; 여기서는 붙박는다)
학습 지지집합 (`tmp/mixed_retrain_2026-07-26/gen_mixed.py`, G-4 가 소스에서 재도출):
  · 평균 방향  dird ~ U[−50°, +50°]        (:279)   — 평가 고정값 0° (중심)
  · 총 방향폭  ~ U[12°, 40°]               (:301)   — 평가 고정값 30° (내부)
  · 수심       = 50.0 점질량               (:54)    — 평가 고정값 50 (동일)
    ★수심은 조건자에도 하드코딩: `scripts/d3_beamform_net.py:30 DEPTH=50.0`,
    `:39` 의 ω-격자가 그 값으로 굳는다 ⟹ 수심 이동은 **출하된 사전 그 자체**(전단부 포함)의 시험.
| 셀 | direction_deg | spread_deg | depth | p_train | 뜻 |
|---|---|---|---|---|---|
| `DIR180` | **180** | 30 | 50 | **0** (경계 밖 130°) | 추종파 기하 — ★PRIMARY |
| `SPR5`   | 0 | **5**  | 50 | **0** (하한 밖 7°)  | 장파정(swell 형) 바다 |
| `DEP12`  | 0 | 30 | **12** | **0** (점질량 밖)   | 천수 — 조건자 ω-격자가 틀린다 |

## ⛔이 배치가 **주장하지 않는 것** (결과 전에 적는다)
- ⛔**실측 음성을 답하지 않는다.** 남은 후보 다섯(관측 기하·창 길이·정상성·스테레오 오차·
  사이트 구성)은 이 배치 뒤에도 **전부 미측정**이다. 수심은 "사이트 구성"과 **인접**할 뿐이다.
- ⛔**(Hs,Tp)·비선형·희소성 축을 다시 답하지 않는다** (B-26 · B-11/B-12 · B-19).
- ⛔**GT 는 선형** — "산다" 문장은 선형 GT 를 명시해 인용한다(B-26 규약 계승).
- ⛔**셀 간 강도 순위·크기 비교 금지**(B-26 F1 교훈: 척도 셋이 순위를 다르게 낸다) ·
  ⛔**세 셀 합산 통계 0개**(등록 자체를 안 한다) · ⛔`B-1`/`B-12`/`B-16`/`B-26` 크기와 직접 비교 금지.
- ⛔**γ 축은 도달 불가** — `run_episode` 가 gamma 를 안 넘긴다(`wavesim/pipeline.py:119-121`,
  jonswap 기본 3.3). wavesim 무수정 규율 아래 **시험 불가**로 보고만 한다(평가 3.3 은 어차피
  학습 U[1.5,4.0] 안이라 "밖" 셀도 못 만든다).

## ⚠사전 노출 공시 (B-25 공시 계승 의무 — 결과 전에 적는다)
- `S15_DIRECTION_2026-08-14` 스모크가 dir180 을 **2짝** 돌렸다(전량 실행은 없었음).
  MEMORY 요약: *"4짝 중 1짝은 `obs:0_5` +7.7% 로 SHIP 이 진다, direction_deg=180 에서만"*.
  ⟹ ★PRIMARY 방향의 사전 증거가 **혼재(약한 역풍)** 로 노출돼 있다 — 이 배치는 blind 가 아니다.
- `DEBATE_blankslate_2026-08-12`: 제어 프리뷰 기하가 dir180 에서 회생한다는 관찰(다른 질문).
- `B-26` 탐색 노트: (Hs,Tp) 지지집합 밖 두 코너에서 벌었다는 **사후** 관찰 + 지지집합 **안**
  `SS-LOWH` 가 최약이라는 반례. ⟹ 어느 결과든 **`SS-LOWH` 반례를 병기**한다(§READING).
- 러너 검증 스모크 1셀(규율 의무)의 수치가 러너 확인 중 노출된다 — `SMOKE.md` 에 기록하고
  판정에 쓰지 않는다.
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
B26 = os.path.join(ROOT, 'tmp', 'B26_SEASTATE_2026-08-27')
S16 = os.path.join(ROOT, 'tmp', 'S16_BASELINE_2026-08-14')
for _p in (HERE, B12, B11, B26, S16, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── 팔·집계 함수는 B-11/B-12 에서 import (중복 구현 금지) ─────────────────────
from gates_b11 import (EST_SEEDS, FILTER_SEEDS, SPEC,          # noqa: F401,E402
                       sign_test_one_sided, med_by_episode, paired)

DATA = os.path.join(HERE, 'data_m3')
SMOKE = os.path.join(HERE, 'smoke_m3')

# ── 생성 상수: B-26 과 같은 계보(gen_s8 규약). ★손복사가 아니라 원본에서 되읽는다 ──
MESH_NX, MESH_NY = 200, 167          # gen_s8.py:40
SRC_EPISODE = os.path.join(ROOT, 'data_finemesh_2026-07-23',
                           'episode_BF4_15comp_seed0_uniform.h5')
RAYDROP = 'uniform'
SEA_STATE = 'BF4'                    # (Hs,Tp) 붙박이 — wavesim SEA_STATE 원본에서 읽는다
# ★PC-1 (생성기 비트동일 양성대조) 대상 — B-26 G-16b 와 같은 동결 에피소드
PC1_PATH = os.path.join(ROOT, 'data_s8_crowded_2026-08-10',
                        'episode_BF4_15comp_seed32_uniform_ra_rf_ds.h5')
PC1 = dict(n_components=15, seed=32)
PC1_DATASETS = ('gt_wave_field', 'lidar_points', 'lidar_valid_mask')
# ★PC-2 (러너 비트동일 양성대조) 대상 — 동결 B-26 SS-BF4 행 (좌표 전부 배포값인 칸)
PC2_EPISODES = ('SS-BF4_15comp_seed1641_uniform_ra_rf_ds',
                'SS-BF4_30comp_seed1641_uniform_ra_rf_ds')
PC2_SEED = 42                        # est_seed == filter_seed == 42 행만 재현한다
B26_DATA = os.path.join(B26, 'data_b26')
B26_RESULTS_GLOB = os.path.join(B26, 'results_b26_*.jsonl')


def gen_params():
    """duration·dt·n_pts·기준 수심을 **동결 원본 에피소드에서 읽는다**(손복사 0)."""
    import h5py
    with h5py.File(SRC_EPISODE, 'r') as f:
        depth = float(f['config'].attrs['depth'])
        n_pts = int(f['config/lidar_config'].attrs.get('n_pts', 0)) or 50
        ts = f['frames/timestamps'][:]
    return dict(duration=float(ts[-1] - ts[0]), dt=float(ts[1] - ts[0]),
                depth=depth, n_pts=n_pts, n_frames=len(ts))


def bf4_seastate():
    from wavesim.pipeline import SEA_STATE as SS
    return float(SS[SEA_STATE]['Hs']), float(SS[SEA_STATE]['Tp'])


# ── 셀 정의 (⛔빈 dict 금지 — 세 좌표를 전부 명시한다) ─────────────────────────
DEPLOY_COORD = dict(direction_deg=0.0, spread_deg=30.0, depth=50.0)  # run_episode 기본값
CELLS = {
    'DIR180': dict(direction_deg=180.0, spread_deg=30.0, depth=50.0),
    'SPR5':   dict(direction_deg=0.0,   spread_deg=5.0,  depth=50.0),
    'DEP12':  dict(direction_deg=0.0,   spread_deg=30.0, depth=12.0),
}
CELL_ORDER = ('DIR180', 'SPR5', 'DEP12')
PRIMARY_CELL = 'DIR180'
NCOMPS = (15, 30)                    # ⛔혼잡 고정 — 희소성 축이 아니다 (B-19)
SEED_POOL = tuple(range(1695, 1758))  # 원장 [1592–1818] 중, B-26 선언 풀(–1680) 뒤
# ★AMENDMENT_1 (2026-08-28, 결과 0행): 시작을 1690→1695 로 — **1681–1694 는 M1 이 소모**
#   (`tmp/M1_NONLIN_REAL_2026-08-28/gates_m1.py:218 SEEDS_CONSUMED`). 종전 풀은 {1690..1694}
#   가 겹쳐 어느 쪽이 먼저 생성하든 **양쪽 G-0 이 FAIL** 했다. 결과와 무관한 원장 사실이고
#   선언 풀 [1690,1757] 안이다. 사유·영향범위 = `AMENDMENT_1.md`.
K_SEEDS = 8
N_EP = len(CELLS) * len(NCOMPS) * K_SEEDS            # 48
N_PER_CELL = len(NCOMPS) * K_SEEDS                   # 16

# ── 문턱: ★코드가 계산한다 (n=16, 단측 부호검정; B-26 규약) ───────────────────
N_FAMILY = 3                          # 셀 3개 = 주장 가족
ALPHA = 0.05
FAMILY_ALPHA = ALPHA / N_FAMILY       # 0.016667
NOMINAL, CONFIRM, WEAK = 12, 13, 4    # /16  (0.03841 · 0.01064 ≤ 0.01667 · n-NOMINAL)
NEG_MIN = 12                          # /16  SPEC25 < ZERO 최소 (음성 대조)
SZ_MIN = 12                           # /16  SHIP < ZERO 최소 (무기술 대비 기술)
MESH_BAD = 'INADEQUATE'


def threshold_table(n=N_PER_CELL):
    """문턱 **재도출** — 사전등록 숫자와 대조한다(G-19)."""
    return {'n': n,
            'p_nominal': sign_test_one_sided(NOMINAL, n),
            'p_confirm': sign_test_one_sided(CONFIRM, n),
            'p_weak_le': sum(comb(n, k) for k in range(0, WEAK + 1)) / 2 ** n,
            'family_alpha': FAMILY_ALPHA,
            'confirm_within_family': sign_test_one_sided(CONFIRM, n) <= FAMILY_ALPHA,
            'weak_is_sibling': WEAK == n - NOMINAL}


# ── "밖" 의 산술 도출 — ★소스에서 재도출한다 (몬테카를로 불필요: 지지집합이 경계 명시형) ──
MIXED = os.path.join(ROOT, 'tmp', 'mixed_retrain_2026-07-26', 'gen_mixed.py')
BFNET = os.path.join(ROOT, 'scripts', 'd3_beamform_net.py')
TRAIN_DIR_SUPPORT = (-50.0, 50.0)     # gen_mixed.py:279  rng.uniform(-50, 50)
TRAIN_SPREAD_SUPPORT = (12.0, 40.0)   # gen_mixed.py:301  rng.uniform(max(12.0, ...), 40.0)
TRAIN_DEPTH = 50.0                    # gen_mixed.py:54   DEPTH = 50.0
BFNET_DEPTH = 50.0                    # d3_beamform_net.py:30  (조건자 ω-격자 하드코딩)


def train_support_facts():
    """gen_mixed.py · d3_beamform_net.py 소스에서 지지집합 경계를 **regex 로 되읽는다**."""
    src = open(MIXED).read()
    m_dir = re.search(r'dird\s*=\s*float\(rng\.uniform\(([-\d.]+),\s*([-\d.]+)\)\)', src)
    m_spr = re.search(r'total_spread\s*=\s*float\(rng\.uniform\(max\(([\d.]+),'
                      r'\s*sig_sub\s*\+\s*[\d.]+\),\s*([\d.]+)\)\)', src)
    m_dep = re.search(r'^DEPTH\s*=\s*([\d.]+)', src, re.M)
    bsrc = open(BFNET).read()
    m_bf = re.search(r'DEPTH\s*=\s*([\d.]+)', bsrc)
    m_om = re.search(r'om\s*=\s*torch\.sqrt\(G\s*\*\s*k\s*\*\s*torch\.tanh\(k\s*\*\s*DEPTH\)\)',
                     bsrc)
    dir_s = (float(m_dir.group(1)), float(m_dir.group(2))) if m_dir else None
    spr_s = (float(m_spr.group(1)), float(m_spr.group(2))) if m_spr else None
    dep = float(m_dep.group(1)) if m_dep else None
    bf_dep = float(m_bf.group(1)) if m_bf else None
    # 평가 기본값 = run_episode 시그니처에서 되읽는다
    import inspect
    from wavesim.pipeline import run_episode
    sig = inspect.signature(run_episode)
    ev = dict(direction_deg=sig.parameters['direction_deg'].default,
              spread_deg=sig.parameters['direction_spread_deg'].default,
              depth=sig.parameters['depth'].default)
    return dict(dir_support=dir_s, spread_support=spr_s, train_depth=dep,
                bfnet_depth=bf_dep, bfnet_om_uses_depth_const=bool(m_om),
                eval_defaults=ev)


def support_ok(facts):
    """(a) 상수가 동결 기대와 같다 (b) 평가 기본값은 **안** (c) 셀 값 셋은 **밖**."""
    if not all([facts['dir_support'], facts['spread_support'],
                facts['train_depth'], facts['bfnet_depth']]):
        return False, '소스 재도출 실패'
    ok = (facts['dir_support'] == TRAIN_DIR_SUPPORT
          and facts['spread_support'] == TRAIN_SPREAD_SUPPORT
          and facts['train_depth'] == TRAIN_DEPTH
          and facts['bfnet_depth'] == BFNET_DEPTH
          and facts['bfnet_om_uses_depth_const'])
    ev = facts['eval_defaults']
    lo, hi = facts['dir_support']
    slo, shi = facts['spread_support']
    ok &= (lo <= ev['direction_deg'] <= hi and slo <= ev['spread_deg'] <= shi
           and ev['depth'] == facts['train_depth'])                  # 평가 = 안
    c = CELLS
    ok &= not (lo <= c['DIR180']['direction_deg'] <= hi)             # 처치 = 밖
    ok &= not (slo <= c['SPR5']['spread_deg'] <= shi)
    ok &= c['DEP12']['depth'] != facts['train_depth']
    return bool(ok), (f"dir{facts['dir_support']} spr{facts['spread_support']} "
                      f"dep{facts['train_depth']}/{facts['bfnet_depth']} eval{ev}")


# ── 표본: 결과 0행에서 · 런 0회로 확정 (mesh/Q 는 jonswap 의 순함수) ───────────
def components(cell, ncomp, seed):
    """★run_episode 가 그릴 것과 **정확히 같은** draw (stream_tag 규약 포함)."""
    from wavesim.wave_generator import jonswap_components
    c = CELLS[cell]
    Hs, Tp = bf4_seastate()
    return jonswap_components(Hs=Hs, Tp=Tp, n_components=ncomp, depth=c['depth'],
                              seed=seed, direction_deg=c['direction_deg'],
                              direction_spread_deg=c['spread_deg'],
                              random_amplitudes=True, random_frequencies=True,
                              stream_tag=f'{SEA_STATE}_{ncomp}')


def _cfg():
    from wavesim.config import WaveSimConfig
    cfg = WaveSimConfig()
    cfg.mesh.nx, cfg.mesh.ny = MESH_NX, MESH_NY
    cfg.hos.enabled = False                       # ⛔선형 GT (§주장하지 않는 것)
    return cfg


def screen(cell, ncomp, seed):
    """생성 전 선별 — **GT 무관·결과 무관**."""
    from wavesim import mesh_adequacy
    from wavesim.wave_generator import check_q_constraint
    comp = components(cell, ncomp, seed)
    q_ok, q_val = check_q_constraint(comp)
    info = mesh_adequacy.check(comp, _cfg(), hos_order=None)
    return dict(mesh_verdict=info['mesh_verdict'],
                var_under=float(info['var_under_resolved']),
                lambda_min=float(info['lambda_min']),
                lambda_peak=float(info['lambda_peak']),
                q_ok=bool(q_ok), q_val=float(q_val))


def fname(cell, ncomp, seed):
    """run_episode 의 실제 파일명 (dtag 는 방향≠0 에서만 붙는다 — 셀 디렉터리로 구분)."""
    d = CELLS[cell]['direction_deg']
    dtag = '' if abs(d) < 1e-9 else f'_dir{int(d)}'
    return f'episode_{SEA_STATE}_{ncomp}comp_seed{seed}_{RAYDROP}{dtag}_ra_rf_ds.h5'


def cell_dir(cell, base=None):
    return os.path.join(base or DATA, cell)


def select():
    """동결 선택 규칙: (셀, ncomp) 마다 풀을 순서대로 걸어 mesh/Q 통과 앞의 8개."""
    eps, rej = [], []
    for cell in CELL_ORDER:
        for ncomp in NCOMPS:
            got = 0
            for seed in SEED_POOL:
                s = screen(cell, ncomp, seed)
                if s['mesh_verdict'] == MESH_BAD or not s['q_ok']:
                    rej.append(dict(cell=cell, ncomp=ncomp, seed=seed, **s))
                    continue
                eps.append(dict(cell=cell, ncomp=ncomp, seed=seed,
                                episode=fname(cell, ncomp, seed)[len('episode_'):-3]
                                + f'@{cell}',
                                path=os.path.join(cell_dir(cell), fname(cell, ncomp, seed)),
                                **CELLS[cell], **s))
                got += 1
                if got == K_SEEDS:
                    break
            assert got == K_SEEDS, f'{cell} {ncomp}comp: 풀 부족 ({got}/{K_SEEDS})'
    return eps, rej


# ★동결 기대값 — `select()` 를 결과 0행에서 돌려 얻었다. G-1 이 매 실행 재대조한다.
# ★AMENDMENT_1 (2026-08-28, 결과 0행): 1690–1697 → **1695–1702** (풀 시작 이동의 귀결;
#   `freeze` 재실행 산출 그대로 — 거부 0건, 6칸 공유는 종전과 같은 모양).
FROZEN_SEEDS = {
    'DIR180|15': tuple(range(1695, 1703)), 'DIR180|30': tuple(range(1695, 1703)),
    'SPR5|15': tuple(range(1695, 1703)), 'SPR5|30': tuple(range(1695, 1703)),
    'DEP12|15': tuple(range(1695, 1703)), 'DEP12|30': tuple(range(1695, 1703)),
}
FROZEN_REJECTED = ()
SEEDS_CONSUMED = tuple(range(1695, 1703))


# ── 배포 상수 되읽기 (B-26 G-20 규약 그대로) ─────────────────────────────────
def deploy_constants():
    from wavesplat.run import DEFAULT_CONFIG
    from wavesplat.optimizer import sequential_update as SU
    return {'DEFAULT_CONFIG.varpro_steps': DEFAULT_CONFIG['varpro_steps'],
            'DEFAULT_CONFIG.learned_init_iters': DEFAULT_CONFIG['learned_init_iters'],
            'DEFAULT_CONFIG.auto_switch_thresh': DEFAULT_CONFIG['auto_switch_thresh'],
            'DEFAULT_CONFIG.lambda_l1': DEFAULT_CONFIG['lambda_l1'],
            'SU.VARPRO_STEPS_DEFAULT': SU.VARPRO_STEPS_DEFAULT,
            'SU.LEARNED_INIT_ITERS_DEFAULT': SU.LEARNED_INIT_ITERS_DEFAULT,
            'SU.AUTO_SWITCH_THRESH_DEFAULT': SU.AUTO_SWITCH_THRESH_DEFAULT,
            'SU.LAMBDA_L1_DEFAULT': SU.LAMBDA_L1_DEFAULT}


DEPLOY_EXPECTED = {'varpro_steps': 400, 'learned_init_iters': 0,
                   'auto_switch_thresh': 0, 'lambda_l1': 1e-4}
ROUTE_EXPECTED = 'learned'

SHIP_CAPTURE = ('sea_state', 'Hs_true', 'Tp_true', 'auto_selected', 'auto_n_selected',
                'auto_real_gate', 'ensemble_k', 'init_iters', 'varpro_steps',
                'varpro_fired', 'n_gaussians', 'avg_nRMSE')
SPEC_CAPTURE = ('Hs', 'Tp', 'depth', 'loc_radius', 'n_ensemble', 'background',
                'filter_class', 'avg_nRMSE', 'zero_nRMSE')

# ── 파생 속성 문턱 (⛔`cfg[k]==cfg[k]` 항등식 금지 — 저장 h5 의 성분에서 잰다) ──
G = 9.81
DIR_COS_NEG = -0.8        # DIR180: 에너지가중 mean cos(θ) ≤ −0.8
DIR_COS_POS = +0.8        # 그 외 셀: ≥ +0.8
SPR5_MAX = 9.5            # SPR5: 실현 방향폭(에너지가중 원형 std, deg) ≤ 9.5
# 공칭 30° 셀 대역 — ★동결 표본의 **정확한 실현값**(성분은 (셀,ncomp,시드)의 순함수라
# 런 0회로 전산 가능). ★AMENDMENT_1 시드 이동(1695–1702) 후 재전산: 30° 셀 [15.86, 51.58]°,
# SPR5 [2.66, 8.29]° — 대역·문턱은 **무변**이고 여유가 종전(1690–1697 의 [10.69, 55.05]° /
# [1.79, 8.69]°)보다 넓다. 분리(G-7c)는 정확값 8.29 < 15.86 으로 성립한다.
# (종전 표본에서의 정정 이력: (10,55) 는 상한이 실현 최대 55.05 를 0.05° 차로 잘랐다 ⟹ (9.6,60).)
SPR30_BAND = (9.6, 60.0)
DISP_TRUE_MAX = 1e-6      # 참 수심 분산관계 잔차(상대 |ω²−gk·tanh(kd)|/ω²) 최대
DISP_50_MIN = 0.10        # DEP12: d=50 가정 잔차 최소 (실산술 최저빈 ≈1.05)
ZERO_BAND = (0.15, 0.40)  # SPEC zero_nRMSE 셀중앙 대역 + 셀 간 ±10% 불변(G-16 유사)


def _read_ep(path):
    """★`gates_data` 의 **유일한 I/O 지점** — 자가진단은 이 함수만 갈아 끼운다."""
    import h5py
    import numpy as np
    with h5py.File(path, 'r') as f:
        a = dict(f['config'].attrs)
        gt = f['frames/gt_wave_field'][:]
        C = f['config/wave_components'][:]          # [A, omega, kx, ky, phi, Q]
    return dict(Hs=float(a['Hs']), Tp=float(a['Tp']),
                hs4std=4 * float(np.std(gt)), n_frames=int(gt.shape[0]),
                mesh_dx=float(a['mesh_dx']), depth=float(a['depth']),
                spread_attr=float(a['direction_spread_deg']),
                ra=bool(a.get('random_amplitudes')),
                rf=bool(a.get('random_frequencies')),
                ds=bool(a.get('decorrelate_stream')),
                A=C[:, 0].astype(float), omega=C[:, 1].astype(float),
                kx=C[:, 2].astype(float), ky=C[:, 3].astype(float))


def derived_coords(a):
    """저장 성분에서 (방향, 방향폭, 분산 잔차) 를 **잰다**."""
    import numpy as np
    w = a['A'] ** 2
    w = w / w.sum()
    th = np.arctan2(a['ky'], a['kx'])
    z = np.sum(w * np.exp(1j * th))
    mean_cos = float(np.cos(np.angle(z)))
    R = float(np.abs(z))
    circ_std_deg = float(np.degrees(np.sqrt(-2.0 * np.log(max(R, 1e-12)))))
    k = np.hypot(a['kx'], a['ky'])
    r_true = float(np.max(np.abs(a['omega'] ** 2 - G * k * np.tanh(k * a['depth']))
                          / a['omega'] ** 2))
    r_50 = float(np.max(np.abs(a['omega'] ** 2 - G * k * np.tanh(k * 50.0))
                        / a['omega'] ** 2))
    return dict(mean_cos=mean_cos, circ_std_deg=circ_std_deg,
                disp_resid_true=r_true, disp_resid_50=r_50)


# ─────────────────────────────────────────────────────────────────────────────
# 게이트 — ★단일출처. 러너·분석기가 같은 함수를 부른다.
# ─────────────────────────────────────────────────────────────────────────────
ROWLESS_OK = ('G-11', 'G-18')         # 0행에서 참이어도 되는 게이트 (B-26 규약)


def gates_runfree():
    out = []
    eps, rej = select()
    got = {f'{c}|{n}': tuple(e['seed'] for e in eps
                             if e['cell'] == c and e['ncomp'] == n)
           for c in CELL_ORDER for n in NCOMPS}
    out.append(('G-1 표본 재도출 = 동결 목록', got == FROZEN_SEEDS,
                f'{len(eps)}/{N_EP} 에피소드'))
    out.append(('G-1b 셀당 16', all(len(v) == K_SEEDS for v in got.values()),
                str(sorted({len(v) for v in got.values()}))))
    out.append(('G-2 선택분 INADEQUATE 0 · 거부목록 일치',
                all(e['mesh_verdict'] != MESH_BAD for e in eps)
                and tuple((r['cell'], r['ncomp'], r['seed']) for r in rej) == FROZEN_REJECTED,
                f'거부 {[(r["cell"], r["ncomp"], r["seed"]) for r in rej]}'))
    out.append(('G-3 Q 제약 48/48', all(e['q_ok'] for e in eps),
                f'{sum(e["q_ok"] for e in eps)}/{len(eps)}'))
    # G-0 시드 원장 충돌 (⛔이 배치 디렉터리 제외 — B-26 규약)
    used = set()
    _mine = os.path.realpath(HERE) + os.sep
    for p in glob.glob(os.path.join(ROOT, '..', '**', 'episode_*seed*.h5'), recursive=True):
        if os.path.realpath(p).startswith(_mine):
            continue
        m = re.search(r'seed(\d+)_', os.path.basename(p))
        if m:
            used.add(int(m.group(1)))
    out.append(('G-0 시드 원장 충돌 0 (B-26/B-12 겹침 금지 포함)',
                not (set(SEEDS_CONSUMED) & used),
                f'충돌 {sorted(set(SEEDS_CONSUMED) & used)}'))
    # ★G-4 지지집합 경계 재도출 (평가=안 · 처치=밖 을 소스에서 증명)
    try:
        ok, msg = support_ok(train_support_facts())
        out.append(('★G-4 지지집합: 학습 경계 소스 재도출 · 평가=안 · 처치 3셀=밖', ok, msg[:150]))
    except Exception as e:                                   # noqa: BLE001
        out.append(('★G-4 지지집합 재도출', False, repr(e)[:80]))
    t = threshold_table()
    out.append(('★G-19 문턱 산술', t['confirm_within_family'] and t['weak_is_sibling']
                and t['p_nominal'] < ALPHA,
                f'NOMINAL {NOMINAL} p={t["p_nominal"]:.5f} · CONFIRM {CONFIRM} '
                f'p={t["p_confirm"]:.5f} ≤ {FAMILY_ALPHA:.5f} · WEAK {WEAK}=n-NOMINAL'))
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


MESH_DX_EXPECTED = 1.2060301507537687


def gates_data(read=None, exists=None):
    """생성 후·추정 전 — 저장 h5 의 **성분에서 좌표를 되잰다** (개입의 파생 측정)."""
    read = read or _read_ep
    exists = exists or os.path.exists
    out = []
    eps, _ = select()
    have = [e for e in eps if exists(e['path'])]
    out.append(('G-6 에피소드 48개 생성됨', len(have) == N_EP, f'{len(have)}/{N_EP}'))
    if not have:
        return out
    meta_ok, nfr, rel = True, set(), []
    dcs = {c: [] for c in CELL_ORDER}       # mean_cos
    sps = {c: [] for c in CELL_ORDER}       # circ_std_deg
    dtr = {c: [] for c in CELL_ORDER}       # disp_resid_true
    d50 = {c: [] for c in CELL_ORDER}       # disp_resid_50
    Hs_bf4, Tp_bf4 = bf4_seastate()
    gp = gen_params()
    for e in have:
        a = read(e['path'])
        d = derived_coords(a)
        dcs[e['cell']].append(d['mean_cos'])
        sps[e['cell']].append(d['circ_std_deg'])
        dtr[e['cell']].append(d['disp_resid_true'])
        d50[e['cell']].append(d['disp_resid_50'])
        rel.append(a['hs4std'] / Hs_bf4)
        nfr.add(a['n_frames'])
        meta_ok &= (abs(a['mesh_dx'] - MESH_DX_EXPECTED) < 1e-9
                    and a['ra'] and a['rf'] and a['ds']
                    and abs(a['Hs'] - Hs_bf4) < 1e-9 and abs(a['Tp'] - Tp_bf4) < 1e-9
                    and abs(a['depth'] - CELLS[e['cell']]['depth']) < 1e-9
                    and abs(a['spread_attr'] - CELLS[e['cell']]['spread_deg']) < 1e-9)
    out.append(('★G-7a 방향 파생측정: DIR180 mean_cos ≤ −0.8 · 그 외 ≥ +0.8',
                bool(dcs['DIR180']) and max(dcs['DIR180']) <= DIR_COS_NEG
                and all(min(dcs[c]) >= DIR_COS_POS for c in ('SPR5', 'DEP12') if dcs[c]),
                f"DIR180 max {max(dcs['DIR180']):.3f} · others min "
                f"{min(min(dcs[c]) for c in ('SPR5', 'DEP12') if dcs[c]):.3f}"
                if dcs['DIR180'] else ''))
    out.append(('★G-7b 방향폭 파생측정: SPR5 ≤ 9.5° · 공칭30° 셀 ∈ [10,55]°',
                bool(sps['SPR5']) and max(sps['SPR5']) <= SPR5_MAX
                and all(SPR30_BAND[0] <= v <= SPR30_BAND[1]
                        for c in ('DIR180', 'DEP12') for v in sps[c]),
                f"SPR5 max {max(sps['SPR5']):.2f}° · 30° 대역 "
                f"[{min(v for c in ('DIR180', 'DEP12') for v in sps[c]):.1f},"
                f"{max(v for c in ('DIR180', 'DEP12') for v in sps[c]):.1f}]°"
                if sps['SPR5'] else ''))
    out.append(('★G-7c **실현 방향폭 분리**: max(SPR5) < min(공칭30°) (겹침 0)',
                bool(sps['SPR5']) and bool(sps['DIR180'])
                and max(sps['SPR5']) < min(v for c in ('DIR180', 'DEP12') for v in sps[c]),
                ''))
    out.append(('★G-7d 분산관계 파생측정: 참 수심 잔차 ≤ 1e-6 전건 · DEP12 의 d=50 잔차 ≥ 0.10',
                all(max(dtr[c]) <= DISP_TRUE_MAX for c in CELL_ORDER if dtr[c])
                and bool(d50['DEP12']) and min(d50['DEP12']) >= DISP_50_MIN
                and all(max(d50[c]) <= DISP_TRUE_MAX for c in ('DIR180', 'SPR5') if d50[c]),
                f"true max {max(max(dtr[c]) for c in CELL_ORDER if dtr[c]):.2e} · "
                f"DEP12 d50 min {min(d50['DEP12']):.3f}" if d50['DEP12'] else ''))
    out.append(('★G-8 에피소드별 |4·std/Hs − 1| ≤ 0.25 (B-26 실측 여유 계승)',
                bool(rel) and max(abs(r - 1) for r in rel) <= 0.25,
                f'max {max(abs(r-1) for r in rel):.4f}' if rel else ''))
    out.append(('G-9 메타 되읽기(mesh 200×167 · ra/rf/ds · BF4 (Hs,Tp) · 셀별 depth/spread attr)',
                meta_ok, ''))
    out.append(('G-10 프레임 수 단일', len(nfr) == 1 and nfr == {gp['n_frames']}, str(nfr)))
    return out


def gates_run(rows):
    out = []
    ok = [r for r in rows if 'error' not in r]
    out.append(('G-11 에러 0행', len(rows) == len(ok), f'{len(rows)-len(ok)}'))
    sh = [r for r in ok if r.get('arm') == 'SHIP-NOW']
    sp = [r for r in ok if r.get('arm') == 'SPEC25']
    out.append(('G-12 SHIP 144행', len(sh) == N_EP * len(EST_SEEDS), f'{len(sh)}/{N_EP*3}'))
    out.append(('G-12b SPEC25 144행', len(sp) == N_EP * len(FILTER_SEEDS), f'{len(sp)}/{N_EP*3}'))
    es, ep_ = {r['episode'] for r in sh}, {r['episode'] for r in sp}
    out.append(('★G-13 두 팔 에피소드 집합 완전 일치', es == ep_ and len(es) == N_EP,
                f'{len(es)}∩{len(ep_)}={len(es & ep_)}'))
    out.append(('★G-14 SHIP 배포값 되읽기(400·0·0·1e-4)',
                bool(sh) and all(r.get('varpro_steps') == DEPLOY_EXPECTED['varpro_steps']
                                 and r.get('learned_init_iters') == DEPLOY_EXPECTED['learned_init_iters']
                                 and r.get('auto_switch_thresh') == DEPLOY_EXPECTED['auto_switch_thresh']
                                 and abs((r.get('lambda_l1') or -1) - DEPLOY_EXPECTED['lambda_l1']) < 1e-12
                                 for r in sh),
                str(sorted({(r.get('varpro_steps'), r.get('learned_init_iters'),
                             r.get('auto_switch_thresh')) for r in sh}))[:70]))
    out.append(('★G-15 라우터가 전 셀에서 학습 경로 (배포 사전이 실제로 시험됐다)',
                bool(sh) and all(r.get('cap_auto_selected') == ROUTE_EXPECTED for r in sh),
                str(sorted({(r.get('cell'), r.get('cap_auto_selected')) for r in sh}))[:120]))
    out.append(('★G-15b VARPRO 발화 전건',
                bool(sh) and all(bool(r.get('cap_varpro_fired')) for r in sh),
                f'{sum(bool(r.get("cap_varpro_fired")) for r in sh)}/{len(sh)}'))
    out.append(('★G-16 SPEC25 되읽기(loc 25 · ens 600 · jonswap)',
                bool(sp) and all(abs(r.get('loc_radius', -1) - SPEC['loc']) < 1e-9
                                 and r.get('n_ensemble') == SPEC['n_ensemble']
                                 and r.get('background') == SPEC['background'] for r in sp), ''))
    Hs_bf4, Tp_bf4 = bf4_seastate()
    out.append(('★G-17 기준선 배경 (Hs,Tp)=BF4 · depth=셀 참값 (비교자 불리 반박 차단)',
                bool(sp) and all(abs(r.get('cap_Hs', -1) - Hs_bf4) < 1e-6
                                 and abs(r.get('cap_Tp', -1) - Tp_bf4) < 1e-6
                                 and abs(r.get('cap_depth', -1) - CELLS[r['cell']]['depth']) < 1e-6
                                 for r in sp),
                str(sorted({(r.get('cell'), round(r.get('cap_Hs', -1), 2),
                             round(r.get('cap_depth', -1), 1)) for r in sp}))[:140]))
    out.append(('★G-18 계측 shim 투명 (잡은 avg_nRMSE == 행의 avg_nRMSE, 비트)',
                all(r.get('cap_avg_nRMSE') == r.get('avg_nRMSE') for r in ok),
                f'{sum(r.get("cap_avg_nRMSE") == r.get("avg_nRMSE") for r in ok)}/{len(ok)}'))
    zero = [r for r in sp if r.get('zero_nRMSE') is not None]
    out.append(('G-18b ZERO 48 에피소드', len({r['episode'] for r in zero}) == N_EP,
                f'{len({r["episode"] for r in zero})}'))
    # ★G-21 ZERO 셀 간 불변 (채점 드리프트 검출 — Hs 가 붙박이이므로 흔들리면 안 된다)
    import statistics as st
    zmed = {c: st.median([r['zero_nRMSE'] for r in zero if r.get('cell') == c])
            for c in CELL_ORDER if any(r.get('cell') == c for r in zero)}
    if len(zmed) == len(CELL_ORDER):
        zok = (all(ZERO_BAND[0] <= v <= ZERO_BAND[1] for v in zmed.values())
               and max(zmed.values()) / min(zmed.values()) < 1.10)
        out.append(('★G-21 ZERO 셀중앙 대역 + 셀 간 ±10% 불변', zok,
                    str({k: round(v, 4) for k, v in zmed.items()})))
    else:
        out.append(('★G-21 ZERO 셀중앙', False, f'셀 {len(zmed)}/{len(CELL_ORDER)}'))
    return out


def gates_pc1(pc1):
    return [('★G-22 PC-1 생성기 비트동일 (좌표=배포값 ⟹ 동결 B-1 코퍼스 재생성)',
             bool(pc1) and all(pc1.get('identical', {}).get(k) for k in PC1_DATASETS),
             str(pc1.get('identical')) + ' ' + str(pc1.get('maxabs', ''))[:60])]


def gates_pc2(pc2):
    """pc2 = run_m3.pc2() 산출: {'rows': [...], 'frozen': {...}} — 비트 대조는 여기서."""
    if not pc2 or not pc2.get('rows'):
        return [('★G-23 PC-2 러너 비트동일 (동결 B-26 SS-BF4 행 재현)', False, '미실행')]
    bad = []
    for r in pc2['rows']:
        key = (r['arm'], r['episode'])
        fz = pc2['frozen'].get('|'.join(key))
        if fz is None or r['avg_nRMSE'] != fz:
            bad.append((key, r.get('avg_nRMSE'), fz))
    return [('★G-23 PC-2 러너 비트동일 (동결 B-26 SS-BF4 행 재현)',
             not bad and len(pc2['rows']) == 2 * len(PC2_EPISODES),
             f'불일치 {bad[:2]}' if bad else f'{len(pc2["rows"])}행 전건 비트동일')]


# ── 집계 · 라벨 · ★결과 전 해석 규칙 ─────────────────────────────────────────
def aggregate(rows):
    ok = [r for r in rows if 'error' not in r]
    out = {}
    for cell in CELL_ORDER:
        sub = [r for r in ok if r.get('cell') == cell]
        ship = med_by_episode(sub, 'SHIP-NOW')
        spec = med_by_episode(sub, 'SPEC25')
        zero = med_by_episode(sub, 'SPEC25', field='zero_nRMSE')
        out[cell] = dict(ship=ship, spec=spec, zero=zero,
                         prim=paired(ship, spec), neg=paired(spec, zero),
                         sz=paired(ship, zero))
    return out


def cell_label(prim, neg, sz):
    """셀 하나의 라벨. ⛔음성대조가 죽으면 우열을 말하지 않는다 (B-26 규약 + SZ 분기)."""
    if neg['n'] == 0 or neg['better'] < NEG_MIN:
        if sz['n'] and sz['better'] >= SZ_MIN:
            return ('BASELINE-NO-SKILL/SHIP-EARNS',
                    f'기준선 무기술 {neg["better"]}/{neg["n"]}, SHIP<ZERO {sz["better"]}/{sz["n"]}')
        return ('NO-SKILL-CELL',
                f'기준선 {neg["better"]}/{neg["n"]} · SHIP<ZERO {sz["better"]}/{sz["n"]} — 전원 무기술')
    w, n = prim['better'], prim['n']
    if w >= CONFIRM:
        return ('HOLDS (CONFIRMATORY)', f'{w}/{n} p={prim["p_one_sided"]:.6f}')
    if w >= NOMINAL:
        return ('HOLDS (NOMINAL)', f'{w}/{n} p={prim["p_one_sided"]:.6f} — 가족단위 미달')
    if w <= WEAK:
        return ('BREAKS', f'{w}/{n} — 부호가 반대로 간다')
    return ('UNDECIDED', f'{w}/{n} — 부호로 갈리지 않는다')


READING = {
    'HOLDS (CONFIRMATORY)': (
        '★그 좌표의 학습 지지집합 **밖**(결합 밀도 0)에서도 배포 사전이 기준선을 이긴다. '
        '세 셀 전부면: *"경계는 평균 방향·방향폭·수심 축을 따라 놓여 있지 않다"* (⛔배제까지 — '
        '지목 아님). ML 심사 사유 ③은 **이 세 구조 좌표에 대해** 실질적으로 닫힌다 — '
        '⛔"전부 닫힌다" 금지: 관측 기하·창 길이·정상성·스테레오 오차·사이트 구성은 남는다. '
        '⚠의무 병기 2건: (a) 선형 GT (b) `SS-LOWH` 반례 — 지지집합 **안**인 칸이 최약이었다 '
        '⟹ *"지지집합 소속이 성능을 예측한다"* 는 이 결과로도 못 쓴다.'),
    'HOLDS (NOMINAL)': (
        '⚠명목만 통과 ⟹ **`EXPLORATORY`**. 위 문장을 쓰되 가족단위 미달을 병기한다.'),
    'BREAKS': (
        '★★강한 **양성** 결과 — ***경계가 학습 지지집합에 붙는다*** (그 좌표에서). '
        '§`sec:boundary` 를 *"자료 영역"* 에서 ***"학습 지지집합"*** 으로 다시 쓰고, '
        '`B-23` 재학습이 비로소 **처방으로 승격**된다(B-26 이 죽인 우선순위가 여기서 되살아난다). '
        '⚠셀별 기제 주의: `DEP12` 의 BREAKS 는 조건자 상수(`d3_beamform_net.py:30`)라는 '
        '**근접 기제**가 있다 — 처방이 재학습이 아니라 **수심 인자화**일 수 있다. 이를 '
        '가르는 후속(조건자만 참 수심으로 재계산)은 **새 사전등록**으로만. '
        '`DIR180`/`SPR5` 의 BREAKS 는 체크포인트의 학습 분포를 직접 지목한다. '
        '⚠`SS-LOWH` 반례 병기 의무는 동일하다.'),
    'UNDECIDED': (
        '그 좌표의 지지집합-밖 수익은 n=16 으로 갈리지 않았다 — **이 라벨도 결과다**: '
        'ML 사유 ③은 그 좌표에서 미결로 남고, MDE(참 승률 0.80 에서 검정력 ≈0.80)를 공시한다. '
        '⛔결과를 보고 n 을 늘리지 않는다(B-8b 정지규칙) — 증축은 새 사전등록으로만.'),
    'BASELINE-NO-SKILL/SHIP-EARNS': (
        '⛔비교자가 무기술 ⟹ 우열 주장 금지. 보고는 `SHIP<ZERO` 로 한정: '
        '*"사전은 그 좌표 밖에서도 무기술 대비 기술을 유지한다(기준선 비교 불가)"*.'),
    'NO-SKILL-CELL': (
        '전원 무기술 — 그 셀은 지지집합 질문에 **답하지 못한다**(레짐 자체가 관측 불능일 수 '
        '있다). 지지집합 귀속 금지. 후속은 새 사전등록으로만.'),
}

# 3셀 분해 읽기 — 라벨 세 개만 읽는다 (추가 검정 0개; 08-23 층간대비 교훈 계승)
DECOMPOSITION = {
    ('hold', 'hold', 'hold'): '세 구조 좌표 전부에서 유지 — 사유 ③에 대한 배제가 세 축 넓어진다.',
    ('break', 'hold', 'hold'): '★방향 축에서만 경계 — 학습 방향 지지집합이 경계 후보로 승격.',
    ('hold', 'break', 'hold'): '★방향폭 축에서만 경계 — 장파정 바다가 경계 후보로 승격.',
    ('hold', 'hold', 'break'): '★수심 축에서만 경계 — 조건자 상수 기제 우선 검증(새 등록).',
    ('break', 'break', 'break'): '★★경계가 지지집합 전반에 붙는다 — B-23 이 1순위 처방.',
}


def decompose(labels):
    def st_(c):
        L = labels[c]
        return 'hold' if L.startswith('HOLDS') else (
            'break' if L == 'BREAKS' else 'undecided')
    key = tuple(st_(c) for c in CELL_ORDER)
    return key, DECOMPOSITION.get(
        key, '조합이 표에 없다 ⟹ 단일 축 귀속 금지, 세 라벨을 그대로 보고한다.')


# ─────────────────────────────────────────────────────────────────────────────
# ★런 전 자가진단 + 변형 하네스 — 게이트가 **실제로 발화하는지** (B-26 규약)
# ⛔파일 0개 · 결과 0개 · 실측값 0개. 픽스처 상수는 임의(0.25·0.60·1.15).
# ─────────────────────────────────────────────────────────────────────────────
FIX_SPEC_BASE = 0.25
FIX_SHIP_MULT = 0.60
FIX_ZERO_MULT = 1.15


def _jit(key):
    return 1.0 + 0.05 * ((hash(key) % 1000) / 1000.0 - 0.5)


def _fixture_rows():
    rows = []
    eps, _ = select()
    for e in eps:
        spec = FIX_SPEC_BASE * _jit(e['episode'])
        for s in EST_SEEDS:
            v = spec * FIX_SHIP_MULT * _jit((e['episode'], s))
            rows.append(dict(arm='SHIP-NOW', episode=e['episode'], cell=e['cell'],
                             est_seed=s, avg_nRMSE=v, cap_avg_nRMSE=v,
                             varpro_steps=400, learned_init_iters=0,
                             auto_switch_thresh=0, lambda_l1=1e-4,
                             cap_auto_selected='learned', cap_varpro_fired=True))
        for s in FILTER_SEEDS:
            rows.append(dict(arm='SPEC25', episode=e['episode'], cell=e['cell'],
                             filter_seed=s, avg_nRMSE=spec, cap_avg_nRMSE=spec,
                             zero_nRMSE=spec * FIX_ZERO_MULT,
                             loc_radius=25.0, n_ensemble=600, background='jonswap',
                             cap_Hs=1.2, cap_Tp=7.0,
                             cap_depth=CELLS[e['cell']]['depth']))
    return rows


def _fixture_read(cell_map):
    """cell_map: {cell: override dict} — 파생좌표 픽스처를 만든다."""
    import numpy as np

    def read(path):
        cell = [c for c in CELL_ORDER if os.sep + c + os.sep in path][0]
        o = dict(CELLS[cell])
        o.update(cell_map.get(cell, {}))
        rng = np.random.default_rng(abs(hash(path)) % 2 ** 31)
        n = 15
        # 표본 std 를 0.7·공칭으로 **표준화** — 픽스처가 대역 밖 꼬리를 뽑아 헛발화하는
        # 것을 막는다(실현 대역은 위 SPR30_BAND 주석의 정확값이 담당한다).
        z0 = rng.normal(0.0, 1.0, n)
        z0 = (z0 - z0.mean()) / max(z0.std(), 1e-9)
        th = np.deg2rad(o['direction_deg'] + z0 * 0.7 * o['spread_deg'])
        om = np.linspace(0.45, 2.7, n)
        from wavesim.wave_generator import wavenumber_from_omega
        k = np.array([wavenumber_from_omega(w, o['depth']) for w in om])
        A = rng.rayleigh(0.1, n)
        Hs, Tp = bf4_seastate()
        return dict(Hs=Hs, Tp=Tp, hs4std=Hs * (1 + rng.uniform(-0.15, 0.15)),
                    n_frames=gen_params()['n_frames'], mesh_dx=MESH_DX_EXPECTED,
                    depth=o['depth'], spread_attr=o['spread_deg'],
                    ra=True, rf=True, ds=True,
                    A=A, omega=om, kx=k * np.cos(th), ky=k * np.sin(th))
    return read


def gate_id(name):
    return name.lstrip('★ ').split()[0]


def selftest(verbose=True):
    """(ok, 보고행들). 변형마다 **정확히 그 게이트**가 죽는지 본다."""
    rep, ok_all = [], True

    def check(desc, cond):
        nonlocal ok_all
        rep.append(('PASS' if cond else 'FAIL', desc))
        ok_all &= bool(cond)

    # 1) 0행 공허 통과 검출
    zero_g = gates_run([])
    bad = [gate_id(n) for n, q, _ in zero_g if q and gate_id(n) not in ROWLESS_OK]
    check(f'0행에서 ROWLESS_OK 밖 통과 게이트 0개 (실제 {bad})', not bad)

    # 2) 픽스처 전건 통과
    rows = _fixture_rows()
    g = gates_run(rows)
    fails = [gate_id(n) for n, q, _ in g if not q]
    check(f'픽스처 run 게이트 전건 PASS (실패 {fails})', not fails)

    # 3) 변형 → 표적 게이트만 죽는다
    import copy
    muts = [
        ('G-12', lambda r: r[1:]),
        ('G-14', lambda r: [{**x, 'varpro_steps': 0} if i == 0 else x
                            for i, x in enumerate(r)]),
        ('G-15', lambda r: [{**x, 'cap_auto_selected': 'v2_konly'}
                            if x['arm'] == 'SHIP-NOW' and i < 3 else x
                            for i, x in enumerate(r)]),
        ('G-15b', lambda r: [{**x, 'cap_varpro_fired': False}
                             if x['arm'] == 'SHIP-NOW' and i < 3 else x
                             for i, x in enumerate(r)]),
        ('G-16', lambda r: [{**x, 'loc_radius': 50.0} if x['arm'] == 'SPEC25' else x
                            for x in r]),
        ('G-17', lambda r: [{**x, 'cap_depth': 50.0}
                            if x['arm'] == 'SPEC25' and x['cell'] == 'DEP12' else x
                            for x in r]),
        ('G-18', lambda r: [{**x, 'cap_avg_nRMSE': (x['avg_nRMSE'] or 0) + 1e-12}
                            if x['arm'] == 'SHIP-NOW' and i < 3 else x
                            for i, x in enumerate(r)]),
        ('G-18b', lambda r: [{**x, 'zero_nRMSE': None}
                             if x['arm'] == 'SPEC25' and x['episode'].endswith('@DIR180')
                             else x for x in r]),
        ('G-21', lambda r: [{**x, 'zero_nRMSE': (x.get('zero_nRMSE') or 0) * 2}
                            if x['arm'] == 'SPEC25' and x['cell'] == 'DEP12' else x
                            for x in r]),
        ('G-11', lambda r: r + [dict(arm='SHIP-NOW', episode='x', cell='DIR180',
                                     error='boom')]),
    ]
    for target, mut in muts:
        mg = gates_run(mut(copy.deepcopy(rows)))
        died = {gate_id(n) for n, q, _ in mg if not q}
        check(f'변형[{target}] → {target} 발화 (죽은 것 {sorted(died)})', target in died)

    # 4) 라벨 규칙 발화 — 픽스처(SHIP 우세)는 HOLDS, 반전은 BREAKS, 균형은 UNDECIDED
    agg = aggregate(rows)
    lab, _ = cell_label(agg['DIR180']['prim'], agg['DIR180']['neg'], agg['DIR180']['sz'])
    check(f'픽스처 라벨 = HOLDS (실제 {lab})', lab.startswith('HOLDS'))
    inv = [{**x, 'avg_nRMSE': (x['avg_nRMSE'] / FIX_SHIP_MULT ** 2
                               if x['arm'] == 'SHIP-NOW' else x['avg_nRMSE']),
            'cap_avg_nRMSE': (x['cap_avg_nRMSE'] / FIX_SHIP_MULT ** 2
                              if x['arm'] == 'SHIP-NOW' else x['cap_avg_nRMSE'])}
           for x in _fixture_rows()]
    agg2 = aggregate(inv)
    lab2, _ = cell_label(agg2['DIR180']['prim'], agg2['DIR180']['neg'], agg2['DIR180']['sz'])
    check(f'반전 픽스처 라벨 = BREAKS (실제 {lab2})', lab2 == 'BREAKS')
    ns = [{**x, 'zero_nRMSE': (x['zero_nRMSE'] * 0.5 if x.get('zero_nRMSE') else None)}
          for x in _fixture_rows()]
    agg3 = aggregate(ns)
    lab3, _ = cell_label(agg3['DIR180']['prim'], agg3['DIR180']['neg'], agg3['DIR180']['sz'])
    check(f'ZERO 강화 픽스처 라벨 = NO-SKILL 계열 (실제 {lab3})',
          lab3 in ('NO-SKILL-CELL', 'BASELINE-NO-SKILL/SHIP-EARNS'))

    # 5) 데이터 게이트: 올바른 픽스처 통과 + 좌표 변형이 표적 발화
    ok_read = _fixture_read({})
    gd = gates_data(read=ok_read, exists=lambda p: True)
    fails = [gate_id(n) for n, q, _ in gd if not q]
    check(f'데이터 픽스처 전건 PASS (실패 {fails})', not fails)
    for target, mutmap in [('G-7a', {'DIR180': dict(direction_deg=0.0)}),
                           ('G-7b', {'SPR5': dict(spread_deg=30.0)}),
                           ('G-7d', {'DEP12': dict(depth=50.0)}),
                           ('G-9', {'DEP12': dict(spread_deg=17.0)})]:
        # G-9 변형: attr 되읽기(spread_attr≠셀값) — G-7b 대역 안 값이라 G-9 만 죽어야 한다
        mg = gates_data(read=_fixture_read(mutmap), exists=lambda p: True)
        died = {gate_id(n) for n, q, _ in mg if not q}
        check(f'데이터 변형[{target}] → {target} 발화 (죽은 것 {sorted(died)})', target in died)

    # 6) PC 게이트 발화
    check('PC-1 빈 dict → FAIL', not gates_pc1({})[0][1])
    check('PC-1 identical 전건 → PASS',
          gates_pc1(dict(identical={k: True for k in PC1_DATASETS}, maxabs={}))[0][1])
    check('PC-2 미실행 → FAIL', not gates_pc2({})[0][1])
    fz = {f'SHIP-NOW|{e}': 0.1 for e in PC2_EPISODES}
    fz.update({f'SPEC25|{e}': 0.2 for e in PC2_EPISODES})
    rows2 = ([dict(arm='SHIP-NOW', episode=e, avg_nRMSE=0.1) for e in PC2_EPISODES]
             + [dict(arm='SPEC25', episode=e, avg_nRMSE=0.2) for e in PC2_EPISODES])
    check('PC-2 비트동일 픽스처 → PASS', gates_pc2(dict(rows=rows2, frozen=fz))[0][1])
    rows3 = [dict(r, avg_nRMSE=r['avg_nRMSE'] + 1e-12) for r in rows2]
    check('PC-2 1e-12 이탈 → FAIL', not gates_pc2(dict(rows=rows3, frozen=fz))[0][1])

    if verbose:
        for s, d in rep:
            print(f'  [{s}] {d}', flush=True)
        n_ok = sum(1 for s, _ in rep if s == 'PASS')
        print(f'  자가진단 {n_ok}/{len(rep)}', flush=True)
    return ok_all, rep


if __name__ == '__main__':
    if sys.argv[1:] and sys.argv[1] == 'freeze':
        eps, rej = select()
        got = {f'{c}|{n}': tuple(e['seed'] for e in eps
                                 if e['cell'] == c and e['ncomp'] == n)
               for c in CELL_ORDER for n in NCOMPS}
        print('FROZEN_SEEDS =', got)
        print('FROZEN_REJECTED =', tuple((r['cell'], r['ncomp'], r['seed']) for r in rej))
        print('SEEDS_CONSUMED =', tuple(sorted({s for v in got.values() for s in v}
                                               | {r['seed'] for r in rej})))
        sys.exit(0)
    if sys.argv[1:] and sys.argv[1] == 'selftest':
        ok, _ = selftest()
        sys.exit(0 if ok else 1)
    for n, q, d in gates_runfree():
        print(f'  [{"PASS" if q else "FAIL"}] {n}: {d}', flush=True)
