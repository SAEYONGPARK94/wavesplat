"""B-22 단일출처 — **학습된 사전(WavePrior)의 절제** (2026-08-27). ⛔결과 0행에서 동결.

## 왜 이 배치인가
★**논문 제목의 주어(「학습된 스펙트럼 사전」)가 합성에서 절제된 적이 한 번도 없다.**
그 주어가 등장하는 유일한 대조는 실측(`B-7`·`B-8`)이고 **거기서는 진다**.
`B-16`/`B-19` 는 *표현*(연속 파수)을 절제했고 통과했다 — 그래서 지금 논문을 지탱하는
절제 증거는 **제목의 주어가 아닌 것**에 붙어 있다. ⟹ **이 배치가 그 칸을 채운다.**

## ★설계 문제 = 교락 4개를 어떻게 가르는가
배포된 두 경로는 **패키지 두 개**이고 넷이 동시에 다르다:

| | 학습 경로(배포) | 해석적 경로(배포) |
|---|---|---|
| 제안원 | 확산 사전(`propose_atoms`) | v2 빔포밍(`initialize_from_frames_v2`) |
| VARPRO | **400 스텝** (`sequential_update.py:913-917`) | **없음** — 호출 지점이 `_initialize_learned` **안에만** 있다 |
| 결합적합 반복 | `learned_init_iters=0` | 코스투파인 **900**(`[200,200,500]`) |
| 앙상블 | k=1 (라우터가 learned 면 앙상블을 안 만든다, `:1924`) | **k=3** median-of-3 |

⛔**그러므로 위원장의 2×2 `{제안원}×{VARPRO}` 는 배포 코드에서 만들 수 없다** — `ANA+VP` 는
패키지에 **존재하지 않는 팔**이고(호출 지점 0개, `grep apply_varpro` = 2행뿐) 만들려면
`wavesplat/**` 를 고쳐야 한다. ⟹ **다른 방식으로 가른다.**

★**해법: 개입을 "라우팅" 이 아니라 "원자 배열" 로 옮긴다.**
`config['learned_atoms']`(dict{kx,ky,energy})는 **이미 노출된 배포 노브**다 —
라우터 자신이 `_auto_select_init`(`:1032`)에서 원자를 그 키에 캐시하고
`learned_init_params`(`learned_init.py:346`)가 그것을 읽는다. ⟹ 그 배열에 **해석적 제안원의
원자**를 넣으면 **파이프라인은 배포 그대로**(VARPRO 400 · iters 0 · 앙상블 k=1 · LS w,φ ·
가지치기/성장 · 추적기)이고 **바뀌는 것은 원자의 출처 하나뿐**이다.
⟹ 교락 4개 중 **셋(VARPRO·반복·앙상블)이 짝짓기로 소거**되고 **제안원만 남는다.**
이것이 `B-16` 이 파수 연속성 하나만 없앤 것과 같은 형태의 절제다.

★그리고 그 팔은 **해석적 제안원에게 불리할 수 있다**(v2 는 원래 900 반복이 뒤따르도록
설계됐다). 그 반론을 닫기 위해 **패키지 수준 대조 `ANA-SHIP`**(=W15 의 `ALL-ANALYTIC`,
`auto_switch_thresh=999999` ⟹ v2_konly 라우팅 + 앙상블 k=3 + 900 반복)를 **같은 배치에서**
같은 에피소드·같은 시드로 돌린다. ⟹ ★**두 수준이 괄호를 이룬다**:
  L1 = 노브 하나(공정한 분리, 비교자에겐 가혹) · L2 = 패키지(비교자에겐 공정, 교락 4개).
**둘 다 통과해야 제목이 산다**(§판정). 어느 쪽도 단독으로는 제목을 사지 못한다.

## ⛔이 배치가 **주장하지 않는 것**
1. ⛔**외부 기준선 비교가 아니다.** `SPEC25`·LETKF 행은 **0개**다. `B-1`·`B-12` 의 문장을
   대체하지도 수정하지도 않는다.
2. ⛔**배포 기본값을 바꾸는 배치가 아니다.** 어떤 결과가 나와도 `auto_switch_thresh` 를
   이 배치로 되돌리지 않는다(그것은 별도 사전등록이 필요한 결정이다).
3. ⛔**실측 판정(`B-7`·`B-8`: 실측에서 학습 경로 무기술)을 뒤집지 않는다.** 층이 다르다.
4. ⚠**체크포인트가 평가와 같은 분포**라는 단서는 그대로다(시드 축만 신선; `B-12` 상속).
5. ⛔`B-16`/`B-19` 의 표현 절제와 **크기를 비교하지 않는다**(다른 개입·다른 팔).

## 되읽기 규약 (`B-19` §3.1 을 그대로 상속 — 함수를 **import** 한다)
설정 dict 되읽기는 `self.config = config`(`sequential_update.py:399`) 때문에 **항등식**이다.
⟹ 증거는 (A) 객체 파생 속성 (B) 결과 dict 계수기 (C) 행동 지문 (D) 라우팅 계수기
(E) 패키지 트립와이어 (F) 환경변수 뿐이고, 설정값은 `cfg_intent` 로만 남긴다.
★**이 배치의 개입(원자 배열)에 대한 (B) 증거는 `auto_n_selected` 다** — 러너가 주입 배열에
`pool_and_select` 를 돌려 낸 기대값과 대조한다(다른 배열이면 다른 수가 나온다).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
B19 = os.path.abspath(os.path.join(HERE, '..', 'B19_KGRID_SPARSE_2026-08-25'))
B12 = os.path.abspath(os.path.join(HERE, '..', 'B12_HOS_EXTEND_2026-08-23'))
B11 = os.path.abspath(os.path.join(HERE, '..', 'B11_HOS_BASELINE_2026-08-23'))
W15 = os.path.abspath(os.path.join(HERE, '..', 'W15_DEPLOY_VERIFY_2026-08-22'))
for _p in (HERE, ROOT, B19, B12, B11, W15):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── 표본·시드·양성대조: `B-12` 에서 import (손복사 0) ─────────────────────────
from gates_b12 import ALL_SEEDS, N_EP                        # noqa: E402
from gates_b11 import HOS, EST_SEEDS                         # noqa: E402
# ── 되읽기·ZERO·비교 함수: `B-19` 에서 import (중복 구현 금지) ────────────────
from gates_b19 import (zero_nrmse, sign_test_one_sided, cmp_pair, cmp_zero,   # noqa: E402
                       effective_from_updater, DERIVED_KEYS, DERIVED_EXPECTED,
                       ENV_PROBE_KEYS, UPDATER_CLASS_EXPECTED, TP_FACTOR,
                       mesh_census, MESH_ATTR)               # noqa: F401
# ── 강제 해석적 임계: `W15` 의 `ALL-ANALYTIC` 정의 그대로 (⛔손복사 금지) ──────
from gates_w15 import ARMS as _W15_ARMS                      # noqa: E402

CROWDED_MIN_NCOMP = 15          # 혼잡 = ncomp >= 15 (`run_b12.crowded_episodes` 의 조건절)
AUTO_SWITCH_THRESH_FORCE_ANALYTIC = int(_W15_ARMS['ALL-ANALYTIC']['auto_switch_thresh'])
VARPRO_OFF = 0                  # ★`pre_varpro_config()['varpro_steps']` 와 같은 값(=0)


def _sig_default(fn, name):
    """함수 **서명의 기본값**을 읽는다 — 리터럴 손복사를 피하는 유일한 단일출처.

    `learned_energy_thresh` 는 이름 붙은 상수가 아니라 두 곳의 `.get(..., 0.03)` 리터럴이다
    (`learned_init.py:349` · `sequential_update.py:1037`). 그 둘이 미러하는 원본이
    `pool_and_select` 의 서명 기본값이므로 여기서 읽는다. ⚠**미러가 어긋나면 이 배치는
    조용히 다른 팔이 된다** ⟹ `G-5e` 가 세 값이 같은지 원본을 열어 검사한다.
    """
    import inspect
    return inspect.signature(fn).parameters[name].default


def shipped_selection_knobs():
    """(energy_thresh, dedup_dk, max_n) — 배포 pooling 기본값을 **서명에서** 읽는다."""
    from wavesplat.optimizer.learned_init import pool_and_select
    return (_sig_default(pool_and_select, 'energy_thresh'),
            _sig_default(pool_and_select, 'dedup_dk'),
            _sig_default(pool_and_select, 'max_n'))


def deploy_constants():
    """배포 상수 — 패키지 단일출처. `B-19` 의 다섯에 이 배치가 쓰는 둘을 더한다."""
    from gates_b19 import deploy_constants as _b19_dc
    from wavesplat.optimizer.sequential_update import DEFAULT_ANALYTIC_ENSEMBLE_K
    out = dict(_b19_dc())
    out['analytic_ensemble_k'] = DEFAULT_ANALYTIC_ENSEMBLE_K
    e_th, ddk, max_n = shipped_selection_knobs()
    out['learned_energy_thresh'] = e_th
    out['learned_dedup_dk'] = ddk
    out['max_n'] = max_n
    return out


# ✅**E. 배포 상수 트립와이어** — *단일출처의 사본이 아니라 **핀***이다.
#    출하 기본값이 나중에 바뀌면 이 배치는 조용히 재라우팅되는 대신 `G-5d` 에서 멈춘다.
DEPLOY_EXPECTED = {'varpro_steps': 400, 'learned_init_iters': 0,
                   'auto_switch_thresh': 0, 'lambda_l1': 1e-4,
                   'init_method': 'auto', 'analytic_ensemble_k': 3,
                   'learned_energy_thresh': 0.03, 'learned_dedup_dk': 0.03,
                   'max_n': 30}

# ─────────────────────────────────────────────────────────────────────────────
# 팔 (8) — ⛔빈 dict 금지. 개입 노브는 **`atoms` (원자 출처) 와 `varpro_steps` 둘뿐**이고
#          나머지 키는 전부 **배포값을 명시**한 것이다(스냅샷 아님).
# ─────────────────────────────────────────────────────────────────────────────
#   atoms=None        → `learned_atoms` 를 넣지 않는다 ⟹ 라우터가 확산 제안을 **인라인**으로 돈다
#   atoms='learned'   → 러너가 `propose_atoms(diffusion_seed=est_seed)` 로 만든 **같은** 원자를 주입
#   atoms='analytic'  → v2 빔포밍 원자 max_n 개 (배포 pooling 규칙 그대로: e_th=0.03)
#   atoms='analytic_n'→ 같은 v2 원자의 **앞 n_L 개**(그리디 접두) + e_th=0 ⟹ **개수 일치**
def _arm_cfg(varpro_steps, thresh, e_th):
    """팔의 설정 **골격**. 문자열은 자리표시자이고 `arm_config()` 가 패키지 값으로 채운다.

    ⛔리터럴을 여기 적지 않는 이유: 출하 기본값이 바뀌면 팔이 조용히 옛 값에 묶인다
    (`gates_b7.py:46` 의 `{}` 사고와 같은 뿌리). ⟹ **자리표시자 + 트립와이어(G-5d)**.
    """
    return {'varpro_steps': varpro_steps,          # 'VP' = 배포값(400) | 0 = 끔
            'auto_switch_thresh': thresh,          # 'TH_L' = 배포값(0) | 999999 = 강제 해석적
            'learned_energy_thresh': e_th,         # 'E_SHIP' = 배포값(0.03) | 0.0 = 개수 일치
            # 아래 넷은 개입이 아니라 **배포값 명시**다(`arm_config` 가 패키지에서 채운다).
            'learned_init_iters': None, 'lambda_l1': None,
            'init_method': None, 'analytic_ensemble_k': None}


ARMS = {
    # ── 배포 경로 그대로 (양성대조 + 배포 기준행) ──────────────────────────────
    'SHIP':        dict(atoms=None,         cfg=_arm_cfg('VP', 'TH_L', 'E_SHIP')),
    # ── L1: 원자 출처만 바뀌는 짝 (VARPRO 켬) ────────────────────────────────
    'LEARN':       dict(atoms='learned',    cfg=_arm_cfg('VP', 'TH_L', 'E_SHIP')),
    'ANA':         dict(atoms='analytic',   cfg=_arm_cfg('VP', 'TH_L', 'E_SHIP')),
    'ANA-N':       dict(atoms='analytic_n', cfg=_arm_cfg('VP', 'TH_L', 0.0)),
    # ── S-2: 같은 짝, VARPRO 끔 (위원장의 2×2 를 **학습 경로 안에서** 만든 것) ──
    'LEARN-NOVP':  dict(atoms='learned',    cfg=_arm_cfg(VARPRO_OFF, 'TH_L', 'E_SHIP')),
    'ANA-NOVP':    dict(atoms='analytic',   cfg=_arm_cfg(VARPRO_OFF, 'TH_L', 'E_SHIP')),
    'ANA-N-NOVP':  dict(atoms='analytic_n', cfg=_arm_cfg(VARPRO_OFF, 'TH_L', 0.0)),
    # ── L2: 패키지 수준 해석적 (W15 `ALL-ANALYTIC` 정의) ──────────────────────
    'ANA-SHIP':    dict(atoms=None,         cfg=_arm_cfg('VP', AUTO_SWITCH_THRESH_FORCE_ANALYTIC,
                                                         'E_SHIP')),
}
ARM_ORDER = ['SHIP', 'LEARN', 'ANA', 'ANA-N',
             'LEARN-NOVP', 'ANA-NOVP', 'ANA-N-NOVP', 'ANA-SHIP']
INJECTED_ARMS = ('LEARN', 'ANA', 'ANA-N', 'LEARN-NOVP', 'ANA-NOVP', 'ANA-N-NOVP')
LEARNED_ROUTE_ARMS = tuple(a for a in ARM_ORDER if a != 'ANA-SHIP')
VP_ON_ARMS = ('SHIP', 'LEARN', 'ANA', 'ANA-N')          # VARPRO 가 **발화해야** 하는 팔
VP_OFF_ARMS = ('LEARN-NOVP', 'ANA-NOVP', 'ANA-N-NOVP')  # 발화하면 안 되는 팔
# ⚠`ANA-SHIP` 은 `varpro_steps=400` 을 **명시하지만 구조적으로 발화하지 않는다**
#   (호출 지점이 `_initialize_learned` 안에만 있다) ⟹ *"설정을 넣었다 ≠ 설정이 먹었다"* 의
#   교과서적 사례라 **결과 전에 적어 두고** `G-5c` 가 `varpro_fired == False` 를 요구한다.

ROUTE_LEARNED = 'learned'
ROUTE_ANALYTIC = 'v2_konly'
ENSEMBLE_K_LEARNED = 1        # 라우터가 learned 면 앙상블을 만들지 않는다(`:1924` 조건절)
ENSEMBLE_K_ANALYTIC = 3       # = DEFAULT_ANALYTIC_ENSEMBLE_K (G-5d 가 패키지와 대조)
INIT_ITERS_LEARNED = 0        # `learned_init_iters=0` ⟹ 결합 경사하강 0 스텝
INIT_ITERS_ANALYTIC = 900     # 코스투파인 [200,200,500] 합 (관측: 배선시험 iters=900)


def arm_config(arm, atoms=None):
    """★팔의 **완성된 설정 dict** — 러너·자가진단이 같은 함수를 부른다(단일출처).

    `_arm_cfg` 의 자리표시자(`'VP'`·`'TH_L'`·`'E_SHIP'`·`None`)를 **패키지 단일출처 값**으로
    채운다 ⟹ 팔 정의는 리터럴이 아니라 *"배포값 + 이 팔이 바꾸는 것"* 으로 읽힌다.
    """
    dep = deploy_constants()
    cfg = dict(ARMS[arm]['cfg'])
    sub = {'VP': dep['varpro_steps'], 'TH_L': dep['auto_switch_thresh'],
           'E_SHIP': dep['learned_energy_thresh']}
    for k, v in list(cfg.items()):
        if isinstance(v, str) and v in sub:
            cfg[k] = sub[v]
        elif v is None:
            cfg[k] = dep[k]
    cfg['learned_dedup_dk'] = dep['learned_dedup_dk']
    cfg['max_n'] = dep['max_n']
    cfg['learned_atoms'] = atoms          # ⛔`None` 도 **명시**한다(키 부재 = 인라인 제안)
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# 원자 제안 — **두 출처**. 러너와 자가진단이 이 함수만 쓴다.
# ─────────────────────────────────────────────────────────────────────────────
def init_context(h5_path):
    """`run_estimation` 이 초기화에 쓰는 것과 **같은 창·같은 인자**를 만든다.

    `run.py:225-241` 축자: `EpisodeDataLoader(path, device=cfg['device'])` →
    `get_init_window(tp_factor=2.0)` → `x_range=(gt_grid_x[0], gt_grid_x[-1])`.
    ⚠**같은 기능이 두 파일에 구현되는 부류**(이 저장소의 반복 결함)를 피할 수 없는 지점이다 —
    `run_estimation` 이 창을 내보내지 않기 때문이다. ⟹ **행에 `tracking_frames` 를 싣고
    `G-8` 이 에피소드 안 24행의 창 일치를 검사**하고, 무엇보다 **`G-10`(양성대조)이
    `SHIP` 을 `B-12` 동결 행과 비트 동일로 묶는다** — 창이 달랐다면 그것이 깨진다.
    """
    from wavesplat.data_loader import EpisodeDataLoader
    ld = EpisodeDataLoader(h5_path, device='cpu')
    frames = ld.get_init_window(tp_factor=TP_FACTOR)
    return (frames, float(ld.depth),
            (float(ld.gt_grid_x[0]), float(ld.gt_grid_x[-1])),
            (float(ld.gt_grid_y[0]), float(ld.gt_grid_y[-1])))


def learned_raw_atoms(h5_path, est_seed):
    """확산 사전의 **원시** 원자 (라우터가 인라인으로 만드는 것과 같은 호출).

    `_auto_select_init:1017-1022` 축자: `propose_atoms(frames, depth, ckpt=DEFAULT_CKPT,
    nsamp=cfg.get('learned_nsamp', 4), diffusion_seed=cfg.get(..., self.seed))`.
    """
    import numpy as np
    from wavesplat.optimizer.learned_init import propose_atoms, DEFAULT_CKPT
    frames, depth, _xr, _yr = init_context(h5_path)
    kx, ky, en = propose_atoms(frames, depth, ckpt=DEFAULT_CKPT, nsamp=4,
                               diffusion_seed=int(est_seed))
    return dict(kx=np.asarray(kx, float).tolist(), ky=np.asarray(ky, float).tolist(),
                energy=np.asarray(en, float).tolist()), 0


def analytic_raw_atoms(h5_path, est_seed, n_request):
    """v2 빔포밍 제안(=배포 해석적 경로의 제안원)의 원자 `n_request` 개.

    `initialize_from_frames_v2` 는 **그리디 잔차 소거**로 peak 를 하나씩 뽑는다 ⟹
    `n_request` 를 키워도 **앞부분은 비트 동일한 접두**다(배선시험에서 30개의 앞 18개가
    18개 요청 결과와 `array_equal` 로 확인됨) ⟹ **한 번 부르고 두 팔이 나눠 쓴다.**

    energy 규약 = `w**2`. 학습 쪽 energy 는 `denorm_atoms` 의 `a = A**2`
    (`learned_init.py:159` 주석 축자) 이므로 **같은 양**이다.

    ⚠**패딩 제거**: peak 가 모자라면 `initializer.py:578-583` 이 **난수 (kx,ky)** 와
    `PAD_W` 진폭으로 채운다. 그것을 그대로 주입하면 해석적 팔에 **난수 원자**를 심는 셈이라
    비교자를 약화시킨다 ⟹ 진폭이 `PAD_W` 와 정확히 같은 항목을 **버리고 개수를 기록**한다.
    (⟹ 이 규칙은 **비교자에게 유리한 방향**이다.)
    """
    import numpy as np
    from wavesplat.optimizer.initializer import initialize_from_frames_v2, PAD_W
    frames, depth, xr, yr = init_context(h5_path)
    kx, ky, w, _phi = initialize_from_frames_v2(frames, int(n_request), depth,
                                                x_range=xr, y_range=yr,
                                                seed=int(est_seed))
    kx = np.asarray(kx, float); ky = np.asarray(ky, float); w = np.asarray(w, float)
    pad = np.isclose(w, np.float32(PAD_W).astype(float), rtol=0, atol=1e-12)
    keep = ~pad
    return (dict(kx=kx[keep].tolist(), ky=ky[keep].tolist(),
                 energy=(w[keep] ** 2).tolist()), int(pad.sum()))


def prefix_atoms(atoms, n):
    """그리디 접두 `n` 개 (⟹ `analytic_n` 팔). 개수가 모자라면 있는 만큼."""
    n = int(n)
    return {k: list(v)[:n] for k, v in atoms.items()}


def pooled_count(atoms, energy_thresh):
    """주입 배열에 **배포 pooling 을 그대로** 돌려 낸 기대 설치 개수.

    ★이것이 개입의 (B)증거다 — 행의 `auto_n_selected` 는 **추정기가 계산한 값**이고
    이 값은 **러너가 주입 배열에서 도출한 기대값**이다. 배열이 안 닿았으면 어긋난다.
    """
    import numpy as np
    from wavesplat.optimizer.learned_init import pool_and_select
    _e, ddk, max_n = shipped_selection_knobs()
    sk, _sy, _se = pool_and_select(np.asarray(atoms['kx'], float),
                                   np.asarray(atoms['ky'], float),
                                   np.asarray(atoms['energy'], float),
                                   energy_thresh=float(energy_thresh),
                                   dedup_dk=float(ddk), max_n=int(max_n))
    return int(len(sk))


def atoms_md5(atoms):
    import hashlib
    import json
    if atoms is None:
        return None
    return hashlib.md5(json.dumps([list(map(float, atoms['kx'])),
                                   list(map(float, atoms['ky'])),
                                   list(map(float, atoms['energy']))],
                                  sort_keys=True).encode()).hexdigest()


# ── 표본 ─────────────────────────────────────────────────────────────────────
def episodes():
    """혼잡 HOS 48 (=`B-12` 표본). 조건절은 `run_b12.crowded_episodes` 의 것을 그대로."""
    import glob
    import re
    out = []
    for p in sorted(glob.glob(os.path.join(HOS, '*.h5'))):
        m = re.match(r'episode_(BF\d)_(\d+)comp_seed(\d+)_(.+)\.h5', os.path.basename(p))
        if m and int(m.group(2)) >= CROWDED_MIN_NCOMP and int(m.group(3)) in ALL_SEEDS:
            out.append(dict(path=p, episode=os.path.basename(p)[len('episode_'):-3],
                            bf=m.group(1), ncomp=int(m.group(2)), gen_seed=int(m.group(3))))
    return out


# ★표본의 비선형성 실현률 — 핀(트립와이어). `B-19` §5.1 규약 상속:
#   ⛔이 표본을 `"HOS M=3 비선형 참값"` 으로 **단일 서술하지 않는다.**
MESH_EXPECTED = {'ADEQUATE': 31, 'DEGRADED': 17}

# ── 양성대조: `B-12`(08-23) 의 동결 `SHIP-NOW` 행 ─────────────────────────────
B12_RESULTS = os.path.join(B12, 'results_b12.jsonl')
POSCTRL_RTOL = 0.0            # 0 = 비트 동일 요구
ZERO_RTOL = 1e-5


def _b12_rows():
    import json
    out = []
    with open(B12_RESULTS) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'error' not in r:
                out.append(r)
    return out


def b12_ship():
    """{(episode, est_seed): avg_nRMSE} — 이 배치의 `SHIP` 이 **비트 동일**해야 한다."""
    return {(r['episode'], r['est_seed']): float(r['avg_nRMSE'])
            for r in _b12_rows() if r.get('arm') == 'SHIP-NOW'}


def b12_zero():
    """{episode: zero_nRMSE} — `B-12` 의 `SPEC25` 행 기록(교차검증용). ⛔우리는 독립 계산한다."""
    return {r['episode']: float(r['zero_nRMSE']) for r in _b12_rows()
            if r.get('arm') == 'SPEC25' and r.get('zero_nRMSE') is not None}


# ── 판정 문턱 (코드가 계산한다) ───────────────────────────────────────────────
ALPHA = 0.05
FAMILY_ALPHA = 0.05 / 2       # 주장 가족 둘: PRIMARY(L1a·L1b·L2 의 **교집합**) · SECONDARY
MIN_N_USED = 44               # 표본 손실이 이보다 크면 SAMPLE-LOSS-UNDECIDED
NEG_INVALIDATING = ('SHIP', 'LEARN')     # 이 팔이 무기술이면 PRIMARY **무효**
NEG_LABEL_SWITCH = ('ANA', 'ANA-N', 'ANA-SHIP')   # 무기술이면 **라벨 수식어**(무효 아님)


def thresholds(n=N_EP):
    """(NOMINAL, CONFIRM, NOT_EARN). 형제 규약: NOT_EARN = n − NOMINAL (`B-12`)."""
    nominal = next((w for w in range(n + 1) if sign_test_one_sided(w, n) <= ALPHA), None)
    confirm = next((w for w in range(n + 1)
                    if sign_test_one_sided(w, n) <= FAMILY_ALPHA), None)
    return nominal, confirm, (n - nominal if nominal is not None else None)


# ─────────────────────────────────────────────────────────────────────────────
# 집계 (★단일출처: 게이트와 분석기가 **같은 함수**를 부른다)
# ─────────────────────────────────────────────────────────────────────────────
def _arm_route(arm):
    return ROUTE_ANALYTIC if arm == 'ANA-SHIP' else ROUTE_LEARNED


def aggregate(rows):
    """median-of-3 집계 + 유효 표본 선정."""
    import statistics as st
    ok = [r for r in rows if 'error' not in r]
    cell = {}
    for r in ok:
        cell.setdefault((r['arm'], r['episode']), []).append(r)
    val = {k: st.median([x['avg_nRMSE'] for x in v]) for k, v in cell.items()}
    eps_all = sorted({e for (_, e) in val})
    good = []
    for e in eps_all:
        rs = [r for r in ok if r['episode'] == e]
        route_ok = all(r.get('auto_selected') == _arm_route(r['arm']) for r in rs)
        inj_ok = all((r.get('atoms_source') is not None) == (r['arm'] in INJECTED_ARMS)
                     for r in rs)
        full = all((a, e) in val for a in ARM_ORDER)
        if route_ok and inj_ok and full:
            good.append(e)
    zero = {r['episode']: r['zero_nRMSE'] for r in ok
            if r['episode'] in good and r.get('zero_nRMSE') is not None}
    return dict(ok=ok, cell=cell, val=val, eps_all=eps_all, good=good,
                zero=zero, n_used=len(good))


def comparisons(val, good):
    """이 배치의 **모든 짝 비교** — 라벨 함수와 분석기가 같은 dict 을 쓴다."""
    return dict(
        L1a=cmp_pair(val, good, 'LEARN', 'ANA'),
        L1b=cmp_pair(val, good, 'LEARN', 'ANA-N'),
        L2=cmp_pair(val, good, 'SHIP', 'ANA-SHIP'),
        S2a=cmp_pair(val, good, 'LEARN-NOVP', 'ANA-NOVP'),
        S2b=cmp_pair(val, good, 'LEARN-NOVP', 'ANA-N-NOVP'),
        # 서술용(주장 아님): VARPRO 가 각 제안원에서 얼마를 하는가
        VP_LEARN=cmp_pair(val, good, 'LEARN', 'LEARN-NOVP'),
        VP_ANA=cmp_pair(val, good, 'ANA-N', 'ANA-N-NOVP'),
        # 서술용: 주입 충실도 (SHIP 대신 LEARN 을 썼을 때)
        SHIP_vs_ANA=cmp_pair(val, good, 'SHIP', 'ANA'),
        SHIP_vs_ANA_N=cmp_pair(val, good, 'SHIP', 'ANA-N'))


# ─────────────────────────────────────────────────────────────────────────────
# 게이트 (★단일출처 — 분석기는 `gates(rows)` 를 **호출만** 한다)
# ⛔**공허한 통과 금지**: 결과 0행에서 참이 되는 게이트를 두지 않는다(W8 `G-4` 부류).
# ─────────────────────────────────────────────────────────────────────────────
ROWLESS_OK = ('G-5d', 'G-5e', 'G-12')      # 행 없이도 판정 가능(패키지·자료만 읽는다)


def _srt(vals):
    return sorted(vals, key=lambda x: (x is None, repr(x)))


def gates(rows, agg=None):
    """`[(이름, 통과여부, 상세)]`. 러너 산출 전체(오류 행 포함)를 받는다."""
    import statistics as st
    if agg is None:
        agg = aggregate(rows)
    ok, cell, val = agg['ok'], agg['cell'], agg['val']
    eps_all, good, zero = agg['eps_all'], agg['good'], agg['zero']
    out = []

    def g(name, cond, detail):
        out.append((name, bool(cond), str(detail)))

    def rows_of(arm):
        return [r for r in ok if r['arm'] == arm]

    g('G-0 오류 0행', bool(rows) and len(ok) == len(rows),
      f'{len(rows) - len(ok)} 행 오류 / {len(rows)} 행')

    g(f'G-1 표본 = {len(ARM_ORDER)}팔 × {N_EP} 에피소드',
      len(val) == len(ARM_ORDER) * N_EP and len(eps_all) == N_EP,
      f'{len(val)}/{len(ARM_ORDER) * N_EP} 셀 · 에피소드 {len(eps_all)}/{N_EP}')

    bad_seed = {k: sorted(x['est_seed'] for x in v) for k, v in cell.items()
                if sorted(x['est_seed'] for x in v) != sorted(EST_SEEDS)}
    g('G-2 셀마다 추정기 시드 3개 전건', bool(cell) and not bad_seed,
      f'{len(cell) - len(bad_seed)}/{len(cell)} 셀 정상'
      + (f' · 이탈 예: {list(bad_seed.items())[:2]}' if bad_seed else ''))

    # ── G-3 개입(원자 주입)이 팔 정의대로 발화했나 ─────────────────────────────
    src = {a: _srt({r.get('atoms_source') for r in rows_of(a)}) for a in ARM_ORDER}
    exp_src = {a: [ARMS[a]['atoms']] for a in ARM_ORDER}
    md5_ok = all(bool(r.get('atoms_md5')) == (r['arm'] in INJECTED_ARMS) for r in ok)
    g('G-3 ★원자 주입 발화 (팔별 출처 · md5 존재)',
      bool(ok) and src == exp_src and md5_ok, f'{src} · md5 규약 {md5_ok}')

    # ── G-4 ★주입이 **먹었다**: 추정기의 `auto_n_selected` == 러너가 배열에서 낸 기대값 ──
    #    ⛔`cfg[k]==cfg[k]` 항등식이 아니다 — 왼쪽은 추정기 객체가, 오른쪽은 러너가 낸다.
    inj = [r for r in ok if r['arm'] in INJECTED_ARMS]
    bad_sel = [(r['arm'], r['episode'], r['est_seed'],
                r.get('auto_n_selected'), r.get('n_pool_expected')) for r in inj
               if r.get('n_pool_expected') is None
               or r.get('auto_n_selected') != r.get('n_pool_expected')]
    g('G-4 ★주입 원자가 실제로 설치됨 (auto_n_selected == 러너 도출 기대값)',
      bool(inj) and not bad_sel,
      f'{len(inj) - len(bad_sel)}/{len(inj)} 행 일치'
      + (f' 예: {bad_sel[:2]}' if bad_sel else ''))

    # ── G-4b ★개수 일치 팔이 실제로 개수를 맞췄나 ─────────────────────────────
    # ⛔초판은 *"한 셀이라도 일치하면 통과"* 였다 — **거의 공허한 게이트**였고 40/48 이
    #   어긋나도 통과했다(런 전 자기검토가 잡았다). ⟹ **셀마다 정확한 기대값**을 요구한다:
    #     기대 = min(n_sel(LEARN), 해석적 제안이 실제로 준 원자 수)
    #   후자가 앞서면 v2 가 그만큼 못 뽑았다는 뜻이므로(패딩 제거) **그 셀 수도 적는다**.
    nsel = {(r['arm'], r['episode'], r['est_seed']): r.get('auto_n_selected') for r in ok}
    navl = {(r['arm'], r['episode'], r['est_seed']): r.get('n_atoms_injected') for r in ok}
    keys = sorted({(e, s) for (_a, e, s) in nsel})
    bad_cnt, short = [], 0
    for k in keys:
        nL, nAN, nAv = nsel.get(('LEARN',) + k), nsel.get(('ANA-N',) + k), navl.get(('ANA',) + k)
        if nL is None or nAN is None or nAv is None:
            bad_cnt.append((k, nL, nAN, nAv))
            continue
        want = min(nL, nAv)
        if nAv < nL:
            short += 1
        if nAN != want:
            bad_cnt.append((k, nL, nAN, want))
    pad = [r.get('n_pad_dropped') or 0 for r in ok if r['arm'] == 'ANA']
    g('G-4b ★ANA-N 원자 수 == min(LEARN 원자 수, 해석적이 실제로 준 수) — 셀마다',
      bool(keys) and not bad_cnt,
      f'이탈 {len(bad_cnt)}/{len(keys)} 셀 · 해석적이 모자란 셀 {short} · '
      f'패딩 제거 총 {sum(pad)} 원자 · '
      f"LEARN n 중앙 {st.median([v for kk, v in nsel.items() if kk[0] == 'LEARN' and v is not None]) if ok else None}"
      f" · ANA n 중앙 {st.median([v for kk, v in nsel.items() if kk[0] == 'ANA' and v is not None]) if ok else None}"
      + (f' 예: {bad_cnt[:2]}' if bad_cnt else ''))

    # ── G-5 객체 파생 속성 (⛔cfg 항등식 아님) ────────────────────────────────
    eff = [r.get('effective') for r in ok]
    have = bool(ok) and all(isinstance(e, dict) and all(k in e for k in DERIVED_KEYS)
                            for e in eff)
    bad_der = sorted({k for r in ok for k, v in DERIVED_EXPECTED.items()
                      if (r.get('effective') or {}).get(k) != v})
    seed_ok = bool(ok) and all((r.get('effective') or {}).get('seed') == r['est_seed']
                               for r in ok)
    cls_ok = bool(ok) and all((r.get('effective') or {}).get('updater_class')
                              == UPDATER_CLASS_EXPECTED for r in ok)
    cap_ok = bool(ok) and all((r.get('effective') or {}).get('n_captured') == 1 for r in ok)
    env_ok = bool(ok) and all(all(v is None for v in
                                  ((r.get('effective') or {}).get('env_probe') or {}).values())
                              for r in ok)
    g('G-5 ★객체 파생 속성 일치 + 시드·클래스·환경',
      have and not bad_der and seed_ok and cls_ok and cap_ok and env_ok,
      f'기대 {DERIVED_EXPECTED} · 이탈 키 {bad_der} · 키존재 {have} · seed {seed_ok} · '
      f'클래스 {cls_ok} · 포착수1 {cap_ok} · 환경변수 미설정 {env_ok}')

    # ── G-5b 행동 지문: 결합 경사하강 반복 수 (팔별로 **다른** 값을 요구한다) ──
    exp_iters = {a: (INIT_ITERS_ANALYTIC if a == 'ANA-SHIP' else INIT_ITERS_LEARNED)
                 for a in ARM_ORDER}
    bad_it = [(r['arm'], r.get('init_loss_n'), r.get('init_iters')) for r in ok
              if r.get('init_loss_n') is None or r.get('init_iters') is None
              or r['init_loss_n'] != exp_iters[r['arm']]
              or r['init_iters'] != exp_iters[r['arm']]]
    g('G-5b 행동 지문 init_loss_n/init_iters == 팔별 기대 (⛔None 통과 아님)',
      bool(ok) and not bad_it,
      f'기대 {exp_iters} · 이탈 {len(bad_it)}/{len(ok)}' + (f' 예: {bad_it[:2]}' if bad_it else ''))

    # ── G-5c VARPRO 실집행 증거 (⛔echo/truthy 아님 — `B-19` §10.5 부류) ──────
    vp_bad = []
    for r in ok:
        want = r['arm'] in VP_ON_ARMS
        fired = bool(r.get('varpro_fired'))
        objs = (r.get('varpro_obj_before'), r.get('varpro_obj_after'))
        if want and (not fired or objs[0] is None or objs[1] is None
                     or r.get('varpro_steps') != DEPLOY_EXPECTED['varpro_steps']):
            vp_bad.append((r['arm'], 'expected-fired', r.get('varpro_steps'), objs))
        if not want and (fired or objs[0] is not None or objs[1] is not None):
            vp_bad.append((r['arm'], 'expected-silent', r.get('varpro_steps'), objs))
    g('G-5c ★VARPRO 실집행/부재 (팔별 · obj 존재로 판정, ⛔echo 아님)',
      bool(ok) and not vp_bad,
      f'발화 기대 {VP_ON_ARMS} · 침묵 기대 {VP_OFF_ARMS}+ANA-SHIP · '
      f'이탈 {len(vp_bad)}/{len(ok)}' + (f' 예: {vp_bad[:2]}' if vp_bad else ''))

    try:
        dep = deploy_constants()
        g('G-5d 배포 상수 트립와이어 (패키지 단일출처 == 이 배치의 핀)',
          dep == DEPLOY_EXPECTED, f'{dep} vs {DEPLOY_EXPECTED}')
    except Exception as exc:                                 # noqa: BLE001
        g('G-5d 배포 상수 트립와이어 (패키지 단일출처 == 이 배치의 핀)', False,
          f'{type(exc).__name__}: {exc}')

    # ── G-5e ★`learned_energy_thresh` 미러 일치 (서명 기본값 vs 두 `.get` 리터럴) ──
    try:
        e_th, ddk, _mx = shipped_selection_knobs()
        import re as _re
        src_li = open(os.path.join(ROOT, 'wavesplat/optimizer/learned_init.py')).read()
        src_su = open(os.path.join(ROOT, 'wavesplat/optimizer/sequential_update.py')).read()
        pat = r"learned_energy_thresh['\"]\s*,\s*([0-9.eE+-]+)\s*\)"
        got = {float(m) for m in _re.findall(pat, src_li) + _re.findall(pat, src_su)}
        g('G-5e ★energy_thresh 미러 일치 (서명 기본값 == 두 곳의 .get 리터럴)',
          bool(got) and got == {float(e_th)}, f'서명 {e_th} · 소스 리터럴 {sorted(got)}')
    except Exception as exc:                                 # noqa: BLE001
        g('G-5e ★energy_thresh 미러 일치', False, f'{type(exc).__name__}: {exc}')

    # ── G-6 라우터·앙상블 (팔별로 **다른** 값을 요구한다) ─────────────────────
    bad_rt = []
    for r in ok:
        want_route = _arm_route(r['arm'])
        want_k = ENSEMBLE_K_ANALYTIC if r['arm'] == 'ANA-SHIP' else ENSEMBLE_K_LEARNED
        emb = r.get('ensemble_member_n')
        if (r.get('auto_selected') != want_route
                or (r.get('effective') or {}).get('init_method') != want_route
                or r.get('ensemble_k') != want_k
                or r.get('auto_real_gate') is not False
                or not isinstance(emb, list) or len(emb) != want_k):
            bad_rt.append((r['arm'], r.get('auto_selected'), r.get('ensemble_k'), emb,
                           r.get('auto_real_gate')))
    g('G-6 ★라우팅·앙상블 팔별 기대 (learned/k=1 vs v2_konly/k=3) · 실측게이트 미발화',
      bool(ok) and not bad_rt,
      f'이탈 {len(bad_rt)}/{len(ok)}' + (f' 예: {bad_rt[:2]}' if bad_rt else ''))

    g('G-7 원자 수 기록 (n_gaussians > 0 — ⛔0·None 통과 아님)',
      bool(ok) and all(isinstance(r.get('n_gaussians'), int)
                       and r['n_gaussians'] > 0 for r in ok),
      f"None {sum(1 for r in ok if r.get('n_gaussians') is None)} · "
      f"0 {sum(1 for r in ok if r.get('n_gaussians') == 0)} / {len(ok)}")

    win = {}
    for r in ok:
        win.setdefault(r['episode'], set()).add(
            (r.get('tracking_frames'), r.get('scored_t_first'), r.get('scored_t_last')))
    bad_win = {e: v for e, v in win.items() if len(v) != 1}
    win_none = [r for r in ok if r.get('tracking_frames') is None
                or r.get('scored_t_first') is None or r.get('scored_t_last') is None]
    g(f'G-8 채점 창이 에피소드 안 {len(ARM_ORDER) * len(EST_SEEDS)}행 전부 동일 '
      '(⛔전 행 None 통과 아님)',
      bool(win) and not bad_win and not win_none,
      f'{len(win) - len(bad_win)}/{len(win)} 일치 · 창 미기록 {len(win_none)}/{len(ok)}')

    zsets = {}
    for r in ok:
        zsets.setdefault(r['episode'], set()).add(r.get('zero_nRMSE'))
    bad_z = {e: v for e, v in zsets.items() if len(v) != 1}
    same_window = bool(ok) and all(r.get('zero_n_frames') == r.get('tracking_frames')
                                   for r in ok)
    none_z = sum(1 for r in ok if r.get('zero_nRMSE') is None)
    g('G-9 ZERO 창 일치 · 에피소드 내 불변 · None 0행',
      bool(zsets) and not bad_z and same_window and none_z == 0,
      f'창 일치 {same_window} · 분산 에피소드 {len(bad_z)} · None {none_z}')

    # ── G-10 ★양성대조: SHIP 이 `B-12` 동결 행과 **비트 동일** ────────────────
    try:
        ref = b12_ship()
    except Exception as exc:                                 # noqa: BLE001
        ref = {}
        g('★G-10 양성대조 기준 파일', False, f'{type(exc).__name__}: {exc}')
    ship = rows_of('SHIP')
    hit = [r for r in ship if (r['episode'], r['est_seed']) in ref]
    bad_pc = [(r['episode'], r['est_seed'], r['avg_nRMSE'], ref[(r['episode'], r['est_seed'])])
              for r in hit
              if abs(r['avg_nRMSE'] - ref[(r['episode'], r['est_seed'])])
              > POSCTRL_RTOL * abs(ref[(r['episode'], r['est_seed'])])]
    g('★G-10 양성대조: SHIP 이 B-12 동결 행과 비트 동일',
      bool(hit) and not bad_pc and len(hit) == len(ship) and bool(ship),
      f'대조 {len(hit)}/{len(ship)} 행 · 불일치 {len(bad_pc)}'
      + (f' 예: {bad_pc[:1]}' if bad_pc else ''))

    # ── G-11 ZERO 교차검증 ────────────────────────────────────────────────────
    try:
        zref = b12_zero()
    except Exception:                                        # noqa: BLE001
        zref = {}
    zp = [(e, s, zref[e]) for e in zsets for s in zsets[e] if e in zref and s is not None]
    z_bad = [(e, s, r) for e, s, r in zp if abs(s - r) > ZERO_RTOL * abs(r)]
    g('★G-11 ZERO 교차검증 (독립 계산 vs B-12 SPEC25 기록)',
      bool(zp) and not z_bad and len(zp) == len(zsets),
      f'{len(zp) - len(z_bad)}/{len(zp)} 일치 (rtol {ZERO_RTOL})'
      + (f' 예: {z_bad[:1]}' if z_bad else ''))

    # ── G-12 ★표본의 비선형성 실현률 = 핀 ────────────────────────────────────
    eps_list = episodes()
    try:
        counts, _per = mesh_census(eps_list)
        g('★G-12 mesh_verdict 분할 = 사전등록 핀 (⛔"HOS M=3" 단일 서술 금지)',
          bool(eps_list) and counts == MESH_EXPECTED and sum(counts.values()) == N_EP,
          f'{counts} vs 핀 {MESH_EXPECTED} (n={sum(counts.values())}/{N_EP})')
    except Exception as exc:                                 # noqa: BLE001
        g('★G-12 mesh_verdict 분할 = 사전등록 핀', False, f'{type(exc).__name__}: {exc}')

    # ── G-13 ★주입 충실도: `LEARN` 대신 `SHIP` 을 써도 PRIMARY 라벨이 같은가 ──
    #    (LEARN 은 확산 제안을 **런 밖에서** 계산해 주입하므로 전역 RNG 소비가 SHIP 과
    #     다르다. 성장 분기가 난수 위상을 뽑으므로 셀에 따라 갈릴 수 있다 ⟹ **재어서 적는다.**)
    same_cells = [e for e in good if ('SHIP', e) in val and ('LEARN', e) in val
                  and val[('SHIP', e)] == val[('LEARN', e)]]
    cmpx = comparisons(val, good)
    n_used = agg['n_used']
    negs = {a: cmp_zero(val, zero, good, a) for a in ARM_ORDER}
    lab_learn = label_primary(cmpx, n_used, negs)[0] if n_used else None
    alt = dict(cmpx)
    alt['L1a'] = cmp_pair(val, good, 'SHIP', 'ANA')
    alt['L1b'] = cmp_pair(val, good, 'SHIP', 'ANA-N')
    lab_ship = label_primary(alt, n_used, negs)[0] if n_used else None
    g('G-13 ★주입 충실도: LEARN↔SHIP 치환에도 PRIMARY 라벨 불변',
      bool(good) and lab_learn is not None and lab_learn == lab_ship,
      f'비트 동일 셀 {len(same_cells)}/{len(good)} · 라벨(LEARN) {lab_learn} · '
      f'라벨(SHIP 치환) {lab_ship}')

    # ── G-14 ★★주입의 **직접 증거**: 런 안에서 확산 제안이 몇 번 돌았나 ──────
    #   주입 팔 0회 · 비주입 팔(SHIP·ANA-SHIP) 1회(라우터가 n_selected 를 계산한다).
    #   ★주입이 실패했다면 라우터가 제안을 돌아 1 이 된다 ⟹ 이 게이트가 잡는다.
    #   ⛔초판은 `init_time_s` 중앙값 비교였는데 **그 값은 라우터의 제안 시간을 포함하지
    #     않는다**(`_initialize_learned` 안에서 t0 를 잡는다) — 1셀 스모크가 0.76 vs 0.75 s
    #     로 반증했다. ⟹ 시간 휴리스틱을 **계수기**로 바꿨다.
    exp_prop = {a: (0 if a in INJECTED_ARMS else 1) for a in ARM_ORDER}
    bad_prop = [(r['arm'], r.get('propose_calls')) for r in ok
                if r.get('propose_calls') != exp_prop[r['arm']]]
    g('G-14 ★★확산 제안 호출 계수 (주입 팔 0회 · 비주입 1회 — 주입의 직접 증거)',
      bool(ok) and not bad_prop,
      f'기대 {exp_prop} · 이탈 {len(bad_prop)}/{len(ok)}'
      + (f' 예: {bad_prop[:2]}' if bad_prop else ''))

    # ── 음성 대조 ─────────────────────────────────────────────────────────────
    # ⛔**연성 음성대조를 `True` 로 통과시키지 않는다** — 자가진단이 잡은 결함이다
    #   (0행에서 여섯 개가 조용히 PASS 했다 = W8 `G-4` 부류의 재현).
    #   ⟹ 연성 게이트의 통과 조건은 *"문턱을 넘었다"* 가 아니라 ***"실제로 측정됐다"*** 이고,
    #      문턱 결과는 **상세와 라벨 수식어**로 간다(§7 의 라벨 전환 규약).
    nominal = thresholds(n_used)[0] if n_used else None
    for a in ARM_ORDER:
        nz = negs[a]
        hard = a in NEG_INVALIDATING
        okz = bool(nominal is not None and nz['n'] and nz['better'] >= nominal)
        measured = bool(nominal is not None and nz['n'] == n_used and n_used)
        g(f'N[{a}] {a} < ZERO ' + (f'≥ {nominal}/{n_used} (미달 ⟹ PRIMARY 무효)' if hard
                                   else f'— **측정 여부**가 통과 조건 (미달 ⟹ 라벨 수식어 COLLAPSE)'),
          okz if hard else measured,
          f"{nz['better']}/{nz['n']} p={nz['p_one_sided']}"
          + ('' if okz else '  ⚠**무기술**' + ('' if hard else ' ⟹ 수식어 COLLAPSE')))
    return out


# ── 라벨 (결과 전 확정) ───────────────────────────────────────────────────────
TITLE_KEEP = ('현행 제목 유지 — 주어 「학습된 스펙트럼 사전」 이 절제로 지탱된다')
TITLE_A = ('제목 (a) 확정 — 주어를 **연속 파수 표현**으로 (B-16·B-19 가 지탱하는 것). '
           '학습 사전은 *속도·단순함 부품* 으로 정직하게 강등한다')


def _band(w, n):
    """('CONFIRM'|'NOMINAL'|'NULL'|'REVERSED')."""
    nominal, confirm, not_earn = thresholds(n)
    if w >= confirm:
        return 'CONFIRM'
    if w >= nominal:
        return 'NOMINAL'
    if w <= not_earn:
        return 'REVERSED'
    return 'NULL'


def label_primary(cmpx, n, negs):
    """★PRIMARY 라벨 — **결과 전 확정**. `L1a`·`L1b`(노브 분리) ∧ `L2`(패키지).

    ⛔교집합 검정(IUT)이다: **셋 다** 통과해야 `PRIOR-EARNS` 다. 어느 하나의 통과로
    제목을 사지 않는다. 각 성분은 가족단위 α 에서 검정한다(IUT 는 다중성 보정이 필요 없으나
    보수적으로 유지한다).
    """
    if n < MIN_N_USED:
        return ('SAMPLE-LOSS-UNDECIDED',
                f'유효 표본 {n} < {MIN_N_USED} ⟹ 문턱 재계산으로 결론을 사지 않는다', None)
    if not all(negs[a]['n'] and negs[a]['better'] >= thresholds(n)[0]
               for a in NEG_INVALIDATING):
        return ('ARM-NO-SKILL-HERE',
                '⛔배포 학습 팔이 이 층에서 참 제로필드를 문턱만큼 이기지 못한다 ⟹ '
                '뺄 기술이 없으므로 절제 비교는 무효다', None)
    b = {k: _band(cmpx[k]['better'], n) for k in ('L1a', 'L1b', 'L2')}
    mod = [a for a in NEG_LABEL_SWITCH
           if negs[a]['n'] and negs[a]['better'] < thresholds(n)[0]]
    sfx = ('' if not mod else
           f'  ⚠수식어 COLLAPSE[{",".join(mod)}] — 그 팔은 참 제로필드조차 못 이긴다 ⟹ '
           '⛔상대 중앙 %를 margin 으로 인용 금지, 그리고 그 비교는 *"두 작동 팔의 차"* 가 '
           '아니라 *"제안원 교체 시 기술의 소멸"* 이다')
    det = ' · '.join(f"{k} {cmpx[k]['better']}/{cmpx[k]['n']}"
                     f"(p={cmpx[k]['p_one_sided']:.4g},{b[k]})" for k in ('L1a', 'L1b', 'L2'))
    if (b['L2'] == 'REVERSED' and b['L1a'] == 'CONFIRM' and b['L1b'] == 'CONFIRM'):
        return ('PRIOR-EARNS-BUT-PACKAGE-LOSES',
                f'⚠★{det} — 사전은 **파이프라인 안에서** 벌지만 **배포 해석적 패키지가 배포 '
                '학습 패키지를 이긴다**. ⟹ 제목의 주어는 사지 못하고(제목 (a)), 그보다 큰 '
                '질문이 열린다: ***배포 기본값(`auto_switch_thresh=0`)이 옳은가.*** '
                '⛔이 배치는 그 결정을 하지 않는다 — 별도 사전등록 배치가 필요하다.' + sfx,
                TITLE_A)
    if 'REVERSED' in b.values():
        rev = [k for k in b if b[k] == 'REVERSED']
        return ('PRIOR-HARMS',
                f'⛔{det} — **해석적 제안원이 이겼다**({rev}). ⟹ ★논문은 (1) 제목 (a) 로 가고 '
                '(2) 기여 목록에서 학습 사전을 *정확도 담지자* 로 적은 문장을 전부 철회하며 '
                '(3) `B-1`·`B-12` 헤드라인의 **담지자 귀속을 재검토**해야 한다'
                '(해석적 팔도 `SPEC25` 를 이기는지 = 별도 사전등록 배치) '
                '(4) 배포 기본값 논의는 이 배치가 하지 않는다.' + sfx, TITLE_A)
    if all(v == 'CONFIRM' for v in b.values()):
        return ('PRIOR-EARNS',
                f'★{det} — 노브 하나(제안원)와 패키지 수준이 **둘 다** 학습 쪽이다.' + sfx,
                TITLE_KEEP)
    if b['L1a'] == 'CONFIRM' and b['L1b'] == 'CONFIRM':
        return ('PRIOR-EARNS-IN-PIPELINE-ONLY',
                f'⚠{det} — 파이프라인 **안**에서는 사전이 담지자지만 **패키지 수준(L2)에서는 '
                '갈리지 않는다** ⟹ 해석적 경로가 900 반복+앙상블로 벌충한다는 뜻이다. '
                '제목의 주어는 사지 못한다.' + sfx, TITLE_A)
    if b['L2'] == 'CONFIRM':
        return ('PIPELINE-EARNS-NOT-PRIOR',
                f'⚠{det} — 배포 패키지는 이기는데 **제안원 교체로는 갈리지 않는다** ⟹ '
                '이득의 담지자는 사전이 아니라 파이프라인(VARPRO·반복·앙상블 구성)이다.'
                + sfx, TITLE_A)
    if any(v == 'NOMINAL' for v in b.values()):
        return ('PRIOR-EARNS-NOMINAL',
                f'⚠{det} — 명목만(가족단위 α={FAMILY_ALPHA} 미달) ⟹ **탐색적**. '
                '명목은 제목을 사지 못한다(결과 전 규칙).' + sfx, TITLE_A)
    return ('PRIOR-NOT-EARNED',
            f'⛔{det} — 부호로 갈리지 않는다 ⟹ ★**제목의 주어가 측정으로 지탱되지 않는다.**'
            + sfx, TITLE_A)


def label_secondary(cmpx, n):
    """S-2: VARPRO 를 끄고도 사전이 버는가 (`S2a` ∧ `S2b`, IUT)."""
    if n < MIN_N_USED:
        return ('SAMPLE-LOSS-UNDECIDED', f'유효 표본 {n} < {MIN_N_USED}')
    b = {k: _band(cmpx[k]['better'], n) for k in ('S2a', 'S2b')}
    det = ' · '.join(f"{k} {cmpx[k]['better']}/{cmpx[k]['n']}"
                     f"(p={cmpx[k]['p_one_sided']:.4g},{b[k]})" for k in ('S2a', 'S2b'))
    if 'REVERSED' in b.values():
        return ('PRIOR-HARMS-WITHOUT-VARPRO', f'⛔{det}')
    if all(v == 'CONFIRM' for v in b.values()):
        return ('PRIOR-EARNS-WITHOUT-VARPRO',
                f'★{det} — 사전의 이득이 VARPRO 에 **의존하지 않는다**')
    if any(v in ('CONFIRM', 'NOMINAL') for v in b.values()):
        return ('PARTIAL-WITHOUT-VARPRO', f'⚠{det} — 부분 통과, 탐색적')
    return ('PRIOR-NEEDS-VARPRO',
            f'⛔{det} — VARPRO 를 끄면 제안원 차이가 사라진다 ⟹ ★이득은 *사전 단독*이 아니라 '
            '**사전+VARPRO 결합**의 것이다(그 문장으로 논문에 적어라)')


# ─────────────────────────────────────────────────────────────────────────────
# ★런 전 게이트 자가진단 — 가짜 행으로 게이트가 **실제로 발화하는지** 확인한다
# ⛔파일을 쓰지 않는다. 추정기를 부르지 않는다(런 0회).
# ⚠공개: 픽스처가 `B-12` 동결 `SHIP` 행을 쓰므로 `N[SHIP]`(=`SHIP<ZERO`) 의 값이 런 전에
#   도출 가능하다. 그것은 새 정보가 아니다(두 동결 배치에서 이미 연역된다). 그래도 적는다.
#   ⛔PRIMARY·SECONDARY 는 픽스처의 **임의 배수**라 아무 정보도 담지 않는다.
# ─────────────────────────────────────────────────────────────────────────────
# ⚠픽스처 배수는 **임의**다(아무 정보도 담지 않는다). 다만 `G-13` 변형이 발화하려면
#   `ANA-SHIP` 이 `ANA`/`ANA-N` 보다 나쁜 자리에 있어야 한다 — 그래야 *"L1 만 뒤집히는"*
#   변형을 만들 수 있다(자가진단이 그 제약을 찾아냈다).
FIXTURE_MULT = {'SHIP': 1.0, 'LEARN': 1.0, 'LEARN-NOVP': 1.06,
                'ANA': 1.25, 'ANA-N': 1.22, 'ANA-NOVP': 1.30, 'ANA-N-NOVP': 1.28,
                'ANA-SHIP': 1.40}


def _fixture_rows():
    ship, zref, eps = b12_ship(), b12_zero(), episodes()
    rows = []
    for e in eps:
        name = e['episode']
        for arm in ARM_ORDER:
            an = ARMS[arm]['atoms']
            analytic_route = (arm == 'ANA-SHIP')
            nsel = 18 if an != 'analytic' else 6
            for s in EST_SEEDS:
                rows.append(dict(
                    arm=arm, est_seed=s, episode=name, bf=e['bf'],
                    ncomp=e['ncomp'], gen_seed=e['gen_seed'],
                    avg_nRMSE=ship[(name, s)] * FIXTURE_MULT[arm],
                    atoms_source=an, atoms_md5=(None if an is None else 'f' * 32),
                    n_atoms_injected=(None if an is None else 40),
                    n_pad_dropped=(0 if an is not None else None),
                    n_pool_expected=(None if an is None else nsel),
                    auto_selected=(ROUTE_ANALYTIC if analytic_route else ROUTE_LEARNED),
                    auto_n_selected=(18 if an != 'analytic' else 6),
                    auto_real_gate=False,
                    ensemble_k=(ENSEMBLE_K_ANALYTIC if analytic_route
                                else ENSEMBLE_K_LEARNED),
                    ensemble_member_n=([19, 19, 19] if analytic_route else [18]),
                    n_gaussians=(19 if analytic_route else 18),
                    init_iters=(INIT_ITERS_ANALYTIC if analytic_route
                                else INIT_ITERS_LEARNED),
                    init_loss_n=(INIT_ITERS_ANALYTIC if analytic_route
                                 else INIT_ITERS_LEARNED),
                    init_time_s=(90.0 if analytic_route else (3.0 if an else 6.5)),
                    propose_calls=(0 if an is not None else 1),
                    varpro_steps=(DEPLOY_EXPECTED['varpro_steps']
                                  if arm in VP_ON_ARMS else VARPRO_OFF),
                    varpro_fired=(arm in VP_ON_ARMS),
                    varpro_obj_before=(1e-2 if arm in VP_ON_ARMS else None),
                    varpro_obj_after=(4e-3 if arm in VP_ON_ARMS else None),
                    tracking_frames=160, scored_t_first=14.1, scored_t_last=30.0,
                    episode_total_frames=301,
                    zero_nRMSE=zref[name], zero_n_frames=160,
                    effective=dict(lambda_l1=1e-4, tracking_mode='w_phi',
                                   tracking_lr=0.005,
                                   init_method=(ROUTE_ANALYTIC if analytic_route
                                                else ROUTE_LEARNED),
                                   real_estimator='bank_mmse', max_steepness=0.15,
                                   seed=s, depth=50.0,
                                   updater_class=UPDATER_CLASS_EXPECTED,
                                   primary_class='SequentialUpdater', n_captured=1,
                                   env_probe={k: None for k in ENV_PROBE_KEYS}),
                    cfg_intent={'⚠': '의도표 — 증거 아님'},
                    git_head='fixture', import_md5={}, wall_s=1.0))
    return rows


def _perturbations():
    """(게이트 접두어, 설명, 변형) — 각 변형은 그 게이트를 **FAIL 로 뒤집어야** 한다."""
    def _mut(fn):
        def apply(rows):
            import copy
            rs = copy.deepcopy(rows)
            fn(rs)
            return rs
        return apply

    def first(rs, arm):
        return next(r for r in rs if r['arm'] == arm)

    return [
        ('G-0', '오류 행 1개', _mut(lambda rs: rs.append(
            dict(arm='ANA', episode=rs[0]['episode'], est_seed=42, error='boom')))),
        ('G-1', '한 셀(3행) 제거', _mut(lambda rs: [rs.remove(r) for r in
                                                 [x for x in rs if x['arm'] == 'ANA-N'][:3]])),
        ('G-2', 'est_seed 를 99 로', _mut(lambda rs: first(rs, 'ANA').__setitem__('est_seed', 99))),
        ('G-3', '주입 팔의 출처를 None 으로',
         _mut(lambda rs: first(rs, 'ANA').__setitem__('atoms_source', None))),
        ('G-3', '주입 팔의 atoms_md5 누락',
         _mut(lambda rs: first(rs, 'LEARN').__setitem__('atoms_md5', None))),
        ('G-4 ', '★주입이 안 먹었다 (auto_n_selected ≠ 기대값)',
         _mut(lambda rs: first(rs, 'ANA').__setitem__('auto_n_selected', 18))),
        ('G-4 ', '★n_pool_expected 가 전 행 None (러너가 안 실었다)',
         _mut(lambda rs: [r.__setitem__('n_pool_expected', None) for r in rs])),
        ('G-4b', 'ANA-N 개수가 LEARN 과 어긋남',
         _mut(lambda rs: [r.__setitem__('auto_n_selected', 5)
                          for r in rs if r['arm'] == 'ANA-N'])),
        ('G-5 ', '파생 lambda_l1 = 1e-3',
         _mut(lambda rs: first(rs, 'SHIP')['effective'].__setitem__('lambda_l1', 1e-3))),
        ('G-5 ', '파생 seed 불일치',
         _mut(lambda rs: first(rs, 'LEARN')['effective'].__setitem__('seed', 7))),
        ('G-5 ', 'S8 환경변수 흔적', _mut(lambda rs: first(rs, 'SHIP')['effective']
                                    ['env_probe'].__setitem__('S8_L1_TRACK_ONLY', '1'))),
        ('G-5 ', '포착된 updater 0개', _mut(lambda rs: first(rs, 'SHIP')['effective']
                                       .__setitem__('n_captured', 0))),
        ('G-5b', 'init_loss_n 전 행 None',
         _mut(lambda rs: [r.__setitem__('init_loss_n', None) for r in rs])),
        ('G-5b', '★ANA-SHIP 이 900 이 아니라 0 (코스투파인이 안 돌았다)',
         _mut(lambda rs: first(rs, 'ANA-SHIP').__setitem__('init_loss_n', 0))),
        ('G-5c', '★VARPRO 건너뜀 (echo/truthy 유지, obj 만 없음)',
         _mut(lambda rs: first(rs, 'LEARN').update(varpro_obj_before=None,
                                                   varpro_obj_after=None))),
        ('G-5c', '★−VP 팔에서 VARPRO 가 발화',
         _mut(lambda rs: first(rs, 'LEARN-NOVP').update(varpro_fired=True,
                                                        varpro_obj_before=1e-2,
                                                        varpro_obj_after=5e-3))),
        ('G-6', '라우터가 v2_konly 로 샘',
         _mut(lambda rs: first(rs, 'LEARN').__setitem__('auto_selected', ROUTE_ANALYTIC))),
        ('G-6', '★ANA-SHIP 이 앙상블을 못 받았다 (k=1)',
         _mut(lambda rs: first(rs, 'ANA-SHIP').update(ensemble_k=1, ensemble_member_n=[19]))),
        ('G-6', '실측 게이트 발화(auto_real_gate=True)',
         _mut(lambda rs: first(rs, 'SHIP').__setitem__('auto_real_gate', True))),
        ('G-7', 'n_gaussians = 0', _mut(lambda rs: first(rs, 'ANA').__setitem__('n_gaussians', 0))),
        ('G-8', '채점 창 3키가 전 행 None',
         _mut(lambda rs: [r.update(tracking_frames=None, scored_t_first=None,
                                   scored_t_last=None) for r in rs])),
        ('G-8', '한 행의 창이 다름',
         _mut(lambda rs: first(rs, 'ANA-SHIP').__setitem__('tracking_frames', 120))),
        ('G-9', 'zero_nRMSE 하나를 None 으로',
         _mut(lambda rs: first(rs, 'ANA-N').__setitem__('zero_nRMSE', None))),
        ('★G-10', 'SHIP 한 행을 1e-12 만큼 어긋나게',
         _mut(lambda rs: first(rs, 'SHIP').__setitem__(
             'avg_nRMSE', first(rs, 'SHIP')['avg_nRMSE'] + 1e-12))),
        ('★G-11', 'zero 를 rtol 밖으로',
         _mut(lambda rs: [r.__setitem__('zero_nRMSE', r['zero_nRMSE'] * 1.001)
                          for r in rs if r['episode'] == rs[0]['episode']])),
        # ⚠배수는 **1.30** 이다: `ANA`(1.25) 보다는 나쁘고 참 제로필드보다는 좋게 —
        #   3.0 으로 하면 `N[SHIP]` 이 먼저 깨져 **두 라벨이 함께** ARM-NO-SKILL-HERE 가 되고
        #   게이트가 발화하지 않는다(자가진단이 그것을 잡았다).
        ('G-13', '★SHIP 치환 시 라벨이 뒤집히게 (SHIP 을 ANA 보다 나쁘게, 기술은 유지)',
         _mut(lambda rs: [r.__setitem__('avg_nRMSE', r['avg_nRMSE'] * 1.30)
                          for r in rs if r['arm'] == 'SHIP'])),
        ('G-14', '★주입 팔에서 확산 제안이 1회 돌았다(= 주입이 안 닿았다)',
         _mut(lambda rs: first(rs, 'LEARN').__setitem__('propose_calls', 1))),
        ('G-14', '★비주입 팔에서 제안이 0회(= 라우터가 안 돌았다)',
         _mut(lambda rs: first(rs, 'ANA-SHIP').__setitem__('propose_calls', 0))),
        ('N[SHIP]', 'SHIP 을 전부 ZERO 위로',
         _mut(lambda rs: [r.__setitem__('avg_nRMSE', r['zero_nRMSE'] * 2)
                          for r in rs if r['arm'] == 'SHIP'])),
        ('N[LEARN]', 'LEARN 을 전부 ZERO 위로',
         _mut(lambda rs: [r.__setitem__('avg_nRMSE', r['zero_nRMSE'] * 2)
                          for r in rs if r['arm'] == 'LEARN'])),
    ]


def selftest(verbose=True):
    """전 게이트 자가진단. `(통과여부, 보고)`. ⛔파일을 쓰지 않는다."""
    log = []

    def say(s):
        log.append(s)
        if verbose:
            print(s, flush=True)

    fails = []
    base = _fixture_rows()
    res = gates(base)
    bad = [n for n, okk, _ in res if not okk]
    say(f'[1/5] 픽스처 {len(base)}행 · 게이트 {len(res)}개 — 통과 {len(res) - len(bad)}/{len(res)}')
    for n, okk, d in res:
        if not okk:
            say(f'      ⛔{n}: {d}')
    if bad:
        fails.append(f'픽스처에서 통과하지 못한 게이트: {bad}')

    empty = gates([])
    leaked = [n for n, okk, _ in empty
              if okk and not any(p in n for p in ROWLESS_OK)]
    say(f'[2/5] rows=[] — 통과 {sum(1 for _, o, _ in empty if o)}/{len(empty)} '
        f'(허용 {list(ROWLESS_OK)}) · 공허한 통과 {len(leaked)}')
    for n in leaked:
        say(f'      ⛔공허한 통과: {n}')
    if leaked:
        fails.append(f'0행에서 통과한 게이트: {leaked}')

    n_flip = 0
    for prefix, desc, mut in _perturbations():
        r2 = gates(mut(base))
        hit = [(n, okk) for n, okk, _ in r2 if n.startswith(prefix)]
        if not hit:
            fails.append(f'{prefix}: 그 이름의 게이트가 없다 ({desc})')
            say(f'      ⛔{prefix} 게이트 없음 — {desc}')
        elif all(okk for _, okk in hit):
            fails.append(f'{prefix}: 변형에도 통과 — {desc}')
            say(f'      ⛔{prefix} 발화 안 함 — {desc}')
        else:
            n_flip += 1
    say(f'[3/5] 변형 {len(_perturbations())}건 중 게이트가 발화한 것 {n_flip}')

    # ── 4. 자료·패키지 게이트가 **입력 부재/틀린 핀**에서 통과하지 않는지 ─────
    step4 = []
    saved_eps, saved_mesh = globals()['episodes'], globals()['MESH_EXPECTED']
    try:
        globals()['episodes'] = lambda: []
        if any(okk for n, okk, _ in gates(base) if 'G-12' in n):
            step4.append('G-12: 에피소드 0개에서 통과(공허)')
        globals()['episodes'] = saved_eps
        globals()['MESH_EXPECTED'] = {'ADEQUATE': 48}
        if any(okk for n, okk, _ in gates(base) if 'G-12' in n):
            step4.append('G-12: 틀린 핀에도 통과')
    finally:
        globals()['episodes'] = saved_eps
        globals()['MESH_EXPECTED'] = saved_mesh
    fails += step4
    say(f'[4/5] 자료 부재·틀린 핀에서 G-12 가 통과하지 않는다: '
        f'{"OK" if not step4 else "⛔ " + "; ".join(step4)}')

    # ── 5. ★라벨 함수의 산술 — 결과 전에 **모든 분기가 도달 가능한지** 확인한다 ──
    #    (`gates_b9`/`gates_b10` 부류: 도달 불가능한 라벨·우연의 절반에서 발화하는 라벨)
    step5 = []
    nominal, confirm, not_earn = thresholds(N_EP)
    fake_neg = {a: dict(n=N_EP, better=N_EP, p_one_sided=0.0, rel_median_pct=-10.0)
                for a in ARM_ORDER}

    def _c(w):
        return dict(n=N_EP, better=w, n_tie=0, p_one_sided=sign_test_one_sided(w, N_EP),
                    rel_median_pct=-10.0, median_a=0.1, median_b=0.2)
    want = {
        'PRIOR-EARNS': (confirm, confirm, confirm),
        'PRIOR-EARNS-IN-PIPELINE-ONLY': (confirm, confirm, N_EP // 2),
        'PIPELINE-EARNS-NOT-PRIOR': (N_EP // 2, N_EP // 2, confirm),
        'PRIOR-EARNS-NOMINAL': (nominal, nominal, nominal),
        'PRIOR-NOT-EARNED': (N_EP // 2, N_EP // 2, N_EP // 2),
        'PRIOR-HARMS': (not_earn, N_EP // 2, N_EP // 2),
        'PRIOR-EARNS-BUT-PACKAGE-LOSES': (confirm, confirm, not_earn),
    }
    for lab, (a, bb, c) in want.items():
        got = label_primary(dict(L1a=_c(a), L1b=_c(bb), L2=_c(c)), N_EP, fake_neg)[0]
        if got != lab:
            step5.append(f'{lab} 도달 불가 (w={a},{bb},{c} → {got})')
    if label_primary(dict(L1a=_c(N_EP), L1b=_c(N_EP), L2=_c(N_EP)), 1, fake_neg)[0] \
            != 'SAMPLE-LOSS-UNDECIDED':
        step5.append('SAMPLE-LOSS-UNDECIDED 도달 불가')
    noskill = {a: dict(n=N_EP, better=0, p_one_sided=1.0, rel_median_pct=10.0)
               for a in ARM_ORDER}
    if label_primary(dict(L1a=_c(N_EP), L1b=_c(N_EP), L2=_c(N_EP)), N_EP, noskill)[0] \
            != 'ARM-NO-SKILL-HERE':
        step5.append('ARM-NO-SKILL-HERE 도달 불가')
    # ★널 라벨이 **우연의 절반에서 발화하지 않는지**(gates_b10 BLOCKER 부류)
    if sign_test_one_sided(not_earn + 1, N_EP) <= 0.5:
        step5.append(f'NOT_EARN={not_earn} 이 우연 이하에서 발화한다')
    fails += step5
    say(f'[5/5] 라벨 분기 도달성·문턱 산술: '
        f'{"OK" if not step5 else "⛔ " + "; ".join(step5)} '
        f'(n={N_EP}: NOMINAL {nominal} · CONFIRM {confirm} · NOT_EARN {not_earn})')

    okk = not fails
    say(('✅ 게이트 자가진단 통과' if okk else '⛔ 게이트 자가진단 실패:\n  - '
         + '\n  - '.join(fails)))
    return okk, '\n'.join(log)


if __name__ == '__main__':
    ok_, _ = selftest()
    sys.exit(0 if ok_ else 1)
