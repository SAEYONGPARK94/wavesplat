"""M7 `gen_m7.py` — HARD 분포 장면 생성기 (`gen_mixed.py` 의 복제 + 두 개의 선언된 편집).

⛔**원본 무수정.**  `tmp/mixed_retrain_2026-07-26/gen_mixed.py` 는 한 바이트도 건드리지 않는다.

★**복제 범위의 판정 (읽는 사람이 제일 먼저 물을 것이므로 먼저 답한다)**
`PREREG §9` 는 이 파일을 *"`gen_mixed.py` 의 **복제** + HARD 앵커 + 비선형 관측 + 참 성분/장 반환"*
이라 정의하고 `G-10` 이 원본과의 **비트 동일성**을 검사한다.  그 둘을 동시에 만족시키는 방법은
둘이다 — (a) 494줄을 텍스트로 베끼거나 (b) **실제로 바뀌는 함수만 텍스트 복제하고 나머지는
원본 객체를 임포트**하거나.  이 파일은 **(b)** 다.  사유:

  · 이 저장소가 이름 붙인 **반복 결함 1위**가 *"같은 기능이 두 파일에 중복 구현돼 한쪽만 계측"*
    이다(λ 3중 기재 · `coarse_to_fine` 2중 구현 · W4 의 개입 기대값 4중 기재).
    `quantise_to_M`·`GeometryBank`·`batch_from_scenes` 를 베끼면 **그 결함을 새로 하나 만든다.**
  · 임포트는 비트 동일성을 *검사*가 아니라 **구조**로 보증한다(같은 객체다).
  · 상수도 전부 `gen_mixed` 에서 **되읽는다** ⟹ 손복사 상수 **0건**(검사기 3번 규약).

⟹ **텍스트로 복제한 것은 실제로 편집이 들어가는 세 함수뿐**이고, 각 편집 지점에 `★EDIT` 표시가
붙어 있다.  그 외 모든 것은 원본 객체다(`_ORIGINALS` 목록이 그것을 나열하고 `--parity` 가 확인한다).

★**선언된 편집 (PREREG §1.5, 이 둘 말고는 없다)**
  **E1 앵커**  `_draw_HsTp` 의 앵커 목록을 **실측 12 레코드의 (Hs, Tp_disp) 중앙값**으로 바꾼다
      (±15% 지터 · 앵커 비율 0.5 = 배포와 같은 규약, 나머지 절반은 **현행 혼합 그대로**).
      값의 출처 = `premise_m7.json:P1_capacity.anchors_hard`(동결 전제, 손복사 0).
      나머지 절반이 *"현행 혼합 그대로"* 라는 것은 `premise_m7.p1_capacity.draw_hard` 가
      `else: return _draw_base(rng)` 로 **원본 `_draw_HsTp` 를 통째로 다시 부르는 것**을 뜻한다
      (= 앵커 0.50 / 배포앵커 0.25 / 넓은 상자 0.25).  ★이 파일은 그 축자를 그대로 복제한다 —
      전제 측정(P1_capacity 의 HARD 열)이 잰 것이 정확히 그 분포이기 때문이다.
  **E2 관측**  전방모델에 **2차 Stokes 속박 조화**를 켠다
      (`scripts/nonlinear_forward.elevation(nonlinear=True)`).  ⛔**라벨(32-슬롯 목표 원자)은
      자유파 그대로** — 배포 표현의 규약이다.

★**추가 반환 (PREREG §9 가 요구, 분포를 바꾸지 않는다)**
  장 손실용으로 `참 성분(ckx,cky,cA,com,cph)` · `seed` · `k_Hs` · `nonlinear` 를 장면 dict 에
  덧붙인다.  **RNG 를 한 번도 더 소모하지 않으므로 BASE 모드 비트 동일성에 영향이 없다**
  (`--parity` 가 rng 상태까지 비교해 그것을 확인한다).

★**순함수 게이트는 생성 *전*에** (`tmp/_audit/DISCIPLINE_gen_gates_2026-08-29.md` P1)
  · `screen()` = 앵커 물리 스크린(`k_p·Hs > BREAKING_KHS`).  문턱은 `gates_m7.BREAKING_KHS` 를
    **되읽는다**(손복사 0).  `G-8` 이 같은 심볼로 되잰다.
  · 생성 중 위반이 찍히면 **즉시 `exit 1`** — `M3` 가 위반값을 화면에 찍은 뒤 46분을 더 돌았다.
  · ⛔`nshard > 1` 은 거부한다(장면 스트림은 단일 순차 RNG 다 — 샤딩하면 시드 회계가 깨진다).

사용:
  python3 tmp/M7_SUPERVISION_2026-08-28/gen_m7.py --parity          # G-10 (BASE 비트 동일)
  python3 tmp/M7_SUPERVISION_2026-08-28/gen_m7.py --screen          # 앵커 물리 스크린 표
  python3 tmp/M7_SUPERVISION_2026-08-28/gen_m7.py --stats --n 400   # 두 스트림 실측(G-6/G-7 재료)
  python3 tmp/M7_SUPERVISION_2026-08-28/gen_m7.py --selftest
"""
import argparse
import json
import os
import statistics as st
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (ROOT, os.path.join(ROOT, 'scripts'),
           os.path.join(ROOT, 'tmp/mixed_retrain_2026-07-26'), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(ROOT)

import gen_mixed as gm                 # ⛔원본 — 읽기 전용
import gates_m7 as G                   # 배치 단일출처(문턱·팔·게이트)

# ── 원본에서 그대로 가져오는 것 (손복사 0건) ─────────────────────────────────
DEPTH = gm.DEPTH
M = gm.M
M_OBS = gm.M_OBS
NOISE_SIGMA = gm.NOISE_SIGMA
K_LIM_HARD = gm.K_LIM_HARD
STRATA = gm.STRATA
WEIGHTS = gm.WEIGHTS
ANCHORS = gm.ANCHORS
ANCHOR_FRAC = gm.ANCHOR_FRAC
ANCHOR_JITTER = gm.ANCHOR_JITTER
HS_BOX = gm.HS_BOX
TP_BOX = gm.TP_BOX
TOBS_FACTOR = gm.TOBS_FACTOR
SEED_LO, SEED_HI = gm.SEED_LO, gm.SEED_HI
REPR_MIN_FRAC = gm.REPR_MIN_FRAC

# ★원본 객체를 그대로 쓰는 것들 — 복제하지 않는다(중복 구현 금지).
_ORIGINALS = ('GeometryBank', 'replicate_to_M', 'energy_graded_to_M', 'energy_graded2_to_M',
              'quantise_to_M', '_expand_parents', 'batch_from_scenes')
GeometryBank = gm.GeometryBank
replicate_to_M = gm.replicate_to_M
energy_graded_to_M = gm.energy_graded_to_M
energy_graded2_to_M = gm.energy_graded2_to_M
quantise_to_M = gm.quantise_to_M
_expand_parents = gm._expand_parents
batch_from_scenes = gm.batch_from_scenes

# 배포 v1 체크포인트의 규약(`train_mixed.py:70` 기본값). Phase A 는 이것으로 돈다.
TARGET_MODE = gm.TARGET_MODE                 # 'replicate'
NGRID = G.NGRID                              # 장 손실 유보 격자(= wp_e2e CONFIGS['A'])

DATA_MODES = ('BASE', 'HARD')
# ★임무 B(`analyze_m7.load_train_manifest`)가 읽는 이름. 평평한 dict 이어야 한다.
GEN_MANIFEST = 'gen_manifest_m7.json'


def _md5(path):
    import hashlib
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def _stop(msg, code=1):
    """★위반값이 찍히면 **즉시** 멈춘다 (M3: 46분/16 에피소드를 더 돌았다)."""
    print(f"\n[GEN-STOP] {msg}", flush=True)
    sys.exit(code)


# ── 순함수 게이트: 앵커 물리 스크린 (생성 전에 돈다) ─────────────────────────
def _kp(Tp, depth=DEPTH):
    from wavesim.wave_generator import wavenumber_from_omega
    return float(wavenumber_from_omega(float(2.0 * np.pi / Tp), depth))


def anchor_table():
    """HARD 앵커 후보 = 동결 전제의 레코드별 (Hs, Tp_disp) 중앙값 12쌍. **손복사 0**.

    ★스크린이 쓰는 가파름은 **함께 동결된 실측값** `P2_real.records[rec]['k_Hs']` 다
    (`premise_m7.p2_real_spread` 가 `realdata_analysis.directional_spread` 의 공간 스펙트럼
    첨두 `kp` 로 잰 값).  `PREREG §1.5` 가 인용한 `LJ18a k·Hs = 1.3521` 이 이 열의 값이고,
    `P3_support.real_median_k_Hs = 0.24447` 도 이 열의 중앙값이다.
    ⚠**분산관계로 다시 계산하면 값이 다르다**(LJ18a 1.3602 · BS11a 0.4407) — `BS11a` 는
    문턱 0.44 의 **바로 위/아래**로 갈린다.  그래서 여기서는 **되계산하지 않고 동결값을 읽는다**
    (그 선택은 `G-8` 이 `['LJ18a']` 로 결과 전에 고정해 둔 것과 일치한다).
    두 추정자를 구별해 부른다: 스크린 = `k_Hs_measured`, 학습 스트림 = `k_Hs_disp`."""
    p = G.premise()
    anc = p['P1_capacity']['anchors_hard']
    rec2 = p['P2_real']['records']
    out = []
    for a in anc:
        r = a['record']
        if r not in rec2:
            _stop(f"전제에 레코드 {r} 의 P2 측정이 없다 — 스크린을 순함수로 돌릴 수 없다")
        # ★교차 확인(항등식 아님): 두 전제 블록이 같은 레코드에 대해 같은 Hs 를 말하는가.
        if abs(float(a['Hs']) - float(rec2[r]['Hs'])) > 1e-9:
            _stop(f"{r}: P1 앵커 Hs {a['Hs']} != P2 실측 Hs {rec2[r]['Hs']}")
        out.append((r, float(a['Hs']), float(a['Tp']), float(rec2[r]['k_Hs'])))
    return out


def screen(table=None, breaking=None):
    """★앵커 물리 스크린 — `k_p·Hs > BREAKING_KHS` 인 앵커는 **학습 앵커에서만** 제외.
    ⛔평가 12 레코드에서는 제외하지 않는다(사후배제 금지) — `G-8` 이 그것도 검사한다.
    문턱은 `gates_m7.BREAKING_KHS` 를 되읽는다(손복사 0)."""
    table = anchor_table() if table is None else table
    thr = G.BREAKING_KHS if breaking is None else breaking
    rows = []
    for rec, Hs, Tp, khs in table:
        rows.append(dict(record=rec, Hs=Hs, Tp=Tp, k_Hs_measured=khs,
                         k_Hs_disp=_kp(Tp) * Hs, excluded=bool(khs > thr)))
    kept = [(r['Hs'], r['Tp']) for r in rows if not r['excluded']]
    return dict(threshold=thr, rows=rows, kept=kept,
                excluded=[r['record'] for r in rows if r['excluded']],
                excluded_if_dispersion_kp=[r['record'] for r in rows
                                           if r['k_Hs_disp'] > thr],
                n_in=len(rows), n_kept=len(kept))


def hard_anchors(verbose=True):
    """스크린을 통과한 앵커 목록. ⛔결과가 사전등록과 어긋나면 **여기서 멈춘다**."""
    s = screen()
    if verbose:
        print(f"  [screen] k_p·Hs > {s['threshold']} 인 앵커 제외: "
              f"{s['excluded']}  ({s['n_kept']}/{s['n_in']} 유지)", flush=True)
    # ⛔커버리지 0 게이트 금지 — 아무것도 거르지 않는 스크린은 항등식이다.
    if s['n_kept'] == s['n_in']:
        _stop(f"앵커 물리 스크린이 아무것도 제외하지 않았다(커버리지 0) — {s['rows']}")
    # ⛔사전등록이 결과 전에 고정한 제외 집합(`gates_m7 G-8`)과 다르면 분포가 바뀐 것이다.
    if s['excluded'] != ['LJ18a']:
        _stop(f"앵커 제외 집합이 사전등록과 다르다: {s['excluded']} != ['LJ18a']")
    if s['n_kept'] < 2:
        _stop(f"앵커가 {s['n_kept']}개만 남았다 — 팔이 구조적으로 빈다")
    return s['kept'], s


# ── ★복제 1/3: `gen_mixed._draw_HsTp` — **편집 0** ───────────────────────────
def _draw_HsTp(rng):
    """`gen_mixed.py:235-240` 축자 복제. ⛔한 글자도 바꾸지 않았다(BASE 팔 = 원본)."""
    if rng.uniform() < ANCHOR_FRAC:
        Hs, Tp = ANCHORS[rng.integers(len(ANCHORS))]
        j = ANCHOR_JITTER
        return (float(Hs * rng.uniform(1 - j, 1 + j)), float(Tp * rng.uniform(1 - j, 1 + j)))
    return float(rng.uniform(*HS_BOX)), float(rng.uniform(*TP_BOX))


# ── ★복제 2/3: HARD 앵커 draw — **편집 E1 은 여기 한 곳뿐** ──────────────────
def _draw_HsTp_hard(rng, anchors):
    """`premise_m7.p1_capacity.draw_hard` **축자 복제**(전제 측정이 잰 바로 그 분포).

    ★EDIT E1 = 앵커 목록 교체.  구조는 원본과 같다:
      · 확률 `ANCHOR_FRAC` 로 **실측 앵커**(±`ANCHOR_JITTER`)
      · 나머지는 `_draw_HsTp(rng)` = **현행 혼합 그대로**(do-not-forget 질량).
        ⚠그 호출이 앵커 주사위를 한 번 더 굴리므로 실현 비율은
        실측앵커 0.50 / 배포앵커 0.25 / 넓은 상자 0.25 다.  **전제가 잰 것이 이 분포다.**
    """
    if rng.uniform() < ANCHOR_FRAC:
        Hs, Tp = anchors[rng.integers(len(anchors))]
        j = ANCHOR_JITTER
        return (float(Hs * rng.uniform(1 - j, 1 + j)), float(Tp * rng.uniform(1 - j, 1 + j)))
    return _draw_HsTp(rng)


# ── ★복제 3/3: `gen_mixed.sample_scene` ──────────────────────────────────────
def sample_scene(rng, bank, stratum=None, data='BASE', anchors=None):
    """`gen_mixed.py:270-351` 복제.  ★EDIT 는 두 곳(`E1`·`E2`)이고 둘 다 표시돼 있다.
    반환 dict 은 원본의 모든 키를 **같은 값으로** 담고, 장 손실용 키를 덧붙인다
    (덧붙이는 값은 이미 뽑힌 것들이라 RNG 를 더 소모하지 않는다)."""
    from wavesim.wave_generator import jonswap_components
    from nonlinear_forward import elevation

    if stratum is None:
        stratum = STRATA[int(rng.choice(len(STRATA), p=WEIGHTS))]
    # ── ★EDIT E1: 앵커 분포 ──────────────────────────────────────────────
    if data == 'BASE':
        Hs, Tp = _draw_HsTp(rng)
    elif data == 'HARD':
        Hs, Tp = _draw_HsTp_hard(rng, anchors)
    else:
        raise ValueError(data)
    # ─────────────────────────────────────────────────────────────────────
    gamma = float(rng.uniform(1.5, 4.0))
    dird = float(rng.uniform(-50, 50))
    seed = int(rng.integers(SEED_LO, SEED_HI))

    if stratum == 'discrete_crowded':
        n_par, M_sub, sig_sub, jit = int(rng.choice([15, 30])), 1, 0.0, False
    elif stratum == 'discrete_sparse':
        n_par, M_sub, sig_sub, jit = int(rng.choice([1, 2])), 1, 0.0, False
    elif stratum == 'discrete_moderate':
        n_par, M_sub, sig_sub, jit = 5, 1, 0.0, False
    elif stratum == 'continuum_mild':
        n_par, M_sub, sig_sub, jit = 15, 31, 10.0, False
    elif stratum == 'continuum_strong':
        n_par, M_sub, sig_sub, jit = 15, 31, 28.0, True
    elif stratum == 'intermediate_spread':
        n_par = int(rng.choice([5, 15, 30]))
        M_sub = int(rng.choice([3, 7, 15]))
        sig_sub = float(rng.uniform(2.0, 20.0))
        jit = bool(rng.uniform() < 0.5)
    else:
        raise ValueError(stratum)

    total_spread = float(rng.uniform(max(12.0, sig_sub + 4.0), 40.0))
    par_spread = float(np.sqrt(max(total_spread ** 2 - sig_sub ** 2, 1e-6)))

    c = jonswap_components(Hs=Hs, Tp=Tp, n_components=n_par, depth=DEPTH, gamma=gamma,
                           direction_deg=dird, direction_spread_deg=par_spread, seed=seed)
    parents = (np.array([x.kx for x in c], np.float32), np.array([x.ky for x in c], np.float32),
               np.array([x.A for x in c], np.float32), np.array([x.omega for x in c], np.float32),
               np.array([x.phi for x in c], np.float32))
    omega_p = 2 * np.pi / Tp
    d_omega = (3.0 - 0.5) * omega_p / max(n_par, 1)
    kx, ky, A, om, ph = _expand_parents(parents, M_sub, sig_sub, jit, d_omega, rng)

    inb = np.hypot(kx, ky) <= K_LIM_HARD
    e_tot = float(np.sum(A ** 2)) + 1e-18
    repr_frac = float(np.sum(A[inb] ** 2)) / e_tot
    if not inb.any():
        return None
    fkx, fky, fA = kx[inb], ky[inb], A[inb]

    n_true = len(fkx)
    if n_true <= M:
        if TARGET_MODE == 'graded2':
            tkx, tky, tA = energy_graded2_to_M(fkx, fky, fA)
        elif TARGET_MODE == 'graded':
            tkx, tky, tA = energy_graded_to_M(fkx, fky, fA)
        else:
            tkx, tky, tA = replicate_to_M(fkx, fky, fA)
    else:
        tkx, tky, tA = quantise_to_M(fkx, fky, fA, rng)
    order = np.lexsort((-tA, np.hypot(tkx, tky)))
    tkx, tky, tA = tkx[order], tky[order], tA[order]

    Tobs = float(rng.uniform(*TOBS_FACTOR)) * Tp
    ox, oy, ot = bank.window(rng, Tobs)
    # ── ★EDIT E2: 관측 전방모델에 2차 Stokes 속박 조화 (라벨은 자유파 그대로) ──
    nonlinear = (data == 'HARD')
    z = (elevation(kx, ky, A, om, ph, ox.astype(np.float64), oy.astype(np.float64),
                   ot.astype(np.float64), DEPTH, nonlinear=nonlinear)
         + rng.normal(0, NOISE_SIGMA, M_OBS)).astype(np.float32)
    # ─────────────────────────────────────────────────────────────────────

    return dict(tkx=tkx, tky=tky, tA=tA,
                P=np.column_stack([ox, oy, ot]).astype(np.float32), z=z,
                stratum=stratum, n_true=int(len(kx)), n_repr=int(n_true),
                n_distinct=int(min(n_true, M)), repr_frac=repr_frac,
                Hs=Hs, Tp=Tp, Tobs=Tobs, sigma_sub=sig_sub, n_par=n_par, M_sub=M_sub,
                # ── PREREG §9 가 요구한 추가 반환 (RNG 소모 0) ──
                ckx=kx, cky=ky, cA=A, com=om, cph=ph,
                seed=seed, data=data, nonlinear=nonlinear,
                k_Hs=float(_kp(Tp) * Hs))


ORIGINAL_KEYS = ('tkx', 'tky', 'tA', 'P', 'z', 'stratum', 'n_true', 'n_repr', 'n_distinct',
                 'repr_frac', 'Hs', 'Tp', 'Tobs', 'sigma_sub', 'n_par', 'M_sub')


def sample_scene_ok(rng, bank, stratum=None, data='BASE', anchors=None, max_tries=20):
    """`gen_mixed.sample_scene_ok` 복제 + ★탈출 시 **즉시 정지**.

    원본은 20회 재시도 후에도 문턱 미달이면 **그 장면을 그냥 돌려준다** — 그러면 학습 스트림에
    표현 불가 장면이 조용히 섞이고 `G-6` 이 사후에야 그것을 본다.  여기서는 그 순간 멈춘다
    (P1 = 행동 가능한 자리, `G-6` = 백스톱).  ⚠**값 경로는 원본과 동일**하므로 비트 동일성은
    영향받지 않는다(정지 분기는 RNG 를 쓰지 않는다)."""
    s = None
    for _ in range(max_tries):
        s = sample_scene(rng, bank, stratum=stratum, data=data, anchors=anchors)
        if s is not None and s['repr_frac'] >= REPR_MIN_FRAC:
            return s
    if s is None:
        _stop(f"[{data}] {max_tries}회 재시도 내내 대역 안 성분이 0개였다")
    _stop(f"[{data}] 표현 필터 탈출: repr_frac {s['repr_frac']:.4f} < {REPR_MIN_FRAC} "
          f"(stratum={s['stratum']} Hs={s['Hs']:.2f} Tp={s['Tp']:.2f}) — G-6 위반")


# ── 학습 장면 스트림 + 게이트 계측 ───────────────────────────────────────────
class SceneStream:
    """★러너가 매니페스트에 그대로 싣는 계측을 **자기가** 들고 있는다.

    `stats()` 가 내는 키는 `gates_m7.gates()` 가 읽는 이름과 같다 —
    `hard_scenes_n` · `hard_scenes_below_repr` · `hard_kHs_median` · `base_kHs_median` ·
    `train_scene_seeds_sample`.  ⛔러너가 그 수를 다시 세지 않는다(단일출처)."""

    SEED_SAMPLE_MAX = 64

    def __init__(self, data, seed, bank, target_mode=None, verbose=True):
        if data not in DATA_MODES:
            raise ValueError(data)
        self.data = data
        self.seed = int(seed)
        self.bank = bank
        self.rng = np.random.default_rng(self.seed)
        self.anchors, self.screen_result = (hard_anchors(verbose=verbose)
                                            if data == 'HARD' else (None, screen()))
        if target_mode is not None:
            global TARGET_MODE
            TARGET_MODE = target_mode
        self.target_mode = TARGET_MODE
        self.n = 0
        self.below_repr = 0          # ★탈출 건수 — 발생하면 sample_scene_ok 가 이미 멈춘다
        self.kHs = []
        self.repr_frac = []
        self.seeds = []
        self.strata_seen = {k: 0 for k in STRATA}
        self.window_widened = 0

    def batch(self, n):
        out = []
        for _ in range(n):
            s = sample_scene_ok(self.rng, self.bank, data=self.data, anchors=self.anchors)
            # ★순함수 게이트를 **장면 단위로** 되잰다 (G-9 술어 = 학습 시드 구간)
            if not (G.TRAIN_SEED_LO <= s['seed'] < G.TRAIN_SEED_HI):
                _stop(f"학습 장면 시드 {s['seed']} 가 구간 밖이다 — G-9 위반")
            if not np.isfinite(s['z']).all():
                _stop(f"관측에 비유한값이 있다 (stratum={s['stratum']} Hs={s['Hs']:.2f})")
            self.n += 1
            self.strata_seen[s['stratum']] += 1
            self.kHs.append(s['k_Hs'])
            self.repr_frac.append(s['repr_frac'])
            if len(self.seeds) < self.SEED_SAMPLE_MAX:
                self.seeds.append(s['seed'])
            if float(s['P'][:, 2].max()) > s['Tobs'] + 1e-6:
                self.window_widened += 1      # 창이 M_OBS 미만이라 레코드 전체로 넓어진 경우
            out.append(s)
        return out

    def stats(self):
        q = lambda v, p: (float(np.quantile(v, p)) if v else None)
        return dict(
            data=self.data, seed=self.seed, target_mode=self.target_mode,
            n_scenes=self.n, below_repr=self.below_repr,
            kHs_median=q(self.kHs, .5), kHs_p90=q(self.kHs, .9),
            repr_frac_median=q(self.repr_frac, .5), repr_frac_p10=q(self.repr_frac, .1),
            repr_frac_min=(float(min(self.repr_frac)) if self.repr_frac else None),
            seeds_sample=list(self.seeds),
            strata_seen=dict(self.strata_seen),
            window_widened=self.window_widened,
            anchors_excluded=self.screen_result['excluded'],
            anchors_kept=self.screen_result['n_kept'],
            anchor_screen_threshold=self.screen_result['threshold'])


def probe_repr_filter(data, n=400, seed=12345, bank=None, anchors=None):
    """★필터 항진 점검 — **거절 없이** 원시 `sample_scene` 을 n회 뽑아 문턱 미달 비율을 잰다.

    `sample_scene_ok` 는 미달 장면을 재시도로 지우므로 그 경로만 보면 `below_repr` 이
    **구조적으로 0**이다(= 항등식).  이 탐침은 **분모가 살아 있는** 측정이고,
    `premise_m7.p1_capacity` 의 `repr_frac.frac_below_min` 과 같은 양이다(비교 가능).
    ⚠전용 RNG 를 쓴다 — 학습 스트림의 난수를 소모하지 않는다."""
    bank = bank or GeometryBank(verbose=False)
    if data == 'HARD' and anchors is None:
        anchors, _ = hard_anchors(verbose=False)
    rng = np.random.default_rng(seed)
    rf, none_n = [], 0
    for _ in range(n):
        s = sample_scene(rng, bank, data=data, anchors=anchors)
        if s is None:
            none_n += 1
            continue
        rf.append(s['repr_frac'])
    below = int(np.sum(np.array(rf) < REPR_MIN_FRAC)) if rf else 0
    return dict(data=data, n=n, n_none=none_n, n_scored=len(rf), below=below,
                frac_below_min=(below / len(rf) if rf else None),
                median=(float(np.median(rf)) if rf else None),
                p10=(float(np.quantile(rf, .1)) if rf else None),
                min=(float(np.min(rf)) if rf else None))


# ── 장 손실 배치 (`e2e_finetune.make_scene_batch` 규약의 이식) ────────────────
def bank_bounds(bank):
    """유보 렌더 격자의 범위 = 기하 은행 발자국의 (min,max).
    `e2e_finetune.finetune` 이 `cd.pool_xy()` 발자국에 대해 한 것과 같은 규약이되,
    M7 의 관측은 `GeometryBank` 에서 오므로 **그 은행의 발자국**을 쓴다."""
    x = np.concatenate([d['x'] for d in bank.donors])
    y = np.concatenate([d['y'] for d in bank.donors])
    return (float(x.min()), float(x.max()), float(y.min()), float(y.max()))


def field_batch(scenes, stats, device, gb, ngrid=None):
    """`e2e_finetune.make_scene_batch` 가 돌려주던 dict 을 `gen_m7` 장면 위에서 만든다.

    ★키 이름·의미·shape 을 **그대로** 맞춘다 — `e2e_finetune.field_loss_from_x0hat` 을
    **수정 없이** 그대로 부르기 위해서다(그 손실 함수가 이 배치의 동결 부품이다).

    ⚠참 장(`gt_field`)은 **관측과 같은 전방모델**로 만든다(HARD 면 2차 Stokes 포함).
      사유: 참 장은 *그 바다의 실제 수면*이고 관측은 그 수면의 잡음 표본이다 — 배포 채점의
      규약과 같다.  ★이것은 HARD 팔에서 **표현이 담을 수 없는 성분을 목표에 남긴다**는 뜻이고,
      그 자체가 §8-3 이 공시한 *표현의 상한*이다.  값은 `gt_nonlinear` 로 행에 적힌다."""
    import torch
    from nonlinear_forward import elevation
    ngrid = NGRID if ngrid is None else ngrid
    B = len(scenes)
    x0, Bmap = batch_from_scenes(scenes, stats, device)
    P = np.stack([s['P'] for s in scenes]).astype(np.float32)
    Z = np.stack([s['z'] for s in scenes]).astype(np.float32)
    Pt = torch.tensor(P, device=device)
    Zt = torch.tensor(Z, device=device)

    gx = np.linspace(gb[0], gb[1], ngrid)
    gy = np.linspace(gb[2], gb[3], ngrid)
    GX, GY = np.meshgrid(gx, gy)
    gxf = GX.reshape(-1).astype(np.float64)
    gyf = GY.reshape(-1).astype(np.float64)
    NG2 = gxf.size

    gt = np.zeros((B, NG2), np.float32)
    ttar = np.zeros(B, np.float32)
    Hs_b = np.zeros(B, np.float32)
    for b, s in enumerate(scenes):
        ttar[b] = 0.5 * s['Tobs']                      # e2e 규약: nowcast = Tobs/2
        Hs_b[b] = s['Hs']
        gt[b] = elevation(s['ckx'], s['cky'], s['cA'], s['com'], s['cph'],
                          gxf, gyf, np.full(NG2, float(ttar[b])), DEPTH,
                          nonlinear=s['nonlinear']).astype(np.float32)
    gxb = torch.tensor(np.repeat(gxf[None].astype(np.float32), B, 0), device=device)
    gyb = torch.tensor(np.repeat(gyf[None].astype(np.float32), B, 0), device=device)
    ttb = torch.tensor(np.repeat(ttar[:, None], NG2, 1), device=device)
    return dict(x0=x0, Bmap=Bmap, Pt=Pt, Zt=Zt, gxb=gxb, gyb=gyb, ttb=ttb,
                gt_field=torch.tensor(gt, device=device),
                Hs=torch.tensor(Hs_b, device=device))


# ── G-10: BASE 모드 비트 동일성 ──────────────────────────────────────────────
def parity(n=64, seed=20260829, modes=('replicate', 'graded', 'graded2'), verbose=True):
    """★`data='BASE'` 에서 이 파일과 **원본 `gen_mixed`** 가 비트 동일한가.

    비교 대상 = 원본이 돌려주는 **모든 키**(`ORIGINAL_KEYS`) + ★**생성 후 RNG 상태**.
    RNG 상태까지 같다는 것은 *"난수를 한 번도 다르게 소모하지 않았다"* 의 증명이고,
    그것이 없으면 값 일치는 우연일 수 있다."""
    global TARGET_MODE
    bank = GeometryBank(verbose=False)          # ★같은 은행 객체를 양쪽에 준다
    worst = 0.0
    detail = []
    for tm in modes:
        gm.TARGET_MODE = tm
        TARGET_MODE = tm
        r_orig = np.random.default_rng(seed)
        r_new = np.random.default_rng(seed)
        m_tm = 0.0
        for i in range(n):
            a = gm.sample_scene_ok(r_orig, bank)
            b = sample_scene_ok(r_new, bank, data='BASE')
            for k in ORIGINAL_KEYS:
                va, vb = a[k], b[k]
                if isinstance(va, np.ndarray):
                    if va.shape != vb.shape or va.dtype != vb.dtype:
                        _stop(f"parity[{tm}] scene {i} key {k}: shape/dtype 불일치")
                    d = float(np.max(np.abs(va.astype(np.float64) - vb.astype(np.float64))))
                elif isinstance(va, str):
                    d = 0.0 if va == vb else float('inf')
                else:
                    d = abs(float(va) - float(vb))
                m_tm = max(m_tm, d)
        same_state = (r_orig.bit_generator.state == r_new.bit_generator.state)
        detail.append(dict(target_mode=tm, n=n, max_abs_delta=m_tm, rng_state_identical=same_state))
        if verbose:
            print(f"  parity[{tm:9s}] n={n}  max|Δ| = {m_tm:.17g}  "
                  f"rng_state_identical={same_state}", flush=True)
        if not same_state:
            _stop(f"parity[{tm}]: RNG 상태가 갈렸다 — 난수 소모가 다르다")
        worst = max(worst, m_tm)
    gm.TARGET_MODE = 'replicate'
    TARGET_MODE = 'replicate'
    scr = screen()
    out = dict(gen_parity_maxabs=worst, n_scenes=n, seed=seed, detail=detail,
               anchors_excluded=scr['excluded'], anchors_kept=scr['n_kept'],
               anchor_screen_threshold=scr['threshold'],
               originals_imported=list(_ORIGINALS),
               gen_mixed_file=os.path.relpath(gm.__file__, ROOT),
               gen_mixed_md5=_md5(gm.__file__), gen_m7_md5=_md5(__file__),
               written_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    return out


# ── 자가시험 / 스트림 실측 ───────────────────────────────────────────────────
def stream_stats(n=400, seed=12345, verbose=True):
    """두 스트림을 n 장면씩 실측한다 — `G-6`·`G-7` 의 재료이고 전제 P1 과 비교 가능."""
    bank = GeometryBank(verbose=False)
    out = {}
    for data in DATA_MODES:
        t0 = time.time()
        s = SceneStream(data, seed, bank, verbose=verbose)
        s.batch(n)
        d = s.stats()
        d['ms_per_scene'] = 1000.0 * (time.time() - t0) / n
        d['probe'] = probe_repr_filter(data, n=n, seed=seed + 1, bank=bank, anchors=s.anchors)
        out[data] = d
        if verbose:
            print(f"  [{data}] k·Hs median {d['kHs_median']:.4f} (p90 {d['kHs_p90']:.4f})  "
                  f"repr_frac med {d['repr_frac_median']:.4f} p10 {d['repr_frac_p10']:.4f} "
                  f"min {d['repr_frac_min']:.4f}  탈출 {d['below_repr']}  "
                  f"창확장 {d['window_widened']}/{n}  {d['ms_per_scene']:.1f} ms/scene",
                  flush=True)
            p = d['probe']
            print(f"         [항진 탐침] 거절 없이 {p['n_scored']}장면: 문턱 미달 "
                  f"{p['below']} ({p['frac_below_min']:.4f})  min {p['min']:.4f}", flush=True)
    if out['HARD']['kHs_median'] <= out['BASE']['kHs_median']:
        _stop(f"HARD 가 더 가파르지 않다 — G-7 위반 "
              f"({out['HARD']['kHs_median']} vs {out['BASE']['kHs_median']})")
    return out


def selftest(n=200):
    print("=== gen_m7 selftest ===", flush=True)
    print("\n[1] 앵커 물리 스크린")
    kept, s = hard_anchors()
    for r in s['rows']:
        print(f"    {r['record']:7s} Hs {r['Hs']:6.3f}  Tp {r['Tp']:5.2f}  "
              f"k·Hs(실측) {r['k_Hs_measured']:7.4f}  k·Hs(분산) {r['k_Hs_disp']:7.4f}  "
              f"{'⛔제외' if r['excluded'] else '유지'}")
    print(f"    ⚠분산관계로 재계산하면 제외 집합이 {s['excluded_if_dispersion_kp']} 가 된다 "
          f"(BS11a 가 문턱 0.44 의 경계에 있다) — 이 스크린은 **동결 실측값**을 쓴다.")

    print("\n[2] 원본 임포트 목록 (복제하지 않은 것)")
    for nm in _ORIGINALS:
        f = globals()[nm]
        print(f"    {nm:20s} <- {getattr(f, '__module__', '?')}  "
              f"same_object={f is getattr(gm, nm)}")
        assert f is getattr(gm, nm)

    print("\n[3] G-10 패리티 (BASE = 원본)")
    p = parity(n=16)
    assert p['gen_parity_maxabs'] == 0.0

    print("\n[4] 두 스트림 실측")
    ss = stream_stats(n=n)

    print("\n[5] 장 손실 배치 (e2e 규약)")
    import torch
    import d3_beamform_cond_diffusion as cd
    stats = torch.load(cd.PRIOR_CKPT, map_location=cd.DEVICE)['stats']
    bank = GeometryBank(verbose=False)
    gb = bank_bounds(bank)
    print(f"    bank footprint bounds x[{gb[0]:.1f},{gb[1]:.1f}] y[{gb[2]:.1f},{gb[3]:.1f}]")
    try:
        pool = cd.pool_xy()
        print(f"    (참고) cd.pool_xy 발자국   x[{pool[:,0].min():.1f},{pool[:,0].max():.1f}] "
              f"y[{pool[:,1].min():.1f},{pool[:,1].max():.1f}]")
    except Exception as e:                                       # noqa: BLE001
        print(f"    (pool_xy 비교 생략: {e})")
    for data in DATA_MODES:
        stt = SceneStream(data, 777, bank, verbose=False)
        sc = stt.batch(8)
        fb = field_batch(sc, stats, cd.DEVICE, gb)
        print(f"    [{data}] " + " ".join(
            f"{k}{tuple(v.shape)}" for k, v in fb.items() if torch.is_tensor(v)))
        assert torch.isfinite(fb['gt_field']).all()
        print(f"           gt_field rms {float(fb['gt_field'].pow(2).mean().sqrt()):.4f}  "
              f"Hs median {float(fb['Hs'].median()):.3f}")

    print("\n[6] 분산관계 패리티 (참 성분 om vs √(gk tanh kd))")
    worst = 0.0
    for data in DATA_MODES:
        stt = SceneStream(data, 5, bank, verbose=False)
        for s2 in stt.batch(20):
            k = np.hypot(s2['ckx'], s2['cky'])
            od = np.sqrt(9.81 * k * np.tanh(k * DEPTH))
            worst = max(worst, float(np.max(np.abs(s2['com'] - od))))
    print(f"    max|Δω| = {worst:.3e}   (e2e 가 잰 값 2e-16 과 같은 규약)")

    print("\nSELFTEST OK", flush=True)
    return dict(screen=s, parity=p, stream=ss, dispersion_max_domega=worst)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--parity', action='store_true', help='G-10: BASE 모드 비트 동일성')
    ap.add_argument('--screen', action='store_true', help='앵커 물리 스크린 표만 출력')
    ap.add_argument('--stats', action='store_true', help='두 스트림 실측(G-6/G-7 재료)')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--n', type=int, default=64)
    ap.add_argument('--seed', type=int, default=20260829)
    ap.add_argument('--nshard', type=int, default=1,
                    help='⛔1 이외는 거부한다(장면 스트림은 단일 순차 RNG)')
    ap.add_argument('--out', default=None, help='결과 json 경로(기본: 배치 안)')
    a = ap.parse_args()

    if a.nshard != 1:
        _stop(f"병렬 생성 금지: --nshard {a.nshard} — 장면 스트림은 단일 순차 RNG 이고 "
              f"샤딩하면 시드 회계(G-9)와 재현이 깨진다")

    if a.screen:
        kept, s = hard_anchors()
        print(json.dumps(s, indent=1, ensure_ascii=False))
    if a.parity:
        out = parity(n=a.n, seed=a.seed)
        path = a.out or os.path.join(HERE, GEN_MANIFEST)
        json.dump(out, open(path, 'w'), indent=1)
        print(f"\nmax|Δ| = {out['gen_parity_maxabs']:.17g}   -> {os.path.relpath(path, ROOT)}")
        if out['gen_parity_maxabs'] != 0.0:
            _stop(f"G-10 실패: max|Δ| = {out['gen_parity_maxabs']}")
        print("G-10 PASS (BASE 모드에서 원본과 비트 동일)")
    if a.stats:
        out = stream_stats(n=a.n, seed=a.seed)
        if a.out:
            json.dump(out, open(a.out, 'w'), indent=1)
    if a.selftest:
        selftest(n=a.n)
