"""M-8 러너 — 규칙은 `gates_m8.py` **단일출처**. ⛔이 파일에 문턱·팔·판정이 없다.

    python3 tmp/M8_LADDER_2026-08-28/run_m8.py smoke        # 1 (record, seed) · 결과 지운다
    python3 tmp/M8_LADDER_2026-08-28/run_m8.py run 0 6      # 샤드 0/6 (본 배치 432런)
    python3 tmp/M8_LADDER_2026-08-28/run_m8.py run          # 직렬 432런

⛔⛔**GPU 금지가 이 배치의 1번 위험이다** (`PREREG §5.1` BLOCKER-A):
`learned_init.propose_atoms:130-132` 가 `config['device']` 를 **받지도 않고** 스스로
`cuda if torch.cuda.is_available() else cpu` 를 고르고, `d3_beamform_cond_diffusion.sample`
은 `torch.manual_seed` 뒤 `torch.randn(..., device=DEVICE)` 를 쓴다 ⟹ **같은 시드라도
CPU 와 CUDA 는 다른 원자**를 낸다. 그래서 `B-2 PROP` 열이 CPU 에서 12/12 재현되지 않는다.
⟹ 이 파일은 `CUDA_VISIBLE_DEVICES=''` 를 **torch 임포트 전 첫 줄**에서 세우고,
`main()` 이 `torch.cuda.is_available()` 로 다시 확인하고 죽으며, **행마다 `cuda_available`
을 싣는다**(`G-DEV` 가 전 행 `False` 를 요구한다).

★**RNG 규율** (`PREREG §6`) — 성장 위상이 `sequential_update.py:1617` `torch.rand` 에서
나오고 실측에서 **실제로 발화한다**(`AA14`: prune 30→10 뒤 +13).
  1. ⛔**원자 캐시 재사용 금지** — 제안 계단은 계단마다 `propose_atoms` 를 **인라인**으로
     돈다. `sample()` 이 `torch.manual_seed` 로 스트림을 리셋하므로 성장 지점까지 소모한
     draw 수가 계단끼리 구조적으로 같다.
  2. R5(`MMSE-PROP`)·부차 `MMSE-PROP-BAND` 는 bank 경로(전역 torch RNG 미사용)라 원자를
     **주입**한다 — 그 원자는 같은 셀 안에서 인라인으로 한 번 만든 것이고 `G-ATOM` 이
     R6–R8 과 md5 일치를 요구한다.
  3. 행에 **소모 draw**(`rng_torch_*`)·RNG 상태 해시·`n_sel`·`apg_*` 를 싣는다.
     ⛔이 값들은 **판정에 쓰지 않는다**(공시·게이트 전용).

★**개입 되읽기는 항등식이 아니다** — `effective_config` 를 입력 dict 복사로 만들지 않는다
(`S13b` 교훈). 읽는 곳:
  · `BankMMSEFit.n_candidates` / `.kxb,.kyb` / `.dk_c` / `.kp`  (bank 후보집합이 **실제로** 바뀌었나)
  · `model.n_gaussians`                                          (성분 수)
  · `res['varpro']['vp_steps']` = varpro `info['steps_executed']` (VARPRO 가 **몇 스텝 돌았나**)
  · `res['k_snap']['fired']`                                     (양자화가 발화했나)
  · `res['n_iters']`                                             (`learned_init_iters` 되읽기)
  · 성장 **가지** 계수기 (`B-2f` 교훈: 함수가 아니라 가지를 센다)
"""
import os
import sys

# ⛔⛔torch 임포트 **전에**. 이 두 줄의 순서가 이 배치의 재현성을 결정한다.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')   # BLAS 는 임포트 시점에 읽는다 ⟹ 여기서 세운다

import hashlib      # noqa: E402
import json         # noqa: E402
import time         # noqa: E402
import warnings     # noqa: E402

warnings.filterwarnings('ignore')

import numpy as np  # noqa: E402
import torch        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for _p in (HERE, ROOT,
           os.path.join(ROOT, 'scripts'),
           os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23'),
           os.path.join(ROOT, 'tmp', 'B3_REFIT_2026-08-23'),
           os.path.join(ROOT, 'tmp', 'bankmmse_port_2026-07-29')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(ROOT)

import gates_m8 as G                                   # noqa: E402
from scene_station import scene_station                # noqa: E402
from zero_ref import zero_ref_nowcast                  # noqa: E402
import make_gif_bank_deployed as MK                    # noqa: E402

_SCENE = {}
_ATOMS = {}
_KP = {}
_ZERO = {}
CAP = {}
RNG = dict(calls=0, elems=0, calls_seed=0, elems_seed=0, seeds=0)


# ── 계측 shim (⛔값을 바꾸지 않는다) ────────────────────────────────────────
def install_shims():
    """전부 **투명 래퍼**다 — 원 함수를 그대로 부르고 반환값을 건드리지 않는다."""
    import wavesplat.optimizer.learned_init as LI
    import wavesplat.optimizer.sequential_update as SU
    import wavesplat.model.spectral_gaussian as SG

    # ① 전역 torch RNG 소모량. `torch.manual_seed` 에서 "시드 이후" 계수기를 리셋한다
    #    ⟹ `sample()` 이 리셋한 뒤의 소모가 계단끼리 비교 가능해진다.
    if not getattr(torch, '_m8_rng_shim', False):
        def _wrap(fn):
            def f(*a, **kw):
                out = fn(*a, **kw)
                try:
                    n = int(out.numel())
                except Exception:                       # noqa: BLE001
                    n = 0
                RNG['calls'] += 1
                RNG['elems'] += n
                RNG['calls_seed'] += 1
                RNG['elems_seed'] += n
                return out
            return f
        for _nm in ('rand', 'randn', 'rand_like', 'randn_like', 'randint', 'randperm'):
            setattr(torch, _nm, _wrap(getattr(torch, _nm)))
        _ms = torch.manual_seed

        def _manual_seed(seed):
            RNG['calls_seed'] = 0
            RNG['elems_seed'] = 0
            RNG['seeds'] += 1
            return _ms(seed)
        torch.manual_seed = _manual_seed
        torch._m8_rng_shim = True

    # ② 확산 제안 호출 수 + **원 원자** (M4 `atoms_md5_inline` 과 비교 가능한 형태)
    if not getattr(LI, '_m8_shim', False):
        _prop = LI.propose_atoms

        def prop(*a, **kw):
            CAP['propose_calls'] = CAP.get('propose_calls', 0) + 1
            r = _prop(*a, **kw)
            try:
                CAP['raw_atoms'] = dict(kx=np.asarray(r[0], float).tolist(),
                                        ky=np.asarray(r[1], float).tolist(),
                                        energy=np.asarray(r[2], float).tolist())
            except Exception:                           # noqa: BLE001
                CAP['raw_atoms'] = None
            return r
        LI.propose_atoms = prop

        # ③ **선택된** 원자 (= 모델에 설치되는 그 집합). R5 가 주입하는 것과 같은 양이다.
        _lip = LI.learned_init_params

        def lip(frames, depth, config=None, seed=None):
            r = _lip(frames, depth, config, seed)
            CAP['sel_kx'] = np.asarray(r[0].detach().cpu().numpy(), float).tolist()
            CAP['sel_ky'] = np.asarray(r[1].detach().cpu().numpy(), float).tolist()
            CAP['n_sel'] = int(r[4])
            return r
        LI.learned_init_params = lip
        LI._m8_shim = True

    # ④ 적응 N **가지** 계수기 + 성분 증감 (⛔함수 호출 수가 아니다 — `B-2f` 교훈)
    if not getattr(SU.SequentialUpdater, '_m8_shim', False):
        _apg = SU.SequentialUpdater._adaptive_prune_grow

        def apg(self, *a, **kw):
            n0 = self.model.n_gaussians
            r = _apg(self, *a, **kw)
            CAP.setdefault('apg_dn', []).append(int(self.model.n_gaussians - n0))
            CAP['apg_calls'] = CAP.get('apg_calls', 0) + 1
            return r
        SU.SequentialUpdater._adaptive_prune_grow = apg
        SU.SequentialUpdater._m8_shim = True

    if not getattr(SG.SpectralGaussianModel, '_m8_shim', False):
        _add, _rem = SG.SpectralGaussianModel.add_gaussians, \
            SG.SpectralGaussianModel.remove_gaussians

        def add(self, n_new, *a, **kw):
            CAP['n_grown'] = CAP.get('n_grown', 0) + int(n_new)
            return _add(self, n_new, *a, **kw)

        def rem(self, keep):
            CAP['n_pruned'] = CAP.get('n_pruned', 0) + int(
                self.n_gaussians - len(keep))
            return _rem(self, keep)
        SG.SpectralGaussianModel.add_gaussians = add
        SG.SpectralGaussianModel.remove_gaussians = rem
        SG.SpectralGaussianModel._m8_shim = True


def _rng_hash():
    h = hashlib.md5()
    h.update(torch.get_rng_state().numpy().tobytes())
    st = np.random.get_state()
    h.update(np.asarray(st[1], dtype=np.uint32).tobytes())
    h.update(repr((st[0], st[2], st[3], st[4])).encode())
    return h.hexdigest()[:16]


# ── 장면 · 원자 · kp (셀 안에서 캐시. ⛔원자는 셀을 넘어 재사용하지 않는다) ──
def scene_for(record):
    if record not in _SCENE:
        _SCENE.clear()
        _SCENE[record] = scene_station(record, win=G.WIN)     # 기본값 = B-2 규약
    return _SCENE[record]


def frames_for(sc):
    t_shift = float(np.asarray(sc['P'])[:, 2].min())
    return MK.frames_from_scene(sc, t_shift, dtype=torch.float32), t_shift


def kp_bank(record, frames, depth):
    """bank 자신의 2단 beamform zoom (GT-free) — `fit_bank_mmse:157-164` 축자.

    `PROP-SNAP`/`PROP-LS-SNAP` 의 `k_snap_dk = kp/10` 과 환대 필터가 이것을 쓴다.
    `G-KP` 가 이 값을 **bank 팔의 객체 되읽기**(`BankMMSEFit.kp`) 와 대조한다."""
    if record in _KP:
        return _KP[record]
    from wavesplat.optimizer import bank_mmse as BM
    from wavesplat.optimizer.sequential_update import SequentialUpdater
    P, z = SequentialUpdater._bank_obs_from_frames(frames)
    P = np.asarray(P, float).astype(np.float32)
    z = np.asarray(z, float).astype(np.float32)
    bx, by = BM._beamform_top(P, z, float(depth), klim=2.0, n=120)
    kp = max(float(np.hypot(bx, by)), 1e-3)
    bx, by = BM._beamform_top(P, z, float(depth), klim=3.5 * kp, n=120)
    _KP.clear()
    _KP[record] = max(float(np.hypot(bx, by)), 1e-3)
    return _KP[record]


def bank_band():
    """환대 `(0.35, 2.5)` — `fit_bank_mmse` 의 **시그니처 기본값**에서 되읽는다(손복사 0)."""
    import inspect
    from wavesplat.optimizer.bank_mmse import fit_bank_mmse
    return tuple(inspect.signature(fit_bank_mmse).parameters['band'].default)


def atoms_for(record, seed, frames, depth):
    """★그 셀의 **제안 원자**(선택 후). 인라인으로 한 번 만든다.

    ⛔디스크 캐시 없음 — `PREREG §6.1.1` 이 캐시 재사용을 금지한다. 셀(record, seed)
    안에서만 메모리에 남고 셀이 바뀌면 버린다."""
    from wavesplat.run import set_seed
    from wavesplat.optimizer.learned_init import learned_init_params
    k = (record, int(seed))
    if k in _ATOMS:
        return _ATOMS[k]
    CAP.clear()
    set_seed(int(seed))
    cfg = G.arm_config('PROP-LS', depth=depth, seed=seed)   # 선택 노브 = 배포값
    learned_init_params(frames, float(depth), cfg, seed=int(seed))
    kx, ky = CAP.get('sel_kx'), CAP.get('sel_ky')
    if not kx:
        raise RuntimeError(f'{record}/{seed}: 제안이 원자를 0개 냈다')
    _ATOMS.clear()
    _ATOMS[k] = (kx, ky, dict(raw=CAP.get('raw_atoms'), n_sel=CAP.get('n_sel')))
    return _ATOMS[k]


def band_filter(kx, ky, kp, band):
    lo, hi = band
    kk = np.hypot(np.asarray(kx, float), np.asarray(ky, float))
    m = (kk > lo * kp) & (kk < hi * kp)
    return ([float(v) for v in np.asarray(kx, float)[m]],
            [float(v) for v in np.asarray(ky, float)[m]],
            float(m.mean()) if len(kk) else 0.0)


def zero_for(record):
    if record not in _ZERO:
        _ZERO[record] = float(zero_ref_nowcast(record, win=G.WIN))
    return _ZERO[record]


# ── 한 런 ───────────────────────────────────────────────────────────────────
def run_one(arm, record, seed):
    from wavesplat.optimizer.sequential_update import make_updater
    from wavesplat.run import set_seed
    t0w = time.time()
    sc = scene_for(record)
    frames, t_shift = frames_for(sc)
    depth = float(sc['depth'])
    t0 = float(sc['Pq'][0, 2]) - t_shift

    need_atoms = arm in G.KSET_ARMS
    need_kp = need_atoms or (arm in G.SNAP_ARMS)
    kp = kp_bank(record, frames, depth) if need_kp else None
    atoms = atoms_band = None
    frac_in_band = None
    if need_atoms:
        kx, ky, meta = atoms_for(record, seed, frames, depth)
        atoms = (kx, ky)
        bx, by, frac_in_band = band_filter(kx, ky, kp, bank_band())
        if not bx:
            raise RuntimeError(f'{record}/{seed}: 환대 안에 남은 원자가 0개')
        atoms_band = (bx, by)

    cfg = G.arm_config(arm, depth=depth, seed=seed, atoms=atoms,
                       atoms_band=atoms_band, kp_bank=kp)

    CAP.clear()
    RNG.update(calls=0, elems=0, calls_seed=0, elems_seed=0, seeds=0)
    set_seed(int(seed))
    rng_after_seed = _rng_hash()
    upd = make_updater(cfg)
    res = upd.initialize(frames,
                         x_range=(float(sc['gxq'].min()), float(sc['gxq'].max())),
                         y_range=(float(sc['gyq'].min()), float(sc['gyq'].max())),
                         verbose=False) or {}
    rng_after_init = _rng_hash()

    gxq, gyq = np.asarray(sc['gxq']), np.asarray(sc['gyq'])
    curve = []
    for li, l_ in enumerate(sc['leads']):
        g = np.asarray(sc['maps_gt'][li], float)
        mm = np.isfinite(g)
        if not mm.any():
            curve.append(float('nan'))
            continue
        Z = upd.predict_field(torch.tensor(gxq, dtype=torch.float32),
                              torch.tensor(gyq, dtype=torch.float32),
                              t0 + float(l_)).cpu().numpy().reshape(len(gyq), len(gxq))
        curve.append(float(np.sqrt(np.mean((Z[mm] - g[mm]) ** 2)) / sc['sg']))
    nowcast, leadmean = curve[0], float(np.nanmean(curve))

    # ── 개입 되읽기 (⛔입력 dict 복사 금지 — **객체**에서) ─────────────────
    bank = getattr(upd, '_bank', None)
    model = getattr(upd, 'model', None)
    vp = res.get('varpro') or {}
    snap = res.get('k_snap') or {}
    dk_c = None if not bank else float(bank[0].dk_c)
    dk_nan = None if not bank else (dk_c != dk_c)
    row = dict(
        # 장치 (BLOCKER-A 의 기계화)
        cuda_available=bool(torch.cuda.is_available()),
        # 경로·라우팅
        path_used=('bank' if bank else 'learned'),
        init_method_readback=str(getattr(upd, 'init_method', '?')),
        auto_selected=getattr(upd, '_auto_selected', None),
        real_gate=getattr(upd, '_auto_real_gate', None),
        ensemble_k=res.get('ensemble_k'),
        # bank 되읽기
        bank_B=(int(bank[0].n_candidates) if bank else None),
        bank_B_all=([int(f.n_candidates) for f in bank] if bank else None),
        bank_dk_c=(None if (dk_c is None or dk_nan) else dk_c),
        bank_dk_c_nan=dk_nan,
        bank_kp=(float(bank[0].kp) if bank else None),
        bank_members=(len(bank) if bank else None),
        bank_seeds=(list(getattr(upd, '_bank_seeds', [])) if bank else None),
        bank_kset_md5=(G.atoms_md5(bank[0].kxb, bank[0].kyb) if bank else None),
        bank_vr=([float(f.vr) for f in bank] if bank else None),
        bank_pexp=([float(f.pexp) for f in bank] if bank else None),
        bank_smult=([float(f.smult) for f in bank] if bank else None),
        # 학습 경로 되읽기
        n_gaussians=(int(model.n_gaussians) if model is not None else 0),
        n_sel=CAP.get('n_sel'),
        atoms_md5=(G.atoms_md5(CAP['sel_kx'], CAP['sel_ky'])
                   if CAP.get('sel_kx') else
                   (G.atoms_md5(*atoms) if atoms else None)),
        raw_atoms_md5=_raw_md5(CAP.get('raw_atoms')),
        n_atoms_raw=(len((CAP.get('raw_atoms') or {}).get('kx', []))
                     if CAP.get('raw_atoms') else None),
        propose_calls=CAP.get('propose_calls', 0),
        kset_n=(len(atoms_band[0]) if arm == 'MMSE-PROP-BAND'
                else (len(atoms[0]) if atoms else None)),
        kset_md5=(G.atoms_md5(*atoms_band) if arm == 'MMSE-PROP-BAND' else None),
        frac_in_band=frac_in_band,
        vp_fired=bool(vp),
        vp_steps_executed=int(vp.get('vp_steps', 0) or 0),
        vp_clamp_hits=vp.get('vp_clamp_hits'),
        vp_k_shift=vp.get('vp_k_shift'),
        vp_obj_before=vp.get('vp_obj_before'), vp_obj_after=vp.get('vp_obj_after'),
        snap_fired=bool(snap.get('fired')),
        snap_dk=snap.get('dk'), snap_shift_median=snap.get('shift_median'),
        snap_unique_cells=snap.get('n_unique_cells'),
        adaptive_n_readback=(None if bank else bool(getattr(upd, 'adaptive_n', None))),
        max_n_readback=int(getattr(upd, 'max_n', -1)),
        init_iters_readback=(None if bank else res.get('n_iters')),
        apg_calls=(None if bank else CAP.get('apg_calls', 0)),
        apg_dn=(None if bank else list(CAP.get('apg_dn', []))),
        n_grown=(None if bank else CAP.get('n_grown', 0)),
        n_pruned=(None if bank else CAP.get('n_pruned', 0)),
        kp_bank_calc=kp,
        # RNG 계측 (⛔공시 전용 — 판정에 쓰지 않는다)
        rng_hash_after_seed=rng_after_seed, rng_hash_after_init=rng_after_init,
        rng_torch_calls=RNG['calls'], rng_torch_elems=RNG['elems'],
        rng_torch_calls_since_seed=RNG['calls_seed'],
        rng_torch_elems_since_seed=RNG['elems_seed'],
        rng_manual_seeds=RNG['seeds'],
        # 장면 불변량 (단일요인)
        depth=depth, n_obs=int(len(sc['z'])), npx=int(sc['npx']),
        obs_hash=G.hash_obs(sc['P'], np.asarray(sc['z'], float)),
        q_hash=G.hash_obs(sc['Pq']),
        gt_hash=G.hash_obs(np.nan_to_num(sc['maps_gt'])),
        sigma_bits=G.bits(sc['sg']), kp_bits=G.bits(sc['kp']),
        t0_abs=float(sc['t0']), dt=float(sc['dt']),
        Hs=float(sc['Hs']), sigma=float(sc['sg']),
        n_leads=int(len(sc['leads'])),
        # 채점 (⛔ZERO_REF=1.0 금지 — 참 제로필드는 서술 전용)
        nowcast=nowcast, leadmean=leadmean, curve=curve,
        nowcast_bits=G.bits(nowcast), leadmean_bits=G.bits(leadmean),
        zero_ref=zero_for(record),
        # 설정 되읽기 (객체 속성 우선. 되읽을 속성이 없는 셋은 값 그대로 공시)
        cap_lambda_l1=getattr(upd, 'lambda_l1', None),
        cap_max_n=int(getattr(upd, 'max_n', -1)),
        cap_prune_threshold=getattr(upd, 'prune_threshold', None),
        cap_grow_residual_threshold=getattr(upd, 'grow_residual_threshold', None),
        cap_use_confidence=getattr(upd, 'use_confidence', None),
        cap_tracking_mode=getattr(upd, 'tracking_mode', None),
        cap_real_estimator=getattr(upd, 'real_estimator', None),
        cap_device=getattr(upd, 'device', None),
        cap_seed=getattr(upd, 'seed', None),
        cap_varpro_steps=cfg.get('varpro_steps'),
        cap_learned_init_iters=cfg.get('learned_init_iters'),
        cap_auto_switch_thresh=cfg.get('auto_switch_thresh'),
        cap_analytic_ensemble_k=cfg.get('analytic_ensemble_k'),
        cap_real_bank_ensemble_k=cfg.get('real_bank_ensemble_k'),
        cap_real_domain_gate=cfg.get('real_domain_gate'),
        cap_bank_dk_frac=cfg.get('bank_dk_frac'),
        cap_bank_k_set_n=(len(cfg['bank_k_set'][0])
                          if cfg.get('bank_k_set') is not None else None),
        cap_k_snap_dk=cfg.get('k_snap_dk'),
        wall_s=time.time() - t0w,
    )
    row.update(G.row_identity(arm, record, seed))
    return row


def _raw_md5(raw):
    if not raw:
        return None
    return hashlib.md5(json.dumps([list(map(float, raw['kx'])),
                                   list(map(float, raw['ky'])),
                                   list(map(float, raw['energy']))],
                                  sort_keys=True).encode()).hexdigest()


# ── 작업 목록 ───────────────────────────────────────────────────────────────
def jobs():
    """★단일출처는 게이트다 (`M4 AMENDMENT_1 A-3`). ⛔여기서 다시 정의하지 않는다."""
    return G.expected_jobs()


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'smoke'
    shard = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    nsh = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    # ── 런 0회 방벽 ────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        raise SystemExit('⛔GPU 가 보인다 — CUDA_VISIBLE_DEVICES 가 안 먹었다. 중단.')
    G.assert_shipping_defaults()
    G.assert_bank_patch()
    reg_ok, reg_msg = G.regress_pass()
    if not reg_ok:
        raise SystemExit(f'⛔패치 회귀가 통과하지 않았다: {reg_msg}\n'
                         f'   계약 = {G.PATCH_CONTRACT}')
    install_shims()
    print(f'  배포 핀 OK · bank 패치 OK · 회귀 OK ({reg_msg}) · '
          f'cuda={torch.cuda.is_available()} · band={bank_band()}', flush=True)

    todo = jobs()
    if len(todo) != G.N_RUNS:
        raise SystemExit(f'⛔작업 수 {len(todo)} != {G.N_RUNS}')
    if mode == 'smoke':
        todo = [j for j in todo if j[1] == G.RECORDS[0] and j[2] == G.ANCHOR_SEED]
        out = os.path.join(HERE, 'smoke_m8.jsonl')
    else:
        todo = [j for i, j in enumerate(todo) if i % nsh == shard]
        out = os.path.join(HERE, f'results_m8_{shard}.jsonl')
    if not todo:
        raise SystemExit('⛔작업 0건 — 샤드 인자를 확인하라.')

    done = {G.key(r) for r in G._load_jsonl(out) if 'error' not in r}
    n_new = 0
    with open(out, 'a') as fh:
        for (arm, record, seed) in todo:
            if (arm, record, int(seed)) in done:
                continue
            try:
                r = run_one(arm, record, seed)
            except Exception as e:                      # noqa: BLE001
                import traceback
                r = dict(G.row_identity(arm, record, seed), error=repr(e),
                         tb=traceback.format_exc()[-900:])
            fh.write(json.dumps(r) + '\n')
            fh.flush()
            n_new += 1
            if 'error' in r:
                tag = str(r['error'])[:80]
            else:
                tag = (f"now={r['nowcast']:.4f} lead={r['leadmean']:.4f} "
                       f"N={r['n_gaussians']} B={r['bank_B']} nsel={r['n_sel']} "
                       f"vp={r['vp_steps_executed']} apg={r['apg_dn']} "
                       f"snap={r['snap_fired']} rng={r['rng_torch_calls_since_seed']} "
                       f"{r['wall_s']:.1f}s")
            print(f'{arm:15s} {record:7s} s{seed} {tag}', flush=True)
    print(f'-> {out}  (신규 {n_new} · 이어받기 {len(done)})', flush=True)

    if mode == 'smoke':
        rows = G._load_jsonl(out)
        print('\n=== 연기 점검 (⛔판정 자료 아님) ===')
        _smoke_report(rows)
        os.remove(out)
        print('smoke 결과 삭제 (판정 자료 아님)')
    return 0


def _smoke_report(rows):
    """연기 런에서 **계측이 실제로 발화하는지**만 본다. ⛔판정하지 않는다."""
    b2 = G.b2_ship_ref()
    m4 = G.m4_prop_ref()
    for r in rows:
        if 'error' in r:
            print(f'  ⛔{r["arm"]}: {r["error"][:120]}')
            continue
        e = G.expect(r['arm'])
        chk = []
        chk.append(('dev', r['cuda_available'] is False))
        chk.append(('path', r['path_used'] == e['path']))
        if e['path'] == 'bank':
            chk.append(('gate', r['real_gate'] is True))
            chk.append(('route', r['init_method_readback'] == e['init_method']))
            chk.append(('ens', r['bank_members'] == e['bank_members']))
            chk.append(('kp', r['bank_kp'] == r['kp_bank_calc']
                        if r['kp_bank_calc'] is not None else True))
            if e['kset']:
                want = (r['kset_md5'] if r['arm'] == 'MMSE-PROP-BAND'
                        else r['atoms_md5'])
                chk.append(('kset', r['bank_kset_md5'] == want and want is not None))
                chk.append(('B=n', r['bank_B'] == r['kset_n']))
                chk.append(('dkNaN', r['bank_dk_c'] is None
                            and r['bank_dk_c_nan'] is True))
            else:
                chk.append(('B', r['bank_B'] is not None))
                chk.append(('dk', r['bank_dk_c'] is not None
                            and r['bank_dk_c_nan'] is False))
        else:
            chk.append(('vp', r['vp_steps_executed'] == e['vp_steps']))
            chk.append(('apg', (r['apg_calls'] or 0) == (1 if e['adaptive_n'] else 0)))
            chk.append(('snap', bool(r['snap_fired']) == e['snap']))
            chk.append(('rng', (r['rng_torch_calls_since_seed'] or 0) > 0))
        print(f'  {r["arm"]:15s} now={r["nowcast"]:.6f} B={r["bank_B"]} '
              f'N={r["n_gaussians"]} nsel={r["n_sel"]} kset={r["kset_n"]} '
              f'grown={r["n_grown"]} pruned={r["n_pruned"]} '
              f'atoms={str(r["atoms_md5"])[:8]} '
              f'| ' + ' '.join(f'{k}={"OK" if v else "⛔"}' for k, v in chk))
    rec = rows[0]['record'] if rows else None
    for arm, ref, tag in (('BANK', b2, 'B-2 SHIP'), ('PROP', m4, 'M4 SHIP-K-INL')):
        rr = [r for r in rows if r.get('arm') == arm and 'error' not in r]
        c = (ref or {}).get(rec) or {}
        if rr and c:
            same = (rr[0]['nowcast'] == c['nowcast']
                    and rr[0]['leadmean'] == c['leadmean'])
            print(f'  ★앵커 {arm} vs {tag}: '
                  f'{"비트 동일 OK" if same else "⛔불일치"}  '
                  f'{rr[0]["nowcast"]!r} vs {c["nowcast"]!r}')


if __name__ == '__main__':
    sys.exit(main())
