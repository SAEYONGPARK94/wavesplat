"""M-10 러너 — 규칙은 `gates_m10.py`, 작업은 `jobs_m10.py` **단일출처**.
⛔이 파일에 문턱·밴드·라벨·판정이 없다.

    python3 tmp/M10_GEOMETRY_2026-08-30/run_m10.py canary       # ★1런 — G-4 전제만 시험
    python3 tmp/M10_GEOMETRY_2026-08-30/run_m10.py smoke        # 6 에피소드 · 결과는 지운다
    python3 tmp/M10_GEOMETRY_2026-08-30/run_m10.py run 0 4      # 샤드 0/4 (본 배치 288런)
    python3 tmp/M10_GEOMETRY_2026-08-30/run_m10.py run          # 직렬 288런

═══════════════════════════════════════════════════════════════════════════════
★**BLOCKER-D1 — 해소됨 (`AMENDMENT_1.md`, 2026-08-30, ⛔결과 0행에서)**

**증상**(양방향 실측, 1셀 `BF4_15comp_seed1581_uniform_ra_rf_ds` seed 42, 스레드 4종=1):

    CUDA 끔(사전등록 §8) : SHIP 0.10320282025107494  vs 동결 0.15112438875560960 → rel Δ **3.171e-01**
    CUDA 켬              : SHIP 0.15112438875560960  vs 동결 0.15112438875560960 → **0.0 (비트)**
                           SPEC25 0.21767047108636230 도 **비트 일치**

`PREREG §8`(`CUDA_VISIBLE_DEVICES=''`)·`gates_m10 G-10`(`cuda_available is False`)과
`gates_m10 G-4`(동결 rel Δ ≤ 1e-6)는 **동시에 성립하지 않았다**.
기제 = `wavesplat/optimizer/learned_init.py:124,133-134` 의 `propose_atoms` 가
`config['device']` 를 **받지도 않고** `cuda if torch.cuda.is_available() else cpu` 를 스스로
고른다(= `M8 PREREG §5.1 BLOCKER-A` 가 이름 붙인 그 결함).
`run_b12.py` 는 스레드도 장치도 고정하지 않았으므로 ★***동결 참조 자료가 GPU 산물이다.***

**처방**(`AMENDMENT_1`, 셋 다 적용):
  (a) `PREREG §8` 의 `CUDA_VISIBLE_DEVICES` 지시 삭제(취소선으로 이력 보존)
  (b) `gates_m10 G-10` 의 `cuda_available is False` 절 **완화**
      — ⛔`cuda_available` 은 스탬프 튜플에 **남아** 단일성은 계속 요구된다
  (c) 이 파일의 `CUDA_OFF = False`

⛔**스레드 고정(OMP 4종 = 1)은 정책과 무관하게 유지**한다: 위 실측이 그것만으로는 비트
재현이 깨지지 않음을 보였다(SHIP 비트 일치 · SPEC25 rel Δ ≤ 4.3e-16).
⟹ ★**이 배치의 재현 통화는 「스레드 고정 · GPU 허용」이다.** 그 조합의 뜻은 `AMENDMENT_1 §3`.
═══════════════════════════════════════════════════════════════════════════════

★**스레드 고정은 호출자에게 맡기지 않는다** — 이 파일이 `import torch` **앞에서** 스스로 세운다
(`M8 run_m8.py:39-44` 규약; `M7 E2` 가 호출자에게 맡겼다가 18시간을 날렸고 고정 후 180배였다).
⚠부수효과 실측: `SPEC25` 벽시계가 37.8 s(동결·미고정) → **52.6 s**(고정) = **+39%**.
   `SHIP` 은 7.9 → 7.7 s 로 변화 없음. 이 증가는 `PREREG §9` 의 여유 1.5배 안이다.

★**계측은 투명 shim 뿐**(`B-26 run_b26.install_shims` 규약) — 팔 정의는 `run_b12` 를
**그대로 부르고 수정하지 않는다**. 값을 바꾸지 않았다는 증거가 `G-4` 다
(*잡은 `avg_nRMSE` == 동결 `avg_nRMSE`*).

★**제로필드는 이 배치가 독립 구현한다**(`zero_reference()`) — `run_s16` 의 함수를 **부르지 않고**
h5/loader 에서 다시 만든다. `G-7` 이 그것을 기준선 반환 `zero_shell` 과 **9셸 전건 비트**로
대조한다(1셀 사전 실측: **9/9 비트 일치**, `shell_frac` 도 9/9, `zero_nRMSE` 도 일치).
⟹ 같은 양의 **독립 두 구현이 일치해야 통과하는 양성대조**다.

★**G-5 의 두 유도**(⛔항등식이 아니다):
   · `n_scored` — SHIP 은 `run.py` 의 `tracking_frames` 길이, SPEC25 는 `run_s16` 의 `sel.sum()`
   · `t_first/t_last` — SHIP 은 자기 `frame_results['t']`, SPEC25 는 **이 배치의 제로 패스**
   두 팔이 서로 다른 코드에서 같은 창을 유도해야 통과한다(`B-13` 「같은 창」 규율).

⚠**집계 규약 통일**(`PREREG §3.3` 공시): SHIP 셸을 반환된 `frame_results` 에서
**기준선 규약**(`t > 2·Tp_true` 프레임의 `nanmean`)으로 다시 집계한다.
`run.py` 의 `shell_frac` 은 **프레임 0 하나**라 그대로 쓰면 두 팔이 다른 양이 된다.
⟹ ⛔이 배치의 SHIP 셸 값은 `run.py` 의 `avg_shell_rmse` 와 **같은 수가 아니다**.
"""
import os
import sys

# ⛔⛔`import torch` **앞**. 이 두 블록의 순서가 이 배치의 재현성을 결정한다.
#    스레드 고정은 정책 무관(항상), CUDA 정책은 아래 한 상수가 소유한다.
CUDA_OFF = False         # ★AMENDMENT_1 (2026-08-30): True → False.  ⛔사유는 성능이 아니라
#                          **동결 재현**(G-4)이다 — 위 BLOCKER-D1 블록을 읽어라.
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'                   # BLAS 는 임포트 시점에 읽는다 ⟹ 여기서 세운다
if CUDA_OFF:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
elif os.environ.get('CUDA_VISIBLE_DEVICES') == '':
    # ★AMENDMENT_1 방벽 — 개정 **전** 의 §8 명령줄을 복사해 붙이는 사고를 여기서 끊는다.
    #   그 명령줄은 `CUDA_VISIBLE_DEVICES=` 를 export 하고, 그러면 이 파일의 정책과 무관하게
    #   G-4 가 144짝 전건에서 실패한다(rel Δ 3.171e-01). ⛔조용히 돌지 못하게 한다.
    raise SystemExit(
        "⛔호출자가 CUDA_VISIBLE_DEVICES='' 를 세웠는데 run_m10.CUDA_OFF 는 False 다.\n"
        "   그 조합은 BLOCKER-D1 그 자체다 — AMENDMENT_1 이 삭제한 **개정 전 §8 명령줄**을\n"
        "   복사한 것이다. 환경변수를 빼고 다시 돌려라 (스레드 4종 고정은 이 파일이 한다).")

import json         # noqa: E402
import time         # noqa: E402
import traceback    # noqa: E402
import warnings     # noqa: E402

warnings.filterwarnings('ignore')

import numpy as np  # noqa: E402
import torch        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for _p in (HERE, ROOT,
           os.path.join(ROOT, 'tmp', 'B11_HOS_BASELINE_2026-08-23'),
           os.path.join(ROOT, 'tmp', 'B12_HOS_EXTEND_2026-08-23'),
           os.path.join(ROOT, 'tmp', 'S16_BASELINE_2026-08-14')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(ROOT)

import gates_m10 as G                                              # noqa: E402
import jobs_m10 as J                                               # noqa: E402
import run_b12                                                     # noqa: E402 ★팔 정의
import run_s16                                                     # noqa: E402 ★기준선 정의

CAP = {}
_ZERO = {}
_FROZEN = None
_CODE_MD5 = None

# ★스모크 규모 — `G-CMP` 가 **발화할 수 있는** 최소치에서 정한다(아래 `_smoke_min_ep()`).
#   ⛔`M3` 는 스모크가 1셀이라 `G-17` 이 288런 전에는 구조적으로 발화 불가였다. 그 재발을 막는다.
SMOKE_SEED_EXTRA_EP = 1     # 이 개수의 에피소드는 시드 3개 전부 돈다(나머지는 첫 시드만)


# ── 계측 shim (⛔값을 바꾸지 않는다 — G-4 가 비트로 증명한다) ──────────────────
def install_shims():
    import wavesplat.run as WR
    from wavesplat.optimizer import sequential_update as SU

    if not getattr(WR, '_m10_shim', False):
        _orig_run = WR.run_estimation

        def _cap_run(*a, **kw):
            r = _orig_run(*a, **kw)
            CAP['ship'] = _reduce_frames(r)
            CAP['ship'].update(
                cap_avg_nRMSE=r.get('avg_nRMSE'),
                cap_init_iters=r.get('init_iters'),
                cap_varpro_declared=r.get('varpro_steps'),
                cap_varpro_fired=r.get('varpro_fired'),
                cap_varpro_obj_before=r.get('varpro_obj_before'),
                cap_varpro_obj_after=r.get('varpro_obj_after'),
                cap_k_snap_fired=r.get('k_snap_fired'),
                cap_ensemble_member_n=r.get('ensemble_member_n'))
            return r
        WR.run_estimation = _cap_run
        WR._m10_shim = True

    if not getattr(SU.SequentialUpdater, '_m10_shim_init', False):
        _orig_ctor = SU.SequentialUpdater.__init__

        def _cap_ctor(self, config, *a, **kw):
            _orig_ctor(self, config, *a, **kw)
            # ★객체 되읽기 — ⛔`cfg[k] == cfg[k]` 항등식이 아니다(S13b 교훈).
            CAP['obj'] = dict(lambda_l1=self.lambda_l1,
                              init_method=self.init_method,      # `auto` 는 아직 미변형
                              tracking_mode=self.tracking_mode,
                              device=str(getattr(self, 'device', 'cpu')))
            # ★⛔빈 dict 팔 금지의 **증거**: 생성자가 실제로 받은 키 수를 센다.
            CAP['cfg_nkeys'] = len(config or {})
        SU.SequentialUpdater.__init__ = _cap_ctor
        SU.SequentialUpdater._m10_shim_init = True

    if not getattr(SU.SequentialUpdater, '_m10_shim_ini2', False):
        _orig_ini = SU.SequentialUpdater.initialize

        def _cap_ini(self, *a, **kw):
            res = _orig_ini(self, *a, **kw)
            vp = (res or {}).get('varpro') or {}
            # ★**실행된** VARPRO 스텝 수 (`varpro.apply_varpro` info['steps_executed']).
            #   ⛔선언값(`config['varpro_steps']`)이 아니다 — 개입 계수기다(M-8 규약).
            if vp.get('vp_steps') is not None:
                CAP['varpro_executed'] = int(vp['vp_steps'])
            CAP['auto_selected'] = getattr(self, 'init_method', None)   # 라우터 사후값
            CAP['final_n'] = (res or {}).get('final_n')
            return res
        SU.SequentialUpdater.initialize = _cap_ini
        SU.SequentialUpdater._m10_shim_ini2 = True

    if not getattr(run_s16, '_m10_shim', False):
        _orig_letkf = run_s16.run_letkf

        def _cap_letkf(*a, **kw):
            row = _orig_letkf(*a, **kw)
            CAP['spec'] = {k: row.get(k) for k in
                           ('shell_rmse', 'zero_shell', 'shell_frac', 'zero_nRMSE',
                            'n_match', 'n_all', 'loc_radius', 'n_ensemble', 'background',
                            'filter_class', 'Hs', 'Tp', 'depth', 'avg_nRMSE')}
            return row
        run_s16.run_letkf = _cap_letkf
        run_s16._m10_shim = True


def _reduce_frames(r):
    """`frame_results` → 기준선 규약의 셸 집계. ★큰 배열은 여기서 버린다.

    `run.py` 의 `frame_results` 는 `tracking_frames` = `t > 2·Tp_true` 와 **정확히 같은 집합**
    이다(`data_loader.get_tracking_frames(tp_factor=2.0)` vs `run_s16:255 sel`).
    ⟹ 여기서의 `nanmean` 은 `run_s16._sm` 과 같은 규약이다.
    ⚠`run.py` 는 관측 0 프레임에 셸 키를 **아예 넣지 않는다**; `run_s16` 은 `nan` 을 넣는다.
      `nanmean` 은 두 경우를 같게 집계한다 ⟹ 팔 간 대조가 성립한다(1셀 실측 9/9 비트).
    """
    fr = r.get('frame_results') or []
    out = dict(n_scored=len(fr),
               t_first=(fr[0].get('t') if fr else None),
               t_last=(fr[-1].get('t') if fr else None))
    for field, name in (('shell_rmse', 'shell_rmse'), ('shell_frac', 'shell_frac')):
        d = {}
        for k in G.SHELL_KEYS:
            bare = k.split(':', 1)[1]           # ⛔`run.py` 는 접두어 없이 저장한다
            a = np.array([(f.get(field) or {}).get(bare, np.nan) for f in fr], dtype=float)
            d[k] = float(np.nanmean(a)) if np.isfinite(a).any() else float('nan')
        out[name] = d
    return out


# ── 제로필드 · 창 · 마스크의 **독립 구현** (⛔run_s16 을 부르지 않는다) ─────────
def zero_reference(path):
    """에피소드 하나에서 `zero_shell`·`shell_frac`·`zero_nRMSE`·채점 창을 다시 만든다.

    ★`G-7` 의 교차구현 양성대조가 이것이다. 규약은 `run_s16` 이 **정의한 것과 같은 것**이고
    (`sqrt(mean(gt[mask]²))` · `t > 2·Tp_true` 의 `nanmean`), 구현은 **독립**이다.
    ⛔`run_s16._shell_masks`·`run_s16._sm` 을 임포트하지 않는다 — 그러면 항등식이 된다.
    실측 단가 **0.92 s/에피소드**(추정기 0회).
    """
    if path in _ZERO:
        return _ZERO[path]
    import h5py
    from wavesplat.data_loader import EpisodeDataLoader
    ld = EpisodeDataLoader(path)
    gx = ld.gt_grid_x.cpu().numpy()
    gy = ld.gt_grid_y.cpu().numpy()
    Xg, Yg = np.meshgrid(gx, gy)
    keys, edges = G.SHELL_KEYS, G.EDGES
    zs = {k: [] for k in keys}
    fr = {k: [] for k in keys}
    zr, ts = [], []
    for i, fm in enumerate(ld):
        ox = fm.x_obs.cpu().numpy()
        oy = fm.y_obs.cpu().numpy()
        gt = fm.gt_field.cpu().numpy()
        zr.append(float(np.sqrt(np.mean(gt ** 2)) / ld.Hs_true))
        ts.append(float(ld.timestamps[i]))
        if len(ox):
            md = np.sqrt((Xg[None, :, :] - ox[:, None, None]) ** 2
                         + (Yg[None, :, :] - oy[:, None, None]) ** 2).min(axis=0)
            for k, (a, b) in zip(keys, zip(edges[:-1], edges[1:])):
                m = (md >= a) & (md < b)
                fr[k].append(float(np.mean(m)))
                zs[k].append(float(np.sqrt(np.mean(gt[m] ** 2))) if np.any(m)
                             else float('nan'))
        else:
            for k in keys:
                fr[k].append(float('nan'))
                zs[k].append(float('nan'))
    t = np.array(ts)
    sel = t > 2.0 * float(ld.Tp_true)
    if not sel.any():
        raise AssertionError('%s: 채점 프레임 0 — 0행 가드' % os.path.basename(path))

    def _sm(d):
        o = {}
        for k, v in d.items():
            a = np.array(v, dtype=float)[sel]
            o[k] = float(np.nanmean(a)) if np.isfinite(a).any() else float('nan')
        return o

    with h5py.File(path, 'r') as f:
        mesh = f['config'].attrs.get('mesh_verdict')
    mesh = mesh.decode() if isinstance(mesh, bytes) else (str(mesh) if mesh is not None
                                                          else None)
    out = dict(zero_shell=_sm(zs), shell_frac=_sm(fr),
               zero_nRMSE=float(np.mean(np.array(zr)[sel])),
               n_scored=int(sel.sum()),
               t_first=float(t[sel][0]), t_last=float(t[sel][-1]),
               n_grid=int(Xg.size), mesh_verdict=mesh,
               Hs=float(ld.Hs_true), Tp=float(ld.Tp_true), depth=float(ld.depth))
    _ZERO[path] = out
    return out


# ── 밴드 합성 (⛔전부 `gates_m10` 의 함수 — 이 파일에 규약이 없다) ──────────────
def _bands(shell, frac, zero_shell, n_grid):
    out = {}
    for b in G.BANDS:
        out['band_' + b] = G.band_rmse(shell, frac, G.BANDS[b])
        out['zero_' + b] = G.band_rmse(zero_shell, frac, G.BANDS[b])
    out['band_cells'] = {b: int(round(G.band_frac(frac, G.BANDS[b]) * n_grid))
                         for b in G.BANDS}
    out['band_frac'] = {b: G.band_frac(frac, G.BANDS[b]) for b in G.BANDS}
    return out


def _effective_prior_device():
    """★확산 사전이 **실제로** 올라간 장치 — ⛔재도출이 아니라 callee 자신의 상태 되읽기.

    `learned_init._load_cond(ckpt, device)` 가 `key = (abspath(ckpt), str(device))` 로
    캐시한다(`learned_init.py:104-107`). 그 키의 두 번째 성분이 **그 프로세스에서 사전이
    실제로 쓴 장치**다.
    ⛔`'cuda' if torch.cuda.is_available() else 'cpu'` 를 여기서 다시 쓰지 않는다 —
      그것이 바로 `BLOCKER-D1` 을 낳은 「같은 기능 두 곳 구현」이다.
    ⚠프로세스 전역이라 런을 가로질러 누적된다 ⟹ 값이 둘 이상이면 배치 중간에 장치가
      갈렸다는 뜻이고 `G-15` 가 그것을 드러낸다.
    """
    from wavesplat.optimizer import learned_init as LI
    devs = sorted({str(k[1]) for k in getattr(LI, '_CACHE', {})})
    return ','.join(devs) if devs else None


def _stamps():
    global _CODE_MD5
    if _CODE_MD5 is None:
        run_s16.preload()
        _CODE_MD5 = run_s16.code_fingerprint()[0]      # ⛔손복사 0 — S16 의 함수 그대로
    from wavesplat.run import DEFAULT_CONFIG
    return dict(device=str(DEFAULT_CONFIG['device']),
                cuda_available=bool(torch.cuda.is_available()),
                omp_threads=os.environ.get('OMP_NUM_THREADS'),
                torch_threads=int(torch.get_num_threads()),
                cuda_off_policy=bool(CUDA_OFF),
                code_md5=_CODE_MD5)


# ── 팔 (⛔`run_b12` 를 수정하지 않는다 · ⛔빈 dict 팔 0개) ──────────────────────
def run_ship(ep, seed):
    CAP.pop('ship', None)
    CAP.pop('obj', None)
    CAP.pop('varpro_executed', None)
    z = zero_reference(ep['path'])
    t0 = time.time()
    base = run_b12.run_ship(ep, seed)                  # ★B-12 의 팔, 손복사 0
    cap = CAP.get('ship') or {}
    obj = CAP.get('obj') or {}
    m = dict(avg_nRMSE=float(base['avg_nRMSE']),
             shell_rmse=cap.get('shell_rmse'), shell_frac=cap.get('shell_frac'),
             zero_shell_batch=z['zero_shell'], zero_shell_ref=None,
             zero_nRMSE=z['zero_nRMSE'],
             n_scored=cap.get('n_scored'), t_first=cap.get('t_first'),
             t_last=cap.get('t_last'),
             obj_init_method=obj.get('init_method'),
             obj_tracking_mode=obj.get('tracking_mode'),
             obj_lambda_l1=obj.get('lambda_l1'),
             obj_device=obj.get('device'),
             cap_varpro_steps=CAP.get('varpro_executed'),      # ★실행된 스텝 수
             cap_varpro_declared=cap.get('cap_varpro_declared'),
             cap_varpro_fired=cap.get('cap_varpro_fired'),
             cap_varpro_obj_before=cap.get('cap_varpro_obj_before'),
             cap_varpro_obj_after=cap.get('cap_varpro_obj_after'),
             cap_init_iters=cap.get('cap_init_iters'),
             cap_ensemble_member_n=cap.get('cap_ensemble_member_n'),
             auto_selected=CAP.get('auto_selected'),
             arm_cfg_nonempty=bool(CAP.get('cfg_nkeys', 0) > 0),
             arm_cfg_nkeys=CAP.get('cfg_nkeys'),
             prior_device=_effective_prior_device(),    # ★AMENDMENT_1 (G-15) — 되읽기
             wall_s=time.time() - t0)
    m.update(_bands(m['shell_rmse'] or {}, m['shell_frac'] or {}, z['zero_shell'],
                    z['n_grid']))
    return m, z


def run_spec(ep, seed):
    CAP.pop('spec', None)
    z = zero_reference(ep['path'])
    t0 = time.time()
    base = run_b12.run_spec(ep, seed)                  # ★S16 의 기준선, 손복사 0
    cap = CAP.get('spec') or {}
    m = dict(avg_nRMSE=float(base['avg_nRMSE']),
             shell_rmse=cap.get('shell_rmse'), shell_frac=cap.get('shell_frac'),
             zero_shell_batch=z['zero_shell'],
             zero_shell_ref=cap.get('zero_shell'),     # ★교차구현 대조 상대 (G-7)
             zero_nRMSE=float(base['zero_nRMSE']),
             n_scored=cap.get('n_match'),              # ★기준선 자신의 창 계수
             t_first=z['t_first'], t_last=z['t_last'],
             loc_radius=base.get('loc_radius'), n_ensemble=base.get('n_ensemble'),
             background=base.get('background'), filter_class=cap.get('filter_class'),
             arm_cfg_nonempty=True, arm_cfg_nkeys=None,
             wall_s=time.time() - t0)
    m.update(_bands(m['shell_rmse'] or {}, m['shell_frac'] or {}, z['zero_shell'],
                    z['n_grid']))
    return m, z


RUNNERS = {'SHIP': run_ship, 'SPEC25': run_spec}


# ── 동결 대조 (G-4 를 러너가 **즉시** 확인한다 — fail-fast) ────────────────────
def frozen():
    global _FROZEN
    if _FROZEN is None:
        _FROZEN = G.load_frozen_b12()
        if not _FROZEN:
            raise AssertionError('B-12 동결 자료가 비었다 — 공허 참조(G-4 가 항진한다)')
    return _FROZEN


def _check_frozen(row):
    """⛔이 함수는 **게이트가 아니다** — `gates_m10` 의 같은 비교를 러너가 미리 부르는 것뿐이다
    (`G._rel`·`G.REPRO_RTOL` 를 그대로 쓴다 = 단일출처). 판정은 `analyze_m10` 만 한다."""
    f = frozen().get(J.row_identity(row))
    if f is None:
        return None, None
    return G._rel(float(row['avg_nRMSE']), f), f


BLOCKER_D1 = """
⛔⛔BLOCKER-D1 — 동결 재현이 깨졌다. **본런을 시작하지 마라.**
   장치 정책 CUDA_OFF=%s · cuda_available=%s · OMP=%s
   %s  재실행 %.17g  vs  B-12 동결 %.17g   rel Δ %.3e  (G-4 한계 %g)
★사전등록의 `PREREG §8`(`CUDA_VISIBLE_DEVICES=''`)·`gates_m10 G-10`(`cuda_available is False`)과
  `gates_m10 G-4`(동결 rel Δ ≤ 1e-6)는 **동시에 성립하지 않는다**:
  동결 `results_b12.jsonl` 은 **GPU 가 보이는 환경**에서 만들어졌고
  `learned_init.propose_atoms` 는 `config['device']` 를 무시하고 스스로 cuda 를 고른다.
⟹ `AMENDMENT_1.md` 로 (a) §8 의 `CUDA_VISIBLE_DEVICES` 지시와 (b) `G-10` 의
  `cuda_available is False` 절을 함께 고치고 **결과 전임을 기계로 증명**한 뒤
  `run_m10.CUDA_OFF = False` 로 내려라. ⛔스레드 고정(OMP 4종=1)은 그대로 둔다 —
  그것만으로는 재현이 깨지지 않는다(실측: 비트 일치).
"""


def _blocker_text(row, rel, f, st):
    return BLOCKER_D1 % (CUDA_OFF, st['cuda_available'], st['omp_threads'],
                         '%s %s s%s' % J.row_identity(row),
                         float(row['avg_nRMSE']), f, rel, G.REPRO_RTOL)


# ── 실행 ───────────────────────────────────────────────────────────────────────
def _done(path):
    done = set()
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if 'error' not in r:
                done.add(J.row_identity(r))
    return done


def execute(jobs, out_path, tag, strict_frozen=True):
    """작업 목록을 돌려 행을 append 한다. ★`jobs_m10` 이 정체성을 소유한다."""
    install_shims()
    st = _stamps()
    eps = {e['episode']: e for e in J.episodes()}
    done = _done(out_path)
    todo = [j for j in jobs if j not in done]
    print('M-10 %s: 작업 %d (완료 %d) -> %s' % (tag, len(todo), len(done),
                                             os.path.basename(out_path)), flush=True)
    print('  장치 스탬프 %r' % (st,), flush=True)
    print('  ★재현 통화 = 스레드 고정(OMP 4종=1) · GPU 허용(CUDA_OFF=%s) — AMENDMENT_1 §3'
          % CUDA_OFF, flush=True)
    n_err = 0
    with open(out_path, 'a') as fh:
        for ident in todo:
            arm, ep_name, seed = ident
            ep = eps[ep_name]
            try:
                metrics, z = RUNNERS[arm](ep, seed)
                row = J.stamp(ident, metrics, bf=ep['bf'], ncomp=ep['ncomp'],
                              gen_seed=ep['seed'], mesh_verdict=z['mesh_verdict'],
                              Hs=z['Hs'], Tp=z['Tp'], depth=z['depth'],
                              n_grid=z['n_grid'], **st)
            except Exception as exc:                                # noqa: BLE001
                n_err += 1
                row = J.stamp(ident, dict(error=repr(exc),
                                          tb=traceback.format_exc()[-600:]), **st)
                fh.write(json.dumps(row) + '\n')
                fh.flush()
                print('  ⛔%-6s %-38s s%-3s %s' % (arm, ep_name[:38], seed,
                                                  repr(exc)[:70]), flush=True)
                continue
            rel, f = _check_frozen(row)
            fh.write(json.dumps(row) + '\n')
            fh.flush()
            print('  %-6s %-38s s%-3s nRMSE=%.6f  relΔ=%s  %s=%.6f  %.1fs'
                  % (arm, ep_name[:38], seed, row['avg_nRMSE'],
                     ('%.1e' % rel) if rel is not None else '--',
                     G.PRIMARY_BAND, row['band_' + G.PRIMARY_BAND] or float('nan'),
                     row['wall_s']), flush=True)
            if strict_frozen and rel is not None and rel > G.REPRO_RTOL:
                print(_blocker_text(row, rel, f, st), flush=True)
                return 2, n_err
    return (1 if n_err else 0), n_err


def cmd_canary():
    """★1런 — `G-4` 전제만 시험한다. 본런 2.4 코어·h 를 태우기 전에 11초로 끊는다."""
    install_shims()
    st = _stamps()
    ep = J.episodes()[0]
    print('M-10 canary: %s s%d · 장치 %r' % (ep['episode'], J.SEEDS['SHIP'][0], st),
          flush=True)
    metrics, _z = run_ship(ep, J.SEEDS['SHIP'][0])
    row = J.stamp(('SHIP', ep['episode'], J.SEEDS['SHIP'][0]), metrics, **st)
    rel, f = _check_frozen(row)
    print('  재실행 %.17g  동결 %.17g  rel Δ %.3e  (한계 %g)'
          % (row['avg_nRMSE'], f, rel, G.REPRO_RTOL))
    if rel > G.REPRO_RTOL:
        print(_blocker_text(row, rel, f, st))
        return 2
    print('  ✅G-4 전제 통과 (비트 일치 %s)' % (row['avg_nRMSE'] == f))
    return 0


def _smoke_min_ep():
    """★`G-CMP` 가 **발화할 수 있는** 최소 에피소드 수.

    `gates_m10.cmp_ok_from` 은 `thresholds(n)['CONFIRM']` 을 부르고 그 안의
    `next(w for w in range(n+1) if p(w,n) <= FAMILY_ALPHA)` 는 **n 이 작으면 StopIteration**
    이다(n=1·3 에서 실제로 죽는다). ⟹ 스모크는 그 문턱이 존재하는 최소 n 이상이어야 한다.
    ⛔`M3` 가 물린 부류(*"스모크 1셀이라 게이트가 본런 전에는 구조적으로 발화 불가"*)의 기계화.
    """
    n = 1
    while True:
        try:
            G.thresholds(n)
            return n
        except StopIteration:
            n += 1
            if n > 64:
                raise


def cmd_smoke():
    """★가장 비싼 팔(`SPEC25`, 52.6 s) + **게이트가 실제로 발화하는 셀**을 포함한다.

    · `G-CMP` 는 `_smoke_min_ep()` 개 이상의 에피소드가 있어야 발화한다 ⟹ 그만큼 돈다.
    · `G-5`·`G-6`·`G-7`·`G-9` 는 한 에피소드에 두 팔이 다 있어야 발화한다 ⟹ 전 에피소드 두 팔.
    · 시드 축(`med_by_episode`)은 에피소드 %d개에서 3시드 전부 돈다.
    ⛔`G-1`(표본 수)만은 스모크에서 **구조적으로 발화 불가**다(48×3 을 요구한다) ⟹ 면제 명시.
    """
    n_ep = _smoke_min_ep()
    eps = [e['episode'] for e in J.episodes()[:n_ep]]
    jobs = []
    for k, ep in enumerate(eps):
        for arm in J.ARMS:
            seeds = J.SEEDS[arm] if k < SMOKE_SEED_EXTRA_EP else J.SEEDS[arm][:1]
            jobs += [(arm, ep, s) for s in seeds]
    out = J.smoke_path(0)
    if os.path.exists(out):
        os.remove(out)
    print('★스모크 설계: 에피소드 %d (G-CMP 최소 %d) · 작업 %d · 시드3 에피소드 %d'
          % (len(eps), n_ep, len(jobs), SMOKE_SEED_EXTRA_EP), flush=True)
    rc, n_err = execute(jobs, out, 'smoke', strict_frozen=True)
    if rc == 2:
        return 2
    rows = [json.loads(l) for l in open(out) if l.strip()]
    print('\n=== 스모크 게이트 (⛔G-1 은 표본 수 게이트라 면제) ===')
    SMOKE_EXEMPT = ('G-1',)
    bad = []
    for name, ok, det, blocking in G.gates(rows, frozen()):
        head = name.split()[0]
        tag = ('면제' if head in SMOKE_EXEMPT else ('PASS' if ok else 'FAIL'))
        print('  [%-4s] %s: %s' % (tag, name, det))
        if not ok and head not in SMOKE_EXEMPT and blocking:
            bad.append(head)
    try:
        cmp_ok, cmp_d = G.cmp_ok_from(rows)
        print('  [%-4s] G-CMP 비교자 유효성: %r' % ('PASS' if cmp_ok else 'FAIL', cmp_d))
    except StopIteration:
        bad.append('G-CMP(발화불가)')
        print('  [FAIL] G-CMP: 에피소드 수가 문턱을 만들 수 없다 — 스모크 규모 결함')
    wall = {a: [r['wall_s'] for r in rows if r['arm'] == a and 'error' not in r]
            for a in J.ARMS}
    print('\n=== 단가 (스레드 고정 상태) ===')
    for a in J.ARMS:
        if wall[a]:
            print('  %-6s n=%d  중앙 %.1f s  합 %.1f s' % (a, len(wall[a]),
                                                          sorted(wall[a])[len(wall[a]) // 2],
                                                          sum(wall[a])))
    est = sum(sorted(wall[a])[len(wall[a]) // 2] * G.N_EP * len(J.SEEDS[a])
              for a in J.ARMS if wall[a])
    print('  ⟹ 본런 288런 추정 %.2f 코어·h (제로 패스 48×0.9 s 별도)' % (est / 3600.0))
    print('\n스모크 %s (차단 FAIL %r · 에러 %d)' % ('PASS' if not bad else 'FAIL', bad, n_err))
    return 1 if bad else 0


def cmd_run(shard_i, nshard):
    jobs = J.shard(J.expected_jobs(), shard_i, nshard)
    rc, n_err = execute(jobs, J.results_path(shard_i), 'run %d/%d' % (shard_i, nshard))
    return rc


def main(argv):
    cmd = argv[1] if len(argv) > 1 else 'smoke'
    if cmd == 'canary':
        return cmd_canary()
    if cmd == 'smoke':
        return cmd_smoke()
    if cmd == 'run':
        i = int(argv[2]) if len(argv) > 2 else 0
        n = int(argv[3]) if len(argv) > 3 else 1
        return cmd_run(i, n)
    print(__doc__)
    return 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
