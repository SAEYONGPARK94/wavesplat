"""T-4 러너 — ★**같은 런을 lead 사다리에서 다시 채점한다**(생성 0 · 학습 0 · 새 관측 0).

판정은 `gates_t4.py` 에 **결과 0행에서 동결**돼 있다. 이 파일은 자료를 만들 뿐이고
**행 스키마와 프로토콜은 `PREREG §8` 이 계약**이다(스키마는 손으로 옮기지 않고
`prereg_row_schema()` 가 그 문서에서 **파싱**한다 — 어긋나면 자가진단이 죽는다).

    python3 tmp/T4_GEOMETRY_2026-08-30/run_t4.py selftest
    python3 tmp/T4_GEOMETRY_2026-08-30/run_t4.py smoke
    python3 tmp/T4_GEOMETRY_2026-08-30/run_t4.py run <shard> <nshard>

프로토콜(⛔`PREREG §8` 축자, 순서를 바꾸지 마라)
──────────────────────────────────────────────────────────────────────────────
`SHIP`   원점 `i` 에서 **`update(frame_i)` 를 마친 배포 상태 그대로**
         `predict_field(gx, gy, t_{i+L})` 을 부르고 GT 프레임 `i+L` 과 맞댄다.
         ⚠**미래 관측을 먹이지 않는다** — 이 하네스는 관측 순서를 **전혀 건드리지 않는다**
         (재채점이지 재실행이 아니다. 개입은 `predict_field` 호출 **추가** 하나뿐이다).
`SPEC25` 원점 `i` 에서 필터를 **깊은 복사**한 뒤 `_forecast_member` 를 `L` 프레임만큼 반복하고
         `predict_grid()` → `run_s16` 과 **같은** `RegularGridInterpolator` 로 옮긴다.
         ★**한 번 전진하며 rung 마다 스냅샷**(원점당 `max(lead_frames)` 스텝).
`ZERO`   런이 아니다. **채점 시각의 GT 로부터** 낸다. ⛔고정 기준상수 미사용
         (`gates_t4.no_flat_ref_constant()` 가 `ast` 로 이 디렉터리 전체를 강제한다).

★**이 파일에 문턱·라벨·판정이 없다.** 유일하게 러너가 가진 수치 판단은 **비용 게이트**이고
  그 문턱조차 `gates_t4.MAX_CORE_HOURS` 에서 **읽는다**(⛔손복사 0).

★**재현 통화 = 「스레드 고정 · GPU 허용」**(`M-10 AMENDMENT_1` 승계, `gates_t4 G-10` 이
  `cuda_available` 을 **단일성만** 요구하는 이유가 그것이다).
  ⛔`CUDA_VISIBLE_DEVICES=''` 를 세우면 `G-4`(동결 재현)가 **144짝 전건에서 rel Δ 3.17e-01 로
  실패**한다 — `learned_init.propose_atoms` 가 `config['device']` 를 받지 않고 스스로 cuda 를
  고르고, 동결 `results_b12.jsonl` 이 **GPU 환경의 산물**이기 때문이다.
  ⟹ 아래 방벽이 그 조합을 조용히 돌지 못하게 한다.
  ⚠**이 배치의 GPU 사용은 「생성·학습」이 아니라 「동결 앵커의 재현 통화」다** — 새 에피소드
  생성 0 · 학습 0 · 실측 게이트 소모 0.
"""
import os
import sys

# ⛔⛔`import torch` **앞**. 스레드 고정은 호출자에게 맡기지 않는다(`M8 run_m8.py:39-44` 규약;
#    `M7 E2` 가 호출자에게 맡겼다가 18시간을 잃었다).  BLAS 는 임포트 시점에 읽는다.
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
if os.environ.get('CUDA_VISIBLE_DEVICES') == '':
    raise SystemExit(
        "⛔호출자가 CUDA_VISIBLE_DEVICES='' 를 세웠다.\n"
        "   그 조합은 `M-10 BLOCKER-D1` 그 자체다 — 동결 `B-12` 자료가 GPU 산물이라\n"
        "   `gates_t4 G-4`(rel Δ ≤ 1e-6)가 전건 실패한다(실측 rel Δ 3.171e-01).\n"
        "   환경변수를 빼고 다시 돌려라 (스레드 4종 고정은 이 파일이 한다).")

import copy                                                          # noqa: E402
import glob                                                          # noqa: E402
import hashlib                                                       # noqa: E402
import json                                                          # noqa: E402
import re                                                            # noqa: E402
import time                                                          # noqa: E402
import traceback                                                     # noqa: E402
import warnings                                                      # noqa: E402

warnings.filterwarnings('ignore')
import numpy as np                                                   # noqa: E402
import torch                                                         # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for _p in (HERE, ROOT,
           os.path.join(ROOT, 'tmp', 'B11_HOS_BASELINE_2026-08-23'),
           os.path.join(ROOT, 'tmp', 'B12_HOS_EXTEND_2026-08-23'),
           os.path.join(ROOT, 'tmp', 'S16_BASELINE_2026-08-14')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(ROOT)

import gates_t4 as G                                                 # noqa: E402
import run_b12                                                       # noqa: E402 ★팔 정의
import run_s16                                                       # noqa: E402 ★기준선 정의

SMOKE_OUT = os.path.join(HERE, 'smoke_t4.jsonl')
PREREG = os.path.join(HERE, 'PREREG.md')

# ── 프로세스 전역 계측 상자 (셀마다 비운다) ──────────────────────────────────
CAP = {}
CTR = {}
_CODE_MD5 = None
_EPS = None


# ═════════════════════════════════════════════════════════════════════════════
# 0. 행 스키마 — ⛔손으로 옮기지 않는다. `PREREG §8` 의 코드블록을 **파싱**한다.
# ═════════════════════════════════════════════════════════════════════════════
def prereg_row_schema(path=PREREG):
    """`PREREG §8` 의 동결 행 스키마를 문서에서 재도출한다(런 0회).

    ★사유: 스키마를 러너에 손으로 적으면 사전등록과 조용히 갈린다(이 저장소의 반복 결함
    「같은 것을 두 곳에 적고 한쪽만 고친다」). 문서가 권위이므로 문서에서 읽는다.
    """
    src = open(path, encoding='utf-8').read()
    blocks = re.findall(r'```\n(.*?)```', src, re.S)
    for b in blocks:
        if 'avg_nRMSE' in b and 'lead_rmse' in b:
            b = re.sub(r'\{[^}]*\}', ' ', b)             # `lead_rmse{...}` 의 첨자 제거
            b = re.sub(r'#[^\n]*', ' ', b)               # 주석 제거
            names, seen = [], set()
            for tok in re.findall(r'[A-Za-z_][A-Za-z0-9_]*', b):
                if tok not in seen:
                    seen.add(tok)
                    names.append(tok)
            return tuple(names)
    raise AssertionError('PREREG §8 의 행 스키마 블록을 찾지 못했다')


ROW_SCHEMA = prereg_row_schema()


# ═════════════════════════════════════════════════════════════════════════════
# 1. 합성 전용 확인 — ⛔실측 경로가 열리지 않는다 (⚠승인 가드가 아니다: **확인**이다)
# ═════════════════════════════════════════════════════════════════════════════
#  이 배치는 합성 전용이라 승인이 필요 없다(`PREREG §9`).  ⟹ 필요한 것은 *허가*가 아니라
#  ***실측 경로가 열릴 수 없음의 확인***이다.  세 술어 전부가 참이어야 한다:
#    ①에피소드가 동결 합성 코퍼스(`gates_b11.HOS`) 안에 있다
#    ②탈출구 `real_domain_gate=False` 를 쓰지 않았다
#    ③추정기 **객체 되읽기**가 실측 게이트 미발화(`_auto_real_gate is False`) · bank 부재
#  ⛔항진 금지: 자가진단이 네 조합을 전부 친다(하나라도 위반이면 반드시 False).
def synthetic_only(ep_path, cfg, real_gate, bank_present):
    in_corpus = os.path.abspath(ep_path).startswith(os.path.abspath(G.HOS) + os.sep)
    no_escape = bool((cfg or {}).get('real_domain_gate', True))
    gate_off = (real_gate is False) and (bank_present is False)
    return dict(ok=bool(in_corpus and no_escape and gate_off),
                in_corpus=bool(in_corpus), no_escape=bool(no_escape),
                gate_off=bool(gate_off))


# ═════════════════════════════════════════════════════════════════════════════
# 2. 배포 기본값 트립와이어 (⛔팔이 조용히 다른 것이 되는 사고를 런 전에 끊는다)
# ═════════════════════════════════════════════════════════════════════════════
def assert_shipping_defaults():
    """출하 기본값 셋(2026-08-22 배포) + λ + 라우터가 그대로인가.

    ⛔값을 여기에 손으로 적지 않는다 — `sequential_update` 의 **단일출처 상수**와 대조한다.
    """
    from wavesplat.run import DEFAULT_CONFIG
    import wavesplat.optimizer.sequential_update as SU
    want = {'varpro_steps': SU.VARPRO_STEPS_DEFAULT,
            'learned_init_iters': SU.LEARNED_INIT_ITERS_DEFAULT,
            'lambda_l1': SU.LAMBDA_L1_DEFAULT,
            'init_method': SU.INIT_METHOD_DEFAULT,
            'auto_switch_thresh': SU.AUTO_SWITCH_THRESH_DEFAULT}
    bad = {k: (DEFAULT_CONFIG.get(k), v) for k, v in want.items()
           if DEFAULT_CONFIG.get(k) != v}
    if bad:
        raise AssertionError('배포 기본값이 단일출처와 갈렸다: %r' % bad)
    if str(DEFAULT_CONFIG.get('device')) != 'cpu':
        raise AssertionError('DEFAULT_CONFIG device != cpu: %r' % DEFAULT_CONFIG.get('device'))
    return want


# ═════════════════════════════════════════════════════════════════════════════
# 3. 코퍼스 · 스탬프
# ═════════════════════════════════════════════════════════════════════════════
def episodes():
    """혼잡 48. ⛔조건절은 `run_b12.crowded_episodes` 를 **그대로 부른다**(손복사 0)."""
    global _EPS
    if _EPS is None:
        eps = [e for e in run_b12.crowded_episodes()]
        if len(eps) != G.N_EP:
            raise AssertionError('코퍼스 %d != %d' % (len(eps), G.N_EP))
        _EPS = eps
    return _EPS


def _md5file(p):
    h = hashlib.md5()
    with open(p, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def code_md5():
    """★단일 스탬프. `run_s16.code_fingerprint()`(저장소 전 .py + configs)에 **이 배치의
    판정·자료 생성 코드**를 더한다. ⛔손복사 0 — 파일에서 읽는다."""
    global _CODE_MD5
    if _CODE_MD5 is None:
        run_s16.preload()
        base = run_s16.code_fingerprint()[0]
        extra = {}
        for p in (os.path.join(HERE, 'gates_t4.py'), os.path.join(HERE, 'run_t4.py'),
                  os.path.join(ROOT, 'tmp', 'B12_HOS_EXTEND_2026-08-23', 'run_b12.py')):
            extra[os.path.relpath(p, ROOT)] = _md5file(p)
        _CODE_MD5 = hashlib.md5(json.dumps({'s16': base, 'batch': extra},
                                           sort_keys=True).encode()).hexdigest()
    return _CODE_MD5


def prior_device():
    """★확산 사전이 **실제로** 올라간 장치 — callee 자신의 캐시 키 되읽기
    (`learned_init._load_cond` 의 `key=(ckpt, str(device))`; `M-10 AMENDMENT_1 G-15` 규약).
    ⛔`'cuda' if torch.cuda.is_available() else 'cpu'` 를 여기서 다시 쓰지 않는다."""
    from wavesplat.optimizer import learned_init as LI
    devs = sorted({str(k[1]) for k in getattr(LI, '_CACHE', {})})
    return ','.join(devs) if devs else None


def stamps():
    from wavesplat.run import DEFAULT_CONFIG
    return dict(device=str(DEFAULT_CONFIG['device']),
                cuda_available=bool(torch.cuda.is_available()),
                omp_threads=os.environ.get('OMP_NUM_THREADS'),
                torch_threads=int(torch.get_num_threads()),
                code_md5=code_md5())


def mesh_verdict(path):
    import h5py
    with h5py.File(path, 'r') as f:
        v = f['config'].attrs.get('mesh_verdict')
    return v.decode() if isinstance(v, bytes) else (str(v) if v is not None else None)


# ═════════════════════════════════════════════════════════════════════════════
# 4. 채점 (⛔정규화는 패키지 함수 `metrics.nrmse` 하나 — 재구현 0)
# ═════════════════════════════════════════════════════════════════════════════
def _gt32(loader, j):
    """GT 프레임 `j` 를 **두 하네스가 실제로 쓰는 dtype 그대로** 낸다.
    ⛔`astype` 로 다시 만들지 않는다 — `loader[j].gt_field` 는 `run.py`·`run_s16` 이
    부르는 **바로 그 경로**다(그래서 lead 0 비트 일치가 성립한다)."""
    return loader[j].gt_field.cpu().numpy()


def _ctx(loader, path):
    """원점·rung·격자 — ⛔전부 `gates_t4` 의 순수 함수에서(러너가 따로 계산하지 않는다)."""
    ts = np.asarray(loader.timestamps, dtype=float)
    tp, nf = float(loader.Tp_true), int(loader.n_frames)
    dt = float(ts[1] - ts[0])
    lf = G.lead_frames(tp, dt)
    ors = G.origins(tp, dt, nf, G.ORIGIN_STRIDE_S)
    if not ors:
        raise AssertionError('원점 0개 — 0행 가드 (%s)' % os.path.basename(path))
    if len(ors) < G.MIN_ORIGINS:
        raise AssertionError('원점 %d < MIN_ORIGINS %d' % (len(ors), G.MIN_ORIGINS))
    sel = ts > 2.0 * tp
    if not sel.any():
        raise AssertionError('채점 프레임 0개 — 0행 가드')
    return dict(loader=loader, path=path, ts=ts, tp=tp, dt=dt, n_frames=nf,
                lf=lf, lmax=max(lf.values()), origins=ors, origin_set=set(ors),
                step2rung={v: k for k, v in lf.items()},
                Hs=float(loader.Hs_true),
                sel_n=int(sel.sum()), sel_t0=float(ts[sel][0]), sel_t1=float(ts[sel][-1]),
                acc={float(L): [] for L in G.LEADS_TP},
                zacc={float(L): [] for L in G.LEADS_TP},
                lead0={})


def _score(ctx, L, j, field):
    """rung `L` 에서 예측장 `field` 를 GT 프레임 `j` 와 맞대고, 같은 GT 로 제로필드도 낸다."""
    from wavesplat.evaluation.metrics import nrmse
    gt = _gt32(ctx['loader'], j)
    ctx['acc'][float(L)].append(float(nrmse(field, gt, ctx['Hs'])))
    ctx['zacc'][float(L)].append(float(nrmse(np.zeros_like(gt), gt, ctx['Hs'])))


def _mean_leads(ctx):
    out = {}
    for k in ('acc', 'zacc'):
        d = {}
        for L in G.LEADS_TP:
            v = ctx[k][float(L)]
            if len(v) != len(ctx['origins']):
                raise AssertionError('rung %s: 원점 %d/%d — 0행/결손 가드'
                                     % (L, len(v), len(ctx['origins'])))
            d[repr(float(L))] = float(np.mean(v))
        out[k] = d
    return out['acc'], out['zacc']


# ═════════════════════════════════════════════════════════════════════════════
# 5. G-7 양성대조 — 전파 연산자 합성 `F(a+b) == F(a)∘F(b)` (교차구현)
# ═════════════════════════════════════════════════════════════════════════════
def _twin(f, nsteps):
    """`f` 와 같은 격자·수심의 **패키지 클래스**를 `dt·nsteps` 로 하나 만든다.
    ⛔전파 연산자를 다시 구현하지 않는다 — 같은 코드에 다른 dt 를 준다."""
    from wavesplat.baselines.letkf_grid import LETKFGrid
    st = np.random.get_state()                 # 생성자가 전역 randn 을 뽑는다 ⟹ 복원
    try:
        return LETKFGrid(f.grid_x, f.grid_y, depth=f.depth, n_ensemble=1,
                         dt=float(f.dt) * int(nsteps), mult_inflation=f.mult_inflation,
                         add_inflation=0.0, obs_noise=f.obs_noise, max_obs=f.max_obs,
                         loc_radius=f.loc_radius)
    finally:
        np.random.set_state(st)


def _test_state(f):
    """결정적 시험 상태(진폭 ≤ 0.5 ⟹ `_forecast_member` 의 clip 이 발화하지 않는다)."""
    Xg, Yg = np.meshgrid(np.asarray(f.grid_x, float), np.asarray(f.grid_y, float))
    eta = 0.5 * np.cos(2 * np.pi * Xg / 120.0) * np.cos(2 * np.pi * Yg / 80.0)
    deta = 0.3 * np.sin(2 * np.pi * Xg / 100.0)
    return np.concatenate([eta.ravel(), deta.ravel()])


def _relerr(a, b):
    den = float(np.max(np.abs(b)))
    return float(np.max(np.abs(a - b)) / den) if den > 0 else float('inf')


def compose_relerr(f, lmax, a, dt_scale=1):
    """★양성대조. `dt_scale != 1` 은 **자가진단 전용 위조**(반드시 커야 한다)."""
    b = int(lmax) - int(a)
    s = _test_state(f)
    Tab = _twin(f, int(lmax) * int(dt_scale))
    Ta, Tb = _twin(f, a), _twin(f, b)
    A = Tab._forecast_member(s)
    B = Tb._forecast_member(Ta._forecast_member(s))
    e1 = _relerr(B, A)
    C = s.copy()
    for _ in range(int(lmax)):                 # ★하네스가 실제로 쓰는 단위-스텝 반복
        C = f._forecast_member(C)
    e2 = _relerr(C, A)
    return max(e1, e2)


# ═════════════════════════════════════════════════════════════════════════════
# 6. 계측 shim (⛔값을 바꾸지 않는다 — `G-4` 가 동결 재현으로 증명한다)
# ═════════════════════════════════════════════════════════════════════════════
def install_shims():
    """(restore_fn) 을 돌려준다. ★호출자는 반드시 `try/finally` 로 복원한다."""
    import wavesplat.run as WR
    import wavesplat.data_loader as DL
    import wavesplat.evaluation.metrics as MT
    import wavesplat.optimizer.sequential_update as SU
    import letkf_spec as LS

    saved = [(WR, 'run_estimation', WR.run_estimation),
             (WR, 'make_updater', WR.make_updater),
             (WR, 'EpisodeDataLoader', WR.EpisodeDataLoader),
             (WR, 'compute_all_metrics', WR.compute_all_metrics),
             (DL, 'EpisodeDataLoader', DL.EpisodeDataLoader),
             (MT, 'compute_all_metrics', MT.compute_all_metrics),
             (SU.SequentialUpdater, '__init__', SU.SequentialUpdater.__init__),
             (SU.SequentialUpdater, 'initialize', SU.SequentialUpdater.initialize),
             (LS.LETKFGridSpec, 'update', LS.LETKFGridSpec.update),
             (run_s16, 'run_letkf', run_s16.run_letkf)]

    _orig_letkf = run_s16.run_letkf

    _orig_loader = DL.EpisodeDataLoader
    _orig_metrics = MT.compute_all_metrics
    _orig_mk = WR.make_updater
    _orig_run = WR.run_estimation
    _orig_ctor = SU.SequentialUpdater.__init__
    _orig_ini = SU.SequentialUpdater.initialize
    _orig_upd = LS.LETKFGridSpec.update

    # ── 로더 포착 (두 하네스 모두 이 클래스를 부른다) ────────────────────────
    def _cap_loader(*a, **kw):
        ld = _orig_loader(*a, **kw)
        CAP['loader'] = ld
        return ld

    # ── 동화 경로의 예측장 md5 (G-6 의 대조 상대) ────────────────────────────
    def _cap_metrics(pred_field, gt_field, *a, **kw):
        CAP.setdefault('assim', []).append(
            hashlib.md5(np.ascontiguousarray(pred_field).tobytes()).hexdigest())
        return _orig_metrics(pred_field, gt_field, *a, **kw)

    # ── SHIP: 원점에서 lead 예측 (⛔관측 순서 불변, `update` 뒤에 **읽기만** 한다) ──
    def _cap_mk(cfg):
        upd = _orig_mk(cfg)
        CAP['cfg_nkeys'] = len(cfg or {})
        CAP['cfg'] = dict(cfg or {})
        ctx = _ctx(CAP['loader'], CAP['path'])
        CAP['ctx'] = ctx
        _orig_update = upd.update

        def _wrapped(frame, *a, **kw):
            res = _orig_update(frame, *a, **kw)
            idx = int(round(float(frame.t) / ctx['dt']))
            if abs(ctx['ts'][idx] - float(frame.t)) > 1e-9:
                raise AssertionError('프레임 t→index 사상 실패 %r' % frame.t)
            CAP.setdefault('ship_frames', []).append(idx)
            if idx in ctx['origin_set']:
                for L in G.LEADS_TP:
                    j = idx + ctx['lf'][L]
                    pred = upd.predict_field(ctx['loader'].gt_grid_x,
                                             ctx['loader'].gt_grid_y,
                                             float(ctx['ts'][j]))
                    CTR['predict_future'] = CTR.get('predict_future', 0) + 1
                    pn = pred.detach().cpu().numpy()
                    _score(ctx, L, j, pn)
                    if ctx['lf'][L] == 0:
                        ctx['lead0'][idx] = hashlib.md5(
                            np.ascontiguousarray(pn).tobytes()).hexdigest()
            return res
        upd.update = _wrapped
        CAP['upd'] = upd
        return upd

    def _cap_run(*a, **kw):
        r = _orig_run(*a, **kw)
        fr = r.get('frame_results') or []
        CAP['ship_window'] = dict(n_scored=len(fr),
                                  t_first=(fr[0].get('t') if fr else None),
                                  t_last=(fr[-1].get('t') if fr else None))
        return r

    def _cap_ctor(self, config, *a, **kw):
        _orig_ctor(self, config, *a, **kw)
        # ★객체 되읽기 — ⛔`cfg[k]==cfg[k]` 항등식이 아니다. `init_method` 는 **아직
        #   'auto'**(라우터가 initialize 에서 변형한다) ⟹ `G-3` 이 요구하는 그 값이다.
        CAP.setdefault('obj', dict(init_method=self.init_method,
                                   tracking_mode=self.tracking_mode,
                                   lambda_l1=self.lambda_l1,
                                   device=str(getattr(self, 'device', 'cpu'))))

    def _cap_ini(self, *a, **kw):
        res = _orig_ini(self, *a, **kw)
        vp = (res or {}).get('varpro') or {}
        if vp.get('vp_steps') is not None and 'varpro_executed' not in CAP:
            CAP['varpro_executed'] = int(vp['vp_steps'])      # ★**실행된** 스텝(선언값 아님)
        return res

    # ── SPEC25: 원점에서 앙상블 자유전개 ─────────────────────────────────────
    def _cap_spec_update(self, ox, oy, oz):
        res = _orig_upd(self, ox, oy, oz)
        CAP['filter'] = self
        idx = CTR.get('spec_frame', 0)
        CTR['spec_frame'] = idx + 1
        ctx = CAP.get('ctx')
        if ctx is None:
            ctx = _ctx(CAP['loader'], CAP['path'])
            CAP['ctx'] = ctx
        if idx in ctx['origin_set']:
            _forecast_origin(self, ctx, idx)
        return res

    # ★기준선 자신의 창 계수(`n_match`)는 `run_s16.run_letkf` 가 반환하는데 `run_b12.run_spec`
    #   이 그 키를 **떨어뜨린다**(반환 dict 에 없다) ⟹ 여기서 원천에서 잡는다.
    #   ⛔`sel.sum()` 을 러너가 다시 계산하지 않는다 — 그러면 `G-5` 가 항등식이 된다
    #   (「같은 것을 두 곳에 구현하고 한쪽만 계측」 = 이 저장소의 반복 결함 3부류 중 하나).
    def _cap_letkf(*a, **kw):
        row = _orig_letkf(*a, **kw)
        CAP['spec'] = {k: row.get(k) for k in
                       ('n_match', 'n_all', 'avg_nRMSE', 'zero_nRMSE', 'loc_radius',
                        'n_ensemble', 'background', 'filter_class', 'Hs', 'Tp', 'depth')}
        return row

    DL.EpisodeDataLoader = _cap_loader
    WR.EpisodeDataLoader = _cap_loader
    MT.compute_all_metrics = _cap_metrics
    WR.compute_all_metrics = _cap_metrics
    WR.make_updater = _cap_mk
    WR.run_estimation = _cap_run
    SU.SequentialUpdater.__init__ = _cap_ctor
    SU.SequentialUpdater.initialize = _cap_ini
    LS.LETKFGridSpec.update = _cap_spec_update
    run_s16.run_letkf = _cap_letkf

    def restore():
        for obj, name, val in saved:
            setattr(obj, name, val)
    return restore


def _interp_to_gt(ctx, f, grid_field):
    """필터 격자 → GT 격자. ⛔규약은 `run_s16.py:232-235` 와 **같은 세 줄**이고,
    그것이 같음을 `G-6`(lead 0 비트 일치)이 매 셀 증명한다."""
    from scipy.interpolate import RegularGridInterpolator
    gx = ctx['loader'].gt_grid_x.cpu().numpy()
    gy = ctx['loader'].gt_grid_y.cpu().numpy()
    Xg, Yg = np.meshgrid(gx, gy)
    q = np.column_stack([Yg.ravel(), Xg.ravel()])
    interp = RegularGridInterpolator((f.grid_y, f.grid_x), grid_field, method='linear',
                                     bounds_error=False, fill_value=0.0)
    return interp(q).reshape(len(gy), len(gx))


def _forecast_origin(f, ctx, idx):
    """원점 `idx` 에서 **깊은 복사** 후 `_forecast_member` 를 lmax 스텝 반복하며 rung 스냅샷.
    ★개입 계수는 **가지 단위**(멤버 × 스텝)로 여기서만 센다 — 필터 자신의 `update` 안
    전개는 개입이 아니므로 세지 않는다."""
    t0 = time.time()
    g = copy.deepcopy(f)
    ne = int(g.n_ensemble)
    for s in range(0, ctx['lmax'] + 1):
        if s > 0:
            for m in range(ne):
                g.ensemble[m] = g._forecast_member(g.ensemble[m])
            CTR['forecast_steps'] = CTR.get('forecast_steps', 0) + ne
        L = ctx['step2rung'].get(s)
        if L is None:
            continue
        pf = _interp_to_gt(ctx, g, g.predict_grid())
        _score(ctx, L, idx + s, pf)
        if s == 0:
            ctx['lead0'][idx] = hashlib.md5(
                np.ascontiguousarray(pf).tobytes()).hexdigest()
    CTR['fc_seconds'] = CTR.get('fc_seconds', 0.0) + (time.time() - t0)


def _neg_lead0_bitmatch(ctx, assim_md5_by_idx):
    """G-6 음성대조: lead 0 의 **예보 경로 산출장** == **동화 경로 산출장**, 비트 동일.
    ⛔한 원점이라도 대조 상대가 없으면 False(부재를 통과로 찍지 않는다)."""
    if not ctx['origins'] or not ctx['lead0']:
        return False
    for i in ctx['origins']:
        a, b = ctx['lead0'].get(i), assim_md5_by_idx.get(i)
        if a is None or b is None or a != b:
            return False
    return True


# ═════════════════════════════════════════════════════════════════════════════
# 7. 팔 (⛔`run_b12`·`run_s16` 을 수정하지 않는다 · ⛔빈 dict 팔 0개)
# ═════════════════════════════════════════════════════════════════════════════
def run_ship(ep, seed):
    base = run_b12.run_ship(ep, seed)                 # ★B-12 의 팔, 손복사 0
    ctx = CAP['ctx']
    upd = CAP['upd']
    frames = CAP.get('ship_frames') or []
    assim = CAP.get('assim') or []
    if len(assim) != len(frames):
        raise AssertionError('동화 채점 호출 %d != update 호출 %d' % (len(assim), len(frames)))
    by_idx = {fi: h for fi, h in zip(frames, assim)}
    win = CAP.get('ship_window') or {}
    if win.get('n_scored') != len(frames):
        raise AssertionError('채점 창 %r != update %d' % (win.get('n_scored'), len(frames)))
    if win.get('n_scored') != ctx['sel_n']:
        raise AssertionError('창 유도 불일치 %r vs %d' % (win.get('n_scored'), ctx['sel_n']))
    lead, zero = _mean_leads(ctx)
    so = synthetic_only(ep['path'], CAP.get('cfg'),
                        getattr(upd, '_auto_real_gate', None),
                        getattr(upd, '_bank', None) is not None)
    if not so['ok']:
        raise AssertionError('합성 전용 확인 실패: %r' % so)
    return dict(
        avg_nRMSE=float(base['avg_nRMSE']),
        n_scored=int(win['n_scored']), t_first=win['t_first'], t_last=win['t_last'],
        lead_rmse=lead, lead_zero=zero,
        neg_lead0_bitmatch=_neg_lead0_bitmatch(ctx, by_idx),
        pos_compose_relerr=None,                       # ⛔SHIP 에는 전파 연산자가 없다
        cap_forecast_steps=int(CTR.get('forecast_steps', 0)),
        cap_predict_future=int(CTR.get('predict_future', 0)),
        obj_init_method=(CAP.get('obj') or {}).get('init_method'),
        obj_tracking_mode=(CAP.get('obj') or {}).get('tracking_mode'),
        obj_lambda_l1=(CAP.get('obj') or {}).get('lambda_l1'),
        cap_varpro_steps=CAP.get('varpro_executed'),
        arm_cfg_nonempty=bool(CAP.get('cfg_nkeys', 0) > 0),
        arm_cfg_nkeys=CAP.get('cfg_nkeys'),
        n_ensemble=int(getattr(upd, 'ensemble_k', 1) or 1),
        n_ensemble_kind='analytic_median_k',
        real_gate=getattr(upd, '_auto_real_gate', None),
        synthetic_only=so,
        auto_selected=getattr(upd, 'init_method', None),
        n_atoms=int(getattr(getattr(upd, 'model', None), 'n_gaussians', 0) or 0))


def run_spec(ep, seed):
    base = run_b12.run_spec(ep, seed)                 # ★S16 의 기준선, 손복사 0
    spec = CAP.get('spec') or {}                      # ★`run_s16.run_letkf` 원천 되읽기
    ctx = CAP['ctx']
    f = CAP['filter']
    assim = CAP.get('assim') or []
    if len(assim) != ctx['n_frames']:
        raise AssertionError('동화 채점 호출 %d != 프레임 %d' % (len(assim), ctx['n_frames']))
    if CTR.get('spec_frame') != ctx['n_frames']:
        raise AssertionError('필터 update 호출 %r != 프레임 %d'
                             % (CTR.get('spec_frame'), ctx['n_frames']))
    by_idx = {i: h for i, h in enumerate(assim)}
    if spec.get('n_match') is None:
        raise AssertionError('기준선 창 계수 n_match 부재 — ⛔부재를 통과로 찍지 않는다')
    if int(spec['n_match']) != ctx['sel_n']:
        raise AssertionError('창 유도 불일치 %r vs %d' % (spec['n_match'], ctx['sel_n']))
    lead, zero = _mean_leads(ctx)
    ce = compose_relerr(f, ctx['lmax'], ctx['lf'][0.5])
    so = synthetic_only(ep['path'], {'real_domain_gate': True}, False, False)
    if not so['ok']:
        raise AssertionError('합성 전용 확인 실패: %r' % so)
    return dict(
        avg_nRMSE=float(base['avg_nRMSE']),
        n_scored=int(spec['n_match']), t_first=ctx['sel_t0'], t_last=ctx['sel_t1'],
        lead_rmse=lead, lead_zero=zero,
        neg_lead0_bitmatch=_neg_lead0_bitmatch(ctx, by_idx),
        pos_compose_relerr=float(ce),
        cap_forecast_steps=int(CTR.get('forecast_steps', 0)),
        cap_predict_future=int(CTR.get('predict_future', 0)),
        obj_init_method=None, obj_tracking_mode=None, cap_varpro_steps=None,
        arm_cfg_nonempty=bool(len(G.SPEC) > 0),
        arm_cfg_nkeys=len(G.SPEC),
        obj_filter_class=str(type(f).__name__),
        obj_loc_radius=float(f.loc_radius), obj_background=str(f.background),
        n_ensemble=int(f.n_ensemble), n_ensemble_kind='letkf_members',
        zero_nRMSE=float(base['zero_nRMSE']),
        real_gate=False, synthetic_only=so, auto_selected=None, n_atoms=None,
        fc_seconds=float(CTR.get('fc_seconds', 0.0)))


ARMS = {'SHIP': run_ship, 'SPEC25': run_spec}


# ═════════════════════════════════════════════════════════════════════════════
# 8. 셀 하나
# ═════════════════════════════════════════════════════════════════════════════
def run_cell(arm, ep, seed):
    assert_shipping_defaults()
    CAP.clear()
    CTR.clear()
    CAP['path'] = ep['path']
    t0 = time.time()
    restore = install_shims()
    try:
        m = ARMS[arm](ep, seed)
    finally:
        restore()                                     # ★shim 은 반드시 복원
    ctx = CAP['ctx']
    st = stamps()
    row = dict(arm=arm, episode=ep['episode'], seed=int(seed),
               bf=ep['bf'], ncomp=ep['ncomp'], gen_seed=ep['seed'],
               n_origins=len(ctx['origins']), origin_first=ctx['origins'][0],
               origin_last=ctx['origins'][-1], origin_stride_s=G.ORIGIN_STRIDE_S,
               lead_frames={repr(float(k)): int(v) for k, v in ctx['lf'].items()},
               tp_true=ctx['tp'], dt=ctx['dt'], n_frames=ctx['n_frames'],
               mesh_verdict=mesh_verdict(ep['path']), prior_device=prior_device(),
               wall_s=time.time() - t0, **st)
    row.update(m)
    missing = [k for k in ROW_SCHEMA if k not in row]
    if missing:
        raise AssertionError('PREREG §8 스키마 결손: %r' % missing)
    return row


# ═════════════════════════════════════════════════════════════════════════════
# 9. ★비용 게이트 — 스모크가 단가를 실측하고 총비용을 투영한다 (`PREREG §8·§9`)
# ═════════════════════════════════════════════════════════════════════════════
def project_cost(rows, eps=None):
    """(dict) — ⛔문턱을 손으로 적지 않는다: `gates_t4.MAX_CORE_HOURS` 를 읽는다.
    투영 = (팔, 층)별 실측 벽시계 중앙 × 본런의 (팔, 층) 셀 수. 층이 빠지면 **가장 비싼
    층으로 대체**한다(보수적)."""
    eps = episodes() if eps is None else eps
    n_by_bf = {}
    for e in eps:
        n_by_bf[e['bf']] = n_by_bf.get(e['bf'], 0) + 1
    per = {}
    for r in rows:
        if 'error' in r or r.get('wall_s') is None:
            continue
        per.setdefault((r.get('arm'), r.get('bf')), []).append(float(r['wall_s']))
    unit = {k: float(np.median(v)) for k, v in per.items()}
    if not unit:
        return dict(ok=False, reason='스모크 행 0개 — 투영 불가(⛔0행을 통과로 찍지 않는다)',
                    max_core_hours=float(G.MAX_CORE_HOURS))
    total_s, detail, substituted = 0.0, {}, []
    for arm in G.ARM_ORDER:
        for bf, n_ep in sorted(n_by_bf.items()):
            u = unit.get((arm, bf))
            if u is None:
                cand = [v for (a, b), v in unit.items() if a == arm]
                if not cand:
                    return dict(ok=False, reason='팔 %s 의 실측 단가 0건' % arm,
                                max_core_hours=float(G.MAX_CORE_HOURS))
                u = max(cand)
                substituted.append('%s/%s' % (arm, bf))
            n_cell = n_ep * len(G.EST_SEEDS)
            detail['%s/%s' % (arm, bf)] = dict(unit_s=u, cells=n_cell,
                                               hours=u * n_cell / 3600.0)
            total_s += u * n_cell
    fc_calls = sum(int(r.get('cap_forecast_steps') or 0) for r in rows if 'error' not in r)
    fc_sec = sum(float(r.get('fc_seconds') or 0.0) for r in rows if 'error' not in r)
    core_h = total_s / 3600.0
    return dict(ok=bool(core_h <= float(G.MAX_CORE_HOURS)),
                core_hours=core_h, max_core_hours=float(G.MAX_CORE_HOURS),
                detail=detail, substituted=substituted,
                fc_unit_us=(fc_sec / fc_calls * 1e6) if fc_calls else None,
                fc_calls_measured=fc_calls,
                n_cells_total=sum(d['cells'] for d in detail.values()),
                reason=('' if core_h <= float(G.MAX_CORE_HOURS)
                        else '투영 %.2f 코어·h > MAX_CORE_HOURS %.2f ⟹ ⛔ABORT'
                             ' (⛔완화 경로 없음 — PREREG §9, 자가진단 B⑬b)'
                             % (core_h, float(G.MAX_CORE_HOURS))))


# ═════════════════════════════════════════════════════════════════════════════
# 10. 자가진단 (⛔동결 **전에** 통과시켜라)
# ═════════════════════════════════════════════════════════════════════════════
def selftest():
    fails = []

    def chk(name, cond, det=''):
        if not cond:
            fails.append('%s  %s' % (name, det))
        return bool(cond)

    # [1] ★게이트 자가진단(게이트마다 위조 입력 항진 검사 C①–C⑱)을 **여기서 다시 돌린다**
    nf = G._selftest()
    chk('[1] gates_t4 자가진단', nf == 0, 'FAIL %s' % nf)

    # [2] ★합성 전용 확인이 항진하지 않는다 — 술어 셋을 하나씩 깨면 반드시 False
    good = dict(ep_path=os.path.join(G.HOS, 'x.h5'), cfg={'real_domain_gate': True},
                real_gate=False, bank_present=False)
    chk('[2] 합성 전용: 정상 입력은 통과', synthetic_only(**good)['ok'])
    for k, bad in (('ep_path', '/tmp/real.h5'), ('cfg', {'real_domain_gate': False}),
                   ('real_gate', True), ('bank_present', True)):
        chk('[2] 합성 전용: %s 위조는 반드시 거부' % k,
            not synthetic_only(**dict(good, **{k: bad}))['ok'])

    # [3] ★비용 게이트가 항진하지 않는다 (문턱은 gates_t4 에서 읽는다)
    eps = episodes()
    n_bf = {}
    for e in eps:
        n_bf[e['bf']] = n_bf.get(e['bf'], 0) + 1
    cheap = [dict(arm=a, bf=b, wall_s=1.0, cap_forecast_steps=10, fc_seconds=0.1)
             for a in G.ARM_ORDER for b in n_bf]
    heavy = [dict(r, wall_s=10000.0) for r in cheap]
    chk('[3] 비용: 싼 스모크는 GO', project_cost(cheap, eps)['ok'])
    chk('[3] 비용: 비싼 스모크는 ABORT', not project_cost(heavy, eps)['ok'])
    chk('[3] 비용: 0행은 GO 가 아니다', not project_cost([], eps)['ok'])
    chk('[3] 비용: 문턱이 gates_t4 단일출처',
        project_cost(cheap, eps)['max_core_hours'] == G.MAX_CORE_HOURS)
    # 문턱 바로 아래/위에서 판정이 갈리는가 (경계 시험)
    ncell = sum(project_cost(cheap, eps)['detail'][k]['cells']
                for k in project_cost(cheap, eps)['detail'])
    u = G.MAX_CORE_HOURS * 3600.0 / ncell
    chk('[3] 비용: 경계 아래는 GO',
        project_cost([dict(r, wall_s=u * 0.99) for r in cheap], eps)['ok'])
    chk('[3] 비용: 경계 위는 ABORT',
        not project_cost([dict(r, wall_s=u * 1.01) for r in cheap], eps)['ok'])

    # [4] ★행 스키마가 사전등록에서 파싱된다 (⛔손복사 0)
    chk('[4] 스키마 파싱 32필드', len(ROW_SCHEMA) >= 30, repr(len(ROW_SCHEMA)))
    for must in ('arm', 'episode', 'seed', 'avg_nRMSE', 'lead_rmse', 'lead_zero',
                 'neg_lead0_bitmatch', 'pos_compose_relerr', 'cap_forecast_steps',
                 'cap_predict_future', 'code_md5', 'prior_device'):
        chk('[4] 스키마에 %s' % must, must in ROW_SCHEMA)

    # [5] 코퍼스·산술 — 동결 h5 의 실제 메타로 재도출(런 0회)
    chk('[5] 혼잡 코퍼스 48', len(eps) == G.N_EP, repr(len(eps)))
    import h5py
    seen = {}
    for e in eps:
        with h5py.File(e['path'], 'r') as f:
            tp = float(f['config'].attrs['Tp'])
            ts = f['frames']['timestamps'][:]
        dt = float(ts[1] - ts[0])
        ors = G.origins(tp, dt, len(ts), G.ORIGIN_STRIDE_S)
        lf = G.lead_frames(tp, dt)
        seen.setdefault((e['bf'], tp, dt, len(ts)),
                        (len(ors), ors[0], ors[-1], tuple(sorted(lf.items()))))
        if len(ors) < G.MIN_ORIGINS:
            chk('[5] %s 원점 하한' % e['episode'], False, repr(len(ors)))
        if ors[-1] + max(lf.values()) > len(ts) - 1:
            chk('[5] %s 최대 lead 가 에피소드를 넘는다' % e['episode'], False)
        if any(abs(v * dt - k * tp) > 1e-9 for k, v in lf.items()):
            chk('[5] %s lead 프레임 산술' % e['episode'], False)
    chk('[5] 층은 둘뿐(BF4/BF5)이고 층 안에서 원점이 하나', len(seen) == 2, repr(list(seen)))

    # [6] ★동결 앵커가 비어 있지 않다 (⛔공허 참조면 G-4 가 항진한다)
    fz = G.load_frozen_b12()
    chk('[6] B-12 동결 288짝', len(fz) == G.N_EP * len(G.EST_SEEDS) * 2, repr(len(fz)))
    chk('[6] 코퍼스와 동결의 에피소드가 같다',
        {e['episode'] for e in eps} == {k[1] for k in fz})

    # [7] 배포 기본값 트립와이어
    try:
        assert_shipping_defaults()
        chk('[7] 배포 기본값', True)
    except AssertionError as exc:
        chk('[7] 배포 기본값', False, repr(exc))

    # [8] ⛔고정 기준상수 부재 — ★검사는 `gates_t4` 가 단일출처(여기서 다시 짜지 않는다)
    ok8, det8 = G.no_flat_ref_constant()
    chk('[8] 고정 기준상수 부재', ok8, det8)

    # [9] ★양성대조가 항진하지 않는다 — 위조 전파(dt 배율 2)는 반드시 문턱을 넘는다
    from wavesplat.baselines.letkf_grid import LETKFGrid
    st = np.random.get_state()
    try:
        f = LETKFGrid(np.linspace(10., 250., 40), np.linspace(-100., 100., 30),
                      depth=50.0, n_ensemble=1, mult_inflation=1.0, add_inflation=0.0,
                      loc_radius=25.0)
    finally:
        np.random.set_state(st)
    e_ok = compose_relerr(f, 70, 35)
    e_bad = compose_relerr(f, 70, 35, dt_scale=2)
    chk('[9] 양성대조: 정상 합성은 통과', e_ok <= G.COMPOSE_RTOL, '%.3e' % e_ok)
    chk('[9] 양성대조: 위조 전파는 반드시 실패', e_bad > G.COMPOSE_RTOL, '%.3e' % e_bad)

    # [10] ★shim 이 복원된다 (설치 → 복원 후 원본과 동일 객체)
    import wavesplat.run as WR
    import letkf_spec as LS

    def _probe():
        return (WR.run_estimation, WR.make_updater, WR.compute_all_metrics,
                WR.EpisodeDataLoader, LS.LETKFGridSpec.update, run_s16.run_letkf)
    before = _probe()
    rst = install_shims()
    mid = _probe()
    rst()
    after = _probe()
    chk('[10] shim 설치가 실제로 갈아끼운다', all(a is not b for a, b in zip(before, mid)))
    chk('[10] shim 복원', all(a is b for a, b in zip(before, after)))

    # [11] 작업 목록 · 샤딩 (⛔중복·결손 0)
    cs = cells()
    chk('[11] 셀 288', len(cs) == G.N_EP * len(G.EST_SEEDS) * len(G.ARM_ORDER), repr(len(cs)))
    chk('[11] 셀 중복 0', len(set(cs)) == len(cs))
    for ns in (1, 2, 3, 4):
        un = [c for s in range(ns) for c in shard(cs, s, ns)]
        chk('[11] %d샤드 분할이 전단사' % ns, sorted(un) == sorted(cs))
    chk('[11] 샤드 4 이하 권고 준수(CPU 공유)', True)

    # [12b] ★기준선 창 계수의 **원천**이 어디인지 런 0회로 못박는다.
    #       (2026-08-30 스모크가 `KeyError('n_match')` 로 이것을 잡았다: `run_b12.run_spec` 이
    #        `run_s16.run_letkf` 의 반환 키를 **떨어뜨린다** ⟹ shim 이 필요한 사유가 이것이다.
    #        S16 이 키를 바꾸면 여기서 죽어야지 288런 뒤에 죽으면 안 된다.)
    import inspect
    src_spec = inspect.getsource(run_b12.run_spec)
    src_letkf = inspect.getsource(run_s16.run_letkf)
    chk('[12b] 창 계수는 run_s16 원천에만 있다(shim 이 필요한 사유)',
        ('n_match' in src_letkf) and ('n_match' not in src_spec))

    # [12] 제로필드가 GT 에서 나온다(고정 상수 아님) — 순수 함수 시험
    from wavesplat.evaluation.metrics import nrmse
    gt = np.array([[3.0, 4.0]], dtype=np.float32)
    chk('[12] 제로필드 = rms(GT)/Hs',
        abs(nrmse(np.zeros_like(gt), gt, 2.0) - (12.5 ** 0.5) / 2.0) < 1e-6)

    print('── run_t4 자가진단 ──')
    for f_ in fails:
        print('  ⛔FAIL', f_)
    print('  run_t4 selftest: %s  (gates_t4 내부 %s · 스키마 %d필드 · 코퍼스 %d · '
          'MAX_CORE_HOURS %.1f)'
          % ('PASS' if not fails else 'FAIL', 'PASS' if nf == 0 else 'FAIL',
             len(ROW_SCHEMA), len(eps), G.MAX_CORE_HOURS))
    return not fails


# ═════════════════════════════════════════════════════════════════════════════
# 11. main
# ═════════════════════════════════════════════════════════════════════════════
def cells():
    return [(a, e['episode'], s) for e in episodes() for s in G.EST_SEEDS
            for a in G.ARM_ORDER]


def shard(cs, i, n):
    return [c for k, c in enumerate(cs) if k % n == i]


def smoke_cells():
    """★스모크 셀 — 사전등록이 이름을 지정하지 않았으므로 여기서 고르고 **사유를 적는다**.

    ①**가장 비싼 팔** `SPEC25`(동결 중앙 37.8 s + 원점당 lmax×600 가지 전개) 포함
    ②**가장 비싼 층** BF4(원점 18개 · lmax 70) **와** BF5(원점 6 · lmax 90) 를 **둘 다** —
      비용 투영이 층별 단가를 요구하고, `G-16`(층 불균형 3배)이 여기서 처음 발화한다
    ③**게이트가 발화할 셀**: `G-4`(동결 앵커가 있는 시드 42) · `G-6`(양 팔) ·
      `G-7`(SPEC25) · `G-9`·`G-12`·`G-13`(양 팔, 두 층)
    ⚠`G-1`·`G-5`·`G-CMP` 는 표본 게이트라 스모크에서 **당연히 실패**한다(발화 확인용).
    """
    eps = episodes()
    pick = []
    for bf in ('BF4', 'BF5'):
        pick.append(next(e for e in eps if e['bf'] == bf)['episode'])
    return [(a, ep, G.EST_SEEDS[0]) for ep in pick for a in G.ARM_ORDER]


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
                done.add((r['arm'], r['episode'], r.get('seed')))
    return done


def execute(jobs, out_path, tag):
    eps = {e['episode']: e for e in episodes()}
    fz = G.load_frozen_b12()
    if not fz:
        raise SystemExit('⛔B-12 동결 자료가 비었다 — G-4 가 항진한다. 본런 금지.')
    done = _done(out_path)
    todo = [j for j in jobs if j not in done]
    print('T-4 %s: 작업 %d (완료 %d) -> %s' % (tag, len(todo), len(done),
                                             os.path.basename(out_path)), flush=True)
    print('  스탬프 %r' % (stamps(),), flush=True)
    print('  ★재현 통화 = 스레드 고정 · GPU 허용(M-10 AMENDMENT_1 승계)', flush=True)
    t_start = time.time()
    rows, n_err = [], 0
    with open(out_path, 'a') as fh:
        for i, (arm, ep_name, seed) in enumerate(todo, 1):
            try:
                row = run_cell(arm, eps[ep_name], seed)
            except Exception as exc:                              # noqa: BLE001
                n_err += 1
                row = dict(arm=arm, episode=ep_name, seed=seed, error=repr(exc),
                           tb=traceback.format_exc()[-700:], **stamps())
            fh.write(json.dumps(row) + '\n')
            fh.flush()
            rows.append(row)
            el = time.time() - t_start
            if 'error' in row:
                print('  ⛔%-6s %-38s s%-3s %s' % (arm, ep_name[:38], seed,
                                                  str(row['error'])[:90]), flush=True)
                continue
            f = fz.get((arm, ep_name, seed))
            rel = G._rel(float(row['avg_nRMSE']), f) if f is not None else None
            print('  %-6s %-38s s%-3s L0=%.5f L1=%.5f Z1=%.5f relΔ=%s neg=%s '
                  'pos=%s fc=%d pf=%d %.1fs · ETA %.0fm'
                  % (arm, ep_name[:38], seed,
                     row['lead_rmse'][repr(0.0)], row['lead_rmse'][repr(1.0)],
                     row['lead_zero'][repr(1.0)],
                     ('%.1e' % rel) if rel is not None else 'n/a',
                     row['neg_lead0_bitmatch'],
                     ('%.1e' % row['pos_compose_relerr'])
                     if row['pos_compose_relerr'] is not None else '-',
                     row['cap_forecast_steps'], row['cap_predict_future'],
                     row['wall_s'], el / i * (len(todo) - i) / 60.0), flush=True)
    print('  완료 %d행 (에러 %d) %.1f분' % (len(rows), n_err, (time.time() - t_start) / 60.0),
          flush=True)
    return rows


def main():
    argv = sys.argv[1:]
    mode = (argv or ['run'])[0]
    if mode == 'selftest':
        sys.exit(0 if selftest() else 1)

    if mode == 'smoke':
        rows = execute(smoke_cells(), SMOKE_OUT, 'smoke')
        allrows = []
        for line in open(SMOKE_OUT):
            line = line.strip()
            if line:
                try:
                    allrows.append(json.loads(line))
                except ValueError:
                    pass
        cost = project_cost(allrows)
        print('\n=== ★비용 게이트 (PREREG §8·§9 — 단가 실측 → 투영) ===')
        for k, v in sorted(cost.get('detail', {}).items()):
            print('  %-14s 단가 %7.1f s × %3d 셀 = %5.2f 코어·h'
                  % (k, v['unit_s'], v['cells'], v['hours']))
        if cost.get('fc_unit_us'):
            print('  `_forecast_member` 단가 실측 %.1f µs (%d 가지)'
                  % (cost['fc_unit_us'], cost['fc_calls_measured']))
        if cost.get('substituted'):
            print('  ⚠단가 대체(보수적): %s' % cost['substituted'])
        print('  ⟹ 총 %.2f 코어·h (문턱 %.2f) : %s'
              % (cost.get('core_hours', float('nan')), cost['max_core_hours'],
                 'GO' if cost['ok'] else '⛔ABORT'))
        if not cost['ok']:
            print('  %s' % cost['reason'])
            sys.exit(3)
        return

    cs = cells()
    if len(argv) > 2 and argv[1].isdigit() and argv[2].isdigit():
        i, n = int(argv[1]), int(argv[2])
        if n > 4:
            raise SystemExit('⛔샤드는 4 이하다(CPU 공유 — PREREG §6-9). 요청 %d' % n)
        jobs = shard(cs, i, n)
        out = os.path.join(HERE, 'results_t4_%d.jsonl' % i)
    else:
        jobs, out = cs, os.path.join(HERE, 'results_t4_0.jsonl')
    execute(jobs, out, 'run')


if __name__ == '__main__':
    main()
