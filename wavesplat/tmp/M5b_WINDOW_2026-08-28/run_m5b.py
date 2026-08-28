"""M-5b 러너 — 규칙은 `gates_m5b.py` **단일출처**. ⛔이 파일에 문턱·판정이 없다.

    python3 tmp/M5b_WINDOW_2026-08-28/run_m5b.py smoke      # 1 (record,win) · 결과 지운다
    python3 tmp/M5b_WINDOW_2026-08-28/run_m5b.py run        # 456 런
    python3 tmp/M5b_WINDOW_2026-08-28/run_m5b.py run 0 6    # 샤드 0/6  (CPU 전용)

⛔**GPU 금지**: `device='cpu'` 를 config 에 명시하고, `CUDA_VISIBLE_DEVICES=''` 를 이
파일이 **직접** 세운다(torch 임포트 전).

★**`M5` 러너와의 차이는 넷뿐이다**(전부 `AMENDMENT_1.md` 의 처방):
  1. 장면을 `cap_mode=G.CAP_MODE`(`'end_anchored'`) 로 **명시** 호출한다.
  2. 행에 `t0_abs` 를 싣는다 — `G-1b` 의 **행 내부 절대 앵커**.
  3. `W >= W_ANCHOR` 인 수준에서 **cap 이전 pool** 을 한 번 더 만들어 `pool_hash`/`pool_n`
     을 싣는다 — `G-5b2`(바뀐 것이 cap 뿐임을 판정 자료 위에서 증명).
  4. ★**밀도 공변**을 싣는다(`dens_per_s`·`dens_last_1s`·`dens_last_02Tp`·`n_last_scan`)
     — `n_obs` 고정의 필연적 부수효과이고 **결과 전 공시** 대상이다(`G-10`).

★**계측은 `effective_config` 를 입력 dict 복사로 만들지 않는다**(CLAUDE.md 금지 항목):
`SequentialUpdater.__init__` 에 투명 shim 을 걸어 **객체 속성**에서 되읽고,
`learned_init.propose_atoms` 에 shim 을 걸어 **조건자에 실제로 들어간 관측 수**를 잡는다.
"""
import json
import os
import sys
import time
import warnings

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')      # ⛔GPU 금지 (torch 임포트 전)
warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for _p in (HERE, ROOT, os.path.join(ROOT, 'scripts'),
           os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23'),
           os.path.join(ROOT, 'tmp', 'B3_REFIT_2026-08-23'),
           os.path.join(ROOT, 'tmp', 'bankmmse_port_2026-07-29')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(ROOT)

import numpy as np                                     # noqa: E402
import torch                                           # noqa: E402
import gates_m5b as G                                  # noqa: E402
from scene_station import scene_station                # noqa: E402
import make_gif_bank_deployed as MK                    # noqa: E402
from premise_m5b import level_seconds                  # noqa: E402

CAP = {}
_SCENES = {}


def assert_shipping_defaults():
    """★런 0회에서 죽는다 — 팔이 **선언한** 배포 3키가 배포 단일출처와 다르면 예외.

    ⛔조용한 흡수 금지(`gates_b7.py:46` 의 `{}` BLOCKER, 2026-08-23)."""
    from wavesplat.run import DEFAULT_CONFIG
    diff = {k: (DEFAULT_CONFIG.get(k), v) for k, v in G.SHIPPED_2026_08_22.items()
            if DEFAULT_CONFIG.get(k) != v}
    if diff:
        raise SystemExit(
            '⛔배포 기본값이 사전등록 선언과 다르다 (shipped, declared) = '
            f'{diff}\n   ⟹ 이 배치를 돌리기 전에 사전등록을 개정하라(AMENDMENT).')
    return True


def assert_scene_contract():
    """★장면 규약이 살아 있는지 **런 0회**에 확인한다.

    ⛔다른 세션이 `scene_station.py` 를 되돌리거나 기본값을 바꾸면 여기서 죽는다."""
    import hashlib
    import inspect
    p = inspect.signature(scene_station).parameters
    if 'cap_mode' not in p:
        raise SystemExit('⛔scene_station 에 cap_mode 가 없다 — 파일이 되돌아갔다.')
    if p['cap_mode'].default != 'uniform':
        raise SystemExit(f'⛔cap_mode 기본값이 {p["cap_mode"].default!r} — uniform 이어야 한다.')
    md5 = hashlib.md5(open(os.path.join(
        ROOT, 'tmp', 'B2_REAL_2026-08-23', 'scene_station.py'), 'rb').read()).hexdigest()
    print(f'  scene_station md5={md5}  cap_mode 기본값={p["cap_mode"].default!r} '
          f'· 이 배치={G.CAP_MODE!r}', flush=True)
    return md5


# ── 투명 계측 shim (값을 바꾸지 않는다) ─────────────────────────────────────
def install_shims():
    from wavesplat.optimizer import sequential_update as SU
    from wavesplat.optimizer import learned_init as LI

    if not getattr(SU.SequentialUpdater, '_m5b_shim', False):
        _orig = SU.SequentialUpdater.__init__

        def _init(self, config, *a, **kw):
            _orig(self, config, *a, **kw)
            CAP['obj'] = dict(
                cap_init_method=getattr(self, 'init_method', None),
                cap_lambda_l1=getattr(self, 'lambda_l1', None),
                cap_tracking_mode=getattr(self, 'tracking_mode', None),
                cap_real_estimator=getattr(self, 'real_estimator', None),
                cap_bank_max_obs=self.config.get('bank_max_obs'),
                cap_varpro_steps=self.config.get('varpro_steps'),
                cap_learned_init_iters=self.config.get('learned_init_iters'),
                cap_auto_switch_thresh=self.config.get('auto_switch_thresh'))
        SU.SequentialUpdater.__init__ = _init
        SU.SequentialUpdater._m5b_shim = True

    if not getattr(LI, '_m5b_shim', False):
        _orig_prop = LI.propose_atoms

        def _prop(*a, **kw):
            frames = kw['frames'] if 'frames' in kw else a[0]
            obs = LI._gather_init_obs(frames)
            CAP['n_obs_learned'] = (min(len(obs[2]), LI.M_OBS_CAP)
                                    if obs is not None else 0)
            CAP['n_obs_learned_pre'] = len(obs[2]) if obs is not None else 0
            return _orig_prop(*a, **kw)
        LI.propose_atoms = _prop
        LI._m5b_shim = True


def _bits(x):
    return np.float64(x).tobytes().hex()


def scene_for(record, win, W, m_cap=None):
    """`m_cap=None` = 판정 장면(`cap_mode` 적용). `m_cap=G.POOL_NO_CAP` = cap 이전 pool.

    ⚠pool 호출은 cap 분기에 **도달하지 않으므로** `cap_mode` 와 무관하다."""
    mc = G.M_CAP if m_cap is None else int(m_cap)
    k = (record, win, round(float(W), 9), mc)
    if _SCENES and next(iter(_SCENES))[:2] != (record, win):
        _SCENES.clear()
    if k not in _SCENES:
        _SCENES[k] = scene_station(record, win=win, W=float(W), noise=G.NOISE,
                                   m_cap=mc, cap_mode=G.CAP_MODE)
    return _SCENES[k]


def tp_of(record, win):
    """이 (record,win) 의 Tp — **장면이 낸 `kp`** 와 분산관계로만 만든다(손복사 0).

    ⚠`kp` 는 조밀한 참 표면에서 온다(GT 유래). 용량 정의에만 쓰고 배포 권고로 올리지 않는다."""
    sc = scene_for(record, win, G.W_ANCHOR)
    kp, d = float(sc['kp']), float(sc['depth'])
    w = np.sqrt(G.G_ACC * kp * np.tanh(min(kp * d, 20.0)))
    return float(2 * np.pi / w)


def run_one(level, arm, record, win):
    from wavesplat.optimizer.sequential_update import make_updater
    from wavesplat.run import DEFAULT_CONFIG, set_seed
    import inspect
    t0w = time.time()
    spec = G.NOOP_SPEC if level == G.NOOP_LEVEL else G.LEVELS[level]
    tp = tp_of(record, win)
    W = float(spec['value']) if spec['kind'] in ('sec', 'noop') \
        else level_seconds(level, tp)
    sc = scene_for(record, win, W)

    P = np.asarray(sc['P'], float)
    t_end_abs = float(P[:, 2].max())
    if spec['origin'] == 'pred_minus':
        # 예측 시각(창의 끝)에서 `ORIG2_TP_FACTOR·Tp` 앞에 원점을 둔다. 관측 집합 불변.
        t_shift = float(sc['Pq'][0, 2]) - G.ORIG2_TP_FACTOR * tp
    else:
        t_shift = float(P[:, 2].min())
    t0 = float(sc['Pq'][0, 2]) - t_shift

    # ── ★cap 이전 pool (W >= W_ANCHOR 수준만) — `G-5b2` 의 되읽기 ──────────
    pool_hash = pool_n = None
    if float(W) >= G.W_ANCHOR:
        pl = scene_for(record, win, W, m_cap=G.POOL_NO_CAP)
        pool_hash = G.hash_obs(pl['P'], pl['z'])
        pool_n = int(len(pl['z']))

    # ── ★밀도 공변 (결과 전 공시. ⛔판정에 쓰지 않는다) ─────────────────────
    t_span = float(P[:, 2].max() - P[:, 2].min())
    dens_per_s = float(len(P) / t_span) if t_span > 0 else float('nan')
    dens_last_1s = int((P[:, 2] > t_end_abs - 1.0).sum())
    dens_last_02Tp = int((P[:, 2] > t_end_abs - 0.2 * tp).sum())
    n_last_scan = int((P[:, 2] >= t_end_abs - G.T_EQ_TOL).sum())

    frames = MK.frames_from_scene(dict(sc, P=P), t_shift, dtype=torch.float32)
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(depth=float(sc['depth']), seed=G.SEED, device='cpu')
    cfg.update(G.ARMS[arm])
    CAP.clear()
    set_seed(G.SEED)
    upd = make_updater(cfg)
    res = upd.initialize(frames,
                         x_range=(float(sc['gxq'].min()), float(sc['gxq'].max())),
                         y_range=(float(sc['gyq'].min()), float(sc['gyq'].max())),
                         verbose=False)
    gxq, gyq = np.asarray(sc['gxq']), np.asarray(sc['gyq'])
    curve = []
    for li, l_ in enumerate(sc['leads']):
        g = np.asarray(sc['maps_gt'][li], float)
        mm = np.isfinite(g)
        if not mm.any():
            curve.append(float('nan')); continue
        Z = upd.predict_field(torch.tensor(gxq, dtype=torch.float32),
                              torch.tensor(gyq, dtype=torch.float32),
                              t0 + float(l_)).cpu().numpy().reshape(len(gyq), len(gxq))
        curve.append(float(np.sqrt(np.mean((Z[mm] - g[mm]) ** 2)) / sc['sg']))
    nowcast, leadmean = curve[0], float(np.nanmean(curve))
    row = dict(level=level, cell=arm, record=record, win=win,
               # ── 개입 되읽기 (⛔항등식 아님: 관측 배열에서 다시 잰다) ──
               W_used=float(W), W_over_Tp=float(W / tp), Tp_used=float(tp),
               t_span=t_span, t_end_abs=t_end_abs, t0_abs=float(sc['t0']),
               t_shift=float(t_shift), t0_pred=float(t0),
               origin=spec['origin'], dt=float(sc['dt']),
               n_obs=int(len(sc['z'])), n_frames_obs=int(len(np.unique(P[:, 2]))),
               n_obs_learned=CAP.get('n_obs_learned'),
               n_obs_learned_pre=CAP.get('n_obs_learned_pre'),
               # ── ★장면 규약 (⚠이것만으로는 항등식 — 관측 가능한 귀결은 G-1b/G-1c) ──
               cap_mode_used=G.CAP_MODE,
               cap_mode_sig=inspect.signature(scene_station).parameters['cap_mode'].default,
               pool_hash=pool_hash, pool_n=pool_n,
               # ── ★밀도 공변 (공시 전용) ──
               dens_per_s=dens_per_s, dens_last_1s=dens_last_1s,
               dens_last_02Tp=dens_last_02Tp, n_last_scan=n_last_scan,
               # ── 불변량 (수준에 걸쳐 비트동일이어야 하는 것들) ──
               obs_hash=G.hash_obs(P, np.asarray(sc['z'], float)),
               q_hash=G.hash_obs(sc['Pq']),
               gt_hash=G.hash_obs(np.nan_to_num(sc['maps_gt'])),
               sigma_bits=_bits(sc['sg']), kp_bits=_bits(sc['kp']),
               # ── 결과 ──
               nowcast=nowcast, leadmean=leadmean, curve=curve,
               nowcast_bits=_bits(nowcast), leadmean_bits=_bits(leadmean),
               n_atoms=(getattr(getattr(upd, 'model', None), 'n_gaussians', 0) or 0),
               varpro_fired=bool((res or {}).get('varpro')),
               real_gate=getattr(upd, '_auto_real_gate', None),
               Hs=float(sc['Hs']), sigma=float(sc['sg']),
               wall_s=time.time() - t0w)
    row.update(CAP.get('obj') or {})
    return row


def cells():
    out = []
    for win in G.WINDOWS:
        for record in G.RECORDS:
            for level in G.LEVEL_ORDER:
                for arm in G.ARM_ORDER:
                    out.append((level, arm, record, win))
    for record in G.RECORDS:              # NOOP 은 win=3 에서만
        for arm in G.ARM_ORDER:
            out.append((G.NOOP_LEVEL, arm, record, G.NOOP_WIN))
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'smoke'
    shard, nshard = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 1)
    assert_shipping_defaults()
    assert_scene_contract()
    install_shims()
    todo = cells()
    assert len(todo) == G.N_RUNS, f'{len(todo)} != {G.N_RUNS}'
    if mode == 'smoke':
        todo = [c for c in todo if c[2] == G.RECORDS[0] and c[3] == G.PRIMARY_WIN]
        out = os.path.join(HERE, 'smoke_m5b.jsonl')
    else:
        todo = [c for i, c in enumerate(todo) if i % nshard == shard]
        out = os.path.join(HERE, f'results_m5b_{shard}.jsonl')
    done = {G.key(r) for r in G._load_jsonl(out) if 'error' not in r}
    with open(out, 'a') as fh:
        for (level, arm, record, win) in todo:
            if (level, arm, record, win) in done:
                continue
            try:
                r = run_one(level, arm, record, win)
            except Exception as e:                      # noqa: BLE001
                import traceback
                r = dict(level=level, cell=arm, record=record, win=win,
                         error=repr(e), tb=traceback.format_exc()[-900:])
            fh.write(json.dumps(r) + '\n'); fh.flush()
            tag = (f"now={r['nowcast']:.4f} lead={r['leadmean']:.4f} N={r['n_atoms']} "
                   f"W={r['W_used']:.2f}s({r['W_over_Tp']:.2f}Tp) n={r['n_obs']} "
                   f"dens={r['dens_per_s']:.1f}/s"
                   if 'error' not in r else str(r['error'])[:70])
            print(f'{level:7s} {arm:5s} {record:7s} w{win} {tag}', flush=True)
    if mode == 'smoke':
        os.remove(out)
        print('smoke 결과 삭제 (판정 자료 아님)')


if __name__ == '__main__':
    main()
