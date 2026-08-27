r"""B-25 러너 — **관측 신뢰가중의 절제** (3팔 × 48 에피소드 × 3 시드 = 432런). 재개·샤딩 가능.

사용:
    python3 tmp/B25_CONFWEIGHT_2026-08-27/run_b25.py --selftest    # 게이트 자가진단만
    python3 tmp/B25_CONFWEIGHT_2026-08-27/run_b25.py <shard> <nshards>
    SMOKE=1 python3 ... run_b25.py                                  # 1셀 (결과 파일은 지울 것)

★**런 전 게이트 자가진단이 자동으로 먼저 돈다.** 실패하면 한 런도 돌리지 않는다.

## 개입은 **둘**이고 둘 다 명시적이다
1. **패키지 노브** `use_confidence` (`run.py:106` · `sequential_update.py:435`) — `NOCONF` 팔.
2. **자료 훅** `EpisodeDataLoader.__getitem__` — `FLATCONF` 팔에서 프레임의 `confidence` 를
   **그 프레임 자신의 평균**으로 치환한다(질량 보존 · 정보 제거).
   ⚠**이것은 관측이 아니라 개입이다** — 그래서 계수기 넷이 행에 실리고 `G-4b`·`G-4c` 가 검사한다.
   ★`__getitem__` 이 **유일한 프레임 생성 지점**이다: `get_init_window(:106-121)` ·
   `get_tracking_frames(:124-131)` · `__iter__(:102-104)` 가 전부 `self[i]` 를 지난다.
   ⛔채점은 오염되지 않는다 — `zero_nrmse` 는 `load_episode` 를 직접 읽고 로더를 안 쓴다(`G-9`).

## 되읽기 — ⛔`cfg[k] == cfg[k]` 항등식 금지
- `make_updater` 를 **관측 훅**으로 감싸 실제 추정기 객체를 붙잡고 `gates_b25.effective_b25()`
  로 **객체 속성** `use_confidence` 를 읽는다(`__init__:435` 가 만든 값).
- ★**소비처가 스스로 증언한다**: `apply_varpro` 의 반환 `rec` 에서 `vp_weighting` 과
  `vp_weighting_fallback` 을 붙잡는다. 후자는 `varpro.py:218-220` 이 `use_confidence=False`
  **를 보고** 남기는 값이다.
  ⚠★**`run.py:530-535` 가 이 두 키를 버린다** — 계수기는 계산되는데 저장되지 않는다
    (저장소의 반복 결함 부류: `RMSE_near`·셸 RMSE·채점 창에 이어 **다섯 번째**).
    ⟹ 패키지를 고치지 않고 훅으로 붙잡는다. **⛔패키지 수정 0.**
- 훅이 거동을 바꾸지 않았다는 증명은 추론이 아니라 **G-5**(`SHIP` 이 `B-22` 동결 행과 비트 동일)다.
"""
import hashlib
import json
import os
import subprocess
import sys
import time
import warnings

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for _p in (HERE, ROOT):
    sys.path.insert(0, _p)
os.chdir(ROOT)
import gates_b25 as G                                        # noqa: E402

IMPORT_PATH = ['wavesplat/run.py',
               'wavesplat/data_loader.py',
               'wavesplat/optimizer/sequential_update.py',
               'wavesplat/optimizer/learned_init.py',
               'wavesplat/optimizer/initializer.py',
               'wavesplat/optimizer/varpro.py',
               'wavesplat/model/confidence.py',
               'wavesplat/evaluation/metrics.py']


def _env():
    try:
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                       cwd=ROOT, text=True).strip()
    except Exception:                                        # noqa: BLE001
        head = None
    md5 = {p: hashlib.md5(open(os.path.join(ROOT, p), 'rb').read()).hexdigest()
           for p in IMPORT_PATH if os.path.exists(os.path.join(ROOT, p))}
    return head, md5


GIT_HEAD, IMPORT_MD5 = _env()

# ── ★관측 훅 ①: `run_estimation` 이 만든 **추정기 객체** ──────────────────────
_CAPTURED = []


def _install_capture():
    import wavesplat.run as WR
    if getattr(WR.make_updater, '_b25_capture', False):
        return
    real = WR.make_updater

    def capturing(config):
        u = real(config)
        _CAPTURED.append(u)
        return u

    capturing._b25_capture = True
    capturing._b25_real = real
    WR.make_updater = capturing


# ── ★관측 훅 ②: `apply_varpro` 의 반환 `rec` (소비처의 증언) ──────────────────
#    ⛔거동을 바꾸지 않는다 — 진짜 함수의 반환값을 **그대로** 돌려준다.
_VPRECS = []


def _install_varpro_capture():
    import wavesplat.optimizer.varpro as VP
    if getattr(VP.apply_varpro, '_b25_capture', False):
        return
    real = VP.apply_varpro

    def capturing(*a, **kw):
        rec = real(*a, **kw)
        _VPRECS.append(rec)
        return rec

    capturing._b25_capture = True
    capturing._b25_real = real
    VP.apply_varpro = capturing


# ── ★개입 훅 ③: 프레임의 `confidence` (FLATCONF 팔) ──────────────────────────
#    ⚠**이것은 개입이다.** 계수기가 행에 실리고 `G-4b`·`G-4c` 가 검사한다.
_CONF = {'mode': G.CONF_RAW, 'seen': 0, 'changed': 0, 'std_max': None,
         'mass_err_max': 0.0, 'sum_c': 0.0, 'sum_c2': 0.0, 'n_pts': 0}


def _conf_reset(mode):
    _CONF.update(mode=mode, seen=0, changed=0, std_max=None, mass_err_max=0.0,
                 sum_c=0.0, sum_c2=0.0, n_pts=0)


def _install_conf_hook():
    import wavesplat.data_loader as DL
    if getattr(DL.EpisodeDataLoader.__getitem__, '_b25_conf', False):
        return
    real = DL.EpisodeDataLoader.__getitem__

    def hooked(self, idx):
        import torch
        f = real(self, idx)
        c = f.confidence
        _CONF['seen'] += 1
        n = int(c.numel())
        if n:
            cd = c.detach().to(torch.float64)
            # ★RAW 통계는 **모드와 무관하게** 원본에서 누적한다 (G-13 이 세 팔 전부에서 검사)
            _CONF['sum_c'] += float(cd.sum())
            _CONF['sum_c2'] += float((cd * cd).sum())
            _CONF['n_pts'] += n
            if _CONF['mode'] == G.CONF_FLAT:
                raw_sum = float(cd.sum())
                flat = c.mean().expand_as(c).clone()      # 프레임 자신의 평균
                f.confidence = flat
                _CONF['changed'] += 1
                out = flat.detach().to(torch.float64)
                if raw_sum > 0:
                    _CONF['mass_err_max'] = max(
                        _CONF['mass_err_max'],
                        abs(float(out.sum()) - raw_sum) / raw_sum)
                s = float(out.std(unbiased=False)) if n > 1 else 0.0
            else:
                s = float(cd.std(unbiased=False)) if n > 1 else 0.0
            _CONF['std_max'] = s if _CONF['std_max'] is None \
                else max(_CONF['std_max'], s)
        return f

    hooked._b25_conf = True
    hooked._b25_real = real
    DL.EpisodeDataLoader.__getitem__ = hooked


def _conf_stats():
    n = _CONF['n_pts']
    if not n:
        return dict(conf_mean_raw=None, conf_std_raw=None)
    m = _CONF['sum_c'] / n
    var = max(_CONF['sum_c2'] / n - m * m, 0.0)
    return dict(conf_mean_raw=m, conf_std_raw=var ** 0.5)


# ─────────────────────────────────────────────────────────────────────────────
def run_one(ep, arm, est_seed):
    from wavesplat.run import run_estimation
    _install_capture()
    _install_varpro_capture()
    _install_conf_hook()
    cfg = G.arm_config(arm, est_seed)          # ★팔 정의는 gates_b25 단일출처
    _CAPTURED.clear()
    _VPRECS.clear()
    _conf_reset(G.conf_mode(arm))
    t0 = time.time()
    r = run_estimation(ep['path'], config=cfg, verbose=False)
    wall = time.time() - t0
    _CONF['mode'] = G.CONF_RAW                 # ⚠런 밖에서는 훅을 중립으로 되돌린다
    eff = G.effective_b25(_CAPTURED[-1] if _CAPTURED else None,
                          n_captured=len(_CAPTURED))
    vp = _VPRECS[-1] if _VPRECS else {}
    z = G.zero_nrmse(ep['path'])               # ★매 행 새로 계산 (로더를 쓰지 않는다 ⟹ G-9)
    return dict(
        arm=arm, est_seed=est_seed, episode=ep['episode'], bf=ep['bf'],
        ncomp=ep['ncomp'], gen_seed=ep['gen_seed'],
        avg_nRMSE=float(r['avg_nRMSE']),
        # ── ★소비처의 증언 (run.py 가 버리는 두 키를 훅이 붙잡았다) ──────────
        vp_weighting=vp.get('vp_weighting'),
        vp_weighting_fallback=vp.get('vp_weighting_fallback'),
        vp_calls=len(_VPRECS), vp_n_obs=vp.get('vp_n_obs'),
        vp_obj_before=vp.get('vp_obj_before'), vp_obj_after=vp.get('vp_obj_after'),
        vp_k_shift=vp.get('vp_k_shift'),
        # ── ★개입 훅 계수기 ─────────────────────────────────────────────────
        conf_frames_seen=_CONF['seen'], conf_frames_changed=_CONF['changed'],
        conf_std_max=_CONF['std_max'],
        conf_mass_rel_err_max=_CONF['mass_err_max'],
        **_conf_stats(),
        # ── 객체 파생 속성 (G-3·G-5c 가 검사하는 되읽기 증거) ────────────────
        effective=eff,
        # ── 배포 계수기 ─────────────────────────────────────────────────────
        varpro_steps=r.get('varpro_steps'), varpro_fired=bool(r.get('varpro_fired')),
        init_iters=r.get('init_iters'), init_loss_n=r.get('init_loss_n'),
        init_time_s=r.get('init_time_s'),
        auto_selected=r.get('auto_selected'), auto_n_selected=r.get('auto_n_selected'),
        auto_real_gate=r.get('auto_real_gate'), ensemble_k=r.get('ensemble_k'),
        ensemble_member_n=r.get('ensemble_member_n'), n_gaussians=r.get('n_gaussians'),
        # ── 채점 창 (B-13 부류 재발 방지) ───────────────────────────────────
        tracking_frames=r.get('tracking_frames'),
        scored_t_first=r.get('scored_t_first'), scored_t_last=r.get('scored_t_last'),
        episode_total_frames=r.get('episode_total_frames'),
        # ── ZERO (참 제로필드; ⛔상수 1.0 아님) ─────────────────────────────
        **z,
        # ── ⚠**의도표** — 러너가 *넣으려 한* 값. ⛔게이트가 증거로 쓰지 않는다.
        cfg_intent={k: cfg.get(k) for k in
                    ('use_confidence', 'lambda_l1', 'init_method',
                     'learned_init_iters', 'auto_switch_thresh', 'varpro_steps',
                     'analytic_ensemble_k')},
        conf_mode_intent=G.conf_mode(arm),
        git_head=GIT_HEAD, import_md5=IMPORT_MD5, wall_s=wall)


def main():
    if '--selftest' in sys.argv:
        sys.exit(0 if G.selftest()[0] else 1)
    print('★런 전 게이트 자가진단 (가짜 행으로 게이트가 발화하는지) ─────────', flush=True)
    if not G.selftest()[0]:
        print('⛔게이트 자가진단 실패 ⟹ 배치를 시작하지 않는다.', flush=True)
        sys.exit(1)
    print('─────────────────────────────────────────────────────────────\n', flush=True)

    shard = int(sys.argv[1]) if len(sys.argv) > 2 else 0
    nsh = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    smoke = os.environ.get('SMOKE') == '1'
    out = os.path.join(HERE, 'results_b25.jsonl' if nsh == 1
                       else f'results_b25_{shard}.jsonl')
    if smoke:
        out = os.path.join(HERE, 'SMOKE_results_b25.jsonl')
    done = set()
    if os.path.exists(out):
        for line in open(out):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'error' not in r:
                done.add((r['arm'], r['episode'], r['est_seed']))
    eps = [e for i, e in enumerate(G.episodes()) if i % nsh == shard]
    seeds = list(G.EST_SEEDS)
    if smoke:
        eps, seeds = eps[:1], seeds[:1]
    hi_n, hi_c, lo_n, lo_c = G.thresholds(G.N_EP)
    print(f'B-25 shard {shard}/{nsh}: 혼잡 에피소드 {len(eps)}개 · 팔 {G.ARM_ORDER} · '
          f'시드 {seeds} ⟹ {len(eps) * len(G.ARM_ORDER) * len(seeds)} 런', flush=True)
    print(f'  문턱(n={G.N_EP}, 양측): NOMINAL >={hi_n} 또는 <={lo_n} · '
          f'CONFIRM >={hi_c} 또는 <={lo_c}', flush=True)
    print(f'  배포 상수 {G.deploy_constants()} · git {GIT_HEAD}', flush=True)
    with open(out, 'a') as fh:
        for ep in eps:
            for s in seeds:
                for arm in G.ARM_ORDER:
                    if (arm, ep['episode'], s) in done:
                        continue
                    try:
                        row = run_one(ep, arm, s)
                    except Exception as e:                   # noqa: BLE001
                        import traceback
                        row = dict(arm=arm, episode=ep['episode'], est_seed=s,
                                   error=repr(e), tb=traceback.format_exc()[-600:])
                    fh.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')
                    fh.flush()
                    print(f"  {ep['episode'][:34]:34s} s{s} {arm:9s} "
                          f"nRMSE {row.get('avg_nRMSE', float('nan')):.5f} "
                          f"| vpw {str(row.get('vp_weighting')):10s} "
                          f"| conf {row.get('conf_frames_changed')}/"
                          f"{row.get('conf_frames_seen')} "
                          f"std_max {row.get('conf_std_max')} "
                          f"| {row.get('wall_s', 0):.1f}s", flush=True)
    print(f'\n완료 → {out}', flush=True)


if __name__ == '__main__':
    main()
