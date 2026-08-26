"""B-19 러너 — 표현 절제(파수 격자 스냅)를 **희소 층**에서. 재개 가능·샤딩 가능.

사용:
    python3 tmp/B19_KGRID_SPARSE_2026-08-25/run_b19.py            # 단일 프로세스 (권장)
    python3 tmp/B19_KGRID_SPARSE_2026-08-25/run_b19.py <shard> <nshards>
    python3 tmp/B19_KGRID_SPARSE_2026-08-25/run_b19.py --selftest # ★게이트 자가진단만
    SMOKE=1 python3 ... run_b19.py                                # 1셀만 (결과 파일은 지울 것)

★**런 전 게이트 자가진단이 자동으로 먼저 돈다** (2026-08-26 신설). 실패하면 배치를 시작하지
않는다. 초판은 게이트가 전부 **분석기 안**에 있어서 이것이 원리적으로 불가능했다.

⚠**샤딩 주의**: `nshards==1` 이면 `results_b19.jsonl` **한 파일**을 쓴다.
`tmp/_audit/check_batch_artifacts.py` 의 H 검사(결과 파일마다 짝 verdict)는 결과가 여럿이면
**각 접미사마다** verdict 을 요구한다 ⟹ `B-16`·`B-15` 가 샤딩 파일명 때문에 그 열에서
`--` 로 떨어져 있다(2026-08-25 확인). 이 배치는 144런 ≈ 20분이므로 **샤딩하지 않는다.**

★배포 진입점 `wavesplat.run.run_estimation` 을 부르고 `k_snap_dk` **하나만** 바꾼다.
배포 상수 다섯은 **패키지 단일출처에서 읽어 명시**한다(스냅샷 아님).

## ★되읽기 — 2026-08-26 BLOCKER 수리 (읽기 전에 이 문단을 읽어라)
⛔**초판은 `lambda_l1=cfg.get('lambda_l1')` 처럼 자기 설정 dict 을 다시 읽어 행에 실었다.**
`SequentialUpdater.__init__` 이 `self.config = config`(`sequential_update.py:399`) 로 입력
dict 을 **복사 없이** 붙들기 때문에, 그 되읽기와 게이트의 대조는
`cfg[k] == cfg[k]` 라는 **항등식**이고 **실패할 수 없다.**
그런데도 게이트 이름이 *"배포 상수 되읽기 일치 PASS"* 로 떠서 **방벽을 붙였다고 믿게 만든다**
(W8 `G-4` 공허한 통과의 재현이고, W7 의 *"설정을 넣었다 ≠ 설정이 먹었다"* 를 인용하고
있었으므로 더 나쁘다).

✅**수리**: `make_updater` 를 **관측 훅**으로 감싸 `run_estimation` 이 실제로 만든 추정기
객체를 붙잡고, `gates_b19.effective_from_updater()` 가 **객체의 파생 속성**을 읽어 행의
`effective` 에 싣는다(`lambda_l1`·`tracking_mode`·`init_method`(경로)·`seed` …).
설정 dict 값은 `cfg_intent` 라는 **다른 이름**으로만 남기고, 게이트는 그것을 **증거로
쓰지 않는다**(S13b 의 *"effective_config 를 입력 dict 복사로 만들지 마라 = 의도표"*).

⚠**훅이 거동을 바꾸지 않는다는 증명은 추론이 아니라 게이트다**: 훅은 `make_updater(config)`
의 반환값을 **그대로** 돌려주고(래핑·상속 없음), `G-11` 이 `SHIP-NOW` 48행을 훅 없이 돌았던
`B-11b` 동결 행과 **비트 동일**로 대조한다. 어긋나면 배치가 그 자리에서 멈춘다.
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
import gates_b19 as G                                        # noqa: E402

# ⚠`git_head` 만으로는 **더러운 작업트리를 못 잡는다**(S13·S13b 가 그것으로 물렸다).
#   ⟹ 임포트 경로 파일의 md5 를 동반 기록한다.
IMPORT_PATH = ['wavesplat/run.py',
               'wavesplat/optimizer/sequential_update.py',
               'wavesplat/optimizer/learned_init.py',
               'wavesplat/optimizer/varpro.py',
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

# ── ★관측 훅: `run_estimation` 이 만든 **추정기 객체**를 붙잡는다 ──────────────
#    ⛔거동을 바꾸지 않는다: 진짜 `make_updater(config)` 의 반환값을 **그대로** 돌려준다
#    (하위클래스·프록시·인자 변경 없음). 증명은 G-11(비트 동일 양성대조)이 한다.
_CAPTURED = []


def _install_capture():
    import wavesplat.run as WR
    if getattr(WR.make_updater, '_b19_capture', False):
        return
    real = WR.make_updater

    def capturing(config):
        u = real(config)
        _CAPTURED.append(u)
        return u

    capturing._b19_capture = True
    capturing._b19_real = real
    WR.make_updater = capturing


def run_one(ep, arm, est_seed):
    from wavesplat.run import run_estimation, DEFAULT_CONFIG
    _install_capture()
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(G.deploy_constants())      # ★명시 — 런타임 스냅샷이 아니라 단일출처 되읽기
    cfg.update(G.ARMS[arm])               # ★개입은 이 한 줄뿐이다
    cfg['seed'] = est_seed
    _CAPTURED.clear()
    t0 = time.time()
    r = run_estimation(ep['path'], config=cfg, verbose=False)
    wall = time.time() - t0
    # ★되읽기는 **객체**에서 한다 (⛔`cfg.get(...)` 항등식 아님 — 모듈 독스트링 참조)
    eff = G.effective_from_updater(_CAPTURED[-1] if _CAPTURED else None,
                                   n_captured=len(_CAPTURED))
    # ★ZERO 는 매 행 새로 계산한다(캐시 없음) ⟹ 에피소드 내 불변성이 진짜 검사가 된다(G-9).
    z = G.zero_nrmse(ep['path'])
    return dict(
        arm=arm, est_seed=est_seed, episode=ep['episode'], bf=ep['bf'],
        ncomp=ep['ncomp'], gen_seed=ep['gen_seed'],
        avg_nRMSE=float(r['avg_nRMSE']),
        # ── 개입 계수기 (되읽기: `_snap_k_to_grid` 가 실제로 쓴 값) ─────────────
        k_snap_dk=(r.get('k_snap_dk')), k_snap_fired=bool(r.get('k_snap_fired')),
        k_snap_cells=r.get('k_snap_cells'),
        k_snap_shift_median=r.get('k_snap_shift_median'),
        # ── ★객체 파생 속성 (G-5 가 검사하는 **유일한** 되읽기 증거) ───────────
        effective=eff,
        # ── 계수기 (init_result 유래, G-5c) ───────────────────────────────────
        # ⚠**이 둘만으로는 부족하다**(2026-08-26 V4 반증): `varpro_steps` 는 `:913` 이
        #   config 에서 읽은 값의 **echo** 이고, `varpro_fired` 는 `bool(init_result['varpro'])`
        #   인데 `varpro.py:201` 이 **건너뛴 경우에도 truthy dict** 을 만든다 ⟹ 둘 다
        #   *"건너뛰었다"* 를 구별하지 못한다.
        varpro_steps=r.get('varpro_steps'), varpro_fired=bool(r.get('varpro_fired')),
        # ★그래서 **실행됐을 때만** 채워지는 값을 함께 싣는다 — `varpro.py:248-258` 의
        #   `rec.update` 는 세 건너뜀 반환(`:205/:209/:213`) **뒤에** 있으므로, 이 둘이
        #   None 이 아니라는 것이 VARPRO 가 실제로 돌았다는 직접 증거다(G-5c).
        varpro_obj_before=r.get('varpro_obj_before'),
        varpro_obj_after=r.get('varpro_obj_after'),
        # ★행동 지문: learned_init_iters=0 이면 결합 경사하강이 한 스텝도 안 돈다
        #   ⟹ loss_history 가 비어 `init_loss_n == 0` (run.py:554 주석). G-5b.
        init_loss_n=r.get('init_loss_n'),
        # ── 라우터·앙상블·원자 수 (★`B-16` 은 없는 키 `final_n` 을 읽어 전 행 None 이었다) ──
        auto_selected=r.get('auto_selected'), auto_n_selected=r.get('auto_n_selected'),
        auto_real_gate=r.get('auto_real_gate'), ensemble_k=r.get('ensemble_k'),
        n_gaussians=r.get('n_gaussians'),
        # ── 채점 창 (B-13: 두 팔이 다른 창에서 채점된 사고를 런 없이 검사한다) ────
        tracking_frames=r.get('tracking_frames'),
        scored_t_first=r.get('scored_t_first'), scored_t_last=r.get('scored_t_last'),
        episode_total_frames=r.get('episode_total_frames'),
        # ── ZERO (참 제로필드 기준; ⛔상수 1.0 아님) ────────────────────────────
        **z,
        # ── ⚠**의도표** — 러너가 *넣으려 한* 값이다. ⛔게이트가 증거로 쓰지 않는다.
        #    (`self.config = config` 때문에 이것과 행의 대조는 항등식이다.)
        cfg_intent={k: cfg.get(k) for k in
                    ('lambda_l1', 'init_method', 'learned_init_iters',
                     'auto_switch_thresh', 'varpro_steps', 'k_snap_dk',
                     'analytic_ensemble_k', 'tracking_mode')},
        # ── 재현 ────────────────────────────────────────────────────────────
        git_head=GIT_HEAD, import_md5=IMPORT_MD5,
        wall_s=wall)


def main():
    if '--selftest' in sys.argv:
        ok, _ = G.selftest()
        sys.exit(0 if ok else 1)
    # ★런 전 게이트 자가진단 — 실패하면 한 런도 돌리지 않는다.
    print('★런 전 게이트 자가진단 (가짜 행으로 게이트가 발화하는지) ─────────', flush=True)
    ok, _ = G.selftest()
    if not ok:
        print('⛔게이트 자가진단 실패 ⟹ 배치를 시작하지 않는다.', flush=True)
        sys.exit(1)
    print('─────────────────────────────────────────────────────────────\n', flush=True)

    shard = int(sys.argv[1]) if len(sys.argv) > 2 else 0
    nsh = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    smoke = os.environ.get('SMOKE') == '1'
    out = os.path.join(HERE, 'results_b19.jsonl' if nsh == 1
                       else f'results_b19_{shard}.jsonl')
    if smoke:
        out = os.path.join(HERE, 'SMOKE_results_b19.jsonl')
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
    if smoke:                       # ★1 셀 = 1 에피소드 × 3 팔 × 1 시드 (개입 발화 확인)
        eps, seeds = eps[:1], seeds[:1]
    nominal, confirm, not_earn = G.thresholds(G.N_EP)
    print(f'B-19 shard {shard}/{nsh}: 희소 에피소드 {len(eps)}개 · 팔 {G.ARM_ORDER} · '
          f'시드 {seeds} ⟹ {len(eps) * len(G.ARM_ORDER) * len(seeds)} 런', flush=True)
    print(f'  문턱(n={G.N_EP}): NOMINAL {nominal} · CONFIRM {confirm} · '
          f'NOT_EARN {not_earn} · NEG_MIN {nominal}', flush=True)
    print(f'  dk: KGRID1={G.DK1} KGRID2={G.DK2} · git {GIT_HEAD}', flush=True)
    with open(out, 'a') as fh:
        for ep in eps:
            for arm in G.ARM_ORDER:
                for s in seeds:
                    if (arm, ep['episode'], s) in done:
                        continue
                    try:
                        row = run_one(ep, arm, s)
                    except Exception as e:                   # noqa: BLE001
                        import traceback
                        row = dict(arm=arm, episode=ep['episode'], est_seed=s,
                                   error=repr(e), tb=traceback.format_exc()[-500:])
                    fh.write(json.dumps(row) + '\n')
                    fh.flush()
                    if 'error' in row:
                        print(f"{arm:8s} {ep['episode'][:36]:36s} s{s} {row['error'][:70]}",
                              flush=True)
                    else:
                        print(f"{arm:8s} {ep['episode'][:36]:36s} s{s} "
                              f"nRMSE={row['avg_nRMSE']:.4f} zero={row['zero_nRMSE']:.4f} "
                              f"snap={row['k_snap_fired']} cells={row['k_snap_cells']} "
                              f"N={row['n_gaussians']} frames={row['tracking_frames']} "
                              f"λ={(row['effective'] or {}).get('lambda_l1')} "
                              f"{row['wall_s']:.1f}s", flush=True)
    if smoke:
        print('SMOKE=1 ⟹ 1셀(3팔×1시드)에서 중단. ★결과 파일을 지울 것: ' + out, flush=True)


if __name__ == '__main__':
    main()
