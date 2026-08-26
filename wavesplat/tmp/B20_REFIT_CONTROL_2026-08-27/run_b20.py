"""B-20 러너 — **재적합 교락 통제**. 재개 가능·샤딩 가능.

사용:
    python3 tmp/B20_REFIT_CONTROL_2026-08-27/run_b20.py --selftest   # ★게이트 자가진단만
    python3 tmp/B20_REFIT_CONTROL_2026-08-27/run_b20.py <shard> <nshards>
    SMOKE=1 python3 ... run_b20.py                # 1셀(1에피소드×3팔×1시드). 결과 파일은 지울 것

★**런 전 게이트 자가진단이 자동으로 먼저 돈다.** 실패하면 한 런도 돌리지 않는다.
★**온라인 드리프트 중단**: `SHIP-NOW`·`KGRID1` 은 `B-19` 와 **같은 구성**이므로 **비트 동일**
  이어야 한다. 한 행이라도 어긋나면 **그 자리에서 exit 2** 한다(임무 지시의 ABORT).
  ⟹ 드리프트는 **첫 런**(에피소드 1의 `SHIP-NOW`)에서 드러난다 — 팔 순서가 그래서 그렇다.

## 훅 둘 — 무엇을 사고 무엇을 증명하는가
1. `make_updater` **관측 훅**(`B-19` 에서 검증된 패턴): `run_estimation` 이 실제로 만든 추정기
   객체를 붙잡아 **파생 속성**을 읽는다. ⛔설정 dict 을 다시 읽지 않는다(항등식).
2. `SequentialUpdater._snap_k_to_grid` **계측 훅**(신규): 진짜 메서드를 호출하기 **전후로**
   `kx, ky, w, φ` 를 스냅샷해 다음을 행에 싣는다 —
     · `w_phi_changed` (재적합이 실제로 발화했는가; `SHIP-NOW` 에서는 **False** 여야 한다)
     · `k_bit_identical_frac` (파수가 **float32 파라미터에서** 비트 동일한 원자 비율)
     · `k_dk_max`, `k_dpsi_max` (파수 섭동이 렌더 장에 만드는 **최대 위상 오차**)
   ⛔거동을 바꾸지 않는다: 진짜 메서드의 반환값을 **그대로** 돌려주고 모델을 건드리지 않는다
   (스냅샷은 `detach().clone()`). **증명은 추론이 아니라 게이트다** — `G-11`·`G-11b` 가
   `SHIP-NOW`·`KGRID1` 96행을 훅 없이 돌았던 `B-19` 동결 행과 **비트 동일**로 대조하고,
   어긋나면 러너가 그 자리에서 멈춘다.

## 왜 `_snap_k_to_grid` 를 `SHIP-NOW` 에서도 훅으로 통과시키는가
`k_snap_dk=None` 이면 그 함수는 `if not dk: return None`(`sequential_update.py:947-948`) 로
즉시 빠져나온다. ⟹ **세 팔이 정확히 같은 코드 경로**를 지나고, `G-3b`(`snap_calls == 1`)와
`G-14`(SHIP 에서 `(w,φ)` 불변)가 그 사실을 **행의 수**로 남긴다.
"""
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import warnings

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for _p in (HERE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(ROOT)
import gates_b20 as G                                          # noqa: E402

# ⚠`git_head` 만으로는 **더러운 작업트리를 못 잡는다** ⟹ 임포트 경로 파일의 md5 동반 기록.
IMPORT_PATH = ['wavesplat/run.py',
               'wavesplat/optimizer/sequential_update.py',
               'wavesplat/optimizer/learned_init.py',
               'wavesplat/optimizer/varpro.py',
               'wavesplat/model/tnse.py',
               'wavesplat/model/dispersion.py',
               'wavesplat/evaluation/metrics.py']


def _env():
    try:
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                       cwd=ROOT, text=True).strip()
    except Exception:                                          # noqa: BLE001
        head = None
    md5 = {p: hashlib.md5(open(os.path.join(ROOT, p), 'rb').read()).hexdigest()
           for p in IMPORT_PATH if os.path.exists(os.path.join(ROOT, p))}
    return head, md5


GIT_HEAD, IMPORT_MD5 = _env()

# ── 훅 1: 추정기 객체 포착 (B-19 패턴 그대로) ─────────────────────────────────
_CAPTURED = []


def _install_capture():
    import wavesplat.run as WR
    if getattr(WR.make_updater, '_b20_capture', False):
        return
    real = WR.make_updater

    def capturing(config):
        u = real(config)
        _CAPTURED.append(u)
        return u

    capturing._b20_capture = True
    capturing._b20_real = real
    WR.make_updater = capturing


# ── 훅 2: `_snap_k_to_grid` 전후 파라미터 계측 (신규) ──────────────────────────
_SNAP = []


def _measure(kx0, ky0, w0, p0, kx1, ky1, w1, p1, depth):
    """파라미터 수준 개입 계측. ⛔모델을 건드리지 않는다 (읽기 전용 사본)."""
    import numpy as np
    from wavesplat.model.dispersion import G as GRAV          # 단일출처
    n = int(kx0.shape[0])
    if n == 0:
        return dict(n_atoms=0, k_bit_identical_frac=None, k_dkx_max=None,
                    k_dky_max=None, k_dk_max=None, k_dpsi_max=None, k_min_mag=None,
                    w_phi_changed=None, w_abs_change_max=None, phi_abs_change_max=None)
    same = int(((kx0 == kx1) & (ky0 == ky1)).sum())            # float32 비트 동일
    a = [np.asarray(t.numpy(), dtype=np.float64) for t in (kx0, ky0, kx1, ky1, w0, p0, w1, p1)]
    x0, y0, x1, y1, ww0, pp0, ww1, pp1 = a
    dkx, dky = np.abs(x1 - x0), np.abs(y1 - y0)
    m0, m1 = np.hypot(x0, y0), np.hypot(x1, y1)
    om0 = np.sqrt(GRAV * m0 * np.tanh(m0 * depth))
    om1 = np.sqrt(GRAV * m1 * np.tanh(m1 * depth))
    dpsi = dkx * G.XMAX_PIN + dky * G.YMAX_PIN + np.abs(om1 - om0) * G.TMAX_PIN
    dphi = np.abs((pp1 - pp0 + math.pi) % (2 * math.pi) - math.pi)
    return dict(n_atoms=n, k_bit_identical_frac=same / n,
                k_dkx_max=float(dkx.max()), k_dky_max=float(dky.max()),
                k_dk_max=float(np.hypot(dkx, dky).max()),
                k_dpsi_max=float(dpsi.max()), k_min_mag=float(m0.min()),
                w_phi_changed=bool(not (np.array_equal(ww0, ww1)
                                        and np.array_equal(pp0, pp1))),
                w_abs_change_max=float(np.abs(ww1 - ww0).max()),
                phi_abs_change_max=float(dphi.max()))


def _install_snap_capture():
    import torch
    from wavesplat.optimizer import sequential_update as SU
    if getattr(SU.SequentialUpdater._snap_k_to_grid, '_b20_capture', False):
        return
    real = SU.SequentialUpdater._snap_k_to_grid

    def capturing(self, frames, verbose=False):
        with torch.no_grad():
            b = tuple(t.detach().to('cpu').clone() for t in
                      (self.model.kx, self.model.ky, self.model.w, self.model.phi))
        rec = real(self, frames, verbose)
        with torch.no_grad():
            a = tuple(t.detach().to('cpu').clone() for t in
                      (self.model.kx, self.model.ky, self.model.w, self.model.phi))
        _SNAP.append(dict(rec=rec, **_measure(b[0], b[1], b[2], b[3],
                                              a[0], a[1], a[2], a[3],
                                              float(self.depth))))
        return rec                                             # ★그대로 돌려준다

    capturing._b20_capture = True
    capturing._b20_real = real
    SU.SequentialUpdater._snap_k_to_grid = capturing


def run_one(ep, arm, est_seed):
    from wavesplat.run import run_estimation, DEFAULT_CONFIG
    _install_capture()
    _install_snap_capture()
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(G.deploy_constants())      # ★명시 — 런타임 스냅샷이 아니라 단일출처 되읽기
    cfg.update(G.ARMS[arm])               # ★개입은 이 한 줄뿐이다
    cfg['seed'] = est_seed
    _CAPTURED.clear()
    _SNAP.clear()
    t0 = time.time()
    r = run_estimation(ep['path'], config=cfg, verbose=False)
    wall = time.time() - t0
    eff = G.effective_from_updater(_CAPTURED[-1] if _CAPTURED else None,
                                   n_captured=len(_CAPTURED))
    snap = dict(_SNAP[-1]) if _SNAP else {}
    z = G.zero_nrmse(ep['path'])          # ★행마다 새로 계산(캐시 없음) ⟹ G-9 가 진짜 검사
    return dict(
        arm=arm, est_seed=est_seed, episode=ep['episode'], bf=ep['bf'],
        ncomp=ep['ncomp'], gen_seed=ep['gen_seed'],
        avg_nRMSE=float(r['avg_nRMSE']),
        # ── 개입 계수기 (패키지가 내보내는 값) ────────────────────────────────
        k_snap_dk=r.get('k_snap_dk'), k_snap_fired=bool(r.get('k_snap_fired')),
        k_snap_cells=r.get('k_snap_cells'),
        k_snap_shift_median=r.get('k_snap_shift_median'),
        # ── ★훅 2: 파라미터 수준 개입 계측 (G-3b · G-7 · G-14 · G-15 · G-15b) ──
        snap_rec=snap.get('rec'), snap_calls=len(_SNAP),
        w_phi_changed=snap.get('w_phi_changed'),
        w_abs_change_max=snap.get('w_abs_change_max'),
        phi_abs_change_max=snap.get('phi_abs_change_max'),
        k_bit_identical_frac=snap.get('k_bit_identical_frac'),
        k_dkx_max=snap.get('k_dkx_max'), k_dky_max=snap.get('k_dky_max'),
        k_dk_max=snap.get('k_dk_max'), k_dpsi_max=snap.get('k_dpsi_max'),
        k_min_mag=snap.get('k_min_mag'),
        # ── ★훅 1: 객체 파생 속성 (G-5 가 검사하는 **유일한** 되읽기 증거) ─────
        effective=eff,
        # ── 계수기 (init_result 유래, G-5c) ───────────────────────────────────
        varpro_steps=r.get('varpro_steps'), varpro_fired=bool(r.get('varpro_fired')),
        varpro_obj_before=r.get('varpro_obj_before'),
        varpro_obj_after=r.get('varpro_obj_after'),
        varpro_k_shift=r.get('varpro_k_shift'),
        init_loss_n=r.get('init_loss_n'),
        # ── 라우터·앙상블·원자 수 ─────────────────────────────────────────────
        auto_selected=r.get('auto_selected'), auto_n_selected=r.get('auto_n_selected'),
        auto_real_gate=r.get('auto_real_gate'), ensemble_k=r.get('ensemble_k'),
        n_gaussians=r.get('n_gaussians'),
        # ── 채점 창 (B-13 부류: 두 팔이 다른 창에서 채점되는 사고) ──────────────
        tracking_frames=r.get('tracking_frames'),
        scored_t_first=r.get('scored_t_first'), scored_t_last=r.get('scored_t_last'),
        episode_total_frames=r.get('episode_total_frames'),
        # ── ZERO (참 제로필드 기준; ⛔상수 1.0 아님) ────────────────────────────
        **z,
        # ── ⚠**의도표** — 러너가 *넣으려 한* 값이다. ⛔게이트가 증거로 쓰지 않는다.
        cfg_intent={k: cfg.get(k) for k in
                    ('lambda_l1', 'init_method', 'learned_init_iters',
                     'auto_switch_thresh', 'varpro_steps', 'k_snap_dk',
                     'analytic_ensemble_k', 'tracking_mode')},
        git_head=GIT_HEAD, import_md5=IMPORT_MD5, wall_s=wall)


def main():
    if '--selftest' in sys.argv:
        ok, _ = G.selftest()
        sys.exit(0 if ok else 1)
    print('★런 전 게이트 자가진단 (가짜 행으로 게이트가 발화하는지) ─────────', flush=True)
    ok, _ = G.selftest()
    if not ok:
        print('⛔게이트 자가진단 실패 ⟹ 배치를 시작하지 않는다.', flush=True)
        sys.exit(1)
    print('─────────────────────────────────────────────────────────────\n', flush=True)

    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    shard = int(args[0]) if len(args) >= 2 else 0
    nsh = int(args[1]) if len(args) >= 2 else 1
    smoke = os.environ.get('SMOKE') == '1'
    out = os.path.join(HERE, 'results_b20.jsonl' if nsh == 1
                       else f'results_b20_{shard}.jsonl')
    if smoke:
        out = os.path.join(HERE, 'SMOKE_results_b20.jsonl')

    # ★온라인 드리프트 기준 (B-19 동결 행)
    ref = {a: G.b19_ref(a) for a in G.POSCTRL_ARMS}
    for a in G.POSCTRL_ARMS:
        if len(ref[a]) != G.N_EP * len(G.EST_SEEDS):
            print(f'⛔B-19 기준 행이 부족하다: {a} {len(ref[a])} 행 ⟹ 중단', flush=True)
            sys.exit(3)

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
    nominal, confirm, not_earn = G.thresholds(G.N_EP)
    d = G.derive_dk0()
    print(f'B-20 shard {shard}/{nsh}: 희소 에피소드 {len(eps)}개 · 팔 {G.ARM_ORDER} · '
          f'시드 {seeds} ⟹ {len(eps) * len(G.ARM_ORDER) * len(seeds)} 런', flush=True)
    print(f'  문턱(n={G.N_EP}): NOMINAL {nominal} · CONFIRM {confirm} · '
          f'NOT_EARN {not_earn} · NEG_MIN {nominal} · family_alpha {G.FAMILY_ALPHA}',
          flush=True)
    print(f'  dk: NOOP-REFIT={G.DK0:g} (상한 {d["bound"]:.4e}, Δψ_max='
          f'{(G.DK0 / 2) * d["lever"]:.3e} rad = {(G.DK0 / 2) * d["lever"] / G.EPS32:.3f}'
          f'×eps32) · KGRID1={G.DK1} (Δψ_max={(G.DK1 / 2) * d["lever"]:.2f} rad)', flush=True)
    print(f'  PP_BAND={G.pp_band():.4f} pp · git {GIT_HEAD}', flush=True)

    with open(out, 'a') as fh:
        for ep in eps:
            for arm in G.ARM_ORDER:
                for s in seeds:
                    if (arm, ep['episode'], s) in done:
                        continue
                    try:
                        row = run_one(ep, arm, s)
                    except Exception as e:                     # noqa: BLE001
                        import traceback
                        row = dict(arm=arm, episode=ep['episode'], est_seed=s,
                                   error=repr(e), tb=traceback.format_exc()[-500:])
                    fh.write(json.dumps(row) + '\n')
                    fh.flush()
                    if 'error' in row:
                        print(f"{arm:11s} {ep['episode'][:34]:34s} s{s} {row['error'][:70]}",
                              flush=True)
                        continue
                    print(f"{arm:11s} {ep['episode'][:34]:34s} s{s} "
                          f"nRMSE={row['avg_nRMSE']:.6f} zero={row['zero_nRMSE']:.4f} "
                          f"snap={row['k_snap_fired']} calls={row['snap_calls']} "
                          f"wφ±={row['w_phi_changed']} bitk={row['k_bit_identical_frac']} "
                          f"Δψ={row['k_dpsi_max']:.3e} N={row['n_gaussians']} "
                          f"{row['wall_s']:.1f}s", flush=True)
                    # ★온라인 드리프트 중단 (임무 지시의 ABORT)
                    if arm in G.POSCTRL_ARMS:
                        want = ref[arm].get((ep['episode'], s))
                        if want is None or row['avg_nRMSE'] != want:
                            print(f'⛔양성대조 드리프트: {arm} {ep["episode"]} s{s} '
                                  f'{row["avg_nRMSE"]!r} != B-19 {want!r} ⟹ **중단**. '
                                  f'PREREG §7.5 에 따라 이 배치는 B-16/B-19 옆에 놓일 수 없다.',
                                  flush=True)
                            sys.exit(2)
    if smoke:
        print('SMOKE=1 ⟹ 1셀에서 중단. ★결과 파일을 지울 것: ' + out, flush=True)


if __name__ == '__main__':
    main()
