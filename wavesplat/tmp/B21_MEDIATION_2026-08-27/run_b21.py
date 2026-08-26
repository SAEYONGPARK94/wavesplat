"""B-21 러너 — 매개(mediation)를 **한 배치에서** 잰다. 재개 가능·샤딩 가능.

사용:
    python3 tmp/B21_MEDIATION_2026-08-27/run_b21.py                 # 단일 프로세스
    python3 tmp/B21_MEDIATION_2026-08-27/run_b21.py <shard> <nshards>
    python3 tmp/B21_MEDIATION_2026-08-27/run_b21.py --selftest      # ★게이트 자가진단만
    SMOKE=1 python3 ... run_b21.py                                  # 1셀 (결과 파일은 지울 것)

★**런 전 게이트 자가진단이 자동으로 먼저 돈다.** 실패하면 한 런도 돌리지 않는다.

## 팔 다섯 — 판정에 쓰는 정의는 전부 `gates_b21` 에서 온다 (⛔손복사 0)
`SHIP`/`KGRID1`/`KGRID2`/`FITSWAP` 은 배포 진입점 `wavesplat.run.run_estimation` 을 부르고
`k_snap_dk` **하나만** 바꾼다. 배포 상수 다섯은 패키지 단일출처에서 읽어 **명시**한다
(⛔`dict(DEFAULT_CONFIG)` 런타임 스냅샷에 기대지 않는다 — S13·S13b·HOSGATE 가 그것으로 물렸다).
`SPEC25` 는 `S16` 의 `run_s16.run_letkf` 를 **그대로 import** 한다(`B-11b`·`B-12`·`B-15` 규약).

## 되읽기 — 무엇이 증거이고 무엇이 아닌가
⛔`cfg.get(k)` 를 행에 실어 게이트가 같은 dict 과 대조하는 것은 **항등식**이다
(`sequential_update.py:399 self.config = config` 가 입력 dict 을 복사 없이 붙든다).
✅`make_updater` 를 **관측 훅**(`gates_b21.install_capture`)으로 감싸 실제로 만들어진
추정기 객체를 붙잡고 `effective_from_updater` 가 **객체의 파생 속성**을 읽는다.
설정 dict 값은 `cfg_intent` 라는 **다른 이름**으로만 남기고 게이트는 그것을 읽지 않는다.
`SPEC25` 는 `letkf_spec.readback(f)` 가 **필터 객체**에서 읽은 `effective_config` 를 싣는다.

## ★채점 창 — 이 배치가 교차배치 결합보다 나은 이유
행마다 `n_scored` 를 싣는다. 그 수는 **세 독립 경로**에서 온다:
  · ablation : `run_estimation` 의 `tracking_frames`
  · `SPEC25` : `run_letkf` 의 `n_match` (`sel = t > 2·Tp` 를 스스로 센 값)
  · `ZERO`   : `gates_b19.zero_nrmse` 가 자료에서 다시 센 `zero_n_frames`
G-8 이 셋의 일치를 요구한다. ⚠**`SPEC25` 행의 `scored_t_first/last` 는 ZERO 경로가 재도출한
값**이다(`run_letkf` 가 시각을 반환하지 않는다) — 하중을 지는 것은 **개수 세 개의 일치**다.

## 실행 순서 (결과 전 고정)
`sparse`(16) → `crowded`(48). ★**순서는 비용상의 일정일 뿐이다** — ⛔`sparse` 결과를 보고
`crowded` 를 돌릴지 말지 정하지 않는다. **두 층은 무조건 돈다.** 층별 문턱·라벨·팔은
`gates_b21` 에 결과 전에 박혀 있고 한쪽 결과가 다른 쪽을 바꾸지 않는다.
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
S16 = os.path.join(ROOT, 'tmp', 'S16_BASELINE_2026-08-14')
for _p in (HERE, ROOT, S16):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(ROOT)
import gates_b21 as G                                        # noqa: E402

# ⚠`git_head` 만으로는 **더러운 작업트리를 못 잡는다**(S13·S13b) ⟹ 임포트 경로 md5 동반.
IMPORT_PATH = ['wavesplat/run.py',
               'wavesplat/optimizer/sequential_update.py',
               'wavesplat/optimizer/learned_init.py',
               'wavesplat/optimizer/varpro.py',
               'wavesplat/evaluation/metrics.py',
               'wavesplat/baselines/letkf_grid.py',
               'tmp/S16_BASELINE_2026-08-14/run_s16.py',
               'tmp/S16_BASELINE_2026-08-14/letkf_spec.py']


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
CFG_INTENT_KEYS = ('lambda_l1', 'init_method', 'learned_init_iters',
                   'auto_switch_thresh', 'varpro_steps', 'k_snap_dk',
                   'analytic_ensemble_k', 'tracking_mode')


def _common(ep, arm, seed, kind, z, wall):
    return dict(arm=arm, stratum=ep['stratum'], episode=ep['episode'], bf=ep['bf'],
                ncomp=ep['ncomp'], gen_seed=ep['gen_seed'],
                seed_kind=kind, seed=seed, wall_s=wall,
                git_head=GIT_HEAD, import_md5=IMPORT_MD5, **z)


def run_ablation(ep, arm, est_seed):
    from wavesplat.run import run_estimation, DEFAULT_CONFIG
    G.install_capture()
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(G.deploy_constants())      # ★명시 — 런타임 스냅샷이 아니라 단일출처 되읽기
    cfg.update(G.ARMS[arm])               # ★개입은 이 한 줄뿐이다
    cfg['seed'] = est_seed
    G._CAPTURED.clear()
    t0 = time.time()
    r = run_estimation(ep['path'], config=cfg, verbose=False)
    wall = time.time() - t0
    eff = G.effective_from_updater(G._CAPTURED[-1] if G._CAPTURED else None,
                                   n_captured=len(G._CAPTURED))
    z = G.zero_nrmse(ep['path'])          # ★매 행 새로 계산(캐시 없음) ⟹ G-9 가 진짜 검사
    row = _common(ep, arm, est_seed, 'est', z, wall)
    row.update(
        avg_nRMSE=float(r['avg_nRMSE']),
        # 개입 계수기 (되읽기: `_snap_k_to_grid` 가 실제로 쓴 값)
        k_snap_dk=r.get('k_snap_dk'), k_snap_fired=bool(r.get('k_snap_fired')),
        k_snap_cells=r.get('k_snap_cells'),
        k_snap_shift_median=r.get('k_snap_shift_median'),
        # ★객체 파생 속성 (G-5 가 검사하는 **유일한** 되읽기 증거)
        effective=eff,
        # VARPRO 실집행 증거 — `varpro_steps` 는 echo 이고 `varpro_fired` 는 건너뛴
        # 경우에도 truthy 다(`varpro.py:201`) ⟹ 실행됐을 때만 채워지는 값을 함께 싣는다.
        varpro_steps=r.get('varpro_steps'), varpro_fired=bool(r.get('varpro_fired')),
        varpro_obj_before=r.get('varpro_obj_before'),
        varpro_obj_after=r.get('varpro_obj_after'),
        init_loss_n=r.get('init_loss_n'),
        auto_selected=r.get('auto_selected'), auto_n_selected=r.get('auto_n_selected'),
        auto_real_gate=r.get('auto_real_gate'), ensemble_k=r.get('ensemble_k'),
        n_gaussians=r.get('n_gaussians'),
        # 채점 창 — G-8
        n_scored=r.get('tracking_frames'), tracking_frames=r.get('tracking_frames'),
        scored_t_first=r.get('scored_t_first'), scored_t_last=r.get('scored_t_last'),
        episode_total_frames=r.get('episode_total_frames'),
        # ⚠**의도표** — 러너가 *넣으려 한* 값. ⛔게이트가 증거로 쓰지 않는다.
        cfg_intent={k: cfg.get(k) for k in CFG_INTENT_KEYS})
    return row


def run_baseline(ep, filter_seed):
    import run_s16                        # ★기준선 정의를 손으로 옮기지 않는다
    t0 = time.time()
    r = run_s16.run_letkf(ep['path'], G.BASELINE_ARM, filter_seed=filter_seed)
    wall = time.time() - t0
    z = G.zero_nrmse(ep['path'])
    row = _common(ep, G.BASELINE_ARM, filter_seed, 'filter', z, wall)
    row.update(
        avg_nRMSE=float(r['avg_nRMSE']),
        baseline_zero_nRMSE=float(r['zero_nRMSE']),   # 기준선이 스스로 낸 값 (G-12 대조용)
        effective_config=r.get('effective_config'),   # ★필터 **객체** 되읽기
        loc_radius=r.get('loc_radius'), n_ensemble=r.get('n_ensemble'),
        background=r.get('background'), filter_class=r.get('filter_class'),
        ms_median=r.get('ms_median'), n_obs_median=r.get('n_obs_median'),
        # ★채점 창 — `n_match` 는 `run_letkf` 가 `t > 2·Tp` 를 스스로 센 값이다.
        #   ⚠시각 두 개는 `run_letkf` 가 반환하지 않으므로 **ZERO 경로가 재도출**한다.
        n_scored=r.get('n_match'), n_all=r.get('n_all'),
        scored_t_first=z.get('zero_t_first'), scored_t_last=z.get('zero_t_last'),
        episode_total_frames=r.get('n_all'))
    return row


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

    shard = int(sys.argv[1]) if len(sys.argv) > 2 else 0
    nsh = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    smoke = os.environ.get('SMOKE') == '1'
    out = os.path.join(HERE, 'results_b21.jsonl' if nsh == 1
                       else f'results_b21_{shard}.jsonl')
    if smoke:
        out = os.path.join(HERE, 'SMOKE_results_b21.jsonl')

    done = set()
    if os.path.exists(out):
        for line in open(out):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'error' not in r:
                done.add((r['arm'], r['episode'], r['seed']))

    eps = [e for st in G.STRATA for e in G.episodes(st)]      # sparse 먼저 (§실행 순서)
    eps = [e for i, e in enumerate(eps) if i % nsh == shard]
    if smoke:
        eps = eps[:1]
    n_runs = 0
    for e in eps:
        n_runs += len(G.ARM_ORDER) * (1 if smoke else len(G.EST_SEEDS))
        n_runs += 1 if smoke else len(G.FILTER_SEEDS)
    print(f'B-21 shard {shard}/{nsh}: 에피소드 {len(eps)}개 · 팔 {G.ALL_ARMS} ⟹ {n_runs} 런',
          flush=True)
    for st in G.STRATA:
        nom, con = G.thresholds(G.STRATUM_N[st])
        print(f'  문턱(n={G.STRATUM_N[st]}, {st}): NOMINAL {nom} · '
              f'CONFIRM(α={G.FAMILY_ALPHA:.5g}) {con}', flush=True)
    print(f'  dk: KGRID1={G.DK1} KGRID2={G.DK2} FITSWAP={G.FITSWAP_DK} · '
          f'TOST Δ={G.TOST_DELTA_PCT}% / Δ_conf={G.TOST_DELTA_CONF_PCT}% · git {GIT_HEAD}',
          flush=True)

    with open(out, 'a') as fh:
        for ep in eps:
            for arm in G.ALL_ARMS:
                seeds = (G.FILTER_SEEDS if arm == G.BASELINE_ARM else G.EST_SEEDS)
                seeds = list(seeds)[:1] if smoke else list(seeds)
                for s in seeds:
                    if (arm, ep['episode'], s) in done:
                        continue
                    try:
                        row = (run_baseline(ep, s) if arm == G.BASELINE_ARM
                               else run_ablation(ep, arm, s))
                    except Exception as exc:                 # noqa: BLE001
                        import traceback
                        row = dict(arm=arm, stratum=ep['stratum'],
                                   episode=ep['episode'], seed=s, error=repr(exc),
                                   tb=traceback.format_exc()[-600:])
                    fh.write(json.dumps(row) + '\n')
                    fh.flush()
                    if 'error' in row:
                        print(f"{arm:8s} {ep['episode'][:34]:34s} s{s} "
                              f"{row['error'][:70]}", flush=True)
                    else:
                        print(f"{arm:8s} {ep['episode'][:34]:34s} s{s} "
                              f"nRMSE={row['avg_nRMSE']:.6f} "
                              f"zero={row['zero_nRMSE']:.4f} "
                              f"snap={row.get('k_snap_fired')} "
                              f"shift={row.get('k_snap_shift_median')} "
                              f"frames={row['n_scored']} {row['wall_s']:.1f}s", flush=True)
    if smoke:
        print('SMOKE=1 ⟹ 1셀에서 중단. ★결과 파일을 지울 것: ' + out, flush=True)


if __name__ == '__main__':
    main()
