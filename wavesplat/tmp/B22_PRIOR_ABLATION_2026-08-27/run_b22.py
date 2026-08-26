"""B-22 러너 — **학습된 사전의 절제** (8팔 × 48 에피소드 × 3 시드 = 1,152런). 재개·샤딩 가능.

사용:
    python3 tmp/B22_PRIOR_ABLATION_2026-08-27/run_b22.py --selftest      # 게이트 자가진단만
    python3 tmp/B22_PRIOR_ABLATION_2026-08-27/run_b22.py <shard> <nshards>
    SMOKE=1 python3 ... run_b22.py                                        # 1셀 (결과 파일은 지울 것)

★**런 전 게이트 자가진단이 자동으로 먼저 돈다.** 실패하면 한 런도 돌리지 않는다.

## 개입은 **원자 배열 하나**다
배포 진입점 `wavesplat.run.run_estimation` 을 부르고, 팔에 따라
`config['learned_atoms']`(이미 노출된 배포 노브 — 라우터 자신이 `:1032` 에서 채운다)에
**학습 원자** 또는 **해석적 원자**를 넣는다. 나머지는 배포 그대로다.
⛔팔은 `gates_b22.arm_config(arm, atoms)` 가 만든다(러너에 설정 리터럴 0개).

## 되읽기 — ⛔`cfg[k] == cfg[k]` 항등식 금지 (`B-19` §3.1 상속)
`make_updater` 를 **관측 훅**으로 감싸 실제로 만들어진 추정기 객체를 붙잡고
`gates_b19.effective_from_updater()` 로 **객체 파생 속성**을 읽는다. 훅은 진짜 함수의 반환값을
**그대로** 돌려준다(래핑·상속 없음). 훅이 거동을 바꾸지 않았다는 증명은 추론이 아니라
**G-10**(SHIP 이 `B-12` 동결 행과 비트 동일)이다.

★**이 배치의 개입에 대한 되읽기**는 `auto_n_selected`(추정기가 계산) vs `n_pool_expected`
(러너가 **주입 배열**에 `pool_and_select` 를 돌려 낸 기대값)의 대조다 — 두 값의 출처가
다르므로 항등식이 아니다(G-4).

## 원자 캐시
`atoms/<episode>__<source>__s<seed>.json` 에 남긴다(작다). ⟹ 개입이 **사후에도 검증 가능**하고
`atoms_md5` 가 행에 실린다. 해석적 제안은 **그리디 접두**라 `max_n` 한 번만 부르고
`ANA`(전체)와 `ANA-N`(앞 n_L 개)이 나눠 쓴다(배선시험에서 접두 비트 동일 확인).
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
import gates_b22 as G                                        # noqa: E402

ATOMS_DIR = os.path.join(HERE, 'atoms')
IMPORT_PATH = ['wavesplat/run.py',
               'wavesplat/optimizer/sequential_update.py',
               'wavesplat/optimizer/learned_init.py',
               'wavesplat/optimizer/initializer.py',
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
_CAPTURED = []


def _install_capture():
    import wavesplat.run as WR
    if getattr(WR.make_updater, '_b22_capture', False):
        return
    real = WR.make_updater

    def capturing(config):
        u = real(config)
        _CAPTURED.append(u)
        return u

    capturing._b22_capture = True
    capturing._b22_real = real
    WR.make_updater = capturing


# ── ★제안 계수기: `propose_atoms`(확산 사전)가 **런 안에서** 몇 번 불렸나 ──────
#    ⛔거동을 바꾸지 않는다(진짜 함수의 반환값을 그대로 돌려준다).
#    ★이것이 주입의 **직접 증거**다: 주입이 실패했으면 라우터가 제안을 돌아 1 이 된다.
#    ⚠러너 자신도 `prepare_atoms` 에서 이 함수를 부르므로 **런 직전에 0 으로 되돌린다.**
_PROPOSE = [0]


def _install_propose_counter():
    import wavesplat.optimizer.learned_init as LI
    if getattr(LI.propose_atoms, '_b22_count', False):
        return
    real = LI.propose_atoms

    def counting(*a, **kw):
        _PROPOSE[0] += 1
        return real(*a, **kw)

    counting._b22_count = True
    counting._b22_real = real
    LI.propose_atoms = counting


# ── 원자 준비 (캐시) ─────────────────────────────────────────────────────────
def _cache_path(ep, source, seed):
    return os.path.join(ATOMS_DIR, f"{ep['episode']}__{source}__s{seed}.json")


def prepare_atoms(ep, est_seed):
    """이 (에피소드, 시드) 에 필요한 원자 셋을 만들고(또는 캐시에서 읽고) 돌려준다.

    반환 `{source: (atoms, meta)}` — `source ∈ {'learned','analytic','analytic_n'}`.
    """
    os.makedirs(ATOMS_DIR, exist_ok=True)
    out = {}
    # 1) 학습 원자 (확산 사전) — 라우터가 인라인으로 만드는 것과 같은 호출
    p = _cache_path(ep, 'learned', est_seed)
    if os.path.exists(p):
        d = json.load(open(p))
    else:
        atoms, pad = G.learned_raw_atoms(ep['path'], est_seed)
        d = dict(atoms=atoms, n_pad_dropped=pad)
        json.dump(d, open(p, 'w'))
    n_L = G.pooled_count(d['atoms'], G.arm_config('LEARN')['learned_energy_thresh'])
    out['learned'] = (d['atoms'], dict(n_pad_dropped=d['n_pad_dropped'], n_pool=n_L))

    # 2) 해석적 원자 (v2 빔포밍) — `max_n` 개를 **한 번** 뽑아 두 팔이 나눠 쓴다
    p = _cache_path(ep, 'analytic', est_seed)
    if os.path.exists(p):
        d2 = json.load(open(p))
    else:
        atoms, pad = G.analytic_raw_atoms(ep['path'], est_seed,
                                          G.arm_config('ANA')['max_n'])
        d2 = dict(atoms=atoms, n_pad_dropped=pad)
        json.dump(d2, open(p, 'w'))
    out['analytic'] = (d2['atoms'], dict(n_pad_dropped=d2['n_pad_dropped'],
                                         n_pool=G.pooled_count(
                                             d2['atoms'],
                                             G.arm_config('ANA')['learned_energy_thresh'])))
    # 3) 개수 일치 팔 = 같은 제안의 **앞 n_L 개** (그리디 접두 = 가장 강한 n_L 개)
    an = G.prefix_atoms(d2['atoms'], n_L)
    out['analytic_n'] = (an, dict(n_pad_dropped=d2['n_pad_dropped'],
                                  n_pool=G.pooled_count(
                                      an, G.arm_config('ANA-N')['learned_energy_thresh'])))
    return out


def run_one(ep, arm, est_seed, atom_bank):
    from wavesplat.run import run_estimation
    _install_capture()
    _install_propose_counter()
    source = G.ARMS[arm]['atoms']
    atoms, meta = (None, {}) if source is None else atom_bank[source]
    cfg = G.arm_config(arm, atoms)          # ★팔 정의는 gates_b22 단일출처
    cfg['seed'] = est_seed
    _CAPTURED.clear()
    _PROPOSE[0] = 0                         # ★러너의 `prepare_atoms` 호출을 세지 않는다
    t0 = time.time()
    r = run_estimation(ep['path'], config=cfg, verbose=False)
    wall = time.time() - t0
    eff = G.effective_from_updater(_CAPTURED[-1] if _CAPTURED else None,
                                   n_captured=len(_CAPTURED))
    z = G.zero_nrmse(ep['path'])            # ★매 행 새로 계산(캐시 없음) ⟹ G-9 가 진짜 검사
    return dict(
        arm=arm, est_seed=est_seed, episode=ep['episode'], bf=ep['bf'],
        ncomp=ep['ncomp'], gen_seed=ep['gen_seed'],
        avg_nRMSE=float(r['avg_nRMSE']),
        # ── 개입 계수기 (원자 주입) ──────────────────────────────────────────
        atoms_source=source,
        atoms_md5=G.atoms_md5(atoms),
        n_atoms_injected=(None if atoms is None else len(atoms['kx'])),
        n_pad_dropped=meta.get('n_pad_dropped'),
        n_pool_expected=meta.get('n_pool'),          # ★러너가 배열에서 도출 (G-4 의 오른쪽)
        # ★런 안에서 확산 제안이 몇 번 돌았나 — 주입의 **직접 증거**(G-14).
        #   ⛔`init_time_s` 로는 잴 수 없다: 그 값은 `_initialize_learned` 안에서 재기 시작해
        #   **라우터의 제안 시간을 포함하지 않는다**(1셀 스모크 실측 0.76 vs 0.75 s).
        propose_calls=_PROPOSE[0],
        # ── 객체 파생 속성 (G-5 가 검사하는 **유일한** 되읽기 증거) ──────────
        effective=eff,
        # ── 계수기 ───────────────────────────────────────────────────────────
        varpro_steps=r.get('varpro_steps'), varpro_fired=bool(r.get('varpro_fired')),
        varpro_obj_before=r.get('varpro_obj_before'),
        varpro_obj_after=r.get('varpro_obj_after'),
        varpro_k_shift=r.get('varpro_k_shift'),
        init_iters=r.get('init_iters'), init_loss_n=r.get('init_loss_n'),
        init_time_s=r.get('init_time_s'),
        # ── 라우터·앙상블·원자 수 ────────────────────────────────────────────
        auto_selected=r.get('auto_selected'), auto_n_selected=r.get('auto_n_selected'),
        auto_real_gate=r.get('auto_real_gate'), ensemble_k=r.get('ensemble_k'),
        ensemble_member_n=r.get('ensemble_member_n'), n_gaussians=r.get('n_gaussians'),
        # ── 채점 창 (B-13: 두 팔이 다른 창에서 채점된 사고를 런 없이 검사) ────
        tracking_frames=r.get('tracking_frames'),
        scored_t_first=r.get('scored_t_first'), scored_t_last=r.get('scored_t_last'),
        episode_total_frames=r.get('episode_total_frames'),
        # ── ZERO (참 제로필드; ⛔상수 1.0 아님) ───────────────────────────────
        **z,
        # ── ⚠**의도표** — 러너가 *넣으려 한* 값. ⛔게이트가 증거로 쓰지 않는다.
        cfg_intent={k: (('<atoms n=%d>' % len(cfg[k]['kx'])) if k == 'learned_atoms'
                        and cfg[k] else cfg.get(k)) for k in
                    ('lambda_l1', 'init_method', 'learned_init_iters',
                     'auto_switch_thresh', 'varpro_steps', 'learned_energy_thresh',
                     'learned_dedup_dk', 'max_n', 'analytic_ensemble_k',
                     'learned_atoms')},
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

    shard = int(sys.argv[1]) if len(sys.argv) > 2 else 0
    nsh = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    smoke = os.environ.get('SMOKE') == '1'
    out = os.path.join(HERE, 'results_b22.jsonl' if nsh == 1
                       else f'results_b22_{shard}.jsonl')
    if smoke:
        out = os.path.join(HERE, 'SMOKE_results_b22.jsonl')
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
    print(f'B-22 shard {shard}/{nsh}: 혼잡 에피소드 {len(eps)}개 · 팔 {G.ARM_ORDER} · '
          f'시드 {seeds} ⟹ {len(eps) * len(G.ARM_ORDER) * len(seeds)} 런', flush=True)
    print(f'  문턱(n={G.N_EP}): NOMINAL {nominal} · CONFIRM {confirm} · '
          f'NOT_EARN {not_earn} · 가족단위 α={G.FAMILY_ALPHA}', flush=True)
    print(f'  배포 상수 {G.deploy_constants()} · git {GIT_HEAD}', flush=True)
    with open(out, 'a') as fh:
        for ep in eps:
            for s in seeds:
                bank = None
                for arm in G.ARM_ORDER:
                    if (arm, ep['episode'], s) in done:
                        continue
                    try:
                        if bank is None:
                            t0 = time.time()
                            bank = prepare_atoms(ep, s)
                            print(f"  [atoms] {ep['episode'][:34]:34s} s{s} "
                                  + ' · '.join(f'{k} raw {len(v[0]["kx"])} pool {v[1]["n_pool"]}'
                                               for k, v in bank.items())
                                  + f' ({time.time() - t0:.1f}s)', flush=True)
                        row = run_one(ep, arm, s, bank)
                    except Exception as e:                   # noqa: BLE001
                        import traceback
                        row = dict(arm=arm, episode=ep['episode'], est_seed=s,
                                   error=repr(e), tb=traceback.format_exc()[-600:])
                    fh.write(json.dumps(row) + '\n')
                    fh.flush()
                    if 'error' in row:
                        print(f"{arm:11s} {ep['episode'][:34]:34s} s{s} {row['error'][:70]}",
                              flush=True)
                    else:
                        print(f"{arm:11s} {ep['episode'][:34]:34s} s{s} "
                              f"nRMSE={row['avg_nRMSE']:.4f} zero={row['zero_nRMSE']:.4f} "
                              f"route={row['auto_selected']} k={row['ensemble_k']} "
                              f"nsel={row['auto_n_selected']}/{row['n_pool_expected']} "
                              f"N={row['n_gaussians']} vp={row['varpro_fired']} "
                              f"prop={row['propose_calls']} "
                              f"it={row['init_loss_n']} {row['wall_s']:.1f}s", flush=True)
    if smoke:
        print('SMOKE=1 ⟹ 1셀에서 중단. ★결과 파일을 지울 것: ' + out, flush=True)


if __name__ == '__main__':
    main()
