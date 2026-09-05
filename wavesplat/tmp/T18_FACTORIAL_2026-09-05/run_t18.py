"""`T18-FACTORIAL` 러너 — ⛔`gates_t18.py` · `analyze_t18.py` **다음에** 쓰였다.

★**행 스키마는 이 파일이 정하지 않는다** — `analyze_t18.py` 독스트링이 정본이고 러너가 따른다.
★**팔 정의는 `gates_t18.arm_config()` 단일출처**다 — 이 파일에 설정 리터럴이 **0개**다.
  (⛔`{}` 로 팔을 정의하지 않는다: `gates_b7.py:46` 이 그렇게 **당일 철회된 값**으로 돌았다.)

## 개입은 두 키뿐이다
배포 진입점 `wavesplat.run.run_estimation` 을 부르고,
  · `config['learned_atoms']` — 원자 **출처**(확산 사전 / v2 빔포밍 개수일치)
  · `config['k_snap_dk']`     — 초기화 **끝**의 파수 격자화 (`None` = 연속)
나머지는 배포 그대로다. ⛔`wavesplat/**` 는 한 바이트도 고치지 않는다.

## 되읽기 규율 (⛔러너 리터럴을 분석기 리터럴과 대조하면 실패할 수 없는 게이트가 된다)
· 팔 정체 = `upd.arm_provenance()` **전체** (`G-PROV`)
· 배포 상수 9종 = `upd.config` **객체 되읽기** (`G-DEPLOY`)
· ★용량 = `resolve_pool_cap` / `resolve_growth_ceiling` / `upd.max_n` **셋 다** (`G-MAXN`)
· 스냅 계수기 = `_snap_k_to_grid` 가 **돌려준 record** (`run.py` 요약이 버리는 `shift_max`·
  `n_atoms` 까지 잡으려고 그 메서드를 관측 훅으로 감쌌다 — 반환값은 그대로 돌려준다)
· ★성장 = **분기 진입**을 센다. ⛔호출이 아니다 — `sequential_update.py:1675-1681` 이
  축자로 요구했다(*"A per-call counter does NOT detect that … Count the BRANCH."*).
  구현: `growth_ceiling` 을 **읽기 관찰 property** 로 바꾼다(값 불변). 그 속성을 읽는 지점이
  `:1697`(분기 조건)과 `:1734`(n_add) **둘뿐**이므로, 한 호출 안의 읽기 로그가
  「분기에 들어갔는가 / 잔차 문턱을 넘었는가」를 **정확히** 준다.

⛔메모리 캡 밖에서 돌리지 마라:
  `systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 -- python3 -u run_t18.py smoke`

사용:
    python3 run_t18.py freeze                 # 표본 + FROZEN*.txt (결과 0행에서)
    python3 run_t18.py selftest               # 게이트 자가진단만
    python3 run_t18.py smoke                  # 앞 2 에피소드 × 5팔 × 3시드 = 30런
    python3 run_t18.py primary [shard n]      # 5팔 × 48 × 3 = 720런  (⛔4샤드까지)
    python3 run_t18.py secc    [shard n]      # 4팔 × 16 × 3 = 192런
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import hashlib                                                     # noqa: E402
import json                                                        # noqa: E402
import subprocess                                                  # noqa: E402
import sys                                                         # noqa: E402
import time                                                        # noqa: E402
import traceback                                                   # noqa: E402
import warnings                                                    # noqa: E402

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gates_t18 as G                                              # noqa: E402
import analyze_t18 as A18                                          # noqa: E402

os.chdir(G.ROOT)
import torch                                                       # noqa: E402
import h5py                                                        # noqa: E402

ATOMS_DIR = os.path.join(HERE, 'atoms')
SAMPLE = os.path.join(HERE, 'sample_t18.json')
FROZEN = os.path.join(HERE, 'FROZEN_md5.txt')
FROZEN_A = os.path.join(HERE, 'FROZEN_md5_ANALYSIS.txt')
ANALYSIS_SIDE = A18.ANALYSIS_SIDE          # ★단일출처 — ⛔러너가 다시 적지 않는다

# ── 동결 대상 (사전등록 §9.1) — ★키는 **실제 상대경로**다 ────────────────────
#   ⛔`T18/…` 같은 **라벨을 쓰지 마라**: `check_batch_artifacts._resolve()` 가
#     「배치 → 부모(tmp) → 저장소」 순으로 푼다(`T15` 가 라벨을 써서 4키가 안 풀렸다).
BATCH_FILES = ('gates_t18.py', 'analyze_t18.py', 'run_t18.py', 'sample_t18.json',
               'PREREG.md', 'AMENDMENT_1_PREDATA.md', 'AMENDMENT_2_PREDATA.md')
SIBLING_FILES = ('B22_PRIOR_ABLATION_2026-08-27/gates_b22.py',
                 'B22_PRIOR_ABLATION_2026-08-27/run_b22.py',
                 'B16_KGRID_2026-08-24/gates_b16.py',
                 'B19_KGRID_SPARSE_2026-08-25/gates_b19.py',
                 'B12_HOS_EXTEND_2026-08-23/gates_b12.py',
                 'B11_HOS_BASELINE_2026-08-23/gates_b11.py',
                 'W15_DEPLOY_VERIFY_2026-08-22/gates_w15.py')
IMPORTS = ('wavesplat/run.py',
           'wavesplat/optimizer/sequential_update.py',
           'wavesplat/optimizer/learned_init.py',
           'wavesplat/optimizer/initializer.py',
           'wavesplat/optimizer/varpro.py',
           'wavesplat/optimizer/coarse_to_fine.py',
           'wavesplat/model/spectral_gaussian.py',
           'wavesplat/model/dispersion.py',
           'wavesplat/evaluation/metrics.py',
           'wavesplat/data_loader.py',
           'wavesim/data_writer.py',
           # ★깊이-2 전이 임포트 (`check_prereg C7t`) — ⛔*"값을 안 움직인다"* 를 주장하는 대신
           #   **동결한다**. 동결은 공짜이고, `T11 A2 §7` 은 `varpro` 를 그렇게 놓쳤다.
           'wavesplat/optimizer/bank_mmse.py',
           'wavesplat/model/tnse.py',
           'wavesplat/model/stokes2.py',
           'wavesplat/model/neural_corrector.py',
           'scripts/d3_beamform_cond_diffusion.py')
CKPT_REL = 'results/direction3/beamform_cond_diffusion_mixed_v1.pt'


def _md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def _sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for c in iter(lambda: fh.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def freeze_map():
    """★동결 지문 — 측정면. ⛔`FROZEN*.txt` 자신은 넣지 않는다(자기 지문이 자기를 바꾼다)."""
    out = {}
    for f in BATCH_FILES:
        if f in ANALYSIS_SIDE:
            continue
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            out[f] = _md5(p)
    for f in SIBLING_FILES:
        p = os.path.join(os.path.dirname(HERE), f)
        if os.path.exists(p):
            out[f] = _md5(p)
    for f in IMPORTS:
        p = os.path.join(G.ROOT, f)
        if os.path.exists(p):
            out[f] = _md5(p)
    return out


def analysis_map():
    return {f: _md5(os.path.join(HERE, f)) for f in ANALYSIS_SIDE
            if os.path.exists(os.path.join(HERE, f))}


def code_digest(live):
    return hashlib.md5(json.dumps(live, sort_keys=True).encode()).hexdigest()


def write_sample():
    """★표본을 **동결**한다 — `gates_b22.episodes()` 를 실물 해소하고 `mesh_verdict` 를 붙인다."""
    eps = G.B22.episodes()
    out = []
    for e in eps:
        with h5py.File(e['path'], 'r') as h:
            v = h['config'].attrs.get(G.B22.MESH_ATTR)
        out.append(dict(episode=e['episode'],
                        path=os.path.relpath(e['path'], G.ROOT), bf=e['bf'],
                        ncomp=e['ncomp'], gen_seed=e['gen_seed'],
                        mesh_verdict=v.decode() if isinstance(v, bytes) else str(v)))
    json.dump(dict(n=len(out), episodes=out,
                   mesh_census=G.MESH_EXPECTED,
                   est_seeds=list(G.EST_SEEDS), secc_seeds=list(G.SECC_SEEDS),
                   arms=G.ARM_ORDER),
              open(SAMPLE, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'✅표본 동결 {len(out)} 에피소드 → {SAMPLE}')


def write_frozen():
    """`FROZEN_md5.txt`(측정면) · `FROZEN_md5_ANALYSIS.txt`(분석면) — **결과 0행에서** 찍는다."""
    with open(FROZEN, 'w', encoding='utf-8') as fh:
        fh.write('# T18-FACTORIAL 측정면 동결 (결과 0행 시점)\n')
        for k, v in sorted(freeze_map().items()):
            fh.write(f'{v}  {k}\n')
        p = os.path.join(G.ROOT, CKPT_REL)
        if os.path.exists(p):
            # ⛔해시가 **첫 칸**이어야 한다(`check_prereg C7s` 가 64-hex 를 거기서 찾는다).
            fh.write(f'{_sha256(p)}  sha256:{CKPT_REL}\n')
    with open(FROZEN_A, 'w', encoding='utf-8') as fh:
        fh.write('# T18-FACTORIAL 분석면 동결 (측정면과 분리 — T11 A3 §2)\n')
        for k, v in sorted(analysis_map().items()):
            fh.write(f'{v}  {k}\n')
    print(f'✅동결 {FROZEN} · {FROZEN_A}')


# ─────────────────────────────────────────────────────────────────────────────
# 관측 훅 — ⛔거동을 바꾸지 않는다(전부 진짜 함수의 반환값을 **그대로** 돌려준다)
# ─────────────────────────────────────────────────────────────────────────────
_CAPTURED = []
_PROPOSE = [0]
_SNAP = [None]
_GROW = dict(calls=0, branch=0, passed=0, atoms=0)
_GC_READS = []
_OBS_ON = [False]


def _install_capture():
    import wavesplat.run as WR
    if getattr(WR.make_updater, '_t18', False):
        return
    real = WR.make_updater

    def capturing(config):
        u = real(config)
        _CAPTURED.append(u)
        return u

    capturing._t18 = True
    capturing._t18_real = real
    WR.make_updater = capturing


def _install_propose_counter():
    import wavesplat.optimizer.learned_init as LI
    if getattr(LI.propose_atoms, '_t18', False):
        return
    real = LI.propose_atoms

    def counting(*a, **kw):
        _PROPOSE[0] += 1
        return real(*a, **kw)

    counting._t18 = True
    counting._t18_real = real
    LI.propose_atoms = counting


def _install_snap_observer():
    """`_snap_k_to_grid` 의 **record 전체**를 잡는다(`run.py` 요약이 `shift_max` 를 버린다)."""
    from wavesplat.optimizer.sequential_update import SequentialUpdater as SU
    if getattr(SU._snap_k_to_grid, '_t18', False):
        return
    real = SU._snap_k_to_grid

    def observing(self, frames, verbose=False):
        rec = real(self, frames, verbose)
        _SNAP[0] = rec
        return rec

    observing._t18 = True
    observing._t18_real = real
    SU._snap_k_to_grid = observing


class _GrowthCeilingProbe:
    """★`growth_ceiling` **읽기 관찰자** — 값은 그대로 돌려주고 읽은 시점만 기록한다.

    ⛔이것이 「호출이 아니라 분기를 센다」의 구현이다. 그 속성을 읽는 지점은
    `sequential_update.py:1697`(분기 조건)과 `:1734`(n_add) **둘뿐**이므로
    한 호출 안의 읽기 로그가 분기 진입 여부를 **정확히** 준다.
    """
    _SLOT = '_t18_growth_ceiling'

    def __get__(self, obj, owner=None):
        if obj is None:
            return self
        v = obj.__dict__[self._SLOT]
        if _OBS_ON[0]:
            model = getattr(obj, 'model', None)
            _GC_READS.append((getattr(model, 'n_gaussians', None), v))
        return v

    def __set__(self, obj, value):
        obj.__dict__[self._SLOT] = value


def _install_growth_observer():
    from wavesplat.optimizer.sequential_update import SequentialUpdater as SU
    if getattr(SU, '_t18_growth_probe', False):
        return
    SU.growth_ceiling = _GrowthCeilingProbe()
    SU._t18_growth_probe = True
    real = SU._adaptive_prune_grow

    def counting(self, frames, x_range, y_range, verbose=False):
        if not _OBS_ON[0]:
            return real(self, frames, x_range, y_range, verbose)
        _GROW['calls'] += 1
        start = len(_GC_READS)
        try:
            return real(self, frames, x_range, y_range, verbose)
        finally:
            reads = _GC_READS[start:]
            entered = [1 for n, c in reads if n is not None and n < c]
            _GROW['branch'] += int(bool(entered))
            _GROW['passed'] += int(len(entered) >= 2)
            del _GC_READS[start:]

    counting._t18 = True
    counting._t18_real = real
    SU._adaptive_prune_grow = counting


def _install_add_counter():
    from wavesplat.model.spectral_gaussian import SpectralGaussianModel as M
    if getattr(M.add_gaussians, '_t18', False):
        return
    real = M.add_gaussians

    def counting(self, n_new, *a, **kw):
        if _OBS_ON[0]:
            _GROW['atoms'] += int(n_new)
        return real(self, n_new, *a, **kw)

    counting._t18 = True
    counting._t18_real = real
    M.add_gaussians = counting


def install_hooks():
    _install_capture()
    _install_propose_counter()
    _install_snap_observer()
    _install_growth_observer()
    _install_add_counter()


# ─────────────────────────────────────────────────────────────────────────────
# 원자 준비 (캐시) — ⛔`B-22` 의 캐시 디렉터리를 건드리지 않는다
# ─────────────────────────────────────────────────────────────────────────────
def _cache_path(ep, source, seed):
    return os.path.join(ATOMS_DIR, f"{ep['episode']}__{source}__s{seed}.json")


def prepare_atoms(ep, est_seed):
    """이 (에피소드, 시드) 의 원자 셋. 반환 `{source: (atoms, meta)}`.

    ★`analytic_n` 은 v2 제안의 **그리디 접두 n_L 개**(사전등록 §1.2 개수 일치) —
    `A-C` 와 `A-Q` 가 **같은 배열**을 받는다는 것이 `G-PAIRSRC` 의 내용이다.
    """
    os.makedirs(ATOMS_DIR, exist_ok=True)
    out = {}
    p = _cache_path(ep, 'learned', est_seed)
    if os.path.exists(p):
        d = json.load(open(p, encoding='utf-8'))
    else:
        atoms, pad = G.B22.learned_raw_atoms(ep['path'], est_seed)
        d = dict(atoms=atoms, n_pad_dropped=pad)
        json.dump(d, open(p, 'w', encoding='utf-8'))
    n_l = G.B22.pooled_count(d['atoms'], G.arm_config('L-C')['learned_energy_thresh'])
    out['learned'] = (d['atoms'], dict(n_pad_dropped=d['n_pad_dropped'], n_pool=n_l))

    p = os.path.join(ATOMS_DIR, f"{ep['episode']}__analytic__s{est_seed}.json")
    if os.path.exists(p):
        d2 = json.load(open(p, encoding='utf-8'))
    else:
        atoms, pad = G.B22.analytic_raw_atoms(ep['path'], est_seed,
                                              G.arm_config('A-C')['max_n'])
        d2 = dict(atoms=atoms, n_pad_dropped=pad)
        json.dump(d2, open(p, 'w', encoding='utf-8'))
    an = G.B22.prefix_atoms(d2['atoms'], n_l)
    out['analytic_n'] = (an, dict(n_pad_dropped=d2['n_pad_dropped'],
                                  n_pool=G.B22.pooled_count(
                                      an, G.arm_config('A-C')['learned_energy_thresh'])))
    return out


def _env():
    try:
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=G.ROOT,
                                       text=True).strip()
    except Exception:                                              # noqa: BLE001
        head = None
    md5 = {p: _md5(os.path.join(G.ROOT, p)) for p in IMPORTS
           if os.path.exists(os.path.join(G.ROOT, p))}
    ck = os.path.join(G.ROOT, CKPT_REL)
    return head, md5, (_sha256(ck) if os.path.exists(ck) else None)


GIT_HEAD, IMPORT_MD5, CKPT_SHA = _env()


def run_one(ep, arm, est_seed, layer, bank, mesh, digest):
    from wavesplat.run import run_estimation
    from wavesplat.optimizer.learned_init import resolve_pool_cap
    from wavesplat.optimizer.sequential_update import resolve_growth_ceiling
    install_hooks()
    source = G.ARMS[arm]['atoms']
    atoms, ameta = (None, {}) if source is None else bank[source]
    cfg = G.arm_config(arm, atoms)                 # ★팔 정의는 gates_t18 단일출처
    cfg['seed'] = est_seed
    _CAPTURED.clear()
    _PROPOSE[0] = 0            # ★러너 자신의 `prepare_atoms` 호출을 세지 않는다
    _SNAP[0] = None
    for k in _GROW:
        _GROW[k] = 0
    del _GC_READS[:]
    _OBS_ON[0] = True
    t0 = time.time()
    try:
        r = run_estimation(ep['path'], config=cfg, verbose=False)
    finally:
        _OBS_ON[0] = False
    wall = time.time() - t0
    upd = _CAPTURED[-1] if _CAPTURED else None
    ucfg = getattr(upd, 'config', None) or {}
    snap = _SNAP[0] if _SNAP[0] is not None else {}
    z = G.B22.zero_nrmse(ep['path'])               # ★매 행 새로 계산(캐시 없음)
    return dict(
        # ── 식별 ────────────────────────────────────────────────────────────
        episode=ep['episode'], bf=ep['bf'], ncomp=ep['ncomp'], gen_seed=ep['gen_seed'],
        arm=arm, est_seed=est_seed, layer=layer,
        # ── 채점 · 창 ───────────────────────────────────────────────────────
        avg_nRMSE=float(r['avg_nRMSE']),
        n_frames_scored=r.get('tracking_frames'),
        scored_t_first=r.get('scored_t_first'), scored_t_last=r.get('scored_t_last'),
        episode_total_frames=r.get('episode_total_frames'),
        # ⛔`**z` 로 흘리지 않는다 — 정적 검사기가 「러너가 안 쓰는 키」로 읽는다(`C17`).
        #   출처는 여전히 `gates_b19.zero_nrmse` 하나다(⛔상수 1.0 아님).
        zero_nRMSE=z['zero_nRMSE'], zero_n_frames=z['zero_n_frames'],
        zero_t_first=z['zero_t_first'], zero_t_last=z['zero_t_last'],
        Hs_true=z['Hs_true'], Tp_true=z['Tp_true'],
        # ── ★개입 ①: 파수 양자화 (⛔`cfg` 되읽기가 아니라 **record 되읽기**) ─
        k_snap_dk=r.get('k_snap_dk'), k_snap_fired=bool(r.get('k_snap_fired')),
        k_snap_cells=r.get('k_snap_cells'),
        k_snap_shift_median=r.get('k_snap_shift_median'),
        k_snap_shift_max=snap.get('shift_max'), k_snap_n_atoms=snap.get('n_atoms'),
        # ── ★개입 ②: 원자 출처 ──────────────────────────────────────────────
        atoms_source=source, atoms_md5=G.B22.atoms_md5(atoms),
        n_atoms_injected=(None if atoms is None else len(atoms['kx'])),
        n_pad_dropped=ameta.get('n_pad_dropped'), n_pool_expected=ameta.get('n_pool'),
        propose_calls=_PROPOSE[0],
        # ── 팔 정체 (⛔의도가 아니라 되읽기) ─────────────────────────────────
        arm_provenance=(upd.arm_provenance() if upd is not None else None),
        auto_selected=r.get('auto_selected'), auto_n_selected=r.get('auto_n_selected'),
        auto_real_gate=r.get('auto_real_gate'), ensemble_k=r.get('ensemble_k'),
        updater_class=type(upd).__name__ if upd is not None else None,
        effective=G.B22.effective_from_updater(upd, n_captured=len(_CAPTURED)),
        deploy_readback={k: ucfg.get(k) for k in G.DEPLOY_EXPECTED},
        # ── ★용량 (2026-09-05 분리 이후 **역할 둘**을 따로 되읽는다) ────────
        pool_cap_readback=(resolve_pool_cap(ucfg) if upd is not None else None),
        growth_ceiling_readback=(resolve_growth_ceiling(ucfg) if upd is not None else None),
        max_n_alias=(getattr(upd, 'max_n', None) if upd is not None else None),
        # ── ★성장 (⛔호출이 아니라 **분기**) ────────────────────────────────
        grow_calls=_GROW['calls'], grow_branch_entries=_GROW['branch'],
        grow_threshold_pass=_GROW['passed'],
        # ⚠`n_grow_atoms` = `add_gaussians` **총합**이다(성장 분기 밖 호출도 포함).
        #   분기 자체의 발화는 위의 `grow_branch_entries` 가 센다.
        n_grow_atoms=_GROW['atoms'],
        # ── 기타 계수기 ─────────────────────────────────────────────────────
        varpro_fired=bool(r.get('varpro_fired')), varpro_steps=r.get('varpro_steps'),
        varpro_obj_before=r.get('varpro_obj_before'),
        varpro_obj_after=r.get('varpro_obj_after'),
        n_gaussians=r.get('n_gaussians'),
        depth=(getattr(upd, 'depth', None) if upd is not None else None),
        mesh_verdict=mesh, wall_s=wall,
        git_head=GIT_HEAD, import_md5=IMPORT_MD5, ckpt_sha256=CKPT_SHA,
        code_md5_row=digest)


def _mesh(ep):
    with h5py.File(ep['path'], 'r') as h:
        v = h['config'].attrs.get(G.B22.MESH_ATTR)
    return v.decode() if isinstance(v, bytes) else str(v)


def main(mode, shard=0, nsh=1):
    print('★런 전 게이트 자가진단 (등록값 재생산 · blocked_by · 사다리) ─────────', flush=True)
    ok, _ = G.selftest(verbose=True)
    if not ok:
        print('⛔게이트 자가진단 실패 ⟹ 한 런도 돌리지 않는다.', flush=True)
        return 1
    print('──────────────────────────────────────────────────────────\n', flush=True)

    all_eps = G.B22.episodes()
    if mode == 'smoke':
        eps, arms, seeds, layer = all_eps[:G.SMOKE_N_EP], G.ARM_ORDER, G.EST_SEEDS, \
            G.LAYER_PRIMARY
        out = os.path.join(HERE, 'results_t18_SMOKE.jsonl')
        meta_o = os.path.join(HERE, 'meta_t18_SMOKE.json')
    elif mode == 'primary':
        eps, arms, seeds, layer = all_eps, G.ARM_ORDER, G.EST_SEEDS, G.LAYER_PRIMARY
        out = os.path.join(HERE, 'results_t18.jsonl' if nsh == 1
                           else f'results_t18_{shard}.jsonl')
        meta_o = os.path.join(HERE, 'meta_t18.json' if nsh == 1
                              else f'meta_t18_{shard}.json')
    elif mode == 'secc':
        eps, arms, seeds, layer = (all_eps[:G.SECC_N_EP], G.CELL_ARMS, G.SECC_SEEDS,
                                   G.LAYER_SECC)
        out = os.path.join(HERE, 'results_t18_secc.jsonl' if nsh == 1
                           else f'results_t18_secc_{shard}.jsonl')
        meta_o = os.path.join(HERE, 'meta_t18_secc.json' if nsh == 1
                              else f'meta_t18_secc_{shard}.json')
    else:
        print(f'⛔알 수 없는 모드 {mode!r}')
        return 2
    eps = [e for i, e in enumerate(eps) if i % nsh == shard]

    md5_start = freeze_map()
    digest = code_digest(md5_start)
    done = set()
    if os.path.exists(out):
        for line in open(out, encoding='utf-8'):
            try:
                rr = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'error' not in rr:
                done.add((rr['arm'], rr['episode'], rr['est_seed']))
    torch.set_num_threads(1)
    n_todo = len(eps) * len(arms) * len(seeds) - len(done)
    cost = G.projected_cost()
    print(f'T18 [{mode}] shard {shard}/{nsh}: 에피소드 {len(eps)} · 팔 {list(arms)} · '
          f'시드 {list(seeds)} ⟹ {len(eps) * len(arms) * len(seeds)} 런 (남은 {n_todo})',
          flush=True)
    print(f"  문턱 축 I {G.THRESH_I}/{G.N_EP} · SEC {G.THRESH_SEC} · MDE q80 "
          f"{G.MDE['I_q80']} · 배포 상수 {G.DEPLOY_EXPECTED}", flush=True)
    print(f"  단가(동결) {cost['unit_costs']} · git {GIT_HEAD}", flush=True)

    t_start, n_done = time.time(), 0
    with open(out, 'a', encoding='utf-8') as fh:
        for ep in eps:
            mesh = _mesh(ep)
            for s in seeds:
                bank = None
                for arm in arms:
                    if (arm, ep['episode'], s) in done:
                        continue
                    try:
                        if bank is None:
                            ta = time.time()
                            bank = prepare_atoms(ep, s)
                            print(f"  [atoms] {ep['episode'][:34]:34s} s{s} " +
                                  ' · '.join(f'{k} raw {len(v[0]["kx"])} pool '
                                             f'{v[1]["n_pool"]}' for k, v in bank.items())
                                  + f' ({time.time() - ta:.1f}s)', flush=True)
                        row = run_one(ep, arm, s, layer, bank, mesh, digest)
                    except Exception as e:                         # noqa: BLE001
                        row = dict(arm=arm, episode=ep['episode'], est_seed=s, layer=layer,
                                   error=repr(e), tb=traceback.format_exc()[-700:])
                    fh.write(json.dumps(row, ensure_ascii=False) + '\n')
                    fh.flush()
                    n_done += 1
                    if 'error' in row:
                        print(f"{arm:5s} {ep['episode'][:34]:34s} s{s} "
                              f"⛔{row['error'][:80]}", flush=True)
                    else:
                        eta = (time.time() - t_start) / n_done * (n_todo - n_done) / 60.0
                        print(f"{arm:5s} {ep['episode'][:34]:34s} s{s} "
                              f"nRMSE={row['avg_nRMSE']:.4f} zero={row['zero_nRMSE']:.4f} "
                              f"snap={row['k_snap_fired']}/{row['k_snap_dk']} "
                              f"nsel={row['auto_n_selected']}/{row['n_pool_expected']} "
                              f"N={row['n_gaussians']} grow={row['grow_branch_entries']}"
                              f"/{row['grow_calls']} vp={row['varpro_fired']} "
                              f"prop={row['propose_calls']} {row['wall_s']:.1f}s "
                              f"(남은 {eta:.0f}분)", flush=True)
    md5_end = freeze_map()
    json.dump(dict(mode=mode, shard=shard, nshards=nsh, n_new_rows=n_done,
                   arms=list(arms), seeds=list(seeds), layer=layer,
                   # ⛔`threads` 와 `threads_readback` 에 **같은 식**을 적으면 하나를 둘로 센 것이다
                   threads_env=os.environ.get('OMP_NUM_THREADS'),
                   threads_readback=int(torch.get_num_threads()),
                   blas_env={v: os.environ.get(v) for v in
                             ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
                   cuda_available=bool(torch.cuda.is_available()),
                   code_md5=md5_end, code_md5_start=md5_start, code_md5_row=digest,
                   md5_stable=(md5_start == md5_end), ckpt_sha256=CKPT_SHA,
                   git_head=GIT_HEAD, projected_cost=cost,
                   t_start=t_start, t_end=time.time(),
                   elapsed_min=round((time.time() - t_start) / 60.0, 2)),
              open(meta_o, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'✅{n_done} 행 · md5 안정 {md5_start == md5_end} · '
          f'{(time.time() - t_start) / 60.0:.1f} 분 → {out}', flush=True)
    return 0


if __name__ == '__main__':
    a = sys.argv[1] if len(sys.argv) > 1 else 'smoke'
    if a == 'freeze':
        write_sample()
        write_frozen()
    elif a == 'selftest':
        _ok, _ = G.selftest(verbose=True)
        sys.exit(0 if _ok else 1)
    else:
        _sh = int(sys.argv[2]) if len(sys.argv) > 3 else 0
        _n = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        assert _n <= 4, '⛔4샤드까지 (8샤드 사고 선례)'
        sys.exit(main(a, _sh, _n))
