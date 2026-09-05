"""`T16-BANDFIX` 러너 — ⛔`analyze_t16.py` · `gates_t16.py` **다음에** 쓰였다(`PREREG §12-1`).

★**행 스키마는 이 파일이 정하지 않는다** — `analyze_t16.py` 독스트링이 정본이고 러너가 따른다.
  ⚠그 스키마는 `PREREG §6.5` 보다 **넷 넓다**(`t_origin`·`seed_applied`·`seed_rng_md5`·
  `equivar_rule`). 사유 = `AMENDMENT_1_PREDATA.md` §4·§6·§3.

★관측·프레임·채점은 `T11 run_t11` 을 **임포트해 비트 재사용**한다(⛔손복사 금지 —
  `G-ANCHOR-T11` 이 성립하려면 필수).

★**되읽기 규율** (⛔러너 리터럴을 분석기 리터럴과 대조하면 실패할 수 없는 게이트가 된다)
  · `(klim, n)` 은 `bank_mmse._beamform_top` 을 **감싸 소비자가 받은 값**을 잡는다(`G-KLIM1`).
  · `k̂₁`·`k̂₂` 는 그 후크가 **돌려받은 첨두**에서 다시 만든다(⛔`klim₂/3.5` 로 역산하지 않는다).
  · 시드는 **적용된 RNG 상태의 지문**으로 적는다(`G-ENV`).
  · 팔 정체는 `upd.arm_provenance()` **전체**(`G-ARM` — 의도가 아니라 되읽기).
  · 배포 상수 9종은 `cfg` **되읽기**(`G-DEPLOY`).

⛔**팔을 `{}` 로 정의하지 않는다** — `BASE = dict(DEFAULT_CONFIG)` 를 **파일·팔마다 새로** 만든다
  (`gates_b7.py:46` 선례 · `T11 v1 BLOCKER-2` 원자 캐시 누출).
⛔`(x or 기본값)` 을 쓰지 않는다 — `0.0` 은 falsy 다(`T14 AMENDMENT_1 §1`). `is None` 으로 가른다.
⛔`SEED16` 은 `gates_t16` 이 들고 있다(20261300; 초안 20261100 은 `T15` 와 60/60 충돌이었다).
⛔`scale_rules_t16` 의 레일을 러너가 재구현하지 않는다 — `SR16.clip_s` 를 부른다.
⛔cwd 를 되돌리지 마라 — `d3_beamform_cond_diffusion.py:25` 의 chdir 은 의존성이다
  (`T13 AMENDMENT_1 §1`).
⛔메모리 캡 밖에서 돌리지 마라:
  `systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 -- python3 -u run_t16.py smoke`
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import hashlib, json, subprocess, sys, tempfile, time, zipfile      # noqa: E402

import numpy as np                                                 # noqa: E402
import torch                                                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_t16 as G                                              # noqa: E402
import scale_rules_t16 as SR16                                     # noqa: E402
# ★`ANALYSIS_SIDE` 단일출처 — ⛔러너가 그 목록을 손으로 다시 적으면 `code_md5_row` 의
#   측정면/분석면 분리가 두 파일에 중복 구현된다(저장소가 그 부류로 여러 번 대가를 치렀다).
#   순환 없음: `analyze_t16` 은 `run_t16` 을 **`main()` 안에서** 임포트한다.
import analyze_t16 as A16                                          # noqa: E402
import run_t11 as R11                                              # noqa: E402
from wavesplat.optimizer import bank_mmse as BM                    # noqa: E402
from wavesplat.optimizer import learned_init as LI                 # noqa: E402
from wavesplat.optimizer.sequential_update import make_updater     # noqa: E402
from wavesplat.model.dispersion import G as GRAV                   # noqa: E402
from wavesplat.model.tnse import render_wave_field                 # noqa: E402
from wavesplat.run import DEFAULT_CONFIG                           # noqa: E402

OUT = os.path.join(HERE, 'results_t16.jsonl')
META = os.path.join(HERE, 'meta_t16.json')
ANALYSIS_SIDE = A16.ANALYSIS_SIDE

# ── 동결 대상 (`PREREG §9.1`) ───────────────────────────────────────────────
# ★키는 **실제 상대경로**다 — `tmp/_audit/check_batch_artifacts.py:_resolve()` 가
#   「배치 → 부모(tmp) → 저장소」 순으로 푼다. ⛔`T14/…` 같은 **라벨을 쓰지 마라**
#   (`T15` 가 라벨을 써서 25키 중 4키가 풀리지 않았다 — `AMENDMENT_1 §14`).
BATCH_FILES = ('gates_t16.py', 'analyze_t16.py', 'run_t16.py', 'scale_rules_t16.py',
               'blind_worker_t16.py', 'sample_t16.json',
               'PREREG.md', 'REACHABILITY.md', 'AMENDMENT_1_PREDATA.md')
IMPORTS = ('wavesplat/optimizer/sequential_update.py',
           'wavesplat/optimizer/learned_init.py',
           'wavesplat/optimizer/bank_mmse.py',
           'wavesplat/optimizer/varpro.py',
           'wavesplat/optimizer/initializer.py',
           'wavesplat/model/spectral_gaussian.py',
           'wavesplat/model/tnse.py',
           'wavesplat/model/dispersion.py',
           'wavesplat/run.py',
           'scripts/d3_beamform_net.py',
           'scripts/d3_beamform_cond_diffusion.py',
           'scripts/d3_setdiff_prior.py',
           # ★깊이-2 전이 임포트 (`check_prereg C7t`) — ⛔*"값을 안 움직인다"* 를 주장하는 대신
           #   **동결한다**. 동결은 공짜이고, `T11 A2 §7` 은 `varpro` 를 그렇게 놓쳤다.
           'wavesplat/optimizer/coarse_to_fine.py',
           'wavesplat/optimizer/relax_estimator.py',
           'wavesplat/evaluation/metrics.py',
           'wavesplat/data_loader.py',
           'wavesim/wave_generator.py')
T14_DIR = os.path.basename(G.T14)          # 'T14_AUTOSCALE_2026-09-04'
T11_DIR = os.path.basename(G.T11)          # 'T11_EXTACC_2026-09-03'
T14_REFS = ('scale_rules_t14.py',)
T11_REFS = ('gates_t11.py', 'run_t11.py', 'sample_t11.json', 'results_t11_SMOKE.jsonl')
CKPT = 'results/direction3/beamform_cond_diffusion_mixed_v1.pt'
FROZEN = os.path.join(HERE, 'FROZEN_md5.txt')
FROZEN_A = os.path.join(HERE, 'FROZEN_md5_A.txt')


def _md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def _sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for c in iter(lambda: fh.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def freeze_map():
    """★동결 지문 — `G-ENV` 가 「지금 다시 계산 == meta == FROZEN」 3중으로 댄다.

    ⛔`FROZEN*.txt` 자신은 넣지 않는다(자기 지문이 자기를 바꾼다).
    """
    out = {}
    for f in BATCH_FILES:
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            out[f] = _md5(p)
    for f in IMPORTS:
        p = os.path.join(G.ROOT, f)
        if os.path.exists(p):
            out[f] = _md5(p)
    for f in T14_REFS:
        p = os.path.join(G.T14, f)
        if os.path.exists(p):
            out[f'{T14_DIR}/{f}'] = _md5(p)
    for f in T11_REFS:
        p = os.path.join(G.T11, f)
        if os.path.exists(p):
            out[f'{T11_DIR}/{f}'] = _md5(p)
    p = os.path.join(G.ROOT, CKPT)
    if os.path.exists(p):
        out['sha256:' + CKPT] = _sha256(p)
    return out


def code_digest(live):
    """★행에 실을 **측정면** 지문. ⛔분석기와 같은 식이어야 한다(그쪽이 대조한다)."""
    meas = {k: v for k, v in live.items() if k not in ANALYSIS_SIDE}
    return hashlib.md5(json.dumps(meas, sort_keys=True).encode()).hexdigest()


def write_frozen():
    """`FROZEN_md5.txt`(측정면) · `FROZEN_md5_A.txt`(분석면) — **결과 0행에서** 찍는다.

    ⛔§9.1 은 분석면을 `FROZEN_md5_A*.txt`(글롭)로 적었으나 분석기는 정확히
    `FROZEN_md5_A.txt` 를 읽는다 ⟹ 이름을 그것으로 확정했다(`AMENDMENT_1 §11`).
    """
    live = freeze_map()
    head = ('# T16-BANDFIX 동결 — **결과 0행 · 추정기 런 0회 시점**\n'
            '# 2026-09-05 · ⛔`run_t16.freeze_map()` 이 낸다. 손으로 고치지 마라.\n'
            '# ★키는 **실제 상대경로**다(배치 → tmp → 저장소 순으로 풀린다).\n')
    with open(FROZEN, 'w', encoding='utf-8') as fh:
        fh.write(head + '# 측정면 (measurement side)\n')
        for k in sorted(live):
            if k not in ANALYSIS_SIDE:
                fh.write(f'{live[k]}  {k}\n')
    with open(FROZEN_A, 'w', encoding='utf-8') as fh:
        fh.write(head + '# 분석면 (analysis side)\n')
        for k in sorted(ANALYSIS_SIDE):
            if k in live:
                fh.write(f'{live[k]}  {k}\n')
    n_m = sum(1 for k in live if k not in ANALYSIS_SIDE)
    print(f'✅동결 — 측정면 {n_m}키 · 분석면 {len(ANALYSIS_SIDE)}키 · '
          f'code_md5_row {code_digest(live)}')
    return live


def write_sample():
    """`sample_t16.json` = Tier-P 60 (⛔손목록이 아니라 `gates_t16.tiers()` 가 계산한다)."""
    tier_p, _ = G.tiers()
    json.dump(sorted(tier_p), open(G.SAMPLE_T16, 'w'), ensure_ascii=False, indent=1)
    print(f'✅표본 {len(tier_p)} → {os.path.basename(G.SAMPLE_T16)}')
    return tier_p


# ── 되읽기 후크 ─────────────────────────────────────────────────────────────
class BeamProbe:
    """★`bank_mmse._beamform_top` 을 감싸 **소비자가 실제로 받은** `(klim, n)` 과
    **돌려준 첨두**를 잡는다.

    ⛔러너가 자기 리터럴을 적는 것이 아니다 — 규칙이 무엇을 넘겼는지는 여기서만 알 수 있다.
    ⛔`__exit__` 에서 반드시 되돌린다(되돌리기 실패 = 프로세스 전역 오염).
    """

    def __enter__(self):
        self.calls, self.peaks = [], []
        self.device = str(BM._DEV)
        orig = BM._beamform_top

        def wrap(P, r, depth, klim=0.6, n=80, _o=orig, _s=self):
            out = _o(P, r, depth, klim=klim, n=n)
            _s.calls.append([float(klim), int(n)])
            _s.peaks.append((float(out[0]), float(out[1])))
            return out
        BM._beamform_top = wrap
        self._undo = lambda: setattr(BM, '_beamform_top', orig)
        return self

    def __exit__(self, *a):
        self._undo()
        return False

    def kp(self, i):
        """후크가 **돌려받은** i번째 첨두에서 `k̂` 를 다시 만든다(바닥 포함)."""
        if i >= len(self.peaks):
            return None
        bx, by = self.peaks[i]
        return max(float(np.hypot(bx, by)), SR16.FLOOR)


class Probe:
    """제안 단계 계측 — 개입이 아니라 **되읽기 계수기**(`T14` 축자)."""

    def __enter__(self):
        self.device = None
        self.n_gathered = None
        ol, og = LI._load_cond, LI._gather_init_obs

        def wl(c, d, _o=ol, _s=self):
            _s.device = str(d)
            return _o(c, d)

        def wg(fr, _o=og, _s=self):
            o = _o(fr)
            _s.n_gathered = None if o is None else int(len(o[0]))
            return o
        LI._load_cond, LI._gather_init_obs = wl, wg
        self._u = [lambda: setattr(LI, '_load_cond', ol),
                   lambda: setattr(LI, '_gather_init_obs', og)]
        return self

    def __exit__(self, *a):
        for f in self._u:
            f()
        return False


def blind_s(x, y, t, z, depth):
    """★별도 프로세스에서 세 규칙을 재실행 — 관측 5배열만 넘긴다(`G-S-BLIND`)."""
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        np.savez(f, x=x, y=y, t=t, z=z, depth=np.float64(depth))
        p = f.name
    try:
        r = subprocess.run([sys.executable, os.path.join(HERE, 'blind_worker_t16.py'), p],
                           capture_output=True, text=True, timeout=1800)
        if r.returncode != 0:
            raise RuntimeError(f'blind_worker 실패 rc={r.returncode}\n{r.stderr[-2000:]}')
        return json.loads(r.stdout.strip().splitlines()[-1])
    finally:
        os.unlink(p)


# ── 규칙 평가 (팔별) ────────────────────────────────────────────────────────
def eval_rule(arm, x, y, tt, zz, depth):
    """★한 팔의 규칙을 **본 1 + 등변 2** 평가하고 전부 되읽는다.

    ⛔`s_rule` 을 얻으려고 규칙을 **두 번** 부르지 않는다 — §8 이 「팔당 4 평가」로
    비용을 등록했다. 비클리핑 값을 한 번 얻고 레일은 `SR16.clip_s` 가 씌운다
    (`s_ap`/`s_band`/`s_bandr` 내부와 **비트 동일**한 식).
    """
    fn_raw = SR16.RULES_RAW[arm]
    is_bf = arm in G.BEAMFORM_ARMS
    with BeamProbe() as pb:
        s_raw = float(fn_raw(x, y, tt, zz, depth))
    with BeamProbe() as p5:
        s_c05 = float(fn_raw(x / 0.5, y / 0.5, tt / np.sqrt(0.5), zz, depth))
    with BeamProbe() as p20:
        s_c20 = float(fn_raw(x / 2.0, y / 2.0, tt / np.sqrt(2.0), zz, depth))
    s_rule = SR16.clip_s(s_raw)
    d = dict(s_rule=s_rule, s_raw=s_raw, s_raw_c05=s_c05, s_raw_c20=s_c20,
             s_clipped=bool(s_rule != s_raw),
             equivar_rule=f'{fn_raw.__module__}.{fn_raw.__qualname__}',
             beamform_calls=list(pb.calls),
             beamform_calls_n=len(pb.calls) + len(p5.calls) + len(p20.calls),
             beamform_device=pb.device,
             klim1_c05=(p5.calls[0][0] if p5.calls else None),
             klim1_c20=(p20.calls[0][0] if p20.calls else None),
             k1hat_raw_c05=p5.kp(0), k1hat_raw_c20=p20.kp(0),
             kp_hat=None, kp_hat_stage1=None, klim1=None, klim2=None, dk1=None,
             floor_margin=None, edge_margin=None, inner_margin=None, inner_ok=None,
             kd_min=None, tp_hat=None)
    if is_bf:
        k1, k2 = pb.kp(0), pb.kp(1)
        klim1, klim2 = pb.calls[0][0], pb.calls[1][0]
        dk1 = 2.0 * klim1 / (SR16.N1 - 1)          # ★후크의 klim 에서 유도(리터럴 아님)
        d.update(kp_hat=k2, kp_hat_stage1=k1, klim1=klim1, klim2=klim2, dk1=dk1,
                 floor_margin=min(k1, k2) / SR16.FLOOR,
                 edge_margin=k1 / klim1,
                 inner_margin=k1 / dk1,
                 inner_ok=bool(k2 <= SR16.ZOOM * k1 * (1.0 + 1e-12)),
                 kd_min=float(k2 * depth),
                 tp_hat=float(2.0 * np.pi
                              / np.sqrt(GRAV * k2 * np.tanh(k2 * depth))))
    return d


# ★규칙 열의 **완전한 선언** — ⛔`s_blind` 를 `rule[...] =` 로만 넣으면 정적 검사가
#   *"분석기가 읽는데 러너가 안 쓴다"* 로 읽는다(`check_prereg C17`). 여기에 이름을 둔다.
NULL_RULE = dict(s_blind=None,
                 s_rule=None, s_raw=None, s_raw_c05=None, s_raw_c20=None, s_clipped=None,
                 equivar_rule=None, beamform_calls=None, beamform_calls_n=None,
                 beamform_device=None, klim1_c05=None, klim1_c20=None,
                 k1hat_raw_c05=None, k1hat_raw_c20=None, kp_hat=None, kp_hat_stage1=None,
                 klim1=None, klim2=None, dk1=None, floor_margin=None, edge_margin=None,
                 inner_margin=None, inner_ok=None, kd_min=None, tp_hat=None)


# ── 추정 ────────────────────────────────────────────────────────────────────
def estimate(s, x, y, tt, z, gx, gy, tq, target, s4, seed):
    """재라벨 `x/s, y/s, t/√s` 로 추정하고 **두 역변환 경로를 다 계산**한다(`G-INVERT`).

    ⛔`BASE` 를 팔 사이에 재사용하지 않는다 — 매번 `dict(DEFAULT_CONFIG)` 를 새로 만든다.
    """
    rs = float(np.sqrt(s))
    ax, ay, at = x / s, y / s, tt / rs
    agx, agy, atq = gx / s, gy / s, tq / rs
    cfg = dict(DEFAULT_CONFIG)                      # ⛔`{}` 아님 (PREREG §1)
    cfg.update(depth=G.DEPTH, seed=seed, device='cpu', init_method='auto')
    assert 'learned_atoms' not in cfg, '제안 캐시가 새 config 에 이미 있다'
    upd = make_updater(cfg)                         # ★출하 진입점 (⛔맨 SequentialUpdater 금지)
    fr = R11.build_frames(ax, ay, at, z, G.GT_CONST)
    rb = R11.readback_frames(fr)
    with Probe() as pr:
        info = upd.initialize(fr, x_range=(float(agx.min()), float(agx.max())),
                              y_range=(float(agy.min()), float(agy.max())), verbose=False)
    gt_after = float(torch.cat([f.gt_field.reshape(-1) for f in fr]).std(unbiased=False))
    # ── P-implicit (`T11` 경로: 원 격자의 상에서 질의 — 역변환이 암묵적) ──
    pi = upd.predict_field(torch.tensor(agx, dtype=torch.float32),
                           torch.tensor(agy, dtype=torch.float32), float(atq))
    pi = np.asarray(pi.detach().cpu()).reshape(target.shape)
    # ── ★P-explicit (배포 가능 경로: 원자를 원 단위로 되돌려 원 격자에 렌더) ──
    m = getattr(upd, 'model', None)
    pe, st2, nnc, amd5 = None, None, None, None
    if m is not None and getattr(m, 'kx', None) is not None:
        kxp = np.asarray(m.kx.detach().cpu(), float) / s
        kyp = np.asarray(m.ky.detach().cpu(), float) / s
        Aa = np.asarray(m.w.detach().cpu(), float)
        ph = np.asarray(m.phi.detach().cpu(), float)
        pe = np.asarray(render_wave_field(kxp, kyp, Aa, ph, G.DEPTH, gx, gy, float(tq)))
        st2 = bool(getattr(m, 'use_stokes2', False))
        nnc = bool(getattr(m, 'use_neural_corrector', False))
        amd5 = hashlib.md5(np.round(np.c_[np.asarray(m.kx.detach().cpu()),
                                          np.asarray(m.ky.detach().cpu())], 6)
                           .tobytes()).hexdigest()
    vp = (info or {}).get('varpro') or {}
    la = cfg.get('learned_atoms')
    pmd5 = None
    if la:
        arr = np.c_[np.asarray(la['kx'], float).ravel(),
                    np.asarray(la['ky'], float).ravel(),
                    np.asarray(la['energy'], float).ravel()]
        pmd5 = hashlib.md5(np.round(arr, 9).tobytes()).hexdigest()
    a_, b_, rel = R11.measured_scales(ax, x, at, tt)
    rb.update(x_scale=a_, t_scale=b_, disp_relerr=rel, proposal_md5=pmd5, atoms_md5=amd5,
              proposal_device=pr.device, n_gathered=pr.n_gathered,
              cap_proposal=int(LI.M_OBS_CAP), gt_std_after=gt_after,
              vp_clamp_hits=vp.get('vp_clamp_hits'), prov=upd.arm_provenance(),
              n_atoms_installed=int(getattr(m, 'n_gaussians', 0) or 0),
              use_stokes2=st2, use_neural_corrector=nnc,
              invert_maxdiff=(float(np.max(np.abs(pi - pe))) if pe is not None else None),
              nrmse_explicit=(R11.nrmse(pe, target, s4) if pe is not None else None),
              deploy_consts={k: cfg.get(k) for k in
                             ('lambda_l1', 'varpro_steps', 'learned_init_iters',
                              'auto_switch_thresh', 'init_method', 'tracking_mode',
                              'analytic_ensemble_k', 'learned_ensemble_k', 'depth')})
    return R11.nrmse(pi, target, s4), rb


# ── 계획 ────────────────────────────────────────────────────────────────────
def plan(mode):
    """⛔실행 순서 = Tier-P 먼저, ANCHOR 마지막(`PREREG §8`).

    ★스모크는 **사전노출된 앵커 2파일에 6팔 전부**(`§7`) — 판정 표본 60 은 건드리지 않는다.
    """
    tier_p, anchor = G.tiers()
    if mode == 'smoke':
        rows = [(b, 'ANCHOR', G.TIERP_ARMS) for b in anchor]
        touched = {b for b, _, _ in rows} & set(tier_p)
        assert not touched, f'⛔스모크가 판정 표본을 건드린다: {sorted(touched)}'
        return rows, tier_p
    return ([(b, 'TIER-P', G.TIERP_ARMS) for b in tier_p]
            + [(b, 'ANCHOR', G.ANCHOR_ARMS) for b in anchor]), tier_p


def main(mode='primary'):
    torch.set_num_threads(1)
    t_start = time.time()
    md5_start = freeze_map()
    digest = code_digest(md5_start)
    ck = md5_start.get('sha256:' + CKPT)
    todo, tier_p = plan(mode)
    out = OUT.replace('.jsonl', '_SMOKE.jsonl') if mode == 'smoke' else OUT
    meta_o = META.replace('.json', '_SMOKE.json') if mode == 'smoke' else META
    if mode == 'smoke':
        print(f'⚠SMOKE — 앵커 {len(todo)}파일(사전노출) · 판정 표본 {len(tier_p)} **미접촉**')
    rows, cache, devs = [], {}, set()
    with zipfile.ZipFile(G.ZIP) as zf:
        for fi, (name, tier, arms) in enumerate(todo, 1):
            t0 = time.time()
            if name not in cache:
                E, gx, gy, tarr = R11.read_mat(zf, name)
                nf = int(G.WIN_S)
                Ew, tw = E[G.J0:G.J0 + nf], tarr[G.J0:G.J0 + nf]
                tgt = Ew[-1]
                s4 = 4.0 * float(np.std(Ew))
                sd = G.seed_of(name, tier, tier_p)
                # ★시드는 **적용된 RNG 상태**로 되읽는다 (⛔리터럴 왕복은 항진, `G-ENV`)
                rng = np.random.default_rng(sd)
                seed_applied = int(rng.bit_generator.seed_seq.entropy)
                seed_md5 = hashlib.md5(json.dumps(rng.bit_generator.state, sort_keys=True,
                                                  default=str).encode()).hexdigest()
                x, y, tt, zz = R11.sample_obs(Ew, gx, gy, tw, G.DENSITY, rng)
                tq = float(tw[-1] - tw[0])
                # ⛔이 파일이 **실제로 도는 팔**의 규칙만 판다 — ANCHOR 층(3팔)은 규칙 팔이 없다.
                #   그래야 §8 의 「60파일 × 16 호출 = 960」 이 되읽기와 맞는다.
                auto_here = [a for a in arms if a in G.AUTO_ARMS]
                rules = {a: eval_rule(a, x, y, tt, zz, G.DEPTH) for a in auto_here}
                bl = ({'s': {}, 'n_calls': {}, 'device': None} if not auto_here
                      else blind_s(x, y, tt, zz, G.DEPTH))
                for a in auto_here:                 # ★blind 호출 수를 합쳐 §8 을 되읽는다
                    rules[a]['beamform_calls_n'] += int(bl['n_calls'].get(a, 0))
                cache[name] = (E, gx, gy, tarr, tgt, s4, sd, seed_applied, seed_md5,
                               x, y, tt, zz, tq, rules, bl)
            (E, gx, gy, tarr, tgt, s4, sd, seed_applied, seed_md5,
             x, y, tt, zz, tq, rules, bl) = cache[name]
            ang, hs = R11.parse(name)
            base = dict(block=name, tier=tier, arm=None, angle=ang, hs=hs, sigma4=s4,
                        seed=sd, seed_applied=seed_applied, seed_rng_md5=seed_md5,
                        frame_index=G.J0 + int(G.WIN_S) - 1, n_frames=int(G.WIN_S),
                        density=G.DENSITY, dt=float(np.median(np.diff(tarr))), tq=tq,
                        e_shape=[int(v) for v in E.shape], mean_convention='raw',
                        axis_convention='E[j][iy, ix]',
                        L_obs_x=float(np.ptp(x)), L_obs_y=float(np.ptp(y)),
                        L_TRAIN=SR16.L_TRAIN, K_TRAIN=SR16.K_TRAIN,
                        ckpt_sha256=ck, code_md5_row=digest)
            got = {}
            for a in arms:
                ta = time.time()
                rule = dict(NULL_RULE)
                if a in G.AUTO_ARMS:
                    rule.update(rules[a])
                    rule['s_blind'] = bl['s'].get(a)
                    devs.add(rule['beamform_device'])
                else:
                    rule['s_blind'] = None
                    rule['s_rule'] = G.S_FIXED.get(a)
                if a == 'ZERO':
                    val = R11.nrmse(np.zeros_like(tgt), tgt, s4)
                    rb = dict(t_origin=float(tt.min()), n_offered=int(len(zz)),
                              n_frames=int(G.WIN_S), gt_std=None, gt_value=None,
                              gt_std_after=None, proposal_md5=None, atoms_md5=None,
                              proposal_device=None, n_gathered=None, cap_proposal=None,
                              vp_clamp_hits=None, prov={}, n_atoms_installed=0,
                              use_stokes2=None, use_neural_corrector=None,
                              invert_maxdiff=None, nrmse_explicit=None, deploy_consts={},
                              x_scale=None, t_scale=None, disp_relerr=None)
                else:
                    val, rb = estimate(rule['s_rule'], x, y, tt, zz, gx, gy, tq,
                                       tgt, s4, sd)
                xs = rb.get('x_scale')             # ⛔`(x or 1)` 금지 — 0.0 은 falsy 다
                rows.append(dict(base, arm=a, nrmse=val,
                                 s_readback=(None if (xs is None or xs == 0)
                                             else 1.0 / xs),
                                 wall_s=round(time.time() - ta, 2), **rule, **rb))
                got[a] = val
            eta = (time.time() - t_start) / fi * (len(todo) - fi) / 60.0
            print(f'  [{fi:2d}/{len(todo)}] {tier:6s} {os.path.basename(name):18s} '
                  + ' '.join(f"s({a[5:]} {rules[a]['s_rule']:.4f} raw "
                             f"{rules[a]['s_raw']:.4f})" for a in sorted(rules)) + ' | '
                  + ' · '.join(f'{k[:9]} {v:.4f}' for k, v in got.items())
                  + f'  ({time.time() - t0:.0f}s · 남은 {eta:.0f}분)', flush=True)
            with open(out, 'w', encoding='utf-8') as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    md5_end = freeze_map()
    json.dump(dict(mode=mode, n_rows=len(rows),
                   n_runs=sum(1 for r in rows if r['arm'] != 'ZERO'),
                   # ⛔`threads` 와 `threads_readback` 에 **같은 식**을 적으면 하나를 둘로 센 것이다
                   #   (`check_prereg C6c`) ⟹ 앞은 **의도(환경변수)**, 뒤는 **되읽기(라이브)**.
                   threads_env=os.environ.get('OMP_NUM_THREADS'),
                   threads_readback=int(torch.get_num_threads()),
                   blas_env={v: os.environ.get(v) for v in
                             ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
                   seed16=G.SEED16, code_md5=md5_end, code_md5_start=md5_start,
                   code_md5_row=digest, md5_stable=(md5_start == md5_end),
                   beamform_device=sorted(d for d in devs if d),
                   cuda_available=bool(torch.cuda.is_available()),
                   files=len(cache), arms=list(G.TIERP_ARMS),
                   # ★§8 이 요구한 것 — `T14` 는 경과 시각을 어디에도 안 적어 mtime 추론을 강요했다
                   t_start=t_start, t_end=time.time(),
                   elapsed_min=round((time.time() - t_start) / 60.0, 2)),
              open(meta_o, 'w'), ensure_ascii=False, indent=1)
    print(f'✅{len(rows)} 행 · md5 안정 {md5_start == md5_end} · '
          f'{(time.time() - t_start) / 60.0:.1f} 분 · 장치 {sorted(d for d in devs if d)}')


if __name__ == '__main__':
    a = sys.argv[1] if len(sys.argv) > 1 else 'primary'
    if a == 'freeze':
        write_sample()
        write_frozen()
    else:
        main(a)
