"""`T17-REALSCALE` 러너 — ⛔`gates_t17.py` · `analyze_t17.py` **다음에** 쓰였다.

★**행 스키마는 이 파일이 정하지 않는다** — `analyze_t17.py` 독스트링이 정본이고 러너가 따른다.
  ⚠그 스키마는 `PREREG §6.4` 보다 **넓다**(`obs_hash_used`·`*_prime`·`s_blind_*`·`rule_ids`·
  `seed_applied`·`rng_md5_seeded`·`gt_present`). 사유는 분석기 독스트링이 열마다 적었다.

★관측·프레임·채점은 `B-2` 를 **임포트해 비트 재사용**한다(⛔손복사 금지 — 앵커가 성립하려면 필수):
  · 장면 = `B2_REAL/scene_station.scene_station` (기본값 = `B-2`·`B-3`·`B-14b` 와 비트 동일)
  · 프레임 = `B2_REAL/run_b2.frames_from` (⟶ `make_gif_bank_deployed.frames_from_scene`)
  · 채점 기준 = `B2_REAL/zero_ref.zero_ref_nowcast` (⛔`ZERO_REF = 1.0` 은 철회됐다)

★**되읽기 규율** (⛔러너 리터럴을 분석기 리터럴과 대조하면 실패할 수 없는 게이트가 된다)
  · `(klim, n)` 과 첨두는 `bank_mmse._beamform_top` 을 **감싸 소비자가 받은 값**을 잡는다.
  · `s` 는 **프레임이 실제로 가진 좌표**의 ptp 비로 되읽는다(`G-S-READBACK`).
  · `depth_used` 는 **소비자**에서 읽는다 — 학습 팔은 `upd.model.depth`, bank 팔은 빔포머가
    받은 `depth` 인자다(⛔`cfg` 를 되읽는 것은 의도표다; 그것도 따로 적지만 게이트는 소비자를 본다).
  · `k̂'·d'` 은 팔의 **재라벨된 관측**에서 규칙을 다시 돌려 만든다(`G-DEPTH-EXACT`).
  · 시드는 `wavesplat.run.set_seed` 를 감싸 **적용된 값 + 직후 RNG 지문**을 잡는다(`G-SEED`).
  · 팔 정체는 `upd.arm_provenance()` **전체**(`G-ARM` — 의도가 아니라 되읽기).

⛔**팔을 `{}` 로 정의하지 않는다** — `BASE = dict(DEFAULT_CONFIG)` 를 **셀·팔마다 새로** 만들고
  배포 상수 6종을 **명시**한다(`gates_b7.py:46` 선례 · `T11 v1 BLOCKER-2` 원자 캐시 누출).
⛔`(x or 기본값)` 을 쓰지 않는다 — `0.0` 은 falsy 다. `is None` 으로 가른다.
⛔cwd 를 되돌리지 마라 — `d3_beamform_cond_diffusion.py:25` 의 chdir 은 의존성이다.
⛔`predict_field` 에는 **축 벡터**를 넘긴다(평탄화 금지 — `T11 A2 §0` OOM 2회).
⛔메모리 캡 밖에서 돌리지 마라:
  `systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 -- python3 -u run_t17.py smoke`
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import hashlib, json, subprocess, sys, tempfile, time, traceback, warnings   # noqa: E402
warnings.filterwarnings('ignore')

import numpy as np                                                 # noqa: E402
import torch                                                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gates_t17 as G                                              # noqa: E402
# ★`ANALYSIS_SIDE` 단일출처 — ⛔러너가 그 목록을 손으로 다시 적으면 측정면/분석면 분리가
#   두 파일에 중복 구현된다(저장소가 그 부류로 여러 번 대가를 치렀다).
import analyze_t17 as A17                                          # noqa: E402
import scale_rules_t14 as SR14                                     # noqa: E402
import scale_rules_t16 as SR16                                     # noqa: E402
import run_b2 as R2                                                # noqa: E402  (chdir(ROOT))
from scene_station import scene_station                            # noqa: E402
from wavesplat.optimizer import bank_mmse as BM                    # noqa: E402
from wavesplat.optimizer import learned_init as LI                 # noqa: E402
from wavesplat.optimizer.sequential_update import make_updater     # noqa: E402
from wavesplat.model.tnse import render_wave_field                 # noqa: E402
from wavesplat.run import DEFAULT_CONFIG                           # noqa: E402
import wavesplat.run as WR                                         # noqa: E402  ⛔bare name 필수

OUT = os.path.join(HERE, 'results_t17.jsonl')
META = os.path.join(HERE, 'meta_t17.json')
SAMPLE = os.path.join(HERE, 'sample_t17.json')
FROZEN = os.path.join(HERE, 'FROZEN_md5.txt')
FROZEN_A = os.path.join(HERE, 'FROZEN_md5_A.txt')
ANALYSIS_SIDE = A17.ANALYSIS_SIDE

# ── 동결 대상 (`PREREG §9.1`) ─────────────────────────────────────────────────────────
# ★키는 **실제 상대경로**다 — `tmp/_audit/check_batch_artifacts.py:_resolve()` 가
#   「배치 → 부모(tmp) → 저장소」 순으로 푼다. ⛔`T14/…` 같은 **라벨을 쓰지 마라**
#   (`T15` 가 라벨을 써서 25키 중 4키가 풀리지 않았다).
BATCH_FILES = ('gates_t17.py', 'analyze_t17.py', 'run_t17.py', 'blind_worker_t17.py',
               'sample_t17.json', 'PREREG.md', 'AMENDMENT_1_PREDATA.md')
IMPORTS = ('wavesplat/optimizer/sequential_update.py',
           'wavesplat/optimizer/learned_init.py',
           'wavesplat/optimizer/bank_mmse.py',
           'wavesplat/optimizer/varpro.py',
           'wavesplat/optimizer/initializer.py',
           'wavesplat/model/spectral_gaussian.py',
           'wavesplat/model/tnse.py',
           'wavesplat/model/dispersion.py',
           'wavesplat/model/confidence.py',
           'wavesplat/run.py',
           'wavesplat/data_loader.py',
           'wavesplat/real_data.py',
           'scripts/d3_beamform_net.py',
           'scripts/d3_beamform_cond_diffusion.py',
           'scripts/d3_setdiff_prior.py',
           'scripts/d3_sa_multirecord.py',
           'scripts/realdata_analysis.py',
           # ★깊이-2 전이 임포트 — ⛔*"값을 안 움직인다"* 를 주장하는 대신 **동결한다**.
           #   동결은 공짜이고 `T11 A2 §7` 은 `varpro` 를 그렇게 놓쳤다.
           'wavesplat/optimizer/coarse_to_fine.py',
           'wavesplat/optimizer/relax_estimator.py',
           'wavesplat/evaluation/metrics.py')
SIBLING_FILES = (
    # ★실측 관측·채점의 **단일 출처** (PREREG §2 · §9.1)
    'B2_REAL_2026-08-23/scene_station.py',
    'B2_REAL_2026-08-23/gates_b2.py',
    'B2_REAL_2026-08-23/zero_ref.py',
    'B2_REAL_2026-08-23/true_zero_field.json',
    'B2_REAL_2026-08-23/run_b2.py',
    'B3_REFIT_2026-08-23/true_zero_perlead.json',
    'bankmmse_port_2026-07-29/make_gif_bank_deployed.py',
    # ★규칙 (⛔손복사 금지 — `G-RULEID` 가 객체 동일성을 단언한다)
    'T16_BANDFIX_2026-09-05/scale_rules_t16.py',
    'T14_AUTOSCALE_2026-09-04/scale_rules_t14.py',
    # ★앵커 자료 (§6.2) — ⛔동결하지 않으면 «앵커가 움직였다» 를 탐지할 수 없다
    'B3_REFIT_2026-08-23/results_b3.jsonl',
    'B14_OBSMODEL_2026-08-24/results_b14b_0.jsonl',
    'B14_OBSMODEL_2026-08-24/results_b14b_1.jsonl',
)
CKPT = 'results/direction3/beamform_cond_diffusion_mixed_v1.pt'
TMP = os.path.dirname(HERE)


def _md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def _sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for c in iter(lambda: fh.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def freeze_map():
    """★동결 지문 — `G-ENV` 가 「지금 다시 계산 == meta == 0행 동결」 3중으로 댄다.

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
    for f in SIBLING_FILES:
        p = os.path.join(TMP, f)
        if os.path.exists(p):
            out[f] = _md5(p)
    p = os.path.join(G.ROOT, CKPT)
    if os.path.exists(p):
        out['sha256:' + CKPT] = _sha256(p)
    return out


def code_digest(live):
    """★행에 실을 **측정면** 지문. ⛔분석기와 같은 식이어야 한다(그쪽이 대조한다)."""
    meas = {k: v for k, v in live.items() if k not in ANALYSIS_SIDE}
    return hashlib.md5(json.dumps(meas, sort_keys=True).encode()).hexdigest()


def write_frozen():
    """`FROZEN_md5.txt`(측정면) · `FROZEN_md5_A.txt`(분석면) — **결과 0행에서** 찍는다."""
    live = freeze_map()
    head = ('# T17-REALSCALE 동결 — **결과 0행 · 추정기 런 0회 시점**\n'
            '# 2026-09-05 · ⛔`run_t17.freeze_map()` 이 낸다. 손으로 고치지 마라.\n'
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
          f'import_md5 {code_digest(live)}')
    return live


def write_sample():
    """`sample_t17.json` = 72셀 (⛔손목록이 아니라 `gates_t17.cells()` 가 계산한다)."""
    ids = [G.cell_id(rec, w) for rec, w in G.cells()]
    json.dump(ids, open(SAMPLE, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'✅표본 {len(ids)} → {os.path.basename(SAMPLE)}')
    return ids


def git_head():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=G.ROOT,
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return None


def rng_md5():
    s = np.random.get_state()
    return hashlib.md5(np.asarray(s[1]).tobytes() + str(s[2]).encode()).hexdigest()


def h_arr(*arrs):
    """관측 지문 — `B-2 run_b2` 의 규약(float64 연속 배열의 md5, 앞 12자)."""
    a = np.column_stack([np.asarray(x, dtype=np.float64).ravel() for x in arrs])
    return hashlib.md5(np.ascontiguousarray(a, dtype=np.float64)).hexdigest()[:12]


# ── 되읽기 후크 ───────────────────────────────────────────────────────────────────────
class BeamProbe:
    """★`bank_mmse._beamform_top` 을 감싸 **소비자가 실제로 받은** `(klim, n, depth)` 와
    **돌려준 첨두**를 잡는다.

    ⛔러너가 자기 리터럴을 적는 것이 아니다 — 규칙/추정기가 무엇을 넘겼는지는 여기서만 알 수 있다.
    ⛔`__exit__` 에서 반드시 되돌린다(되돌리기 실패 = 프로세스 전역 오염).
    """

    def __enter__(self):
        self.calls, self.peaks, self.depths = [], [], []
        self.device = str(BM._DEV)
        orig = BM._beamform_top

        def wrap(P, r, depth, klim=0.6, n=80, _o=orig, _s=self):
            out = _o(P, r, depth, klim=klim, n=n)
            _s.calls.append([float(klim), int(n)])
            _s.depths.append(float(depth))
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


class ArmProbe:
    """팔 하나를 도는 동안의 계측 — 개입이 아니라 **되읽기 계수기**.

    잡는 것: 제안 장치·모은 관측 수 · 적용된 시드와 직후 RNG 지문 · bank 빔포머 호출 수.
    """

    def __enter__(self):
        self.device = None
        self.n_gathered = None
        self.seed_applied = None
        self.rng_md5_seeded = None
        self.bank_calls = 0
        self.bank_depths = []
        self.proposal_md5 = None
        self.n_raw = None
        self.n_sel = None
        self._undo = []

        # ★제안 지문은 **`pool_and_select` 가 실제로 고른 원자**에서 만든다(`T15` 관용구).
        #   ⛔`cfg['learned_atoms']` 를 읽는 방법은 **명시 `init_method='learned'` 경로에서
        #   구조적으로 실패한다** — 그 캐시는 `_auto_select_init`(라우터) 안에서만 쓰이고
        #   이 배치의 `PROP*` 팔은 라우터를 타지 않는다. **스모크가 잡았다**(전건 None).
        ops = LI.pool_and_select

        def wp(kx, ky, en, *a, _o=ops, _s=self, **kw):
            _s.n_raw = int(len(kx))
            out = _o(kx, ky, en, *a, **kw)
            skx, sky, sen = (np.asarray(out[0]), np.asarray(out[1]), np.asarray(out[2]))
            _s.n_sel = int(len(skx))
            arr = (np.c_[skx.ravel(), sky.ravel(), sen.ravel()] if len(skx)
                   else np.zeros((0, 3)))
            _s.proposal_md5 = hashlib.md5(np.round(arr, 9).tobytes()).hexdigest()
            return out
        LI.pool_and_select = wp
        self._undo.append(lambda: setattr(LI, 'pool_and_select', ops))

        ol, og = LI._load_cond, LI._gather_init_obs

        def wl(c, d, _o=ol, _s=self):
            _s.device = str(d)
            return _o(c, d)

        def wg(fr, _o=og, _s=self):
            o = _o(fr)
            _s.n_gathered = None if o is None else int(len(o[0]))
            return o
        LI._load_cond, LI._gather_init_obs = wl, wg
        self._undo += [lambda: setattr(LI, '_load_cond', ol),
                       lambda: setattr(LI, '_gather_init_obs', og)]

        # ★시드는 **적은 값**이 아니라 **적용된 값**을 기록한다(`G-SEED`, `T15` 관용구).
        #   ⛔러너 리터럴을 분석기 리터럴과 대조하면 **실패할 수 없는 게이트**가 된다.
        oss = WR.set_seed

        def wss(seed, _o=oss, _s=self):
            _o(seed)
            _s.seed_applied = int(seed)
            _s.rng_md5_seeded = rng_md5()
        WR.set_seed = wss
        self._undo.append(lambda: setattr(WR, 'set_seed', oss))

        # ★`PREREG §10` — 규칙이 빌려 쓰는 빔포머가 **학습 경로에서 실행되지 않는다**를
        #   주장으로 두지 않고 **센다**. bank 팔에서는 반대로 0 이 아니어야 한다.
        obt = BM._beamform_top

        def wbt(P, r, depth, klim=0.6, n=80, _o=obt, _s=self):
            _s.bank_calls += 1
            _s.bank_depths.append(float(depth))
            return _o(P, r, depth, klim=klim, n=n)
        BM._beamform_top = wbt
        self._undo.append(lambda: setattr(BM, '_beamform_top', obt))
        return self

    def __exit__(self, *a):
        for f in reversed(self._undo):
            f()
        return False


def blind_s(x, y, t, z, depth):
    """★별도 프로세스에서 규칙을 재실행 — 관측 5배열만 넘긴다(`G-S-BLIND`)."""
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        np.savez(f, x=x, y=y, t=t, z=z, depth=np.float64(depth))
        p = f.name
    try:
        r = subprocess.run([sys.executable, os.path.join(HERE, 'blind_worker_t17.py'), p],
                           capture_output=True, text=True, timeout=1800)
        if r.returncode != 0:
            raise RuntimeError(f'blind_worker 실패 rc={r.returncode}\n{r.stderr[-2000:]}')
        return json.loads(r.stdout.strip().splitlines()[-1])
    finally:
        os.unlink(p)


# ── 규칙 평가 (셀 단위) ───────────────────────────────────────────────────────────────
def eval_rules(x, y, t, z, depth):
    """★셀의 세 규칙을 한 번씩 돌리고 소비자 값을 **전부 되읽는다**.

    ⛔`s_rule` 을 얻으려고 규칙을 두 번 부르지 않는다 — 비클리핑을 한 번 얻고 레일은
    `SR16.clip_s` 가 씌운다(`s_bandr`/`s_ap` 내부와 **비트 동일**한 식).
    ⚠`AUTO-AP` 는 빔포머를 **0회** 부른다 — 그것 자체를 되읽어 적는다.
    """
    with BeamProbe() as pb:
        s_raw_bandr = float(SR16.s_bandr_raw(x, y, t, z, depth))
    n_bandr = len(pb.calls)
    with BeamProbe() as pa:
        s_raw_ap = float(SR14.s_ap_raw(x, y, t, z, depth))
    with BeamProbe() as po:
        s_raw_old = float(SR14.s_band_raw(x, y, t, z, depth))
    k1, k2 = pb.kp(0), pb.kp(1)
    klim1, klim2 = pb.calls[0][0], pb.calls[1][0]
    dk1 = 2.0 * klim1 / (SR16.N1 - 1)          # ★후크의 klim 에서 유도(리터럴 아님)
    return dict(
        s_raw_bandr=s_raw_bandr, s_rule_bandr=SR16.clip_s(s_raw_bandr),
        s_raw_ap=s_raw_ap, s_rule_ap=SR16.clip_s(s_raw_ap),
        s_raw_band_old=s_raw_old, s_rule_band_old=SR16.clip_s(s_raw_old),
        kp1_hat=k1, kp2_hat=k2, klim1=klim1, klim2=klim2, dk1=dk1,
        # ★`margin` 은 **2단** 첨두다 — `PREREG §0.2` 프로브 B 가 `margin = kp_rel/dk1` 이라
        #   적었고 `kp_rel = SR16.kp_hat_rel(...)` 은 2단 값을 돌려준다(항등 `≡ L_obs/λ̂_p`).
        #   ⛔초판은 1단(`k1`)을 썼고 **스모크가 잡았다**(BS11a 67.974 vs 등록 226.136).
        margin=(None if k2 is None else k2 / dk1),
        margin_stage1=(None if k1 is None else k1 / dk1),
        # `edge_hit`/`edge_margin` 은 **1단**이다(§6.3: "1단 argmax 가 격자 가장자리").
        edge_margin=(None if k1 is None else k1 / klim1),
        edge_hit=(None if k1 is None else bool(k1 > klim1 * (1.0 - 2.0 / (SR16.N1 - 1)))),
        floor_margin=(None if (k1 is None or k2 is None) else min(k1, k2) / SR16.FLOOR),
        kd=(None if k2 is None else float(k2 * depth)),
        beamform_calls=[list(c) for c in pb.calls],
        beamform_calls_rule=n_bandr + len(pa.calls) + len(po.calls),
        beamform_device=pb.device,
        # ⛔`id()` 는 **프로세스마다 다르다** — 러너↔분석기 대조에 쓸 수 없다(스모크가 잡았다).
        #   신원은 `module.qualname` 으로 적고, 객체 동일성은 분석기가 자기 안에서 단언한다.
        rule_qualnames={k: G.qualname_of(f) for k, f in
                        (('s_bandr', SR16.s_bandr), ('s_bandr_raw', SR16.s_bandr_raw),
                         ('s_ap', SR14.s_ap), ('s_ap_raw', SR14.s_ap_raw),
                         ('s_band_raw', SR14.s_band_raw),
                         ('kp_hat_rel', SR16.kp_hat_rel), ('clip_s', SR16.clip_s))},
        rule_ids_in_runner={'s_bandr': id(SR16.s_bandr), 's_ap': id(SR14.s_ap)})


def eval_prime(ax, ay, at, z, depth_eff):
    """★팔의 **재라벨된 관측**에서 규칙을 다시 돌린다 — `G-DEPTH-EXACT` 의 `k̂'·d'` 항.

    ⛔이것이 없으면 `k'd' = (s·k)(d/s) = kd` 는 **산수 항등식**이 되어 게이트가 항진한다.
    여기서는 재라벨된 좌표에서 규칙이 **실제로 낸** 첨두를 쓰므로 지수·수심을 틀리면 죽는다.
    """
    with BeamProbe() as pb:
        SR16.kp_hat_rel(ax, ay, at, z, depth_eff)
    k1, k2 = pb.kp(0), pb.kp(1)
    klim1 = pb.calls[0][0]
    return dict(kp1_hat_prime=k1, kp2_hat_prime=k2, klim1_prime=klim1,
                dk1_prime=2.0 * klim1 / (SR16.N1 - 1),
                kd_prime=(None if k2 is None else float(k2 * depth_eff)),
                beamform_depth_prime=(pb.depths[0] if pb.depths else None),
                n_calls_prime=len(pb.calls))


def measured_scales(x_used, x_orig, t_used, t_orig):
    """★재척도가 **실제로 적용된 배율**을 나눠서 되읽고 분산관계 보존을 잰다.

    `x' = a·x` ⟹ `k' = k/a` · `t' = b·t` ⟹ `ω' = ω/b`.
    `ω=√(g k tanh(k d))` 가 `d' = d/s` 아래 보존되려면 **`b = √a`** 다.
    ⛔해석적 항등식이 아니다 — 지수를 잘못 쓰면(예: `t·√s`) 이 값이 커진다.
    """
    mx = np.abs(x_orig) > 1e-9
    mt = np.abs(t_orig) > 1e-9
    a = float(np.median(x_used[mx] / x_orig[mx])) if mx.any() else 1.0
    b = float(np.median(t_used[mt] / t_orig[mt])) if mt.any() else 1.0
    return a, b, float(abs(b - np.sqrt(a)) / np.sqrt(a))


# ── 팔 하나 ───────────────────────────────────────────────────────────────────────────
def run_arm(arm, s, sc, x, y, t, z, t_shift, t0, depth, seed, prime_cache):
    """★셀·팔마다 **새 config** 를 만든다 (⛔제안 캐시 누출 차단 · `T11 v1 BLOCKER-2`)."""
    rs = float(np.sqrt(s))
    ax, ay, at = x / s, y / s, (t - t_shift) / rs
    gxq, gyq = np.asarray(sc['gxq']), np.asarray(sc['gyq'])
    agx, agy = gxq / s, gyq / s
    depth_eff = float(depth) / float(s)

    key = repr(float(s))
    if key not in prime_cache:
        prime_cache[key] = eval_prime(ax, ay, at, z, depth_eff)
    prime = dict(prime_cache[key])

    cfg = dict(DEFAULT_CONFIG)                      # ⛔`{}` 아님 (PREREG §1)
    cfg.update(depth=depth_eff, seed=int(seed), device='cpu')
    # ★배포 상수를 **명시**한다 (⛔런타임 스냅샷에 맡기지 않는다 — `B-7` BLOCKER)
    cfg.update(G.deploy_consts())
    cfg.update(G.ARM_CFG[arm])
    assert 'learned_atoms' not in cfg, '제안 캐시가 새 config 에 이미 있다'

    with ArmProbe() as pr:
        WR.set_seed(int(seed))                      # ★후크가 **적용된** 값을 잡는다
        upd = make_updater(cfg)                     # ★출하 진입점 (⛔맨 SequentialUpdater 금지)
        frames = R2.frames_from({'P': np.column_stack([ax, ay, at]), 'z': z}, 0.0)
        gt_present = any(getattr(f, 'gt_field', None) is not None for f in frames)
        n_offered = int(sum(len(f.x_obs) for f in frames))
        t_origin = float(min(float(f.t) for f in frames))
        info = upd.initialize(frames,
                              x_range=(float(agx.min()), float(agx.max())),
                              y_range=(float(agy.min()), float(agy.max())),
                              verbose=False)
    assert 'learned_atoms' not in cfg, '호출자 config 가 변형됐다 (캐시 역류)'

    # ── 채점: 미래 프레임 τ=0 이 종점(`B-2` 계승), lead 곡선은 기술통계 ─────────────
    curve, Zn = [], None
    for li, l_ in enumerate(np.asarray(sc['leads'])):
        g = np.asarray(sc['maps_gt'][li], float)
        mm = np.isfinite(g)
        if not mm.any():
            curve.append(float('nan'))
            continue
        Z = upd.predict_field(torch.tensor(agx, dtype=torch.float32),
                              torch.tensor(agy, dtype=torch.float32),
                              (t0 + float(l_)) / rs)
        Z = np.asarray(Z.detach().cpu()).reshape(len(gyq), len(gxq))
        if li == 0:
            Zn = Z
        curve.append(float(np.sqrt(np.mean((Z[mm] - g[mm]) ** 2)) / sc['sg']))

    # ── ★P-explicit (배포 가능 경로: 원자를 원 단위로 되돌려 원 격자·원 시각에 렌더) ──
    m = getattr(upd, 'model', None)
    pe, st2, nnc, amd5, depth_used = None, None, None, None, None
    if m is not None and getattr(m, 'kx', None) is not None:
        kxp = np.asarray(m.kx.detach().cpu(), float) / s
        kyp = np.asarray(m.ky.detach().cpu(), float) / s
        Aa = np.asarray(m.w.detach().cpu(), float)
        ph = np.asarray(m.phi.detach().cpu(), float)
        pe = np.asarray(render_wave_field(kxp, kyp, Aa, ph, float(depth), gxq, gyq, t0))
        st2 = bool(getattr(m, 'use_stokes2', False))
        nnc = bool(getattr(m, 'use_neural_corrector', False))
        amd5 = hashlib.md5(np.round(np.c_[np.asarray(m.kx.detach().cpu()),
                                          np.asarray(m.ky.detach().cpu())], 6)
                           .tobytes()).hexdigest()
        depth_used = float(getattr(m, 'depth'))     # ★소비자 되읽기 (모델 객체)
    if depth_used is None and pr.bank_depths:
        depth_used = float(pr.bank_depths[0])       # ★소비자 되읽기 (빔포머가 받은 인자)

    vp = info if isinstance(info, dict) else {}
    vpd = vp.get('varpro')
    pmd5 = pr.proposal_md5             # ★후크가 잡은 **실제로 고른** 제안 원자의 지문
    a_, b_, rel = measured_scales(ax, x, at, t - t_shift)
    ucfg = getattr(upd, 'config', {})
    n_at = getattr(m, 'n_gaussians', None) if m is not None else None
    g0 = np.asarray(sc['maps_gt'][0], float)
    m0 = np.isfinite(g0)
    nrmse_e = (None if pe is None else
               float(np.sqrt(np.mean((pe[m0] - g0[m0]) ** 2)) / sc['sg']))
    row = dict(
        arm=arm, s=float(s), s_readback=(None if a_ == 0 else 1.0 / a_),
        x_scale=a_, t_scale=b_, disp_relerr=rel,
        obs_hash_used=h_arr(ax, ay, at, z),
        depth=float(depth), depth_used=depth_used,
        depth_cfg_readback=float(ucfg.get('depth')) if ucfg.get('depth') is not None else None,
        nowcast=curve[0], leadmean=float(np.nanmean(curve)), curve=curve,
        nrmse_explicit=nrmse_e,
        invert_maxdiff=(None if (pe is None or Zn is None) else float(np.max(np.abs(Zn - pe)))),
        n_atoms=(0 if n_at is None else int(n_at)),
        varpro_fired=bool(vpd),
        varpro_steps=vp.get('varpro_steps'),
        vp_clamp_hits=(vpd.get('vp_clamp_hits') if isinstance(vpd, dict) else None),
        init_iters=vp.get('n_iters'), ensemble_k=vp.get('ensemble_k'),
        proposal_md5=pmd5, atoms_md5=amd5,
        n_proposal_raw=pr.n_raw, n_proposal_sel=pr.n_sel,
        use_stokes2=st2, use_neural_corrector=nnc,
        proposal_device=pr.device, n_gathered=pr.n_gathered,
        cap_proposal=int(LI.M_OBS_CAP),
        n_offered=n_offered, t_origin=t_origin, gt_present=gt_present,
        beamform_calls_bank=int(pr.bank_calls),
        seed=int(seed), seed_applied=pr.seed_applied, rng_md5_seeded=pr.rng_md5_seeded,
        arm_provenance=upd.arm_provenance(),
        arm_cfg=dict(G.ARM_CFG[arm]),
        effective={k: ucfg.get(k) for k in G.DEPLOY_KEYS},
    )
    row.update(prime)
    return row


# ── 계획 ──────────────────────────────────────────────────────────────────────────────
def plan(mode):
    """⛔실행 순서 = 셀 순(record × win), 팔은 셀 안에서 6개 연속(장면 1회 로드 재사용).

    ★스모크(`PREREG §7`) = `BS11a` win=3(**`G-EDGE` 가 발화하는 유일한 셀** + 레일 클립 +
      `G-RAILID`) 와 `AA14` win=3(|Δω/ω| 최대 9.46% ⟹ `G-DEPTH-EXACT` 가 값을 가른다).
    """
    if mode == 'smoke':
        return [('BS11a', G.ANCHOR_WIN), ('AA14', G.ANCHOR_WIN)]
    return G.cells()


def main(mode='primary'):
    torch.set_num_threads(1)
    t_start = time.time()
    md5_start = freeze_map()
    digest = code_digest(md5_start)
    ck = md5_start.get('sha256:' + CKPT)
    gh = git_head()
    todo = plan(mode)
    out = OUT.replace('.jsonl', '_SMOKE.jsonl') if mode == 'smoke' else OUT
    meta_o = META.replace('.json', '_SMOKE.json') if mode == 'smoke' else META
    if mode == 'smoke':
        print(f'⚠SMOKE — {len(todo)}셀 × {G.N_ARM}팔 = {len(todo) * G.N_ARM}런 '
              f'(⛔판정 표본 {G.N_CELL} 중 {len(todo)}만 · 본실행이 전수를 다시 돈다)')
    rows, devs = [], set()
    for ci, (rec, win) in enumerate(todo, 1):
        t0c = time.time()
        cid = G.cell_id(rec, win)
        sc = scene_station(rec, win=win, W=G.W_OBS, m_scan=G.M_SCAN, m_cap=G.M_CAP,
                          noise=G.NOISE, cap_mode=G.CAP_MODE)
        P = np.asarray(sc['P'], float)
        z = np.asarray(sc['z'], float)
        x, y, t = P[:, 0], P[:, 1], P[:, 2]
        depth = float(sc['depth'])
        t_shift = float(t.min())
        t0 = float(sc['Pq'][0, 2]) - t_shift
        zt = G.zero_true(rec, win)                  # ★함수다 (⛔상수 1.0 금지)
        rules = eval_rules(x, y, t, z, depth)
        bl = blind_s(x, y, t, z, depth)
        devs.add(rules['beamform_device'])
        base = dict(cell_id=cid, record=rec, win=int(win), site=G.site_of(rec),
                    date=G.date_of(rec), phase='A',
                    n_obs=int(len(z)), sigma=float(sc['sg']), Hs=float(sc['Hs']),
                    m_scan=int(G.M_SCAN), m_cap=int(G.M_CAP), noise=float(G.NOISE),
                    W_obs=float(G.W_OBS), cap_mode=str(G.CAP_MODE),
                    leads=[float(v) for v in np.asarray(sc['leads'])],
                    t_start=float(t.min()), t_end=float(t.max()),
                    obs_hash=h_arr(x, y, t, z), q_hash=h_arr(np.asarray(sc['Pq'], float)),
                    zero_true=float(zt),
                    s_blind_bandr=bl['s']['bandr'], s_blind_ap=bl['s']['ap'],
                    s_blind_band_old=bl['s']['band_old'],
                    beamform_calls_blind=int(sum(bl['n_calls'].values())),
                    git_head=gh, import_md5=digest, ckpt_sha256=ck)
        base.update(rules)
        prime_cache, got = {}, {}
        for arm in G.ARM_ORDER:
            ta = time.time()
            s = G.s_of_arm(arm, rules['s_rule_bandr'], rules['s_rule_ap'])
            try:
                r = run_arm(arm, s, sc, x, y, t, z, t_shift, t0, depth, G.SEED,
                            prime_cache)
                r.update(base, wall_s=round(time.time() - ta, 2))
                r['s_clipped'] = bool(
                    arm == 'PROP-BANDR' and rules['s_rule_bandr'] != rules['s_raw_bandr']
                    or arm == 'PROP-AP' and rules['s_rule_ap'] != rules['s_raw_ap'])
                r['s_raw'] = (rules['s_raw_bandr'] if arm == 'PROP-BANDR' else
                              rules['s_raw_ap'] if arm == 'PROP-AP' else float(s))
            except Exception as e:                                  # noqa: BLE001
                r = dict(base, arm=arm, s=float(s), error=repr(e),
                         tb=traceback.format_exc()[-1200:],
                         wall_s=round(time.time() - ta, 2))
            rows.append(r)
            got[arm] = r.get('nowcast', r.get('error'))
            if r.get('proposal_device'):
                devs.add(r['proposal_device'])
        eta = (time.time() - t_start) / ci * (len(todo) - ci) / 60.0
        print(f'  [{ci:2d}/{len(todo)}] {cid:12s} z*={zt:.4f} '
              f"s(bandr {rules['s_rule_bandr']:.4f} raw {rules['s_raw_bandr']:.6f} · "
              f"ap {rules['s_rule_ap']:.4f}) edge={rules['edge_hit']} "
              f"margin={rules['margin']:.3f} | "
              + ' · '.join(f'{k[:10]} '
                           + (f'{v:.4f}' if isinstance(v, float) else str(v)[:40])
                           for k, v in got.items())
              + f'  ({time.time() - t0c:.0f}s · 남은 {eta:.0f}분)', flush=True)
        with open(out, 'w', encoding='utf-8') as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    md5_end = freeze_map()
    json.dump(dict(mode=mode, n_rows=len(rows), n_cells=len(todo),
                   n_runs=sum(1 for r in rows if 'error' not in r),
                   n_errors=sum(1 for r in rows if 'error' in r),
                   # ⛔`threads` 와 `threads_readback` 에 **같은 식**을 적으면 하나를 둘로 센 것이다
                   #   ⟹ 앞은 **의도(환경변수)**, 뒤는 **되읽기(라이브)**.
                   threads_env=os.environ.get('OMP_NUM_THREADS'),
                   threads_readback=int(torch.get_num_threads()),
                   blas_env={v: os.environ.get(v) for v in
                             ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
                   seed=int(G.SEED), seed17_base=int(G.SEED17),
                   code_md5=md5_end, code_md5_start=md5_start, import_md5=digest,
                   md5_stable=(md5_start == md5_end),
                   git_head=gh, ckpt_sha256=ck,
                   devices=sorted(d for d in devs if d),
                   cuda_available=bool(torch.cuda.is_available()),
                   arms=list(G.ARM_ORDER),
                   # ★§8 이 요구한 것 — `T14`·`B-2` 는 경과 시각을 안 적어 mtime 추론을 강요했다.
                   #   ⚠행의 동명 키(`t_start`/`t_end`)는 **관측창 시각**이다(분석기 독스트링).
                   t_start=t_start, t_end=time.time(),
                   elapsed_min=round((time.time() - t_start) / 60.0, 2)),
              open(meta_o, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'✅{len(rows)} 행 · 에러 {sum(1 for r in rows if "error" in r)} · '
          f'md5 안정 {md5_start == md5_end} · {(time.time() - t_start) / 60.0:.1f} 분 · '
          f'장치 {sorted(d for d in devs if d)}')


if __name__ == '__main__':
    a = sys.argv[1] if len(sys.argv) > 1 else 'primary'
    if a == 'freeze':
        write_sample()
        write_frozen()
    else:
        main(a)
