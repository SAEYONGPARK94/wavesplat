"""T3-NOW 러너 — ★**실측 나우캐스트 결손을 두 노브(앙상블 · VARPRO)로 가른다.**

판정은 `gates_t3.py` 에 **결과 0행에서 동결**돼 있다. 이 파일은 자료를 만들 뿐이다.
⛔문턱·팔·시드·오프셋·기대값을 여기서 **다시 만들지 않는다** — 전부 `gates_t3` 를 **호출**한다.

프로토콜(전문 = `G.PROTOCOL`, 그것이 정본이다). 요지:
  ① `scene_station(record, win, noise=NOISE)` → `frames_from_scene(sc, t_shift, float32)`
  ② 멤버마다 `set_seed(cfg['seed'])` → `make_updater(cfg)` → `initialize(...)` →
     `predict_field(tx, ty, t0)` → σ정규화 RMSE
  ③ 멤버 장을 **점별 median** (`G.MEDIAN_RULE`) → 같은 식으로 `e_med3`
★`update()` 는 **0회**다 — lead 0 은 초기화 창 이후 관측이 들어오기 **전** 이다.
⛔②의 세 줄은 `T2 run_t2.py:run()` 의 lead-0 경로 축자다. 인자 형(torch float32)을 바꾸면
**앵커가 깨진다**.

## ★장치 정책 (이 배치의 핵심 공시 — `PREREG §1.4`, `premise_t3.json:device_sensitivity`)

`T2`(cuda 가용)와 `M8`(cuda 차단)의 **같은 셀**을 비트 비교하면 bank 는 **12/12 일치**인데
제안 경로는 **0/12**(최대 상대차 245.1%)다. 사유는 `learned_init.propose_atoms:132-133` 이
`config['device']` 를 **받지도 않고** 스스로 `cuda if available else cpu` 를 고르고
`cd.sample` 이 그 장치에서 표집하기 때문이다.
⟹ ★**이 배치는 `T2` 와 같은 환경 하나로 고정한다**: `cfg['device']='cpu'` (그 값은
`G.DEPLOY_PIN` 에서 읽는다) **이면서 CUDA 는 가용한 채로 둔다**.
⛔**`CUDA_VISIBLE_DEVICES=''` 를 세우지 마라** — 그러면 `M8` 환경이 되고
`G-ANCHOR PROP-V400M3`(36/36 비트)이 **구조적으로 통과 불가**가 된다.
⚠**공시**: 그러므로 `PROP` 두 팔의 확산 표집(`cd.sample`)과 bank 의 beamform zoom
(`bank_mmse.py:41 _DEV`)은 **GPU 에서 돈다** — `T2` 144런이 이미 그랬다. 선택이 아니라
앵커가 정한 것이고, 행마다 `device`·`cuda_available` 을 실어 `G-7` 이 단일 스탬프를 요구한다.

    python3 tmp/T3_NOWCAST_2026-08-30/run_t3.py selftest
    T3_APPROVED=1 python3 …/run_t3.py smoke --approved
    T3_APPROVED=1 python3 …/run_t3.py run <shard> <nshard> --approved
"""
import os

# ★스레드는 러너가 **스스로** 고정한다(⛔호출자에게 맡기지 마라 — M7 E2 가 18시간을 잃었다).
#   ⚠`import torch` **앞**이어야 효력이 있다(BLAS 는 임포트 시점에 읽는다).
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
# ⛔여기에 `CUDA_VISIBLE_DEVICES=''` 를 넣지 마라 — 위 「장치 정책」 참조.

import hashlib                                                      # noqa: E402
import json                                                         # noqa: E402
import sys                                                          # noqa: E402
import time                                                         # noqa: E402
import traceback                                                    # noqa: E402
import warnings                                                     # noqa: E402

warnings.filterwarnings('ignore')
import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for _p in (HERE, ROOT, os.path.join(ROOT, 'scripts'),
           os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23'),
           os.path.join(ROOT, 'tmp', 'B3_REFIT_2026-08-23'),
           os.path.join(ROOT, 'tmp', 'bankmmse_port_2026-07-29')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(ROOT)

import gates_t3 as G                                                # noqa: E402
from scene_station import scene_station                             # noqa: E402
import make_gif_bank_deployed as MK                                 # noqa: E402

OUT = os.path.join(HERE, 'results_t3.jsonl')
SMOKE_OUT = os.path.join(HERE, 'smoke_t3.jsonl')
SMOKE_CELL = ('BS13a', 1)      # ⚠**이미 노출된 셀**(`B-3 PREREG` 스모크 기록 · `T2` 동일)

# ── ★승인 가드 (선례 `M7 E1` · `T2`) ────────────────────────────────────────
ENV = 'T3_APPROVED'
FLAG = '--approved'
ASK = (f'`{G.NAME}` 의 `PROP-V400M3`·`PROP-V0M3` 두 팔은 실측 12 레코드에 제안 표현을\n'
       '**강제**합니다(`real_domain_gate=False`). 상시 비승인 정책이고 `W12`·`B-2`·\n'
       '`M7 E1`·`T2-STAT` 때 각각 명시 승인으로 풀렸습니다.\n'
       f'사전 기대는 C1·C2 UNDECIDED~양성, C3 음성 쪽. 가족 MDE q80={G.mde()["q80"]}.\n'
       f'행 {G.expected_rows()} · 추정기 런 {G.expected_runs()}.\n'
       '승인 근거는 `APPROVAL_T3_2026-08-30.md` 에 축자로 있습니다.\n'
       f'⟹ `{ENV}=1 … run_t3.py … {FLAG}` 로 **둘 다** 주어야 돕니다.')


def approved(argv, env):
    """★순함수. ⚠가드도 *"항진하지 않는가"* 를 시험해야 한다 — 항상 거부/항상 통과하는
    가드는 그 자체로 결함이다. `selftest()` 가 네 조합을 전부 친다."""
    has_flag = FLAG in list(argv)
    has_env = env.get(ENV) == '1'
    return has_flag, has_env, (has_flag and has_env)


# ── 장 해시 · median 연산자 ────────────────────────────────────────────────
def field_md5(Z):
    """장의 비트 지문. ⛔멤버 장과 median 장이 **같은 dtype·같은 배치**여야 비교가 성립한다."""
    return hashlib.md5(
        np.ascontiguousarray(np.asarray(Z, dtype=np.float32)).tobytes()).hexdigest()


def median_field(fields):
    """★`G.MEDIAN_RULE` 의 구현 — `sequential_update.py:1337-1338` 의 축자 거울.

    `np.median(np.stack(members, axis=0), axis=0)`. k 가 홀수라 결과는 **어느 점에서든
    멤버 값 중 하나**이고 보간이 없다 ⟹ float32 로 반올림된 멤버들의 median 은
    float64 median 의 float32 반올림과 **비트 동일**하다(반올림이 단조이므로).
    ★그 항등식의 구속력 있는 증명은 `G-MIRROR` 다(하네스 ≡ 패키지 k=3, 36/36).
    """
    return np.ascontiguousarray(
        np.median(np.stack([np.asarray(f, dtype=np.float32) for f in fields], axis=0),
                  axis=0), dtype=np.float32)


def score(Z, g, mm, sg):
    """⛔`T2 run_t2.py:214` 축자 — 앵커가 비트로 성립하는 이유."""
    return float(np.sqrt(np.mean((Z[mm] - g[mm]) ** 2)) / sg)


# ── 앵커 (동결 `T2` 와의 **비트** 대조) ─────────────────────────────────────
_ANCHOR = {}


def anchor_value(arm, record, win):
    """동결 팔의 lead-0 값. ⛔앵커가 없는 팔은 `None`(= `G-ANCHOR-N` 이 요구하는 것)."""
    if arm not in G.ANCHORS:
        return None, False
    _path, farm, ffield, _mine = G.ANCHORS[arm]
    if not _ANCHOR:
        _ANCHOR.update(G.anchor_index())
    r = _ANCHOR.get((farm, record, win))
    return (None, True) if r is None else (r.get(ffield), True)


def anchor_bitmatch(arm, record, win, e_m0):
    v, has = anchor_value(arm, record, win)
    if not has:
        return None
    return bool(v is not None and v == e_m0)


# ── 행 조립 (★`run()` 과 자가진단이 **같은 함수**를 부른다 — 중복 구현 금지) ──
MEM_KEYS = ('e', 'md5', 'ens_k', 'bank_seeds', 'ana_k', 'vp_fired', 'vp_steps',
            'vp_k_shift', 'n_atoms', 'init_method', 'real_gate', 'real_estimator',
            'seed', 'dseed')
CELL_KEYS = ('e_med3', 'md5_med', 'spread_rms', 'zero_lead0', 'anchor_bitmatch',
             'device', 'sigma', 'npx', 'wall_s')


def build_row(arm, record, win, M, C):
    """★`G.ROW_CONTRACT` 의 35 키를 전건 채운다. ⛔멤버 값은 전부 **객체 되읽기**다."""
    off = list(G.ARMS[arm]['MEM'])
    return dict(
        batch=G.BATCH, arm=arm, record=record, win=win, approved=True,
        n_members=len(off), member_offsets=off,
        member_seed_used=[G.member_overrides(arm, o) for o in off],
        e_members=M['e'], e_m0=M['e'][0], e_med3=C['e_med3'],
        field_md5_members=M['md5'], field_md5_med=C['md5_med'],
        spread_rms=C['spread_rms'], zero_lead0=C['zero_lead0'],
        ens_k_readback=M['ens_k'], bank_seeds=M['bank_seeds'],
        analytic_ensemble_k_readback=M['ana_k'],
        vp_fired=M['vp_fired'], vp_steps_executed=M['vp_steps'],
        vp_k_shift=M['vp_k_shift'],
        n_atoms=M['n_atoms'], init_method_readback=M['init_method'],
        real_gate=M['real_gate'], real_estimator_readback=M['real_estimator'],
        seed_readback=M['seed'], diffusion_seed_readback=M['dseed'],
        deploy_pin_ok=True, anchor_bitmatch=C['anchor_bitmatch'],
        device=C['device'], cuda_available=bool(torch.cuda.is_available()),
        omp_threads=os.environ.get('OMP_NUM_THREADS'),
        sigma=C['sigma'], npx=C['npx'], wall_s=C['wall_s'])


# ── 한 셀 (= 한 행, 멤버 1~3 런) ────────────────────────────────────────────
def run(record, win, arm):
    import wavesplat.optimizer.learned_init as LI
    from wavesplat.optimizer.sequential_update import make_updater
    from wavesplat.run import DEFAULT_CONFIG, set_seed

    t0w = time.time()
    G.assert_shipping_defaults()          # ★트립와이어: 출하 기본값이 사전등록 뒤 움직였나

    # ① 장면 (⛔`with_future_obs` 불필요 — `update()` 0회)
    sc = scene_station(record, win=win, noise=G.NOISE)
    t_shift = float(np.asarray(sc['P'])[:, 2].min())
    t0 = float(sc['Pq'][0, 2]) - t_shift
    frames = MK.frames_from_scene(sc, t_shift, dtype=torch.float32)
    gxq, gyq = np.asarray(sc['gxq']), np.asarray(sc['gyq'])
    tx = torch.tensor(gxq, dtype=torch.float32)
    ty = torch.tensor(gyq, dtype=torch.float32)
    x_range = (float(gxq.min()), float(gxq.max()))
    y_range = (float(gyq.min()), float(gyq.max()))
    g = np.asarray(sc['maps_gt'][0], float)          # ★lead 0 **단독**
    mm = np.isfinite(g)
    if not mm.any():
        raise RuntimeError(f'{record}/w{win}: lead 0 의 GT 가 전부 NaN 이다')

    M = {k: [] for k in MEM_KEYS}
    fields = []

    # ── 계측 shim: 확산 시드 **되읽기**(⛔입력 dict 복사 아님 — 실제로 넘어간 인자) ──
    cap = {}
    orig_prop = LI.propose_atoms

    def wrapped_prop(*a, **kw):
        cap['dseed'] = kw.get('diffusion_seed', a[4] if len(a) > 4 else None)
        cap['calls'] = cap.get('calls', 0) + 1
        return orig_prop(*a, **kw)

    LI.propose_atoms = wrapped_prop
    try:
        for off in G.ARMS[arm]['MEM']:
            # ⛔팔 config·시드를 러너에서 다시 만들지 않는다 — `gates_t3` 가 단일출처다.
            cfg = dict(DEFAULT_CONFIG)
            cfg.update(depth=float(sc['depth']), device=G.DEPLOY_PIN['device'])
            cfg.update(G.arm_config(arm))
            cfg.update(G.member_overrides(arm, off))

            cap.clear()
            set_seed(int(cfg['seed']))
            upd = make_updater(cfg)
            res = upd.initialize(frames, x_range=x_range, y_range=y_range,
                                 verbose=False) or {}
            Z = upd.predict_field(tx, ty, t0).cpu().numpy().reshape(
                len(gyq), len(gxq))
            fields.append(Z)
            M['e'].append(score(Z, g, mm, sc['sg']))
            M['md5'].append(field_md5(Z))

            # ── 개입 되읽기 (⛔팔 정의가 아니라 **추정기 객체 / 반환 기록**에서) ──
            vp = res.get('varpro') or {}
            M['vp_fired'].append(bool(vp))
            M['vp_steps'].append(int(vp.get('vp_steps', 0) or 0))
            M['vp_k_shift'].append(vp.get('vp_k_shift'))
            bank = getattr(upd, '_bank', None)
            model = getattr(upd, 'model', None)
            M['ens_k'].append(res.get('bank_members'))
            M['bank_seeds'].append(list(getattr(upd, '_bank_seeds', []))
                                   if bank is not None else None)
            M['ana_k'].append(res.get('ensemble_k'))
            M['n_atoms'].append(int(getattr(model, 'n_gaussians', 0) or 0))
            M['init_method'].append(getattr(upd, 'init_method', None))
            rg = getattr(upd, '_auto_real_gate', None)
            M['real_gate'].append(bool(rg) if rg is not None else (bank is not None))
            M['real_estimator'].append(getattr(upd, 'real_estimator', None))
            M['seed'].append(getattr(upd, 'seed', None))
            M['dseed'].append(cap.get('dseed'))
    finally:
        LI.propose_atoms = orig_prop

    # ── ★멤버 장을 점별 median (하네스 소유 — 패키지는 learned 를 안 감싼다) ──
    if len(fields) > 1:
        Zmed = median_field(fields)
        e_med3 = score(Zmed, g, mm, sc['sg'])
        md5_med = field_md5(Zmed)
        # E3(서술 전용): 멤버 간 점별 분산의 RMS. ⚠모집단 분산(ddof=0), float64 누산.
        var = np.var(np.stack([np.asarray(f, np.float64) for f in fields], axis=0),
                     axis=0)
        spread = float(np.sqrt(np.mean(var[mm])) / sc['sg'])
    else:
        e_med3 = md5_med = spread = None

    C = dict(e_med3=e_med3, md5_med=md5_med, spread_rms=spread,
             zero_lead0=G.zero_lead0(record, win),
             anchor_bitmatch=anchor_bitmatch(arm, record, win, M['e'][0]),
             device=str(G.DEPLOY_PIN['device']), sigma=float(sc['sg']),
             npx=int(sc['npx']), wall_s=time.time() - t0w)
    row = build_row(arm, record, win, M, C)
    missing = [k for k in G.ROW_CONTRACT if k not in row]
    if missing:                    # ⛔조용한 결손 금지 — `G-0f` 가 잡기 전에 여기서 죽는다
        raise AssertionError(f'⛔행 계약 미충족: {missing}')
    return row


# ── 자가진단 (⛔동결 **전에** 통과시켜라) ───────────────────────────────────
def selftest():
    fails = []
    G.selftest()
    G.assert_shipping_defaults()

    # [1] ★승인 가드 항진 검사 — 네 조합 중 **정확히 하나**만 통과해야 한다
    truth = {(f, e): approved(['run'] + ([FLAG] if f else []), {ENV: '1'} if e else {})[2]
             for f in (False, True) for e in (False, True)}
    if sum(truth.values()) != 1 or not truth[(True, True)]:
        fails.append(f'승인 가드가 항진/불통이다: {truth}')

    # [2] 팔 정의 — ⛔빈 dict 금지 · 러너는 팔 config 를 **만들지 않고 부른다**
    for a in G.ARM_ORDER:
        if not G.arm_config(a):
            fails.append(f'{a}: 빈 dict 팔')
        mo = [G.member_overrides(a, o) for o in G.ARMS[a]['MEM']]
        if len(mo) != G.expected_members(a):
            fails.append(f'{a}: 멤버 수 불일치 {len(mo)}')
        if any('seed' not in m for m in mo):
            fails.append(f'{a}: 멤버 오버레이에 seed 가 없다')

    # [3] ★행 계약 — **러너의 조립기**가 35 키를 전건 내는가 (⛔픽스처가 아니라 이 코드)
    dummy_M = {k: [None] * 3 for k in MEM_KEYS}
    dummy_M['e'] = [1.0, 1.0, 1.0]
    dummy_C = {k: None for k in CELL_KEYS}
    row = build_row('PROP-V400M3', G.RECORDS[0], G.WINDOWS[0], dummy_M, dummy_C)
    if set(row) != set(G.ROW_CONTRACT):
        fails.append(f'행 계약 불일치: 빠짐={sorted(set(G.ROW_CONTRACT) - set(row))} '
                     f'초과={sorted(set(row) - set(G.ROW_CONTRACT))}')

    # [4] ★median 연산자가 `G.MEDIAN_RULE` 과 같은가 + **공허하지 않은가**(≠ 평균)
    rng = np.random.default_rng(0)
    F = [rng.standard_normal((7, 5)).astype(np.float32) for _ in range(3)]
    med = median_field(F)
    S = np.stack(F, axis=0)
    if not np.array_equal(med, np.median(S, axis=0).astype(np.float32)):
        fails.append('median 연산자가 `np.median(stack, axis=0)` 에서 이탈했다')
    if med.dtype != np.float32:
        fails.append(f'median dtype 이 float32 가 아니다: {med.dtype}')
    if np.array_equal(med, S.mean(axis=0).astype(np.float32)):
        fails.append('median 검사가 공허하다 — 평균과 구분되지 않는다')
    # ⛔k 가 홀수이므로 median 은 **어느 점에서든 멤버 값 중 하나**여야 한다
    if not np.all((med == F[0]) | (med == F[1]) | (med == F[2])):
        fails.append('median 이 멤버 값이 아닌 값을 냈다(보간 발생)')
    # ★거울 항등식의 산술적 근거: float32 반올림은 단조 ⟹ median 과 교환된다
    S64 = np.stack([f.astype(np.float64) for f in F], axis=0)
    if not np.array_equal(med, np.median(S64, axis=0).astype(np.float32)):
        fails.append('float32 median 이 float64 median 의 반올림과 다르다')

    # [5] ★앵커 자료가 36셀 전건 있는가 (⛔없으면 `G-ANCHOR` 가 구조적으로 실패한다)
    for arm in G.ANCHORS:
        miss = [(r, w) for r in G.RECORDS for w in G.WINDOWS
                if anchor_value(arm, r, w)[0] is None]
        if miss:
            fails.append(f'앵커 {arm}: {len(miss)} 셀 결손 {miss[:3]}')
        # ⛔공허 방지 — 앵커 값이 서로 달라야 비트 대조가 의미를 갖는다
        vals = {anchor_value(arm, r, w)[0] for r in G.RECORDS for w in G.WINDOWS}
        if len(vals) < G.N_PAIRS:
            fails.append(f'앵커 {arm}: 서로 다른 값이 {len(vals)}/{G.N_PAIRS}')
    if anchor_bitmatch('PROP-V0M3', G.RECORDS[0], G.WINDOWS[0], 1.0) is not None:
        fails.append('앵커 없는 팔이 None 을 싣지 않는다(`G-ANCHOR-N` 위반)')

    # [6] ⛔고정 기준상수 부재 — ★검사는 `gates_t3` 가 **단일출처**다(⛔여기서 다시 짜지 마라)
    ok6, det6 = G._no_flat_ref_constant()
    if not ok6:
        fails.append(f'배치가 고정 기준상수를 코드로 쓴다: {det6}')

    # [7] 작업 목록이 프로토콜 도출값과 같은가 (⛔"144"·"360" 을 손으로 적지 않는다)
    cs = cells()
    if len(cs) != G.expected_rows():
        fails.append(f'셀 수 {len(cs)} != {G.expected_rows()}')
    runs = sum(G.expected_members(a) for _, _, a in cs)
    if runs != G.expected_runs():
        fails.append(f'런 수 {runs} != {G.expected_runs()}')

    # [8] ★장치 정책 — 앵커 환경(`T2`)과 같은 스탬프인가
    t2cuda = sorted({r.get('cuda_available') for r in _ANCHOR.values()}) \
        if _ANCHOR else sorted({True})
    if not torch.cuda.is_available():
        fails.append('⛔CUDA 가 보이지 않는다 — `T2` 앵커 환경(cuda 가용)이 아니다. '
                     '`CUDA_VISIBLE_DEVICES` 를 확인하라(제안 경로가 0/12 로 갈린다).')
    if t2cuda != [True]:
        fails.append(f'앵커 자료의 cuda 스탬프가 [True] 가 아니다: {t2cuda}')
    if G.DEPLOY_PIN['device'] != 'cpu':
        fails.append(f'장치 핀이 cpu 가 아니다: {G.DEPLOY_PIN["device"]}')

    for f in fails:
        print('  FAIL', f)
    print(f'run_t3 selftest: {"PASS" if not fails else "FAIL"}  '
          f'guard={sum(truth.values())}/4 rows={len(cs)} runs={runs} '
          f'anchors={{{", ".join(f"{a}: {G.N_PAIRS}" for a in G.ANCHORS)}}} '
          f'vp={ {a: G.expected_vp_steps(a) for a in G.ARM_ORDER} } '
          f'cuda={torch.cuda.is_available()} device={G.DEPLOY_PIN["device"]}')
    return not fails


# ── main ───────────────────────────────────────────────────────────────────
def cells():
    return [(rec, win, arm) for rec in G.RECORDS for win in G.WINDOWS
            for arm in G.ARM_ORDER]


def main():
    argv = sys.argv[1:]
    mode = (argv or ['run'])[0]
    if mode == 'selftest':
        sys.exit(0 if selftest() else 1)

    has_flag, has_env, ok = approved(argv, os.environ)
    if not ok:
        print(ASK)
        print(f'\n⛔거부: 플래그={has_flag} 환경변수={has_env}')
        sys.exit(2)

    if mode == 'smoke':
        # ★스모크 = **네 팔 전부**(가장 비싼 `PROP-V400M3` 포함) · 결과 파일에 **안 쓴다**.
        #   셀 `BS13a`/창 1 = 이미 노출된 셀 ⟹ 새 셀 노출을 만들지 않는다.
        jobs = [(SMOKE_CELL[0], SMOKE_CELL[1], a) for a in G.ARM_ORDER]
        out_path = SMOKE_OUT
    else:
        shard, nshard = ((int(argv[1]), int(argv[2])) if len(argv) > 2
                         and argv[1].isdigit() else (0, 1))
        jobs = [c for i, c in enumerate(cells()) if i % nshard == shard]
        out_path = OUT

    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'error' not in r:
                done.add((r['record'], r['win'], r['arm']))
    t_start = time.time()
    with open(out_path, 'a') as fh:
        for i, (rec, win, arm) in enumerate(jobs, 1):
            if (rec, win, arm) in done:
                continue
            try:
                out = run(rec, win, arm)
            except Exception as exc:                                  # noqa: BLE001
                out = dict(batch=G.BATCH, arm=arm, record=rec, win=win,
                           error=repr(exc), tb=traceback.format_exc()[-700:])
            fh.write(json.dumps(out) + '\n'); fh.flush()
            el = time.time() - t_start
            if 'error' in out:
                print(f'[{i}/{len(jobs)}] {arm:12s} {rec:6s} w{win} '
                      f'⛔{str(out["error"])[:90]}', flush=True)
                continue
            med = ('   —   ' if out['e_med3'] is None else f'{out["e_med3"]:.5f}')
            spr = ('  —  ' if out['spread_rms'] is None
                   else f'{out["spread_rms"]:.4f}')
            print(f'[{i}/{len(jobs)}] {arm:12s} {rec:6s} w{win} '
                  f'm0={out["e_m0"]:.5f} med={med} sprd={spr} '
                  f'z={out["zero_lead0"]:.4f} vp={out["vp_steps_executed"]} '
                  f'N={out["n_atoms"]} anc={out["anchor_bitmatch"]} '
                  f'{out["wall_s"]:.1f}s · {out["wall_s"] / out["n_members"]:.1f}s/런 '
                  f'· ETA {el / i * (len(jobs) - i) / 60:.0f}m', flush=True)


if __name__ == '__main__':
    main()
