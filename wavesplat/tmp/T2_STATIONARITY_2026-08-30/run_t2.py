"""T2-STAT 러너 — ★**제안 표현에 「재적합」을 이식한다**.

판정은 `gates_t2.py` 에 **결과 0행에서 동결**돼 있다. 이 파일은 자료를 만들 뿐이다.

프로토콜(⛔`B-3`·`B-7b` 와 글자 그대로 같다 — 그래서 앵커가 **비트**로 성립한다):
lead ℓ 마다 **① 먼저 예측·채점 → ② 그 다음 그 시각의 관측 투입**. 절대 뒤집지 마라.

★이 배치가 새로 하는 것은 하나뿐이다 — `PROP-REFIT2` 팔에서 하네스가 **bank 의 거울**을 돈다:

    ① 관측을 롤링 창(`MEMORY_S=40`)에 넣고 오래된 것을 버린다      (== `_bank_update:1303-1307`)
    ② `t - t_last_fit >= PROP_REFIT_S(2.0)` 이면 그 창으로 재적합    (== `_bank_update:1308-1320`)
    ③ 창은 `OBS_CAP=600` 으로 **같은 `default_rng(0)` 추출**로 자른다 (== `_bank_fit_window:1179-1181`)

⛔**①의 창 조립은 손으로 다시 짜지 않는다** — 패키지의 `_bank_obs_from_frames` 를
**그대로 호출**한다(같은 기능을 두 곳에 구현해 한쪽만 계측하는 것이 이 저장소의 반복 결함).
★그리고 ③의 **정직성은 `G-MIRROR` 가 지킨다**: `SHIP-REFIT2` 쪽 해시는 **패키지가 실제로
`fit_bank_mmse` 에 넘긴 배열**에서 뜬다(하네스 헬퍼가 아니다) ⟹ 하네스가 패키지 규약에서
이탈하면 해시가 갈려 **게이트가 잡는다**.

⚠**재적합 시 전역 RNG 를 다시 seed 하지 않는다** — bank 도 하지 않는다(`_bank_seeds` 고정).
그 결과 `PROP-REFIT2` 는 `PROP-FROZEN` 보다 전역 draw 를 더 소모하는데, 그 노출은
**처치 안에** 있고 팔 사이 교락이 아니다. 증명은 `G-3`(lead 0–2 비트 동일)이다.

★승인 가드는 **이중**(플래그+환경변수)이고 자가진단이 **항진 검사**를 한다(선례 `M7 E1`).

    python3 tmp/T2_STATIONARITY_2026-08-30/run_t2.py selftest
    T2_APPROVED=1 python3 …/run_t2.py smoke --approved
    T2_APPROVED=1 python3 …/run_t2.py run <shard> <nshard> --approved
"""
import os

# ★스레드는 러너가 **스스로** 고정한다(⛔호출자에게 맡기지 마라 — M7 E2 가 18시간을 잃었다).
#   ⚠`import torch` **앞**이어야 효력이 있다.
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

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
           os.path.join(ROOT, 'tmp', 'B7_PROP_TRACK_2026-08-23'),
           os.path.join(ROOT, 'tmp', 'bankmmse_port_2026-07-29')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(ROOT)

import gates_t2 as G                                                # noqa: E402
from scene_station import scene_station                             # noqa: E402
import make_gif_bank_deployed as M                                  # noqa: E402

OUT = os.path.join(HERE, 'results_t2.jsonl')
SMOKE_OUT = os.path.join(HERE, 'smoke_t2.jsonl')

# ── ★승인 가드 (선례 `M7 E1`) ───────────────────────────────────────────────
ENV = 'T2_APPROVED'
FLAG = '--approved'
ASK = ('`T2-STAT` 의 PROP 두 팔은 실측 12 레코드에 **learned 를 강제**합니다\n'
       '(`real_domain_gate=False`). 상시 비승인 정책이고 `W12`·`B-2`·`M7 E1` 때 각각\n'
       f'명시 승인으로 풀렸습니다. 사전 기대는 UNDECIDED 쪽, MDE q80={G.mde()["q80"]}.\n'
       f'승인 근거는 `APPROVAL_T2_2026-08-30.md` 에 축자로 있습니다.\n'
       f'⟹ `{ENV}=1 … run_t2.py … {FLAG}` 로 **둘 다** 주어야 돕니다.')


def approved(argv, env):
    """★순함수. ⚠가드도 *"항진하지 않는가"* 를 시험해야 한다 — 항상 거부/항상 통과하는
    가드는 그 자체로 결함이다. `selftest()` 가 네 조합을 전부 친다."""
    has_flag = FLAG in list(argv)
    has_env = env.get(ENV) == '1'
    return has_flag, has_env, (has_flag and has_env)


# ── 창 규약 ────────────────────────────────────────────────────────────────
def _cap(P, z, cap=G.OBS_CAP):
    """== `_bank_fit_window:1179-1181` 의 캡. ⚠하네스 쪽 구현이고, 패키지와 어긋나면
    `G-MIRROR` 가 잡는다(SHIP 해시는 **패키지가 실제로 넘긴 배열**에서 뜬다)."""
    if len(z) > cap:
        ii = np.random.default_rng(0).choice(len(z), cap, replace=False)
        return P[ii], z[ii]
    return P, z


def _hash(P, z):
    return hashlib.md5(np.ascontiguousarray(np.asarray(P, float)).tobytes()
                       + np.ascontiguousarray(np.asarray(z, float)).tobytes()).hexdigest()


def _frames_from_window(P, z, dtype=torch.float32):
    """(P, z) → 시간-major FrameData 목록. == `frames_from_scene:71` 의 grouping 규약."""
    from wavesplat.data_loader import FrameData
    out = []
    for tt in np.unique(P[:, 2]):
        m = P[:, 2] == tt
        n = int(m.sum())
        out.append(FrameData(t=float(tt),
                             x_obs=torch.tensor(P[m, 0], dtype=dtype),
                             y_obs=torch.tensor(P[m, 1], dtype=dtype),
                             z_obs=torch.tensor(z[m], dtype=dtype),
                             confidence=torch.ones(n, dtype=dtype),
                             incidence_angles=torch.zeros(n, dtype=dtype)))
    return out


# ── 앵커 (동결 배치와의 **비트** 대조) ──────────────────────────────────────
_ANCHOR_CACHE = {}


def anchor_rows(arm):
    if arm not in G.ANCHORS:
        return None
    path, cell = G.ANCHORS[arm]
    if path not in _ANCHOR_CACHE:
        rows = []
        with open(path) as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        _ANCHOR_CACHE[path] = rows
    return {(r['record'], r['win']): r for r in _ANCHOR_CACHE[path]
            if r.get('cell') == cell and 'error' not in r}


def anchor_bitmatch(arm, record, win, per_lead, fc_mean):
    a = anchor_rows(arm)
    if a is None:
        return None                       # ⛔`PROP-REFIT2` 에는 앵커가 없다 — 신규 칸이다
    r = a.get((record, win))
    if r is None:
        return False
    return bool(r.get('per_lead') == per_lead and r.get('fc_mean') == fc_mean)


# ── 한 셀 ──────────────────────────────────────────────────────────────────
def run(record, win, arm):
    from wavesplat.data_loader import FrameData
    from wavesplat.optimizer.sequential_update import make_updater
    import wavesplat.optimizer.sequential_update as SU
    import wavesplat.optimizer.bank_mmse as BM
    from wavesplat.run import DEFAULT_CONFIG, set_seed

    t0w = time.time()
    G.assert_shipping_defaults()                      # ★트립와이어: 출하 기본값이 움직였나
    sc = scene_station(record, win=win, noise=G.NOISE, with_future_obs=True)
    t_shift = float(np.asarray(sc['P'])[:, 2].min())
    t0 = float(sc['Pq'][0, 2]) - t_shift
    frames = M.frames_from_scene(sc, t_shift, dtype=torch.float32)

    cfg = dict(DEFAULT_CONFIG)
    cfg.update(depth=float(sc['depth']), seed=G.SEED, device='cpu')
    cfg.update(G.ARMS[arm]['EST'])                    # ⛔빈 dict 금지 — 모든 키 명시
    prop_refit_s = G.ARMS[arm]['HARN']['prop_refit_s']

    # ── 계측 shim ───────────────────────────────────────────────────────────
    fitwin, bank_hashes, bank_n = [], [], []          # bank 재적합(가지 단위)
    orig_fit = SU.SequentialUpdater._bank_fit_window
    orig_bmm = BM.fit_bank_mmse
    _seen = {'h': None}

    def wrapped_fit(self, *a, **kw):
        fitwin.append(1)
        _seen['h'] = None                             # 이 재적합의 첫 fit_bank_mmse 만 해시
        return orig_fit(self, *a, **kw)

    def wrapped_bmm(P, z, *a, **kw):
        # ★해시는 **패키지가 실제로 넘긴(캡 적용 후) 배열**에서 뜬다 — 하네스 헬퍼가 아니다.
        if _seen['h'] is None:
            _seen['h'] = _hash(P, z)
            bank_hashes.append(_seen['h'])
            bank_n.append(int(len(z)))
        return orig_bmm(P, z, *a, **kw)

    SU.SequentialUpdater._bank_fit_window = wrapped_fit
    BM.fit_bank_mmse = wrapped_bmm
    try:
        set_seed(G.SEED)
        upd = make_updater(cfg)
        x_range = (float(sc['gxq'].min()), float(sc['gxq'].max()))
        y_range = (float(sc['gyq'].min()), float(sc['gyq'].max()))
        upd.initialize(frames, x_range=x_range, y_range=y_range, verbose=False)
        n_init_fit = len(fitwin)                      # 초기화 중 호출은 **개입이 아니다**
        bank_hashes.clear(); bank_n.clear()           # 초기 적합 해시도 개입이 아니다

        # ★제안 쪽 롤링 창 — 조립은 **패키지 코드**가 한다(⛔중복 구현 금지)
        Pw, zw = SU.SequentialUpdater._bank_obs_from_frames(frames)
        t_last_fit = float(np.max(Pw[:, 2]))          # == `_bank_update` 의 `_bank_last_fit_t:1145`
        reinit, prop_hashes, prop_n = 0, [], []

        gxq, gyq = np.asarray(sc['gxq']), np.asarray(sc['gyq'])
        tx = torch.tensor(gxq, dtype=torch.float32)
        ty = torch.tensor(gyq, dtype=torch.float32)
        Pf = np.asarray(sc['P_fut'], float)
        zf = np.asarray(sc['z_fut'], float)
        per_lead = {}
        for li, l_ in enumerate(sc['leads']):
            # ① 먼저 예측·채점
            g = np.asarray(sc['maps_gt'][li], float)
            mm = np.isfinite(g)
            if mm.sum() > 0:
                Z = upd.predict_field(tx, ty, t0 + float(l_)).cpu().numpy().reshape(
                    len(gyq), len(gxq))
                per_lead[float(l_)] = float(np.sqrt(np.mean((Z[mm] - g[mm]) ** 2)) / sc['sg'])
            # ② 그 다음 그 시각 관측 투입
            if li == 0:
                continue
            tt = t0 + float(l_) + t_shift
            m = np.abs(Pf[:, 2] - tt) < 1e-6
            if m.sum() == 0:
                continue
            nq = int(m.sum())
            fr = FrameData(t=float(l_) + t0,
                           x_obs=torch.tensor(Pf[m, 0], dtype=torch.float32),
                           y_obs=torch.tensor(Pf[m, 1], dtype=torch.float32),
                           z_obs=torch.tensor(zf[m], dtype=torch.float32),
                           confidence=torch.ones(nq, dtype=torch.float32),
                           incidence_angles=torch.zeros(nq, dtype=torch.float32))
            upd.update(fr)

            # ── ★하네스 거울 (`PROP-REFIT2` 에서만 발화) ──────────────────
            Pn, zn = SU.SequentialUpdater._bank_obs_from_frames([fr])
            if len(zn):
                Pw = np.vstack([Pw, Pn]); zw = np.concatenate([zw, zn])
            keep = Pw[:, 2] >= float(fr.t) - G.MEMORY_S     # == `:1306`
            Pw, zw = Pw[keep], zw[keep]
            if prop_refit_s is not None and float(fr.t) - t_last_fit >= float(prop_refit_s):
                Pc, zc = _cap(Pw, zw)
                prop_hashes.append(_hash(Pc, zc)); prop_n.append(int(len(zc)))
                upd.initialize(_frames_from_window(Pc, zc),
                               x_range=x_range, y_range=y_range, verbose=False)
                reinit += 1
                t_last_fit = float(fr.t)                     # == `:1320`

        fcv = [v for k, v in per_lead.items() if G.LEAD_FC[0] <= k <= G.LEAD_FC[1]]
        pl = {str(k): v for k, v in sorted(per_lead.items())}
        fc_mean = float(np.mean(fcv)) if fcv else None
        rg = getattr(upd, '_auto_real_gate', None)
        is_bank = getattr(upd, '_bank', None) is not None
        refit_calls = len(fitwin) - n_init_fit
        return dict(
            arm=arm, cell=arm, record=record, win=win,
            lead0=per_lead.get(0.0), fc_mean=fc_mean, per_lead=pl,
            # ★개입 계수 — **가지** 단위, 팔마다 하나씩(⛔함수 단위로는 못 잡는다)
            refit_calls=refit_calls, reinit_calls=reinit, init_fit_calls=n_init_fit,
            refit_window_hashes=(bank_hashes if refit_calls else prop_hashes) or [],
            refit_window_n=(bank_n if refit_calls else prop_n) or [],
            # ★경로 되읽기 — ⛔입력 dict 가 아니라 **추정기 객체**에서 읽는다
            n_atoms=int(getattr(getattr(upd, 'model', None), 'n_gaussians', 0) or 0),
            init_method_readback=getattr(upd, 'init_method', None),
            real_gate=(bool(rg) if rg is not None else is_bank),
            refit_readback=getattr(upd, '_bank_refit_every_s', None),
            prop_refit_readback=prop_refit_s,
            deploy_pin_ok=True,
            anchor_bitmatch=anchor_bitmatch(arm, record, win, pl, fc_mean),
            device=str(cfg['device']), cuda_available=bool(torch.cuda.is_available()),
            omp_threads=os.environ.get('OMP_NUM_THREADS'),
            approved=True, n_fut_obs=int(len(zf)), sigma=float(sc['sg']),
            wall_s=time.time() - t0w)
    finally:
        SU.SequentialUpdater._bank_fit_window = orig_fit
        BM.fit_bank_mmse = orig_bmm


# ── 자가진단 (⛔동결 **전에** 통과시켜라) ───────────────────────────────────
def selftest():
    fails = []
    G.selftest()
    G.assert_shipping_defaults()

    # [1] 승인 가드 항진 검사 — 네 조합 중 **정확히 하나**만 통과해야 한다
    truth = {(f, e): approved(['run'] + ([FLAG] if f else []), {ENV: '1'} if e else {})[2]
             for f in (False, True) for e in (False, True)}
    if sum(truth.values()) != 1 or not truth[(True, True)]:
        fails.append(f'승인 가드가 항진/불통이다: {truth}')

    # [2] 팔 정의에 빈 dict 0건 + 하네스/추정기 키 분리
    for a in G.ARM_ORDER:
        if not G.ARMS[a]['EST']:
            fails.append(f'{a}: 빈 dict 팔')
        if 'prop_refit_s' in G.ARMS[a]['EST']:
            fails.append(f'{a}: 하네스 키가 추정기 config 로 샜다')

    # [3] ★재적합 스케줄러가 프로토콜 위에서 정말 7회 발화하는가 — 러너 코드로 재현
    last, n = 0.0, 0
    for l_ in G.LEADS_OBS:
        if l_ - last >= G.PROP_REFIT_S:
            n += 1; last = l_
    if n != G.N_REFIT_EXPECTED:
        fails.append(f'스케줄러 {n} != {G.N_REFIT_EXPECTED}')

    # [4] ★캡 헬퍼가 패키지 규약과 같은가 (같은 seed·같은 추출)
    rng = np.random.default_rng(0)
    P = np.arange(700 * 3, dtype=float).reshape(700, 3); z = np.arange(700, dtype=float)
    ii = rng.choice(700, G.OBS_CAP, replace=False)
    Pc, zc = _cap(P, z)
    if not (np.array_equal(Pc, P[ii]) and np.array_equal(zc, z[ii])):
        fails.append('캡 헬퍼가 `default_rng(0).choice` 규약에서 이탈했다')
    if len(_cap(P[:100], z[:100])[1]) != 100:
        fails.append('캡이 미달 창을 건드렸다')

    # [5] 창 재조립이 시간-major 순서를 지키는가
    Pw = np.array([[1., 1., 5.], [2., 2., 3.], [3., 3., 5.], [4., 4., 3.]])
    zw = np.array([1., 2., 3., 4.])
    fr = _frames_from_window(Pw, zw)
    if [f.t for f in fr] != [3.0, 5.0] or [len(f.z_obs) for f in fr] != [2, 2]:
        fails.append('창 재조립이 시간-major 가 아니다')

    # [6] 앵커 파일이 36셀 전건 있는가 (⛔없으면 `G-ANCHOR` 가 구조적으로 실패한다)
    for a in G.ANCHORS:
        rr = anchor_rows(a)
        if len(rr) != G.N_PAIRS:
            fails.append(f'앵커 {a}: {len(rr)}/{G.N_PAIRS} 셀')

    # [7] ⛔고정 기준상수 부재 — ★검사는 `gates_t2` 가 **단일출처**다(⛔여기서 다시 짜지 마라).
    #     2026-08-30: 원래 구현이 문자열 검색이라 **금지문 자신에 걸려 구조적으로 항상 실패**했다.
    #     동결 전에 `ast` 판정으로 교체했다 — `AMENDMENT_0_PREFREEZE.md`.
    ok7, det7 = G._no_flat_ref_constant()
    if not ok7:
        fails.append(f'배치가 고정 기준상수를 코드로 쓴다: {det7}')

    for f in fails:
        print('  FAIL', f)
    print(f'run_t2 selftest: {"PASS" if not fails else "FAIL"}  '
          f'guard={sum(truth.values())}/4 refits={G.N_REFIT_EXPECTED} '
          f'anchors={ {a: len(anchor_rows(a)) for a in G.ANCHORS} }')
    return not fails


# ── main ───────────────────────────────────────────────────────────────────
def cells():
    return [(rec, win, arm) for rec in G.RECORDS for win in G.WINDOWS for arm in G.ARM_ORDER]


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
        # ★스모크 = 가장 비싼 팔(`PROP-REFIT2`) + 게이트가 발화할 셀. 결과 파일에 **안 쓴다**.
        #  셀 `BS13a`/창 1 = `B-3 PREREG` 가 이미 n=1 엿보기를 공시한 셀.
        jobs = [('BS13a', 1, a) for a in G.ARM_ORDER]
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
                out = dict(arm=arm, cell=arm, record=rec, win=win, error=repr(exc),
                           tb=traceback.format_exc()[-700:])
            fh.write(json.dumps(out) + '\n'); fh.flush()
            el = time.time() - t_start
            print(f'[{i}/{len(jobs)}] {arm:12s} {rec:6s} w{win} ' + (
                f'lead0={out["lead0"]:.4f} fc={out["fc_mean"]:.4f} '
                f'refit={out["refit_calls"]} reinit={out["reinit_calls"]} '
                f'N={out["n_atoms"]} anchor={out["anchor_bitmatch"]} '
                f'{out["wall_s"]:.1f}s · ETA {el/i*(len(jobs)-i)/60:.0f}m'
                if 'error' not in out else str(out['error'])[:90]), flush=True)


if __name__ == '__main__':
    main()
