"""M3b 러너 — 수심을 **실제로 집행**한다. ⛔패키지 수정 0줄 (전부 in-process shim).

★팔 정의는 `B-12` 에서 **그대로 import** 한다(손복사 0). 이 파일이 덧붙이는 것은 셋뿐이다:
  ① **RT shim** — `wavesplat.data_loader.load_episode` 를 감싸 h5 참값 수심을 주입한다.
     ⛔`wavesim.data_writer.load_episode` 를 패치하면 **도달하지 않는다** —
       `wavesplat/data_loader.py:12` 가 `from … import load_episode` 로 **모듈 수준 바인딩**을
       떠 놓았기 때문이다(`PREREG.md` §10-1). 패치 지점은 **한 곳**이고 그것이 여기다.
     ★`RT50` = shim 을 **떼어낸다**(= `_ORIG` 그대로 = 오늘의 배포 거동).
       `RT12` = shim 을 **건다**. ⟹ `PC-A` 가 두 상태의 비트동일을 배치 안에서 다시 잰다.
  ② **조건자 재바인딩** — `CKPT50-RT12-C12` 팔만 `d3_beamform_net.{DEPTH,KX_G,KY_G,OM_G}` 를
     d=12 로 갈아 끼운다. ⛔의도표 금지 — `cond_readback()` 이 **`OM_G` 텐서에서 수심을 역산**해
     행에 싣는다(`G-14`).
  ③ **계측 shim** — `run_estimation` 반환 / `SequentialUpdater` **객체** / `run_letkf` 반환을
     행에 싣는다. ★ship 쪽 실효 수심(`obj_depth`)이 `PC-C`(`G-8`)의 유일한 근거다.

★**러너가 게이트의 모형이다**: `expected_jobs()` / `expect_counts()` / `row_identity()` 가
  단일출처이고, 분석기와 스모크가 **같은 함수**를 부른다(⛔게이트에 기대 개수를 손으로 적지 않는다).

★**스모크는 전 팔 × `DEP12` 를 덮는다**(`gates_m3b.SMOKE_PLAN`). ⛔`if not smoke:` 분기로
  게이트를 끄지 않는다 — M3 의 `G-17` 이 288런 전에 발화 불가였던 결함의 처방이다.

## ★★노출 공시 — 2026-08-30 하네스 스모크 (⛔결과 전에 적는다)
`SMOKE.md` §0 은 08-29 probe 가 `DEP12` **처치 팔의 `avg_nRMSE` 를 한 값도 보지 않았다**고
기록했다. ★**오늘의 하네스 스모크는 그것을 깬다** — 러너가 런마다 `nRMSE` 를 찍으므로
**에피소드 1개(`BF4_15comp_seed1695`) × 시드 42 × 전 5팔**의 값이 노출됐다(= 48짝 중 **1짝**).
⟹ ⛔**숨기지 않고 공시한다.** 남는 재량이 0 인 근거는 셋이다:
  ① 문턱·팔·읽기·분해·기술게이트는 **전부 `gates_m3b.py`** 에 있고 그 파일은 **08-29 동결**이다
     (`FROZEN_md5.txt` 첫 묶음, 오늘 한 줄도 고치지 않았다).
  ② `analyze_m3b.py` 는 **이 스모크보다 먼저 쓰였다**(검사기의 `analyzer<results` 가 검사한다).
  ③ 이 파일(`run_m3b.py`)은 판정을 낳지 않는다 — 행을 만들 뿐이고, 오늘 함께 동결된다.
⚠**그래도 이월되는 편향**: 나는 이제 그 1짝에서 `RT12` 가 `RT50` 보다 낮다는 것을 안다.
  ⟹ ⛔**단가·판정을 이 값으로 재해석하지 않는다.** 스모크 산출물은 삭제했다.

## ★단가 실측 (2026-08-30 스모크 5런, 1스레드 · `DEP12` 1에피소드 · 시드 42)
`CKPT50-RT50` 11.1 s · `CKPT50-RT12` 5.5 s · `CKPT50-RT12-C12` 5.7 s ·
`SPEC25-RT50` 50.5 s · `SPEC25-RT12` 49.3 s
⟹ `SMOKE.md` §2 의 보수적 상한(ship 12.9 s · spec 66.1 s)을 **밑돈다**(추정과 어긋나지 않는다).
본체 240런 직렬 ≈ **1.6 h**, 8샤드 ≈ **12–15 분**. ⛔시간은 어떤 판정에도 쓰지 않는다.

    python3 tmp/M3b_DEPTH_2026-08-29/run_m3b.py --selftest    # 게이트 자가진단
    python3 tmp/M3b_DEPTH_2026-08-29/run_m3b.py pcbref        # PC-B 동결 참조가 비지 않는지
    python3 tmp/M3b_DEPTH_2026-08-29/run_m3b.py pca           # PC-A 8런 ⛔실패 시 정지
    python3 tmp/M3b_DEPTH_2026-08-29/run_m3b.py smoke         # 전 팔 5런
    python3 tmp/M3b_DEPTH_2026-08-29/run_m3b.py run {i} 8     # ×8 샤드
"""
import os

# ★스레드 고정 — ⛔호출자에게 맡기지 않는다(`M7 E2` 가 부하 115/16코어, 한 런 11,800 s).
#   import 보다 **먼저** 세워야 BLAS 가 이 값을 읽는다.
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'

import glob                                                    # noqa: E402
import json                                                    # noqa: E402
import math                                                    # noqa: E402
import sys                                                     # noqa: E402
import time                                                    # noqa: E402
import traceback                                               # noqa: E402
import warnings                                                # noqa: E402

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for _p in (HERE, ROOT,
           os.path.join(ROOT, 'tmp', 'M3_SUPPORT_2026-08-28'),
           os.path.join(ROOT, 'tmp', 'B12_HOS_EXTEND_2026-08-23'),
           os.path.join(ROOT, 'tmp', 'B11_HOS_BASELINE_2026-08-23'),
           os.path.join(ROOT, 'tmp', 'S16_BASELINE_2026-08-14')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
_SCRIPTS = os.path.join(ROOT, 'scripts')          # ★뒤에 붙인다(이름 가림 최소화)
if _SCRIPTS not in sys.path:
    sys.path.append(_SCRIPTS)
os.chdir(ROOT)

import gates_m3b as G                                          # noqa: E402
import run_b12                                                 # noqa: E402  ★팔 정의
import run_s16                                                 # noqa: E402  ★기준선 정의

CAP = {}


# ═════════════════════════════════════════════════════════════════════════════
# 1. ★RT shim — **패치 지점은 `wavesplat.data_loader.load_episode` 하나다**
# ═════════════════════════════════════════════════════════════════════════════
_RT = {'orig': None, 'inject': None, 'mode': None}


def _install_rt():
    """원본을 붙잡고 주입 shim 을 만든다(설치는 `apply_rt` 가 한다). 멱등."""
    import wavesplat.data_loader as DL
    if _RT['orig'] is not None:
        return
    import h5py
    _RT['orig'] = DL.load_episode

    def _inject(path):
        d = _RT['orig'](path)
        with h5py.File(path, 'r') as f:
            d['depth'] = float(f['config'].attrs['depth'])     # ★h5 참값
        return d

    _RT['inject'] = _inject


def apply_rt(mode):
    """`RT50` ⟹ 원본 그대로(= 오늘의 배포). `RT12` ⟹ 참값 주입 shim."""
    _install_rt()
    import wavesplat.data_loader as DL
    DL.load_episode = _RT['orig'] if mode == G.RT50 else _RT['inject']
    _RT['mode'] = mode
    return mode


def rt_installed():
    import wavesplat.data_loader as DL
    return DL.load_episode is _RT['inject']


# ═════════════════════════════════════════════════════════════════════════════
# 2. ★조건자 — `beamform()` 이 `KX_G/KY_G/OM_G` 를 **호출 시점 모듈 전역**으로 조회한다
# ═════════════════════════════════════════════════════════════════════════════
_BN = {}


def _bn():
    if 'm' not in _BN:
        import d3_beamform_net as BN                           # noqa: N806
        _BN['m'] = BN
    return _BN['m']


def cond_readback():
    """⛔의도표 금지 — **`OM_G` 텐서에서** 조건자 수심을 역산한다.

    `om² = g·k·tanh(k·d)` ⟹ `d = atanh(om²/(g k))/k`. 최소 |k| 셀에서 푼다
    (`k=0.013469`: `tanh(kd)` 가 d=12 에서 0.160, d=50 에서 0.587 로 **양쪽 다 비포화**).
    실측 오차 ≈ 5e-6 (float32 격자) ⟹ 소수 3자리 반올림이 정확히 12.0 / 50.0 을 준다.
    """
    import torch
    BN = _bn()                                                 # noqa: N806
    k = torch.hypot(BN.KX_G.double(), BN.KY_G.double()).reshape(-1)
    om = BN.OM_G.double().reshape(-1)
    i = int(torch.argmin(k))
    kk = float(k[i])
    x = float(om[i] ** 2 / (BN.G * kk))
    return math.atanh(min(max(x, -1 + 1e-15), 1 - 1e-15)) / kk


def apply_cond(depth):
    """조건자 ω-격자를 `depth` 로 재바인딩(필요할 때만). 되읽은 실효 수심을 돌려준다."""
    BN = _bn()                                                 # noqa: N806
    d = float(depth)
    if float(BN.DEPTH) != d:
        BN.DEPTH = d
        BN.KX_G, BN.KY_G, BN.OM_G = BN.kgrid()
    return cond_readback()


# ═════════════════════════════════════════════════════════════════════════════
# 3. 계측 shim (M3/B-26 규약 — 값은 건드리지 않는다)
# ═════════════════════════════════════════════════════════════════════════════
def install_shims():
    import wavesplat.run as WR                                 # noqa: N806
    from wavesplat.optimizer import sequential_update as SU    # noqa: N806

    if not getattr(WR, '_m3b_shim', False):
        _orig_run = WR.run_estimation

        def _cap_run(*a, **kw):
            r = _orig_run(*a, **kw)
            CAP['ship'] = {f'cap_{k}': r.get(k) for k in G.SHIP_CAPTURE}
            return r
        WR.run_estimation = _cap_run
        WR._m3b_shim = True

    if not getattr(SU.SequentialUpdater, '_m3b_shim', False):
        _orig_init = SU.SequentialUpdater.__init__

        def _cap_init(self, config, *a, **kw):
            _orig_init(self, config, *a, **kw)
            # ★객체 되읽기 — ⛔입력 dict 복사(의도표) 금지(S13b 교훈).
            CAP['obj'] = {k: getattr(self, k, None) for k in G.OBJ_CAPTURE}
            CAP['obj_n'] = CAP.get('obj_n', 0) + 1
        SU.SequentialUpdater.__init__ = _cap_init
        SU.SequentialUpdater._m3b_shim = True

    if not getattr(run_s16, '_m3b_shim', False):
        _orig_letkf = run_s16.run_letkf

        def _cap_letkf(*a, **kw):
            row = _orig_letkf(*a, **kw)
            CAP['spec'] = {f'cap_{k}': row.get(k) for k in G.SPEC_CAPTURE}
            return row
        run_s16.run_letkf = _cap_letkf
        run_s16._m3b_shim = True


# ═════════════════════════════════════════════════════════════════════════════
# 4. ★작업 모형 — 러너가 단일출처. 게이트·분석기·스모크가 **이 함수들**을 부른다.
# ═════════════════════════════════════════════════════════════════════════════
def expected_jobs(smoke=False):
    """(에피소드 × 팔 × 시드) 전건. ★스모크도 **전 팔**을 덮는다(`SMOKE_PLAN`)."""
    eps = G.episodes()
    if smoke:
        eps = eps[:int(G.SMOKE_PLAN['n_episodes'])]
    jobs = []
    for ep in eps:
        for arm in G.ARM_ORDER:
            seeds = tuple(G.ARMS[arm]['seeds'])
            if smoke:
                seeds = seeds[:int(G.SMOKE_PLAN['n_seeds'])]
            for s in seeds:
                jobs.append(dict(ep=ep, arm=arm, seed=int(s)))
    return jobs


def expect_counts(jobs):
    """`gates_run(rows, expect=…)` 에 넣을 팔별 기대 행수. ⛔손으로 적지 않는다."""
    c = {a: 0 for a in G.ARM_ORDER}
    for j in jobs:
        c[j['arm']] += 1
    return c


def row_identity(r):
    """(팔, 에피소드, 시드) — 이어받기·중복 판정의 단일출처."""
    a = r.get('arm')
    fld = G.ARMS[a]['seed_field'] if a in G.ARMS else None
    s = (r.get(fld) if fld else
         (r.get('est_seed') if r.get('est_seed') is not None else r.get('filter_seed')))
    return (a, r.get('episode'), s)


def harness_matches_runner():
    """★변형 하네스는 **러너를 모형**해야 한다 — `gates_m3b._rows()` 가 만드는 합성 행이
    `expected_jobs()` 와 팔별 개수·행 정체성에서 일치하는지 기계로 확인한다.
    ⛔둘이 갈리면 자가진단은 통과하는데 본 배치가 다른 것을 돌린다(= 저장소의 반복 결함 부류).
    """
    jobs = expected_jobs()
    want = expect_counts(jobs)
    syn = G._rows()
    got = {a: 0 for a in G.ARM_ORDER}
    for r in syn:
        got[r['arm']] = got.get(r['arm'], 0) + 1
    ids = {row_identity(r) for r in syn}
    out = [('★T-R1 합성 하네스 팔별 행수 == expect_counts(expected_jobs())',
            got == want, f'{got} vs {want}'),
           ('★T-R2 row_identity 가 합성 행 전건을 유일하게 가른다',
            len(ids) == len(syn) == len(jobs), f'{len(ids)}/{len(syn)}/{len(jobs)}'),
           ('★T-R3 스모크 계획이 전 팔을 1행씩 덮는다',
            expect_counts(expected_jobs(smoke=True)) == {a: 1 for a in G.ARM_ORDER},
            str(expect_counts(expected_jobs(smoke=True))))]
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 5. 한 런
# ═════════════════════════════════════════════════════════════════════════════
def run_one(ep, arm, seed):
    d = G.ARMS[arm]
    apply_rt(d['rt'])
    if d['family'] == 'ship':
        cond_before = apply_cond(d['cond_depth'])
        CAP.pop('ship', None)
        CAP.pop('obj', None)
        CAP.pop('obj_n', None)
        row = run_b12.run_ship(ep, seed)                       # ★B-12 의 팔, 손복사 0
        row.update(CAP.get('ship', {}))
        row.update({f'obj_{k}': v for k, v in (CAP.get('obj') or {}).items()})
        row['obj_init_calls'] = CAP.get('obj_n')
        cond_after = cond_readback()
        if abs(cond_after - cond_before) > 1e-9:
            raise RuntimeError(f'조건자가 런 중에 움직였다 {cond_before} → {cond_after}')
        if abs(cond_after - float(d['cond_depth'])) > 1e-2:
            raise RuntimeError(f'조건자 되읽기 불일치 {cond_after} != {d["cond_depth"]}')
        row['cap_cond_depth'] = float(round(cond_after, 3))
        row['cap_cond_depth_raw'] = cond_after
    else:
        CAP.pop('spec', None)
        row = run_b12.run_spec(ep, seed)                       # ★S16 의 기준선, 손복사 0
        row.update(CAP.get('spec', {}))
    row['arm'] = arm            # ★팔 이름에 문자열 `SHIP` 0회 (`G-H`/`T13`)
    row['rt'] = d['rt']
    row['rt_shim_installed'] = bool(rt_installed())
    row['cell'] = ep.get('cell')
    row['ncomp'] = ep['ncomp']
    row['gen_seed'] = ep['seed']
    return row


# ═════════════════════════════════════════════════════════════════════════════
# 6. PC-A (비트불변 양성대조) — ⛔FAIL ⟹ BATCH-VOID
# ═════════════════════════════════════════════════════════════════════════════
PCA_PATH = os.path.join(HERE, 'pca_m3b.json')


def _pca_episode(name):
    cand = glob.glob(os.path.join(G.B26_DATA, '**', f'episode_{name}.h5'), recursive=True)
    assert cand, f'PC-A 에피소드 없음: {name}'
    return dict(path=sorted(cand)[0], episode=name, bf='SS-BF4',
                ncomp=(15 if '15comp' in name else 30), seed=1641, cell=None)


def pca():
    """참 수심 50 코퍼스에서 shim off/on 이 `avg_nRMSE` **비트 동일**임을 배치 안에서 잰다."""
    install_shims()
    rows = []
    for name in G.PCA_EPISODES:
        ep = _pca_episode(name)
        for mode, tag in ((G.RT50, 'off'), (G.RT12, 'on')):
            apply_rt(mode)
            apply_cond(50.0)
            CAP.pop('ship', None)
            CAP.pop('obj', None)
            r = run_b12.run_ship(ep, G.PCA_SEED)
            rows.append(dict(arm='CKPT50', episode=name, shim=tag,
                             avg_nRMSE=r['avg_nRMSE'], wall_s=r['wall_s'],
                             depth_readback=(CAP.get('obj') or {}).get('depth')))
            print(f'  PC-A CKPT50 {name[:34]:34s} shim={tag} '
                  f'nRMSE={r["avg_nRMSE"]!r} d={rows[-1]["depth_readback"]} '
                  f'{r["wall_s"]:.1f}s', flush=True)
            CAP.pop('spec', None)
            r2 = run_b12.run_spec(ep, G.PCA_SEED)
            rows.append(dict(arm='SPEC25', episode=name, shim=tag,
                             avg_nRMSE=r2['avg_nRMSE'], wall_s=r2['wall_s'],
                             depth_readback=(CAP.get('spec') or {}).get('cap_depth')))
            print(f'  PC-A SPEC25 {name[:34]:34s} shim={tag} '
                  f'nRMSE={r2["avg_nRMSE"]!r} d={rows[-1]["depth_readback"]} '
                  f'{r2["wall_s"]:.1f}s', flush=True)
    apply_rt(G.RT50)
    out = dict(rows=rows)
    with open(PCA_PATH, 'w') as fh:
        json.dump(out, fh, indent=1)
    ok = True
    for n_, q_, d_ in G.gates_pca(out):
        print(f'  [{"PASS" if q_ else "FAIL"}] {n_}: {d_}', flush=True)
        ok = ok and bool(q_)
    print('★PC-A ' + ('PASS' if ok else 'FAIL ⟹ ⛔BATCH-VOID (본 배치를 돌리지 마라)'),
          flush=True)
    return out, ok


# ═════════════════════════════════════════════════════════════════════════════
# 7. PC-B (M3 재현) — ⚠FAIL 은 VOID 가 **아니다**(`gates_m3b.READING_PCB`)
# ═════════════════════════════════════════════════════════════════════════════
#   M3 의 에피소드 이름은 `…_ra_rf_ds@DEP12` 로 셀 접미어를 달고 있고 M3b 는 달지 않는다.
#   ⟹ `@` 앞만 쓴다. 팔 이름도 다르므로 **계열(family)** 로 맞춘다.
M3_ARM_FAMILY = {'SHIP-NOW': 'ship', 'SPEC25': 'spec'}
PCB_ARM_FAMILY = {'CKPT50-RT50': 'ship', 'SPEC25-RT50': 'spec'}


def _ep_base(name):
    return str(name).split('@')[0]


def frozen_m3_dep12():
    """동결 M3 결과에서 `DEP12` 96행의 `avg_nRMSE` 를 키로 모은다(외부 동결 참조)."""
    fz = {}
    for p in sorted(glob.glob(G.M3_RESULTS_GLOB)):
        with open(p) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if 'error' in r or r.get('cell') != G.CELL:
                    continue
                fam = M3_ARM_FAMILY.get(r.get('arm'))
                if fam is None:
                    continue
                s = r.get('est_seed') if fam == 'ship' else r.get('filter_seed')
                fz[f'{fam}|{_ep_base(r["episode"])}|{s}'] = r['avg_nRMSE']
    return fz


def build_pcb(rows):
    """`RT50` 두 팔의 행을 동결 M3 행과 짝짓는다(추가 런 0회)."""
    out = []
    for r in rows:
        if 'error' in r:
            continue
        fam = PCB_ARM_FAMILY.get(r.get('arm'))
        if fam is None:
            continue
        s = r.get(G.ARMS[r['arm']]['seed_field'])
        out.append(dict(key=f'{fam}|{_ep_base(r["episode"])}|{s}', arm=r['arm'],
                        episode=r['episode'], avg_nRMSE=r['avg_nRMSE']))
    return dict(rows=out, frozen=frozen_m3_dep12())


# ═════════════════════════════════════════════════════════════════════════════
# 8. main
# ═════════════════════════════════════════════════════════════════════════════
def _load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if 'error' not in r:
                    done.add(row_identity(r))
    return done


def _selftest_all():
    """게이트 자가진단 + ★러너-하네스 정합. 어느 하나라도 실패하면 추정을 돌리지 않는다."""
    ok, _f = G.selftest()
    for n_, q_, d_ in harness_matches_runner():
        print(f'  [{"ok" if q_ else "XX"}] {n_}: {d_}', flush=True)
        ok = ok and bool(q_)
    return ok


def main():
    if '--selftest' in sys.argv:
        sys.exit(0 if _selftest_all() else 1)
    which = (sys.argv[1:] or ['run'])[0]

    if which == 'pcbref':
        # ★외부 동결 자료를 참조하는 게이트는 **실제 파일에 대고** 돌려 ref 가 비지 않음을 확인.
        fz = frozen_m3_dep12()
        want = 2 * G.N_EP * len(G.EST_SEEDS)
        print(f'PC-B 동결 참조: {len(fz)}/{want}행 '
              f'(예: {sorted(fz)[0] if fz else "없음"})')
        sys.exit(0 if len(fz) == want else 1)

    print('★런 전 게이트 자가진단 ─────────────────────────────', flush=True)
    ok = _selftest_all()
    if not ok:
        print('⛔자가진단 실패 — 러너를 돌리지 않는다.', flush=True)
        sys.exit(1)

    # ★데이터 게이트 — ⛔**스모크에서도, PC-A 에서도 끄지 않는다**
    #   (M3 의 `if not smoke:` 분기가 `gates_data()` 를 통째로 건너뛴 결함의 처방).
    print('★데이터 게이트 ───────────────────────────────────', flush=True)
    gd = G.gates_data()
    for n_, q_, d_ in gd:
        print(f'  [{"PASS" if q_ else "FAIL"}] {n_}: {d_}', flush=True)
    if not all(q_ for _, q_, _ in gd):
        print('⛔데이터 게이트 실패 — 추정을 돌리지 않는다.', flush=True)
        sys.exit(1)

    if which == 'pca':
        _out, pass_ = pca()
        sys.exit(0 if pass_ else 1)

    smoke = which == 'smoke'
    shard, nshard = ((int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3
                     else (0, 1))

    install_shims()
    jobs = expected_jobs(smoke=smoke)
    mine = jobs if smoke else jobs[shard::nshard]
    out_path = os.path.join(HERE, 'results_m3b_smoke.jsonl' if smoke
                            else f'results_m3b_{shard}.jsonl')
    done = _load_done(out_path)
    print(f'=== M3b {which} shard {shard}/{nshard}: 작업 {len(mine)}/{len(jobs)} '
          f'· 이미 완료 {len(done)}', flush=True)

    fresh = []
    with open(out_path, 'a') as fh:
        for j in mine:
            ep, arm, s = j['ep'], j['arm'], j['seed']
            if (arm, ep['episode'], s) in done:
                continue
            t0 = time.time()
            try:
                row = run_one(ep, arm, s)
            except Exception as e:                             # noqa: BLE001
                row = dict(arm=arm, episode=ep['episode'], cell=ep.get('cell'),
                           rt=G.ARMS[arm]['rt'], error=repr(e),
                           tb=traceback.format_exc()[-600:], wall_s=time.time() - t0)
                row[G.ARMS[arm]['seed_field']] = s
            fh.write(json.dumps(row) + '\n')
            fh.flush()
            fresh.append(row)
            if 'error' in row:
                print(f'  {arm:16s} {ep["episode"][:36]:36s} s{s} '
                      f'⛔{str(row["error"])[:70]}', flush=True)
            else:
                d_eff = (row.get('obj_depth') if G.ARMS[arm]['family'] == 'ship'
                         else row.get('cap_depth'))
                print(f'  {arm:16s} {ep["episode"][:36]:36s} s{s} '
                      f'nRMSE={row["avg_nRMSE"]:.4f} d={d_eff} '
                      f'{row["wall_s"]:.1f}s', flush=True)

    if smoke:
        # ★스모크에서도 **모든 런 게이트가 돈다** — 기대 개수만 매개변수화한다.
        rows = []
        with open(out_path) as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        print('★스모크 런 게이트 (expect = expect_counts(expected_jobs(smoke=True))) ───',
              flush=True)
        exp = expect_counts(jobs)
        n_ok = 0
        gr = G.gates_run(rows, expect=exp)
        for n_, q_, d_ in gr:
            print(f'  [{"PASS" if q_ else "FAIL"}] {n_}: {d_}', flush=True)
            n_ok += bool(q_)
        print(f'  스모크 게이트 {n_ok}/{len(gr)}', flush=True)
        w = [r['wall_s'] for r in fresh if 'error' not in r]
        if w:
            per = {}
            for r in fresh:
                if 'error' not in r:
                    per.setdefault(r['arm'], []).append(r['wall_s'])
            print('★단가(스모크 실측): '
                  + ' · '.join(f'{a} {sum(v)/len(v):.1f}s' for a, v in per.items()),
                  flush=True)
        print(f'⚠스모크 산출물 = {out_path} — ⛔판정에 쓰지 마라(삭제 대상).', flush=True)


if __name__ == '__main__':
    main()
