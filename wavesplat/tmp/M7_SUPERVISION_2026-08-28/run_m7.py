"""M7 러너 — E1(실측 12 레코드) · E2(합성 헤드라인 회귀 + `S-1`) · E3(`M1` 자료 재사용, 보고 전용).

⛔**규칙은 `gates_m7.py` 가 유일하게 소유한다**(팔·문턱·검정·해석). 작업 목록과 행 정체성은
`jobs_m7.py` 가 유일하게 소유하고 **이 러너와 `analyze_m7.py` 가 같은 함수를 부른다**.

## 팔을 어떻게 정의하는가 — ⛔**빈 dict 로 정의하지 않는다**(`gates_b7.py:46` 전례)
* 학습 5칸 + `SHIPPED` : **`learned_ckpt` 한 키**로 갈린다(`PREREG §1.1` 축자).
  E1 은 `{'real_domain_gate': False, 'init_method': 'learned', 'learned_ckpt': …,
  'learned_diffusion_seed': …}`, E2/E3 는 배포 진입점의 config 에 같은 키를 **투명 shim** 으로
  넣는다(`_ARM['ckpt'] is None` 이면 원본 인자를 **그대로** 넘겨 비트 동일).
* `BANK` : 배포 실측 경로. **키를 명시**한다(`real_domain_gate=True · init_method='auto' ·
  real_estimator='bank_mmse'` = 출하 기본값과 같은 값이지만 **행에 남는다**).
* `SPEC25` : `S16` 정의(`run_s16.run_letkf`)를 **그대로 import** — 손복사 0.
* `ZERO` : ⛔**런 0회**. `B-12` 규약대로 `SPEC25` 행의 `zero_nRMSE` 열에서 분석기가 **합성**한다.

## ★"팔이 정말 그 넷을 썼나" 는 **되읽기**로만 인정한다 (`G-4`)
`learned_init.propose_atoms` 에 **투명 shim**(값 무수정, 인자 포착)을 걸어 **실제로 열린 파일**의
경로와 sha256 을 행에 싣는다. 입력 dict 복사(=의도표)를 쓰지 않는다 — `S13b` 교훈.
같은 이유로 `SequentialUpdater.__init__` 에서 배포 상수를 **객체에서** 되읽는다.

## ⛔디바이스 — `CUDA_VISIBLE_DEVICES` 를 세우지 않는다 (`AMENDMENT_1 §5.2`)
`learned_init.py:133-134` 가 스스로 cuda 를 고르고 `config['device']` 를 읽지 않는다. E1 계보
(`B-2 PROP`)와 E2 계보(`B-12`)가 **GPU 가시 상태**에서 측정됐다. 행에 `cuda_available` 을 남긴다.

## 실행
    python3 tmp/M7_SUPERVISION_2026-08-28/run_m7.py --selftest
    python3 tmp/M7_SUPERVISION_2026-08-28/run_m7.py e2 <shard> 8      # ⛔8 을 바꾸지 마라
    python3 tmp/M7_SUPERVISION_2026-08-28/run_m7.py e3 <shard> 8      # ⛔M1 판정 동결 후
    M7_E1_APPROVED=1 python3 …/run_m7.py e1 --approved                # ★사용자 승인 필수
"""
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
import warnings

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import jobs_m7 as J                                                     # noqa: E402
import gates_m7 as G                                                    # noqa: E402

os.chdir(ROOT)

# ── ★E1 승인 가드 (PREREG §3 E1 · AMENDMENT_1 §0 · §7 Step 5) ─────────────────
E1_ENV = 'M7_E1_APPROVED'
E1_FLAG = '--approved'
E1_ASK = (
    'M7 의 E1 은 실측 12 레코드에 **learned 를 강제**합니다(`real_domain_gate=False`).\n'
    '이는 비승인 정책이고 `W12`·`B-2` 때 각각 승인으로 풀렸습니다. 사전 기대는 **널**이며\n'
    f'(`B-8b`), MDE 는 12개 중 {G.mde()["min_wins"]}개 일치입니다. 돌릴까요?')


def e1_approved(argv, env):
    """★E1 승인 술어 — **순함수**로 떼어 놓는다.

    ⚠가드도 *"항진하지 않는가"* 를 시험해야 한다: 네 조합 중 **정확히 하나**만 참이어야
    하고(항상 거부하는 가드는 그 자체로 결함이다) 자가진단이 그것을 검사한다.
    """
    has_flag = E1_FLAG in list(argv)
    has_env = env.get(E1_ENV) == '1'
    return has_flag, has_env, (has_flag and has_env)

# ── 포착 상태 (투명 shim 이 채운다) ───────────────────────────────────────────
CAP = {}
_ARM = {'ckpt': None}
_SHA = {}


def sha256_of(path):
    if path in _SHA:
        return _SHA[path]
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    _SHA[path] = h.hexdigest()
    return _SHA[path]


def install_shims():
    """★전부 **투명**하다 — 값을 바꾸는 것은 `run_estimation` 의 `learned_ckpt` 주입 하나뿐이고
    그것은 `_ARM['ckpt'] is None` 일 때 **인자를 그대로 넘긴다**(비트 동일)."""
    import wavesplat.run as WR
    from wavesplat.model.spectral_gaussian import SpectralGaussianModel as SGM
    from wavesplat.optimizer import learned_init as LI
    from wavesplat.optimizer import sequential_update as SU
    from wavesplat.optimizer import varpro as VP

    if not getattr(LI, '_m7_shim', False):
        _orig = LI.propose_atoms

        def _cap_propose(frames, depth, ckpt=LI.DEFAULT_CKPT, *a, **kw):
            # ★값 무수정. 실제로 열리는 파일을 그대로 기록한다(의도표 아님).
            CAP['ckpt_path'] = os.path.abspath(ckpt)
            CAP['ckpt_sha256'] = sha256_of(ckpt)
            CAP['propose_calls'] = CAP.get('propose_calls', 0) + 1
            return _orig(frames, depth, ckpt=ckpt, *a, **kw)
        LI.propose_atoms = _cap_propose
        LI._m7_shim = True

    if not getattr(SGM, '_m7_shim', False):
        _orig_add = SGM.add_gaussians

        def _cap_add(self, n_new, *a, **kw):
            CAP['add_calls'] = CAP.get('add_calls', 0) + 1
            CAP['add_atoms'] = CAP.get('add_atoms', 0) + int(n_new)
            return _orig_add(self, n_new, *a, **kw)
        SGM.add_gaussians = _cap_add
        SGM._m7_shim = True

    if not getattr(VP, '_m7_shim', False):
        _orig_vp = VP.apply_varpro

        def _cap_vp(*a, **kw):
            rec = _orig_vp(*a, **kw)
            CAP['vp'] = dict(rec or {})
            return rec
        VP.apply_varpro = _cap_vp
        VP._m7_shim = True

    if not getattr(SU.SequentialUpdater, '_m7_shim', False):
        _orig_init = SU.SequentialUpdater.__init__

        def _cap_su(self, config, *a, **kw):
            _orig_init(self, config, *a, **kw)
            # ★객체 되읽기 — 입력 dict 복사(의도표)를 쓰지 않는다 (S13b 교훈)
            CAP['obj'] = dict(lambda_l1=getattr(self, 'lambda_l1', None),
                              init_method=getattr(self, 'init_method', None),
                              tracking_mode=getattr(self, 'tracking_mode', None),
                              real_estimator=getattr(self, 'real_estimator', None),
                              adaptive_n=getattr(self, 'adaptive_n', None))
            CAP['obj_updater'] = self
        SU.SequentialUpdater.__init__ = _cap_su
        SU.SequentialUpdater._m7_shim = True

    if not getattr(WR, '_m7_shim', False):
        _orig_run = WR.run_estimation

        def _inject(*a, **kw):
            if _ARM['ckpt'] is None:
                return _orig_run(*a, **kw)              # ★비트 동일 통과
            if 'config' in kw:
                cfg = dict(kw['config'] or {})
                cfg['learned_ckpt'] = _ARM['ckpt']
                kw = dict(kw, config=cfg)
            elif len(a) >= 2:
                cfg = dict(a[1] or {})
                cfg['learned_ckpt'] = _ARM['ckpt']
                a = (a[0], cfg) + tuple(a[2:])
            else:
                kw = dict(kw, config={'learned_ckpt': _ARM['ckpt']})
            return _orig_run(*a, **kw)
        WR.run_estimation = _inject
        WR._m7_shim = True


def _reset_cap():
    for k in ('ckpt_path', 'ckpt_sha256', 'propose_calls', 'add_calls', 'add_atoms',
              'vp', 'obj', 'obj_updater'):
        CAP.pop(k, None)


def _provenance(extra=None):
    """행마다 붙는 출처·계측 스탬프. ⚠`add_calls` 는 **가지 단위 계수기**(`B-2f` 교훈)."""
    import torch
    vp = CAP.get('vp') or {}
    upd = CAP.get('obj_updater')
    mdl = getattr(upd, 'model', None)
    out = dict(
        # PREREG §5 ③ — 전역 RNG 결합 계측: `add_calls` 와 `final_n` 을 행에 남긴다.
        final_n=(int(getattr(mdl, 'n_gaussians', 0) or 0) if mdl is not None else 0),
        ckpt_path=(os.path.relpath(CAP['ckpt_path'], ROOT)
                   if CAP.get('ckpt_path') else None),
        ckpt_sha256=CAP.get('ckpt_sha256'),
        propose_calls=int(CAP.get('propose_calls', 0)),
        add_calls=int(CAP.get('add_calls', 0)),
        add_atoms=int(CAP.get('add_atoms', 0)),
        varpro_obj_before=vp.get('vp_obj_before'),
        varpro_obj_after=vp.get('vp_obj_after'),
        varpro_steps_exec=vp.get('vp_steps'),
        obj_readback=CAP.get('obj'),
        cuda_available=bool(torch.cuda.is_available()),
        cuda_device=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        pid=os.getpid(), git_head=_git_head(), host_ts=time.time())
    out.update(extra or {})
    return out


_GIT = {}


def _git_head():
    if 'h' not in _GIT:
        try:
            _GIT['h'] = subprocess.check_output(
                ['git', '-C', os.path.dirname(ROOT), 'rev-parse', 'HEAD'],
                text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:                                              # noqa: BLE001
            _GIT['h'] = None
    return _GIT['h']


# ── E1 : 실측 12 레코드 ───────────────────────────────────────────────────────
def _e1_arm_config(cell, diffusion_seed):
    """⛔빈 dict 금지 — 키를 전부 명시한다."""
    if cell == G.REF_BANK:
        return dict(real_domain_gate=True, init_method='auto', real_estimator='bank_mmse')
    ck = J.cell_ckpt(cell)
    if ck is None or not os.path.exists(ck):
        raise FileNotFoundError(f'{cell}: 체크포인트 없음 {ck}')
    return dict(real_domain_gate=False, init_method='learned', learned_ckpt=ck,
                learned_diffusion_seed=int(diffusion_seed))


def run_e1_job(ident):
    """`run_b2.run` 을 **그대로** 부른다 — 장면·잡음·질의격자·채점식을 복제하지 않는다.
    팔은 `gates_b2.ARMS` 에 임시 등록해 넘긴다(모듈 상태만, 파일 무수정)."""
    import gates_b2
    import run_b2
    from zero_ref import zero_ref_nowcast
    import zero_ref as ZR
    _, cell, record, dseed = ident
    key = f'M7|{cell}|ds{dseed}'
    gates_b2.ARMS[key] = _e1_arm_config(cell, 0 if dseed is None else dseed)
    _reset_cap()
    _ARM['ckpt'] = None                     # E1 은 make_updater 경로 — 주입 shim 은 쉰다
    t0 = time.time()
    try:
        row = run_b2.run(record, key)
    finally:
        gates_b2.ARMS.pop(key, None)
    upd = CAP.get('obj_updater')
    row = J.stamp(ident, row, **_provenance(dict(
        arm_config=_e1_arm_config(cell, 0 if dseed is None else dseed),
        route_init_method=getattr(upd, 'init_method', None),
        route_real_gate=getattr(upd, '_auto_real_gate', None),
        route_bank=(getattr(upd, '_bank', None) is not None),
        zero_ref=float(zero_ref_nowcast(record, win=G.REAL_WIN_PRIMARY)),
        zero_ref_source=os.path.relpath(os.path.abspath(ZR.__file__), ROOT).replace('.pyc', '.py'),
        isolated=bool(os.environ.get('M7_ISOLATED')),
        wall_s=time.time() - t0)))
    return row


# ── E2 / E3 : 합성 ────────────────────────────────────────────────────────────
def _episode_by_key(exp):
    return {e['episode']: e for e in J.episodes(exp)}


def run_synth_job(ident, ep):
    """`B-12` 팔 정의(`run_b12.run_ship`/`run_spec`)를 **그대로** 부른다 — 복제 0."""
    import run_b12
    exp, cell, _key, seed = ident
    _reset_cap()
    t0 = time.time()
    if cell == J.SPEC_ARM:
        _ARM['ckpt'] = None
        row = run_b12.run_spec(ep, seed)
    else:
        ck = J.cell_ckpt(cell)
        if ck is None or not os.path.exists(ck):
            raise FileNotFoundError(f'{cell}: 체크포인트 없음 {ck}')
        _ARM['ckpt'] = ck
        try:
            row = run_b12.run_ship(ep, seed)
        finally:
            _ARM['ckpt'] = None
    row.pop('arm', None)
    return J.stamp(ident, row, **_provenance(dict(
        gt_arm=ep.get('gt_arm'), corner=ep.get('corner'), ncomp=ep.get('ncomp'),
        gen_seed=ep.get('seed'), isolated=bool(os.environ.get('M7_ISOLATED')),
        wall_s=time.time() - t0)))


# ── 격리 워커 (PREREG §5 완화책 ②) ───────────────────────────────────────────
_SENTINEL = '@@M7ROW@@'


def _run_one(ident):
    if ident[0] == 'e1':
        return run_e1_job(ident)
    return run_synth_job(ident, _episode_by_key(ident[0])[ident[2]])


def _run_isolated(ident):
    env = dict(os.environ, M7_ISOLATED='1')
    cmd = [sys.executable, os.path.abspath(__file__), '__job',
           ident[0], str(ident[1]), str(ident[2]),
           '-' if ident[3] is None else str(ident[3])]
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True,
                       text=True, encoding='utf-8', errors='replace')
    for line in p.stdout.splitlines():
        if line.startswith(_SENTINEL):
            return json.loads(line[len(_SENTINEL):])
    raise RuntimeError(f'격리 워커 실패 rc={p.returncode}: '
                       f'{(p.stderr or p.stdout)[-400:]}')


# ── 본 실행 ───────────────────────────────────────────────────────────────────
def _done_set(path):
    done = set()
    if not os.path.exists(path):
        return done
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if 'error' not in r:
            done.add(J.row_identity(r))
    return done


def main(argv):
    if '--selftest' in argv:
        ok, _ = selftest()
        return 0 if ok else 1

    if argv and argv[0] == '__job':                     # 격리 워커 (내부 전용)
        exp, cell, unit, seed = argv[1], argv[2], argv[3], argv[4]
        ident = (exp, cell, unit, None if seed == '-' else int(seed))
        install_shims()
        print(_SENTINEL + json.dumps(_run_one(ident)), flush=True)
        return 0

    exp = (argv[:1] or ['e2'])[0]
    if exp not in J.EXPERIMENTS:
        print(f'⛔ 사용법: run_m7.py {{{"|".join(J.EXPERIMENTS)}}} [shard nshard] '
              f'[{E1_FLAG}] | --selftest', flush=True)
        return 2

    # ★E1 승인 가드 — 환경변수 **와** 명시 플래그가 둘 다 없으면 돌지 않는다.
    if exp == 'e1':
        has_flag, has_env, approved = e1_approved(argv, os.environ)
        if not approved:
            print('⛔ E1 은 사용자 승인 없이 실행되지 않는다 (PREREG §3 · AMENDMENT_1 §7 Step 5).',
                  flush=True)
            print('   승인 요청 문안:\n     ' + E1_ASK.replace('\n', '\n     '), flush=True)
            print(f'   승인 후 명령:\n     {E1_ENV}=1 python3 '
                  f'tmp/M7_SUPERVISION_2026-08-28/run_m7.py e1 {E1_FLAG}', flush=True)
            print(f'   (현재: 플래그 {has_flag} · 환경변수 {has_env})', flush=True)
            return 3

    pos = [a for a in argv[1:] if not a.startswith('-')]
    shard_i, nshard = (int(pos[0]), int(pos[1])) if len(pos) >= 2 else (0, 1)
    isolate = ('--isolate' in argv) or (exp == 'e1' and '--no-isolate' not in argv)

    ok_self, fails = selftest(verbose=False)
    print('★런 전 자가진단 ' + ('PASS' if ok_self else 'FAIL'), flush=True)
    if not ok_self:
        for f in fails:
            print(f'  ⛔ {f}', flush=True)
        print('⛔ 자가진단 실패 ⟹ 배치를 시작하지 않는다.', flush=True)
        return 1

    if exp == 'e3':                                     # G-14 선행 확인 (M1 다중성 보호)
        vf = os.path.join(ROOT, 'tmp/M1_NONLIN_REAL_2026-08-28/verdict_m1.json')
        pk = (json.load(open(vf)).get('pair_key') if os.path.exists(vf) else None)
        if not pk:
            print(f'⛔ E3 는 M1 판정 동결 후에만 돈다 — pair_key 없음 ({vf})', flush=True)
            return 1
        print(f'  M1 pair_key = {pk} ⟹ G-14 선행 충족', flush=True)

    install_shims()
    jobs = J.shard(J.expected_jobs(exp), shard_i, nshard)
    out = J.results_path(exp, shard_i, nshard)
    done = _done_set(out)
    todo = [j for j in jobs if j not in done]
    import torch
    print(f'M7 {exp} shard {shard_i}/{nshard}: 작업 {len(jobs)} · 완료 {len(jobs)-len(todo)} · '
          f'남음 {len(todo)} · 격리 {isolate} · cuda {torch.cuda.is_available()} -> {out}',
          flush=True)
    if exp != 'e1' and not isolate:
        print('  ⚠E2/E3 는 동결 계보(`run_b12`/`M1`)와 같이 **한 프로세스**에서 돈다. '
              '매 런 시작의 `set_seed()` 가 전역 RNG 를 되돌린다(run.py:212).', flush=True)

    with open(out, 'a') as fh:
        for k, ident in enumerate(todo, 1):
            try:
                row = _run_isolated(ident) if isolate else _run_one(ident)
            except Exception as exc:                                    # noqa: BLE001
                row = J.stamp(ident, dict(error=repr(exc),
                                          tb=traceback.format_exc()[-900:]))
            fh.write(json.dumps(row) + '\n')
            fh.flush()
            tag = (f"nRMSE={row.get('avg_nRMSE'):.4f}" if row.get('avg_nRMSE') is not None
                   else (f"nowcast={row.get('nowcast'):.4f} lead={row.get('leadmean'):.4f}"
                         if row.get('nowcast') is not None
                         else str(row.get('error'))[:70]))
            print(f'[{k}/{len(todo)}] {ident[1]:8s} {str(ident[2])[:34]:34s} '
                  f's{ident[3]} {tag} {row.get("wall_s", 0) or 0:.1f}s', flush=True)
    print(f'완료: {out}', flush=True)
    return 0


# ── 자가진단 (런 0회 · GPU 0회 · 파일 쓰기 0회) ───────────────────────────────
def selftest(verbose=True):
    def say(s):
        if verbose:
            print(s, flush=True)
    ok, fails = J.selftest(verbose=verbose)
    fails = list(fails)
    say('★run_m7 자가진단 ─────────────────────────────────────────')

    # [1] 팔 정의가 **빈 dict 가 아니다** (gates_b7:46 부류)
    for c in J.REAL_ARMS:
        try:
            cfg = _e1_arm_config(c, 0)
        except FileNotFoundError:
            cfg = None                                   # 체크포인트 미생성(학습 전) — 정상
        if cfg is not None and not cfg:
            fails.append(f'{c}: 팔이 빈 dict 로 정의됐다')
    say(f'  [1] E1 팔 정의: 빈 dict 0건 (BANK={_e1_arm_config(G.REF_BANK, None)})')

    # [2] E1 승인 가드 — ★**항진 검사 포함**: 네 조합 중 **정확히 하나**만 통과해야 한다.
    truth = {(f, e): e1_approved(['e1'] + ([E1_FLAG] if f else []),
                                 {E1_ENV: '1'} if e else {})[2]
             for f in (False, True) for e in (False, True)}
    if sum(truth.values()) != 1 or not truth[(True, True)]:
        fails.append(f'E1 가드가 항진/불통이다: {truth}')
    saved = os.environ.pop(E1_ENV, None)
    try:
        for (flag, env), passes in truth.items():
            if passes:
                continue                                  # (True, True) 는 실제로 돌므로 제외
            if env:
                os.environ[E1_ENV] = '1'
            else:
                os.environ.pop(E1_ENV, None)
            rc = main(['e1'] + ([E1_FLAG] if flag else []))
            if rc != 3:
                fails.append(f'E1 가드: 플래그={flag} 환경={env} 에서 rc={rc} (기대 3)')
    finally:
        os.environ.pop(E1_ENV, None)
        if saved is not None:
            os.environ[E1_ENV] = saved
    say('  [2] E1 승인 가드: 4조합 중 (플래그·환경) 둘 다일 때만 통과 · 나머지 3조합 exit 3 ✓')

    # [3] 주입 shim 의 **투명성** — ckpt 가 None 이면 인자를 그대로 넘긴다
    seen = {}

    class _Fake:
        @staticmethod
        def run_estimation(path, config=None, verbose=False):
            seen['cfg'] = config
            return {'avg_nRMSE': 0.1}
    orig = _Fake.run_estimation
    _ARM['ckpt'] = None
    cfg_in = {'seed': 42}

    def _inject(*a, **kw):                                # `install_shims` 와 같은 논리
        if _ARM['ckpt'] is None:
            return orig(*a, **kw)
        cfg = dict(kw.get('config') or {})
        cfg['learned_ckpt'] = _ARM['ckpt']
        return orig(*a, **dict(kw, config=cfg))
    _inject('x.h5', config=cfg_in)
    if seen['cfg'] is not cfg_in:
        fails.append('주입 shim 이 ckpt=None 에서 config 객체를 바꿔치기했다(비투명)')
    _ARM['ckpt'] = '/tmp/x.pt'
    _inject('x.h5', config=cfg_in)
    if seen['cfg'].get('learned_ckpt') != '/tmp/x.pt' or 'learned_ckpt' in cfg_in:
        fails.append('주입 shim 이 learned_ckpt 를 넣지 않았거나 원본 dict 을 오염시켰다')
    _ARM['ckpt'] = None
    say('  [3] 주입 shim: None 통과=동일객체 · 주입 시 원본 dict 무오염 ✓')

    # [4] 결과 경로 규약 (검사기 H 의 접미사 짝)
    p1 = os.path.basename(J.results_path('e2', 3, 8))
    p2 = os.path.basename(J.results_path('e1', 0, 1))
    if p1 != 'results_m7_e2_3.jsonl' or p2 != 'results_m7_e1.jsonl':
        fails.append(f'결과 경로 규약 이탈: {p1} · {p2}')
    say(f'  [4] 결과 경로: {p1} · {p2} ✓')

    for f in fails:
        say(f'  ⛔ {f}')
    say(f'run_m7 자가진단 {"PASS" if not fails else "FAIL"} ({len(fails)} 결함)')
    return (not fails), fails


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
