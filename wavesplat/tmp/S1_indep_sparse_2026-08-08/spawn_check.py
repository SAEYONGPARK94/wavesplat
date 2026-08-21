"""AMENDMENT 1 F6 — spawn 경로를 **논증하지 말고 측정한다**.

CON 지적: `run_rx2b.DATA` 는 모듈 전역이고 `mp.get_context('spawn')` 자식은 모듈을 재임포트하므로
부모에서 바꾼 값이 자식에 전파되지 않을 수 있다. 게다가 S1 은 데이터 디렉터리 **두 개**에 걸친다.

`run_s1.py` 는 `R.DATA = REPO` 를 자기 모듈 최상단에서 하고, 자식이 `run_s1` 을 재임포트하므로
전파된다 — **그러나 그건 논증이다.** 이 파일이 실제 spawn 워커 안에서 다음을 잰다:

  1. 워커가 보는 `R.DATA` 가 REPO 인가
  2. 서로 **다른 두 디렉터리**의 에피소드가 워커에서 실제로 열리는가
  3. AMENDMENT 1 F5 로 조립한 `AUTO-G1P1` 셀이 워커에서 실제로 적용되는가
     (env 게이트 + `_beamform_coarse` 몽키패치 + cfg)

S1 데이터가 아직 없어도 되도록 **기존 독립 집합**(indep2 / indep3)을 쓴다. S1 결과 파일은 만들지 않는다.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def probe(job):
    """워커 안에서 실행된다. 경로·셀 적용 상태를 되돌려준다."""
    import os as _os
    import run_s1 as S
    import wavesplat.optimizer.initializer as I

    seen_data = S.R.DATA
    path = _os.path.join(seen_data, job['episode'])
    exists = _os.path.exists(path)

    undo = S.R.apply_cell(job['cell'])
    try:
        env = {k: _os.environ.get(k) for k in S.R.RX2_ENV_KEYS}
        patched = getattr(I._beamform_coarse, '__name__', '?')
        cfg = S.R.cell_config(job['cell'], job['episode']) if exists else {}
    finally:
        undo()
    env_after = {k: _os.environ.get(k) for k in S.R.RX2_ENV_KEYS}

    return dict(cell=job['cell'], pid=_os.getpid(), seen_data=seen_data,
                episode=job['episode'], exists=exists, env_in_cell=env,
                env_after_undo=env_after, beamform=patched,
                init_method=cfg.get('init_method'), n_gauss=cfg.get('n_gaussians'))


def main():
    import multiprocessing as mp
    import run_s1 as S

    jobs = [
        dict(cell='X-BASE',
             episode='data_indep2_2026-08-07/episode_BF4_15comp_seed0_uniform_ra_rf_ds.h5'),
        dict(cell='AUTO-G1P1',
             episode='data_indep3_2026-08-07/episode_BF5_30comp_seed7_uniform_ra_rf_ds.h5'),
        dict(cell='ORACLE-K',
             episode='data_indep4_2026-08-07/episode_BF4_15comp_seed10_uniform_ra_rf_ds.h5'),
    ]
    print('AMENDMENT 1 F6 — spawn 워커 경로/셀 적용 측정')
    print(f'부모 REPO = {S.REPO}')
    print(f'조립 셀 AUTO-G1P1 = {S.R.CELLS.get("AUTO-G1P1")}')
    ctx = mp.get_context('spawn')
    with ctx.Pool(3, initializer=S.R._init_worker) as pool:
        out = pool.map(probe, jobs)

    ok = True
    for r in out:
        good_path = (r['seen_data'] == S.REPO) and r['exists']
        print(f"\n[{r['cell']}] pid={r['pid']}")
        print(f"   워커가 본 R.DATA : {r['seen_data']}  "
              f"{'✅' if r['seen_data'] == S.REPO else '⛔ 전파 실패'}")
        print(f"   에피소드 열림     : {r['exists']}  {'✅' if r['exists'] else '⛔'}  "
              f"({r['episode'].split('/')[0]})")
        print(f"   셀 안 env        : {r['env_in_cell']}")
        print(f"   undo 후 env      : {r['env_after_undo']}")
        print(f"   _beamform_coarse : {r['beamform']}")
        print(f"   cfg init_method  : {r['init_method']}   n_gaussians={r['n_gauss']}")
        ok &= good_path

    exp = {'X-BASE': ('v2_konly', '0', 'bf_is_original'),
           'AUTO-G1P1': ('auto', '1', 'bf_is_patched'),
           'ORACLE-K': ('oracle', '0', 'bf_is_original')}
    print('\n' + '=' * 66)
    for r in out:
        want_init, want_env, _ = exp[r['cell']]
        e1 = r['init_method'] == want_init
        e2 = r['env_in_cell'].get('RX2_GROW_SUBTRACT') == want_env
        e3 = all(v == '0' for v in r['env_after_undo'].values())
        e4 = (r['beamform'] == 'bf') == (r['cell'] == 'AUTO-G1P1')
        print(f"  {r['cell']:<11s} init {'✅' if e1 else '⛔'}  env {'✅' if e2 else '⛔'}  "
              f"undo {'✅' if e3 else '⛔'}  krange {'✅' if e4 else '⛔'}")
        ok &= (e1 and e2 and e3 and e4)
    print(f"SPAWN CHECK: {'PASS ✅ — run_s1.py 그대로 간다' if ok else 'FAIL ⛔ — 자체 run_one 필요'}")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
