"""M2 러너 — ★기준선 정의를 **손으로 옮기지 않는다**.

`run_s16.run_letkf` 를 그대로 부르고, 새 팔은 **`run_s16.ARMS` 에 항목만 등록**한다
(= `S17` 의 선례: *"기준선 변형은 `letkf_spec.py` 를 무수정으로 재사용하고 팔 정의(dict)만
추가한다"*). ⛔`run_s16.py`·`letkf_spec.py`·`wavesplat/**`·`wavesim/**` 무수정.

    python3 tmp/M2_LOCTUNE_2026-08-28/run_m2.py --selftest   # 게이트 자가진단만
    python3 tmp/M2_LOCTUNE_2026-08-28/run_m2.py smoke        # 1셀 (산출물은 지운다)
    python3 tmp/M2_LOCTUNE_2026-08-28/run_m2.py pc           # 양성대조 12+12 런
    python3 tmp/M2_LOCTUNE_2026-08-28/run_m2.py run i 8      # 샤드 i/8
"""
import json
import os
import sys
import time
import traceback
import warnings

warnings.filterwarnings('ignore')
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_k, '1')          # ★스레드 고정 (미고정 시 load thrash)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for _p in (HERE, ROOT,
           os.path.join(ROOT, 'tmp', 'B26_SEASTATE_2026-08-27'),
           os.path.join(ROOT, 'tmp', 'B12_HOS_EXTEND_2026-08-23'),
           os.path.join(ROOT, 'tmp', 'B11_HOS_BASELINE_2026-08-23'),
           os.path.join(ROOT, 'tmp', 'S16_BASELINE_2026-08-14')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(ROOT)

import gates_m2 as G                                            # noqa: E402
import run_b12                                                  # noqa: E402  SHIP 팔
import run_s16                                                  # noqa: E402  기준선 팔

CAP = {}


def register_arms():
    """★개입은 이것 하나 — `ARMS` 에 항목 등록. `run_letkf` 는 `ARMS[arm]['loc']` 을
    그대로 `LETKFGridSpec(loc_radius=...)` 에 넣는다(`run_s16.py:208,214`)."""
    for name, spec in G.NEW_ARMS.items():
        if name in run_s16.ARMS and run_s16.ARMS[name] != spec:
            raise RuntimeError(f'{name} 이 다른 정의로 이미 있다: {run_s16.ARMS[name]}')
        run_s16.ARMS[name] = dict(spec)


def install_shim():
    """★투명 계측 — 값은 건드리지 않는다(`G-18` 이 비트로 검사)."""
    if getattr(run_s16, '_m2_shim', False):
        return
    _orig = run_s16.run_letkf

    def _cap(*a, **kw):
        row = _orig(*a, **kw)
        CAP['spec'] = {'cap_avg_nRMSE': row.get('avg_nRMSE'),
                       'cap_loc_radius': row.get('loc_radius'),
                       'cap_filter_class': row.get('filter_class'),
                       'cap_Hs': row.get('Hs'), 'cap_Tp': row.get('Tp'),
                       'cap_zero_nRMSE': row.get('zero_nRMSE')}
        return row
    run_s16.run_letkf = _cap
    run_s16._m2_shim = True


KEEP = ('avg_nRMSE', 'zero_nRMSE', 'loc_radius', 'n_ensemble', 'background',
        'filter_class', 'filter_seed', 'effective_config', 'Hs', 'Tp', 'depth',
        'n_match', 'n_all', 'ms_median', 'shell_rmse', 'zero_shell', 'shell_frac',
        'hs_obs4std')


def run_spec(ep, arm, filter_seed, pc=False):
    CAP.pop('spec', None)
    t0 = time.time()
    row = run_s16.run_letkf(ep['path'], arm, filter_seed=filter_seed)
    out = dict(arm=arm, episode=ep['episode'], corner=ep['corner'],
               ncomp=ep['ncomp'], gen_seed=ep['seed'], filter_seed=filter_seed,
               wall_s=time.time() - t0, pc=bool(pc))
    out.update({k: row.get(k) for k in KEEP})
    out.update(CAP.get('spec', {}))
    return out


def run_ship(ep, est_seed):
    """★양성대조 전용 — `B-12` 의 팔 정의를 그대로(손복사 0)."""
    row = run_b12.run_ship(dict(ep, bf=ep['corner']), est_seed)
    row.update(corner=ep['corner'], ncomp=ep['ncomp'], gen_seed=ep['seed'], pc=True)
    return row


def units(which, shard=0, nshard=1):
    """작업 단위 목록 — 샤딩은 **추정만**(생성 0회)."""
    u = []
    if which == 'pc':
        for ep in G.pc_episodes():
            for s in G.FILTER_SEEDS:
                u.append(('spec', ep, G.INCUMBENT_ARM, s))
            for s in G.EST_SEEDS:
                u.append(('ship', ep, G.SHIP_ARM, s))
    elif which == 'smoke':
        ep = [e for e in G.episodes() if e['corner'] == G.PRIMARY_CORNER][0]
        for a in ('SPEC12', 'SPEC6'):
            u.append(('spec', ep, a, G.FILTER_SEEDS[0]))
    else:
        for ep in G.episodes():
            for a in ('SPEC12', 'SPEC6'):
                for s in G.FILTER_SEEDS:
                    u.append(('spec', ep, a, s))
    return [x for i, x in enumerate(u) if i % nshard == shard]


def main():
    if '--selftest' in sys.argv:
        sys.exit(0 if G.selftest()[0] else 1)
    print('★런 전 게이트 자가진단 ──────────────────────────────────', flush=True)
    if not G.selftest()[0]:
        print('⛔게이트 자가진단 실패 ⟹ 배치를 시작하지 않는다.', flush=True)
        sys.exit(1)
    rf = list(G.gates_runfree()) + list(G.gates_data())
    for n_, q_, d_ in rf:
        print(f'  [{"PASS" if q_ else "FAIL"}] {n_}: {d_}', flush=True)
    if not all(g[1] for g in rf):
        print('⛔런 0회 게이트 FAIL ⟹ 중단', flush=True)
        sys.exit(1)
    print('──────────────────────────────────────────────────────────\n', flush=True)

    which = (sys.argv[1:] or ['run'])[0]
    shard, nshard = ((int(sys.argv[2]), int(sys.argv[3]))
                     if len(sys.argv) > 3 else (0, 1))
    # ★AMENDMENT_1 (2026-08-28, 결과 0행): PC 파일명 `results_m2_pc.jsonl` → `results_m2_9.jsonl`.
    #   검사기 `_sfx()` 는 끝의 `_<숫자>` 만 벗긴다 — `_pc` 는 접미사 `m2_pc` 가 되어
    #   `verdict_m2_pc.json` 부재로 paired=False = 인용 불가(적대적 관문이 가짜 배치로 재현).
    #   9 는 본배치 샤드 0–7 밖의 예약 숫자다. 분석기 glob `results_m2*.jsonl` 이 그대로 집고,
    #   PC 행 식별은 파일명이 아니라 행의 `pc` 필드다(`gates_m2.aggregate` 가 `not r.get('pc')`).
    out = os.path.join(HERE, {'smoke': 'smoke_results_m2.jsonl',
                              'pc': 'results_m2_9.jsonl'}.get(
        which, f'results_m2_{shard}.jsonl'))
    register_arms()
    install_shim()
    done = set()
    if os.path.exists(out):
        for line in open(out):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'error' not in r:
                done.add((r['arm'], r['episode'],
                          r.get('filter_seed') if r.get('est_seed') is None
                          else r.get('est_seed')))
    us = units(which, shard, nshard)
    print(f'M2 {which} shard {shard}/{nshard}: {len(us)} 런 -> {out}', flush=True)
    with open(out, 'a') as fh:
        for kind, ep, arm, s in us:
            if (arm, ep['episode'], s) in done:
                continue
            try:
                row = (run_ship(ep, s) if kind == 'ship'
                       else run_spec(ep, arm, s, pc=(which == 'pc')))
            except Exception as exc:                            # noqa: BLE001
                row = dict(arm=arm, episode=ep['episode'], corner=ep['corner'],
                           filter_seed=s, error=repr(exc),
                           tb=traceback.format_exc()[-500:], pc=(which == 'pc'))
            fh.write(json.dumps(row) + '\n')
            fh.flush()
            print(f'{arm:7s} {ep["episode"][:40]:40s} s{s} ' +
                  (f'nRMSE={row["avg_nRMSE"]:.6f} loc={row.get("loc_radius")} '
                   f'{row.get("wall_s", 0):.1f}s'
                   if 'error' not in row else str(row['error'])[:70]), flush=True)


if __name__ == '__main__':
    main()
