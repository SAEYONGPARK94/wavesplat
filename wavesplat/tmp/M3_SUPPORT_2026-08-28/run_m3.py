"""M3 러너 — 팔 정의를 `B-12` 에서 **그대로 import** 하고 **계측만 덧붙인다** (B-26 규약).

★`run_b12.run_ship` / `run_b12.run_spec` 를 수정하지 않는다(동결물). 투명 shim 으로
`run_estimation`/`run_letkf` 의 **반환 객체**와 `SequentialUpdater` **객체 속성**을 행에
싣는다(G-18 이 *잡은 avg_nRMSE == 행의 avg_nRMSE* 를 비트로 요구).

★PC-2: 동결 `B-26` `SS-BF4` 에피소드 2개(15/30comp seed1641)를 est/filter 시드 42 로
재실행해 동결 `results_b26_*.jsonl` 행과 `avg_nRMSE` **비트 동일**을 요구한다(G-23)
— "좌표를 배포값으로 두면 동결 행과 비트 동일" 의 러너 수준 절반이다.

    python3 tmp/M3_SUPPORT_2026-08-28/run_m3.py --selftest      # 게이트 자가진단만
    python3 tmp/M3_SUPPORT_2026-08-28/run_m3.py pc2             # 러너 양성대조 (≈5분)
    python3 tmp/M3_SUPPORT_2026-08-28/run_m3.py smoke           # 1셀 (결과는 지운다)
    python3 tmp/M3_SUPPORT_2026-08-28/run_m3.py run 0 8         # 샤드 0/8
"""
import os

# ★스레드 고정 — 규율 (미고정 시 8샤드 load 87 thrash). import 보다 먼저.
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'

import glob                                                    # noqa: E402
import json                                                    # noqa: E402
import sys                                                     # noqa: E402
import time                                                    # noqa: E402
import traceback                                               # noqa: E402
import warnings                                                # noqa: E402

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for _p in (HERE, ROOT,
           os.path.join(ROOT, 'tmp', 'B12_HOS_EXTEND_2026-08-23'),
           os.path.join(ROOT, 'tmp', 'B11_HOS_BASELINE_2026-08-23'),
           os.path.join(ROOT, 'tmp', 'S16_BASELINE_2026-08-14')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(ROOT)

import gates_m3 as G                                           # noqa: E402
import run_b12                                                 # noqa: E402  ★팔 정의
import run_s16                                                 # noqa: E402  ★기준선 정의

CAP = {}


def install_shims():
    """★투명 계측 — 값은 건드리지 않는다(G-18 이 비트로 검사). B-26 과 동형."""
    import wavesplat.run as WR
    from wavesplat.optimizer import sequential_update as SU

    if not getattr(WR, '_m3_shim', False):
        _orig_run = WR.run_estimation

        def _cap_run(*a, **kw):
            r = _orig_run(*a, **kw)
            CAP['ship'] = {f'cap_{k}': r.get(k) for k in G.SHIP_CAPTURE}
            return r
        WR.run_estimation = _cap_run
        WR._m3_shim = True

    if not getattr(SU.SequentialUpdater, '_m3_shim', False):
        _orig_init = SU.SequentialUpdater.__init__

        def _cap_init(self, config, *a, **kw):
            _orig_init(self, config, *a, **kw)
            CAP['obj'] = dict(lambda_l1=self.lambda_l1, init_method=self.init_method,
                              tracking_mode=self.tracking_mode,
                              real_estimator=self.real_estimator)
        SU.SequentialUpdater.__init__ = _cap_init
        SU.SequentialUpdater._m3_shim = True

    if not getattr(run_s16, '_m3_shim', False):
        _orig_letkf = run_s16.run_letkf

        def _cap_letkf(*a, **kw):
            row = _orig_letkf(*a, **kw)
            CAP['spec'] = {f'cap_{k}': row.get(k) for k in G.SPEC_CAPTURE}
            return row
        run_s16.run_letkf = _cap_letkf
        run_s16._m3_shim = True


def run_ship(ep, est_seed):
    CAP.pop('ship', None)
    CAP.pop('obj', None)
    row = run_b12.run_ship(ep, est_seed)            # ★B-12 의 팔, 손복사 0
    row.update(CAP.get('ship', {}))
    row.update({f'obj_{k}': v for k, v in CAP.get('obj', {}).items()})
    row['lambda_l1'] = CAP.get('obj', {}).get('lambda_l1')     # ★객체 되읽기
    row['cell'] = ep.get('cell')
    row['ncomp'] = ep['ncomp']
    row['gen_seed'] = ep['seed']
    for k in ('direction_deg', 'spread_deg', 'depth'):
        row[k] = ep.get(k)
    return row


def run_spec(ep, filter_seed):
    CAP.pop('spec', None)
    row = run_b12.run_spec(ep, filter_seed)         # ★S16 의 기준선, 손복사 0
    row.update(CAP.get('spec', {}))
    row['cell'] = ep.get('cell')
    row['ncomp'] = ep['ncomp']
    row['gen_seed'] = ep['seed']
    for k in ('direction_deg', 'spread_deg', 'depth'):
        row[k] = ep.get(k)
    return row


def episodes(smoke=False):
    eps, _ = G.select()
    if smoke:
        eps = [e for e in eps if e['cell'] == G.PRIMARY_CELL][:1]
        for e in eps:
            e['path'] = os.path.join(G.cell_dir(e['cell'], G.SMOKE),
                                     G.fname(e['cell'], e['ncomp'], e['seed']))
    for e in eps:                       # run_b12.run_ship 가 요구하는 키
        e['bf'] = G.SEA_STATE
    return eps


def frozen_b26_rows():
    """동결 B-26 결과에서 PC-2 대상 행의 avg_nRMSE 를 모은다 (seed 42 행, 팔별)."""
    fz = {}
    for p in sorted(glob.glob(G.B26_RESULTS_GLOB)):
        for line in open(p):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'error' in r or r.get('episode') not in G.PC2_EPISODES:
                continue
            s = r.get('est_seed') if r.get('arm') == 'SHIP-NOW' else r.get('filter_seed')
            if s == G.PC2_SEED:
                fz[f"{r['arm']}|{r['episode']}"] = r['avg_nRMSE']
    return fz


def pc2():
    """동결 B-26 SS-BF4 에피소드 재실행 → 동결 행과 비트 대조. 산출 = pc2_m3.json."""
    install_shims()
    rows = []
    for name in G.PC2_EPISODES:
        ncomp = 15 if '15comp' in name else 30
        ep = dict(path=os.path.join(G.B26_DATA, 'SS-BF4', f'episode_{name}.h5'),
                  episode=name, bf='SS-BF4', ncomp=ncomp, seed=1641, cell=None)
        if not os.path.exists(ep['path']):                      # B-26 저장 배치 탐색
            cand = glob.glob(os.path.join(G.B26_DATA, '**', f'episode_{name}.h5'),
                             recursive=True)
            assert cand, f'PC-2 에피소드 없음: {name}'
            ep['path'] = cand[0]
        rows.append(dict(arm='SHIP-NOW', episode=name,
                         avg_nRMSE=run_b12.run_ship(ep, G.PC2_SEED)['avg_nRMSE']))
        rows.append(dict(arm='SPEC25', episode=name,
                         avg_nRMSE=run_b12.run_spec(ep, G.PC2_SEED)['avg_nRMSE']))
    out = dict(rows=rows, frozen=frozen_b26_rows())
    json.dump(out, open(os.path.join(HERE, 'pc2_m3.json'), 'w'), indent=1)
    for name, ok, msg in G.gates_pc2(out):
        print(f'  [{"PASS" if ok else "FAIL"}] {name}: {msg}', flush=True)
    return out


def main():
    if '--selftest' in sys.argv:
        sys.exit(0 if G.selftest()[0] else 1)
    which = (sys.argv[1:] or ['run'])[0]
    if which == 'pc2':
        pc2()
        return
    # ★런 전 자가진단 — 게이트가 실제로 발화하는지 (B-26 규약: 기계를 앞에 세운다)
    print('★런 전 게이트 자가진단 ─────────────────────────────', flush=True)
    ok, _ = G.selftest()
    if not ok:
        print('⛔자가진단 실패 — 러너를 돌리지 않는다.', flush=True)
        sys.exit(1)
    smoke = which == 'smoke'
    shard, nshard = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 1)
    install_shims()
    eps = episodes(smoke=smoke)
    if not smoke:                       # 생성물 파생측정 게이트 — 추정 전 강제
        gd = G.gates_data()
        for n, q, d in gd:
            print(f'  [{"PASS" if q else "FAIL"}] {n}: {d}', flush=True)
        if not all(q for _, q, _ in gd):
            print('⛔데이터 게이트 실패 — 추정을 돌리지 않는다.', flush=True)
            sys.exit(1)
    out_path = os.path.join(HERE, 'results_m3_smoke.jsonl' if smoke
                            else f'results_m3_{shard}.jsonl')
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'error' not in r:
                done.add((r['arm'], r['episode'],
                          r.get('est_seed') or r.get('filter_seed')))
    jobs = [e for i, e in enumerate(eps) if i % nshard == shard]
    print(f'=== M3 {which} shard {shard}/{nshard}: {len(jobs)} 에피소드', flush=True)
    with open(out_path, 'a') as fh:
        for ep in jobs:
            for s in (G.EST_SEEDS if not smoke else G.EST_SEEDS[:1]):
                if ('SHIP-NOW', ep['episode'], s) in done:
                    continue
                try:
                    row = run_ship(ep, s)
                except Exception as e:                          # noqa: BLE001
                    row = dict(arm='SHIP-NOW', episode=ep['episode'], cell=ep['cell'],
                               est_seed=s, error=repr(e),
                               tb=traceback.format_exc()[-500:])
                fh.write(json.dumps(row) + '\n')
                fh.flush()
                print(f"SHIP-NOW {ep['episode'][:40]:40s} s{s} "
                      + (f"nRMSE={row['avg_nRMSE']:.4f} {row['wall_s']:.1f}s"
                         if 'error' not in row else str(row['error'])[:60]), flush=True)
            for s in (G.FILTER_SEEDS if not smoke else G.FILTER_SEEDS[:1]):
                if ('SPEC25', ep['episode'], s) in done:
                    continue
                try:
                    row = run_spec(ep, s)
                except Exception as e:                          # noqa: BLE001
                    row = dict(arm='SPEC25', episode=ep['episode'], cell=ep['cell'],
                               filter_seed=s, error=repr(e),
                               tb=traceback.format_exc()[-500:])
                fh.write(json.dumps(row) + '\n')
                fh.flush()
                print(f"SPEC25   {ep['episode'][:40]:40s} f{s} "
                      + (f"nRMSE={row['avg_nRMSE']:.4f} zero={row['zero_nRMSE']:.4f} "
                         f"{row['wall_s']:.1f}s"
                         if 'error' not in row else str(row['error'])[:60]), flush=True)


if __name__ == '__main__':
    main()
