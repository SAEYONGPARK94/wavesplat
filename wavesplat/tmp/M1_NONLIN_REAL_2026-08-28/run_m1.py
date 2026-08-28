"""M1 러너 — 팔 정의를 `B-12` 에서 **그대로 import**, 계측만 덧붙인다 (`B-26` 규약 축자).

★`run_b12.run_ship` / `run_b12.run_spec` 를 수정하지 않는다(동결물).
★투명 shim 으로 **반환 객체**·**`SequentialUpdater` 객체 속성**에서 값을 잡는다
  (⛔입력 dict 복사 = 의도표 금지). `G-18` 이 shim 투명성을 **비트로** 요구한다.

    python3 tmp/M1_NONLIN_REAL_2026-08-28/run_m1.py smoke      # 1짝 (결과는 지운다)
    python3 tmp/M1_NONLIN_REAL_2026-08-28/run_m1.py run 0 8    # 샤드 0/8 (총 288 런)
"""
import json
import os
import sys
import traceback
import warnings

warnings.filterwarnings('ignore')

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

import gates_m1 as G                                           # noqa: E402
import run_b12                                                 # noqa: E402  ★팔 정의
import run_s16                                                 # noqa: E402  ★기준선 정의

CAP = {}


def install_shims():
    import wavesplat.run as WR
    from wavesplat.optimizer import sequential_update as SU

    if not getattr(WR, '_m1_shim', False):
        _orig = WR.run_estimation

        def _cap(*a, **kw):
            r = _orig(*a, **kw)
            CAP['ship'] = {f'cap_{k}': r.get(k) for k in G.SHIP_CAPTURE}
            return r
        WR.run_estimation = _cap
        WR._m1_shim = True

    if not getattr(SU.SequentialUpdater, '_m1_shim', False):
        _orig_init = SU.SequentialUpdater.__init__

        def _cap_init(self, config, *a, **kw):
            _orig_init(self, config, *a, **kw)
            CAP['obj'] = dict(lambda_l1=self.lambda_l1, init_method=self.init_method,
                              tracking_mode=self.tracking_mode,
                              real_estimator=self.real_estimator)
        SU.SequentialUpdater.__init__ = _cap_init
        SU.SequentialUpdater._m1_shim = True

    if not getattr(run_s16, '_m1_shim', False):
        _orig_letkf = run_s16.run_letkf

        def _cap_letkf(*a, **kw):
            row = _orig_letkf(*a, **kw)
            CAP['spec'] = {f'cap_{k}': row.get(k) for k in G.SPEC_CAPTURE}
            return row
        run_s16.run_letkf = _cap_letkf
        run_s16._m1_shim = True


def _tag(row, ep):
    row['gt_arm'] = ep['gt_arm']
    row['corner'] = ep['corner']
    row['ncomp'] = ep['ncomp']
    row['gen_seed'] = ep['seed']
    row['episode'] = ep['episode']          # ★'LIN|15comp_seed1681' — GT 팔이 키에 들어간다
    return row


def run_ship(ep, est_seed):
    CAP.pop('ship', None)
    CAP.pop('obj', None)
    row = run_b12.run_ship(ep, est_seed)
    row.update(CAP.get('ship', {}))
    row.update({f'obj_{k}': v for k, v in CAP.get('obj', {}).items()})
    row['lambda_l1'] = CAP.get('obj', {}).get('lambda_l1')       # ★객체 되읽기
    return _tag(row, ep)


def run_spec(ep, filter_seed):
    CAP.pop('spec', None)
    row = run_b12.run_spec(ep, filter_seed)
    row.update(CAP.get('spec', {}))
    return _tag(row, ep)


def episodes(smoke=False):
    eps, _ = G.select()
    if smoke:
        first = sorted({(e['ncomp'], e['seed']) for e in eps})[0]
        eps = [e for e in eps if (e['ncomp'], e['seed']) == first]
        for e in eps:
            e['path'] = os.path.join(G.SMOKE, e['gt_arm'], os.path.basename(e['path']))
    return eps


def main():
    if '--selftest' in sys.argv:
        sys.exit(0 if G.selftest()[0] else 1)
    print('★런 전 게이트 자가진단 ─────────────────────────────────────', flush=True)
    if not G.selftest()[0]:
        print('⛔게이트 자가진단 실패 ⟹ 배치를 시작하지 않는다.', flush=True)
        sys.exit(1)
    print('────────────────────────────────────────────────────────────\n', flush=True)
    which = (sys.argv[1:] or ['run'])[0]
    shard, nshard = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 1)
    smoke = which == 'smoke'
    out = os.path.join(HERE, 'smoke_results_m1.jsonl' if smoke else f'results_m1_{shard}.jsonl')
    install_shims()
    eps = [e for i, e in enumerate(episodes(smoke)) if i % nshard == shard]
    done = set()
    if os.path.exists(out):
        for line in open(out):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'error' not in r:
                done.add((r['arm'], r['episode'], r.get('est_seed') or r.get('filter_seed')))
    print(f'M1 {which} shard {shard}/{nshard}: {len(eps)} 에피소드 -> {out}', flush=True)
    with open(out, 'a') as fh:
        for ep in eps:
            for arm, fn, seeds, key in (('SHIP-NOW', run_ship, G.EST_SEEDS, 'est_seed'),
                                        ('SPEC25', run_spec, G.FILTER_SEEDS, 'filter_seed')):
                for s in seeds:
                    if (arm, ep['episode'], s) in done:
                        continue
                    try:
                        row = fn(ep, s)
                    except Exception as exc:                    # noqa: BLE001
                        row = dict(arm=arm, episode=ep['episode'], gt_arm=ep['gt_arm'],
                                   corner=ep['corner'], **{key: s}, error=repr(exc),
                                   tb=traceback.format_exc()[-500:])
                    fh.write(json.dumps(row) + '\n')
                    fh.flush()
                    print(f'{arm:9s} {ep["episode"][:30]:30s} {key[0]}{s} ' +
                          (f'nRMSE={row["avg_nRMSE"]:.4f} {row["wall_s"]:.1f}s'
                           if 'error' not in row else str(row['error'])[:70]), flush=True)


if __name__ == '__main__':
    main()
