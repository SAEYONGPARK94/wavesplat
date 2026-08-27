"""B-26 러너 — 팔 정의를 `B-12` 에서 **그대로 import** 하고 **계측만 덧붙인다**.

★`run_b12.run_ship` / `run_b12.run_spec` 를 **수정하지 않는다**(동결물이다).
그런데 그 두 함수는 반환 객체를 버리고 **입력 dict 을 행에 적는다** — CLAUDE.md 가
*"`effective_config` 를 입력 dict 복사로 만들지 마라(= 의도표)"* 라 금지한 부류다.
⟹ ★**투명 shim** 을 건다: `wavesplat.run.run_estimation` 과
`run_s16.run_letkf` 를 감싸 **반환 객체**에서, 그리고 `SequentialUpdater.__init__` 에서
**객체 속성**(`self.lambda_l1`·`self.init_method`·`self.tracking_mode`)을 잡아 행에 싣는다.
⛔shim 은 값을 바꾸지 않는다 — `G-18` 이 *잡은 `avg_nRMSE` == 행의 `avg_nRMSE`* 를
**비트로** 요구해서 그것을 증명한다.

    python3 tmp/B26_SEASTATE_2026-08-27/run_b26.py smoke        # 1셀 (결과는 지운다)
    python3 tmp/B26_SEASTATE_2026-08-27/run_b26.py run          # 576 런
    python3 tmp/B26_SEASTATE_2026-08-27/run_b26.py run 0 8      # 샤드 0/8
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
           os.path.join(ROOT, 'tmp', 'B12_HOS_EXTEND_2026-08-23'),
           os.path.join(ROOT, 'tmp', 'B11_HOS_BASELINE_2026-08-23'),
           os.path.join(ROOT, 'tmp', 'S16_BASELINE_2026-08-14')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(ROOT)

import gates_b26 as G                                          # noqa: E402
import run_b12                                                 # noqa: E402  ★팔 정의
import run_s16                                                 # noqa: E402  ★기준선 정의

CAP = {}


def install_shims():
    """★투명 계측. 값은 건드리지 않는다(G-18 이 비트로 검사)."""
    import wavesplat.run as WR
    from wavesplat.optimizer import sequential_update as SU

    if not getattr(WR, '_b26_shim', False):
        _orig_run = WR.run_estimation

        def _cap_run(*a, **kw):
            r = _orig_run(*a, **kw)
            CAP['ship'] = {f'cap_{k}': r.get(k) for k in G.SHIP_CAPTURE}
            return r
        WR.run_estimation = _cap_run
        WR._b26_shim = True

    if not getattr(SU.SequentialUpdater, '_b26_shim', False):
        _orig_init = SU.SequentialUpdater.__init__

        def _cap_init(self, config, *a, **kw):
            _orig_init(self, config, *a, **kw)
            CAP['obj'] = dict(lambda_l1=self.lambda_l1, init_method=self.init_method,
                              tracking_mode=self.tracking_mode,
                              real_estimator=self.real_estimator)
        SU.SequentialUpdater.__init__ = _cap_init
        SU.SequentialUpdater._b26_shim = True

    if not getattr(run_s16, '_b26_shim', False):
        _orig_letkf = run_s16.run_letkf

        def _cap_letkf(*a, **kw):
            row = _orig_letkf(*a, **kw)
            CAP['spec'] = {f'cap_{k}': row.get(k) for k in G.SPEC_CAPTURE}
            return row
        run_s16.run_letkf = _cap_letkf
        run_s16._b26_shim = True


def run_ship(ep, est_seed):
    CAP.pop('ship', None)
    CAP.pop('obj', None)
    row = run_b12.run_ship(ep, est_seed)            # ★B-12 의 팔, 손복사 0
    row.update(CAP.get('ship', {}))
    row.update({f'obj_{k}': v for k, v in CAP.get('obj', {}).items()})
    row['lambda_l1'] = CAP.get('obj', {}).get('lambda_l1')     # ★객체 되읽기
    row['corner'] = ep['corner']
    row['ncomp'] = ep['ncomp']
    row['gen_seed'] = ep['seed']
    return row


def run_spec(ep, filter_seed):
    CAP.pop('spec', None)
    row = run_b12.run_spec(ep, filter_seed)         # ★S16 의 기준선, 손복사 0
    row.update(CAP.get('spec', {}))
    row['corner'] = ep['corner']
    row['ncomp'] = ep['ncomp']
    row['gen_seed'] = ep['seed']
    return row


def episodes(smoke=False):
    eps, _ = G.select()
    if smoke:
        eps = [e for e in eps if e['corner'] == G.PRIMARY_CORNER][:1]
        for e in eps:
            e['path'] = os.path.join(G.SMOKE, os.path.basename(e['path']))
    # `run_b12.run_ship` 이 요구하는 키(`bf`)를 코너 이름으로 채운다
    for e in eps:
        e['bf'] = e['corner']
    return eps


def main():
    # ★중단 게이트 — **게이트가 실제로 발화하는지** 먼저 확인한다(런 0회, 파일 0개).
    #   저장소가 반복해서 물린 결함이 *"게이트가 있는데 검사하지 않는다"* 부류다
    #   (W8 `G-4` 공허한 통과 · W7 `G-3` 오설계) ⟹ 576런 앞에 기계를 세운다.
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
    out = os.path.join(HERE, 'smoke_results_b26.jsonl' if smoke
                       else f'results_b26_{shard}.jsonl')
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
    print(f'B-26 {which} shard {shard}/{nshard}: {len(eps)} 에피소드 -> {out}', flush=True)
    with open(out, 'a') as fh:
        for ep in eps:
            for arm, fn, seeds, key in (('SHIP-NOW', run_ship, G.EST_SEEDS, 'est_seed'),
                                        ('SPEC25', run_spec, G.FILTER_SEEDS, 'filter_seed')):
                for s in seeds:
                    if (arm, ep['episode'], s) in done:
                        continue
                    try:
                        row = fn(ep, s)
                    except Exception as exc:                   # noqa: BLE001
                        row = dict(arm=arm, episode=ep['episode'], corner=ep['corner'],
                                   **{key: s}, error=repr(exc),
                                   tb=traceback.format_exc()[-500:])
                    fh.write(json.dumps(row) + '\n')
                    fh.flush()
                    print(f'{arm:9s} {ep["episode"][:38]:38s} {key[0]}{s} ' +
                          (f'nRMSE={row["avg_nRMSE"]:.4f} {row["wall_s"]:.1f}s'
                           if 'error' not in row else str(row['error'])[:70]), flush=True)


if __name__ == '__main__':
    main()
