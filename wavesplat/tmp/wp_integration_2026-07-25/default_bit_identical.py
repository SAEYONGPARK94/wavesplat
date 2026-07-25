"""Default-path preservation control for the additive init_method='learned' edits.

Replays recorded v2_konly (default) jobs through the edited code and requires avg_nRMSE
AND init_nrmse to match the recorded head-to-head values to 1e-10. Because the learned
branch is purely additive (only entered when init_method=='learned'), the default path
must be byte-identical. RAN 2026-07-25: 3/3 OK (see WP_INTEGRATION report).

  python3 tmp/wp_integration_2026-07-25/default_bit_identical.py
"""
import json, warnings
warnings.filterwarnings('ignore')

REF = 'tmp/wp_crowded_2026-07-25/ws_runs.jsonl'
FINE = 'data_finemesh_2026-07-23/'


def main():
    rows = [json.loads(l) for l in open(REF)]
    rows = [r for r in rows if r.get('err') is None]
    sample, seen = [], set()
    for r in rows:
        key = (r['episode'].split('_')[1], r['seed'])
        if key in seen:
            continue
        seen.add(key); sample.append(r)
        if len(sample) >= 3:
            break
    import wavesplat.optimizer.sequential_update as su
    su.TRACK_PHASE_PREDICT = False
    from wavesplat.run import run_estimation
    allok = True
    for r in sample:
        out = run_estimation(FINE + r['episode'],
                             config={'adaptive_n': True, 'seed': r['seed'], 'device': 'cpu'})
        ok = (f"{float(out['avg_nRMSE']):.10f}" == f"{r['avg_nRMSE']:.10f}") and \
             (f"{float(out['init_nrmse']):.10f}" == f"{r['init_nrmse']:.10f}")
        allok &= ok
        print(f"{'OK ' if ok else 'DIFF'} {r['episode']:38s} s{r['seed']:<3} "
              f"avg rec={r['avg_nRMSE']:.10f} got={float(out['avg_nRMSE']):.10f} | "
              f"init rec={r['init_nrmse']:.10f} got={float(out['init_nrmse']):.10f}", flush=True)
    print("\nDEFAULT PATH BIT-IDENTICAL:", allok)


if __name__ == '__main__':
    main()
