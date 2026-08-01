"""Integrity control (not an endpoint): the 2026-08-01 harness must reproduce
the 2026-07-29 rolling-refit numbers bit-for-bit on a CONSUMED window, which is
outside the PRIMARY 21 and therefore cannot influence any prereg verdict.

frozen == arm B, refit2 == arm A. Reference: tmp/bankmmse_port_2026-07-29/
rolling_refit.json (AA15a_w1: frozen 1.2718, refit2 0.9417).
"""
import os, sys, json
os.environ.setdefault('OMP_NUM_THREADS', '4')
import warnings; warnings.filterwarnings('ignore')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'bankmmse_port_2026-07-29'))
import run_rolling_refit as rr
import run_prereg_heldout as ph

REF = json.load(open('tmp/bankmmse_port_2026-07-29/rolling_refit.json'))


def main():
    ok = True
    for record, win in [('AA15a', 1), ('YS17', 1)]:
        st = rr.build_stream(record, win)
        rr.assert_replication(st, record, win)
        taus = [float(v) for v in np.arange(0.0, st['lead_max'] + 1e-9, 1.0)]
        cache, rcache = {}, {}
        kp = ph.bank_fit(st, st['t0'], 40.0, None, cache)[0][0].kp
        for arm, ref_arm in (('B', 'frozen'), ('A', 'refit2')):
            r = ph.run_bank_arm(st, taus, ph.ARM_SPEC[arm], kp, cache, rcache)
            got = float(np.mean(r['curve']))
            exp = REF[f'{record}_w{win}']['arms'][ref_arm]['sustained_mean']
            dc = float(np.max(np.abs(np.array(r['curve'])
                                     - np.array(REF[f'{record}_w{win}']['arms'][ref_arm]['curve']))))
            good = dc == 0.0
            ok &= good
            print(f'{record}_w{win} {arm} (={ref_arm}): got {got:.6f} '
                  f'ref {exp:.6f}  max|curve diff| {dc:.3e}  '
                  f'{"BIT-IDENTICAL" if good else "MISMATCH"}', flush=True)
    print('CONTROL PASS' if ok else 'CONTROL FAIL')


if __name__ == '__main__':
    main()
