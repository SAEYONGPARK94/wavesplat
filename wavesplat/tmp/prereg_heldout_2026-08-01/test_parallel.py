"""Integrity + cost check for the arm-C worker: a member's field must not
depend on whether it ran in-process or in a spawned worker alongside the other
seeds.  Also times sequential vs 3-way parallel."""
import os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '4')
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'bankmmse_port_2026-07-29'))
import run_rolling_refit as rr
from v2k import v2_member

W_MAX = rr.W_MAX


def main():
    st = rr.build_stream('AA14', 0)
    m = (st['P_all'][:, 2] >= st['t0'] - W_MAX) & (st['P_all'][:, 2] <= st['t0'] + 1e-9)
    P, z = st['P_all'][m], st['z_all'][m]
    if len(z) > 600:
        ii = np.random.default_rng(0).choice(len(z), 600, replace=False)
        P, z = P[ii], z[ii]
    t_shift = float(P[:, 2].min())
    rel = [st['t0'] - t_shift, st['t0'] + 1.0 - t_shift]
    args = [(P, z, st['depth'], st['gxq'], st['gyq'], t_shift, rel, s)
            for s in (42, 1042, 2042)]

    t = time.time()
    solo = v2_member(args[0])
    t_solo = time.time() - t
    print(f'in-process seed 42: {t_solo:.1f}s  field[0] rms '
          f'{np.sqrt(np.mean(solo[0]**2)):.5f}', flush=True)

    ctx = mp.get_context('spawn')
    t = time.time()
    with ctx.Pool(3) as pool:
        par = pool.map(v2_member, args)
    t_par = time.time() - t
    d = float(np.max(np.abs(par[0] - solo)))
    print(f'spawn pool(3), 3 seeds: {t_par:.1f}s  |seed42_par - seed42_solo|_max '
          f'= {d:.3e}', flush=True)
    print('BIT-IDENTICAL' if d == 0.0 else 'MISMATCH -- run arm C sequentially')
    print(f'speedup vs 3x sequential: {3*t_solo/t_par:.2f}x')


if __name__ == '__main__':
    main()
