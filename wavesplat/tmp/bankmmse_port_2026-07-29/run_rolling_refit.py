"""Rolling-refit streaming validation. Protocol: PROTOCOL_ROLLING_REFIT.md.

Replicates scene_real's observation generation (asserted bit-identical on the
memory window) and extends it into the lead span, then compares the deployed
frozen-fit behaviour against refit cadences {2,5,10} s using the PACKAGE
bank-MMSE port (wavesplat.optimizer.bank_mmse.fit_bank_mmse -- the deployed
math, fit/render split so one fit serves many evaluation times).
"""
import os, sys, json, time
os.environ.setdefault('OMP_NUM_THREADS', '4')
import warnings; warnings.filterwarnings('ignore')
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
os.chdir(ROOT)

OUT = 'tmp/bankmmse_port_2026-07-29/rolling_refit.json'
G = 9.81
W_MAX = 40.0
SEEDS = [42, 1042, 2042]
CADENCES = [2.0, 5.0, 10.0]
WINDOWS = [(r, w) for r in ('AA15a', 'AA15b', 'YS17') for w in (1, 3, 5)]


def build_stream(record, win):
    """scene_real's obs generation, continued into the lead span.

    Returns dict with the full stream (P_all, z_all), the frozen-protocol
    window (for the replication assert), GT frames, grid and timing.
    """
    from realdata_analysis import directional_spread
    from d3_falcon_real_operator import RECORDS, resolve_direction
    from d3_sa_multirecord import MRecord

    cfg = RECORDS[record]
    U, depth, lead_max = cfg['U'], cfg['depth'], cfg['lead_max']
    rec = MRecord(cfg['path'], stride=cfg['stride'])
    nfw = int(round(W_MAX / rec.dt))
    n_lead = int(round(lead_max / rec.dt))
    cx = 0.5 * (rec.gx.min() + rec.gx.max())
    cy = 0.5 * (rec.gy.min() + rec.gy.max())
    Xg, Yg = np.meshgrid(rec.gx, rec.gy)
    lo, hi = nfw + 1, rec.T_all - n_lead - 2
    i0 = int(np.linspace(lo, hi, 6).astype(int)[win])

    rng = np.random.default_rng(700 + win)
    Zmem = rec.window(i0 - nfw + 1, nfw)
    Zfut = rec.window(i0, n_lead + 1)
    t0 = float(rec.t[i0])
    spread, lam_p, pdir_ax, kp = directional_spread(
        np.nan_to_num(Zmem[-1]), rec.dx, rec.dy)
    om_p = np.sqrt(G * kp * np.tanh(kp * depth))
    pdir, _ = resolve_direction(Zmem, rec, kp, pdir_ax, om_p / kp)
    w = np.array([np.cos(np.radians(pdir)), np.sin(np.radians(pdir))])
    h = -w
    T_tr = W_MAX + lead_max

    def ship_pos(t):
        return np.array([cx, cy]) + U * (t - (t0 - W_MAX) - T_tr / 2) * h

    r0, r1 = cfg['r_fan']
    noise = 0.02
    m_scan = 12
    P_all, z_all = [], []

    def sample_frame(t_j, Z_j):
        sp = ship_pos(t_j)
        RX, RY = Xg - sp[0], Yg - sp[1]
        R = np.hypot(RX, RY)
        ca = (RX * h[0] + RY * h[1]) / np.maximum(R, 1e-9)
        m = ((R >= r0) & (R <= r1) & (ca >= 0.5) & np.isfinite(Z_j))
        idx = np.flatnonzero(m)
        if len(idx) == 0:
            return
        pick = rng.choice(idx, min(m_scan, len(idx)), replace=False)
        iy, ix = np.unravel_index(pick, Z_j.shape)
        P_all.append(np.column_stack([rec.gx[ix], rec.gy[iy],
                                      np.full(len(pick), t_j)]))
        z_all.append(Z_j[iy, ix] + rng.normal(0, noise, len(pick)))

    for j in range(nfw):                       # memory window (as scene_real)
        sample_frame(float(rec.t[i0 - nfw + 1 + j]), Zmem[j])
    for jj in range(1, n_lead + 1):            # continuation into the lead span
        sample_frame(float(rec.t[i0 + jj]), Zfut[jj])

    P_all = np.vstack(P_all)
    z_all = np.concatenate(z_all)

    sq = max(1, int(round(4.0 / rec.dx)))
    Zq = Zfut[:, ::sq, ::sq]
    return dict(P_all=P_all, z_all=z_all, t0=t0, dt=float(rec.dt),
                depth=float(depth), lead_max=float(lead_max),
                gxq=rec.gx[::sq], gyq=rec.gy[::sq], Zq=Zq,
                sg=float(np.std(Zq[np.isfinite(Zq)])))


def assert_replication(st, record, win):
    """The memory-window part of the stream must equal sc['P'] exactly."""
    import d3_compare_gifs as cg
    sc = cg.scene_real(record, win)
    m = st['P_all'][:, 2] <= st['t0'] + 1e-9
    m &= st['P_all'][:, 2] >= st['t0'] - W_MAX
    P, z = st['P_all'][m], st['z_all'][m]
    if len(z) > 600:
        ii = np.random.default_rng(0).choice(len(z), 600, replace=False)
        P, z = P[ii], z[ii]
    assert np.array_equal(P, sc['P']) and np.array_equal(z, sc['z']), \
        f'stream replication mismatch on {record} w{win}'


def fit_at(st, t_f, fit_cache, timing):
    """Deployed fit on the trailing 40 s of the stream up to t_f (cap 600)."""
    from wavesplat.optimizer.bank_mmse import fit_bank_mmse
    key = round(t_f, 3)
    if key in fit_cache:
        return fit_cache[key]
    m = (st['P_all'][:, 2] >= t_f - W_MAX) & (st['P_all'][:, 2] <= t_f + 1e-9)
    P, z = st['P_all'][m], st['z_all'][m]
    if len(z) > 600:
        ii = np.random.default_rng(0).choice(len(z), 600, replace=False)
        P, z = P[ii], z[ii]
    s0 = 0.2 * float(np.std(z))
    t = time.time()
    fits = [fit_bank_mmse(P, z, st['depth'], s0, kp_hint=None, seed=s)
            for s in SEEDS]
    timing.append(time.time() - t)
    fit_cache[key] = fits
    return fits


def render_med(st, fits, tau, render_cache):
    key = (id(fits), round(tau, 3))
    if key in render_cache:
        return render_cache[key]
    gx, gy = st['gxq'], st['gyq']
    Xq, Yq = np.meshgrid(gx, gy)
    Q = np.column_stack([Xq.ravel(), Yq.ravel(),
                         np.full(Xq.size, st['t0'] + tau)])
    med = np.median(np.stack([f.render(Q) for f in fits], 0), 0) \
        .reshape(len(gy), len(gx))
    render_cache[key] = med
    return med


def main():
    results = {}
    for record, win in WINDOWS:
        key = f'{record}_w{win}'
        st = build_stream(record, win)
        assert_replication(st, record, win)
        taus = [float(v) for v in np.arange(0.0, st['lead_max'] + 1e-9, 1.0)]
        fit_cache, render_cache, timing = {}, {}, []
        entry = {'lead_max': st['lead_max'], 'arms': {}}
        for arm in ['frozen'] + [f'refit{int(R)}' for R in CADENCES]:
            R = None if arm == 'frozen' else float(arm[5:])
            curve = []
            for tau in taus:
                t_f = st['t0'] if R is None else \
                    st['t0'] + R * int(np.floor(tau / R + 1e-9))
                fits = fit_at(st, t_f, fit_cache, timing)
                pred = render_med(st, fits, tau, render_cache)
                j = int(round(tau / st['dt']))
                g = st['Zq'][j]
                m = np.isfinite(g)
                curve.append(float(np.sqrt(np.mean((pred[m] - g[m]) ** 2))
                                   / st['sg']))
            entry['arms'][arm] = dict(curve=curve,
                                      sustained_mean=float(np.mean(curve)))
        entry['n_fits'] = len(fit_cache)
        entry['fit_wall_s_m3'] = [round(v, 2) for v in timing]
        results[key] = entry
        line = ' '.join(f"{a}={v['sustained_mean']:.4f}"
                        for a, v in entry['arms'].items())
        print(f'{key}: {line} | fits {len(fit_cache)} '
              f'(m3 wall med {np.median(timing):.1f}s)', flush=True)
        json.dump(results, open(OUT, 'w'), indent=1)

    # record-level summary
    summary = {}
    for record in ('AA15a', 'AA15b', 'YS17'):
        ks = [k for k in results if k.startswith(record)]
        summary[record] = {a: float(np.mean(
            [results[k]['arms'][a]['sustained_mean'] for k in ks]))
            for a in results[ks[0]]['arms']}
    results['summary_record_mean'] = summary
    json.dump(results, open(OUT, 'w'), indent=1)
    print('\nSUMMARY (sustained mean nRMSE, record avg):', flush=True)
    for r, v in summary.items():
        print(' ', r, ' '.join(f'{a}={x:.4f}' for a, x in v.items()), flush=True)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
