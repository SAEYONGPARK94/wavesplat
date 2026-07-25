"""DESIGN-PHASE (pre-freeze, in-sample allowed): decisive feasibility test for the two
candidate mechanisms. Uses ONLY existing atom caches + read-only episode GT.

Q1 (candidate 1 = PEAK-ANCHORED FILTERING): on the catastrophic BF2_1comp slope episodes,
    does the diffusion proposal place ANY selected atom near the TRUE swell / the beamform
    dominant peak? If yes, keeping atoms within a kp-scaled radius of a significant beamform
    peak isolates the good atom and discards the scatter. If no, candidate 1 cannot rescue.

Q2 (candidate 2 = HOLDOUT-RESIDUAL SAMPLE SELECTION): split the 4 diffusion samples (32 atoms
    each) and ask whether held-out obs residual after LS w/phi separates a GOOD sample (atom
    near truth) from the garbage samples on the catastrophic episodes. Gross-garbage rejection.
"""
import os, sys, json, re
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import h5py
sys.path.insert(0, os.path.abspath('.'))
from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.optimizer.initializer import _collect_observations, _beamform_coarse, MIN_DK
from wavesplat.optimizer.learned_init import pool_and_select, ls_fit_wphi
from wavesplat.model.tnse import render_wave_field

FINE = 'data_finemesh_2026-07-23/'
BATCHC = 'tmp/rerun_2026-07-25/batchC_ws_phasefix.jsonl'
LANES = [(42, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
G = 9.81


def comp(ep): return int(re.search(r'_(\d+)comp', ep).group(1))
def med(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else float('nan')


def beamform_peaks(frames, depth, frac=0.1, n_extract=30):
    """Return list of (kx,ky,power) significant peaks (>= frac*P1), v2 convention."""
    x, y, z, t = _collect_observations(frames)
    if x is None or len(x) < 10:
        return [], None
    power, kx_axis, ky_axis = _beamform_coarse(x, y, z, t, depth, (-4, 4), (-4, 4), 81, 81)
    dkx = kx_axis[1]-kx_axis[0]; dky = ky_axis[1]-ky_axis[0]
    cx = np.argmin(np.abs(kx_axis)); cy = np.argmin(np.abs(ky_axis))
    power[max(0, cy-1):cy+2, max(0, cx-1):cx+2] = 0.0
    supx = int(MIN_DK/dkx)+1; supy = int(MIN_DK/dky)+1
    peaks = []; P = power.copy()
    for _ in range(n_extract):
        idx = int(np.argmax(P.ravel())); iy, ix = np.unravel_index(idx, P.shape)
        pk = float(P[iy, ix])
        if pk < 1e-12: break
        peaks.append((float(kx_axis[ix]), float(ky_axis[iy]), pk))
        yl, yh = max(0, iy-supy), min(P.shape[0], iy+supy+1)
        xl, xh = max(0, ix-supx), min(P.shape[1], ix+supx+1)
        P[yl:yh, xl:xh] = 0.0
    if not peaks: return [], None
    p1 = peaks[0][2]
    sig = [(kx, ky, pw) for (kx, ky, pw) in peaks if pw >= frac*p1]
    return sig, peaks[0]


def true_comps(ep):
    with h5py.File(FINE+ep, 'r') as f:
        wc = np.array(f['config/wave_components'])  # [A, om, kx, ky, phi, Q]
    return wc[:, 2], wc[:, 3], wc[:, 0]  # kx, ky, A


def main():
    eps = sorted(set(json.loads(l)['episode'] for l in open(BATCHC)))
    sparse = [e for e in eps if comp(e) in (1, 2)]
    cache = {}
    for src in ('tmp/wp_integration_2026-07-25/atoms_cache.jsonl',
                'tmp/unified_2026-07-26/atoms_cache_extra.jsonl'):
        for l in open(src):
            d = json.loads(l)
            if d.get('kx') is not None and d.get('err') is None:
                cache[(d['episode'], d['diff_seed'])] = d

    print("="*118)
    print("Q1: distance from TRUE swell to nearest SELECTED proposal atom (median over 5 diff_seeds), kp-units")
    print("   dom_bf = dist from beamform dominant peak to nearest selected atom (kp-units); bf_err = |bf peak - true|/kp")
    print("="*118)
    print(f"{'episode':38s} {'d_true/kp':>10} {'d_true_abs':>10} {'dom_bf/kp':>10} {'bf_err/kp':>10} {'nsig_bf':>7} {'nsel':>5}")
    rows = []
    for ep in sparse:
        L = EpisodeDataLoader(FINE+ep)
        fr = L.get_init_window(2.0)
        sig, dom = beamform_peaks(fr, L.depth)
        tkx, tky, tA = true_comps(ep)
        # dominant true component
        di = int(np.argmax(tA)); tkxd, tkyd = tkx[di], tky[di]
        kp = float(np.hypot(tkxd, tkyd))
        d_true, d_dom, nsel_list = [], [], []
        for (E, D) in LANES:
            d = cache.get((ep, D))
            if d is None: continue
            sk, sy, se = pool_and_select(np.asarray(d['kx'], float), np.asarray(d['ky'], float),
                                         np.asarray(d['energy'], float))
            nsel_list.append(len(sk))
            if len(sk) == 0:
                d_true.append(np.inf); d_dom.append(np.inf); continue
            # nearest selected atom to the dominant true component
            dist = np.min(np.hypot(sk - tkxd, sy - tkyd))
            d_true.append(dist)
            if dom is not None:
                dd = np.min(np.hypot(sk - dom[0], sy - dom[1]))
                d_dom.append(dd)
        d_true_med = med(d_true); d_dom_med = med(d_dom)
        bf_err = float(np.hypot(dom[0]-tkxd, dom[1]-tkyd))/kp if dom is not None else np.nan
        rows.append(dict(ep=ep, comp=comp(ep), var=('slope' if 'slope' in ep else 'uniform'),
                         d_true_kp=d_true_med/kp, d_true_abs=d_true_med, dom_bf_kp=d_dom_med/kp,
                         bf_err_kp=bf_err, nsig=len(sig), nsel=med(nsel_list), kp=kp,
                         bad=('BF2_1comp' in ep and 'slope' in ep)))
        mark = ' <== catastrophic' if rows[-1]['bad'] else ''
        print(f"{ep:38s} {d_true_med/kp:10.3f} {d_true_med:10.3f} {d_dom_med/kp:10.3f} "
              f"{bf_err:10.3f} {len(sig):7d} {med(nsel_list):5.0f}{mark}")

    json.dump(rows, open('tmp/learned_only_2026-07-26/design_diag_q1.json', 'w'), indent=0)

    bad = [r for r in rows if r['bad']]
    good1 = [r for r in rows if r['comp'] == 1 and not r['bad']]
    print("\nSUMMARY Q1 (candidate 1 feasibility):")
    print(f"  catastrophic BF2_1comp_slope (n={len(bad)}): d_true/kp med={med([r['d_true_kp'] for r in bad]):.3f} "
          f"[{min(r['d_true_kp'] for r in bad):.3f},{max(r['d_true_kp'] for r in bad):.3f}]; "
          f"dom_bf/kp med={med([r['dom_bf_kp'] for r in bad]):.3f}; bf_err/kp med={med([r['bf_err_kp'] for r in bad]):.3f}")
    print(f"  other 1comp (n={len(good1)}): d_true/kp med={med([r['d_true_kp'] for r in good1]):.3f}")
    print("  => candidate-1 CAN work iff proposal HAS an atom near truth (d_true/kp small) AND")
    print("     the beamform peak is at truth (bf_err/kp small) so anchoring keeps that atom.")

    # ---- Q2: per-sample held-out residual selection on the catastrophic episodes ----
    print("\n" + "="*118)
    print("Q2: candidate-2 per-sample (32-atom) held-out residual vs per-sample atom-quality, catastrophic eps")
    print("="*118)
    M = 32
    q2 = []
    for ep in [r['ep'] for r in bad]:
        L = EpisodeDataLoader(FINE+ep); fr = L.get_init_window(2.0); depth = L.depth
        tkx, tky, tA = true_comps(ep); di = int(np.argmax(tA)); tkxd, tkyd = tkx[di], tky[di]
        kp = float(np.hypot(tkxd, tkyd))
        # held-out split of init obs
        ox, oy, oz, ot = _collect_observations(fr)
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(oz)); nh = len(oz)//3
        ho = idx[:nh]; fitidx = idx[nh:]
        class F:  # lightweight frame-like for ls_fit on a subset
            pass
        for (E, D) in LANES:
            d = cache.get((ep, D))
            if d is None: continue
            rkx = np.asarray(d['kx'], float); rky = np.asarray(d['ky'], float); ren = np.asarray(d['energy'], float)
            nsamp = len(rkx)//M
            per = []
            for s in range(nsamp):
                sl = slice(s*M, (s+1)*M)
                sk, sy, se = pool_and_select(rkx[sl], rky[sl], ren[sl])
                if len(sk) == 0:
                    per.append((np.inf, np.inf, 0)); continue
                dtrue = float(np.min(np.hypot(sk - tkxd, sy - tkyd)))
                # LS fit w/phi on fit-subset, residual on held-out
                from wavesplat.model.tnse import fit_amplitudes_phases
                fit_xyz = np.column_stack([ox[fitidx], oy[fitidx], ot[fitidx], oz[fitidx]]).astype(np.float64)
                A, phi = fit_amplitudes_phases(sk, sy, fit_xyz, depth, lambda_reg=0.5)
                # render at held-out points, residual
                def render_pts(kx, ky, w, phi, X, Y, T):
                    out = np.zeros(len(X))
                    for i in range(len(kx)):
                        km = np.hypot(kx[i], ky[i]); om = np.sqrt(G*km*np.tanh(km*depth))
                        out += w[i]*np.cos(kx[i]*X+ky[i]*Y-om*T+phi[i])
                    return out
                pred = render_pts(sk, sy, np.abs(A), phi, ox[ho], oy[ho], ot[ho])
                resid = float(np.sqrt(np.mean((pred - oz[ho])**2)))
                per.append((resid, dtrue/kp, len(sk)))
            # does min-residual sample = min-d_true sample?
            resids = [p[0] for p in per]; dts = [p[1] for p in per]
            best_res = int(np.argmin(resids)); best_dt = int(np.argmin(dts))
            q2.append(dict(ep=ep, diff_seed=D, per=per,
                           picks_good=(per[best_res][1] < 0.375),
                           best_dt_val=dts[best_dt], best_res_dt=per[best_res][1],
                           any_good=(min(dts) < 0.375)))
    # summarize
    print(f"{'episode':38s} {'ds':>3} {'residuals(4samp)':>34} {'d_true/kp(4samp)':>34} {'pick_good':>9}")
    for r in q2:
        rs = ' '.join(f'{p[0]:.3f}' for p in r['per'])
        dt = ' '.join(f'{p[1]:.2f}' for p in r['per'])
        print(f"{r['ep']:38s} {r['diff_seed']:3d} {rs:>34} {dt:>34} {str(r['picks_good']):>9}")
    n_anygood = sum(1 for r in q2 if r['any_good'])
    n_pickgood = sum(1 for r in q2 if r['picks_good'])
    print(f"\nSUMMARY Q2 (candidate 2): lanes where SOME sample is good (d_true/kp<0.375): {n_anygood}/{len(q2)}")
    print(f"  lanes where min-held-out-residual sample IS the good one: {n_pickgood}/{len(q2)}")
    print("  => candidate 2 alive iff both counts high (a good sample exists AND residual finds it).")
    json.dump(q2, open('tmp/learned_only_2026-07-26/design_diag_q2.json', 'w'), indent=0, default=float)


if __name__ == '__main__':
    main()
