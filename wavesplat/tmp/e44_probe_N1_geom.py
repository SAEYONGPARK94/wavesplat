"""e44 probe N1-geom — WHERE does the argmax's +0.037 score gap live?

For a selected subset of si=300..347 seas, re-run the full 952,560-hypothesis
enumeration (exactly as e43c) and then:
  1. swap-correct the argmax's direction errors (label swap (l1,l2,w)->(l2,l1,1-w)
     is GAMMA-IDENTICAL, so the stored per-lobe errors are bookkeeping, not physics)
  2. block-substitution decomposition of the commit gap:
       sc_full      = commit to argmax node              (as e43c)
       sc_node_orc  = commit to true node                (as e43c)
       sc_hybD      = argmax rings & weight, TRUE dirs   (fix directions only)
       sc_hybR      = argmax dirs, TRUE rings & weight   (fix rings/weight only)
     => gap := sc_node_orc - sc_full splits into direction / ring+w / interaction.

Read-only wrt existing scripts; output tmp/e44_probe_N1_geom.json
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from e29_gennet_C import DEPTH
from e37_bimodal_ladder import build_sea_crosstp, ang_err_deg, net_gamma_map
from e29_gate0_ladder import solve_with_grid
from e42_detector_v2 import JointEngine, DIRS, A_GRID, W_GRID, MIN_SEP_DEG
from d3_sa_achievability_gap import dispersion_lam

OUT = 'tmp/e44_probe_N1_geom.json'

# swap-tie candidates {304,322,337}, genuine-far {306,307,318,321,323},
# near-but-costly {305,341}, near-good controls {303,331}
SEAS = [303, 304, 305, 306, 307, 318, 321, 322, 323, 331, 337, 341]


def nearest(arr, v):
    return int(np.argmin(np.abs(np.asarray(arr) - v)))


def nearest_dir(v):
    d = np.angle(np.exp(1j * (DIRS - v)))
    return int(np.argmin(np.abs(d)))


def main():
    t0 = time.time()
    rows = []
    for si in SEAS:
        S = build_sea_crosstp(si)
        eng = JointEngine(S)
        keys = [(round(float(x), 3), round(float(m), 5))
                for m in DIRS for x in A_GRID]
        eng.build_lobes(keys)
        lut = {k: eng._lobe[k] for k in keys}

        def idx_of(i_dir, i_a):
            return lut[(round(float(A_GRID[i_a]), 3),
                        round(float(DIRS[i_dir]), 5))]

        hyp, II, JJ, WW = [], [], [], []
        for i1 in range(len(DIRS)):
            for i2 in range(len(DIRS)):
                sep = abs(np.rad2deg(np.angle(np.exp(1j * (DIRS[i2] - DIRS[i1])))))
                if sep < MIN_SEP_DEG:
                    continue
                for ia1 in range(len(A_GRID)):
                    for ia2 in range(len(A_GRID)):
                        for iw in range(len(W_GRID)):
                            hyp.append((i1, ia1, i2, ia2, iw))
                            II.append(idx_of(i1, ia1))
                            JJ.append(idx_of(i2, ia2))
                            WW.append(W_GRID[iw])
        loge = eng.logev_pairs(II, JJ, WW)
        best = int(np.argmax(loge))
        i1, ia1, i2, ia2, iw = hyp[best]
        w = float(W_GRID[iw])

        # true node
        a1t = (2 * np.pi / dispersion_lam(7.0, DEPTH)) / S['kp']
        a2t = (2 * np.pi / dispersion_lam(S['Tp2'], DEPTH)) / S['kp']
        Ti = dict(i_d1=nearest_dir(S['d1']), i_a1=nearest(A_GRID, a1t),
                  i_d2=nearest_dir(S['d2']), i_a2=nearest(A_GRID, a2t),
                  i_w=nearest(W_GRID, S['r2']))
        r2 = float(S['r2'])

        # ---- swap correction: match argmax lobes to true lobes
        eA = (ang_err_deg(DIRS[i1], S['d1']) + ang_err_deg(DIRS[i2], S['d2']))
        eB = (ang_err_deg(DIRS[i1], S['d2']) + ang_err_deg(DIRS[i2], S['d1']))
        swap = bool(eB < eA)
        if not swap:   # L1<->T1, L2<->T2
            d_err = (float(ang_err_deg(DIRS[i1], S['d1'])),
                     float(ang_err_deg(DIRS[i2], S['d2'])))
            ring_rel = (float(abs(A_GRID[ia1] - a1t) / a1t),
                        float(abs(A_GRID[ia2] - a2t) / a2t))
            w_err = float(abs(w - r2))
            td1, td2 = Ti['i_d1'], Ti['i_d2']          # matched true dirs
            tr1, tr2 = Ti['i_a1'], Ti['i_a2']          # matched true rings
            tw1, tw2 = 1 - r2, r2                      # matched true weights
        else:          # L1<->T2, L2<->T1
            d_err = (float(ang_err_deg(DIRS[i1], S['d2'])),
                     float(ang_err_deg(DIRS[i2], S['d1'])))
            ring_rel = (float(abs(A_GRID[ia1] - a2t) / a2t),
                        float(abs(A_GRID[ia2] - a1t) / a1t))
            w_err = float(abs((1 - w) - r2))
            td1, td2 = Ti['i_d2'], Ti['i_d1']
            tr1, tr2 = Ti['i_a2'], Ti['i_a1']
            tw1, tw2 = r2, 1 - r2

        G = eng.grids
        # commit arms
        g_full = (1 - w) * G[idx_of(i1, ia1)] + w * G[idx_of(i2, ia2)]
        g_orc = (1 - r2) * G[idx_of(Ti['i_d1'], Ti['i_a1'])] \
            + r2 * G[idx_of(Ti['i_d2'], Ti['i_a2'])]
        # hybrid-D: TRUE dirs, argmax rings & weight (per matched lobe)
        g_hD = (1 - w) * G[idx_of(td1, ia1)] + w * G[idx_of(td2, ia2)]
        # hybrid-R: argmax dirs, TRUE rings & weight
        g_hR = tw1 * G[idx_of(i1, tr1)] + tw2 * G[idx_of(i2, tr2)]

        sc_full = solve_with_grid(S, g_full)
        sc_orc = solve_with_grid(S, g_orc)
        sc_hD = solve_with_grid(S, g_hD)
        sc_hR = solve_with_grid(S, g_hR)
        Yn = net_gamma_map(S)
        sc_cur = solve_with_grid(S, np.maximum(10.0 ** Yn, 1e-8))

        rows.append(dict(
            si=si, swap=swap, d_err=d_err, ring_rel=ring_rel, w_err=w_err,
            w_argmax=w, r2=r2,
            dlogE_best_minus_true=float(loge[best]
                                        - loge[np.argmax([h == (Ti['i_d1'], Ti['i_a1'], Ti['i_d2'], Ti['i_a2'], Ti['i_w']) for h in hyp])]) if (Ti['i_d1'], Ti['i_a1'], Ti['i_d2'], Ti['i_a2'], Ti['i_w']) in set(hyp) else None,
            sc_full=float(sc_full), sc_node_orc=float(sc_orc),
            sc_hybD=float(sc_hD), sc_hybR=float(sc_hR), sc_cur=float(sc_cur)))
        r = rows[-1]
        gap = r['sc_node_orc'] - r['sc_full']
        print(f"si{si} swap={int(swap)} dErr {d_err[0]:5.1f}/{d_err[1]:5.1f} "
              f"ring {ring_rel[0]:.2f}/{ring_rel[1]:.2f} wErr {w_err:.2f} | "
              f"full {sc_full:+.3f} orc {sc_orc:+.3f} gap {gap:+.3f} | "
              f"fixD {sc_hD - sc_full:+.3f} fixRW {sc_hR - sc_full:+.3f} | "
              f"cur {sc_cur:+.3f}  [{time.time()-t0:.0f}s]", flush=True)
        del eng
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    a = lambda k: np.array([r[k] for r in rows], float)
    gap = a('sc_node_orc') - a('sc_full')
    fixD = a('sc_hybD') - a('sc_full')
    fixRW = a('sc_hybR') - a('sc_full')
    print(f"\n=== e44 N1 geom (n={len(rows)}) ===")
    print(f"  gap (orc-full)          : mean {gap.mean():+.4f}")
    print(f"  fix DIRS only recovers  : mean {fixD.mean():+.4f}")
    print(f"  fix RING+W only recovers: mean {fixRW.mean():+.4f}")
    de = np.array([max(r['d_err']) for r in rows])
    print(f"  swap-corrected max dir err: median {np.median(de):.1f} deg "
          f"(stored errors were swap-uncorrected)")
    sub = [r for r in rows if max(r['d_err']) < 30]
    if sub:
        g2 = np.array([r['sc_node_orc'] - r['sc_full'] for r in sub])
        print(f"  seas with corrected dir err <30 deg: n={len(sub)}, "
              f"gap mean {g2.mean():+.4f}")
    json.dump(dict(rows=rows), open(OUT, 'w'), indent=1, default=float)
    print(f"saved -> {OUT} ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
