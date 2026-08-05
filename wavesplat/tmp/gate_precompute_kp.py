"""M-3G — precompute sensor beamform kp for all bwtask training scenes.
2-stage beamform_top (identical recipe to M-3 eval / net_gamma_mmse kp_hint=None).
Cache -> tmp/bwtask_kp_beam.npy, indexed like load_all('tmp/prior_train/bwtask').
Usage: python3 tmp/gate_precompute_kp.py
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from d3_prior_net_taskft import load_all, DEPTH
from d18_relax_estimator import beamform_top

OUT = 'tmp/bwtask_kp_beam.npy'
DATA = 'tmp/prior_train/bwtask'


def beamform_kp(P, z):
    P32, z32 = np.asarray(P, np.float32), np.asarray(z, np.float32)
    bx, by = beamform_top(P32, z32, DEPTH, klim=2.0, n=120)
    kp = max(np.hypot(bx, by), 1e-3)
    bx, by = beamform_top(P32, z32, DEPTH, klim=3.5 * kp, n=120)
    return max(np.hypot(bx, by), 1e-3)


def main():
    D = load_all(DATA)
    n = len(D['P'])
    kp_e = np.array([m[0] for m in D['meta']])
    kp_beam = np.zeros(n)
    t0 = time.time()
    for i in range(n):
        kp_beam[i] = beamform_kp(D['P'][i], D['z'][i])
        if i % 1000 == 0:
            print(f"  {i}/{n}  [{time.time()-t0:.0f}s]", flush=True)
    np.save(OUT, kp_beam)
    # bias diagnostics (report for red-team / verdict)
    rel = kp_beam / kp_e
    print(f"\nsaved {n} kp_beam -> {OUT}  ({time.time()-t0:.0f}s)")
    print(f"kp_e   : median {np.median(kp_e):.4f}  range [{kp_e.min():.4f},{kp_e.max():.4f}]")
    print(f"kp_beam: median {np.median(kp_beam):.4f}  range [{kp_beam.min():.4f},{kp_beam.max():.4f}]")
    print(f"kp_beam/kp_e: median {np.median(rel):.3f}  IQR [{np.percentile(rel,25):.3f},{np.percentile(rel,75):.3f}]")
    print(f"  frac beam>1.2x oracle: {np.mean(rel>1.2):.2f}  |  beam<0.8x: {np.mean(rel<0.8):.2f}")
    print(f"  degenerate (kp_beam<=1e-3 or >5): {int(((kp_beam<=1.1e-3)|(kp_beam>5)).sum())}")


if __name__ == '__main__':
    main()
