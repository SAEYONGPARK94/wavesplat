"""Probe patch geometry / timing of all 13 Bergamasco .nc records.
NO scoring — data properties only (used to derive the frozen fan-geometry rule)."""
import os, sys, json, glob
os.environ.setdefault('OMP_NUM_THREADS', '1')
import warnings; warnings.filterwarnings('ignore')
import numpy as np
sys.path.insert(0, 'scripts')
from d3_sa_gonogo import WassRecord

FILES = sorted(glob.glob('data/real_bergamasco/*/*.nc'))
out = []
for p in FILES:
    try:
        r = WassRecord(p)
        Lx = float(r.gx.max() - r.gx.min()); Ly = float(r.gy.max() - r.gy.min())
        out.append(dict(path=p, nx=int(r.n1), ny=int(r.n0), T_all=int(r.T_all),
                        dx=float(r.dx), dy=float(r.dy), dt=float(r.dt),
                        Lx=Lx, Ly=Ly, dur=float(r.t[-1] - r.t[0]),
                        valid_frac=float(r.valid.mean())))
        r.f.close()
    except Exception as e:
        out.append(dict(path=p, error=repr(e)))
for o in out:
    if 'error' in o:
        print(f"{o['path']}: ERROR {o['error']}"); continue
    print(f"{o['path'].split('real_bergamasco/')[1]:55s} "
          f"grid {o['ny']:4d}x{o['nx']:4d} d={o['dx']:.2f}/{o['dy']:.2f}m "
          f"L={o['Ly']:6.1f}x{o['Lx']:6.1f}m  T={o['T_all']:5d} dt={o['dt']:.3f} "
          f"dur={o['dur']:6.1f}s valid={o['valid_frac']:.2f}")
json.dump(out, open('tmp/heldout_real_2026-07-26/geometry.json', 'w'), indent=1)
