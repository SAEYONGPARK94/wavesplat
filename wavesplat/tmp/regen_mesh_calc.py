"""Compute required mesh resolution from real shipped wave_components.
Read-only: does not touch data/*.h5 contents, only inspects."""
import h5py, glob, numpy as np, re, sys
sys.path.insert(0, '.')
from wavesim import mesh_adequacy

# ---- gather in-scope shipped files: BF2-BF5, uniform/slope, comp in {1,2,5,15,30}, exclude dir180 ----
files = sorted(glob.glob('data/episode_BF[2-5]_*comp_seed*_*.h5'))
def parse(fn):
    m = re.match(r'episode_(BF\d)_(\d+)comp_seed(\d+)_(uniform|slope|fixedT)(_dir\d+)?(_pk)?\.h5', fn.split('/')[-1])
    if not m: return None
    bf, ncomp, seed, model, dirtag, pk = m.groups()
    return dict(bf=bf, ncomp=int(ncomp), seed=int(seed), model=model, dirtag=dirtag, pk=pk)

rows = []
for fp in files:
    meta = parse(fp)
    if meta is None:
        continue
    if meta['dirtag'] or meta['pk']:
        continue
    if meta['model'] not in ('uniform','slope'):
        continue
    with h5py.File(fp,'r') as f:
        comp = f['config/wave_components'][:]  # [A, omega, kx, ky, phi, Q]
        cfg_attrs = dict(f['config'].attrs)
    A = comp[:,0]; kx = comp[:,2]; ky = comp[:,3]
    k = np.hypot(kx,ky)
    lam = 2*np.pi/np.maximum(k,1e-12)
    var = A**2
    keep = var >= 0.01*var.sum()
    lam_min = lam[keep].min()
    lam_peak = lam[np.argmax(A)]
    rows.append(dict(fp=fp, bf=meta['bf'], ncomp=meta['ncomp'], seed=meta['seed'],
                      model=meta['model'], lam_min=lam_min, lam_peak=lam_peak,
                      n_true_comp=len(A)))

print(f"Loaded {len(rows)} in-scope shipped files (BF2-5, uniform/slope, no dir180/pk)")

# Group by (bf,ncomp) -- lam should be seed/model-independent? Actually kx/ky include
# random *direction* draw (rng), so |k| depends only on omega (seed-independent draw of
# omega bin centers -- wait omega_centers is deterministic (linspace), only phases/
# directions are randomized) -- but direction changes kx,ky COMPONENTS not the magnitude
# |k| (direction rotation preserves |k|). So lam should be IDENTICAL across seed and
# across raydrop model for the same (bf, ncomp). Verify this.
import collections
groups = collections.defaultdict(list)
for r in rows:
    groups[(r['bf'], r['ncomp'])].append(r)

print("\n=== per (BF, ncomp): lambda_min / lambda_peak variability across seed+model ===")
for key in sorted(groups):
    grp = groups[key]
    lam_mins = [g['lam_min'] for g in grp]
    lam_peaks = [g['lam_peak'] for g in grp]
    print(f"{key}: n_files={len(grp)}  lam_min range=[{min(lam_mins):.4f},{max(lam_mins):.4f}]"
          f"  lam_peak range=[{min(lam_peaks):.4f},{max(lam_peaks):.4f}]")

