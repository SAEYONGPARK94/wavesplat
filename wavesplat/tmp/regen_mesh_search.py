"""Search for minimal uniform mesh spacing d (with dx=(240)/(nx-1), dy=(200)/(ny-1),
nx,ny chosen so dx~=dy) that makes mesh_adequacy.check() report ADEQUATE for every
in-scope (BF,ncomp) family, using the REAL wave_components read from shipped h5 files."""
import h5py, glob, re, sys
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, '.')
from wavesim import mesh_adequacy

files = sorted(glob.glob('data/episode_BF[2-5]_*comp_seed*_*.h5'))
def parse(fn):
    m = re.match(r'episode_(BF\d)_(\d+)comp_seed(\d+)_(uniform|slope|fixedT)(_dir\d+)?(_pk)?\.h5', fn.split('/')[-1])
    if not m: return None
    bf, ncomp, seed, model, dirtag, pk = m.groups()
    return dict(bf=bf, ncomp=int(ncomp), seed=int(seed), model=model, dirtag=dirtag, pk=pk)

groups = {}  # (bf,ncomp) -> components array (use seed0 uniform as representative; verify seed-invariance of |k| already done)
for fp in files:
    meta = parse(fp)
    if meta is None or meta['dirtag'] or meta['pk'] or meta['model'] not in ('uniform','slope'):
        continue
    key = (meta['bf'], meta['ncomp'])
    if key in groups:
        continue
    if meta['seed'] != 0 or meta['model'] != 'uniform':
        continue
    with h5py.File(fp, 'r') as f:
        comp = f['config/wave_components'][:]
    groups[key] = comp

print("Groups found:", sorted(groups.keys()))

def make_components(comp_arr):
    return [SimpleNamespace(A=row[0], kx=row[2], ky=row[3]) for row in comp_arr]

def make_cfg(nx, ny):
    return SimpleNamespace(
        gt_grid=SimpleNamespace(x_range=(10.0,250.0), y_range=(-100.0,100.0)),
        mesh=SimpleNamespace(nx=nx, ny=ny))

X_SPAN, Y_SPAN = 240.0, 200.0

def nxny_for_d(d):
    nx = int(round(X_SPAN/d)) + 1
    ny = int(round(Y_SPAN/d)) + 1
    return nx, ny

# fine-to-coarse binary-ish search over d
def verdict_all(d):
    nx, ny = nxny_for_d(d)
    cfg = make_cfg(nx, ny)
    out = {}
    for key, comp in groups.items():
        comps = make_components(comp)
        info = mesh_adequacy.check(comps, cfg, hos_order=None)
        out[key] = info
    return nx, ny, out

# Sweep d from 0.3 to 7.0
print(f"\n{'d':>6} {'nx':>4} {'ny':>4} {'dx':>6} {'dy':>6}  worst_verdict  #ADEQ #DEG #INADEQ  worst_group")
import numpy as np
for d in [7.0,6.9,6.15,5,4,3,2.5,2,1.5,1.4,1.37,1.36,1.35,1.3,1.2,1.1,1.05,1.04,1.0,0.9,0.8,0.7,0.6,0.5]:
    nx, ny = nxny_for_d(d)
    dx = X_SPAN/(nx-1); dy = Y_SPAN/(ny-1)
    _, _, out = verdict_all(d)
    verdicts = {k: v['mesh_verdict'] for k,v in out.items()}
    n_adeq = sum(1 for v in verdicts.values() if v=='ADEQUATE')
    n_deg = sum(1 for v in verdicts.values() if v=='DEGRADED')
    n_inadeq = sum(1 for v in verdicts.values() if v=='INADEQUATE')
    # worst = INADEQUATE > DEGRADED > ADEQUATE
    order = {'INADEQUATE':2,'DEGRADED':1,'ADEQUATE':0}
    worst_key = max(verdicts, key=lambda k: order[verdicts[k]])
    print(f"{d:6.2f} {nx:4d} {ny:4d} {dx:6.3f} {dy:6.3f}  worst={verdicts[worst_key]:10s} "
          f"{n_adeq:5d} {n_deg:4d} {n_inadeq:7d}  worst_group={worst_key}")
