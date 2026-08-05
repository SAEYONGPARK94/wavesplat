import h5py, glob, re, sys
from types import SimpleNamespace
sys.path.insert(0, '.')
from wavesim import mesh_adequacy

files = sorted(glob.glob('data/episode_BF[2-5]_*comp_seed0_uniform.h5'))
def parse(fn):
    m = re.match(r'episode_(BF\d)_(\d+)comp_seed(\d+)_(uniform|slope|fixedT)(_dir\d+)?(_pk)?\.h5', fn.split('/')[-1])
    return m.groups() if m else None

def make_components(comp_arr):
    return [SimpleNamespace(A=row[0], kx=row[2], ky=row[3]) for row in comp_arr]

def cfg(nx,ny):
    return SimpleNamespace(gt_grid=SimpleNamespace(x_range=(10.0,250.0), y_range=(-100.0,100.0)),
                            mesh=SimpleNamespace(nx=nx, ny=ny))

rows=[]
for fp in files:
    meta = parse(fp)
    if meta is None or meta[4] or meta[5]:
        continue
    bf, ncomp = meta[0], int(meta[1])
    with h5py.File(fp,'r') as f:
        comp = f['config/wave_components'][:]
    comps = make_components(comp)
    r40 = mesh_adequacy.check(comps, cfg(40,30))
    r180 = mesh_adequacy.check(comps, cfg(180,150))
    r200 = mesh_adequacy.check(comps, cfg(200,167))
    rows.append((bf,ncomp,r40,r180,r200))

print(f"{'BF':4s}{'ncomp':6s}{'lam_peak':9s}{'lam_min':8s}| {'40x30':>9s} {'pts/lam':>8s} {'var_und':>8s} | {'180x150':>9s} {'pts/lam':>8s} {'var_und':>8s} | {'200x167':>9s}")
for bf,ncomp,r40,r180,r200 in rows:
    print(f"{bf:4s}{ncomp:<6d}{r40['lambda_peak']:9.2f}{r40['lambda_min']:8.2f}| "
          f"{r40['mesh_verdict']:>9s} {r40['pts_per_lambda']:8.2f} {100*r40['var_under_resolved']:7.1f}% | "
          f"{r180['mesh_verdict']:>9s} {r180['pts_per_lambda']:8.2f} {100*r180['var_under_resolved']:7.1f}% | "
          f"{r200['mesh_verdict']:>9s}")
