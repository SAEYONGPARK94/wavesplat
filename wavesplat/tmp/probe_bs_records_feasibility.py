"""Feasibility probe for the record 5 -> 13 expansion (E-10 power).

Question: the RECORDS registry in d3_falcon_real_operator.py says
"only records whose patch can host a (possibly scaled) forward fan".
Were BS_2011 (4) / BS_2013 (3) excluded for a *geometric* reason, or are
they simply unregistered?

Prints, per record: patch extent, dx/dy, dt, T, valid fraction, Hs, and
the largest forward fan radius the patch can host.
"""
import sys
import glob
import numpy as np

sys.path.insert(0, 'scripts')
from d3_sa_gonogo import WassRecord  # noqa: E402

REGISTERED = {
    'YS_2017/Surfaces_20170513_050000.nc': ('YS17', (20.0, 100.0), 2),
    'AA_2015/Surfaces_20150305_103500.nc': ('AA15a', (8.0, 40.0), 1),
    'AA_2014/Surfaces_20140327_091000.nc': ('AA14', (8.0, 40.0), 1),
    'AA_2015/Surfaces_20150515_090000.nc': ('AA15b', (8.0, 40.0), 1),
    'LJ_2018/Surfaces_20180103_093938_Grid2_short.nc': ('LJ18g', (20.0, 100.0), 2),
}

rows = []
for path in sorted(glob.glob('data/real_bergamasco/*/*.nc')):
    key = '/'.join(path.split('/')[-2:])
    try:
        r = WassRecord(path)
    except Exception as e:              # noqa: BLE001
        rows.append((key, f'LOAD FAIL: {e}'))
        continue

    Lx = float(np.ptp(r.gx))
    Ly = float(np.ptp(r.gy))
    vf = float(r.valid.mean())

    # Hs from a few frames over the valid region
    idx = np.linspace(0, r.T_all - 1, 24).astype(int)
    z = []
    for i in idx:
        Z = r.window(i, 1)[0] if hasattr(r, 'window') else None
        if Z is not None:
            z.append(Z[r.valid])
    Hs = 4.0 * float(np.nanstd(np.concatenate(z))) if z else float('nan')

    # Largest forward fan the patch can host: fan is a half-disc of radius
    # r_max ahead of the sensor. Sensor sits at the patch edge, so we need
    # r_max <= min(Ly, Lx / 2) roughly (fan spans +-60 deg of 120 deg FOV).
    r_max = min(Ly, Lx / 2.0)

    reg = REGISTERED.get(key)
    tag = reg[0] if reg else '-- UNREGISTERED --'
    need = reg[1][1] if reg else None

    rows.append((key, dict(tag=tag, Lx=Lx, Ly=Ly, dx=r.dx, dy=r.dy, dt=r.dt,
                           T=r.T_all, valid=vf, Hs=Hs, r_max=r_max,
                           need=need)))
    r.f.close()

print(f"{'record':<46}{'tag':<20}{'extent [m]':>16}{'dx':>6}{'dt':>6}"
      f"{'T':>6}{'valid':>8}{'Hs':>7}{'r_fan_max':>11}")
print('-' * 125)
for key, v in rows:
    if isinstance(v, str):
        print(f'{key:<46}{v}')
        continue
    fits = ''
    if v['need'] is not None:
        fits = '  (registered, uses r=%.0f)' % v['need']
    print(f"{key:<46}{v['tag']:<20}"
          f"{v['Lx']:7.0f} x{v['Ly']:6.0f}"
          f"{v['dx']:6.2f}{v['dt']:6.2f}{v['T']:6d}"
          f"{v['valid']:8.2f}{v['Hs']:7.2f}{v['r_max']:11.0f}{fits}")
