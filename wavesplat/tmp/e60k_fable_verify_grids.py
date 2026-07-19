"""fable red-team v2 -- independent verification of R3 grid sizes.

Loads every BS record (and the current 5) directly and reports grid extent,
dx, and whether an r_fan=(8,40) or (2,10) fan chord can fit inside.
Also re-derives the site-day mapping from the FILENAMES (primary source),
not from any hardcoded dict.
"""
import os
import re
import sys

import numpy as np

sys.path.insert(0, 'scripts')
from d3_sa_multirecord import MRecord  # noqa: E402

FILES = {
    'BS11a': 'data/real_bergamasco/BS_2011/Surfaces_20111004_113800_short.nc',
    'BS11b': 'data/real_bergamasco/BS_2011/Surfaces_20111001_161800_short.nc',
    'BS11c': 'data/real_bergamasco/BS_2011/Surfaces_20111004_130700_short.nc',
    'BS11d': 'data/real_bergamasco/BS_2011/Surfaces_20111004_153000_short.nc',
    'BS13a': 'data/real_bergamasco/BS_2013/Surfaces_20130925_121501_short.nc',
    'BS13b': 'data/real_bergamasco/BS_2013/Surfaces_20130930_102001_short.nc',
    'BS13c': 'data/real_bergamasco/BS_2013/Surfaces_20130922_130001_short.nc',
    'AA15a': 'data/real_bergamasco/AA_2015/Surfaces_20150305_103500.nc',
    'AA15b': 'data/real_bergamasco/AA_2015/Surfaces_20150515_090000.nc',
    'AA14': 'data/real_bergamasco/AA_2014/Surfaces_20140327_091000.nc',
    'YS17': 'data/real_bergamasco/YS_2017/Surfaces_20170513_050000.nc',
    'LJ18g': 'data/real_bergamasco/LJ_2018/Surfaces_20180103_093938_Grid2_short.nc',
}

print(f'{"rec":7s} {"site-day (from filename)":22s} {"Lx x Ly [m]":16s} '
      f'{"dx":6s} fits(8,40)? fits(2,10)?')
for name, path in sorted(FILES.items()):
    m = re.search(r'Surfaces_(\d{4})(\d{2})(\d{2})_', path)
    day = f'{m.group(1)}-{m.group(2)}-{m.group(3)}'
    site = os.path.basename(os.path.dirname(path)).split('_')[0]
    try:
        rec = MRecord(path, stride=1, robust=False)
        lx = float(rec.gx.max() - rec.gx.min())
        ly = float(rec.gy.max() - rec.gy.min())
        dx = float(rec.dx)
        # a forward fan needs a chord of length r1-r0 starting r0 from the
        # sensor edge; minimal hosting box ~ r1 in one dimension
        f84 = 'yes' if min(lx, ly) >= 40.0 else 'NO'
        f21 = 'yes' if min(lx, ly) >= 10.0 else 'NO'
        print(f'{name:7s} {site}_{day:17s} {lx:6.1f} x {ly:6.1f}   '
              f'{dx:5.2f} {f84:>10s} {f21:>10s}')
    except Exception as e:
        print(f'{name:7s} {site}_{day:17s} LOAD FAIL: {e}')
