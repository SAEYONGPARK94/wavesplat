"""FROZEN held-out record configuration for the scene_real path.

Rule (pre-registered; derived from the 5 DESIGN records only, never from any score):
  depth, stride  : taken verbatim from the pre-existing committed config
                   scripts/d3_sa_multirecord.py RECORDS (BS site = 30.0 m, LJ = 45.0 m;
                   BS stride 1, LJ stride 2).  No new numbers invented.
  Lmin           : min(Lx, Ly) of the patch, measured from the file.
  scale          : Lmin / 55.0   (55.0 m = AA15b, the SMALLEST design patch that hosts
                   the design '0.4x-scaled' fan r_fan=(8,40) m at U=1.2 m/s).
  r_fan          : (8*scale, 40*scale)      -- geometric similarity to the AA recipe
  U              : 1.2*scale                -- keeps ship-traverse / patch-size fixed
  lead_max       : 14.0 (AA design-family value, unscaled; lead-0 nowcast is the endpoint)

No file under scripts/ is modified: the entries are injected into the live
d3_falcon_real_operator.RECORDS dict at runtime (additive only).
"""
import numpy as np

DESIGN = ['YS17', 'AA15a', 'AA14', 'AA15b', 'LJ18g']

# path, depth, stride  -- depth/stride from d3_sa_multirecord.RECORDS (or same-site value)
_HELDOUT_RAW = {
    'BS11a': ('data/real_bergamasco/BS_2011/Surfaces_20111004_113800_short.nc', 30.0, 1),
    'BS11b': ('data/real_bergamasco/BS_2011/Surfaces_20111001_161800_short.nc', 30.0, 1),
    'BS11c': ('data/real_bergamasco/BS_2011/Surfaces_20111004_130700_short.nc', 30.0, 1),
    'BS11d': ('data/real_bergamasco/BS_2011/Surfaces_20111004_153000_short.nc', 30.0, 1),
    'BS13a': ('data/real_bergamasco/BS_2013/Surfaces_20130925_121501_short.nc', 30.0, 1),
    'BS13b': ('data/real_bergamasco/BS_2013/Surfaces_20130930_102001_short.nc', 30.0, 1),
    'BS13c': ('data/real_bergamasco/BS_2013/Surfaces_20130922_130001_short.nc', 30.0, 1),
    'LJ18n': ('data/real_bergamasco/LJ_2018/Surfaces_20180103_093938_Grid1_short.nc', 45.0, 2),
}
# Lmin [m] measured by tmp/heldout_real_2026-07-26/probe_geometry.py (data property only)
_LMIN = {'BS11a': 13.5, 'BS11b': 13.0, 'BS11c': 13.5, 'BS11d': 14.5,
         'BS13a': 19.0, 'BS13b': 17.0, 'BS13c': 19.0, 'LJ18n': 40.0}
_REF_LMIN, _REF_FAN, _REF_U, _LEAD = 55.0, (8.0, 40.0), 1.2, 14.0


def heldout_records():
    out = {}
    for k, (path, depth, stride) in _HELDOUT_RAW.items():
        s = _LMIN[k] / _REF_LMIN
        out[k] = dict(path=path, depth=depth, stride=stride,
                      r_fan=(_REF_FAN[0] * s, _REF_FAN[1] * s), U=_REF_U * s,
                      lead_max=_LEAD, note=f'HELD-OUT; Lmin={_LMIN[k]} m, scale={s:.4f}')
    return out


def install():
    """Inject held-out entries into the live RECORDS dict (additive, no file edits)."""
    import d3_falcon_real_operator as fro
    for k, v in heldout_records().items():
        if k in fro.RECORDS:                       # idempotent (pool re-uses workers)
            assert fro.RECORDS[k] == v, f'{k} entry changed after install'
            continue
        assert k not in DESIGN, f'{k} collides with a design record'
        fro.RECORDS[k] = v
    return sorted(heldout_records())
