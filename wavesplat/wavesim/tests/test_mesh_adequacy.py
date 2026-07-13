"""The raycast mesh must be able to represent the sea it renders.

Regression guard for the bug that made the BF2 1-component "information limit"
(docs/E2_MESH_AUDIT_prereg_2026-07-13.md): the shipped default mesh renders an
8.2 m wave at 1.2 points per wavelength, below Nyquist, so the LiDAR returned
facet error instead of waves.
"""
import pytest

from wavesim import mesh_adequacy
from wavesim.config import WaveSimConfig
from wavesim.wave_generator import jonswap_components


def _cfg(nx, ny):
    cfg = WaveSimConfig()
    cfg.mesh.nx, cfg.mesh.ny = nx, ny
    return cfg


def _bf2_1comp():
    return jonswap_components(Hs=0.3, Tp=4.0, n_components=1, depth=50.0, seed=0)


def test_spacing_matches_gt_extent():
    dx, dy = mesh_adequacy.mesh_spacing(_cfg(40, 30))
    assert dx == pytest.approx(240.0 / 39, rel=1e-6)
    assert dy == pytest.approx(200.0 / 29, rel=1e-6)


def test_default_mesh_is_sub_nyquist_for_bf2_1comp():
    info = mesh_adequacy.check(_bf2_1comp(), _cfg(40, 30))
    assert info['lambda_min'] == pytest.approx(8.2, abs=0.2)
    assert info['pts_per_lambda'] < mesh_adequacy.PTS_PER_LAMBDA_NYQUIST
    assert not info['mesh_adequate']
    assert not info['mesh_above_nyquist']


def test_strict_mode_refuses_to_render_what_it_cannot_represent():
    info = mesh_adequacy.check(_bf2_1comp(), _cfg(40, 30))
    with pytest.raises(ValueError, match='NYQUIST'):
        mesh_adequacy.enforce(info, strict=True, verbose=False)


def test_fine_mesh_resolves_bf2_1comp():
    info = mesh_adequacy.check(_bf2_1comp(), _cfg(180, 150))
    assert info['pts_per_lambda'] >= mesh_adequacy.PTS_PER_LAMBDA_OK
    assert info['mesh_adequate']
    mesh_adequacy.enforce(info, strict=True, verbose=False)   # must not raise


def test_verdict_is_driven_by_unresolvable_variance_not_the_peak():
    """A mesh can render the peak beautifully and still drop a fifth of the sea.

    This is the BF4 benchmark case: 9.5 points per peak wavelength looks fine,
    but ~20% of the variance sits below the mesh's resolving power, and the
    measured operator error was 2.4x the nominal noise.  The verdict must see
    that, which is why it keys on variance rather than on the peak.
    """
    comps = jonswap_components(Hs=1.2, Tp=7.0, n_components=15, depth=50.0,
                               seed=0)
    info = mesh_adequacy.check(comps, _cfg(40, 30))
    assert info['pts_per_lambda_peak'] > mesh_adequacy.PTS_PER_LAMBDA_OK
    assert info['var_under_resolved'] > mesh_adequacy.VAR_UNDER_OK
    assert info['mesh_verdict'] == mesh_adequacy.DEGRADED
    assert not info['mesh_adequate']


def test_hos_order_accounts_for_bound_harmonics():
    comps = jonswap_components(Hs=1.2, Tp=8.0, n_components=60, depth=50.0,
                               seed=0)
    linear = mesh_adequacy.check(comps, _cfg(40, 30))
    third = mesh_adequacy.check(comps, _cfg(40, 30), hos_order=3)
    assert third['lambda_min'] == pytest.approx(linear['lambda_min'] / 3)
