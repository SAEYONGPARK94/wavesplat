"""The fixed-threshold detection law had no tests at all.

Two things need locking down.  (1) The (mask, n_raw) contract, including the
zero-hit path -- callers unpack two values, and returning one silently breaks
them.  (2) The cap must subsample uniformly: if it ever became a top-k by power
it would reintroduce the preferential sampling that H3-OD (2026-07-11) found to
be "death-manufacturing", and it would do so invisibly.
"""
import numpy as np
import pytest

from wavesim.raydrop_threshold import T_MANUAL_50, apply_dropout_fixed_T


def _scene(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    hit_mask = np.ones(n, dtype=bool)
    dist = rng.uniform(10.0, 250.0, n)
    inc = np.arctan(dist / 5.0)          # 5 m sensor height -> grazing at range
    return hit_mask, dist, inc


def test_returns_mask_and_raw_count():
    hit_mask, dist, inc = _scene()
    mask, n_raw = apply_dropout_fixed_T(hit_mask, dist, inc,
                                        np.random.default_rng(0))
    assert mask.dtype == bool and len(mask) == len(hit_mask)
    assert n_raw == int(mask.sum())      # uncapped: raw count == returned count


def test_zero_hits_still_honours_the_two_value_contract():
    empty = np.zeros(0, dtype=bool)
    mask, n_raw = apply_dropout_fixed_T(empty, np.zeros(0), np.zeros(0),
                                        np.random.default_rng(0))
    assert len(mask) == 0
    assert n_raw == 0


def test_cap_truncates_but_raw_count_reports_the_full_yield():
    hit_mask, dist, inc = _scene()
    _, n_uncapped = apply_dropout_fixed_T(hit_mask, dist, inc,
                                          np.random.default_rng(0))
    mask, n_raw = apply_dropout_fixed_T(hit_mask, dist, inc,
                                        np.random.default_rng(0), cap=100)
    assert mask.sum() == 100             # storage policy
    assert n_raw == n_uncapped           # the yield, undamaged by the cap
    assert n_raw > 100                   # the cap really is binding here


def test_cap_samples_uniformly_not_by_power():
    """A top-k cap would keep only the near returns; a uniform one must not."""
    hit_mask, dist, inc = _scene(n=20000)
    _, n_raw = apply_dropout_fixed_T(hit_mask, dist, inc,
                                     np.random.default_rng(0))
    mask, _ = apply_dropout_fixed_T(hit_mask, dist, inc,
                                    np.random.default_rng(0), cap=500)
    kept = dist[mask]
    full = dist[apply_dropout_fixed_T(hit_mask, dist, inc,
                                      np.random.default_rng(0))[0]]
    # the kept range distribution must track the full one, not collapse to near
    assert kept.mean() == pytest.approx(full.mean(), rel=0.15)
    assert kept.max() > 0.8 * full.max()


def test_threshold_is_the_datasheet_relative_calibration():
    # 10% reflectivity detectable at 250 m, 50% detection under exponential
    # speckle, normalised at d_ref = 125 m.
    assert T_MANUAL_50 == pytest.approx(0.10 / 4.0 * np.log(2.0))
