"""Tests for wavesplat.evaluation.metrics."""
import numpy as np
import pytest  # noqa: F401




# ---------------------------------------------------------------------------
# rmse_observed (2026-08-07).  Added so the tracker's damage can be split into
# "where the sensor saw" and "where it did not" -- Rx-6 measured that turning the
# tracker off improves the unobserved field 2.12x more than it improves nRMSE,
# and only the observed half distinguishes overfitting from a misspecified
# objective.  These tests lock in that it is an exact complement.
# ---------------------------------------------------------------------------

def test_rmse_observed_is_exact_complement_of_unobserved():
    from wavesplat.evaluation.metrics import rmse_observed, rmse_unobserved, rmse
    rng = np.random.default_rng(0)
    gx = np.linspace(0, 250, 80)
    gy = np.linspace(-100, 100, 60)
    gt = rng.normal(0, 1, (60, 80))
    pred = gt + rng.normal(0, 0.3, (60, 80))
    ox = rng.uniform(20, 200, 71)
    oy = rng.uniform(-60, 60, 71)

    ro = rmse_observed(pred, gt, gx, gy, ox, oy)
    ru = rmse_unobserved(pred, gt, gx, gy, ox, oy)

    Xg, Yg = np.meshgrid(gx, gy)
    md = np.full(Xg.shape, np.inf)
    for j in range(len(ox)):
        md = np.minimum(md, np.sqrt((Xg - ox[j]) ** 2 + (Yg - oy[j]) ** 2))
    mo, mu = md < 10.0, md >= 10.0

    assert np.all(mo ^ mu), "masks must partition the grid"
    assert np.isclose(ro, rmse(pred[mo], gt[mo]))
    assert np.isclose(ru, rmse(pred[mu], gt[mu]))
    # variance-weighted recombination must recover the whole-grid RMSE
    tot = np.sqrt((mo.sum() * ro ** 2 + mu.sum() * ru ** 2) / md.size)
    assert np.isclose(tot, rmse(pred, gt))


def test_rmse_observed_no_observations_is_nan():
    from wavesplat.evaluation.metrics import rmse_observed
    gx = np.linspace(0, 10, 5)
    gy = np.linspace(0, 10, 5)
    z = np.zeros((5, 5))
    assert np.isnan(rmse_observed(z, z, gx, gy, np.array([]), np.array([])))
