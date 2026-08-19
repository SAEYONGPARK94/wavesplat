"""Regression test for the (a, b) -> phi sign convention in
`SequentialUpdater.amplitude_refit`  (W4 / E1, 2026-08-19).

WHY THIS EXISTS
---------------
The shipped forward model is  z = sum_i w_i * cos(theta_i + phi_i)
(`model/spectral_gaussian.py:6,124-127`), so expanding the cosine gives

    w cos(theta + phi) = (w cos phi) cos theta  -  (w sin phi) sin theta
                       =      a      cos theta  +       b      sin theta
    ==>  a = w cos phi,   b = -w sin phi   ==>   phi = atan2(-b, a)

Five independent least-squares sites in this package already use that
convention (`model/tnse.py:216`, `optimizer/initializer.py:448`,
`optimizer/relax_estimator.py:68`, `optimizer/cs_initializer.py:160`,
`optimizer/deep_relax.py:232`).  `optimizer/sequential_update.py:1335`
was the lone outlier, computing `np.arctan2(b, a)` -- a phase reflected
about 0, which makes the refitted model reconstruct a DIFFERENT field
than the one that was just fitted.

The bug is silent in the function's own report: `residual_after` is
computed from `A @ coeffs` (`sequential_update.py:1354`), i.e. straight
from the cos/sin coefficients, which are correct.  Only the (w, phi)
written back into the model at `:1356-1361` is wrong.  So the ONLY way
to catch this is a round-trip through (w, phi) -- which is what this
test does.

Repository discipline (CLAUDE.md, "테스트가 배포 기본값을 전혀 지키지
않는다"): this test calls the real shipped module, on the real shipped
class, through the real public method.  Nothing is re-implemented.

HOW IT WORKS
------------
`amplitude_refit` calls `np.arctan2` exactly once.  We wrap that call
with a spy so we can read the module's own float64 (w, phi) BEFORE they
are truncated to float32 by `torch.tensor(..., dtype=torch.float32)` at
`:1356-1361`.  That gives a genuine machine-zero (<= 1e-12) assertion.
A second assertion checks the float32 state actually stored in the model
(tolerance 1e-6, the float32 floor).  A third test flips the convention
back through the same spy hook and asserts the round-trip then breaks --
the negative control that proves the test has teeth.

RUN
    python3 -m pytest tmp/W4_SOLVER_2026-08-19/patches/test_phase_convention.py -v
(from /home/vexit/wavesplat/wavesplat).  When patch 1 is applied, move
this file to `wavesplat/tests/test_phase_convention.py` so it joins the
standing suite.
"""

import numpy as np
import pytest
import torch

from wavesplat.data_loader import FrameData
from wavesplat.model.dispersion import G
from wavesplat.model.spectral_gaussian import SpectralGaussianModel
from wavesplat.optimizer import sequential_update as su
from wavesplat.optimizer.sequential_update import SequentialUpdater

# ── Known ground truth.  Phases chosen in all four quadrants and away from
#    0 / +-pi/2 / +-pi, where atan2(b, a) and atan2(-b, a) coincide.
TRUE_W = np.array([0.80, 0.35, 0.55, 0.20])
TRUE_PHI = np.array([0.70, -2.40, 2.90, -0.55])
INIT_KX = np.array([0.060, -0.030, 0.110, 0.045])
INIT_KY = np.array([0.010, 0.070, -0.050, -0.090])
DEPTH = 50.0
FRAME_TIMES = (0.0, 0.5, 1.0, 1.5, 2.0)
N_OBS_PER_FRAME = 64


def _build_case():
    """Return (updater, model, frames, theta_stack, z_stack).

    The model is given the TRUE (kx, ky) but deliberately WRONG (w, phi):
    `amplitude_refit` freezes k and must recover (w, phi) exactly.

    Observations are generated with the MODEL's own float32 k / omega
    arrays, combined in exactly the order `amplitude_refit` uses
    (`sequential_update.py:1310-1313`), so the design matrix built inside
    the method is bit-identical to the theta used here.  Do NOT promote
    them to float64 first: `omega * t` is evaluated in float32 inside the
    method, and that alone leaves a ~5e-9 residual that would swamp the
    1e-12 tolerance -- the test would then be measuring numpy casting
    rather than the sign bug.
    """
    model = SpectralGaussianModel(n_gaussians=len(TRUE_W), depth=DEPTH)
    with torch.no_grad():
        model.kx.copy_(torch.as_tensor(INIT_KX, dtype=torch.float32))
        model.ky.copy_(torch.as_tensor(INIT_KY, dtype=torch.float32))
        # Wrong starting point on purpose -- the refit has to move.
        model.w.copy_(torch.full((len(TRUE_W),), 0.10))
        model.phi.copy_(torch.zeros(len(TRUE_W)))

    # float32, exactly as amplitude_refit reads them at :1292-1294.
    kx = model.kx.detach().cpu().numpy()
    ky = model.ky.detach().cpu().numpy()
    omega = model._omega().detach().cpu().numpy()

    # Sanity: the model's dispersion is the one we think it is.
    kmag = np.sqrt(kx.astype(np.float64) ** 2 + ky.astype(np.float64) ** 2)
    assert np.allclose(omega.astype(np.float64),
                       np.sqrt(G * kmag * np.tanh(kmag * DEPTH)), rtol=1e-5)

    rng = np.random.default_rng(20260819)
    frames, thetas, zs = [], [], []
    for t in FRAME_TIMES:
        x = rng.uniform(0.0, 150.0, N_OBS_PER_FRAME)
        y = rng.uniform(-60.0, 60.0, N_OBS_PER_FRAME)
        theta = (kx[None, :] * x[:, None]
                 + ky[None, :] * y[:, None]
                 - omega[None, :] * t)
        # Shipped forward model, noise-free: exact recovery must be possible.
        z = np.sum(TRUE_W[None, :] * np.cos(theta + TRUE_PHI[None, :]), axis=1)
        frames.append(FrameData(
            t=float(t),
            x_obs=torch.as_tensor(x, dtype=torch.float64),
            y_obs=torch.as_tensor(y, dtype=torch.float64),
            z_obs=torch.as_tensor(z, dtype=torch.float64),
            confidence=torch.ones(N_OBS_PER_FRAME, dtype=torch.float64),
            incidence_angles=torch.zeros(N_OBS_PER_FRAME, dtype=torch.float64),
        ))
        thetas.append(theta)
        zs.append(z)

    updater = SequentialUpdater({'depth': DEPTH})
    updater.model = model
    assert updater._bank is None, "bank route would short-circuit amplitude_refit"

    return updater, model, frames, np.vstack(thetas), np.concatenate(zs)


def _roundtrip_rms(theta, z, w, phi):
    """RMS error of reconstructing z from (w, phi) via the shipped forward model."""
    z_hat = np.sum(np.asarray(w)[None, :]
                   * np.cos(theta + np.asarray(phi)[None, :]), axis=1)
    return float(np.sqrt(np.mean((z_hat - z) ** 2)))


class _Arctan2Spy:
    """Wrap np.arctan2, recording args+result; optionally flip the sign of the
    first argument to emulate the *wrong* convention (negative control)."""

    def __init__(self, real, flip=False):
        self._real = real
        self._flip = flip
        self.calls = []

    def __call__(self, y, x, *args, **kwargs):
        y_eff = -np.asarray(y) if self._flip else y
        out = self._real(y_eff, x, *args, **kwargs)
        self.calls.append((np.asarray(y), np.asarray(x), np.asarray(out)))
        return out


def _refit_with_spy(flip=False):
    """Run the real `amplitude_refit`, returning what the module actually computed."""
    updater, model, frames, theta, z = _build_case()
    spy = _Arctan2Spy(np.arctan2, flip=flip)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(su.np, 'arctan2', spy)
        result = updater.amplitude_refit(frames, verbose=False)

    assert len(spy.calls) == 1, (
        f"expected exactly one np.arctan2 call inside amplitude_refit, "
        f"got {len(spy.calls)} -- the spy no longer isolates the phase step")
    y_arg, x_arg, phi_ls = spy.calls[0]
    # |b| is sign-agnostic here, so w is recoverable whichever convention is used.
    w_ls = np.sqrt(x_arg ** 2 + y_arg ** 2)
    return result, model, theta, z, w_ls, phi_ls


def test_refit_is_an_exact_least_squares_fit():
    """Precondition: the LS solve itself is exact (this passes with EITHER
    convention -- it is *not* the regression test, it only proves the setup
    is noise-free and well-conditioned, so any round-trip failure below is
    attributable to the sign alone)."""
    result, _, _, z, _, _ = _refit_with_spy()
    assert result, "amplitude_refit returned {} -- it never reached the LS solve"
    assert result['residual_after'] <= 1e-12, (
        f"design-matrix residual {result['residual_after']:.3e} is not machine "
        f"zero; the synthetic case is not exactly representable")
    assert float(np.sqrt(np.mean(z ** 2))) > 0.3, "signal is too weak to be a test"


def test_phase_convention_roundtrip_is_machine_zero():
    """★THE REGRESSION TEST.  Reconstructing the observations from the (w, phi)
    that amplitude_refit produced must return the observations.

    Fails on the pre-patch tree (`np.arctan2(b, a)`, sequential_update.py:1335)
    with a residual LARGER than the rms of the signal itself -- i.e. worse than
    predicting nothing.  `residual_after` does NOT catch this, because it is
    computed from `A @ coeffs`, before the (w, phi) conversion."""
    _, model, theta, z, w_ls, phi_ls = _refit_with_spy()

    rms_z = float(np.sqrt(np.mean(z ** 2)))

    # (a) float64, straight from the module's own arctan2 output.
    resid = _roundtrip_rms(theta, z, w_ls, phi_ls)
    assert resid <= 1e-12, (
        f"(w, phi) round-trip residual {resid:.3e} > 1e-12 "
        f"(signal rms {rms_z:.3f}). sequential_update.py:1335 must be "
        f"np.arctan2(-b, a): the forward model is w*cos(theta + phi), so "
        f"b = -w*sin(phi).  Same convention as model/tnse.py:216.")

    # (b) the float32 state actually written back into the model at :1356-1361.
    w_state = model.w.detach().cpu().numpy().astype(np.float64)
    phi_state = model.phi.detach().cpu().numpy().astype(np.float64)
    resid_state = _roundtrip_rms(theta, z, w_state, phi_state)
    assert resid_state <= 1e-6, (
        f"model state round-trip residual {resid_state:.3e} > 1e-6 "
        f"(float32 storage floor); the refit did not land in the model")

    # (c) and it recovered the truth, not just some self-consistent pair.
    phi_err = np.abs(np.angle(np.exp(1j * (phi_state - TRUE_PHI))))
    assert np.max(np.abs(w_state - TRUE_W)) <= 1e-5, "amplitudes not recovered"
    assert np.max(phi_err) <= 1e-5, "phases not recovered"


def test_wrong_convention_is_detected():
    """NEGATIVE CONTROL.  Force the module back to `atan2(b, a)` through the same
    hook and assert the round-trip breaks.  Without this, a test that trivially
    passes would look identical to a test that works."""
    _, _, theta, z, w_ls, phi_ls = _refit_with_spy(flip=True)

    rms_z = float(np.sqrt(np.mean(z ** 2)))
    resid_wrong = _roundtrip_rms(theta, z, w_ls, phi_ls)

    assert resid_wrong > rms_z, (
        f"the wrong convention produced residual {resid_wrong:.3f} <= signal rms "
        f"{rms_z:.3f}; this test case cannot distinguish the two conventions "
        f"and would not have caught the bug")
    assert resid_wrong > 1e6 * 1e-12, "negative control is not separated from the tolerance"
