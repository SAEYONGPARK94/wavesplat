#!/usr/bin/env python3
"""S16 baseline repair -- JONSWAP spectral background-error covariance for LETKFGrid.

PREREG.md (md5 in FROZEN_md5.txt) section 4 is the specification; this file implements it.

    *** THE REPOSITORY PACKAGE IS NOT MODIFIED. ***
    `LETKFGridSpec` is a subclass living inside this batch directory only.

WHAT IS BROKEN (PREREG section 0, four independent paths)
---------------------------------------------------------
`wavesplat/baselines/letkf_grid.py:88`

    self.ensemble = np.random.randn(n_ensemble, self.n_state) * 0.1

is the entire background-error covariance model:  spatially WHITE (correlation length
zero -> no wave physics) with a spread of 0.1 m, against a true field whose rms is
Hs/4 = 0.30 m (BF4) / 0.50 m (BF5).  Neither the spectrum nor the spread nor the
localisation radius has ever been an axis in any tuning sweep in this repository.

WHAT THIS CHANGES (exactly one concept)
---------------------------------------
The background-error covariance model, from "white, 0.1 m" to "isotropic JONSWAP,
Hs/4".  Everything else -- forecast operator, observation operator, the ETKF transform,
inflation MAGNITUDES, obs_noise, max_obs, n_ensemble, the Gaspari-Cohn function -- is
inherited unchanged from the stock class.

  1. modes           : the SAME rfft2 mode set the forecast operator uses
                       (letkf_grid.py:76-80), so background and forecast are consistent
  2. dispersion      : omega = sqrt(g|k| tanh(|k| d))     -- the filter's own relation
  3. JONSWAP         : S_J(w) = w^-5 exp(-1.25 (wp/w)^4) gamma^exp(-(w-wp)^2/(2 s^2 wp^2))
                       gamma = 3.3, s = 0.07 (w<=wp) / 0.09 (w>wp), wp = 2 pi / Tp
                       (the leading alpha g^2 cancels in the rescale, so it is set to 1)
  4. direction       : ISOTROPIC, D(theta) = 1/(2 pi).  No directional oracle is given.
                       Jacobian to wavenumber space:  S_k = S_J(w) D(theta) c_g / |k|,
                       c_g = dw/dk = (w/2k)(1 + 2kd/sinh(2kd))
                       mode variance = S_k * dkx * dky,  dkx = 2pi/(nx dx), dky = 2pi/(ny dy)
  5. realisation     : Hhat_m(k) = sigma_k exp(i phi),  phi ~ U(0, 2pi)
                       eta   = irfft2(Hhat)
                       etadot= irfft2(-i s omega Hhat)  <- dispersion-consistent cross-cov,
                       because eta = Re sum c e^{i(k.x - wt)}  =>  d eta/dt = Re sum (-i w) c e^{...}
                       *** s = +-1, a FAIR COIN per member and per mode (AMENDMENT_1, B1). ***
                       rfft2 keeps only the kx >= 0 half plane, so a fixed s = +1 makes
                       EVERY mode of EVERY member propagate towards +x -- which is exactly
                       the true field's direction in this corpus (direction_deg = 0) and
                       therefore an undeclared directional oracle inside the very eta/etadot
                       cross-covariance the Kalman gain uses.  MEASURED before the fix:
                       corr(etadot, d eta/dx) = -0.9092, corr(etadot, d eta/dy) = +0.0001.
                       The coin makes propagation genuinely isotropic; |corr| <= 0.15 on BOTH
                       axes is a pre-registered gate (ISOTROPY_TOL, checked by G-7).
  6. rescale         : multiply eta AND etadot by (Hs/4)/std(eta), so the background
                       spread is exactly the true field rms regardless of FFT normalisation
  7. centre          : subtract the ensemble mean -> background MEAN is the zero field
                       (the honest "no information" prior); spread loses only sqrt(1-1/Ne)
  8. inflation       : additive inflation is drawn from the SAME model (magnitude
                       unchanged at add_inflation), so the repair is not undone by
                       re-injecting white noise every cycle

RNG DISCIPLINE (PREREG section 2)
---------------------------------
The stock class draws the background, the additive inflation and the observation
thinning from the SAME global numpy stream, so changing the background also changes
which observations get thinned.  Here background + inflation use a DEDICATED
Generator(filter_seed); the global stream is saved/restored around the base __init__
and is therefore consumed by `_thin_observations` ONLY.  Consequence: `SPEC*` and
`WHITER50` see BIT-IDENTICAL thinning subsets, so `SPEC_best - WHITER50` is the net
effect of the repair with no confound.

DOCUMENTED IMPLEMENTATION DEVIATION (measured by PREREG E-3)
-----------------------------------------------------------
The stock class adds inflation to the perturbations INSIDE update(), after the
forecast.  Rewriting update() would mean copying 130 lines, so instead the base
`add_inflation` is set to 0 and the inflation is applied to the ensemble in this
class's `update()` just BEFORE delegating -- i.e. one forecast step earlier.  The
forecast is linear and spectrum-preserving, so this is immaterial in expectation; its
total size is nevertheless MEASURED by the `WHITER50` (this class, white background)
vs `WHITE50` (stock class) contrast, which is a pre-registered arm for exactly this.
Both `SPEC*` and `WHITER50` go through this identical path.
"""
import numpy as np

from wavesplat.baselines.letkf_grid import LETKFGrid

GRAVITY = 9.81
JONSWAP_GAMMA = 3.3
ISOTROPY_TOL = 0.15        # AMENDMENT_1 B1: |corr(etadot, d eta/dx or dy)| must be <= this


def jonswap_omega(omega, omega_p, gamma=JONSWAP_GAMMA):
    """Unnormalised JONSWAP density S(omega) (the leading alpha*g^2 is dropped)."""
    omega = np.asarray(omega, dtype=float)
    out = np.zeros_like(omega)
    ok = omega > 1e-9
    w = omega[ok]
    sig = np.where(w <= omega_p, 0.07, 0.09)
    r = np.exp(-((w - omega_p) ** 2) / (2.0 * (sig ** 2) * (omega_p ** 2)))
    out[ok] = w ** -5.0 * np.exp(-1.25 * (omega_p / w) ** 4) * gamma ** r
    return out


def _corr(a, b):
    """Pearson correlation over every element of two same-shaped arrays."""
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    a = a - a.mean()
    b = b - b.mean()
    den = a.std() * b.std()
    return float((a * b).mean() / den) if den > 0 else float('nan')


def group_velocity(k, omega, depth):
    """c_g = dw/dk for w^2 = g k tanh(k d)."""
    k = np.asarray(k, dtype=float)
    kd = np.clip(k * depth, 0.0, 30.0)          # sinh overflows past ~30
    with np.errstate(divide='ignore', invalid='ignore'):
        cg = np.where(k > 1e-9,
                      omega / (2.0 * np.maximum(k, 1e-12))
                      * (1.0 + 2.0 * kd / np.sinh(np.maximum(2.0 * kd, 1e-12))),
                      0.0)
    return np.nan_to_num(cg, nan=0.0, posinf=0.0, neginf=0.0)


class LETKFGridSpec(LETKFGrid):
    """LETKFGrid with a JONSWAP (or, as a control, white) background-error covariance.

    background : 'jonswap'  -> PREREG section 4 repair
                 'white'    -> stock model (randn * white_amp), same code path otherwise
    """

    def __init__(self, grid_x, grid_y, background='jonswap', Hs_bg=None, Tp_bg=None,
                 gamma=JONSWAP_GAMMA, white_amp=0.1, filter_seed=42, **kw):
        if background not in ('jonswap', 'white'):
            raise ValueError('background must be jonswap|white')
        self.background = background
        self.Hs_bg = None if Hs_bg is None else float(Hs_bg)
        self.Tp_bg = None if Tp_bg is None else float(Tp_bg)
        self.jonswap_gamma = float(gamma)
        self.white_amp = float(white_amp)
        self.filter_seed = int(filter_seed)
        self.direction_model = 'isotropic'
        self._rng = np.random.default_rng(self.filter_seed)

        # The base __init__ draws randn from the GLOBAL stream; save/restore it so the
        # global stream is consumed by observation thinning only.
        _state = np.random.get_state()
        super().__init__(grid_x, grid_y, **kw)
        np.random.set_state(_state)

        if background == 'jonswap' and (self.Hs_bg is None or self.Tp_bg is None):
            raise ValueError('jonswap background needs Hs_bg and Tp_bg')

        # additive inflation is applied by THIS class (see module docstring)
        self._add_inf = float(self.add_inflation)
        self.add_inflation = 0.0

        self._build_spectrum()
        self.ensemble = self._draw_background(self.n_ensemble, centre=True)

    # ---------------- background model ----------------
    def _build_spectrum(self):
        """sigma_k over the SAME rfft2 modes the forecast operator uses."""
        kx = 2 * np.pi * np.fft.fftfreq(self.nx, d=self.dx)[:self.nx // 2 + 1]
        ky = 2 * np.pi * np.fft.fftfreq(self.ny, d=self.dy)
        KX, KY = np.meshgrid(kx, ky)                      # (ny, nkx) == rfft2 layout
        K = np.sqrt(KX ** 2 + KY ** 2)
        self._K = K
        self._omega_bg = np.sqrt(GRAVITY * K * np.tanh(np.clip(K * self.depth, 0, 30)))
        if self.background != 'jonswap':
            self._sigma_k = None
            return
        omega_p = 2.0 * np.pi / self.Tp_bg
        S_w = jonswap_omega(self._omega_bg, omega_p, self.jonswap_gamma)
        cg = group_velocity(K, self._omega_bg, self.depth)
        D = 1.0 / (2.0 * np.pi)                            # isotropic
        with np.errstate(divide='ignore', invalid='ignore'):
            S_k = np.where(K > 1e-9, S_w * D * cg / np.maximum(K, 1e-12), 0.0)
        S_k = np.nan_to_num(S_k, nan=0.0, posinf=0.0, neginf=0.0)
        dkx = 2 * np.pi / (self.nx * self.dx)
        dky = 2 * np.pi / (self.ny * self.dy)
        self._sigma_k = np.sqrt(np.maximum(S_k * dkx * dky, 0.0))
        self._sigma_k[K <= 1e-9] = 0.0

    def _draw_background(self, n, centre=True, target_std=None):
        """(n, n_state) draw from the background model.

        jonswap: random-phase realisation rescaled so std(eta) == target_std
                 (default Hs_bg/4), with etadot = irfft2(-i s omega Hhat), s a fair
                 +-1 coin per member and mode, so the eta / etadot cross-covariance
                 obeys the dispersion relation AND propagation is isotropic
                 (AMENDMENT_1 B1; without the coin every mode propagates to +x).
        white  : randn * white_amp (stock model), same shape / centring convention.
        """
        if self.background == 'white':
            amp = self.white_amp if target_std is None else target_std
            X = self._rng.standard_normal((n, self.n_state)) * amp
        else:
            tgt = (self.Hs_bg / 4.0) if target_std is None else target_std
            phase = self._rng.uniform(0.0, 2.0 * np.pi,
                                      size=(n,) + self._sigma_k.shape)
            H = self._sigma_k[None, :, :] * np.exp(1j * phase)
            sgn = self._rng.choice(np.array([-1.0, 1.0]), size=phase.shape)
            eta = np.fft.irfft2(H, s=(self.ny, self.nx), axes=(-2, -1))
            deta = np.fft.irfft2(-1j * sgn * self._omega_bg[None, :, :] * H,
                                 s=(self.ny, self.nx), axes=(-2, -1))
            sd = eta.reshape(n, -1).std(axis=1)
            sc = np.where(sd > 1e-12, tgt / np.maximum(sd, 1e-12), 0.0)[:, None, None]
            eta = eta * sc
            deta = deta * sc
            X = np.concatenate([eta.reshape(n, -1), deta.reshape(n, -1)], axis=1)
        if centre and n > 1:
            X = X - X.mean(axis=0, keepdims=True)
        return X

    # ---------------- filter hooks ----------------
    def update(self, obs_x, obs_y, obs_z):
        """Additive inflation from the background model, then the stock update."""
        if self._add_inf > 0.0:
            self.ensemble = self.ensemble + self._draw_background(
                self.n_ensemble, centre=True, target_std=self._add_inf)
        return super().update(obs_x, obs_y, obs_z)

    def reset(self):
        self._rng = np.random.default_rng(self.filter_seed)
        self.ensemble = self._draw_background(self.n_ensemble, centre=True)

    # ---------------- introspection (PREREG G-7 / effective_config read-back) --------
    def background_diagnostics(self):
        """Proof, from the object itself, that the repair is in the ensemble."""
        E = self.ensemble
        eta = E[:, :self.n_grid].reshape(-1, self.ny, self.nx)
        deta = E[:, self.n_grid:].reshape(-1, self.ny, self.nx)
        m = eta.mean(axis=0)
        # nearest-neighbour spatial correlation of the member perturbations
        a = eta - eta.mean(axis=(1, 2), keepdims=True)
        num = float(np.mean(a[:, :, :-1] * a[:, :, 1:]))
        den = float(np.mean(a * a))
        lag1_x = num / den if den > 0 else float('nan')
        num_y = float(np.mean(a[:, :-1, :] * a[:, 1:, :]))
        lag1_y = num_y / den if den > 0 else float('nan')
        # dispersion consistency: std(etadot)/std(eta) should sit near the
        # energy-weighted rms omega of the prescribed spectrum
        if self._sigma_k is not None and self._sigma_k.sum() > 0:
            w2 = float(np.sum(self._sigma_k ** 2 * self._omega_bg ** 2)
                       / np.sum(self._sigma_k ** 2))
            w_rms_spec = float(np.sqrt(w2))
        else:
            w_rms_spec = float('nan')
        # AMENDMENT_1 B1 -- PROPAGATION ISOTROPY.  A background whose members all
        # travel the same way has corr(etadot, d eta/dx) -> -1 for +x propagation.
        # The four original G-7 diagnostics are invariant to the sign of omega and
        # cannot see this; these two are the gate.  Threshold: |corr| <= ISOTROPY_TOL
        # on BOTH axes (frozen in AMENDMENT_1 before any full run).
        gx = np.gradient(eta, self.dx, axis=2)
        gy = np.gradient(eta, self.dy, axis=1)
        cdx, cdy = _corr(deta, gx), _corr(deta, gy)
        return dict(background=self.background,
                    corr_detadot_dx=cdx, corr_detadot_dy=cdy,
                    isotropy_tol=ISOTROPY_TOL,
                    isotropy_ok=bool(abs(cdx) <= ISOTROPY_TOL
                                     and abs(cdy) <= ISOTROPY_TOL),
                    eta_std=float(eta.std()),
                    eta_target=(self.Hs_bg / 4.0 if self.Hs_bg else None),
                    ens_mean_abs=float(np.abs(m).max()),
                    lag1_corr_x=lag1_x, lag1_corr_y=lag1_y,
                    detadot_over_deta=float(deta.std() / max(eta.std(), 1e-12)),
                    omega_rms_spectrum=w_rms_spec,
                    n_modes_nonzero=(int((self._sigma_k > 0).sum())
                                     if self._sigma_k is not None else 0))


READBACK_KEYS = ('background', 'Hs_bg', 'Tp_bg', 'jonswap_gamma', 'white_amp',
                 'direction_model', 'filter_seed', 'loc_radius', 'n_ensemble',
                 'mult_inflation', 'add_inflation', '_add_inf', 'obs_noise', 'max_obs',
                 'depth', 'dt', 'max_eta', 'nx', 'ny', 'n_grid', 'n_state', 'dx', 'dy', 'g')


def readback(f):
    """effective_config read back FROM THE OBJECT (discipline 4), never from intent."""
    out = {'filter_class': type(f).__name__}
    for k in READBACK_KEYS:
        v = getattr(f, k, '<absent>')
        out[k] = (v if isinstance(v, (int, float, str, bool, type(None)))
                  else (float(v) if isinstance(v, np.floating) else str(v)))
    return out
