"""
hos_simulator.py
Higher-Order Spectral (HOS) method for nonlinear ocean wave simulation.

References:
  - Dommermuth & Yue (1987), J. Fluid Mech. 184, pp. 267-288
  - West, Brueckner, Janda, Milder & Milton (1987), J. Geophys. Res. 92(C11)
  - Ducrozet, Bonnefoy, Le Touze & Ferrant (2016), Comp. Phys. Comm. 203

Physical basis:
  - Potential flow: irrotational, incompressible, inviscid
  - Free-surface boundary conditions (KFSBC + DFSBC)
  - Perturbation expansion of velocity potential to order M
  - FFT-based pseudo-spectral evaluation of nonlinear products
  - Full dispersion: omega = sqrt(g * k * tanh(k * d))
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List


G = 9.81  # gravitational acceleration [m/s^2]


@dataclass
class HOSConfig:
    """Configuration for HOS simulator."""
    Lx: float = 500.0       # domain length x [m]
    Ly: float = 400.0       # domain length y [m]
    Nx: int = 256            # spectral modes x
    Ny: int = 256            # spectral modes y
    depth: float = 50.0      # water depth [m]
    M: int = 3               # nonlinear order (1=linear, 2=2nd order, 3=3rd order)
    g: float = G


class HOSSimulator:
    """
    Pure Python HOS wave simulator.

    Solves free-surface potential flow using perturbation expansion
    up to order M. State variables are surface elevation eta and
    surface potential phi_s, stored in Fourier space.
    """

    def __init__(self, config: HOSConfig):
        self.cfg = config
        self.Lx = config.Lx
        self.Ly = config.Ly
        self.Nx = config.Nx
        self.Ny = config.Ny
        self.depth = config.depth
        self.M = config.M
        self.g = config.g
        self.t = 0.0

        # Dealiasing factor: physical grid = ceil((M+1)/2) * spectral grid
        self.dealiasing_factor = (self.M + 1 + 1) // 2  # ceil((M+1)/2)
        self.Nx_phys = self.Nx * self.dealiasing_factor
        self.Ny_phys = self.Ny * self.dealiasing_factor

        # Wavenumber grids (spectral)
        self.dkx = 2 * np.pi / self.Lx
        self.dky = 2 * np.pi / self.Ly
        kx_1d = np.fft.fftfreq(self.Nx, d=self.Lx / (2 * np.pi * self.Nx))
        ky_1d = np.fft.fftfreq(self.Ny, d=self.Ly / (2 * np.pi * self.Ny))
        self.KX, self.KY = np.meshgrid(kx_1d, ky_1d, indexing='ij')
        self.K = np.sqrt(self.KX**2 + self.KY**2)
        self.K[0, 0] = 1e-12  # avoid division by zero

        # Precompute L_n operators: vertical derivative multipliers at z=0
        # L_n(k) = |k|^n for even n, |k|^n * tanh(|k|*d) for odd n
        kd = self.K * self.depth
        self.tanh_kd = np.tanh(np.clip(kd, 0, 20))  # clip for numerical stability
        self._precompute_Ln()

        # Dispersion relation: omega = sqrt(g * |k| * tanh(|k| * d))
        self.omega = np.sqrt(self.g * self.K * self.tanh_kd)
        self.omega[0, 0] = 0.0

        # State variables in Fourier space (Nx, Ny) complex arrays
        self.eta_hat = np.zeros((self.Nx, self.Ny), dtype=complex)
        self.phi_s_hat = np.zeros((self.Nx, self.Ny), dtype=complex)

        # Physical space grids
        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny
        x = np.arange(self.Nx) * self.dx
        y = np.arange(self.Ny) * self.dy
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')

    def _precompute_Ln(self):
        """Precompute L_n(k) operators for n = 1..M+1."""
        self.Ln = {}
        for n in range(1, self.M + 2):
            if n % 2 == 1:  # odd
                self.Ln[n] = self.K**n * self.tanh_kd
            else:  # even
                self.Ln[n] = self.K**n

    def _to_physical(self, f_hat: np.ndarray) -> np.ndarray:
        """FFT spectral → physical space with dealiasing (zero-padding)."""
        if self.dealiasing_factor == 1:
            return np.real(np.fft.ifft2(f_hat)) * (self.Nx * self.Ny)

        # Zero-pad to dealiased grid
        f_pad = np.zeros((self.Nx_phys, self.Ny_phys), dtype=complex)
        # Copy spectral coefficients
        nx2 = self.Nx // 2
        ny2 = self.Ny // 2
        f_pad[:nx2, :ny2] = f_hat[:nx2, :ny2]
        f_pad[:nx2, -ny2:] = f_hat[:nx2, -ny2:]
        f_pad[-nx2:, :ny2] = f_hat[-nx2:, :ny2]
        f_pad[-nx2:, -ny2:] = f_hat[-nx2:, -ny2:]

        return np.real(np.fft.ifft2(f_pad)) * (self.Nx_phys * self.Ny_phys)

    def _to_spectral(self, f_phys: np.ndarray) -> np.ndarray:
        """FFT physical → spectral space with dealiasing (truncation)."""
        if self.dealiasing_factor == 1:
            return np.fft.fft2(f_phys) / (self.Nx * self.Ny)

        f_hat_full = np.fft.fft2(f_phys) / (self.Nx_phys * self.Ny_phys)

        # Truncate to spectral grid
        nx2 = self.Nx // 2
        ny2 = self.Ny // 2
        f_hat = np.zeros((self.Nx, self.Ny), dtype=complex)
        f_hat[:nx2, :ny2] = f_hat_full[:nx2, :ny2]
        f_hat[:nx2, -ny2:] = f_hat_full[:nx2, -ny2:]
        f_hat[-nx2:, :ny2] = f_hat_full[-nx2:, :ny2]
        f_hat[-nx2:, -ny2:] = f_hat_full[-nx2:, -ny2:]

        return f_hat

    def _compute_W(self, eta_hat: np.ndarray,
                   phi_s_hat: np.ndarray) -> np.ndarray:
        """
        Compute vertical velocity W = dphi/dz at z=eta via perturbation expansion.

        Dommermuth & Yue (1987) formulation:
        1. Compute phi^(m) from surface potential phi_s and eta
        2. Compute W from Taylor expansion at z=eta
        """
        M = self.M

        # phi^(m) Fourier coefficients at z=0
        phi_m_hat = [None] * (M + 1)  # index 0 unused, 1..M

        # Order 1: phi^(1) = phi_s
        phi_m_hat[1] = phi_s_hat.copy()

        # Get eta in physical space (dealiased)
        eta_phys = self._to_physical(eta_hat)

        # Higher orders: phi^(m) = -sum_{j=1}^{m-1} FFT[eta^j/j! * IFFT[L_j * phi^(m-j)]]
        eta_powers = {1: eta_phys}  # cache powers of eta
        for m in range(2, M + 1):
            phi_m_hat[m] = np.zeros_like(phi_s_hat)
            for j in range(1, m):
                # Get eta^j in physical space
                if j not in eta_powers:
                    eta_powers[j] = eta_powers[j - 1] * eta_phys
                eta_j = eta_powers[j]
                factorial_j = math.factorial(j)

                # L_j * phi^(m-j) in physical space
                Lj_phi = self._to_physical(self.Ln[j] * phi_m_hat[m - j])

                # Product in physical space, then back to spectral
                product = eta_j * Lj_phi / factorial_j
                phi_m_hat[m] -= self._to_spectral(product)

        # Compute W via Taylor expansion: W = sum over orders
        # W^(m) = sum_{j=0}^{m-1} eta^j/j! * IFFT[L_{j+1} * phi^(m-j)]
        W_phys = np.zeros_like(eta_phys)
        for m in range(1, M + 1):
            for j in range(0, m):
                if j not in eta_powers and j > 0:
                    eta_powers[j] = eta_powers[j - 1] * eta_phys
                eta_j = eta_powers.get(j, np.ones_like(eta_phys))
                factorial_j = math.factorial(j)

                Ljp1_phi = self._to_physical(self.Ln[j + 1] * phi_m_hat[m - j])
                W_phys += eta_j * Ljp1_phi / factorial_j

        return W_phys

    def _rhs(self, eta_hat: np.ndarray,
             phi_s_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute right-hand sides of free-surface evolution equations.

        KFSBC: d_eta/dt = W*(1 + |grad_h eta|^2) - grad_h phi_s . grad_h eta
        DFSBC: d_phi_s/dt = -g*eta - 0.5*|grad_h phi_s|^2
                            + 0.5*W^2*(1 + |grad_h eta|^2)
        """
        # Compute W (vertical velocity at z=eta)
        W_phys = self._compute_W(eta_hat, phi_s_hat)

        # Horizontal gradients in physical space
        eta_x = self._to_physical(1j * self.KX * eta_hat)
        eta_y = self._to_physical(1j * self.KY * eta_hat)
        phi_x = self._to_physical(1j * self.KX * phi_s_hat)
        phi_y = self._to_physical(1j * self.KY * phi_s_hat)

        grad_eta_sq = eta_x**2 + eta_y**2
        grad_phi_sq = phi_x**2 + phi_y**2

        # KFSBC
        deta_dt_phys = (W_phys * (1.0 + grad_eta_sq)
                        - phi_x * eta_x - phi_y * eta_y)

        # DFSBC
        eta_phys = self._to_physical(eta_hat)
        dphi_dt_phys = (-self.g * eta_phys
                        - 0.5 * grad_phi_sq
                        + 0.5 * W_phys**2 * (1.0 + grad_eta_sq))

        # Back to spectral
        deta_hat = self._to_spectral(deta_dt_phys)
        dphi_hat = self._to_spectral(dphi_dt_phys)

        return deta_hat, dphi_hat

    def step(self, dt: float):
        """Advance one time step using RK4 integration."""
        eta0 = self.eta_hat.copy()
        phi0 = self.phi_s_hat.copy()

        # k1
        k1_eta, k1_phi = self._rhs(eta0, phi0)

        # k2
        k2_eta, k2_phi = self._rhs(
            eta0 + 0.5 * dt * k1_eta,
            phi0 + 0.5 * dt * k1_phi)

        # k3
        k3_eta, k3_phi = self._rhs(
            eta0 + 0.5 * dt * k2_eta,
            phi0 + 0.5 * dt * k2_phi)

        # k4
        k4_eta, k4_phi = self._rhs(
            eta0 + dt * k3_eta,
            phi0 + dt * k3_phi)

        # Update
        self.eta_hat = eta0 + (dt / 6.0) * (k1_eta + 2*k2_eta + 2*k3_eta + k4_eta)
        self.phi_s_hat = phi0 + (dt / 6.0) * (k1_phi + 2*k2_phi + 2*k3_phi + k4_phi)
        self.t += dt

    def get_eta(self) -> np.ndarray:
        """Return surface elevation in physical space on spectral grid."""
        return np.real(np.fft.ifft2(self.eta_hat)) * (self.Nx * self.Ny)

    def get_eta_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Evaluate surface elevation at arbitrary (x, y) points via spectral interpolation.

        Args:
            x, y: arrays of query positions [m] (any shape, will be flattened)

        Returns:
            eta: surface elevation at query points (same shape as x)
        """
        shape = x.shape
        x_flat = x.ravel()
        y_flat = y.ravel()

        # Direct Fourier evaluation (exact, no grid interpolation)
        eta = np.zeros(len(x_flat))
        for ix in range(self.Nx):
            for iy in range(self.Ny):
                a = self.eta_hat[ix, iy]
                if abs(a) < 1e-15:
                    continue
                kx = self.KX[ix, iy]
                ky = self.KY[ix, iy]
                eta += np.real(a * np.exp(1j * (kx * x_flat + ky * y_flat)))
        return eta.reshape(shape)

    def initialize_jonswap(self, Hs: float, Tp: float,
                           gamma: float = 3.3,
                           direction_deg: float = 0.0,
                           direction_spread_deg: float = 30.0,
                           seed: Optional[int] = None):
        """
        Initialize eta and phi_s from JONSWAP spectrum.

        Sets up a linear wave field (valid initial condition for HOS).
        The nonlinear evolution will naturally develop bound harmonics.

        Args:
            Hs: significant wave height [m]
            Tp: peak period [s]
            gamma: JONSWAP peak enhancement factor
            direction_deg: mean direction [degrees from +x]
            direction_spread_deg: directional spreading [degrees, std dev]
            seed: random seed
        """
        rng = np.random.default_rng(seed)
        omega_p = 2 * np.pi / Tp
        dir_rad = np.deg2rad(direction_deg)
        spread_rad = np.deg2rad(direction_spread_deg)

        # JONSWAP spectrum S(omega)
        def jonswap_1d(omega):
            sigma = np.where(omega <= omega_p, 0.07, 0.09)
            alpha_j = np.exp(-0.5 * ((omega / omega_p - 1) / sigma)**2)
            S = (0.3125 * Hs**2 * omega_p**4 / omega**5
                 * np.exp(-1.25 * (omega_p / omega)**4)
                 * gamma**alpha_j)
            return S

        # Convert to 2D directional spectrum S(kx, ky)
        # For each (kx, ky) mode, compute omega from dispersion, then S(omega)
        # with directional spreading D(theta)

        phases = rng.uniform(0, 2 * np.pi, (self.Nx, self.Ny))

        eta_hat = np.zeros((self.Nx, self.Ny), dtype=complex)
        phi_hat = np.zeros((self.Nx, self.Ny), dtype=complex)

        for ix in range(self.Nx):
            for iy in range(self.Ny):
                k_mag = self.K[ix, iy]
                if k_mag < 1e-10 or k_mag > 20.0:
                    continue

                kx = self.KX[ix, iy]
                ky = self.KY[ix, iy]
                omega = self.omega[ix, iy]

                if omega < 0.3 * omega_p or omega > 3.0 * omega_p:
                    continue

                # 1D spectrum value
                S_omega = jonswap_1d(omega)

                # Directional spreading: cos^2s distribution (simplified Gaussian)
                theta_k = np.arctan2(ky, kx)
                d_theta = theta_k - dir_rad
                # Wrap to [-pi, pi]
                d_theta = (d_theta + np.pi) % (2 * np.pi) - np.pi
                D_theta = np.exp(-0.5 * (d_theta / spread_rad)**2)
                D_theta /= (spread_rad * np.sqrt(2 * np.pi))

                # Jacobian: S(kx,ky) = S(omega) * D(theta) * |d_omega/dk| / k
                # d_omega/dk = g * tanh(kd) / (2*omega) + g*k*d*(1-tanh^2(kd)) / (2*omega)
                kd = k_mag * self.depth
                tanh_kd = np.tanh(min(kd, 20.0))
                domega_dk = (self.g * (tanh_kd + k_mag * self.depth * (1 - tanh_kd**2))
                             / (2 * omega))

                S_2d = S_omega * D_theta * domega_dk / max(k_mag, 1e-10)

                # Amplitude from spectral density:
                # variance = S_2d * dkx * dky
                # A = sqrt(2 * variance)
                dk2 = self.dkx * self.dky
                variance = S_2d * dk2
                if variance < 0:
                    continue
                A = np.sqrt(2 * variance)

                phi_phase = phases[ix, iy]

                # eta_hat(k) = A/2 * exp(i*phi)  (complex amplitude)
                # (factor 1/2 because eta = Re[sum eta_hat * exp(ikx)])
                eta_hat[ix, iy] = A * np.exp(1j * phi_phase)

                # phi_s from linear theory: phi_s = -i * (omega/k) * eta / tanh(kd)
                # In Fourier: phi_s_hat = -i * omega / (k * tanh(kd)) * eta_hat
                # But more precisely: W = omega * eta (linear), so
                # phi_s_hat = i * omega * eta_hat / (k * tanh(kd))
                # Wait: from linear theory:
                #   eta = A cos(kx - omega*t + phi)
                #   phi_s = (A * omega / k) * sin(kx - omega*t + phi) / tanh(kd)
                # In complex: phi_s_hat = -i * omega / L1 * eta_hat
                # where L1 = k * tanh(kd)
                L1 = k_mag * tanh_kd
                if L1 > 1e-10:
                    phi_hat[ix, iy] = -1j * omega / L1 * eta_hat[ix, iy]

        # Normalize to match target Hs
        # eta in physical space
        eta_phys = np.real(np.fft.ifft2(eta_hat)) * (self.Nx * self.Ny)
        Hs_actual = 4.0 * np.std(eta_phys)
        if Hs_actual > 1e-10:
            scale = Hs / Hs_actual
            eta_hat *= scale
            phi_hat *= scale

        self.eta_hat = eta_hat
        self.phi_s_hat = phi_hat
        self.t = 0.0

    def initialize_from_components(self, components: list):
        """
        Initialize from WaveComponent list (for validation against linear wavesim).

        Args:
            components: list of WaveComponent (from wave_generator.py)
        """
        self.eta_hat[:] = 0
        self.phi_s_hat[:] = 0

        for c in components:
            # Find nearest wavenumber bin
            ix = round(c.kx / self.dkx) % self.Nx
            iy = round(c.ky / self.dky) % self.Ny

            # Complex amplitude: eta = A * cos(kx - omega*t + phi)
            # At t=0: eta = A * cos(k.x + phi) = Re[A*exp(i*phi)*exp(ik.x)]
            a_complex = c.A * np.exp(1j * c.phi)
            self.eta_hat[ix, iy] += a_complex

            # Surface potential from linear theory
            k_mag = self.K[ix, iy]
            L1 = self.Ln[1][ix, iy]
            if L1 > 1e-10:
                omega = self.omega[ix, iy]
                self.phi_s_hat[ix, iy] += -1j * omega / L1 * a_complex

        self.t = 0.0

    def compute_energy(self) -> Tuple[float, float]:
        """
        Compute kinetic and potential energy.

        Returns:
            (E_kinetic, E_potential)
        """
        eta_phys = self.get_eta()
        E_pot = 0.5 * self.g * np.mean(eta_phys**2) * self.Lx * self.Ly

        # Kinetic energy: E_kin = 0.5 * integral phi_s * W dx dy (to leading order)
        phi_s_phys = np.real(np.fft.ifft2(self.phi_s_hat)) * (self.Nx * self.Ny)
        W_phys = self._compute_W(self.eta_hat, self.phi_s_hat)
        # Truncate W to spectral grid
        if W_phys.shape != phi_s_phys.shape:
            # W was computed on dealiased grid, need spectral grid value
            W_hat = self._to_spectral(W_phys)
            W_on_grid = np.real(np.fft.ifft2(W_hat)) * (self.Nx * self.Ny)
        else:
            W_on_grid = W_phys
        E_kin = 0.5 * np.mean(phi_s_phys * W_on_grid) * self.Lx * self.Ly

        return E_kin, E_pot

    def run_to(self, t_target: float, dt: float,
               store_interval: Optional[float] = None) -> dict:
        """
        Run simulation to t_target, optionally storing snapshots.

        Args:
            t_target: target time [s]
            dt: time step [s]
            store_interval: if set, store eta every this many seconds

        Returns:
            dict with 'times' and 'eta_fields' if store_interval is set
        """
        results = {'times': [], 'eta_fields': []}
        next_store = self.t if store_interval else float('inf')

        while self.t < t_target - 1e-10:
            # Store snapshot if needed
            if store_interval and self.t >= next_store - 1e-10:
                results['times'].append(self.t)
                results['eta_fields'].append(self.get_eta().copy())
                next_store += store_interval

            actual_dt = min(dt, t_target - self.t)
            self.step(actual_dt)

        # Final snapshot
        if store_interval:
            results['times'].append(self.t)
            results['eta_fields'].append(self.get_eta().copy())

        return results

    def get_surface_gradients(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute surface gradients deta/dx, deta/dy from spectral coefficients.

        Returns:
            (eta_x, eta_y): (Nx, Ny) arrays of surface gradients
        """
        eta_x = np.real(np.fft.ifft2(1j * self.KX * self.eta_hat)) * (self.Nx * self.Ny)
        eta_y = np.real(np.fft.ifft2(1j * self.KY * self.eta_hat)) * (self.Nx * self.Ny)
        return eta_x, eta_y

    def get_surface_normals(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute surface normals from spectral gradients.

        Normal = (-deta/dx, -deta/dy, 1) / |(-deta/dx, -deta/dy, 1)|

        Returns:
            (NX, NY, NZ): (Nx, Ny) arrays of unit normal components
        """
        eta_x, eta_y = self.get_surface_gradients()
        mag = np.sqrt(eta_x**2 + eta_y**2 + 1.0)
        return -eta_x / mag, -eta_y / mag, 1.0 / mag

    def evaluate_at(self, X: np.ndarray, Y: np.ndarray) -> Tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate surface elevation and normals at arbitrary (X, Y) grid points
        via spectral interpolation.

        Compatible with gt_generator.evaluate_field() interface.

        Args:
            X, Y: 2D arrays of query positions [m]

        Returns:
            (Z, NX, NY, NZ): surface elevation and unit normal components
        """
        shape = X.shape
        x_flat = X.ravel()
        y_flat = Y.ravel()
        n_pts = len(x_flat)

        # Gather nonzero spectral coefficients for efficiency
        mask = np.abs(self.eta_hat) > 1e-15
        ix_nz, iy_nz = np.where(mask)
        a_nz = self.eta_hat[mask]
        kx_nz = self.KX[mask]
        ky_nz = self.KY[mask]

        # Vectorized evaluation: sum_k a_k * exp(i*(kx*x + ky*y))
        # Shape: (n_modes,) x (n_pts,) via outer products
        eta = np.zeros(n_pts)
        eta_x = np.zeros(n_pts)
        eta_y = np.zeros(n_pts)

        # Process in batches to limit memory (n_modes × n_pts can be large)
        batch_size = 1000
        for start in range(0, len(a_nz), batch_size):
            end = min(start + batch_size, len(a_nz))
            a_batch = a_nz[start:end]
            kx_batch = kx_nz[start:end]
            ky_batch = ky_nz[start:end]

            # phase: (batch, n_pts)
            phase = np.outer(kx_batch, x_flat) + np.outer(ky_batch, y_flat)
            exp_phase = np.exp(1j * phase)

            eta += np.real(a_batch[:, None] * exp_phase).sum(axis=0)
            eta_x += np.real(1j * kx_batch[:, None] * a_batch[:, None] * exp_phase).sum(axis=0)
            eta_y += np.real(1j * ky_batch[:, None] * a_batch[:, None] * exp_phase).sum(axis=0)

        Z = eta.reshape(shape)
        eta_x = eta_x.reshape(shape)
        eta_y = eta_y.reshape(shape)

        # Normal = (-deta/dx, -deta/dy, 1) / |...|
        mag = np.sqrt(eta_x**2 + eta_y**2 + 1.0)
        NX = -eta_x / mag
        NY = -eta_y / mag
        NZ = 1.0 / mag

        return Z, NX, NY, NZ

    def get_Hs(self) -> float:
        """Compute significant wave height from current state."""
        eta = self.get_eta()
        return 4.0 * np.std(eta)
