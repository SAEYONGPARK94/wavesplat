"""d3_n5_core.py -- N5: the derived network core (three primitives).

Not a network someone liked the look of. Every piece is forced by a measured
result from this project, and the two failure modes that have killed learned
components here (support memorization; variance that collapses off the training
distribution) are made ARCHITECTURALLY impossible, not merely discouraged.

Provenance of each primitive (proofs in docs/N5_DERIVED_NETWORK_2026-07-10.md):

  P1  EP-neuron       the activation is the truncated-Gaussian moment match,
                      DERIVED. This is bug-fix #40 -- getting either sign wrong
                      makes v_hat > vc, every site clamps to tau=0, and the
                      constraints silently do nothing. The reference formulas
                      live in scripts/d3_e5_shadow_constraints.py::ep_solve; they
                      are copied here verbatim, not re-derived.

  P2  information     every learned correction delta_i to bank coefficient i is
      gate            emitted as c_i * delta_i, where c_i is the S1 posterior
                      contraction computed from the CURRENT Phi, gam, sigma^2 --
                      a pure computation with zero parameters. A mode the data
                      cannot constrain (c_i ~ 0) cannot be moved by the network.
                      This is the T1 impossibility theorem.

  P3  wave-group      the state is the complex amplitude a_k on the normalized
      equivariance    dispersion shell (k/k_p, psi) -- the ONLY representation
                      that ever transferred to real data (G3 lesson). A time
                      shift acts as a per-mode phase rotation; a module that
                      respects it must be complex-linear (diagonal in the mode
                      basis) with magnitude-only nonlinearities.

NO NOVELTY CLAIM. Deep unfolding, KalmanNet, learned EP messages (Heess,
Tarlow, Winn 2013), and steerable / complex-valued networks are all prior art.
A separate litcheck task is running; until it reports, this file makes no
novelty claim. The defence here is that the composition is FORCED by the
measurements, and is verifiable to machine precision (see d3_n5_tests.py).

No training happens in this module. fp64 throughout for the reductions.
"""
import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

SQ2PI = math.sqrt(2 * math.pi)


def _norm_pdf(x):
    return torch.exp(-0.5 * x * x) / SQ2PI


def _norm_cdf(x):
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


# ============================================================ P1: EP-neuron
def ep_neuron(mc, vc, h, sgn=1.0, slack=None):
    """Truncated-Gaussian moment-match unit -- the DERIVED activation (#40).

    Cavity marginal N(mc, vc) on a scalar linear functional x = phi^T a, meeting
    the one-sided event  sgn * (h - x) > 0  (sgn=+1: x < h ; sgn=-1: x > h). The
    site (a Gaussian message on that row) that makes the augmented linear solve
    reproduce the truncated posterior's first two moments has

        alpha = sgn * (h - mc) / st,   st = sqrt(vc + slack^2),  lam = phi/Phi(alpha)
        E[x | event]   = mc - sgn * (vc / st) * lam
        Var[x | event] = vc - (vc^2 / (vc + slack^2)) * lam * (lam + alpha)
        tau = 1/Var - 1/vc     (site precision, clamped >= 0)
        nu  = E/Var - mc/vc    (site precision * mean)

    slack = None (hard truncation, st = sqrt(vc)) recovers the classic form
        Var[x | event] = vc * (1 - lam^2 - alpha*lam).
    slack = sh softens the height h into h + N(0, sh^2), a probit site -- the
    constraint height built from a NOISY range is itself uncertain.

    IDENTITY PROPERTY. When the event is already satisfied with room to spare
    (alpha -> +inf), lam -> 0, so E -> mc, Var -> vc, tau -> 0, nu -> 0: the unit
    is the identity and adds nothing to the solve. This is why an unused /
    ignorable (MCAR) site changes the answer by exactly zero.

    Getting either sign wrong makes Var > vc, hence tau < 0; the .clamp_min(0.0)
    then zeros the site and the constraint silently vanishes (the bug #40 warns
    about). The clamp is a guard, not the fix -- the signs above are the fix.

    Args, all broadcastable torch tensors (or scalars) in fp64:
        mc, vc : cavity mean and variance of x = phi^T a
        h      : constraint height
        sgn    : +1 for x < h, -1 for x > h
        slack  : None or per-site sd of the height (probit softening)
    Returns:
        tau, nu : site precision and precision-weighted mean (same shape as mc)
    """
    vc = vc.clamp_min(1e-12) if torch.is_tensor(vc) else vc
    sh2 = torch.zeros_like(mc) if slack is None else slack ** 2
    st = torch.sqrt(vc + sh2)
    al = sgn * (h - mc) / st
    Z = _norm_cdf(al).clamp_min(1e-12)
    lam = _norm_pdf(al) / Z
    m_hat = mc - sgn * (vc / st) * lam
    v_hat = (vc - (vc * vc / (vc + sh2)) * lam * (lam + al)).clamp_min(1e-10)
    tau = ((1.0 / v_hat) - (1.0 / vc)).clamp_min(0.0)
    nu = m_hat / v_hat - mc / vc
    bad = ~torch.isfinite(tau) | ~torch.isfinite(nu)
    tau = torch.where(bad, torch.zeros_like(tau), tau)
    nu = torch.where(bad, torch.zeros_like(nu), nu)
    return tau, nu


# =================================================== P2: information gate
def contraction_gate(Phi, gam, s2):
    """S1 posterior contraction c_i per bank coefficient -- zero parameters.

        c_i = 1 - Var_post(a_i) / gamma_i = gamma_i * phi_i^T K^-1 phi_i,
        K = Phi diag(gam) Phi^T + s2 I         (the n x n "kernel" form)

    c_i in [0,1]: the fraction of prior variance the data remove from mode i.
    c_i = 0  <=> the observations say nothing about that mode (an added bank atom
    is a free parameter with no information); c_i -> 1 <=> the mode is pinned.

    KERNEL vs PRECISION form. The identity above (Woodbury) also reads
        c_i = 1 - [ (Phi^T Phi / s2 + diag(1/gam))^-1 ]_ii / gamma_i.
    We use the KERNEL form because it needs one n x n solve (n = #observations,
    ~600) against the precision form's 2B x 2B solve (2B ~ 3800), and because it
    never forms 1/gam, which blows up for the near-zero prior weights the
    beamform prior produces. clamp(0,1) removes fp round-off past the endpoints.

    Args:
        Phi : (n, 2B) design [cos | sin], SAME atom order as gam
        gam : (2B,) prior variance per coefficient
        s2  : scalar observation noise variance
    Returns:
        c : (2B,) contraction per coefficient, in [0,1]
    """
    n = Phi.shape[0]
    K = (Phi * gam[None]) @ Phi.T + s2 * torch.eye(
        n, device=Phi.device, dtype=Phi.dtype)
    M = torch.linalg.solve(K, Phi)                 # K^-1 Phi   (n, 2B)
    c = gam * (Phi * M).sum(0)                      # (2B,)
    return c.clamp(0.0, 1.0)


def variance_floor(c, gam):
    """Per-mode posterior-variance floor (1 - c_i) * gamma_i (theorem T2).

    For the linear-Gaussian model Var_post(a_i) = (1 - c_i) * gamma_i exactly.
    A learned variance head must report var_out_i = max(net_var_i, floor_i); it
    then can never claim to know a mode better than the data allow, and on an
    uninformed mode (c_i ~ 0) the floor is the FULL prior gamma_i -- overconfident
    collapse is impossible. This bounds the per-mode COEFFICIENT variance only;
    end-to-end field z-std calibration needs the full pipeline (see the doc).
    """
    return (1.0 - c).clamp_min(0.0) * gam


def physics_posterior(Phi, z, gam, s2, sites=None):
    """The physics posterior-mean coefficients a (primal / normal-equations).

        A = Phi^T Phi / s2 + diag(1/gam) + sum_k tau_k f_k f_k^T
        b = Phi^T z / s2 + sum_k nu_k f_k
        a = A^-1 b

    With sites=None this is exactly the linear-Gaussian MMSE coefficient vector,
    equal (Woodbury) to the kernel form gam * (Phi^T K^-1 z) that d3_gpu_mmse
    uses. With EP sites (Fc, tau, nu) it is the augmented solve of ep_solve --
    every tau_k = 0 leaves it identical to the naive solve (the R1 reduction).

    Args:
        Phi : (n, 2B) design
        z   : (n,) observations
        gam : (2B,) prior variance
        s2  : scalar noise variance
        sites : optional (Fc, tau, nu) with Fc (K, 2B), tau (K,), nu (K,)
    Returns:
        a : (2B,) posterior-mean coefficients
    """
    A = Phi.T @ Phi / s2 + torch.diag(1.0 / gam)
    b = Phi.T @ z / s2
    if sites is not None:
        Fc, tau, nu = sites
        A = A + (Fc * tau[:, None]).T @ Fc
        b = b + Fc.T @ nu
    return torch.linalg.solve(A, b)


class GatedDelta(nn.Module):
    """out = a_physics + c (*) delta,  delta = scale * tanh(module(features)).

    An arbitrary learned module proposes a per-coefficient correction delta; the
    tanh bounds it to [-scale, scale] (Delta_max = scale) and the contraction
    gate c multiplies it elementwise. The learned correction to mode i is thus
    capped at c_i * scale in magnitude: on a mode the data cannot see (c_i ~ 0)
    the network is powerless. This is the whole point (theorem T1).

    module: any nn.Module mapping features -> tensor shaped like a_physics.
    """

    def __init__(self, module, scale=1.0):
        super().__init__()
        self.module = module
        self.scale = float(scale)

    def delta(self, features):
        return self.scale * torch.tanh(self.module(features))

    def forward(self, a_physics, c, features):
        return a_physics + c * self.delta(features)


# =================================================== P3: wave-group equivariance
def phase_rotate(a, om, dt):
    """Advance complex amplitudes by dt: a_k -> a_k * exp(-i om_k dt).

    The exact physics time-evolution of a free wave component, and a group
    action: phase_rotate(phase_rotate(a, om, s), om, t) == phase_rotate(a, om, s+t).

    a  : (..., B, 2) real, channel 0 = Re, channel 1 = Im
    om : (B,) angular frequencies on the shell
    dt : scalar time shift
    """
    ph = -om * dt
    cph, sph = torch.cos(ph), torch.sin(ph)
    re, im = a[..., 0], a[..., 1]
    return torch.stack([re * cph - im * sph, re * sph + im * cph], -1)


class PhaseEquivariantBlock(nn.Module):
    """Minimal complex-linear (diagonal) layer + magnitude nonlinearity.

    Acts per mode:  a_k -> g(|a_k|) * w_k * a_k , with w_k a learned complex
    weight and g a learned magnitude-only gain (an MLP of |a_k|). It COMMUTES
    exactly with per-mode phase rotation, because

        * |a_k| is invariant to phase, so g is unchanged by phase_rotate, and
        * multiplication by the diagonal complex weight w_k commutes with the
          diagonal phase rotation exp(-i om_k dt).

    Mode MIXING (a full complex matrix) would commute only with a GLOBAL phase,
    not the per-mode rotation the dispersion relation dictates; the symmetry
    therefore forbids it, which is why the block is diagonal. A complex bias
    would break equivariance (a constant does not rotate) and is omitted.

    Represents complex numbers as a real 2-channel tensor (..., B, 2).
    """

    def __init__(self, B, hidden=16, eps=1e-12, dtype=torch.float64):
        super().__init__()
        self.w_re = nn.Parameter(torch.randn(B, dtype=dtype))
        self.w_im = nn.Parameter(torch.randn(B, dtype=dtype))
        self.mag = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, 1)).to(dtype)
        self.eps = eps

    def forward(self, a):
        re, im = a[..., 0], a[..., 1]
        r = torch.sqrt(re * re + im * im + self.eps)          # phase-invariant
        g = 1.0 + self.mag(r.unsqueeze(-1)).squeeze(-1)       # magnitude-only
        out_re = g * (self.w_re * re - self.w_im * im)
        out_im = g * (self.w_re * im + self.w_im * re)
        return torch.stack([out_re, out_im], -1)


if __name__ == '__main__':
    print('d3_n5_core: import-only module. Run scripts/d3_n5_tests.py.')
