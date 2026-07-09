"""Can a network that eats the beamform map output the wave field?

The beamform map our nets consume is a POWER map:

    pw(k) = (sum_i z_i cos psi_i)^2 + (sum_i z_i sin psi_i)^2 = |c(k)|^2

The squared modulus of the complex beamform coefficient c(k). Squaring throws
the phase away, and the field is nothing but phase: eta(x,t) = Re sum_k c(k)
exp(i psi). So a power->field network would have to invent the phase it was
never given.

This measures that directly. Backproject the same observations three ways and
correlate with the truth:

  complex   c(k)              -> the phase is there
  power     |c(k)| e^{i 0}    -> phase zeroed  (what a power map contains)
  power+rnd |c(k)| e^{i U}    -> phase randomized

If `complex` recovers the field and the two `power` variants do not, then no
architecture, no amount of data, and no loss function can turn our beamform
map into a field: the information was destroyed at the input.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..'))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'scripts')

import d3_static_field as SF

G = 9.81
NK = 41            # k-grid per axis


def main():
    sc = SF.scene_sim('linear', seed=0)
    P, z = sc['P'], sc['z']
    depth, kp = sc['depth'], sc['kp']
    cx, cy = sc['center']

    # dispersion-constrained k-grid, same band the estimator's bank uses
    klim = 2.5 * kp
    ax = np.linspace(-klim, klim, NK)
    KX, KY = np.meshgrid(ax, ax, indexing='xy')
    kx, ky = KX.ravel(), KY.ravel()
    kk = np.hypot(kx, ky)
    sel = (kk > 0.35 * kp) & (kk < klim)
    kx, ky, kk = kx[sel], ky[sel], kk[sel]
    om = np.sqrt(G * kk * np.tanh(kk * depth))

    # complex beamform coefficient of the observations
    psi = kx[:, None] * P[:, 0][None] + ky[:, None] * P[:, 1][None] \
        - om[:, None] * P[:, 2][None]
    c = (z[None] * np.exp(-1j * psi)).sum(1) / len(z)
    print(f"obs {len(z)}, bank {len(kx)} candidates")

    # query: the nowcast field on the domain grid
    Xq, Yq = np.meshgrid(sc['gx'], sc['gy'])
    t0 = sc['t0']
    eta = sc['make_eta']()
    gt = eta(Xq.ravel(), Yq.ravel(), t0).reshape(Xq.shape)

    psq = kx[:, None] * Xq.ravel()[None] + ky[:, None] * Yq.ravel()[None] \
        - om[:, None] * t0
    rng = np.random.default_rng(0)

    variants = {
        'complex  c(k)          ': c,
        'power    |c(k)|        ': np.abs(c).astype(complex),
        'power+rnd|c(k)|e^{iU}  ': np.abs(c) * np.exp(
            1j * rng.uniform(0, 2 * np.pi, len(c))),
    }
    inside, _ = SF.fan_mask(Xq, Yq, cx, cy, sc['hvec'])
    print(f"\n{'backprojection from':<24} {'corr(all)':>10} "
          f"{'corr(in-fan)':>13}")
    print('-' * 50)
    for name, ck in variants.items():
        f = 2.0 * np.real((ck[:, None] * np.exp(1j * psq)).sum(0))
        f = f.reshape(Xq.shape)
        r_all = np.corrcoef(f.ravel(), gt.ravel())[0, 1]
        r_in = np.corrcoef(f[inside], gt[inside])[0, 1]
        print(f"{name:<24} {r_all:>10.3f} {r_in:>13.3f}")

    print("\n(the estimator never uses the power map for the field: its MMSE "
          "reads the raw z. the net gets the power map, and so can only "
          "return a power — the prior variance gamma(k).)")


if __name__ == '__main__':
    sys.exit(main())
