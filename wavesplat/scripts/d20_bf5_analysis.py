"""
Why is BF5-30c marginal for the deployable estimator while BF4-30c is fine?
Oracle is good for both (~0.03) ⇒ the information is there; the gap is component
ESTIMATION. Compare the cases on the axes that make estimation hard.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
from d6_integrated_eval import tag_to_file

G = 9.81


def main():
    tags = ['BF4_5c_s0', 'BF4_15c_s0', 'BF4_30c_s0', 'BF5_30c_s1']
    print(f"{'tag':12s} {'Hs':>5} {'Tp':>5} {'Ncmp':>5} {'PRener':>6} {'|k|rng':>12} "
          f"{'cg(p)':>6} {'cgT/λp':>7} {'pts/fr':>7} {'maxStp':>6}")
    for tag in tags:
        L = EpisodeDataLoader('data/' + tag_to_file(tag))
        wc = np.array(L.wave_components)  # A,omega,kx,ky,phi,Q
        A = wc[:, 0]; om = wc[:, 1]; kx = wc[:, 2]; ky = wc[:, 3]
        kmag = np.sqrt(kx**2 + ky**2)
        E = A**2
        pr_energy = (E.sum()**2) / (E**2).sum()          # eff # energy-carrying comps
        # peak component (max energy)
        ip = int(np.argmax(E)); kp = kmag[ip]
        cgp = G * (np.tanh(kp*L.depth) + kp*L.depth*(1-np.tanh(kp*L.depth)**2)) / (2*np.sqrt(G*kp*np.tanh(kp*L.depth)))
        lam_p = 2*np.pi/kp
        cgT_over_lam = cgp * 15.0 / lam_p                  # synthetic aperture in wavelengths over 15s
        # lidar points/frame
        npf = []
        for i in range(150, 260, 20):
            npf.append(len(L[i].x_obs))
        max_steep = float(np.max(kmag * A))
        print(f"{tag:12s} {L.Hs_true:5.2f} {L.Tp_true:5.1f} {len(A):5d} {pr_energy:6.1f} "
              f"[{kmag.min():.3f},{kmag.max():.2f}] {cgp:6.2f} {cgT_over_lam:7.2f} "
              f"{np.mean(npf):7.0f} {max_steep:6.3f}")
    print("\nPRener=energy participation ratio (higher=energy spread over more comps=harder)")
    print("cgT/λp=synthetic aperture (in peak wavelengths) over 15s (lower=less temporal leverage)")


if __name__ == '__main__':
    main()
