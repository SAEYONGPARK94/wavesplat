"""
Non-stationary seas — the upper bound on the fusion window.

Real seas evolve; our estimator fits ONE constant set of components over the window.
Too-long a window then spans a changed sea. We synthesize controlled non-stationary
fields (per-component amplitudes modulated on a coherence time T_coh) from a real
episode's components + LiDAR geometry, and reconstruct at t* with growing windows.

Expectation: a sweet spot — nRMSE falls (information) then RISES (non-stationarity)
as T_obs grows, with the minimum near ~T_coh. Exponential recency weighting
(forgetting τ) should track the latest sea and stay robust. This complements Dir-I:
useful window is  T_wall  <  T_obs  <  T_coherence.

Run: python3 scripts/d22_nonstationary.py
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
from d6_integrated_eval import tag_to_file
from d15_sequential_fusion import gather_frames

G = 9.81
TAG = 'BF4_15c_s0'
I_STAR = 250


def env(t, T_coh, phases, depth_mod=0.7):
    """Per-component slow amplitude envelope (non-stationary if T_coh finite)."""
    if T_coh is None:
        return np.ones((len(phases), np.size(t)))
    return 1.0 + depth_mod * np.sin(2 * np.pi * np.atleast_1d(t)[None, :] / T_coh + phases[:, None])


def synth_z(P, comps, depth, T_coh, phases):
    A, om, kx, ky, phi = comps
    psi = kx[:, None]*P[None, :, 0] + ky[:, None]*P[None, :, 1] - om[:, None]*P[None, :, 2] + phi[:, None]
    e = env(P[:, 2], T_coh, phases)                       # (N, M)
    return ((A[:, None] * e) * np.cos(psi)).sum(0)


def synth_field(gx, gy, t, comps, depth, T_coh, phases):
    A, om, kx, ky, phi = comps
    Xg, Yg = np.meshgrid(gx, gy)
    e = env(np.array([t]), T_coh, phases)[:, 0]           # (N,)
    z = np.zeros_like(Xg)
    for i in range(len(A)):
        z += A[i]*e[i]*np.cos(kx[i]*Xg + ky[i]*Yg - om[i]*t + phi[i])
    return z


def wls_fit(kx, ky, P, z, depth, w):
    """Weighted LS of (cos,sin) coeffs; w: per-obs weights. Returns A,phi."""
    k = np.sqrt(kx**2+ky**2+1e-12); om = np.sqrt(G*k*np.tanh(k*depth)+1e-12)
    psi = kx[None]*P[:, 0:1] + ky[None]*P[:, 1:2] - om[None]*P[:, 2:3]
    D = np.concatenate([np.cos(psi), np.sin(psi)], 1); N = len(kx)
    W = w[:, None]
    Anorm = D.T @ (W*D) + 1e-2*np.eye(2*N); b = D.T @ (w*z)
    c = np.linalg.solve(Anorm, b)
    return np.sqrt(c[:N]**2+c[N:]**2), np.arctan2(-c[N:], c[:N])


def render(kx, ky, A, phi, gx, gy, t, depth):
    k = np.sqrt(kx**2+ky**2+1e-12); om = np.sqrt(G*k*np.tanh(k*depth)+1e-12)
    Xg, Yg = np.meshgrid(gx, gy); z = np.zeros_like(Xg)
    for i in range(len(kx)):
        z += A[i]*np.cos(kx[i]*Xg+ky[i]*Yg-om[i]*t+phi[i])
    return z


def main():
    L = EpisodeDataLoader('data/'+tag_to_file(TAG)); depth = L.depth; Hs = L.Hs_true
    wc = np.array(L.wave_components)
    comps = (wc[:, 0], wc[:, 1], wc[:, 2], wc[:, 3], wc[:, 4])
    gx = L.gt_grid_x.numpy(); gy = L.gt_grid_y.numpy()
    rng = np.random.default_rng(0); phases = rng.uniform(0, 2*np.pi, len(wc))
    t_star = float(L[I_STAR].t)
    T_obs_list = [2, 4, 6, 8, 10, 12, 15, 20]
    T_coh_list = [(None, 'stationary'), (20., 'T_coh=20s'), (10., 'T_coh=10s'), (6., 'T_coh=6s')]

    res = {}
    for T_coh, name in T_coh_list:
        gt = synth_field(gx, gy, t_star, comps, depth, T_coh, phases)
        uni, exp = [], []
        for T in T_obs_list:
            P, _ = gather_frames(L, I_STAR, int(round(T/0.1)))
            z = synth_z(P, comps, depth, T_coh, phases)
            # uniform weight
            A1, p1 = wls_fit(wc[:, 2], wc[:, 3], P, z, depth, np.ones(len(z)))
            f1 = render(wc[:, 2], wc[:, 3], A1, p1, gx, gy, t_star, depth)
            uni.append(float(np.sqrt(np.mean((f1-gt)**2))/Hs))
            # exponential recency weight (forgetting tau=4s)
            w = np.exp(-(t_star - P[:, 2]) / 4.0)
            A2, p2 = wls_fit(wc[:, 2], wc[:, 3], P, z, depth, w)
            f2 = render(wc[:, 2], wc[:, 3], A2, p2, gx, gy, t_star, depth)
            exp.append(float(np.sqrt(np.mean((f2-gt)**2))/Hs))
        res[name] = dict(T_obs=T_obs_list, uniform=uni, exp_weighted=exp)
        best_u = T_obs_list[int(np.argmin(uni))]
        print(f"{name:14s} uniform: " + " ".join(f"{v:.3f}" for v in uni) + f"  (min@T={best_u}s)")
        print(f"{'':14s} exp-wt : " + " ".join(f"{v:.3f}" for v in exp))
    json.dump(res, open('results/direction3/nonstationary.json', 'w'), indent=2)

    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    cols = ['k', 'tab:blue', 'tab:orange', 'tab:red']
    for (name, r), c in zip(res.items(), cols):
        a1.plot(r['T_obs'], r['uniform'], 'o-', color=c, label=name)
        a2.plot(r['T_obs'], r['exp_weighted'], 'o-', color=c, label=name)
    for ax, ti in [(a1, 'uniform window (equal weight)'), (a2, 'exponential forgetting (τ=4s)')]:
        ax.set_xlabel('fusion window $T_{obs}$ [s]'); ax.grid(alpha=0.25); ax.legend(fontsize=8); ax.set_title(ti)
    a1.set_ylabel('field nRMSE at $t^*$')
    fig.suptitle('Non-stationary seas: a fusion-window sweet spot (uniform) that recency weighting removes\n'
                 'useful window  T_wall < T_obs < T_coherence', fontsize=12)
    fig.tight_layout(); fig.savefig('results/direction3/fig_nonstationary.png', dpi=140)
    print('saved nonstationary.json + fig_nonstationary.png')


if __name__ == '__main__':
    main()
