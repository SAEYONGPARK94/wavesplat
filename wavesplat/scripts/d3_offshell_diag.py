"""P4 (2026-07-08): off-shell energy diagnostic — mechanism test for the
K3 bad-window reproduction (GAP_RMSE_persona_debate_2026-07-08.md).

Hypothesis: cross-Tp crossing seas generate 2nd-order bound waves at
k1+-k2, w1+-w2 — energy OFF the free-wave dispersion shell that the
estimator hard-constrains. Prediction: off-shell fraction (in the
estimator's k-band) is small for unimodal / same-Tp seas and large for
cross-Tp seas, and correlates with per-sea skill collapse in
falcon_hos_probe_bimod90cross_n8.json.

Method: rebuild the exact probe seas (same seeds/config), record
eta(x,y,t) for T_REC seconds after spinup, Hann-windowed FFT in time +
periodic FFT in space -> E(kx,ky,w>0). Off-shell = |w - w_disp(|k|)| >
DELTA within the k-band [0.3, 3]*kp_lo..hi. Report fraction per sea.
"""
import json

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesim.hos_simulator import HOSSimulator, HOSConfig

G = 9.81
HS, TP, DEPTH = 2.0, 8.0, 50.0
TP2 = 5.6
SPINUP = 10.0
DT_HOS = 0.1
T_REC, DT_REC = 40.0, 0.5
OUT = 'results/direction3/offshell_diag.json'


def build_sea(mode, si, M=3):
    cfg = HOSConfig(Lx=500.0, Ly=400.0, Nx=128, Ny=128, depth=DEPTH, M=M)
    sim = HOSSimulator(cfg)
    if mode == 'unimodal':
        sim.initialize_jonswap(Hs=HS, Tp=TP, direction_deg=0.0,
                               direction_spread_deg=25.0, seed=9000 + si)
    else:
        tp2 = TP if mode == 'bimod90same' else TP2
        sim.initialize_jonswap(Hs=HS * np.sqrt(0.5), Tp=TP,
                               direction_deg=0.0,
                               direction_spread_deg=25.0, seed=9000 + si)
        ea, pa = sim.eta_hat.copy(), sim.phi_s_hat.copy()
        sim.initialize_jonswap(Hs=HS * np.sqrt(0.5), Tp=tp2,
                               direction_deg=90.0,
                               direction_spread_deg=25.0, seed=19000 + si)
        sim.eta_hat = ea + sim.eta_hat
        sim.phi_s_hat = pa + sim.phi_s_hat
        sim.t = 0.0
    return sim, cfg


def offshell_fraction(sim, cfg, deltas=(0.25, 0.35)):
    n_rec = int(round(T_REC / DT_REC)) + 1
    frames = np.empty((n_rec, cfg.Ny, cfg.Nx))
    t_next = SPINUP
    for j in range(n_rec):
        while sim.t < t_next - 1e-9:
            sim.step(DT_HOS)
        frames[j] = sim.get_eta()
        t_next += DT_REC
    if not np.all(np.isfinite(frames)):
        return None
    frames = frames - frames.mean()
    win = np.hanning(n_rec)[:, None, None]
    F = np.fft.fftn(frames * win, axes=(0, 1, 2))
    E = np.abs(F) ** 2
    om = 2 * np.pi * np.fft.fftfreq(n_rec, DT_REC)
    ky = 2 * np.pi * np.fft.fftfreq(cfg.Ny, cfg.Ly / cfg.Ny)
    kx = 2 * np.pi * np.fft.fftfreq(cfg.Nx, cfg.Lx / cfg.Nx)
    KY, KX = np.meshgrid(ky, kx, indexing='ij')
    K = np.hypot(KX, KY)
    om_d = np.sqrt(G * K * np.tanh(K * DEPTH))
    kp1 = (2 * np.pi / TP) ** 2 / G
    kp2 = (2 * np.pi / TP2) ** 2 / G
    kband = (K >= 0.3 * min(kp1, kp2)) & (K <= 3.0 * max(kp1, kp2))
    mpos = om > 0.3                            # exclude DC / set-down band
    res = {}
    for delta in deltas:
        on_e = off_e = 0.0
        for io in np.flatnonzero(mpos):
            sl = E[io][kband]
            d = np.abs(om[io] - om_d[kband])
            on_e += sl[d <= delta].sum()
            off_e += sl[d > delta].sum()
        res[f'offshell_frac_d{delta}'] = float(off_e / (on_e + off_e))
    return res


def main():
    out = {'configs': {}}
    for mode, n_seas in [('unimodal', 4), ('bimod90same', 4),
                         ('bimod90cross', 8)]:
        rows = []
        for si in range(n_seas):
            sim, cfg = build_sea(mode, si)
            r = offshell_fraction(sim, cfg)
            if r is None:
                print(f"{mode} sea{si}: NaN field — skipped", flush=True)
                continue
            r['si'] = si
            rows.append(r)
            print(f"{mode} sea{si}: " + " ".join(
                f"{k}={v:.3f}" for k, v in r.items() if k != 'si'),
                flush=True)
        out['configs'][mode] = rows
    json.dump(out, open(OUT, 'w'), indent=1)

    # correlate with per-sea skill of the n8 cross run (W=20, leads<=10)
    try:
        from scipy.stats import spearmanr
        d = json.load(open(
            'results/direction3/falcon_hos_probe_bimod90cross_n8.json'))
        leads = np.asarray(d['leads'], float)
        skills = []
        for pred, gt in d['raw']['20.0']:
            pred, gt = np.asarray(pred, float), np.asarray(gt, float)
            n = min(len(pred), len(gt), len(leads))
            m = (leads[:n] <= 10) & np.isfinite(gt[:n]) & np.isfinite(pred[:n])
            skills.append(np.corrcoef(pred[:n][m], gt[:n][m])[0, 1])
        fr = [r['offshell_frac_d0.25']
              for r in out['configs']['bimod90cross']]
        n = min(len(fr), len(skills))
        rho, p = spearmanr(fr[:n], skills[:n])
        print(f"\ncross seas: offshell_frac vs skill(W20): "
              f"Spearman {rho:+.2f} (p={p:.3f}, n={n})")
        out['cross_skill_spearman'] = dict(rho=float(rho), p=float(p), n=n)
        json.dump(out, open(OUT, 'w'), indent=1)
    except Exception as e:
        print('skill join failed:', e)


if __name__ == '__main__':
    main()
