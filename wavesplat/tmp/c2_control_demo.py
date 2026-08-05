"""C2 — envelope-gated feedforward heave-compensation demo (pre-registered:
tmp/C2_PREREG_control_2026-07-22.md). Does the CRB operability envelope, used to
modulate a feedforward gain, beat the best constant-gain controller on HOS seas?

Task: residual heave = eta_true - g*eta_hat; minimize residual variance (felt motion).
Controllers (gain per query point): naive-full g=1 | naive-best-const (train-fit) |
envelope-gated g(sigma_CRB) (train-calibrated) | oracle g(true-error) (test upper-bound).
Metric R = 1 - sum(res^2)/sum(eta_true^2). Primary: R_gated > R_naive_const (sea-block boot CI).

Usage: python3 tmp/c2_control_demo.py [--n 16]
Out: results/direction3/c2_control_demo.json
"""
import json, os, sys, time, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts'); sys.path.insert(0, 'tmp')
from hos_s8_envelope import (SEAS, build_sea, draw, two_stage_kp, fit_theta_and_pred, crb_envelope)


def ls_gain(yt, yh):
    """best constant LS gain g minimizing sum(yt-g*yh)^2."""
    d = float(np.sum(yh * yh))
    return (float(np.sum(yt * yh)) / d) if d > 1e-12 else 0.0


def fit_wiener(sig, yt, yh):
    """primary gated policy: g(sigma)=g0/(1+c*sigma^2), monotone-decreasing (Wiener form).
    grid over c>=0, closed-form g0 per c; pick min residual energy. Returns (g0, c)."""
    s2 = sig ** 2
    best = (0.0, 0.0, np.inf)
    for c in np.concatenate([[0.0], np.geomspace(1e-2, 1e5, 60)]):
        w = 1.0 / (1.0 + c * s2); wy = w * yh
        d = float(np.sum(wy * wy))
        g0 = float(np.sum(yt * wy)) / d if d > 1e-12 else 0.0
        res = float(np.sum((yt - g0 * wy) ** 2))
        if res < best[2]:
            best = (g0, float(c), res)
    return best[0], best[1]


def wiener_gain(sig, g0, c):
    return g0 / (1.0 + c * sig ** 2)


def binned_gain_curve(sig, yt, yh, nbin=10):
    """quantile-bin by sig; per-bin LS gain. returns (edges, gains) for lookup."""
    edges = np.quantile(sig, np.linspace(0, 1, nbin + 1))
    edges[0] -= 1e-9; edges[-1] += 1e-9
    b = np.clip(np.digitize(sig, edges) - 1, 0, nbin - 1)
    gains = np.array([ls_gain(yt[b == j], yh[b == j]) if (b == j).sum() >= 5 else np.nan
                      for j in range(nbin)])
    # fill empty bins by nearest finite
    for j in range(nbin):
        if not np.isfinite(gains[j]):
            fin = np.where(np.isfinite(gains))[0]
            gains[j] = gains[fin[np.argmin(np.abs(fin - j))]] if len(fin) else 0.0
    return edges, gains


def apply_curve(sig, edges, gains):
    b = np.clip(np.digitize(sig, edges) - 1, 0, len(gains) - 1)
    return gains[b]


def R_of(yt, g):
    """variance-reduction R given per-point gain g applied to yh (yh carried via closure)."""
    pass  # replaced below


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=16)
    ap.add_argument('--nbin', type=int, default=10); a = ap.parse_args()
    N = min(a.n, len(SEAS))
    CACHE = 'tmp/c2_cache'; os.makedirs(CACHE, exist_ok=True)
    print(f"C2 control demo: {N} HOS seas (train=even idx, test=odd idx), nbin={a.nbin}\n", flush=True)
    per = []  # per-sea dict of arrays
    t0 = time.time()
    for si in range(N):
        Tp, Hs = SEAS[si]
        cf = f'{CACHE}/sea_{si}.npz'
        if os.path.exists(cf):
            d = np.load(cf, allow_pickle=True)
            per.append(dict(si=si, sea=str(d['sea']), sig=d['sig'], yh=d['yh'], yt=d['yt']))
            print(f"  [{si:2d}] {str(d['sea']):12s} CACHED nq={len(d['yt']):4d}  [{time.time()-t0:.0f}s]", flush=True)
            continue
        try:
            sea = build_sea(Tp, Hs, 0)
            Ps, zs, Q, gt = draw(sea, np.random.default_rng(900000 + 31 * si))
            Hs_e = 4 * np.std(zs); kp = two_stage_kp(Ps, zs)
            theta, pred, emp = fit_theta_and_pred(Ps, zs, Q, kp)
            sig = crb_envelope(theta, Ps, Q, Hs_e)
            fin = np.isfinite(sig) & np.isfinite(pred) & np.isfinite(gt)
            rec = dict(si=si, sea=f'Tp{Tp:g}_Hs{Hs:g}', sig=sig[fin], yh=pred[fin], yt=gt[fin])
            per.append(rec)
            np.savez(cf, sea=rec['sea'], sig=rec['sig'], yh=rec['yh'], yt=rec['yt'])
            print(f"  [{si:2d}] {rec['sea']:12s} nq={int(fin.sum()):4d}  "
                  f"sig[{np.nanmin(sig[fin]):.3f},{np.nanmax(sig[fin]):.3f}]  [{time.time()-t0:.0f}s]", flush=True)
        except Exception as ex:
            import traceback; print(f"  sea {si} ERR {ex}"); traceback.print_exc()

    train = [p for p in per if p['si'] % 2 == 0]
    test = [p for p in per if p['si'] % 2 == 1]
    tr_sig = np.concatenate([p['sig'] for p in train]); tr_yt = np.concatenate([p['yt'] for p in train]); tr_yh = np.concatenate([p['yh'] for p in train])

    # --- fit policies on TRAIN ---
    g_const = ls_gain(tr_yt, tr_yh)                                  # naive-best-const
    g0, cc = fit_wiener(tr_sig, tr_yt, tr_yh)                        # PRIMARY gated (Wiener, monotone)
    edges, gains = binned_gain_curve(tr_sig, tr_yt, tr_yh, a.nbin)   # secondary robustness (binned)
    sq = np.quantile(tr_sig, [0.1, 0.5, 0.9])
    print(f"\nTRAIN policies: best-const g={g_const:+.3f} | Wiener g0={g0:+.3f} c={cc:.3g}")
    print(f"  Wiener gain @ sigma p10/p50/p90 ({sq[0]:.3f}/{sq[1]:.3f}/{sq[2]:.3f}): "
          f"{wiener_gain(sq[0],g0,cc):.2f}/{wiener_gain(sq[1],g0,cc):.2f}/{wiener_gain(sq[2],g0,cc):.2f}")

    def R_controllers(seas):
        """compute R for each controller on a list of sea dicts (pooled)."""
        yt = np.concatenate([p['yt'] for p in seas]); yh = np.concatenate([p['yh'] for p in seas])
        sig = np.concatenate([p['sig'] for p in seas]); denom = float(np.sum(yt * yt))
        def R(g): return 1.0 - float(np.sum((yt - g * yh) ** 2)) / denom
        # proper upper bound for sigma-conditioned policies: bin TEST by sigma, per-bin LS gain on TEST
        se, sg = binned_gain_curve(sig, yt, yh, a.nbin); R_sig_oracle = R(apply_curve(sig, se, sg))
        # diagnostic reference: bin by realized |error| (NOT an upper bound over policy classes)
        e = np.abs(yt - yh); oe, og = binned_gain_curve(e, yt, yh, a.nbin); R_err_ref = R(apply_curve(e, oe, og))
        return dict(full=R(1.0), const=R(g_const),
                    gated=R(wiener_gain(sig, g0, cc)),          # PRIMARY (train-fit Wiener)
                    gated_bin=R(apply_curve(sig, edges, gains)),  # robustness (train-fit binned)
                    sig_oracle=R_sig_oracle,                    # upper bound: best sigma-conditioned (test-fit)
                    err_ref=R_err_ref)                          # diagnostic: error-tier reference

    Rtest = R_controllers(test)
    delta = Rtest['gated'] - Rtest['const']
    gap = (Rtest['gated'] - Rtest['const']) / (Rtest['sig_oracle'] - Rtest['const'] + 1e-12)

    # --- sea-block bootstrap on TEST for delta (primary Wiener policy, fixed train fit) ---
    rng = np.random.default_rng(7)
    boot = []
    for _ in range(2000):
        samp = [test[i] for i in rng.integers(0, len(test), len(test))]
        yt = np.concatenate([p['yt'] for p in samp]); yh = np.concatenate([p['yh'] for p in samp])
        sig = np.concatenate([p['sig'] for p in samp]); denom = float(np.sum(yt * yt))
        def R(g): return 1.0 - float(np.sum((yt - g * yh) ** 2)) / denom
        boot.append(R(wiener_gain(sig, g0, cc)) - R(g_const))
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    # per-test-sea delta (consistency)
    persea = []
    for p in test:
        denom = float(np.sum(p['yt'] * p['yt']))
        def R(g): return 1.0 - float(np.sum((p['yt'] - g * p['yh']) ** 2)) / denom
        persea.append(dict(sea=p['sea'], d=R(wiener_gain(p['sig'], g0, cc)) - R(g_const)))

    win = delta > 0 and lo > 0
    verdict = ("WIN: envelope-gated control beats best-const (CI lower>0) -> decision-value headline authorized"
               if win else
               "NO-WIN: gated does not beat best-const (CI includes 0) -> fall back to C1 ceiling headline, envelope=stat UQ")

    print(f"\n=== C2 RESULT (TEST = {len(test)} HOS seas, {sum(len(p['yt']) for p in test)} query pts) ===")
    print(f"  R  naive-full (g=1)     {Rtest['full']:+.4f}   (harm if < R_const: full-trust hurts)")
    print(f"  R  naive-best-const     {Rtest['const']:+.4f}   [primary baseline]")
    print(f"  R  envelope-gated       {Rtest['gated']:+.4f}   [ours, Wiener]  (binned {Rtest['gated_bin']:+.4f})")
    print(f"  R  sigma-oracle (test)  {Rtest['sig_oracle']:+.4f}   [upper bound: best sigma-conditioned]")
    print(f"  R  err-tier ref         {Rtest['err_ref']:+.4f}   [diagnostic, not a bound]")
    print(f"  Δ (gated - const)       {delta:+.4f}   boot95% [{lo:+.4f}, {hi:+.4f}]  (2000 sea-block)")
    print(f"  sigma-oracle gap closed {gap*100:.0f}%   (gated vs best achievable from sigma)")
    print(f"  per-test-sea Δ>0        {sum(1 for s in persea if s['d']>0)}/{len(persea)}")
    print(f"\n=== C2 VERDICT === {verdict}")

    json.dump(dict(N=N, n_test=len(test), n_train=len(train), g_const=g_const,
                   wiener=dict(g0=g0, c=cc), gated_curve=dict(edges=edges.tolist(), gains=gains.tolist()),
                   R=Rtest, delta=delta, boot_lo=lo, boot_hi=hi, oracle_gap=gap,
                   per_sea=persea, verdict=verdict),
              open('results/direction3/c2_control_demo.json', 'w'), indent=1, default=float)
    print(f"saved results/direction3/c2_control_demo.json ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
