"""Red-team CPU checks for G0-coherent (s2b/s2c) and E5 shadow constraints.

1. Saturation & clip artefacts of p_of_s at high rate (the small-p question).
2. Fisher weight at s=0 vs prior-averaged E_s[w(s)]  (linearisation check).
3. Ray r-distribution: modelled uniform-r vs the sensor's el-uniform grid.
4. Rate-knob invariance of dlnp/ds (the 'structural' claim's real content).
5. E5 geometry: g(r)/sigma_s, active window sigma_eta/g, per-ray survival
   log-evidence -ln Phi(g/sigma_s'), predicted alpha at the margin.
6. MC check of the truncated-Gaussian moment formulas used in ep_solve.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

# ---- replicate p_of_s / calibrate without importing torch-heavy modules ----
from wavesim.raydrop_uniform import fresnel_reflectance

H_SENS, D_REF = 5.0, 125.0
KP = 2 * np.pi / 80.0
SIGMA = 1.0
SIG0 = 0.05 * 4 * SIGMA

def p_of_s(s, r, eta_cal):
    g = np.arctan(H_SENS / r)
    th = np.pi / 2 - g - s
    th = np.clip(th, 0.0, np.pi / 2 - 1e-9)
    P = eta_cal * fresnel_reflectance(th) * np.cos(th) / ((r / D_REF) ** 2 + 1e-9)
    return np.clip(P, 1e-12, 1 - 1e-12)

def calibrate(r, rate):
    lo, hi = 1e-9, 1e9
    for _ in range(80):
        mid = np.sqrt(lo * hi)
        m = p_of_s(0.0, r, mid).mean()
        lo, hi = (mid, hi) if m < rate else (lo, mid)
    return np.sqrt(lo * hi)

rng = np.random.default_rng(0)
r_uni = rng.uniform(15.0, 200.0, 20000)          # the script's modelled rays

print("== 1. saturation at high rate (uniform-r modelled rays) ==")
for rate in (0.001, 0.05, 0.3):
    ec = calibrate(r_uni, rate)
    p0 = p_of_s(0.0, r_uni, ec)
    clip_hi = (p0 >= 1 - 1e-9).mean()
    print(f" rate {rate:5g}: eta {ec:.3e}  frac p clipped@1 {clip_hi:.4f}  "
          f"p quantiles 5/50/95%: {np.quantile(p0,[.05,.5,.95])}")

print("\n== 2. Fisher weight at s=0 vs prior-averaged (flat-prior slope sd) ==")
# flat-prior along-ray slope sd: sigma_s^2 = SIGMA^2 * E[(k.rhat)^2] over bank
# annulus (0.1,2.5)kp uniform-in-area: E[k^2] = 3.128 kp^2, E[(k.rhat)^2]=E[k^2]/2
sig_s = SIGMA * KP * np.sqrt(3.128 / 2)
print(f" flat-prior along-ray slope sd  sigma_s = {sig_s:.4f}")
ss = rng.normal(0, sig_s, 200000)
for rate in (0.001, 0.3):
    ec = calibrate(r_uni, rate)
    for rr in (25.0, 100.0, 200.0):
        h = 1e-4
        p0 = p_of_s(0.0, np.array([rr]), ec)[0]
        dp0 = (p_of_s(h, np.array([rr]), ec)[0] - p_of_s(-h, np.array([rr]), ec)[0]) / (2*h)
        w0 = dp0**2 / (p0*(1-p0))
        # prior-averaged expected Fisher info E_s[ p'(s)^2 / (p(1-p)) ]
        ps = p_of_s(ss, rr, ec)
        dps = (p_of_s(ss+h, rr, ec) - p_of_s(ss-h, rr, ec)) / (2*h)
        wbar = np.mean(dps**2/(ps*(1-ps)))
        # prior-averaged amplitude info per fired ray ~ E[p]/SIG0^2 (per unit |phi|^2)
        amp0, ampbar = p0/SIG0**2, ps.mean()/SIG0**2
        print(f" rate {rate:5g} r={rr:5.0f}: w(0) {w0:9.3g}  E[w] {wbar:9.3g} "
              f"(x{wbar/max(w0,1e-30):7.1f})   amp(0) {amp0:8.3g}  E[amp] {ampbar:8.3g} "
              f"(x{ampbar/amp0:6.1f})   ratio w/amp: s0 {w0/amp0:8.3g}  avg {wbar/ampbar:8.3g}")

print("\n== 3. real fired-ray r-density (el-uniform) vs modelled uniform-r ==")
el = np.deg2rad(np.linspace(-25.0, -1.146, 48))
r_grid = H_SENS / np.tan(-el)
inb = (r_grid > 15) & (r_grid < 200)
print(f" el-uniform grid: {inb.mean():.3f} of rays land in [15,200] m; "
      f"quantiles of r: {np.quantile(r_grid[inb],[.1,.5,.9]).round(1)}")
print(f" modelled uniform-r median r = 107.5; sensor grid median r = "
      f"{np.median(r_grid[inb]):.1f} -> uniform-r OVERWEIGHTS far range (big dlnp/ds)")

print("\n== 4. rate-knob invariance: dlnp/ds vs eta ==")
for rr in (25.0, 100.0, 200.0):
    vals = []
    for rate in (0.001, 0.01, 0.3):
        ec = calibrate(r_uni, rate)
        h = 1e-4
        p0 = p_of_s(0.0, np.array([rr]), ec)[0]
        dp0 = (p_of_s(h, np.array([rr]), ec)[0] - p_of_s(-h, np.array([rr]), ec)[0])/(2*h)
        vals.append(dp0/p0)
    print(f" r={rr:5.0f}: dlnp/ds at rates .001/.01/.3 = {np.round(vals,3)}")

print("\n== 5. E5 geometry (Hs=1.2 Tp=8 spread30, depth 50) ==")
from wavesim.wave_generator import jonswap_components
cs = jonswap_components(1.2, 8.0, n_components=60, depth=50.0,
                        direction_spread_deg=30.0, seed=0)
kx = np.array([c.kx for c in cs]); ky = np.array([c.ky for c in cs])
A = np.array([c.A for c in cs])
sig_eta = np.sqrt((A**2/2).sum())
# along-ray slope for a ray at azimuth 0 (waves spread 30deg): use k_x
sig_sx = np.sqrt((A**2 * kx**2 / 2).sum())
print(f" sigma_eta {sig_eta:.3f} m   along-x slope sd {sig_sx:.4f}")
for rr in (25., 50., 100., 150., 200.):
    g = np.arctan(H_SENS/rr)
    a_near = g / sig_sx                    # tightness of the near-hit slope bit
    surv = -np.log(norm.cdf(a_near))       # per-ray log-evidence of the bit
    dstar = sig_eta / np.tan(g)            # window where clearance < 1 sigma_eta
    alpha_margin = np.tan(g)*3.0 / sig_eta # alpha at the 3 m margin
    print(f" r={rr:5.0f}: g {np.degrees(g):5.2f}deg  g/sig_s {a_near:5.2f} "
          f" -lnPhi {surv:7.4f} nats  window sigma_eta/g {dstar:6.1f} m "
          f" alpha@3m-margin {alpha_margin:5.2f}")

print("\n== 6. MC check of truncated moments used in ep_solve ==")
mc, vc, hcv = 0.3, 0.8, 0.5
x = rng.normal(mc, np.sqrt(vc), 4_000_000)
for sgn, name in ((1.0, 'x<h'), (-1.0, 'x>h')):
    keep = x < hcv if sgn > 0 else x > hcv
    al = sgn*(hcv-mc)/np.sqrt(vc); Z = norm.cdf(al); lam = norm.pdf(al)/Z
    m_hat = mc - sgn*np.sqrt(vc)*lam
    v_hat = vc*(1-lam*lam-al*lam)
    print(f" {name}: MC mean {x[keep].mean():+.5f} formula {m_hat:+.5f} | "
          f"MC var {x[keep].var():.5f} formula {v_hat:.5f}")
