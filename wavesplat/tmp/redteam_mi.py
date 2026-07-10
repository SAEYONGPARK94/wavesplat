"""The correct currency for a Bernoulli mask bit: mutual information I(m; s),
computed by 1D quadrature. Fisher-at-s=0 (what s2b/s2c used) and prior-averaged
Fisher E[w] are both wrong for bits: the first ignores the law's curvature over
the prior, the second violates the bit entropy cap. I(m;s) is exact, cheap and
sampling-free. Data-processing: p(m|a) = p(m|s(a)) => I(m;a) = I(m;s).

Also: E5's per-ray shadow budget -ln Phi(g/sigma_s) with the sensor's real
hit r-density, vs the amplitude nats that arrive with the same rays.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from scipy.stats import norm

from wavesim.raydrop_uniform import fresnel_reflectance

H_SENS, D_REF = 5.0, 125.0
KP = 2*np.pi/80.0; SIGMA = 1.0; SIG0 = 0.2
SIG_S = 0.0982                      # flat-prior along-ray slope sd (s2 sea)

def p_of_s(s, r, eta_cal):
    g = np.arctan(H_SENS/r)
    th = np.clip(np.pi/2 - g - s, 0.0, np.pi/2 - 1e-9)
    return np.clip(eta_cal*fresnel_reflectance(th)*np.cos(th)/((r/D_REF)**2+1e-9),
                   1e-15, 1-1e-15)

# sensor ray population: el-uniform grid -> r density prop 5/(r^2+25), in [15,200]
rr = np.linspace(15.0, 200.0, 400)
dens = H_SENS/(rr**2 + H_SENS**2); dens /= np.trapezoid(dens, rr)

# quadrature nodes for s ~ N(0, SIG_S^2)
qs, qw = np.polynomial.hermite_e.hermegauss(201)
ss = qs*SIG_S; ws = qw/np.sum(qw)

def calibrate_pop(rate):
    """eta so that the POPULATION mean of E_s[p(s,r)] over the ray density = rate.
    (fair: the real sensor's rate is measured over real slopes, not s=0)"""
    lo, hi = 1e-9, 1e12
    for _ in range(90):
        mid = np.sqrt(lo*hi)
        pm = np.array([np.sum(ws*p_of_s(ss, r, mid)) for r in rr])
        m = np.trapezoid(pm*dens, rr)
        lo, hi = (mid, hi) if m < rate else (lo, mid)
    return np.sqrt(lo*hi)

def h2(p):  # binary entropy, nats
    return -(p*np.log(p) + (1-p)*np.log(1-p))

amp_per_return = 0.5*np.log(1 + SIGMA**2/SIG0**2)
print(f"amp info per return (elevation functional): {amp_per_return:.3f} nats")
print(f"{'rate':>7} {'I_mask tot':>11} {'I_amp tot':>10} {'ratio':>7}   "
      f"I_mask(r) at r=25/100/200 per hit")
for rate in (1e-3, 1e-2, 1e-1, 3e-1):
    ec = calibrate_pop(rate)
    Im = np.zeros_like(rr); pm = np.zeros_like(rr)
    for i, r in enumerate(rr):
        p = p_of_s(ss, r, ec)
        pbar = np.sum(ws*p)
        Im[i] = h2(pbar) - np.sum(ws*h2(p))          # I(m;s) exact
        pm[i] = pbar
    tot_mask = np.trapezoid(Im*dens, rr)                 # nats per fired(hit) ray
    tot_amp = np.trapezoid(pm*dens, rr)*amp_per_return
    i25, i100, i200 = [Im[np.argmin(abs(rr-v))] for v in (25,100,200)]
    print(f"{rate:7.3g} {tot_mask:11.3e} {tot_amp:10.3e} {tot_mask/tot_amp:7.2%}   "
          f"{i25:.2e} / {i100:.2e} / {i200:.2e}")

print("\nbit caps at deployed rate: H(pbar) profile r=25/100/200:")
ec = calibrate_pop(1e-3)
for v in (25,100,200):
    p = p_of_s(ss, v, ec); pbar = np.sum(ws*p)
    print(f"  r={v:3d}: pbar {pbar:.2e}  H(pbar) {h2(pbar):.2e} nats")

# ---------------- E5 shadow budget at its own sea ----------------
print("\n== E5 budget: survival bit per returned ray vs its amplitude ==")
sig_eta, sig_sx, noise = 0.300, 0.0305, 0.05
amp_e5 = 0.5*np.log(1 + sig_eta**2/noise**2)
g = np.arctan(H_SENS/rr)
surv = -norm.logcdf(g/sig_sx)                        # PRIOR-sd upper bound
tot = np.trapezoid(surv*dens, rr)                        # per returned ray, r-avg
print(f"amp per return {amp_e5:.3f} nats; survival bit r-avg {tot:.4f} nats "
      f"-> ratio {tot/amp_e5:.2%}")
print(f"survival bit at r=25/50/100/150/200: "
      + "/".join(f"{surv[np.argmin(abs(rr-v))]:.4f}" for v in (25,50,100,150,200)))
# 600 rays: total budget vs amplitude total
print(f"600 rays: shadow total <= {600*tot:.1f} nats vs amp total {600*amp_e5:.0f} nats")
# what nRMSE gain can ~N nats buy? crude: nRMSE ~ exp(-nats/dof_effective)
