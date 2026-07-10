"""Does the mask/amp ratio move under a THRESHOLD detection law?

Real LiDAR detection: return iff received power > threshold T (with speckle).
Exponential speckle: p(s) = P(A(s)*X > T), X ~ Exp(1)  =>  p = exp(-T/A(s)).
The rate knob is T (or laser power), NOT a multiplicative factor on p:
    dlnp/ds = (T/A) * dlnA/ds = (-ln p) * dlnA/ds
so sparser sensing (smaller p) makes the mask bit MORE log-sensitive.
Same A(s) radiometric shape as the simulator (Fresnel*cos/d^2).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from wavesim.raydrop_uniform import fresnel_reflectance

H_SENS, D_REF = 5.0, 125.0
SIG_S = 0.0982; SIGMA = 1.0; SIG0 = 0.2

def A_of_s(s, r):
    g = np.arctan(H_SENS/r)
    th = np.clip(np.pi/2 - g - s, 0.0, np.pi/2 - 1e-9)
    return fresnel_reflectance(th)*np.cos(th)/((r/D_REF)**2 + 1e-9)

rr = np.linspace(15.0, 200.0, 300)
dens = H_SENS/(rr**2 + H_SENS**2); dens /= np.trapezoid(dens, rr)
qs, qw = np.polynomial.hermite_e.hermegauss(301)
ss = qs*SIG_S; ws = qw/np.sum(qw)

def h2(p):
    p = np.clip(p, 1e-300, 1-1e-15)
    return -(p*np.log(p) + (1-p)*np.log(1-p))

def stats(T):
    Im = np.zeros_like(rr); pm = np.zeros_like(rr)
    for i, r in enumerate(rr):
        p = np.exp(-T/np.maximum(A_of_s(ss, r), 1e-30))
        pbar = np.sum(ws*p)
        Im[i] = h2(pbar) - np.sum(ws*h2(p)); pm[i] = pbar
    rate = np.trapezoid(pm*dens, rr)
    mask = np.trapezoid(Im*dens, rr)
    amp = rate*0.5*np.log(1 + SIGMA**2/SIG0**2)
    return rate, mask, amp

def find_T(rate):
    lo, hi = 1e-12, 1e3
    for _ in range(80):
        mid = np.sqrt(lo*hi)
        r, _, _ = stats(mid)
        lo, hi = (lo, mid) if r < rate else (mid, hi)
    return np.sqrt(lo*hi)

print(f"{'rate':>8} {'I_mask':>10} {'I_amp':>10} {'ratio':>8}")
for rate in (1e-3, 1e-2, 1e-1, 3e-1):
    T = find_T(rate)
    r, m, a = stats(T)
    print(f"{r:8.3g} {m:10.3e} {a:10.3e} {m/a:8.2%}")
