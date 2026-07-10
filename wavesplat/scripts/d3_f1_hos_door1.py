"""d3_f1_hos_door1.py  --  FALSIFIER F1: does "Door 1 is closed" survive HOS?

============================ WHY THIS SCRIPT EXISTS ==========================
E5 (docs/E5_DOOR1_IS_ONE_CHANNEL_2026-07-10.md) proved a single-channel closure
of Door 1 FOR A LINEAR GAUSSIAN SEA: the entire pre-hit ray-inequality set of a
returned ray collapses to one one-sided bit on the ALONG-RAY SLOPE at the hit,
worth -ln Phi(g/sigma_s) nats, and the mask channel is worth I(m;s) nats.  On
the reference sea (Hs 1.2, Tp 8, depth 50, spread 30, h=5) the sensor's real
1/r^2 hit density buries both:
    shadow budget  = 0.0127 nats/ray = 0.70 % of 1.805 amplitude nats/ray
                     -> ceiling  Delta nRMSE ~ +0.002
    mask MI ratio  = 2.46 % at deployed return rate 1e-3.
The closure theorem assumes (A1) a LINEAR GAUSSIAN field.  Non-Gaussianity is
the named falsifier F1: nonlinear seas have sharper crests / flatter troughs,
so the along-ray slope distribution skews and its tail fattens (crest-front
steepness).  If sigma_s grows or the positive tail fattens, g/sigma_s drops,
P(s<g) shrinks, the buried bit grows -- and it grows exactly at FAR range where
the linear bit is already largest.  Project standing rule: NO headline enters
the thesis until it reproduces on HOS.  Door 1 has never touched HOS.  F1 does.

s := d eta / d(range), the OUTWARD along-ray horizontal slope (increasing r,
away from the sensor).  s>0 is a sensor-facing near-face; a face steeper than
the grazing angle g=atan(h/r) shadows what is behind it, so the survival /
un-shadowed condition is s<g and the bit is -ln P(s<g).  Waves are run TOWARD
the sensor (direction 180, sensor looks +x) -- the control-relevant and
WORST-CASE geometry, because that is when nonlinear crest fronts face the
sensor and load the positive tail of s.  (Linear s is symmetric, so the linear
anchor is direction-independent; only the HOS skew depends on it.)

Statistics are computed on the POOLED ENSEMBLE slope distribution (all
seeds x instants), because that is the ensemble object the frozen sigma_s=0.0305
and 0.70 % were derived from -- a single 512 m sea realisation carries only a
few independent wave crossings across the fan, so its per-realisation sigma_s
is a noisy 0.022-0.035 draw.  Per-seed values give the clustered SE.

============================ PRE-REGISTRATION (frozen before running) =========
Fields, all matched to the reference sea (Hs 1.2, Tp 8, depth 50, spread 30,
h=5, ranges 15-200 m), waves toward the sensor:
  hos_m1          : HOSSimulator M=1 (linear evolution, full spectral grid).
                    THE MATCHED LINEAR ANCHOR: must reproduce the frozen numbers
                    (sigma_s~0.0305, bits {25:.000 50:.0005 100:.052 150:.148
                    200:.231}, budget ~0.70 %) -- it is the linear baseline for
                    the HOS ratio, same machinery as hos_m3.
  hos_m3          : HOSSimulator M=3 (3rd-order nonlinear).  THE PRIMARY TEST.
  hos_m3_steep    : HOS M=3 at Hs=3.0 (steep) -- NOT gated.  POSITIVE CONTROL:
                    proves the instrument is not blind to nonlinearity (its slope
                    skew must be >> hos_m3's).
  lin_jonswap     : jonswap_components (60 comps), analytic.  Coarser spectrum
                    (sigma_s~0.025) -- the self-consistent LINEAR baseline for:
  stokes2         : analytic 2nd-order directional bound wave (Sharma-Dean, all
                    pair sum+diff interactions) on the same 60 comps.  The
                    CONSERVATIVE cross-check: it resolves EVERY harmonic, whereas
                    the HOS grid truncates 2nd harmonics above its Nyquist and so
                    UNDER-represents crest-front steepness.  B4: bound waves are
                    the dominant non-Gaussianity, corr(measured, rho^2)=+0.997.
                    Reported as the multiplicative inflation stokes2/lin_jonswap.

MEASUREMENTS (n>=8 seeds, pooled headline + mean +- clustered SE):
  V1  sigma_s(HOS)/sigma_s(linear), + skewness and excess kurtosis of s.
  V2  shadow budget from the EMPIRICAL slope CDF (NOT Phi): per-ray -ln P(s<g)
      at r in {25,50,100,150,200} m, then the 1/r^2 (h/(r^2+h^2)) weighted total
      vs the amplitude nats/ray, as a % (linear anchor ~0.70 %).
  V3  mask MI ratio I(m;s)/amp with the EMPIRICAL slope density in the redteam
      quadrature, at deployed return rate 1e-3.
  V4  hit rate on the HOS mesh (Moller-Trumbore first hit, MESH_DX 2 m over
      x in [0,260], y in [-240,240]) vs flat-sea -- occlusion must not create
      missingness.

GATES (frozen now):  the closure claim SURVIVES iff
  (a)  HOS weighted shadow budget <= 1.4 % of amplitude nats  (<= 2x the linear
       0.70 %, i.e. ceiling Delta nRMSE <= +0.004),   AND
  (b)  HOS mask MI ratio <= 5 %.
If EITHER is exceeded -> F1 FIRES: "closed" may NOT be written in the thesis;
quantify the reopened prize.  Gates are applied to hos_m3 (the literal HOS gate);
the stokes2 inflation (all harmonics) is reported as a bandwidth-complete
sensitivity, applied to the frozen 0.70 % anchor.  Either outcome is a result;
nothing is softened.  HOSSimulator is numerically stable here (smoke-tested:
Hs conserved, no NaN at M=3), so it is the primary; stokes2 corroborates.  Had
HOS blown up, the fallback would have been stokes2 alone, labelled order-2 only.
=============================================================================
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from wavesim.wave_generator import jonswap_components
from wavesim.hos_simulator import HOSSimulator, HOSConfig
from wavesim.lidar_simulator import build_mesh, raycast_surface
from wavesim.raydrop_uniform import fresnel_reflectance
from d3_door1b_field_bias import ray_grid, H_SENS, R_MIN, R_MAX
from generate_nn_training_data import _compute_stokes2_correction_np

G = 9.81
OUT = 'results/direction3/f1_hos_door1.json'

HS, TP, DEPTH, SPREAD = 1.2, 8.0, 50.0, 30.0
NCOMP = 60
WDIR = 180.0
LX = LY = 512.0
NX = NY = 128
CX, CY = 156.0, 256.0
SPINUP = 12.0
INSTANTS = (12.0, 14.0, 16.0)
DT = 0.10
FD = 0.20
AMP_NOISE = 0.05

RGRID = np.linspace(R_MIN, R_MAX, 400)
DENS = H_SENS / (RGRID ** 2 + H_SENS ** 2)
DENS = DENS / np.trapezoid(DENS, RGRID)
RTAB = (25, 50, 100, 150, 200)

_O, _D, _FOOT = ray_grid()
_FR = np.hypot(_FOOT[:, 0], _FOOT[:, 1])
_INB = (_FR > R_MIN) & (_FR < R_MAX)
FOOT = _FOOT[_INB]
FR = _FR[_INB]
UX, UY = FOOT[:, 0] / FR, FOOT[:, 1] / FR


# ============================ field slope samplers =========================
def sea_components(seed, Hs=HS):
    cs = jonswap_components(Hs, TP, n_components=NCOMP, depth=DEPTH,
                            direction_deg=WDIR, direction_spread_deg=SPREAD,
                            seed=seed)
    return dict(kx=np.array([c.kx for c in cs]), ky=np.array([c.ky for c in cs]),
                om=np.array([c.omega for c in cs]), A=np.array([c.A for c in cs]),
                ph=np.array([c.phi for c in cs]))


def eta_linear(S, x, y, t, stokes2=False):
    psi = np.outer(x, S['kx']) + np.outer(y, S['ky']) - t * S['om'][None] + S['ph']
    z = (S['A'][None] * np.cos(psi)).sum(1)
    if stokes2:
        z = z + _compute_stokes2_correction_np(
            S['kx'], S['ky'], S['A'], S['ph'], S['om'], DEPTH,
            np.asarray(x, float), np.asarray(y, float), float(t))
    return z


def slopes_analytic(S, t, stokes2=False):
    xp, yp = CX + FOOT[:, 0], CY + FOOT[:, 1]
    zf = eta_linear(S, xp + FD * UX, yp + FD * UY, t, stokes2)
    zb = eta_linear(S, xp - FD * UX, yp - FD * UY, t, stokes2)
    return (zf - zb) / (2 * FD)


def hos_modes(sim, rel=1e-5):
    a = sim.eta_hat
    m = np.abs(a) > rel * np.abs(a).max()
    return sim.KX[m].copy(), sim.KY[m].copy(), a[m].copy()


def hos_eval(kx, ky, a, xs, ys, grad=True):
    ex = np.zeros(len(xs)); ey = np.zeros(len(xs)); ez = np.zeros(len(xs))
    B = 512
    for i in range(0, len(kx), B):
        kb, lb, ab = kx[i:i + B], ky[i:i + B], a[i:i + B]
        e = np.exp(1j * (np.outer(kb, xs) + np.outer(lb, ys)))
        if grad:
            ex += np.real(1j * kb[:, None] * ab[:, None] * e).sum(0)
            ey += np.real(1j * lb[:, None] * ab[:, None] * e).sum(0)
        else:
            ez += np.real(ab[:, None] * e).sum(0)
    return (ex, ey) if grad else ez


def slopes_hos(sim):
    kx, ky, a = hos_modes(sim)
    ex, ey = hos_eval(kx, ky, a, CX + FOOT[:, 0], CY + FOOT[:, 1], grad=True)
    return UX * ex + UY * ey


def eta_hos_fan(sim):
    kx, ky, a = hos_modes(sim)
    return hos_eval(kx, ky, a, CX + FOOT[:, 0], CY + FOOT[:, 1], grad=False)


def hos_run(seed, M, Hs=HS):
    sim = HOSSimulator(HOSConfig(Lx=LX, Ly=LY, Nx=NX, Ny=NY, depth=DEPTH, M=M))
    sim.initialize_jonswap(Hs=Hs, Tp=TP, direction_deg=WDIR,
                           direction_spread_deg=SPREAD, seed=seed)
    return sim


# ============================ per-field statistics =========================
def moments(s):
    m, sd = float(s.mean()), float(s.std())
    z = (s - m) / sd
    return dict(mean=m, sd=sd, skew=float((z ** 3).mean()),
                exkurt=float((z ** 4).mean() - 3.0))


def shadow_budget(s, sigma_eta):
    ss = np.sort(s)
    g_tab = np.arctan(H_SENS / np.array(RTAB, float))
    bit_tab = -np.log(np.clip(np.searchsorted(ss, g_tab, 'right') / len(ss),
                              1e-12, 1.0))
    g_grid = np.arctan(H_SENS / RGRID)
    bit_grid = -np.log(np.clip(np.searchsorted(ss, g_grid, 'right') / len(ss),
                               1e-12, 1.0))
    total = float(np.trapezoid(bit_grid * DENS, RGRID))
    amp = 0.5 * np.log(1 + sigma_eta ** 2 / AMP_NOISE ** 2)
    return dict(bits={str(r): float(b) for r, b in zip(RTAB, bit_tab)},
                total_nats=total, amp_nats=float(amp), ratio=float(total / amp),
                ceiling_dnrmse=float(total * (0.002 / 0.0127)))


# ---- V3 mask MI quadrature (vectorised; p_of_s is linear in eta_cal) --------
MI_DREF = 125.0
RR = np.linspace(R_MIN, R_MAX, 400)
MI_DENS = H_SENS / (RR ** 2 + H_SENS ** 2)
MI_DENS = MI_DENS / np.trapezoid(MI_DENS, RR)


def _base_matrix(s_nodes):
    """base(r, s) = R_fresnel(th) cos(th) / (r/dref)^2, th = pi/2 - g - s.
    p_of_s = clip(eta_cal * base).  (n_r, n_nodes)."""
    g = np.arctan(H_SENS / RR)[:, None]
    th = np.clip(np.pi / 2 - g - s_nodes[None, :], 0.0, np.pi / 2 - 1e-9)
    return (fresnel_reflectance(th) * np.cos(th)
            / ((RR[:, None] / MI_DREF) ** 2 + 1e-9))


def _h2(p):
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def mask_mi(s_nodes, s_w, rate=1e-3, sigma_amp=1.0, sig0=0.2):
    base = _base_matrix(s_nodes)                          # (n_r, n_nodes)
    lo, hi = 1e-9, 1e12
    for _ in range(90):
        mid = np.sqrt(lo * hi)
        pm = (np.clip(mid * base, 1e-15, 1 - 1e-15) * s_w[None]).sum(1)
        m = np.trapezoid(pm * MI_DENS, RR)
        lo, hi = (mid, hi) if m < rate else (lo, mid)
    ec = np.sqrt(lo * hi)
    P = np.clip(ec * base, 1e-15, 1 - 1e-15)              # (n_r, n_nodes)
    pbar = (P * s_w[None]).sum(1)
    Im = _h2(pbar) - (_h2(P) * s_w[None]).sum(1)
    tot_mask = float(np.trapezoid(Im * MI_DENS, RR))
    amp_per = 0.5 * np.log(1 + sigma_amp ** 2 / sig0 ** 2)
    tot_amp = float(np.trapezoid(pbar * MI_DENS, RR) * amp_per)
    return tot_mask / tot_amp


def empirical_density(s, nbins=301):
    lo, hi = np.percentile(s, 0.05), np.percentile(s, 99.95)
    edges = np.linspace(lo, hi, nbins + 1)
    cnt, _ = np.histogram(s, edges)
    w = cnt / cnt.sum()
    return 0.5 * (edges[:-1] + edges[1:]), w


def gauss_density(sigma, n=201):
    qs, qw = np.polynomial.hermite_e.hermegauss(n)
    return qs * sigma, qw / np.sum(qw)


# ============================ V4 hit rate ==================================
MESH_DX = 2.0
DOM_X4, DOM_Y4 = (0.0, 260.0), (-240.0, 240.0)


def hit_rate_flat():
    r = np.hypot(_FOOT[:, 0], _FOOT[:, 1])
    return float(((_D[:, 2] < 0) & (r > R_MIN) & (r < R_MAX)).mean())


def hit_rate_field(eval_z):
    gx = np.arange(DOM_X4[0], DOM_X4[1] + MESH_DX, MESH_DX)
    gy = np.arange(DOM_Y4[0], DOM_Y4[1] + MESH_DX, MESH_DX)
    X, Y = np.meshgrid(gx, gy)
    Z = eval_z(X.ravel(), Y.ravel()).reshape(X.shape)
    V, T = build_mesh(X, Y, Z)
    hm, hp, _ = raycast_surface(_O, _D, V, T, range_min=R_MIN, range_max=R_MAX,
                                batch_size=8000)
    r = np.hypot(hp[:, 0], hp[:, 1])
    return float((hm & (r > R_MIN) & (r < R_MAX)).mean())


def clustered(v):
    v = np.asarray(v, float)
    se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float('inf')
    return float(v.mean()), float(se)


FIELDS = ('hos_m1', 'hos_m3', 'hos_m3_steep', 'lin_jonswap', 'stokes2')


def field_stats(slopes, sigeta):
    """headline stats on a pooled slope array."""
    mo = moments(slopes)
    bud = shadow_budget(slopes, sigeta)
    nodes, w = empirical_density(slopes)
    mi = mask_mi(nodes, w, rate=1e-3)
    return mo, bud, mi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seas', type=int, default=8)
    ap.add_argument('--v4-seeds', type=int, default=4)
    a = ap.parse_args()

    print(__doc__)
    print(f"run: seas={a.seas}  instants/seed={len(INSTANTS)}  fan rays="
          f"{len(FOOT)}  HOS {NX}x{NY} L={LX:.0f} M=3\n", flush=True)

    seed_slopes = {f: {} for f in FIELDS}       # [field][seed] -> pooled array
    seed_sig = {f: {} for f in FIELDS}

    for seed in range(a.seas):
        S = sea_components(seed)
        Ssteep = sea_components(seed, Hs=3.0)          # noqa: F841 (parallel seed)
        sim1 = hos_run(seed, 1)
        sim3 = hos_run(seed, 3)
        simS = hos_run(seed, 3, Hs=3.0)
        for sm in (sim1, sim3, simS):
            while sm.t < SPINUP - 1e-9:
                sm.step(min(DT, SPINUP - sm.t))

        acc = {f: [] for f in FIELDS}
        sig = {f: [] for f in FIELDS}
        for tgt in INSTANTS:
            for sm in (sim1, sim3, simS):
                while sm.t < tgt - 1e-9:
                    sm.step(min(DT, tgt - sm.t))
            acc['hos_m1'].append(slopes_hos(sim1))
            acc['hos_m3'].append(slopes_hos(sim3))
            acc['hos_m3_steep'].append(slopes_hos(simS))
            acc['lin_jonswap'].append(slopes_analytic(S, tgt, stokes2=False))
            acc['stokes2'].append(slopes_analytic(S, tgt, stokes2=True))
            sig['hos_m1'].append(np.std(eta_hos_fan(sim1)))
            sig['hos_m3'].append(np.std(eta_hos_fan(sim3)))
            sig['hos_m3_steep'].append(np.std(eta_hos_fan(simS)))
            sig['lin_jonswap'].append(
                np.std(eta_linear(S, CX + FOOT[:, 0], CY + FOOT[:, 1], tgt)))
            sig['stokes2'].append(np.std(
                eta_linear(S, CX + FOOT[:, 0], CY + FOOT[:, 1], tgt, stokes2=True)))
        for f in FIELDS:
            seed_slopes[f][seed] = np.concatenate(acc[f])
            seed_sig[f][seed] = float(np.mean(sig[f]))
        print(f"seed {seed:2d} sig_s(per-seed): "
              f"m1 {seed_slopes['hos_m1'][seed].std():.4f} "
              f"m3 {seed_slopes['hos_m3'][seed].std():.4f} "
              f"stk2 {seed_slopes['stokes2'][seed].std():.4f} "
              f"steep {seed_slopes['hos_m3_steep'][seed].std():.4f} | "
              f"skew m3 {moments(seed_slopes['hos_m3'][seed])['skew']:+.3f} "
              f"steep {moments(seed_slopes['hos_m3_steep'][seed])['skew']:+.3f}",
              flush=True)

    # -------------------- pooled headline + clustered SE ------------------
    res = {'sea': dict(Hs=HS, Tp=TP, depth=DEPTH, spread=SPREAD, h=H_SENS,
                       ranges=[R_MIN, R_MAX], ncomp=NCOMP, wdir=WDIR),
           'hos': dict(Nx=NX, Ny=NY, Lx=LX, M=3, spinup=SPINUP,
                       instants=list(INSTANTS)),
           'n_sea': a.seas, 'note': 'headline = pooled ensemble; +-SE clustered '
           'by seed', 'fields': {}}
    for f in FIELDS:
        pooled = np.concatenate([seed_slopes[f][s] for s in range(a.seas)])
        sigeta = float(np.mean([seed_sig[f][s] for s in range(a.seas)]))
        mo, bud, mi = field_stats(pooled, sigeta)
        # per-seed for SE
        ps = {'sd': [], 'skew': [], 'exkurt': [], 'ratio': [], 'mi': [],
              'bits': {str(r): [] for r in RTAB}}
        for s in range(a.seas):
            arr = seed_slopes[f][s]
            m2 = moments(arr); b2 = shadow_budget(arr, seed_sig[f][s])
            n2, w2 = empirical_density(arr)
            ps['sd'].append(m2['sd']); ps['skew'].append(m2['skew'])
            ps['exkurt'].append(m2['exkurt']); ps['ratio'].append(b2['ratio'])
            ps['mi'].append(mask_mi(n2, w2, rate=1e-3))
            for r in RTAB:
                ps['bits'][str(r)].append(b2['bits'][str(r)])
        res['fields'][f] = dict(
            sigma_s=mo['sd'], sigma_s_se=clustered(ps['sd'])[1],
            skew=mo['skew'], skew_se=clustered(ps['skew'])[1],
            exkurt=mo['exkurt'], exkurt_se=clustered(ps['exkurt'])[1],
            budget_ratio=bud['ratio'], budget_ratio_se=clustered(ps['ratio'])[1],
            budget_nats=bud['total_nats'], sigma_eta=sigeta,
            amp_nats=bud['amp_nats'], ceiling_dnrmse=bud['ceiling_dnrmse'],
            mi_ratio=mi, mi_ratio_se=clustered(ps['mi'])[1],
            bits=bud['bits'],
            bits_se={r: clustered(ps['bits'][r])[1] for r in ps['bits']})
    F = res['fields']
    res['sigma_s_ratio_hos'] = F['hos_m3']['sigma_s'] / F['hos_m1']['sigma_s']
    res['sigma_s_ratio_stokes2'] = F['stokes2']['sigma_s'] / F['lin_jonswap']['sigma_s']
    res['budget_inflation_hos'] = F['hos_m3']['budget_ratio'] / F['hos_m1']['budget_ratio']
    res['budget_inflation_stokes2'] = (F['stokes2']['budget_ratio']
                                       / F['lin_jonswap']['budget_ratio'])
    res['mi_inflation_hos'] = F['hos_m3']['mi_ratio'] / F['hos_m1']['mi_ratio']
    res['mi_inflation_stokes2'] = F['stokes2']['mi_ratio'] / F['lin_jonswap']['mi_ratio']

    # -------------------- V4 hit rate -------------------------------------
    flat = hit_rate_flat()
    hr_m3, hr_lin = [], []
    for seed in range(a.v4_seeds):
        S = sea_components(seed)
        sim3 = hos_run(seed, 3)
        while sim3.t < SPINUP - 1e-9:
            sim3.step(min(DT, SPINUP - sim3.t))
        for tgt in INSTANTS[:2]:
            while sim3.t < tgt - 1e-9:
                sim3.step(min(DT, tgt - sim3.t))
            kx, ky, av = hos_modes(sim3)
            hr_m3.append(hit_rate_field(
                lambda X, Y: hos_eval(kx, ky, av, X + CX, Y + CY, grad=False)))
            hr_lin.append(hit_rate_field(
                lambda X, Y, S=S, tgt=tgt: eta_linear(S, X + CX, Y + CY, tgt)))
    res['v4'] = {'flat': flat, 'hos_m3': float(np.mean(hr_m3)),
                 'lin_jonswap': float(np.mean(hr_lin)),
                 'hos_m3_se': clustered(hr_m3)[1]}

    # anchors: reproduce redteam 2.46% (Gaussian sigma_s=0.0982) + Gaussian@our sea
    an, aw = gauss_density(0.0982)
    res['anchor_redteam_mi_sig0982'] = mask_mi(an, aw, rate=1e-3)
    gn, gw = gauss_density(F['hos_m1']['sigma_s'])
    res['linear_gauss_mi_hosm1sig'] = mask_mi(gn, gw, rate=1e-3)

    # -------------------- GATES -------------------------------------------
    hb, hmi = F['hos_m3']['budget_ratio'], F['hos_m3']['mi_ratio']
    # stokes2 (all harmonics) applied to the frozen 0.70% anchor:
    stk_budget_anchored = 0.0070 * res['budget_inflation_stokes2']
    stk_mi_anchored = 0.0246 * res['mi_inflation_stokes2']
    gate_a = hb <= 0.014
    gate_b = hmi <= 0.05
    gate_a_stk = stk_budget_anchored <= 0.014
    gate_b_stk = stk_mi_anchored <= 0.05
    fires = not (gate_a and gate_b)
    res['gates'] = dict(gate_a_budget=bool(gate_a), gate_b_mi=bool(gate_b),
                        gate_a_stokes2_anchored=bool(gate_a_stk),
                        gate_b_mi_stokes2_anchored=bool(gate_b_stk),
                        stk_budget_anchored=stk_budget_anchored,
                        stk_mi_anchored=stk_mi_anchored,
                        F1_FIRES=bool(fires))

    os.makedirs('results/direction3', exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(res, f, indent=2)

    # -------------------- tables ------------------------------------------
    print("\n" + "=" * 82)
    print(f"V1  ALONG-RAY SLOPE  s=d eta/d(range)  (pooled headline, SE clustered "
          f"n={a.seas})")
    print(f"{'field':<14}{'sigma_s':>16}{'skew':>15}{'exkurt':>15}")
    for f in FIELDS:
        d = F[f]
        print(f"{f:<14}{d['sigma_s']:>9.4f}+-{d['sigma_s_se']:.4f}"
              f"{d['skew']:>+10.3f}+-{d['skew_se']:.3f}"
              f"{d['exkurt']:>+10.3f}+-{d['exkurt_se']:.3f}")
    print(f"  sigma_s ratio  hos_m3/hos_m1={res['sigma_s_ratio_hos']:.4f}   "
          f"stokes2/lin_jonswap={res['sigma_s_ratio_stokes2']:.4f}")

    print("\nV2  SHADOW BIT  -ln P(s<g) [nats/ray] (empirical CDF)   + weighted "
          "budget & ceiling")
    hdr = ''.join(f"{'r='+str(r):>9}" for r in RTAB)
    print(f"{'field':<14}{hdr}{'budget%':>10}{'ceilDnR':>9}")
    for f in FIELDS:
        d = F[f]
        row = ''.join(f"{d['bits'][str(r)]:>9.4f}" for r in RTAB)
        print(f"{f:<14}{row}{100*d['budget_ratio']:>8.2f}%{d['ceiling_dnrmse']:>+9.4f}")
    print("  frozen linear anchor: bits{25:.000 50:.0005 100:.052 150:.148 "
          "200:.231} budget 0.70% ceil +0.002")
    print(f"  budget inflation  hos_m3/hos_m1={res['budget_inflation_hos']:.3f}   "
          f"stokes2/lin={res['budget_inflation_stokes2']:.3f} "
          f"(-> anchored {100*stk_budget_anchored:.2f}%)")

    print("\nV3  MASK MI RATIO  I(m;s)/amp @ rate 1e-3")
    for f in FIELDS:
        print(f"  {f:<14}{100*F[f]['mi_ratio']:>7.2f}% +- {100*F[f]['mi_ratio_se']:.2f}")
    print(f"  anchor redteam sigma_s=0.0982: {100*res['anchor_redteam_mi_sig0982']:.2f}%"
          f" (target 2.46%)   Gaussian@hos_m1 sigma_s: "
          f"{100*res['linear_gauss_mi_hosm1sig']:.2f}%")
    print(f"  MI inflation  hos_m3/hos_m1={res['mi_inflation_hos']:.3f}  "
          f"stokes2/lin={res['mi_inflation_stokes2']:.3f} "
          f"(-> anchored {100*stk_mi_anchored:.2f}%)")

    print(f"\nV4  HIT RATE   flat {flat:.3f}   hos_m3 {res['v4']['hos_m3']:.3f}"
          f"+-{res['v4']['hos_m3_se']:.3f}   lin {res['v4']['lin_jonswap']:.3f}"
          f"   (occlusion != missingness)")

    print("\n" + "=" * 82)
    print("GATES (frozen):")
    print(f"  (a) HOS budget ratio <= 1.4% : {100*hb:.2f}%  -> "
          f"{'PASS' if gate_a else 'FAIL'}")
    print(f"  (b) HOS mask MI     <= 5.0%  : {100*hmi:.2f}%  -> "
          f"{'PASS' if gate_b else 'FAIL'}")
    print(f"  conservative (stokes2 all-harmonics, on frozen anchor): "
          f"budget {100*stk_budget_anchored:.2f}% -> {'PASS' if gate_a_stk else 'FAIL'}"
          f"   MI {100*stk_mi_anchored:.2f}% -> {'PASS' if gate_b_stk else 'FAIL'}")
    print(f"  positive control hos_m3_steep skew {F['hos_m3_steep']['skew']:+.3f} "
          f"(vs hos_m3 {F['hos_m3']['skew']:+.3f}) -- instrument is not blind")
    print("\n>>> F1 " + ("FIRES: 'closed' may NOT be written -- HOS reopens the "
          "prize." if fires else "DOES NOT FIRE: the closure survives HOS."))
    print(f">>> wrote {OUT}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
