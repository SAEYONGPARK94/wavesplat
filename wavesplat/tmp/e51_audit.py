"""Independent audit of e51 CRB. Fresh re-derivation, no trust in e51 outputs."""
import os, sys, json
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')

from e29_gennet_C import DEVICE, _MASK, DEPTH
from e37_bimodal_ladder import build_sea_crosstp
from e42_detector_v2 import JointEngine, GAMMA, SPREAD
from e41_regime_prize_ladder import lobe_grid, tp_of_k
from d3_sa_achievability_gap import dispersion_lam
from e51_crb import build_C, fim_crb, EPS, PNAMES, _atoms

torch.set_grad_enabled(False)

def k_of_Tp(Tp):
    return 2*np.pi/dispersion_lam(Tp, DEPTH)

# ================= CHECK 1: build_C vs JointEngine K primitive =================
print("="*70)
print("CHECK 1: build_C reproduces JointEngine covariance K")
print("="*70)
for si in [700, 704, 716]:
    S = build_sea_crosstp(si)
    theta = [S['d1'], 7.0, S['d2'], S['Tp2'], S['r2']]
    d1, Tp1, d2, Tp2, w = theta
    eng = JointEngine(S)
    # --- e51 build_C (float64) ---
    K_bc = build_C(eng, theta).cpu().numpy()

    # --- Reconstruct via JointEngine's EXACT primitive ops (float32 path) ---
    # g_i exactly as JointEngine.build_lobes builds them (Tp_eff, mu, GAMMA, SPREAD)
    g1 = lobe_grid(eng.kp, [(Tp1, GAMMA, 1.0)], [(d1, SPREAD, 1.0)])[_MASK]
    g2 = lobe_grid(eng.kp, [(Tp2, GAMMA, 1.0)], [(d2, SPREAD, 1.0)])[_MASK]
    Phi32 = eng.Phi  # float32
    def M_of(g):
        gam = torch.tensor(np.concatenate([g, g]), dtype=torch.float32, device=DEVICE)
        return torch.einsum('nk,k,mk->nm', Phi32, gam, Phi32)
    M1 = M_of(g1); M2 = M_of(g2)
    Mtot = (1-w)*M1 + w*M2
    stot = (1-w)*float(g1.sum()) + w*float(g2.sum())
    scale = eng.vz / max(stot, 1e-30)
    K_je = (Mtot*scale + eng.s2*eng.eye).cpu().numpy()

    # --- Also run the REAL logev_pairs path with rounded (a,mu) keys ---
    a1 = round(float(k_of_Tp(Tp1)/eng.kp), 3); mu1 = round(float(d1), 5)
    a2 = round(float(k_of_Tp(Tp2)/eng.kp), 3); mu2 = round(float(d2), 5)
    eng.build_lobes([(a1, mu1), (a2, mu2)])
    i = eng.idx(a1, mu1); j = eng.idx(a2, mu2)
    Mi = eng.M[i]; Mj = eng.M[j]
    si_s = float(eng.sums[i]); sj_s = float(eng.sums[j])
    Mtot2 = (1-w)*Mi + w*Mj
    stot2 = (1-w)*si_s + w*sj_s
    K_real = (Mtot2*(eng.vz/max(stot2,1e-30)) + eng.s2*eng.eye).cpu().numpy()

    ld_bc = np.linalg.slogdet(K_bc)[1]
    ld_je = np.linalg.slogdet(K_je)[1]
    ld_real = np.linalg.slogdet(K_real)[1]
    relmax = np.max(np.abs(K_bc-K_je))/np.max(np.abs(K_bc))
    relmax_r = np.max(np.abs(K_bc-K_real))/np.max(np.abs(K_bc))
    print(f" si{si}: logdet build_C={ld_bc:.6f} JEprim={ld_je:.6f} "
          f"JEreal(rounded a,mu)={ld_real:.6f}")
    print(f"        max|dC|/max|C| primitive={relmax:.2e}  real(rounded)={relmax_r:.2e}")

# ================= CHECK 2: independent FIM/CRB + medians =================
print("\n"+"="*70)
print("CHECK 2: re-derive medians from stored rowsX and recompute a few FIM")
print("="*70)
J = json.load(open('results/direction3/e51_crb.json'))
rowsX = J['rowsX']
for key in ['crb_d1_deg','crb_d2_deg','crb_Tp2','crb_w']:
    v = np.array([r[key] for r in rowsX])
    print(f"  {key:12s} median={np.median(v):.4f}  (stored medX={J['medX'][key]:.4f})"
          f"  n={len(v)} nNaN={np.isnan(v).sum()}")
conds = np.array([r['cond'] for r in rowsX])
print(f"  cond median={np.median(conds):.1f}  max={conds.max():.1f} "
      f"  >1e3: {(conds>1e3).sum()}  >1e2: {(conds>1e2).sum()}")
wv = np.array([r['crb_w'] for r in rowsX])
print(f"  crb_w > prior_std(0.087): {(wv>0.087).sum()}/{len(wv)} seas; "
      f"median ratio {np.median(wv)/0.087:.2f}")

# recompute FIM from scratch for a few seas, compare stored sqrt(CRB)
print("  recompute FIM for si 700,704,716,730(high cond):")
for si in [700, 704, 716, 730]:
    S = build_sea_crosstp(si)
    theta = [S['d1'], 7.0, S['d2'], S['Tp2'], S['r2']]
    sc, cond, F = fim_crb(S, theta)
    stored = [r for r in rowsX if r['si']==si][0]
    print(f"   si{si}: recompute d2={np.rad2deg(sc[2]):.3f} w={sc[4]:.4f} cond={cond:.1f}"
          f" | stored d2={stored['crb_d2_deg']:.3f} w={stored['crb_w']:.4f} cond={stored['cond']:.1f}"
          f" | F symm err={np.max(np.abs(F-F.T)):.1e} PD={np.all(np.linalg.eigvalsh(F)>0)}")

# ================= CHECK 3: EPS convergence =================
print("\n"+"="*70)
print("CHECK 3: finite-difference EPS convergence (0.5x, 1x, 2x)")
print("="*70)
import e51_crb as E
sub = list(range(700, 712))
def median_crb(scale):
    E.EPS = {k: v*scale for k, v in EPS.items()}
    d2s, ws, d1s = [], [], []
    for si in sub:
        S = build_sea_crosstp(si)
        theta = [S['d1'], 7.0, S['d2'], S['Tp2'], S['r2']]
        sc, cond, _ = E.fim_crb(S, theta)
        d1s.append(np.rad2deg(sc[0])); d2s.append(np.rad2deg(sc[2])); ws.append(sc[4])
    return np.median(d1s), np.median(d2s), np.median(ws)
for s in [0.5, 1.0, 2.0]:
    d1m, d2m, wm = median_crb(s)
    print(f"  EPS x{s}: median sqrt(CRB) d1={d1m:.3f}deg d2={d2m:.3f}deg w={wm:.4f}")
E.EPS = dict(EPS)

# ================= CHECK 4: GUARD1 partial correlation =================
print("\n"+"="*70)
print("CHECK 4: GUARD1 aperture vs sqrt(CRB(d2)) — Spearman + PARTIAL")
print("="*70)
from scipy import stats
apert = np.array([r['aperture'] for r in rowsX])
d2 = np.array([r['crb_d2_deg'] for r in rowsX])
sep = np.array([r['sep_deg'] for r in rowsX])
r2 = np.array([r['r2'] for r in rowsX])
Tp2 = np.array([r['Tp2'] for r in rowsX])
cond = np.array([r['cond'] for r in rowsX])
rho, p = stats.spearmanr(apert, d2)
print(f"  raw Spearman aperture vs crb_d2: {rho:+.3f} (p={p:.3f})")

def rankz(x): return stats.rankdata(x)
def partial_spearman(x, y, Z):
    # rank-based partial correlation controlling for columns of Z
    xr = rankz(x); yr = rankz(y); Zr = np.column_stack([rankz(z) for z in Z.T])
    Zr = np.column_stack([np.ones(len(xr)), Zr])
    bx = np.linalg.lstsq(Zr, xr, rcond=None)[0]; rx = xr - Zr@bx
    by = np.linalg.lstsq(Zr, yr, rcond=None)[0]; ry = yr - Zr@by
    r = np.corrcoef(rx, ry)[0,1]
    nn = len(xr); k = Zr.shape[1]-1
    t = r*np.sqrt((nn-2-k)/max(1e-12,(1-r**2)))
    pv = 2*stats.t.sf(abs(t), nn-2-k)
    return r, pv
for label, Z in [('control sep', np.column_stack([sep])),
                 ('control sep,r2,Tp2', np.column_stack([sep, r2, Tp2])),
                 ('control sep,r2,Tp2,cond', np.column_stack([sep, r2, Tp2, cond]))]:
    pr, pv = partial_spearman(apert, d2, Z)
    print(f"  partial Spearman ({label}): {pr:+.3f} (p={pv:.3f})")
# what dominates crb_d2? correlations with candidate drivers
print("  drivers of crb_d2_deg (raw Spearman):")
for nm, arr in [('sep_deg', sep), ('r2', r2), ('Tp2', Tp2), ('aperture', apert),
                ('cond', cond)]:
    rr, pp = stats.spearmanr(arr, d2)
    print(f"     {nm:9s}: {rr:+.3f} (p={pp:.3f})")
