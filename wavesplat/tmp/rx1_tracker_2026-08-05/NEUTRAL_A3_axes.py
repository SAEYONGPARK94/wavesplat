"""A3: is 'shrinkage' the right word? + secondary-outcome redundancy.

Identity used: normalized MSE (relative to Var(gt)) = 1 - 2*rho*sqrt(v) + v,
minimised at v* = rho^2 with floor 1 - rho^2.
Skill_Score = 1 - MSE/Var(gt), so nMSE = 1 - SS is directly available.
"""
import json, os, itertools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 's1s2_2026-08-05', 'results.jsonl')
rows = [json.loads(l) for l in open(SRC) if l.strip() and 'error' not in json.loads(l)]
EPS = sorted({r['episode'] for r in rows}); SEEDS = sorted({r['seed'] for r in rows})
ARMS = sorted({r['arm'] for r in rows})

def cell(arm, ep, seed):
    v = [r for r in rows if r['arm'] == arm and r['episode'] == ep and r['seed'] == seed]
    return v[0]

print('=' * 84)
print('A3.1  IDENTITY CHECK: is (1 - SS) == 1 - 2*rho*sqrt(VR) + VR on frame-averaged values?')
print('=' * 84)
err = []
for arm in ARMS:
    for ep in EPS:
        for s in SEEDS:
            c = cell(arm, ep, s)
            v, rho, ss = c['Variance_Ratio'], c['Pearson_rho'], c['Skill_Score']
            lhs = 1 - ss
            rhs = 1 - 2 * rho * np.sqrt(v) + v
            err.append(lhs - rhs)
err = np.array(err)
print(f'  n={len(err)}  mean resid {err.mean():+.4f}  median {np.median(err):+.4f}  '
      f'max|resid| {np.abs(err).max():.4f}')
print('  (non-zero because all three are frame-AVERAGES of a per-frame identity;')
print('   sign/size tells you how much Jensen slack the frame-averaging introduces)')

print()
print('=' * 84)
print('A3.2  SHRINKAGE vs DECORRELATION SPLIT  (per arm, mean over 12 cells)')
print('=' * 84)
print(f'{"arm":<16s} {"rho":>7s} {"VR":>7s} {"nMSE=1-SS":>10s} {"decorr floor":>13s} '
      f'{"shrink part":>12s} {"shrink %":>9s} {"v*=rho^2":>9s} {"VR/v*":>7s}')
summary = {}
for arm in ARMS:
    rr = [cell(arm, ep, s) for ep in EPS for s in SEEDS]
    rho = np.mean([c['Pearson_rho'] for c in rr]); v = np.mean([c['Variance_Ratio'] for c in rr])
    nmse = np.mean([1 - c['Skill_Score'] for c in rr])
    # per-cell split then average (avoids averaging-then-splitting bias)
    floors, shr, tot = [], [], []
    for c in rr:
        rh, vv = c['Pearson_rho'], c['Variance_Ratio']
        floors.append(1 - rh ** 2); shr.append((np.sqrt(vv) - rh) ** 2)
        tot.append(1 - 2 * rh * np.sqrt(vv) + vv)
    f, sh, t = np.mean(floors), np.mean(shr), np.mean(tot)
    summary[arm] = (rho, v, nmse, f, sh, t)
    print(f'{arm:<16s} {rho:7.4f} {v:7.4f} {nmse:10.4f} {f:13.4f} {sh:12.4f} '
          f'{100*sh/t:8.1f}% {rho**2:9.4f} {v/rho**2:7.3f}')
print()
print('  READ: "decorr floor" = 1-rho^2 = the nMSE that NO amplitude rescaling can remove.')
print('        "shrink part"  = (sqrt(VR)-rho)^2 = the part a single scalar rescale WOULD remove.')

print()
print('=' * 84)
print('A3.3  PAIRED ON->OFF: which axis actually moved? (oracle placement, med3 per episode)')
print('=' * 84)
def med3(arm, ep, key, f=lambda c, k: c[k]):
    return float(np.median([f(cell(arm, ep, s), key) for s in SEEDS]))
for ep in EPS:
    a = [cell('S2-T1f', ep, s) for s in SEEDS]; b = [cell('X-T1f-notrack', ep, s) for s in SEEDS]
    ra, va = np.median([c['Pearson_rho'] for c in a]), np.median([c['Variance_Ratio'] for c in a])
    rb, vb = np.median([c['Pearson_rho'] for c in b]), np.median([c['Variance_Ratio'] for c in b])
    print(f'  {ep[8:-3]:<24s} rho {ra:.4f}->{rb:.4f} (d{rb-ra:+.4f})  '
          f'VR {va:.4f}->{vb:.4f} (d{vb-va:+.4f})  '
          f'floor {1-ra**2:.4f}->{1-rb**2:.4f}  shrink {(np.sqrt(va)-ra)**2:.4f}->{(np.sqrt(vb)-rb)**2:.4f}')

print()
print('=' * 84)
print('A3.4  IS Hs_error A DUPLICATE OF |VR-1| ?   (tautology: Hs_err = |4*std(pred)/Hs-1|,')
print('        VR = var(pred)/var(gt)  =>  if 4*std(gt_frame) == Hs_true then Hs_err=|sqrt(VR)-1|)')
print('=' * 84)
X, Y, Z, A_, B_, C_ = [], [], [], [], [], []
for arm in ARMS:
    for ep in EPS:
        for s in SEEDS:
            c = cell(arm, ep, s)
            X.append(abs(c['Variance_Ratio'] - 1)); Y.append(c['Hs_error'])
            Z.append(abs(np.sqrt(c['Variance_Ratio']) - 1))
            A_.append(c['nRMSE']); B_.append(c['Pearson_rho']); C_.append(c['Skill_Score'])
X, Y, Z = np.array(X), np.array(Y), np.array(Z)
print(f'  n={len(X)} cells (all 11 arms x 4 ep x 3 seed)')
print(f'  Pearson r(|VR-1|, Hs_error)          = {np.corrcoef(X, Y)[0,1]:+.4f}')
print(f'  Pearson r(|sqrt(VR)-1|, Hs_error)    = {np.corrcoef(Z, Y)[0,1]:+.4f}   <- the tautological form')
from scipy.stats import spearmanr
print(f'  Spearman rho(|VR-1|, Hs_error)       = {spearmanr(X, Y).statistic:+.4f}')
print(f'  Spearman rho(|sqrt(VR)-1|, Hs_error) = {spearmanr(Z, Y).statistic:+.4f}')
print(f'  mean |Hs_error - |sqrt(VR)-1||       = {np.mean(np.abs(Y-Z)):.4f}  '
      f'(median {np.median(np.abs(Y-Z)):.4f}, max {np.abs(Y-Z).max():.4f})')
print(f'  regression Hs_error ~ |sqrt(VR)-1| : slope {np.polyfit(Z,Y,1)[0]:.4f}  '
      f'intercept {np.polyfit(Z,Y,1)[1]:+.4f}  R^2 {np.corrcoef(Z,Y)[0,1]**2:.4f}')

print()
print('  Full pairwise |Pearson r| among the 5 PRE-SPECIFIED SECONDARY outcomes (n=132 cells):')
names = ['rho', '|VR-1|', 'Hs_error', 'amp_rel_error', 'skill']
amp = []
for arm in ARMS:
    for ep in EPS:
        for s in SEEDS:
            amp.append(cell(arm, ep, s)['spec'].get('amp_rel_error_mean', np.nan))
M = np.vstack([np.array(B_), X, Y, np.array(amp), np.array(C_)])
ok = np.all(np.isfinite(M), axis=0)
M = M[:, ok]
print(f'  (finite cells: {ok.sum()})')
R = np.corrcoef(M)
print('        ' + ''.join(f'{n:>16s}' for n in names))
for i, n in enumerate(names):
    print(f'  {n:<14s}' + ''.join(f'{R[i,j]:+16.4f}' for j in range(len(names))))
w, _ = np.linalg.eigh(np.corrcoef(M))
w = np.sort(w)[::-1]
print(f'\n  eigenvalues of the 5x5 correlation matrix: {[round(x,3) for x in w]}')
print(f'  effective dimensionality (participation ratio (sum w)^2/sum w^2) = '
      f'{w.sum()**2/np.sum(w**2):.2f} of 5')
print(f'  components needed for 90% of variance: '
      f'{int(np.searchsorted(np.cumsum(w)/w.sum(), 0.90))+1}')
