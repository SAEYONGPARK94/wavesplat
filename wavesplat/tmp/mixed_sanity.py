"""Did the mixed retrain actually learn variable cardinality? Draw unconditional samples
from mixed vs _long prior and compare the participation-ratio (effective #components)
distribution. Mixed should now generate sparse spectra; _long should stay ~dense.
Run: python3 tmp/mixed_sanity.py
"""
import os, sys, numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, 'scripts')
import d3_setdiff_posterior as DP
from d3_setdiff_prior import SetDenoiser, denorm_atoms, sample

def pr_of(ckpt, n=400):
    ck = torch.load(ckpt, map_location=DP.DEVICE); stats = ck['stats']
    m = SetDenoiser(**ck.get('arch', {})).to(DP.DEVICE); m.load_state_dict(ck['model']); m.eval()
    abar = ck['abar'].to(DP.DEVICE); betas = ck['betas'].to(DP.DEVICE)
    prs = []
    with torch.no_grad():
        for _ in range(0, n, 50):
            Z = sample(m, abar, betas, n=50)
            _, _, a = denorm_atoms(Z, stats); a = torch.clamp(a, min=0).cpu().numpy()
            prs.append((a.sum(1) ** 2) / ((a ** 2).sum(1) + 1e-12))
    return np.concatenate(prs)

for name, ck in [('_long (old)', 'results/direction3/setdiff_prior_long.pt'),
                 ('mixed (new)', 'results/direction3/setdiff_prior_mixed.pt')]:
    if not os.path.exists(ck):
        print(f"{name}: {ck} missing"); continue
    pr = pr_of(ck)
    print(f"{name:12s} generated PR: mean {pr.mean():.1f}  min {pr.min():.1f}  "
          f"p5 {np.percentile(pr,5):.1f}  frac<6 {(pr<6).mean():.2f}")
