"""
Direction III — over-smoothing diagnostic for the persona review.

The sharpest critique of the 30-ep result: grid-free DPS BMA "wins" on complex
seas at nRMSE ~0.22, but a zero-predictor already gets ~0.25 on a field with
std≈Hs/4. Is the BMA a real reconstruction or just an ensemble average smoothed
toward zero? We compute the reviewer-defense panel (variance ratio, Pearson,
skill) for the BMA on simple + complex seas, vs the zero-predictor baseline.

Run: python3 scripts/d7_diag_smoothing.py
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.evaluation.metrics import variance_ratio, pearson_correlation, skill_score
import d3_setdiff_posterior as DP
from d3_setdiff_prior import SetDenoiser
from d6_integrated_eval import gather_W, W, TI

CKPT = 'results/direction3/setdiff_prior_v2.pt'
NSAMP = 10


def main():
    ck = torch.load(CKPT, map_location=DP.DEVICE)
    stats = ck['stats']
    model = SetDenoiser(**ck.get('arch', {})).to(DP.DEVICE)
    model.load_state_dict(ck['model']); model.eval()
    abar = ck['abar'].to(DP.DEVICE); betas = ck['betas'].to(DP.DEVICE)

    eps = ['episode_BF4_5comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_uniform.h5',
           'episode_BF5_30comp_seed1_uniform.h5']
    print(f"{'episode':34s} {'nRMSE':>7} {'VR':>6} {'Pear':>6} {'Skill':>7} {'persVR':>7} {'zeroNR':>7}")
    rows = []
    for ep in eps:
        L = EpisodeDataLoader('data/' + ep); Hs = L.Hs_true; depth = L.depth
        gx = L.gt_grid_x.numpy(); gy = L.gt_grid_y.numpy()
        frames = L.get_tracking_frames(2.0)
        P, z, _ = gather_W(L, TI, W)
        tgt = frames[TI + 1]
        target = {'gx': gx, 'gy': gy, 't': float(tgt.t), 'gt': tgt.gt_field.numpy(), 'Hs': Hs}
        gt = target['gt']
        Xd = DP.dps_sample(model, stats, abar, betas, P, z, depth, guidance=DP.GUID if hasattr(DP,'GUID') else 0.1, n=NSAMP, seed=0)
        fields = [DP.atoms_to_field(Xd[b], stats, P, z, depth, target)[0] for b in range(NSAMP)]
        fields = [f for f in fields if f is not None]
        F = np.stack(fields); bma = F.mean(0)
        nr = float(np.sqrt(np.mean((bma - gt)**2)) / Hs)
        vr = variance_ratio(bma, gt); pe = pearson_correlation(bma, gt); sk = skill_score(bma, gt)
        pers_vr = float(np.mean([variance_ratio(f, gt) for f in fields]))   # per-sample VR
        zero_nr = float(np.sqrt(np.mean(gt**2)) / Hs)                       # zero-predictor nRMSE
        print(f"{ep:34s} {nr:7.3f} {vr:6.2f} {pe:6.2f} {sk:7.2f} {pers_vr:7.2f} {zero_nr:7.3f}")
        rows.append(dict(ep=ep, nrmse=nr, VR=vr, Pearson=pe, Skill=sk,
                         persample_VR=pers_vr, zero_nrmse=zero_nr))
    json.dump(rows, open('results/direction3/diag_smoothing.json', 'w'), indent=2)
    print("\nVR≈1 good; VR<<1 = over-smoothed (predicting near-zero). "
          "Pearson = real shape match. Skill>0 = beats zero-predictor.")


if __name__ == '__main__':
    main()
