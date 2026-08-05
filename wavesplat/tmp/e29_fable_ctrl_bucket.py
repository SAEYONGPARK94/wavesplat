"""e29_fable_ctrl_bucket.py — debate check: is B's EX_Hs loss an Hs-OOD
failure or a mid-aperture regression mislabeled as OOD?

Scale-invariance argument: with bw=0, Hs only scales signal AND noise
(sigma = nfrac*Hs), inputs are max-normalized, corr is scale-free, and the
MMSE solve is scale-equivariant (gam ~ var(z), s2 ~ (nfrac*Hs)^2). So for a
scale-invariant estimator the corr distribution of (Hs 4-5, side 1.2-1.8,
bw 0) is IDENTICAL to (Hs 1-3, side 1.2-1.8, bw 0). The only Hs-sensitive
path is FiLM's log_std(z) feature.

CTRL bucket = hs (1,3), side (1.2,1.8), bw 0  <- matched to EX_Hs except Hs.
  If d(film-cur) on CTRL ~ -0.03 -> EX_Hs is an aperture-band regression,
  NOT sea-state extrapolation.  If ~0 -> Hs really matters -> the FiLM path
  (informativeness conditioning) is the culprit, falsifying "~0 contribution".
Also re-run EX_Hs with a different rng to check stability.
"""
import sys, os, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import e28_gennet_B as B   # chdirs to repo root, provides everything

DEVICE = B.DEVICE

ck = torch.load(B.CKPT, map_location=DEVICE, weights_only=False)
film_net = B.FiLMPriorNet().to(DEVICE)
film_net.backbone.load_state_dict(ck['model'])
film_net.film.load_state_dict(ck['film'])
film_net.film.set_norm(ck['feat_mu'], ck['feat_sd'])
film_net.eval()
film_ls = torch.tensor(float(ck['log_smult']), device=DEVICE)

cur = torch.load(B.CUR_NET, map_location=DEVICE, weights_only=False)
cur_net = B.PriorNet().to(DEVICE)
cur_net.load_state_dict(cur['model'])
cur_net.eval()
cur_ls = torch.tensor(float(cur.get('log_smult', np.log(3.0))), device=DEVICE)

BUCKETS = {
    'CTRL_hs13_mid': dict(hs=(1.0, 3.0), side=(1.2, 1.8), bw=0.0, seed=555001),
    'EXHs_rerun':    dict(hs=(4.0, 5.0), side=(1.2, 1.8), bw=0.0, seed=555002),
}
N_PER = 24
out = {}
for name, reg in BUCKETS.items():
    rng = np.random.default_rng(reg['seed'])
    samples, Vf = [], []
    for _ in range(N_PER):
        s = B.gen_widened(rng, hs_range=reg['hs'], side_range=reg['side'],
                          bw_max=0.0)
        samples.append(s)
        Vf.append(B.inform_features(s['P'], s['z'], s['kp_e'], s['nfrac'],
                                    s['Hs'], s['X']))
    r = B._predict_all(samples, film_net, film_ls, cur_net, cur_ls, Vf)
    d = r['film'] - r['cur']
    m, lo, hi, win, n = B._paired_ci(d)
    # also: film with CONST informativeness (feature vector = train mean)
    vmean = ck['feat_mu']
    rc = B._predict_all(samples, film_net, film_ls, cur_net, cur_ls,
                        [np.asarray(vmean, np.float32)] * N_PER)
    dconst = rc['film'] - rc['cur']
    out[name] = dict(film=float(r['film'].mean()), cur=float(r['cur'].mean()),
                     hand=float(r['hand'].mean()), d=m, ci90=[lo, hi],
                     win=win, n=n,
                     d_constinform=float(dconst.mean()),
                     hs_mean=float(np.mean([s['Hs'] for s in samples])))
    print(f"{name:14s} film {r['film'].mean():+.3f} cur {r['cur'].mean():+.3f} "
          f"hand {r['hand'].mean():+.3f} | d {m:+.4f} CI90[{lo:+.3f},{hi:+.3f}] "
          f"win {win}/{n} | d(const-inform) {dconst.mean():+.4f}",
          flush=True)

json.dump(out, open('results/direction3/e29_fable_ctrl_bucket.json', 'w'),
          indent=1, default=float)
print('saved -> results/direction3/e29_fable_ctrl_bucket.json')
