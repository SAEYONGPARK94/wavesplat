"""PRE-REGISTERED evaluation for the fan-geometry retrain (frozen BEFORE training completes).

PRIMARY: fresh-seed (rng 120000) pure-fan (fan_frac=1.0) gen_sample scenes, n=30.
  Metric = gt_q nRMSE (fan-mode task queries = the encounter-forecast region).
  Test = paired one-sided Wilcoxon (alpha 0.05): fanTF-net better than analytic hand.
SECONDARY (reported, not decision-primary): fanTF vs old footprint-trained 2ndlobeTF;
  by-nc split; real-record re-test is run separately via run_wp_realdata --ckpt fanTF.
NO other tests count as primary. (Pre-registered 2026-07-24 night, per memory protocol.)
"""
import numpy as np
from scipy import stats as ss
import d3_prior_net_data as G
from d3_sa_nomp import mmse_bank_estimate
from d3_prior_net_eval import get_net, net_gamma_mmse

net_fan = get_net('results/direction3/prior_net_fanTF.pt')
net_old = get_net('results/direction3/prior_net_2ndlobeTF.pt')
rng = np.random.default_rng(120000)


def nr(f, gt): return float(np.sqrt(np.mean((f - gt) ** 2)) / max(np.std(gt), 1e-9))


H, FN, OLD, NCs = [], [], [], []
for r in range(30):
    s = G.gen_sample(rng, fan_frac=1.0, bimodal_frac=0.3, bw_inject=0.15,
                     nc=int(np.random.default_rng(500 + r).choice([5, 15, 30, 60, 200])))
    P, z, Q, gt = s['P'], s['z'], s['Q'], s['gt_q']
    kp = s['kp_e']; s0 = 0.05 * s['Hs']
    H.append(nr(mmse_bank_estimate(P, z, G.DEPTH, Q, sigma0=s0, kp_hint=kp, seed=r), gt))
    FN.append(nr(net_gamma_mmse(net_fan, P, z, G.DEPTH, Q, sigma0=s0, seed=r, kp_hint=kp), gt))
    OLD.append(nr(net_gamma_mmse(net_old, P, z, G.DEPTH, Q, sigma0=s0, seed=r, kp_hint=kp), gt))
    NCs.append(len(s.get('meta', [0]*5)))  # placeholder; nc recorded via print
    print(f"[{r:2d}] hand {H[-1]:.3f}  fanTF {FN[-1]:.3f}  oldTF {OLD[-1]:.3f}", flush=True)

H, FN, OLD = map(np.array, (H, FN, OLD))
d_primary = H - FN
w = ss.wilcoxon(d_primary, alternative='greater')
print("\n=== PRE-REGISTERED PRIMARY (fan geometry, n=30, one-sided) ===")
print(f"  hand {H.mean():.3f}  fanTF {FN.mean():.3f}  d {d_primary.mean():+.4f}  "
      f"wins {int((FN < H).sum())}/30  Wilcoxon(one-sided) p={w.pvalue:.4f}")
print(f"  ⟹ PRIMARY {'PASS' if w.pvalue < 0.05 and d_primary.mean() > 0 else 'FAIL'}")
d_sec = OLD - FN
w2 = ss.wilcoxon(d_sec) if np.any(d_sec != 0) else None
print("\n=== SECONDARY ===")
print(f"  old-footprint-TF {OLD.mean():.3f} vs fanTF {FN.mean():.3f}: d {d_sec.mean():+.4f} "
      f"(fan better {int((FN < OLD).sum())}/30, p={w2.pvalue if w2 else 1:.4f})")
print(f"  old vs hand: d {(H-OLD).mean():+.4f} ({int((OLD < H).sum())}/30)")
