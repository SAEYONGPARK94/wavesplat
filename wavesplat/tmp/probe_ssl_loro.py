"""Decisive test: does the SSL encoder predict the kurtosis of a sea it never saw?

The confound probe showed record ID is 79% linearly decodable (chance 8.3%).
So the rec_kurt R^2 = 0.70 can be produced by fingerprint-then-look-up: the
probe's training set contains patches from the very record it is scored on.

Leave-one-record-out removes the lookup. The probe is fit on 11 records and
asked for the 12th record's kurtosis, which it has never been told. A
fingerprint of an unseen record indexes nothing.

Two tiers:
  LORO        11 SSL-training records -> the held-out 12th.
  YS_2017     never seen by the SSL encoder either. The strict tier.

Scored across records (n=12), because a sea state is one sample, not 6144.
Random-init control throughout.
"""
import os
import sys

import numpy as np
import torch

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'scripts')

from d3_ssl_pretrain import MAE, Patches, excess_kurtosis, HELD_OUT, DEV

CKPT = 'results/direction3/ssl_mae.pt'
N_BATCH, BS = 96, 64
RIDGE = 1e-2


def collect(model, ds, seed=7):
    rng = np.random.default_rng(seed)
    F, K, R = [], [], []
    model.eval()
    with torch.no_grad():
        for _ in range(N_BATCH):
            xb, ridx = ds.batch(rng, BS, want_rec=True)
            xb = xb.to(DEV)
            inp = torch.stack([xb, torch.zeros_like(xb)], 1)
            F.append(model.features(inp).cpu())
            K.append(excess_kurtosis(xb).cpu())
            R.append(torch.tensor(ridx))
    return (torch.cat(F).double().numpy(), torch.cat(K).double().numpy(),
            torch.cat(R).numpy())


def fit(F, y):
    A = np.concatenate([F, np.ones((len(F), 1))], 1)
    G = A.T @ A + RIDGE * np.eye(A.shape[1])
    return np.linalg.solve(G, A.T @ y)


def apply(W, F):
    return np.concatenate([F, np.ones((len(F), 1))], 1) @ W


def loro(F, K, R):
    """Per-record: fit on the other 11, predict this record's mean kurtosis."""
    recs = np.unique(R)
    true, pred = [], []
    for r in recs:
        tr, te = R != r, R == r
        rk = {x: K[R == x].mean() for x in recs if x != r}
        W = fit(F[tr], np.array([rk[x] for x in R[tr]]))
        true.append(K[te].mean())
        pred.append(apply(W, F[te]).mean())     # pooled over that record
    return np.array(true), np.array(pred)


def r2(t, p):
    return float(1 - ((t - p) ** 2).sum() / max(((t - t.mean()) ** 2).sum(), 1e-12))


def strict(F, K, R, Fh, Kh):
    """Fit on all 12 training records; predict the never-seen YS_2017."""
    rk = {x: K[R == x].mean() for x in np.unique(R)}
    W = fit(F, np.array([rk[x] for x in R]))
    return float(Kh.mean()), float(apply(W, Fh).mean())


def main():
    ds = Patches(HELD_OUT, exclude=True)
    dh = Patches(HELD_OUT, exclude=False)      # YS_2017 only
    ck = torch.load(CKPT, map_location=DEV)
    print(f"checkpoint step {ck['step']}, held-out masked MSE {ck['val']:.4f}\n")

    models = {}
    models['SSL-trained'] = MAE(dim=ck['dim']).to(DEV)
    models['SSL-trained'].load_state_dict(ck['model'])
    models['random-init'] = MAE(dim=ck['dim']).to(DEV)

    for tag in ['random-init', 'SSL-trained']:
        m = models[tag]
        F, K, R = collect(m, ds)
        Fh, Kh, _ = collect(m, dh, seed=11)
        t, p = loro(F, K, R)
        rho = float(np.corrcoef(t, p)[0, 1])
        kt, kp = strict(F, K, R, Fh, Kh)
        print(f"{tag}")
        print(f"  LORO across {len(t)} records:  R2 {r2(t, p):+.3f}   "
              f"corr {rho:+.3f}   spread(true) {t.std():.3f}")
        print(f"  strict {HELD_OUT}:  true kurt {kt:+.3f}   "
              f"pred {kp:+.3f}   err {abs(kt - kp):.3f}")
    print(f"\nA fingerprint cannot index an unseen record. If LORO R2 collapses "
          f"toward 0\nwhile the in-sample rec_kurt R2 stayed 0.70, the headline "
          f"was lookup.")


if __name__ == '__main__':
    sys.exit(main())
