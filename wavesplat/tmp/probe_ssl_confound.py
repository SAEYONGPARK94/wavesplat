"""Does the SSL encoder know non-Gaussianity, or does it just recognise the record?

W7's headline: a linear probe on frozen features predicts the RECORD-level
excess kurtosis with R^2 = 0.761, a gain of +0.345 over a random-init control.
That was offered as evidence that the representation understands how far from
Gaussian a sea is.

There is a cheaper explanation. Each record has its own grid spacing, stereo
noise texture and dead-pixel pattern. If the encoder can simply identify WHICH
record a patch came from, then predicting that record's kurtosis is a lookup,
not an understanding. The rec_kurt probe would be measuring record
fingerprinting.

Three probes on the same frozen features:

  1. record ID       12-way linear classifier. High accuracy => the confound
                     is available.
  2. rec_kurt        the original claim.
  3. within-record   patch kurtosis AFTER subtracting its record's mean. A
     kurtosis         fingerprint cannot help here: the record is held constant.
                     THIS is the signal CL1 actually needs, because CL1 must
                     tell a locally non-Gaussian patch from a Gaussian one
                     inside the same sea.

Random-init control for all three.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..'))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'scripts')

from d3_ssl_pretrain import MAE, Patches, excess_kurtosis, HELD_OUT, DEV

CKPT = 'results/direction3/ssl_mae.pt'
N_BATCH, BS = 96, 64


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


def lin_r2(F, y):
    n = int(0.7 * len(F))
    A = np.concatenate([F, np.ones((len(F), 1))], 1)
    W, *_ = np.linalg.lstsq(A[:n], y[:n], rcond=None)
    p = A[n:] @ W
    ss = ((y[n:] - p) ** 2).sum()
    st = ((y[n:] - y[n:].mean()) ** 2).sum()
    return float(1 - ss / max(st, 1e-12))


def lin_acc(F, r):
    """One-vs-rest least squares classifier — a linear probe for record ID."""
    n = int(0.7 * len(F))
    labs = np.unique(r)
    A = np.concatenate([F, np.ones((len(F), 1))], 1)
    Y = (r[:, None] == labs[None]).astype(float)
    W, *_ = np.linalg.lstsq(A[:n], Y[:n], rcond=None)
    pred = labs[np.argmax(A[n:] @ W, 1)]
    return float((pred == r[n:]).mean()), len(labs)


def report(tag, F, K, R):
    acc, nlab = lin_acc(F, R)
    rk = {x: K[R == x].mean() for x in np.unique(R)}
    Y = np.array([rk[x] for x in R])
    within = K - Y                       # record mean removed
    print(f"  {tag:<12} record-ID acc {acc:.3f} (chance {1/nlab:.3f})   "
          f"rec_kurt R2 {lin_r2(F, Y):+.3f}   "
          f"within-record kurt R2 {lin_r2(F, within):+.3f}")
    return {'id_acc': acc, 'chance': 1 / nlab, 'rec_kurt': lin_r2(F, Y),
            'within_kurt': lin_r2(F, within)}


def main():
    ds = Patches(HELD_OUT, exclude=True)     # the training records
    ck = torch.load(CKPT, map_location=DEV)
    print(f"checkpoint step {ck['step']}, held-out masked MSE {ck['val']:.4f}\n")

    trained = MAE(dim=ck['dim']).to(DEV)
    trained.load_state_dict(ck['model'])
    rand = MAE(dim=ck['dim']).to(DEV)        # same arch, untrained

    print("linear probes on frozen features:")
    r_rand = report('random-init', *collect(rand, ds))
    r_tr = report('SSL-trained', *collect(trained, ds))

    print(f"\ngains over random init:")
    print(f"  record ID     {r_tr['id_acc'] - r_rand['id_acc']:+.3f}")
    print(f"  rec_kurt      {r_tr['rec_kurt'] - r_rand['rec_kurt']:+.3f}")
    print(f"  within-record {r_tr['within_kurt'] - r_rand['within_kurt']:+.3f}"
          f"   <- the one CL1 needs")
    if r_tr['id_acc'] > 0.8:
        print("\n[!] record ID is almost perfectly linearly decodable: the "
              "rec_kurt result is confounded by fingerprinting.")


if __name__ == '__main__':
    sys.exit(main())
