"""Did the bound-wave test fail, or was it silenced?

d3_cbf_boundwave trained on the RAW bound-wave field, whose variance is 0.0055
— 190x smaller than the free-wave target (1.04). Same lr, same weight decay
1e-2, same zero-initialised head. A 190x weaker gradient against unchanged
regularization is a good way to train a model that outputs zero, and "outputs
zero" is exactly the relative MSE of 1.0 we observed.

So before concluding "no network can synthesize the bound wave":

  1. normalize the target to unit variance (loss scale O(1))
  2. drop the frozen ridge scaffold — on this target the ridge is actively
     harmful (rel 1.074), so `resid` started below the zero-predictor
  3. weight decay 1e-4, not 1e-2
  4. train a plain CNN from scratch

If it still cannot beat the zero-predictor, the claim stands: c(k) does not
support the bound wave in a way a conv net can reach. If it can, the earlier
run measured the optimizer, not the physics.

Reference ceiling: the free field itself is only recovered to rel MSE 0.507,
i.e. corr ~ 0.70. The bound wave is quadratic in the free field, so its
attainable correlation is at best ~corr^2 ~ 0.49 -> rel MSE ~ 0.76.
Anything meaningfully below 1.0 is a real signal.
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..'))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'scripts')

from d3_cbf_boundwave import CACHE, NG2, N_VAL
from d3_complex_beamform_field import NK, CNN

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PlainCNN(nn.Module):
    def __init__(self):
        super().__init__()
        c = CNN()
        self.enc = c.enc
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.1),
                                  nn.Linear(256 * 5 * 5, 1024), nn.SiLU(),
                                  nn.Linear(1024, NG2 * NG2))

    def forward(self, x):
        return self.head(self.enc(x)).view(-1, NG2, NG2)


def main():
    d = np.load(CACHE)
    X, Fb, Ff = d['X'], d['Fb'], d['Ff']
    sb = float(np.std(Fb))
    Y = (Fb / sb).astype(np.float32)          # unit-variance target
    print(f"scenes {len(X)}  raw bound var {np.var(Fb):.5f} -> normalized 1.0")

    Xv = torch.tensor(X[:N_VAL]).to(DEV)
    Yv = torch.tensor(Y[:N_VAL]).to(DEV)
    Xt = torch.tensor(X[N_VAL:]).to(DEV)
    Yt = torch.tensor(Y[N_VAL:]).to(DEV)
    print(f"train {len(Xt)} / val {len(Xv)}")

    net = PlainCNN().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    steps = 15000
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    rng = np.random.default_rng(0)
    var_v = float((Yv ** 2).mean())
    best, t0 = np.inf, time.time()
    for step in range(1, steps + 1):
        net.train()
        ii = torch.tensor(rng.choice(len(Xt), 64, replace=False), device=DEV)
        xb, yb = Xt[ii], Yt[ii]
        if rng.random() < 0.5:
            xb, yb = torch.flip(xb, [2]), torch.flip(yb, [1])
        loss = ((net(xb) - yb) ** 2).mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 1500 == 0:
            net.eval()
            with torch.no_grad():
                p = net(Xv)
                vl = float(((p - Yv) ** 2).mean())
                amp = float(p.std() / Yv.std())
                cr = float(torch.corrcoef(torch.stack(
                    [p.flatten(), Yv.flatten()]))[0, 1])
            best = min(best, vl)
            print(f"  step {step:6d}  rel MSE {vl/var_v:.3f}  corr {cr:+.3f}  "
                  f"amp {amp:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"\nbest rel MSE {best/var_v:.3f}   (1.0 = zero-predictor)")
    print("theoretical ceiling from corr(free)~0.70:  ~0.76")


if __name__ == '__main__':
    sys.exit(main())
