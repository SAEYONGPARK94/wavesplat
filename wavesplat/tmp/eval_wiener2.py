"""Train wiener2 at the full 99k and score it on the SAME eval protocol as
MMSE / linear / cnn (in-fan and out-of-fan field nRMSE, 64 fresh scenes).

The scaling curves are validation MSE on cached scenes; the physics baseline
was never measured on that axis. Only this puts wiener2 and MMSE side by side.
"""
import sys, os, json, numpy as np, torch
sys.path.insert(0, 'wavesplat/scripts'); os.chdir('wavesplat'); sys.path.insert(0, 'scripts')
import d3_cbf_scaling2 as S2
import d3_complex_beamform_field as CBF

N_VAL = S2.N_VAL
X, F = S2.extend_cache(100000)
Ns = [len(X) - N_VAL]
rv, Ws = S2.ridge_curve(X, F, Ns)
n = Ns[0]
Xv = torch.tensor(X[:N_VAL]).to(S2.DEV); Fv = torch.tensor(F[:N_VAL]).to(S2.DEV)
Xt = torch.tensor(X[N_VAL:N_VAL+n]).to(S2.DEV); Ft = torch.tensor(F[N_VAL:N_VAL+n]).to(S2.DEV)
W0 = torch.tensor(Ws[n].T, dtype=torch.float32, device=S2.DEV)

# rebuild + train, keeping the trained net (S2.train returns only the score)
net = S2.Wiener2(W0).to(S2.DEV)
params = [p for p in net.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-2)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 12000)
rng = np.random.default_rng(0); best = np.inf; best_sd = None
for step in range(1, 12001):
    net.train()
    ii = torch.tensor(rng.choice(len(Xt), 64, replace=False), device=S2.DEV)
    xb, fb = Xt[ii], Ft[ii]
    if rng.random() < 0.5:
        xb, fb = torch.flip(xb, [2]), torch.flip(fb, [1])
    loss = ((net(xb) - fb) ** 2).mean()
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
    if step % 1000 == 0:
        net.eval()
        with torch.no_grad(): vl = float(((net(Xv) - Fv) ** 2).mean())
        if vl < best:
            best = vl; best_sd = {k: v.detach().clone() for k, v in net.state_dict().items()}
        print(f"  step {step} val {vl:.4f}", flush=True)
net.load_state_dict(best_sd); net.eval()
print(f"wiener2 best val {best:.4f}")

# also a ridge-only module for reference on the same eval
lin = CBF.LinearMap().to(S2.DEV)
with torch.no_grad(): lin.fc.weight.copy_(W0)
lin.eval()

ev = CBF.evaluate({'wiener2': net, 'linear99k': lin}, n_val=64, n_mmse=32)
print(f"\n{'estimator':<12} {'in-fan':>9} {'out-of-fan':>12} {'amp':>8}")
print('-'*45)
for k, v in ev.items():
    f = lambda x: '   --' if x is None else f'{x:9.3f}'
    print(f"{k:<12} {f(v['in'])} {f(v['out']):>12} {v['amp']:>8.3f}")
json.dump({'wiener2_val': best, 'eval': ev}, open('results/direction3/cbf_wiener2_eval.json','w'), indent=1)
