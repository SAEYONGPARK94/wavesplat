"""C2 robustness (cache-only, no HOS rebuild): is the envelope-gated WIN split-specific?
Loads cached seas 0..N-1; runs (a) even/odd split, (b) odd/even swap, (c) leave-one-sea-out
(fit Wiener+const on the rest, test on held-out sea). Reports Delta=R_gated-R_const each.
Usage: python3 tmp/c2_robust.py [--n 16]
"""
import os, sys, glob, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'tmp')
from c2_control_demo import ls_gain, fit_wiener, wiener_gain

CACHE = 'tmp/c2_cache'


def load(n):
    per = []
    for si in range(n):
        f = f'{CACHE}/sea_{si}.npz'
        if os.path.exists(f):
            d = np.load(f, allow_pickle=True)
            per.append(dict(si=si, sea=str(d['sea']), sig=d['sig'], yh=d['yh'], yt=d['yt']))
    return per


def R(yt, yh, g): return 1.0 - float(np.sum((yt - g * yh) ** 2)) / float(np.sum(yt * yt))


def delta_split(train, test):
    trs = np.concatenate([p['sig'] for p in train]); trt = np.concatenate([p['yt'] for p in train]); trh = np.concatenate([p['yh'] for p in train])
    g_const = ls_gain(trt, trh); g0, cc = fit_wiener(trs, trt, trh)
    tes = np.concatenate([p['sig'] for p in test]); tet = np.concatenate([p['yt'] for p in test]); teh = np.concatenate([p['yh'] for p in test])
    return R(tet, teh, wiener_gain(tes, g0, cc)) - R(tet, teh, g_const)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=16); a = ap.parse_args()
    per = load(a.n)
    print(f"C2 robustness: {len(per)} cached seas\n")

    ev = [p for p in per if p['si'] % 2 == 0]; od = [p for p in per if p['si'] % 2 == 1]
    d_eo = delta_split(ev, od); d_oe = delta_split(od, ev)
    print(f"(a) split even->odd  Δ = {d_eo:+.4f}   [pre-registered]")
    print(f"(b) split odd->even  Δ = {d_oe:+.4f}   [swap]")

    # leave-one-sea-out: fit on all-but-one, test on held-out
    loso = []
    for i in range(len(per)):
        train = [per[j] for j in range(len(per)) if j != i]
        loso.append((per[i]['sea'], delta_split(train, [per[i]])))
    pos = sum(1 for _, d in loso if d > 0)
    ds = np.array([d for _, d in loso])
    print(f"\n(c) leave-one-sea-out ({len(loso)} folds): Δ>0 in {pos}/{len(loso)}")
    print(f"    LOSO Δ  mean {ds.mean():+.4f}  median {np.median(ds):+.4f}  min {ds.min():+.4f}  max {ds.max():+.4f}")
    for s, d in sorted(loso, key=lambda x: x[1]):
        print(f"      {s:12s} {d:+.4f}{'  <-- neg' if d < 0 else ''}")
    verdict = ("ROBUST: gated>const in both splits AND LOSO majority (win not split-specific)"
               if d_eo > 0 and d_oe > 0 and pos >= 0.75 * len(loso) else
               "FRAGILE: win depends on split -> caution")
    print(f"\n=== ROBUSTNESS === {verdict}")


if __name__ == '__main__':
    main()
