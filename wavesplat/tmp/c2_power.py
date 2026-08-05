"""C2b rescue (pre-reg tmp/C2b_PREREG_rescue_2026-07-22.md): power via seed-replicates (c)
+ operational-regime Hs>=2 restriction (b). Two pre-fixed tests, honor rule frozen.

Cache: seed 0 -> tmp/c2_cache/sea_{si}.npz (existing) ; seed k>0 -> tmp/c2_cache/sea_{si}_s{k}.npz.
Split by (Tp,Hs) parity (all seeds of a sea share the split; block=(si,seed)).
Usage: python3 tmp/c2_power.py --n 32 --seeds 0,1    (builds missing, then analyzes)
Out: results/direction3/c2_power.json
"""
import json, os, sys, time, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts'); sys.path.insert(0, 'tmp')
from hos_s8_envelope import (SEAS, build_sea, draw, two_stage_kp, fit_theta_and_pred, crb_envelope)
from c2_control_demo import ls_gain, fit_wiener, wiener_gain, binned_gain_curve, apply_curve

CACHE = 'tmp/c2_cache'


def cache_path(si, seed):
    return f'{CACHE}/sea_{si}.npz' if seed == 0 else f'{CACHE}/sea_{si}_s{seed}.npz'


def get_sea(si, seed, t0):
    Tp, Hs = SEAS[si]; cf = cache_path(si, seed)
    if os.path.exists(cf):
        d = np.load(cf, allow_pickle=True)
        return dict(si=si, seed=seed, Hs=Hs, sea=str(d['sea']), sig=d['sig'], yh=d['yh'], yt=d['yt'])
    sea = build_sea(Tp, Hs, seed)
    Ps, zs, Q, gt = draw(sea, np.random.default_rng(900000 + 31 * si + 7919 * seed))
    Hs_e = 4 * np.std(zs); kp = two_stage_kp(Ps, zs)
    theta, pred, emp = fit_theta_and_pred(Ps, zs, Q, kp)
    sig = crb_envelope(theta, Ps, Q, Hs_e)
    fin = np.isfinite(sig) & np.isfinite(pred) & np.isfinite(gt)
    rec = dict(si=si, seed=seed, Hs=Hs, sea=f'Tp{Tp:g}_Hs{Hs:g}', sig=sig[fin], yh=pred[fin], yt=gt[fin])
    np.savez(cf, sea=rec['sea'], sig=rec['sig'], yh=rec['yh'], yt=rec['yt'])
    print(f"  built [{si:2d} s{seed}] {rec['sea']:12s} nq={int(fin.sum()):4d}  [{time.time()-t0:.0f}s]", flush=True)
    return rec


def R(yt, yh, g): return 1.0 - float(np.sum((yt - g * yh) ** 2)) / float(np.sum(yt * yt))


def analyze(train, test, nbin=10, tag=''):
    trs = np.concatenate([p['sig'] for p in train]); trt = np.concatenate([p['yt'] for p in train]); trh = np.concatenate([p['yh'] for p in train])
    g_const = ls_gain(trt, trh); g0, cc = fit_wiener(trs, trt, trh)

    def pooled(seas):
        yt = np.concatenate([p['yt'] for p in seas]); yh = np.concatenate([p['yh'] for p in seas]); sig = np.concatenate([p['sig'] for p in seas])
        return R(yt, yh, wiener_gain(sig, g0, cc)) - R(yt, yh, g_const)
    delta = pooled(test)
    rng = np.random.default_rng(7); boot = []
    for _ in range(2000):
        samp = [test[i] for i in rng.integers(0, len(test), len(test))]
        boot.append(pooled(samp))
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    persea = [(p['sea'], pooled([p])) for p in test]
    pos = sum(1 for _, d in persea if d > 0)
    win = delta > 0 and lo > 0
    print(f"  [{tag}] test={len(test)} blocks  Δ={delta:+.4f}  boot95%[{lo:+.4f},{hi:+.4f}]  per-block {pos}/{len(persea)}  {'WIN' if win else 'no-win'}")
    return dict(tag=tag, n_test=len(test), g_const=g_const, wiener=dict(g0=g0, c=cc),
                delta=delta, lo=lo, hi=hi, pos=pos, n_block=len(persea), win=win)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=32)
    ap.add_argument('--seeds', type=str, default='0,1'); a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(',')]
    N = min(a.n, len(SEAS)); t0 = time.time()
    print(f"C2b power+regime: N={N} (Tp,Hs) x seeds={seeds}\n", flush=True)
    per = [get_sea(si, sd, t0) for sd in seeds for si in range(N)]

    train = [p for p in per if p['si'] % 2 == 0]; test = [p for p in per if p['si'] % 2 == 1]
    print(f"\ntrain blocks={len(train)}  test blocks={len(test)}  (seeds pooled)\n")
    res = {}
    res['power'] = analyze(train, test, tag='Primary-1 power (all Hs)')
    tr2 = [p for p in train if p['Hs'] >= 2.0]; te2 = [p for p in test if p['Hs'] >= 2.0]
    res['operational'] = analyze(tr2, te2, tag='Primary-2 operational (Hs>=2)')

    p1, p2 = res['power']['win'], res['operational']['win']
    verdict = ("RESCUE (both primaries WIN)" if p1 and p2 else
               f"PARTIAL RESCUE (only {'power' if p1 else 'operational'} WIN) -> conditional/context-labeled" if p1 or p2 else
               "NO RESCUE (both no-win) -> fall back to C1 headline, C2 = suggestive-unestablished")
    print(f"\n=== C2b VERDICT === {verdict}")
    json.dump(dict(N=N, seeds=seeds, res=res, verdict=verdict),
              open('results/direction3/c2_power.json', 'w'), indent=1, default=float)
    print(f"saved results/direction3/c2_power.json ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
