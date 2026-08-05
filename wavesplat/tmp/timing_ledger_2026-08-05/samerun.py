"""Same-run, same-box, same-config WS-vs-baseline latency comparison.

Both arms come from the 2026-07-25 chain on data_finemesh_2026-07-23:
  batchC_ws_phasefix.jsonl  -> WaveSplat (shipped defaults, 5 estimator seeds)
  batchA_baselines.jsonl    -> ETKF(Ne=100,40x30) / LETKF(loc50,Ne=100) / GPR-1 / GPR-disp,
                               EXACTLY the paper's tuned configs, filter seed 42.

Both were measured with the same defective time.time() on the same box, so the
constant relative bias (-3.84%, see LEDGER Sec.3) cancels in the ratio.
"""
import json, os, re, collections
import numpy as np

ROOT = '/home/vexit/wavesplat/wavesplat'
STEP_MS = 1247.1        # measured 2026-08-05, mean of observed backward steps
PERIOD_S = 32.50        # measured, sd < 1e-3 s
EXPECTED_REL_BIAS = -STEP_MS / (PERIOD_S * 1000.0)
NF = {'BF2': 220, 'BF3': 190, 'BF4': 160, 'BF5': 120}
TAB12 = ['BF2_1comp_seed0_uniform', 'BF3_1comp_seed0_uniform', 'BF3_2comp_seed0_uniform',
         'BF3_2comp_seed0_slope', 'BF3_5comp_seed0_uniform', 'BF3_5comp_seed0_slope',
         'BF4_1comp_seed0_uniform', 'BF4_5comp_seed0_uniform', 'BF4_15comp_seed0_uniform',
         'BF4_30comp_seed0_uniform', 'BF5_15comp_seed0_uniform', 'BF5_30comp_seed0_uniform']


def norm(e):
    return e.replace('episode_', '').replace('.h5', '')


def sea(e):
    return re.search(r'BF\d', e).group(0)


def ws_per_episode(path, key='elapsed_ms', frac=0.5):
    """Return {episode: (raw_mean, clean_mean, median, n_dropped)} over estimator seeds."""
    G = collections.defaultdict(list)
    for l in open(os.path.join(ROOT, path)):
        r = json.loads(l)
        if isinstance(r.get(key), (int, float)) and np.isfinite(r[key]):
            G[norm(r['episode'])].append(r[key])
    out = {}
    for e, v in G.items():
        v = np.array(v, float); med = np.median(v)
        shift = STEP_MS / NF[sea(e)]
        keep = v[(med - v) < frac * shift]
        out[e] = (v.mean(), keep.mean(), med, len(v) - len(keep))
    return out


def base_per_episode(path):
    G = collections.defaultdict(dict)
    for l in open(os.path.join(ROOT, path)):
        r = json.loads(l)
        if isinstance(r.get('ms'), (int, float)):
            G[r['method']][norm(r['episode'])] = r['ms']
    return G


print(f"expected constant relative bias on ANY mean-aggregated timing on this box: "
      f"{EXPECTED_REL_BIAS*100:+.3f}%  (= -{STEP_MS:.1f} ms / {PERIOD_S:.2f} s)")

WS = ws_per_episode('tmp/rerun_2026-07-25/batchC_ws_phasefix.jsonl')
B = base_per_episode('tmp/rerun_2026-07-25/batchA_baselines.jsonl')

for scope, eps in (('100 episodes', sorted(WS)), ('12-episode subset', TAB12)):
    print(f"\n=== {scope} (n={len(eps)}) — ALL ARMS FROM THE 2026-07-25 CHAIN ===")
    wsr = np.array([WS[e][0] for e in eps if e in WS])
    wsc = np.array([WS[e][1] for e in eps if e in WS])
    wsm = np.array([WS[e][2] for e in eps if e in WS])
    drop = sum(WS[e][3] for e in eps if e in WS)
    print(f"  {'WaveSplat':14s} n={len(wsr):3d}  raw mean={wsr.mean():7.2f}  clean mean={wsc.mean():7.2f}  "
          f"median-of-seeds mean={wsm.mean():7.2f}  med={np.median(wsm):7.2f}  dropped={drop}")
    for m in sorted(B):
        v = np.array([B[m][e] for e in eps if e in B[m]], float)
        if len(v) == 0:
            continue
        corr = v / (1 + EXPECTED_REL_BIAS)     # de-bias: raw is low by 3.84%
        print(f"  {m:14s} n={len(v):3d}  raw mean={v.mean():7.2f}  de-biased={corr.mean():7.2f}  "
              f"med={np.median(v):7.2f}  |  ratio vs WS(clean)={v.mean()/wsc.mean():5.2f}x  "
              f"ratio vs WS(median)={np.median(v)/np.median(wsm):5.2f}x")
