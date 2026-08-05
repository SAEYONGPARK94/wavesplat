"""Contamination audit: a swallowed backward clock step shifts an episode's mean
per-frame time by STEP_MS / n_frames.  Detect records whose timing sits an integer
number of such shifts below their episode's clean median."""
import json, os, re, glob
import numpy as np

STEP_MS = 1150.0                      # measured 1215/1228 ms today; deficits imply ~1130-1212
NFRAMES = {'BF2': 220, 'BF3': 190, 'BF4': 160, 'BF5': 120}   # tracking frames (tp_factor=2)

def sea(ep):
    m = re.search(r'BF\d', ep); return m.group(0) if m else 'BF4'

def audit(path, key, group_keys, label, nframes_mult=1.0):
    rows = [json.loads(l) for l in open(path)]
    rows = [r for r in rows if r.get(key) is not None and np.isfinite(r[key])]
    groups = {}
    for r in rows:
        g = tuple(r.get(k) for k in group_keys)
        groups.setdefault(g, []).append(r)
    flagged = []
    for g, rs in groups.items():
        v = np.array([r[key] for r in rs], float)
        med = np.median(v)
        n = NFRAMES[sea(rs[0]['episode'])] * nframes_mult
        shift = STEP_MS / n
        for r in rs:
            k = (med - r[key]) / shift
            if k >= 0.5:
                flagged.append((r, round(float(k), 2), med, shift))
    print(f"\n=== {label}  ({path})  n={len(rows)}  groups={len(groups)}")
    v = np.array([r[key] for r in rows], float)
    print(f"    value: min={v.min():.3f} med={np.median(v):.3f} max={v.max():.3f} | negatives={(v<0).sum()}")
    print(f"    FLAGGED (>=0.5 step swallowed): {len(flagged)}  ({len(flagged)/len(rows)*100:.1f}%)")
    hist = {}
    for _, k, _, _ in flagged:
        hist[int(round(k))] = hist.get(int(round(k)), 0) + 1
    print(f"    steps-swallowed histogram: {dict(sorted(hist.items()))}")
    for r, k, med, sh in sorted(flagged, key=lambda x: -x[1])[:8]:
        print(f"      {r['episode'][:42]:42s} seed={r.get('seed', r.get('method'))} "
              f"{key}={r[key]:8.3f} (clean med {med:6.3f}, {k:.2f} steps)")
    return flagged, len(rows)

tot_f = tot_n = 0
for path, key, gk, lab, mult in [
    ('tmp/rerun_2026-07-25/batchB_ws.jsonl', 'elapsed_ms', ['episode'], 'batchB WS (pre-phasefix)', 1.0),
    ('tmp/rerun_2026-07-25/batchC_ws_phasefix.jsonl', 'elapsed_ms', ['episode'], 'batchC WS (phasefix)', 1.0),
    ('tmp/rerun_2026-07-25/batchA_baselines.jsonl', 'ms', ['episode', 'method'], 'batchA baselines', 2.0),
]:
    if os.path.exists(path):
        f, n = audit(path, key, gk, lab, mult); tot_f += len(f); tot_n += n
print(f"\nTOTAL flagged {tot_f} / {tot_n}")
