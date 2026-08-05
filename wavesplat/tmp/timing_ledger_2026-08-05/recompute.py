"""Timing provenance recompute.  Read-only over raw JSONL; no experiments re-run.

Exclusion rule (stated once, applied everywhere):
  A backward CLOCK_REALTIME step of magnitude STEP_MS swallowed inside one
  update() call makes that frame record ~ -STEP_MS instead of ~+6.5 ms, so the
  episode mean (mean over n_frames) drops by ~STEP_MS/n_frames.  For an episode
  group (same episode, 5 estimator seeds) we take the group median as the clean
  reference and flag any record sitting >= FRAC * (STEP_MS / n_frames) below it.
  FRAC=0.5 is the probe's rule; we also report FRAC=0.75 and a rule-free
  alternative (median-based aggregation, which needs no exclusion at all).
"""
import json, os, re, sys
import numpy as np

ROOT = '/home/vexit/wavesplat/wavesplat'
STEP_MS = 1150.0
NF = {'BF2': 220, 'BF3': 190, 'BF4': 160, 'BF5': 120}  # tracking frames, tp_factor=2


def sea(ep):
    m = re.search(r'BF\d', ep)
    return m.group(0) if m else 'BF4'


def load(path, key):
    rows = [json.loads(l) for l in open(os.path.join(ROOT, path))]
    return [r for r in rows if isinstance(r.get(key), (int, float)) and np.isfinite(r[key])]


def analyse(path, key, label, frac=0.5):
    rows = load(path, key)
    G = {}
    for r in rows:
        G.setdefault(r['episode'], []).append(r)
    clean, dirty = [], []
    per_ep_raw_med, per_ep_clean_med = [], []
    for ep, rs in G.items():
        v = np.array([r[key] for r in rs], float)
        med = np.median(v)
        shift = STEP_MS / NF[sea(ep)]
        keep = v[(med - v) < frac * shift]
        drop = v[(med - v) >= frac * shift]
        clean += list(keep); dirty += list(drop)
        per_ep_raw_med.append(med)
        per_ep_clean_med.append(np.median(keep) if len(keep) else np.nan)
    allv = np.array([r[key] for r in rows], float)
    c = np.array(clean, float)
    return dict(
        label=label, path=path, n=len(rows), n_groups=len(G),
        n_dropped=len(dirty), pct_dropped=100.0 * len(dirty) / len(rows),
        raw_mean=float(allv.mean()), raw_median=float(np.median(allv)),
        raw_min=float(allv.min()), raw_max=float(allv.max()),
        n_negative=int((allv < 0).sum()),
        clean_mean=float(c.mean()), clean_median=float(np.median(c)),
        clean_std=float(c.std(ddof=1)),
        # paper-style two-stage: per-episode median over seeds, then mean over episodes
        medseed_then_mean_raw=float(np.nanmean(per_ep_raw_med)),
        medseed_then_mean_clean=float(np.nanmean(per_ep_clean_med)),
        medseed_then_median_raw=float(np.nanmedian(per_ep_raw_med)),
        dropped_values=sorted(float(x) for x in dirty)[:12],
    )


def show(d):
    print(f"\n--- {d['label']}  [{d['path']}]")
    print(f"    n={d['n']} groups={d['n_groups']}  negatives={d['n_negative']}  "
          f"range=[{d['raw_min']:.3f}, {d['raw_max']:.3f}]")
    print(f"    RAW    mean={d['raw_mean']:8.3f}  median={d['raw_median']:8.3f}")
    print(f"    CLEAN  mean={d['clean_mean']:8.3f}  median={d['clean_median']:8.3f}  "
          f"(sd {d['clean_std']:.3f})   dropped {d['n_dropped']}/{d['n']} = {d['pct_dropped']:.1f}%")
    print(f"    DELTA  mean {d['raw_mean']-d['clean_mean']:+.3f} ms "
          f"({(d['raw_mean']-d['clean_mean'])/d['clean_mean']*100:+.2f}%)   "
          f"median {d['raw_median']-d['clean_median']:+.3f} ms")
    print(f"    two-stage (med over seeds -> mean over eps): raw={d['medseed_then_mean_raw']:.3f} "
          f"clean={d['medseed_then_mean_clean']:.3f}  | med-of-med raw={d['medseed_then_median_raw']:.3f}")
    print(f"    dropped values (lowest 12): {[round(x,2) for x in d['dropped_values']]}")


if __name__ == '__main__':
    out = {}
    for path, key, label in [
        ('tmp/rerun_2026-07-24/ep100_runs.jsonl', 'elapsed_ms', 'WS 100ep (2026-07-24, pre-phasefix)'),
        ('tmp/rerun_2026-07-25/batchB_ws.jsonl', 'elapsed_ms', 'WS 100ep batchB (pre-phasefix rerun)'),
        ('tmp/rerun_2026-07-25/batchC_ws_phasefix.jsonl', 'elapsed_ms', 'WS 100ep batchC (PHASEFIX = paper arm)'),
    ]:
        for frac in (0.5, 0.75):
            d = analyse(path, key, f'{label}  FRAC={frac}', frac)
            show(d)
            out[f'{path}|{frac}'] = d
    # wall_s sanity
    for path in ('tmp/rerun_2026-07-24/ep100_runs.jsonl',
                 'tmp/rerun_2026-07-25/batchC_ws_phasefix.jsonl'):
        rows = load(path, 'wall_s')
        w = np.array([r['wall_s'] for r in rows])
        print(f"\n    wall_s {path}: n={len(w)} min={w.min():.3f} med={np.median(w):.3f} "
              f"max={w.max():.3f} negatives={(w<0).sum()}")
    json.dump(out, open(os.path.join(ROOT, 'tmp/timing_ledger_2026-08-05/recompute.json'), 'w'), indent=1)
