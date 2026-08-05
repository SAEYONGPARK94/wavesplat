import json, os, re, glob
import numpy as np
STEP_MS = 1150.0
NF = {'BF2': 220, 'BF3': 190, 'BF4': 160, 'BF5': 120}
def sea(ep):
    m = re.search(r'BF\d', ep); return m.group(0) if m else 'BF4'

# --- batchA baselines: group by (method, sea) since only 1 record per (ep,method) ---
rows = [json.loads(l) for l in open('tmp/rerun_2026-07-25/batchA_baselines.jsonl')]
print("=== batchA baselines, grouped by (method, sea state) ===")
G = {}
for r in rows:
    G.setdefault((r['method'], sea(r['episode'])), []).append(r)
nflag = 0
for (m, s), rs in sorted(G.items()):
    v = np.array([r['ms'] for r in rs]); med = np.median(v)
    # baselines iterate ALL frames; n unknown -> use tracking-frame count as a LOWER bound on n
    shift_hi = STEP_MS / NF[s]                 # if they saw only NF frames (max possible shift)
    flag = [r for r in rs if med - r['ms'] >= 0.5 * shift_hi]
    nflag += len(flag)
    print(f"  {m:9s} {s}  n={len(rs):3d} med={med:7.2f} min={v.min():7.2f} "
          f"spread=[{v.min():.1f},{v.max():.1f}] max_shift_if_1step={shift_hi:5.2f}ms flagged={len(flag)}")
print(f"  batchA total flagged (loose criterion): {nflag}/{len(rows)}")

# --- broad scan: any negative timing-like number anywhere in repo JSON/JSONL ---
print("\n=== broad scan for negative timing values in repo results ===")
KEYS = ('elapsed_ms', 'avg_inference_ms', 'inference_ms', 'ms', 'elapsed_s', 'init_time_s',
        'latency_ms', 'time_ms', 'ms_per_frame')
hits = []
def walk(o, path, f):
    if isinstance(o, dict):
        for k, v in o.items():
            if k in KEYS and isinstance(v, (int, float)) and v < 0:
                hits.append((f, path + '/' + k, v))
            walk(v, path + '/' + str(k), f)
    elif isinstance(o, list):
        for i, v in enumerate(o[:4000]):
            walk(v, f'{path}[{i}]', f)
files = []
for pat in ('results/**/*.json', 'tmp/**/*.json', 'tmp/**/*.jsonl', 'paper/**/*.json'):
    files += glob.glob(pat, recursive=True)
scanned = 0
for f in sorted(set(files)):
    try:
        if os.path.getsize(f) > 80e6: continue
        if f.endswith('.jsonl'):
            for i, line in enumerate(open(f)):
                try: walk(json.loads(line), f'L{i}', f)
                except Exception: pass
        else:
            walk(json.load(open(f)), '', f)
        scanned += 1
    except Exception:
        pass
print(f"scanned {scanned} files; negative-timing hits: {len(hits)}")
byfile = {}
for f, p, v in hits:
    byfile.setdefault(f, []).append(v)
for f, vs in sorted(byfile.items(), key=lambda x: -len(x[1])):
    a = np.array(vs)
    print(f"  {len(vs):4d}  min={a.min():9.3f} max={a.max():8.3f}  {f}")
