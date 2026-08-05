"""NEUTRAL A0 — inventory every JSONL artifact from today: arms, cfg, init_nrmse, nRMSE."""
import json, os, glob, collections

TMP = "/home/vexit/wavesplat/wavesplat/tmp"
FILES = [
    "rx1_tracker_2026-08-05/results.jsonl",
    "rx1_tracker_2026-08-05/results2.jsonl",
    "rx1_tracker_2026-08-05/CON_probe_results.jsonl",
    "s1s2_2026-08-05/results.jsonl",
    "e1_grid_2026-08-05/results.jsonl",
    "e1_grid_2026-08-05/probe.jsonl",
    "e2_grid_sep_2026-08-05/results.jsonl",
    "e2_grid_sep_2026-08-05/probe.jsonl",
]

for f in FILES:
    p = os.path.join(TMP, f)
    if not os.path.exists(p):
        print(f"MISSING {f}")
        continue
    rows = []
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    print(f"\n=== {f}  n={len(rows)}")
    if not rows:
        continue
    print("  keys:", sorted(rows[0].keys()))
    arms = collections.Counter(r.get("arm", "?") for r in rows)
    print("  arms:", dict(arms))
    # cfg signature per arm
    sig = collections.defaultdict(set)
    for r in rows:
        cfg = r.get("cfg", {}) or {}
        s = (cfg.get("init_method"), tuple(cfg.get("n_gaussians") or []), cfg.get("max_n"),
             cfg.get("adaptive_n"), cfg.get("prune_threshold"))
        sig[r.get("arm", "?")].add(s)
    for a in sorted(sig):
        for s in sorted(sig[a], key=str):
            print(f"    {a:22s} init={s[0]} ladder={s[1]} max_n={s[2]} adaptive={s[3]} prune={s[4]}")
    has_init = sum(1 for r in rows if r.get("init_nrmse") is not None)
    print(f"  rows with init_nrmse: {has_init}/{len(rows)}")
    eps = sorted(set(r.get("episode", "?") for r in rows))
    print(f"  episodes ({len(eps)}):", eps[:8], "..." if len(eps) > 8 else "")
    seeds = sorted(set(r.get("seed") for r in rows if r.get("seed") is not None))
    print("  seeds:", seeds)
