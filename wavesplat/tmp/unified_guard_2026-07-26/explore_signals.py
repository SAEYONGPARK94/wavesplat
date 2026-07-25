"""DESIGN-PHASE exploration (pre-freeze): characterize learned-proposal atoms per regime.

Goal: find ONE GT-free 'disagreement' signal that separates the BF2_1comp over-proposal
edge (raw n_selected 'many' but really one clean swell) from genuine crowding, so a guard
can veto the learned route without flipping any crowded decision.

Reads ONLY the atom caches (raw pooled atoms) -> no GPU, no tracker. Uses the repo's
pool_and_select for n_selected reproduction. Prints distributions by comp count.
"""
import json, re, sys, os
import numpy as np

sys.path.insert(0, os.path.abspath('.'))
from wavesplat.optimizer.learned_init import pool_and_select

S1_CACHE = 'tmp/wp_integration_2026-07-25/atoms_cache.jsonl'
X_CACHE = 'tmp/unified_2026-07-26/atoms_cache_extra.jsonl'
DIFF_SEEDS = (0, 1, 2, 3, 4)


def comp(ep):
    return int(re.search(r'_(\d+)comp', ep).group(1))


def load_cache():
    rows = {}
    for src in (S1_CACHE, X_CACHE):
        for l in open(src):
            d = json.loads(l)
            if d.get('kx') is not None and d.get('err') is None:
                rows[(d['episode'], d['diff_seed'])] = d
    return rows


def merge_clusters(kx, ky, en, radius):
    """Single-linkage agglomerate of a SELECTED set within Euclidean `radius` in k-space.
    Returns energy-weighted centroid (kx,ky), summed energy, cluster count."""
    n = len(kx)
    if n == 0:
        return np.array([]), np.array([]), np.array([]), 0
    parent = list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb
    for i in range(n):
        for j in range(i + 1, n):
            if (kx[i] - kx[j])**2 + (ky[i] - ky[j])**2 < radius * radius:
                union(i, j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    mkx, mky, men = [], [], []
    for idxs in groups.values():
        w = np.array([en[i] for i in idxs]); w = w / w.sum() if w.sum() > 0 else np.ones(len(idxs))/len(idxs)
        mkx.append(float(np.sum(w * np.array([kx[i] for i in idxs]))))
        mky.append(float(np.sum(w * np.array([ky[i] for i in idxs]))))
        men.append(float(np.sum([en[i] for i in idxs])))
    return np.array(mkx), np.array(mky), np.array(men), len(groups)


def main():
    rows = load_cache()
    # aggregate per (comp)
    by_comp = {}
    per_record = []
    for (ep, ds), d in rows.items():
        c = comp(ep)
        rkx = np.asarray(d['kx'], float); rky = np.asarray(d['ky'], float); ren = np.asarray(d['energy'], float)
        sk, sy, se = pool_and_select(rkx, rky, ren, energy_thresh=0.03, dedup_dk=0.03, max_n=30)
        n_sel = len(sk)
        rec = dict(ep=ep, ds=ds, comp=c, n_sel=n_sel)
        if n_sel > 0:
            kmag = np.sqrt(sk**2 + sy**2)
            e_order = np.argsort(-se)
            kp = float(kmag[e_order[0]])           # |k| of dominant selected atom
            etot = se.sum()
            rec['kp'] = kp
            rec['dom_frac'] = float(se.max() / etot) if etot > 0 else 0.0
            rec['top2_frac'] = float(np.sort(se)[::-1][:2].sum() / etot) if etot > 0 else 0.0
            rec['top3_frac'] = float(np.sort(se)[::-1][:3].sum() / etot) if etot > 0 else 0.0
            # merged counts: absolute radius and kp-scaled
            for tag, rad in [('abs015', 0.15), ('abs0225', 0.225),
                             ('kp15', 0.15 * kp), ('kp25', 0.25 * kp), ('kp40', 0.40 * kp)]:
                _, _, _, nm = merge_clusters(sk, sy, se, rad)
                rec['nmerge_' + tag] = nm
        else:
            rec['kp'] = float('nan'); rec['dom_frac'] = float('nan')
            for tag in ('abs015','abs0225','kp15','kp25','kp40'):
                rec['nmerge_' + tag] = 0
        per_record.append(rec)
        by_comp.setdefault(c, []).append(rec)

    def stat(vals):
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        if not vals: return 'n/a'
        return f"med={np.median(vals):.2f} [{np.min(vals):.2f},{np.max(vals):.2f}]"

    print("="*100)
    print("Per-comp distributions (median [min,max]) across all diff_seeds")
    print("="*100)
    hdr = f"{'comp':>4} {'N':>4} | {'n_sel':>16} | {'dom_frac':>16} | {'top2_frac':>16} | {'kp':>14}"
    print(hdr)
    for c in sorted(by_comp):
        recs = by_comp[c]
        print(f"{c:>4} {len(recs):>4} | {stat([r['n_sel'] for r in recs]):>16} | "
              f"{stat([r.get('dom_frac') for r in recs]):>16} | {stat([r.get('top2_frac') for r in recs]):>16} | "
              f"{stat([r.get('kp') for r in recs]):>14}")

    print("\n" + "="*100)
    print("MERGED component counts by comp (median [min,max])")
    print("="*100)
    print(f"{'comp':>4} {'N':>4} | " + " | ".join(f"{t:>14}" for t in ('nmerge_abs015','nmerge_abs0225','nmerge_kp15','nmerge_kp25','nmerge_kp40')))
    for c in sorted(by_comp):
        recs = by_comp[c]
        print(f"{c:>4} {len(recs):>4} | " + " | ".join(
            f"{stat([r['nmerge_'+t] for r in recs]):>14}" for t in ('abs015','abs0225','kp15','kp25','kp40')))

    # Focus: the misroute regime = comp 1,2 records with n_sel>=8 (would route learned)
    print("\n" + "="*100)
    print("MISROUTE CANDIDATES: sparse (comp 1,2) records with raw n_sel>=8 (currently route learned)")
    print("="*100)
    mis = [r for r in per_record if r['comp'] in (1,2) and r['n_sel'] >= 8]
    print(f"count={len(mis)}")
    for r in sorted(mis, key=lambda x:(x['comp'],x['ep'],x['ds'])):
        print(f"  {r['ep']:34s} c{r['comp']} d{r['ds']} n_sel={r['n_sel']:2d} dom_frac={r.get('dom_frac',0):.3f} "
              f"top2={r.get('top2_frac',0):.3f} kp={r.get('kp',0):.3f} "
              f"nm[abs015={r['nmerge_abs015']} kp25={r['nmerge_kp25']} kp40={r['nmerge_kp40']}]")

    # Crowded records with n_sel>=8 (must NOT be vetoed) -- check merged counts stay high
    print("\n" + "="*100)
    print("CROWDED (comp 15,30) records routing learned (n_sel>=8): merged-count floor check")
    print("="*100)
    crw = [r for r in per_record if r['comp'] in (15,30) and r['n_sel'] >= 8]
    print(f"count={len(crw)}  (these must stay routed learned under any guard)")
    for t in ('abs015','abs0225','kp25','kp40'):
        vals = [r['nmerge_'+t] for r in crw]
        print(f"  merged[{t}]: min={min(vals)} median={np.median(vals):.1f}  (# with merged<8: {sum(v<8 for v in vals)}, <6: {sum(v<6 for v in vals)}, <4:{sum(v<4 for v in vals)})")
        vals_df = [r['dom_frac'] for r in crw]
    print(f"  dom_frac on crowded: max={max(r['dom_frac'] for r in crw):.3f} median={np.median([r['dom_frac'] for r in crw]):.3f}")
    print(f"  dom_frac on sparse-misroute: min={min(r['dom_frac'] for r in mis):.3f} median={np.median([r['dom_frac'] for r in mis]):.3f}")

    json.dump([{k:(v if not isinstance(v,float) or np.isfinite(v) else None) for k,v in r.items()} for r in per_record],
              open('tmp/unified_guard_2026-07-26/explore_signals.json','w'), indent=0)
    print("\nwrote explore_signals.json (", len(per_record), "records )")


if __name__ == '__main__':
    main()
