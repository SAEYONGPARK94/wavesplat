"""B3 INDEPENDENT RE-DERIVATION (path ii).

Written from scratch against tmp/B3_rx3_2026-08-06/PREREG.md.  It deliberately does NOT
import or reuse analyze_b3.py, and it differs structurally on purpose:

  * panel quantities (m, decorr, shrink, |VR-1|) are RECOMPUTED from the stored per-frame
    arrays `vr_frames` / `rho_frames` instead of trusting the runner's aggregates;
  * the sign test is scipy.stats.binomtest, not a hand-rolled binomial sum;
  * medians come from statistics.median on plain python lists, not numpy;
  * episodes/strata are rebuilt by parsing the filename, not from a hard-coded list.

If path (i) and path (ii) disagree anywhere, that is reported as a discrepancy, not
silently reconciled.
"""
import json
import os
import re
import statistics as st
import sys

from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))

# PREREG sec.6, frozen before the run
MDE = {'ALL-12': 0.0134, 'crowded': 0.0329, 'mid': 0.0042, 'sparse': 0.0014}


def ncomp(ep):
    return int(re.search(r'BF\d_(\d+)comp', ep).group(1))


def stratum(ep):
    n = ncomp(ep)
    return 'sparse' if n <= 2 else ('mid' if n == 5 else 'crowded')


def read(p):
    out = []
    with open(p) as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            d = json.loads(ln)
            if 'error' not in d:
                out.append(d)
    return out


def panel_from_frames(rec):
    """Recompute the orthogonal split from the RAW per-frame arrays.
    m = (1-rho^2) + (sqrt(v)-rho)^2 ; |VR-1| formed per frame then averaged (PREREG)."""
    v = rec.get('vr_frames') or []
    r = rec.get('rho_frames') or []
    if not v or len(v) != len(r):
        return {}
    dec, shr, av = [], [], []
    for vi, ri in zip(v, r):
        s = (max(vi, 0.0)) ** 0.5
        dec.append(1.0 - ri * ri)
        shr.append((s - ri) ** 2)
        av.append(abs(vi - 1.0))
    n = len(v)
    return {'m': sum(d + s for d, s in zip(dec, shr)) / n,
            'decorr': sum(dec) / n, 'shrink': sum(shr) / n,
            'absVR': sum(av) / n, 'rho': sum(r) / n}


def main():
    rows = read(os.path.join(HERE, 'results.jsonl'))
    nulls = {d['episode']: d['zero_null_tracking_nRMSE_mean']
             for d in json.load(open(os.path.join(
                 REPO, 'tmp/B2_g1p1_ext_2026-08-06/nulls12.json')))}

    episodes = sorted({r['episode'] for r in rows}, key=lambda e: (stratum(e), ncomp(e), e))
    cells = sorted({r['cell'] for r in rows})
    seeds = sorted({r['seed'] for r in rows})
    print(f"[rederive] {len(rows)} records | {len(cells)} cells | {len(episodes)} episodes "
          f"| seeds {seeds}")
    print(f"[rederive] strata: " + ", ".join(
        f"{s}={sum(1 for e in episodes if stratum(e) == s)}"
        for s in ('sparse', 'mid', 'crowded')))

    # ---- seeds -> episode medians, on RECOMPUTED panel quantities --------------------
    lvl = {}
    for c in cells:
        for e in episodes:
            sub = [r for r in rows if r['cell'] == c and r['episode'] == e]
            if not sub:
                continue
            pf = [panel_from_frames(r) for r in sub]
            lvl[(c, e)] = {
                'nRMSE': st.median([r['nRMSE'] for r in sub]),
                'm': st.median([p['m'] for p in pf]),
                'decorr': st.median([p['decorr'] for p in pf]),
                'shrink': st.median([p['shrink'] for p in pf]),
                'absVR': st.median([p['absVR'] for p in pf]),
                'rho': st.median([p['rho'] for p in pf]),
                'n_seed': len(sub),
            }

    # ---- cross-check: recomputed panel vs runner-stored aggregates --------------------
    worst = 0.0
    for r in rows:
        p = panel_from_frames(r)
        if not p:
            continue
        for a, b in (('m', 'm_mean'), ('decorr', 'decorr_mean'), ('shrink', 'shrink_mean'),
                     ('absVR', 'absVR_framewise'), ('rho', 'Pearson_rho')):
            if r.get(b) is not None:
                worst = max(worst, abs(p[a] - r[b]))
    print(f"[rederive] recomputed-from-frames vs runner aggregates: max|diff| = {worst:.3e}")

    # ---- null screen -----------------------------------------------------------------
    keep = [e for e in episodes if lvl[('AUTO', e)]['nRMSE'] < nulls[e]]
    print(f"[rederive] null screen (control AUTO < flat-sea null): kept {len(keep)}/"
          f"{len(episodes)}; dropped {[e[8:-3] for e in episodes if e not in keep] or 'none'}")

    # ---- paired deltas ---------------------------------------------------------------
    def report(cell, eps, label):
        eps = [e for e in eps if (cell, e) in lvl]
        if not eps:
            return None
        d = [lvl[(cell, e)]['nRMSE'] - lvl[('AUTO', e)]['nRMSE'] for e in eps]
        k = sum(1 for x in d if x < 0)
        n = len(d)
        p = binomtest(k, n, 0.5, alternative='two-sided').pvalue
        rel = [(lvl[(cell, e)]['nRMSE'] - lvl[('AUTO', e)]['nRMSE']) / lvl[('AUTO', e)]['nRMSE']
               for e in eps]
        pan = {}
        for key in ('rho', 'absVR', 'm', 'decorr', 'shrink'):
            pan[key] = st.median([lvl[(cell, e)][key] - lvl[('AUTO', e)][key] for e in eps])
        lv = st.median([lvl[(cell, e)]['nRMSE'] for e in eps])
        nl = st.median([nulls[e] for e in eps])
        print(f"  {cell:14s} {label:16s} lvl={lv:.4f} null={nl:.4f} "
              f"{'PASS' if lv < nl else 'FAIL':4s} | d={st.median(d):+.4f} {k}/{n} p={p:.4f} "
              f"rel={st.median(rel):+.1%} | drho={pan['rho']:+.4f} "
              f"d|VR-1|={pan['absVR']:+.4f} dm={pan['m']:+.4f} "
              f"ddec={pan['decorr']:+.4f} dshr={pan['shrink']:+.4f}")
        return st.median(d), k, n, p

    for label, eps in [('ALL-12', episodes), ('ALL-12/screened', keep)] + [
            (s, [e for e in keep if stratum(e) == s]) for s in ('crowded', 'mid', 'sparse')]:
        print(f"\n-- block {label} (n={len(eps)}) --")
        for c in cells:
            if c == 'AUTO':
                continue
            report(c, eps, label)

    # ---- pre-registered verdict ------------------------------------------------------
    print("\n== VERDICT (rederived, PREREG sec.7) : AUTO-NOTRACK vs AUTO ==")
    for name, eps, mkey in [('ALL-12', keep, 'ALL-12'),
                            ('crowded', [e for e in keep if stratum(e) == 'crowded'], 'crowded')]:
        d = [lvl[('AUTO-NOTRACK', e)]['nRMSE'] - lvl[('AUTO', e)]['nRMSE'] for e in eps]
        n = len(d)
        k = sum(1 for x in d if x < 0)
        med = st.median(d)
        if name == 'ALL-12':
            unan, coin = (k >= 11 or k <= n - 11), (4 <= k <= 8)
        else:
            unan = (k == n or k == 0)
            coin = not unan
        verdict = ('MATTERS' if (abs(med) >= MDE[mkey] and unan)
                   else 'NULL' if (abs(med) < MDE[mkey] and coin) else 'INCONCLUSIVE')
        print(f"  {name:10s} n={n} delta_med={med:+.4f} k={k}/{n} "
              f"p={binomtest(k, n, 0.5).pvalue:.4f} MDE={MDE[mkey]} -> {verdict}")

    # ---- integrity: Phase-1 invariance ------------------------------------------------
    print("\n== Phase-1 invariance (rederived) ==")
    bad = 0
    for e in episodes:
        for s in seeds:
            ref = [r for r in rows if r['cell'] == 'AUTO' and r['episode'] == e and r['seed'] == s]
            if not ref:
                continue
            ref = ref[0]
            for c in cells:
                cur = [r for r in rows if r['cell'] == c and r['episode'] == e and r['seed'] == s]
                if not cur:
                    continue
                cur = cur[0]
                if (cur['init_nrmse'] != ref['init_nrmse']
                        or cur.get('auto_selected') != ref.get('auto_selected')
                        or cur.get('ensemble_k') != ref.get('ensemble_k')
                        or cur.get('N') != ref.get('N')):
                    bad += 1
                    if bad <= 5:
                        print(f"   MISMATCH {c} {e[8:-3]} seed{s}: "
                              f"init {cur['init_nrmse']} vs {ref['init_nrmse']}, "
                              f"route {cur.get('auto_selected')} vs {ref.get('auto_selected')}, "
                              f"N {cur.get('N')} vs {ref.get('N')}")
    print(f"   total Phase-1 mismatches: {bad}")


if __name__ == '__main__':
    main()
