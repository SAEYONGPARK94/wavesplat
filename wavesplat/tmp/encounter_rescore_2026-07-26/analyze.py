"""Scoring for the encounter re-score (W9-c). Metric frozen in PREREG.md section 5.

skill(cell, arm, d, lead) = 1 - nRMSE_cell / floor_cell, where
  rms(ep, seed)  = sqrt(mean over the 4 origins of (pred-gt)^2)
  rms_ep         = MEDIAN over seeds of rms(ep, seed)        <- median-of-k, mandatory
  nRMSE_cell     = mean over the 10 episodes of rms_ep / Hs
  floor          = same construction on the zero predictor (gt itself)
"""
import os, sys, json, argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); os.chdir(ROOT)
sys.path.insert(0, 'tmp/encounter_rescore_2026-07-26')
from common import DISTS, LEADS, HS, ORIGINS, episodes, EpisodeGeom

D = 'tmp/encounter_rescore_2026-07-26'


def load(fn):
    rows = []
    for line in open(f'{D}/{fn}'):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get('err') is None and d.get('pred') is not None:
            rows.append(d)
    return rows


def collect(rows):
    """-> {(cell, ep, seed): {'e2': (norg, nlead, ndist) err, 'gt2': ..., 'map': [...]}}"""
    acc = {}
    for r in rows:
        k = (r['cell'], r['episode'], r['seed'])
        a = acc.setdefault(k, {'org': {}, 'map': {}})
        a['org'][r['i_ref']] = (np.array(r['pred']) - np.array(r['gt']),
                                np.array(r['gt']))
        a['map'][r['i_ref']] = (r['map_rmse'], r['map_rms0'])
    return acc


def cell_stats(acc, cell, verbose=False):
    eps = episodes(cell)
    norg = len(ORIGINS[cell])
    per_ep_rms, per_ep_rms0, per_ep_map, per_ep_map0, incomplete = [], [], [], [], []
    for ep in eps:
        seeds = sorted({k[2] for k in acc if k[0] == cell and k[1] == ep})
        rms_s, map_s = [], []
        for s in seeds:
            a = acc.get((cell, ep, s))
            if a is None or len(a['org']) != norg:
                incomplete.append((ep, s, len(a['org']) if a else 0))
                continue
            E = np.stack([a['org'][i][0] for i in ORIGINS[cell]])      # (norg,nlead,ndist)
            rms_s.append(np.sqrt((E ** 2).mean(0)))                    # (nlead,ndist)
            map_s.append(np.sqrt(np.mean([a['map'][i][0] ** 2 for i in ORIGINS[cell]])))
        if not rms_s:
            continue
        per_ep_rms.append(np.median(np.stack(rms_s), axis=0))
        per_ep_map.append(float(np.median(map_s)))
        # zero-predictor floor: rebuilt from the episode itself (estimator-independent)
        G = EpisodeGeom(cell, ep)
        Gt = np.stack([G.targets(i)['gt'] for i in ORIGINS[cell]])
        per_ep_rms0.append(np.sqrt((Gt ** 2).mean(0)))
        per_ep_map0.append(float(np.sqrt(np.mean(
            [np.mean(G.eta_grid(G.gx, G.gy, float(G.ts[i])) ** 2)
             for i in ORIGINS[cell]]))))
    if not per_ep_rms:
        return None
    R = np.stack(per_ep_rms) / HS          # (nep, nlead, ndist)
    R0 = np.stack(per_ep_rms0) / HS
    nr, fl = R.mean(0), R0.mean(0)
    return dict(n_ep=len(per_ep_rms), nrmse=nr, floor=fl, skill=1 - nr / fl,
                per_ep_skill=1 - R / R0, incomplete=incomplete,
                map_nrmse=float(np.mean(per_ep_map) / HS),
                map_floor=float(np.mean(per_ep_map0) / HS),
                map_skill=float(1 - np.mean(per_ep_map) / np.mean(per_ep_map0)))


def rstar(skill, tau, leads=(0.0, 1.0, 2.0)):
    """Largest CONTIGUOUS distance whose min-skill over the given leads >= tau."""
    li = [LEADS.index(l) for l in leads]
    best = None
    for j, d in enumerate(DISTS):
        if min(skill[i, j] for i in li) >= tau:
            best = d
        else:
            break
    return best   # None == fails already at d=0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', nargs='+', default=['c', 'cbig', 'a', 'abig'])
    ap.add_argument('--out', default=f'{D}/results.json')
    a = ap.parse_args()
    arms = {}
    for arm, fn in (('ws', 'ws_enc.jsonl'), ('cond', 'cond_enc.jsonl')):
        if os.path.exists(f'{D}/{fn}'):
            arms[arm] = collect(load(fn))
    out = {'dists': list(DISTS), 'leads': list(LEADS), 'cells': {}}
    for cell in a.cells:
        out['cells'][cell] = {}
        for arm, acc in arms.items():
            if not any(k[0] == cell for k in acc):
                continue
            st = cell_stats(acc, cell)
            if st is None:
                continue
            out['cells'][cell][arm] = dict(
                n_ep=st['n_ep'], nrmse=st['nrmse'].tolist(), floor=st['floor'].tolist(),
                skill=st['skill'].tolist(),
                rstar50=rstar(st['skill'], 0.5), rstar30=rstar(st['skill'], 0.3),
                map_skill=st['map_skill'], map_nrmse=st['map_nrmse'],
                map_floor=st['map_floor'],
                per_ep_skill=st['per_ep_skill'].tolist(),
                incomplete=st['incomplete'])
    json.dump(out, open(a.out, 'w'), indent=1)

    # ---- console report ----
    for cell, arms_d in out['cells'].items():
        print(f"\n===== cell {cell.upper()} " + "=" * 60)
        for arm, st in arms_d.items():
            print(f"  [{arm}] n_ep={st['n_ep']}  global-map skill (lead 0, full grid) "
                  f"= {st['map_skill']:+.3f}  (nRMSE {st['map_nrmse']:.4f} / floor {st['map_floor']:.4f})")
            print(f"      skill by distance [m] x lead [s]      "
                  + "  ".join(f"{d:>6.0f}" for d in DISTS))
            for i, L in enumerate(LEADS):
                print(f"        lead {L:>3.0f} s                        "
                      + "  ".join(f"{st['skill'][i][j]:+6.3f}" for j in range(len(DISTS))))
            r50, r30 = st['rstar50'], st['rstar30']
            print(f"      R*(0.5, lead<=2s) = {('%g m' % r50) if r50 is not None else 'NONE'}"
                  f"   R*(0.3, lead<=2s) = {('%g m' % r30) if r30 is not None else 'NONE'}")
        if 'ws' in arms_d and 'cond' in arms_d:
            sw = np.array(arms_d['ws']['skill']); sc = np.array(arms_d['cond']['skill'])
            print(f"      learned - analytic  Dskill        "
                  + "  ".join(f"{d:>6.0f}" for d in DISTS))
            for i, L in enumerate(LEADS):
                print(f"        lead {L:>3.0f} s                        "
                      + "  ".join(f"{sc[i][j]-sw[i][j]:+6.3f}" for j in range(len(DISTS))))
    print(f"\nsaved {a.out}")


if __name__ == '__main__':
    main()
