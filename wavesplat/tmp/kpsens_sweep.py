"""kp-sensitivity sweep — grounds the kp-robust B-2 redesign. NO retrain.
Evaluates failure-window T-L/T-N at kp_hint = m * sea['kp'] (true peak) for a
range of multipliers m, on the frozen 11 failure seas / 18 draws (same rng as
E-B2K). Maps how the loss-swap advantage depends on kp error. Reuses existing
checkpoints. Out: results/direction3/kpsens_sweep.json
"""
import json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts'); sys.path.insert(0, 'tmp')

from e15_hos_multidraw import (SEAS, build_sea, T_MANUAL_50, N_OBS, LEADS,
                               DEPTH, NOISE, R_MIN, FOV_H, z_at)
from d3_prior_net_eval import net_gamma_mmse
from e60k_hos_arena import load_model, ARMS

MULT = [0.7, 0.85, 1.0, 1.15, 1.3]   # kp = m * true peak
FROZEN = 'results/direction3/e60k_hos_arena.json'
OUT = 'results/direction3/kpsens_sweep.json'


def build_draw(sea, rng):
    P, zv, rr = [], [], []
    for sc in sea['scans']:
        pw = sc['amp'] * rng.exponential(1.0, len(sc['amp'])); keep = pw > T_MANUAL_50
        P.append(np.column_stack([sc['xy'][keep], np.full(int(keep.sum()), sc['trel'])]))
        zv.append(sc['z'][keep]); rr.append(sc['r'][keep])
    P, zv, rr = np.vstack(P), np.concatenate(zv), np.concatenate(rr)
    zv = zv + rng.normal(0, NOISE, len(zv))
    edges = np.linspace(R_MIN, rr.max() + 1e-6, 13); b = np.clip(np.digitize(rr, edges) - 1, 0, 11)
    take = [rng.choice(np.where(b == j)[0], min((b == j).sum(), max(-(-N_OBS // 12), 1)), False)
            for j in range(12) if (b == j).sum()]
    sel = np.concatenate(take); Ps, zs = P[sel], zv[sel]
    rf = np.sqrt(rng.uniform((15.0 / 200.0) ** 2, 1.0, 800)) * 200.0
    af = rng.uniform(-np.deg2rad(FOV_H), np.deg2rad(FOV_H), 800)
    qx, qy = rf * np.cos(af), rf * np.sin(af)
    Q = np.vstack([np.column_stack([qx, qy, np.full(800, l_)]) for l_ in LEADS])
    gt = np.concatenate([z_at(sea['leads'][li], qx, qy) for li in range(len(LEADS))])
    return Ps, zs, Q, gt


def main():
    models = {k: load_model(v) for k, v in ARMS.items()}
    frozen = {(r['sea'], r['draw']): r for r in json.load(open(FROZEN))['rows']}
    fail = {(s, d) for (s, d), r in frozen.items() if r['hand'] < 0.5}
    print(f"failure draws: {len(fail)} across {len({s for s,_ in fail})} seas", flush=True)
    rows = []
    if os.path.exists(OUT):
        rows = json.load(open(OUT)).get('rows', [])
    done = {(r['sea'], r['draw']) for r in rows}
    t0 = time.time()
    for si, (Tp, Hs) in enumerate(SEAS):
        tag = f'Tp{Tp:g}_Hs{Hs:g}'
        draws = [d for d in range(5) if (tag, d) in fail and (tag, d) not in done]
        if not draws:
            continue
        sea = build_sea(Tp, Hs, 0)
        kp_true = float(sea['kp'])
        for d in draws:
            Ps, zs, Q, gt = build_draw(sea, np.random.default_rng(900_000 + 7919 * si + 31 * d))
            row = {'sea': tag, 'draw': d, 'kp_true': kp_true, 'hand': frozen[(tag, d)]['hand']}
            for m in MULT:
                kp = m * kp_true
                for name, (net, head) in models.items():
                    pn = np.asarray(net_gamma_mmse(net, Ps, zs, DEPTH, Q, NOISE, seed=0,
                                                   kp_hint=kp, use_em=False, smult_head=head))
                    row[f'{name}_m{m}'] = float(np.corrcoef(pn, gt)[0, 1])
            rows.append(row)
        json.dump(dict(rows=rows, mult=MULT), open(OUT, 'w'), indent=1, default=float)
        print(f"  {tag:12s} done [{time.time()-t0:.0f}s]", flush=True)
    analyze(rows)


def analyze(rows):
    from scipy import stats
    seas = {}
    for r in rows:
        seas.setdefault(r['sea'], []).append(r)
    def arm(r, a, m): return np.mean([r[f'{a}_s{s}_m{m}'] for s in (0, 1, 2)])
    print(f"\n=== failure-window T-L / T-N vs kp multiplier (k={len(seas)} seas, cluster mean) ===")
    print(f"  {'m':>5s} | {'T-L':>8s} {'sign':>5s} | {'T-N':>8s} {'sign':>5s} | {'T':>6s} {'L':>6s} {'N':>6s}")
    out = {}
    for m in MULT:
        tl, tn, Ts, Ls, Ns = [], [], [], [], []
        for s, rs in seas.items():
            tl.append(np.mean([arm(r, 'T', m) - arm(r, 'L', m) for r in rs]))
            tn.append(np.mean([arm(r, 'T', m) - arm(r, 'N', m) for r in rs]))
            Ts.append(np.mean([arm(r, 'T', m) for r in rs]))
            Ls.append(np.mean([arm(r, 'L', m) for r in rs]))
            Ns.append(np.mean([arm(r, 'N', m) for r in rs]))
        tl, tn = np.array(tl), np.array(tn)
        out[m] = dict(TL=float(tl.mean()), TN=float(tn.mean()),
                      TL_pos=int((tl > 0).sum()), k=len(tl))
        print(f"  {m:5.2f} | {tl.mean():+8.4f} {int((tl>0).sum()):>3d}/{len(tl)} | "
              f"{tn.mean():+8.4f} {int((tn>0).sum()):>3d}/{len(tn)} | "
              f"{np.mean(Ts):6.3f} {np.mean(Ls):6.3f} {np.mean(Ns):6.3f}")
    print("  (ref: E-B2K oracle-kp T-L +0.139 | M-3 beamform-kp T-L +0.022)")
    json.dump(dict(rows=rows, mult=MULT, summary=out), open(OUT, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
