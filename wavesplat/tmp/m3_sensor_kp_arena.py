"""M-3 (1단계) — E-B2K eval-side sensor-kp test. Prereg = tmp/M3_sensor_kp_prereg_2026-07-22.md.
Re-evaluates the 9 arms with kp beamformed from the 600 sensor points (kp_hint=None)
on the SAME frozen draws; frozen oracle-hand defines the failure set (no peeking).
Tests whether T-L(+0.139)/T-N(+0.107) survive sensor kp input at inference.
Usage: python3 tmp/m3_sensor_kp_arena.py
Out:   results/direction3/m3_sensor_kp_arena.json
"""
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts'); sys.path.insert(0, 'tmp')

from e15_hos_multidraw import (SEAS, build_sea, lam_p, T_MANUAL_50, N_OBS,
                               LEADS, DEPTH, NOISE, R_MIN, FOV_H, z_at, DEVICE)
from d3_prior_net_eval import net_gamma_mmse
from d18_relax_estimator import beamform_top
from e60k_hos_arena import load_model, ARMS


def beamform_kp(Ps, zs, depth):
    """2-stage beamform kp from the sensor points -- IDENTICAL recipe to
    net_gamma_mmse's internal kp_hint=None path (d3_prior_net_eval L71-79).
    Computed ONCE per draw and shared across arms (kp depends only on P,z)."""
    P32, z32 = Ps.astype(np.float32), zs.astype(np.float32)
    bx, by = beamform_top(P32, z32, depth, klim=2.0, n=120)
    kp = max(np.hypot(bx, by), 1e-3)
    bx, by = beamform_top(P32, z32, depth, klim=3.5 * kp, n=120)
    return max(np.hypot(bx, by), 1e-3)

OUT = 'results/direction3/m3_sensor_kp_arena.json'
FROZEN = 'results/direction3/e60k_hos_arena.json'
DRAWS = 5


def draw_sensor(sea, rng, models):
    """one_draw sampling body VERBATIM (identical rng consumption), then eval
    every arm with kp_hint=None (beamformed from the 600 obs). hand not recomputed
    (frozen oracle-hand used for the failure set)."""
    P, zv, rr = [], [], []
    for sc in sea['scans']:
        pw = sc['amp'] * rng.exponential(1.0, len(sc['amp']))
        keep = pw > T_MANUAL_50
        P.append(np.column_stack([sc['xy'][keep], np.full(int(keep.sum()), sc['trel'])]))
        zv.append(sc['z'][keep]); rr.append(sc['r'][keep])
    P, zv, rr = np.vstack(P), np.concatenate(zv), np.concatenate(rr)
    zv = zv + rng.normal(0, NOISE, len(zv))
    edges = np.linspace(R_MIN, rr.max() + 1e-6, 13)
    b = np.clip(np.digitize(rr, edges) - 1, 0, 11)
    take = []
    for j in range(12):
        idx = np.where(b == j)[0]
        k = min(len(idx), max(-(-N_OBS // 12), 1))
        if k:
            take.append(rng.choice(idx, k, replace=False))
    sel = np.concatenate(take)
    Ps, zs = P[sel], zv[sel]
    rf = np.sqrt(rng.uniform((15.0 / 200.0) ** 2, 1.0, 800)) * 200.0
    af = rng.uniform(-np.deg2rad(FOV_H), np.deg2rad(FOV_H), 800)
    qx, qy = rf * np.cos(af), rf * np.sin(af)
    Q, gt = [], []
    for li, l_ in enumerate(LEADS):
        gt.append(z_at(sea['leads'][li], qx, qy))
        Q.append(np.column_stack([qx, qy, np.full(800, l_)]))
    Q, gt = np.vstack(Q), np.concatenate(gt)
    kp_sensor = beamform_kp(Ps, zs, DEPTH)   # <-- SENSOR kp, computed ONCE, shared
    out = {'n_obs': int(len(zs)), 'kp_sensor': float(kp_sensor)}
    for name, (net, head) in models.items():
        pn = np.asarray(net_gamma_mmse(net, Ps, zs, DEPTH, Q, NOISE, seed=0,
                                       kp_hint=kp_sensor,   # shared beamformed kp
                                       use_em=False, smult_head=head))
        out[name] = float(np.corrcoef(pn, gt)[0, 1])
    return out


def main():
    models = {k: load_model(v) for k, v in ARMS.items()}
    # --- runtime self-check: optimized shared-kp must equal per-arm kp_hint=None ---
    si0 = next(i for i, (Tp, Hs) in enumerate(SEAS) if f'Tp{Tp:g}_Hs{Hs:g}' == 'Tp9_Hs4')
    s_chk = build_sea(*SEAS[si0], 0)
    opt = draw_sensor(s_chk, np.random.default_rng(900_000 + 7919 * si0 + 31 * 3), models)
    # slow path (kp_hint=None per arm) on the SAME draw
    r2 = np.random.default_rng(900_000 + 7919 * si0 + 31 * 3)
    P, zv, rr = [], [], []
    for sc in s_chk['scans']:
        pw = sc['amp'] * r2.exponential(1.0, len(sc['amp'])); keep = pw > T_MANUAL_50
        P.append(np.column_stack([sc['xy'][keep], np.full(int(keep.sum()), sc['trel'])]))
        zv.append(sc['z'][keep]); rr.append(sc['r'][keep])
    P, zv, rr = np.vstack(P), np.concatenate(zv), np.concatenate(rr)
    zv = zv + r2.normal(0, NOISE, len(zv))
    edges = np.linspace(R_MIN, rr.max() + 1e-6, 13); b = np.clip(np.digitize(rr, edges) - 1, 0, 11)
    take = [r2.choice(np.where(b == j)[0], min((b == j).sum(), max(-(-N_OBS // 12), 1)), False)
            for j in range(12) if (b == j).sum()]
    sel = np.concatenate(take); Ps, zs = P[sel], zv[sel]
    rf = np.sqrt(r2.uniform((15.0/200.0)**2, 1.0, 800))*200.0; af = r2.uniform(-np.deg2rad(FOV_H), np.deg2rad(FOV_H), 800)
    qx, qy = rf*np.cos(af), rf*np.sin(af)
    Qc = np.vstack([np.column_stack([qx, qy, np.full(800, l_)]) for l_ in LEADS])
    gtc = np.concatenate([z_at(s_chk['leads'][li], qx, qy) for li in range(len(LEADS))])
    maxd = 0.0
    for name, (net, head) in models.items():
        pn = np.asarray(net_gamma_mmse(net, Ps, zs, DEPTH, Qc, NOISE, seed=0,
                                       kp_hint=None, use_em=False, smult_head=head))
        maxd = max(maxd, abs(float(np.corrcoef(pn, gtc)[0, 1]) - opt[name]))
    print(f"SELF-CHECK opt(shared-kp) vs slow(per-arm None): max|d| {maxd:.2e} "
          f"-> {'OK' if maxd < 1e-9 else 'MISMATCH-ABORT'}", flush=True)
    if maxd >= 1e-9:
        print("*** ABORT: optimization not bit-identical"); sys.exit(1)
    frozen = {(r['sea'], r['draw']): r for r in json.load(open(FROZEN))['rows']}
    # only failure-window draws (frozen oracle-hand < 0.5) are needed for the verdict
    fail_draws = {(r['sea'], r['draw']) for r in frozen.values() if r['hand'] < 0.5}
    print(f"failure-window draws to eval (sensor kp): {len(fail_draws)} "
          f"across {len({s for s, _ in fail_draws})} seas", flush=True)
    rows = []
    if os.path.exists(OUT):
        rows = json.load(open(OUT)).get('rows', [])
    done = {(r['sea'], r['draw']) for r in rows}
    t0 = time.time()
    for si, (Tp, Hs) in enumerate(SEAS):
        tag = f'Tp{Tp:g}_Hs{Hs:g}'
        draws_here = [r for r in range(DRAWS) if (tag, r) in fail_draws]
        if not draws_here or all((tag, r) in done for r in draws_here):
            continue
        sea = build_sea(Tp, Hs, 0)
        lam = lam_p(Tp)
        for r in draws_here:
            if (tag, r) in done:
                continue
            res = draw_sensor(sea, np.random.default_rng(900_000 + 7919 * si + 31 * r), models)
            fr = frozen[(tag, r)]
            res.update(sea=tag, Tp=float(Tp), Hs=float(Hs), lam_p=lam, draw=r,
                       hand=fr['hand'])   # frozen oracle-hand -> failure set unchanged
            rows.append(res)
        json.dump(dict(rows=rows, note='eval-side sensor kp (kp_hint=None); frozen hand'),
                  open(OUT, 'w'), indent=1, default=float)
        print(f"  {tag:12s} done [{time.time()-t0:.0f}s]", flush=True)

    # ---- verdict on frozen failure set ----
    analyze(rows)


def wild_p(vals, B=4000, seed=0):
    v = np.array(vals); m = v.mean()
    rng = np.random.default_rng(seed)
    t0 = m / (v.std(ddof=1) / np.sqrt(len(v)) + 1e-12)
    cnt = 0
    for _ in range(B):
        w = rng.choice([-1, 1], len(v))
        vs = v * w  # sign-flip wild (mean-0 null)
        ts = vs.mean() / (vs.std(ddof=1) / np.sqrt(len(v)) + 1e-12)
        if ts >= t0:
            cnt += 1
    return (cnt + 1) / (B + 1)


def analyze(rows):
    from scipy import stats
    # failure window = hand<0.5, cluster by sea
    fw = [r for r in rows if r['hand'] < 0.5]
    seas = {}
    for r in fw:
        seas.setdefault(r['sea'], []).append(r)
    def arm(r, a):
        return np.mean([r[f'{a}_s{s}'] for s in (0, 1, 2)])
    tl, tn = [], []
    for s, rs in seas.items():
        tl.append(np.mean([arm(r, 'T') - arm(r, 'L') for r in rs]))
        tn.append(np.mean([arm(r, 'T') - arm(r, 'N') for r in rs]))
    tl, tn = np.array(tl), np.array(tn)
    k = len(tl)
    print(f"\n=== failure window (hand<0.5, k={k} seas) SENSOR kp ===")
    for lab, v in [('T-L', tl), ('T-N', tn)]:
        t = v.mean() / (v.std(ddof=1) / np.sqrt(k))
        p1 = stats.t.sf(t, k - 1)
        wp = wild_p(v)
        npos = int((v > 0).sum())
        print(f"  {lab}: mean {v.mean():+.4f}  t({k-1}) {t:+.2f}  p_t {p1:.4f}  "
              f"wild {wp:.4f}  {npos}/{k}")
    print("  (compare frozen ORACLE: T-L +0.1389 wild .0022 10/11 | T-N +0.1068 wild .0048 8/11)")
    json.dump(dict(k=int(k), TL=list(map(float, tl)), TN=list(map(float, tn)),
                   TL_mean=float(tl.mean()), TN_mean=float(tn.mean())),
              open('results/direction3/m3_sensor_kp_verdict.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
