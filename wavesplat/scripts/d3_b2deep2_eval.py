"""B-2 DEEP-2 -- mechanism decomposition, UQ duel, seed-3, HOS generalization.

Pre-registered in docs/B1B2_DEEP_prereg_2026-07-12.md ("B-2 심화2"). Four
sub-tasks, all reusing the frozen 60-window judgment (d3_b2deep_eval pattern:
window_data + net_gamma_mmse(use_em=False, smult_head, kp_hint)) and the ckpts
results/direction3/b2deep_{T,L}_s{0,1,2}.pt ('model'+'head').

  --task decomp_uq  (1) gamma/sigma decomposition: 6 arms on the 60 windows --
                    {T-net,L-net} x {own head, other's head, fixed smult=3.25}.
                    (2) UQ duel: T-s0 vs L-s0 posterior variance -> pooled
                    z-score / cov95. Gate G-B2D2-UQ: |zstd(T)-1| < |zstd(L)-1|.
  --task seed3      T-s2/L-s2 60-window corr -> G-B2D-1 direction 3/3 (s0,s1
                    reused from b2deep_eval.json).
  --task hos        T-s0 vs L-s0 on HOS M=3, 8 cells (seed{0,1,2,3} x Hs{1.2,3})
                    whole-field corr (A-1 machinery, arms swapped to T/L nets).

Writes/merges results/direction3/b2deep2_eval.json (one key per task).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from d3_falcon_real_operator import RECORDS
from d3_sa_multirecord import MRecord
from d3_prior_net_eval import net_gamma_mmse, DEVICE
from d3_prior_net_train import PriorNet
from d3_l1_s2head_train import SmultHead
from d3_gate_promotion_eval import window_data, RECS

OUT = 'results/direction3/b2deep2_eval.json'
REF = 'results/direction3/b2deep_eval.json'
FIXED_SMULT = 3.25
CKPT = {'T_s0': 'results/direction3/b2deep_T_s0.pt',
        'L_s0': 'results/direction3/b2deep_L_s0.pt',
        'T_s1': 'results/direction3/b2deep_T_s1.pt',
        'L_s1': 'results/direction3/b2deep_L_s1.pt',
        'T_s2': 'results/direction3/b2deep_T_s2.pt',
        'L_s2': 'results/direction3/b2deep_L_s2.pt'}


def load_model(ckpt):
    ck = torch.load(ckpt, map_location=DEVICE)
    net = PriorNet().to(DEVICE)
    net.load_state_dict(ck['model'])
    net.eval()
    head = SmultHead().to(DEVICE)
    head.load_state_dict(ck['head'])
    head.eval()
    return net, head, ck.get('step'), ck.get('val')


def merge_out(key, payload):
    doc = {}
    if os.path.exists(OUT):
        doc = json.load(open(OUT))
    doc[key] = payload
    json.dump(doc, open(OUT, 'w'), indent=1)


def iter_windows():
    """Yield (name, wi, i0, P, z, Pq, gt_q, leads, Hs, kp, v) for the 60."""
    for name in RECS:
        cfg = RECORDS[name]
        rec = MRecord(cfg['path'], stride=cfg['stride'])
        nfw = int(round(40.0 / rec.dt))
        n_lead = int(round(cfg['lead_max'] / rec.dt))
        i0s = np.linspace(nfw + 1, rec.T_all - n_lead - 2, 12).astype(int)
        for wi, i0 in enumerate(i0s):
            P, z, Pq, gt_q, leads, Hs, kp, _ = window_data(
                rec, cfg, int(i0), nfw, n_lead, wi)
            v = np.isfinite(gt_q) & (leads <= 10.0)
            if v.sum() < 5:
                continue
            yield name, wi, int(i0), cfg, P, z, Pq, gt_q, leads, Hs, kp, v


# ----------------------------------------------------------------------------
def task_decomp_uq():
    ref = {(r['record'], r['wi']): r for r in json.load(open(REF))['rows']}
    Tn, Th, _, _ = load_model(CKPT['T_s0'])
    Ln, Lh, _, _ = load_model(CKPT['L_s0'])
    rows = []
    zT_all, zL_all = [], []           # pooled standardized residuals
    for (name, wi, i0, cfg, P, z, Pq, gt_q, leads, Hs, kp,
         v) in iter_windows():
        t0 = time.time()
        sig0 = 0.05 * Hs
        d = cfg['depth']

        def call(net, head=None, fixed=None, var=False):
            return net_gamma_mmse(net, P, z, d, Pq, sig0, seed=i0,
                                  kp_hint=kp, use_em=False, smult_head=head,
                                  smult_fixed=fixed, return_var=var)

        fT_own, vT = call(Tn, head=Th, var=True)
        fL_own, vL = call(Ln, head=Lh, var=True)
        fT_Lhead = call(Tn, head=Lh)
        fL_Thead = call(Ln, head=Th)
        fT_fix = call(Tn, fixed=FIXED_SMULT)
        fL_fix = call(Ln, fixed=FIXED_SMULT)

        def c(fq):
            return float(np.corrcoef(fq[v], gt_q[v])[0, 1])

        r = ref.get((name, wi), {})
        row = {'record': name, 'wi': wi, 'hand': r.get('hand', float('nan')),
               'T_own': c(fT_own), 'L_own': c(fL_own),
               'T_Lhead': c(fT_Lhead), 'L_Thead': c(fL_Thead),
               'T_fix': c(fT_fix), 'L_fix': c(fL_fix)}
        rows.append(row)
        # UQ: standardized residual at valid query points
        zT = (fT_own[v] - gt_q[v]) / np.sqrt(vT[v] + 1e-12)
        zL = (fL_own[v] - gt_q[v]) / np.sqrt(vL[v] + 1e-12)
        zT_all.append(zT)
        zL_all.append(zL)
        print(f"[{name} w{wi}] hand {row['hand']:+.2f} | "
              f"Town {row['T_own']:+.2f} Tfix {row['T_fix']:+.2f} "
              f"TLh {row['T_Lhead']:+.2f} | Lown {row['L_own']:+.2f} "
              f"Lfix {row['L_fix']:+.2f} LTh {row['L_Thead']:+.2f} "
              f"[{time.time()-t0:.0f}s]", flush=True)

    h = np.array([r['hand'] for r in rows])
    bad = h < 0.5

    def m(k, sub=None):
        a = np.array([r[k] for r in rows])
        return float(a[sub].mean()) if sub is not None else float(a.mean())

    keys = ('T_own', 'L_own', 'T_fix', 'L_fix', 'T_Lhead', 'L_Thead')
    means_all = {k: m(k) for k in keys}
    means_bad = {k: m(k, bad) for k in keys}

    # --- gamma/sigma decomposition on failure windows (the interesting set) ---
    def decomp(sub):
        total = m('T_own', sub) - m('L_own', sub)          # full T advantage
        gamma = m('T_fix', sub) - m('L_fix', sub)          # floors equal->prior
        sigma = total - gamma                              # residual = floor
        # within-net floor swaps (does the OTHER floor move each net?)
        T_floor_eff = m('T_own', sub) - m('T_Lhead', sub)  # T gets L's floor
        L_floor_eff = m('L_Thead', sub) - m('L_own', sub)  # L gets T's floor
        return {'total_adv': total, 'gamma_share': gamma, 'sigma_share': sigma,
                'gamma_frac': float(gamma / total) if abs(total) > 1e-9
                else float('nan'),
                'T_own_minus_TLhead': T_floor_eff,
                'L_Thead_minus_L_own': L_floor_eff}

    dec_bad = decomp(bad)
    dec_all = decomp(slice(None))

    # --- UQ ---
    zT = np.concatenate(zT_all)
    zL = np.concatenate(zL_all)

    def uq(z):
        return {'zstd': float(z.std()), 'zmean': float(z.mean()),
                'cov95': float((np.abs(z) <= 1.96).mean()),
                'cov68': float((np.abs(z) <= 1.0).mean()), 'n': int(len(z))}

    uqT, uqL = uq(zT), uq(zL)
    gate_uq = abs(uqT['zstd'] - 1.0) < abs(uqL['zstd'] - 1.0)

    payload = {
        'n': len(rows), 'n_bad': int(bad.sum()),
        'means_all': means_all, 'means_bad': means_bad,
        'decomp_bad': dec_bad, 'decomp_all': dec_all,
        'UQ': {'T_s0': uqT, 'L_s0': uqL,
               'G_B2D2_UQ_pass': bool(gate_uq),
               'absdev_T': abs(uqT['zstd'] - 1.0),
               'absdev_L': abs(uqL['zstd'] - 1.0)},
        'rows': rows,
    }
    merge_out('decomp_uq', payload)

    print("\n===== gamma/sigma decomposition (failure windows hand<0.5) =====")
    print(f"  total T-L advantage : {dec_bad['total_adv']:+.3f}")
    print(f"  gamma share (fixed floor 3.25): {dec_bad['gamma_share']:+.3f} "
          f"({dec_bad['gamma_frac']*100:.0f}% of total)")
    print(f"  sigma share (floor) : {dec_bad['sigma_share']:+.3f}")
    print(f"  T_own - T(Lhead)    : {dec_bad['T_own_minus_TLhead']:+.3f}")
    print(f"  L(Thead) - L_own    : {dec_bad['L_Thead_minus_L_own']:+.3f}")
    print("\n===== UQ (pooled query pts, leads<=10) =====")
    print(f"  T_s0 zstd {uqT['zstd']:.3f} cov95 {uqT['cov95']:.3f} "
          f"cov68 {uqT['cov68']:.3f} (n {uqT['n']})")
    print(f"  L_s0 zstd {uqL['zstd']:.3f} cov95 {uqL['cov95']:.3f} "
          f"cov68 {uqL['cov68']:.3f}")
    print(f"  G-B2D2-UQ {'PASS' if gate_uq else 'FAIL'} "
          f"(|zstd_T-1| {abs(uqT['zstd']-1):.3f} < "
          f"|zstd_L-1| {abs(uqL['zstd']-1):.3f})")
    print(f"wrote {OUT}", flush=True)


# ----------------------------------------------------------------------------
def task_seed3():
    ref = {(r['record'], r['wi']): r for r in json.load(open(REF))['rows']}
    Tn, Th, Ts, Tv = load_model(CKPT['T_s2'])
    Ln, Lh, Ls, Lv = load_model(CKPT['L_s2'])
    print(f"T_s2 best step {Ts} val {Tv:.4f} | "
          f"L_s2 best step {Ls} val {Lv:.4f}", flush=True)
    rows = []
    for (name, wi, i0, cfg, P, z, Pq, gt_q, leads, Hs, kp,
         v) in iter_windows():
        t0 = time.time()
        sig0 = 0.05 * Hs
        d = cfg['depth']
        diagT, diagL = {}, {}
        fT = net_gamma_mmse(Tn, P, z, d, Pq, sig0, seed=i0, kp_hint=kp,
                            use_em=False, smult_head=Th, diag_out=diagT)
        fL = net_gamma_mmse(Ln, P, z, d, Pq, sig0, seed=i0, kp_hint=kp,
                            use_em=False, smult_head=Lh, diag_out=diagL)
        r = ref.get((name, wi), {})
        rows.append({'record': name, 'wi': wi, 'hand': r.get('hand'),
                     'T_s2': float(np.corrcoef(fT[v], gt_q[v])[0, 1]),
                     'L_s2': float(np.corrcoef(fL[v], gt_q[v])[0, 1]),
                     'T_s2_smult': float(diagT.get('smult', float('nan'))),
                     'L_s2_smult': float(diagL.get('smult', float('nan')))})
        print(f"[{name} w{wi}] hand {rows[-1]['hand']:+.2f} "
              f"Ts2 {rows[-1]['T_s2']:+.2f} Ls2 {rows[-1]['L_s2']:+.2f} "
              f"[{time.time()-t0:.0f}s]", flush=True)

    h = np.array([r['hand'] for r in rows])
    bad = h < 0.5
    dT = np.array([r['T_s2'] for r in rows])
    dL = np.array([r['L_s2'] for r in rows])
    d = dT - dL
    m2_bad = float(d[bad].mean())
    # reuse s0,s1 bad-window means from the reference summary
    rs = json.load(open(REF))['summary']['G_B2D_1']
    m0, m1 = rs['seed0_mean_d'], rs['seed1_mean_d']
    dirs = [m0 > 0, m1 > 0, m2_bad > 0]
    payload = {
        'n': len(rows), 'n_bad': int(bad.sum()),
        'T_s2_all': float(dT.mean()), 'L_s2_all': float(dL.mean()),
        'T_s2_bad': float(dT[bad].mean()), 'L_s2_bad': float(dL[bad].mean()),
        'd_bad_seed2': m2_bad, 'd_bad_seed0': m0, 'd_bad_seed1': m1,
        'direction_3of3': bool(all(dirs)),
        'smult_T_s2': [float(np.nanmean([r['T_s2_smult'] for r in rows])),
                       float(np.nanstd([r['T_s2_smult'] for r in rows]))],
        'smult_L_s2': [float(np.nanmean([r['L_s2_smult'] for r in rows])),
                       float(np.nanstd([r['L_s2_smult'] for r in rows]))],
        'ckpt_step_val': {'T_s2': [Ts, Tv], 'L_s2': [Ls, Lv]},
        'rows': rows,
    }
    merge_out('seed3', payload)
    print(f"\nseed2 bad-window d(T-L) {m2_bad:+.3f}  "
          f"(s0 {m0:+.3f}, s1 {m1:+.3f})  "
          f"direction 3/3 {'YES' if all(dirs) else 'NO'}")
    print(f"wrote {OUT}", flush=True)


# ----------------------------------------------------------------------------
def task_hos():
    from d3_h7_hos_arena import (surface_snap, scan_hits, z_at, SCAN_REL,
                                 SPINUP, NOISE)
    from d3_f1_hos_door1 import hos_run, sea_components, DT
    from d3_door1b_field_bias import R_MIN, FOV_H
    SEEDS = [0, 1, 2, 3]
    HS_LIST = [1.2, 3.0]
    LEADS = [0.0, 2.0, 5.0, 8.0]
    Tn, Th, _, _ = load_model(CKPT['T_s0'])
    Ln, Lh, _, _ = load_model(CKPT['L_s0'])

    def run_cell(seed, Hs, rng):
        sim = hos_run(seed, 3, Hs=Hs)
        while sim.t < SPINUP - 1e-9:
            sim.step(min(DT, SPINUP - sim.t))
        P, zv, rr = [], [], []
        t_end = SPINUP + 20.0
        for trel in SCAN_REL:
            tgt = t_end + trel
            while sim.t < tgt - 1e-9:
                sim.step(min(DT, tgt - sim.t))
            snap = surface_snap(sim)
            xy, z, r = scan_hits(snap, rng)
            P.append(np.column_stack([xy, np.full(len(z), trel)]))
            zv.append(z)
            rr.append(r)
        P, zv, rr = np.vstack(P), np.concatenate(zv), np.concatenate(rr)
        zv = zv + rng.normal(0, NOISE, len(zv))
        edges = np.linspace(R_MIN, rr.max() + 1e-6, 13)
        b = np.clip(np.digitize(rr, edges) - 1, 0, 11)
        take = []
        for j in range(12):
            idx = np.where(b == j)[0]
            k = min(len(idx), 167)
            if k:
                take.append(rng.choice(idx, k, replace=False))
        sel = np.concatenate(take)
        Ps, zs = P[sel], zv[sel]
        rf = np.sqrt(rng.uniform((15.0 / 200.0) ** 2, 1.0, 800)) * 200.0
        af = rng.uniform(-np.deg2rad(FOV_H), np.deg2rad(FOV_H), 800)
        qx, qy = rf * np.cos(af), rf * np.sin(af)
        Q, gt = [], []
        for l_ in LEADS:
            tgt = t_end + l_
            while sim.t < tgt - 1e-9:
                sim.step(min(DT, tgt - sim.t))
            gt.append(z_at(surface_snap(sim), qx, qy))
            Q.append(np.column_stack([qx, qy, np.full(800, l_)]))
        Q, gt = np.vstack(Q), np.concatenate(gt)
        S = sea_components(seed, Hs=Hs)
        kp = float(np.hypot(S['kx'], S['ky'])[np.argmax(S['A'])])
        fT = np.asarray(net_gamma_mmse(Tn, Ps, zs, 50.0, Q, NOISE, seed=seed,
                                       kp_hint=kp, use_em=False, smult_head=Th))
        fL = np.asarray(net_gamma_mmse(Ln, Ps, zs, 50.0, Q, NOISE, seed=seed,
                                       kp_hint=kp, use_em=False, smult_head=Lh))
        row = {'seed': seed, 'Hs': Hs, 'n_obs': int(len(zs)),
               'T': float(np.corrcoef(fT, gt)[0, 1]),
               'L': float(np.corrcoef(fL, gt)[0, 1])}
        for li, l_ in enumerate(LEADS):
            s_ = slice(800 * li, 800 * (li + 1))
            row[f'T_l{int(l_)}'] = float(np.corrcoef(fT[s_], gt[s_])[0, 1])
            row[f'L_l{int(l_)}'] = float(np.corrcoef(fL[s_], gt[s_])[0, 1])
        return row

    # resumable + per-cell independent rng (each cell reproducible in any
    # order; T and L share the same obs/queries within a cell -> fair duel).
    done = {}
    if os.path.exists(OUT):
        prev = json.load(open(OUT)).get('hos', {})
        done = {(r['seed'], r['Hs']): r for r in prev.get('rows', [])}
    rows = []
    for Hs in HS_LIST:
        for seed in SEEDS:
            if (seed, Hs) in done:
                rows.append(done[(seed, Hs)])
                print(f"Hs {Hs} seed {seed} [cached] "
                      f"T {done[(seed,Hs)]['T']:+.3f} "
                      f"L {done[(seed,Hs)]['L']:+.3f}", flush=True)
                continue
            t0 = time.time()
            cell_rng = np.random.default_rng(7000 + 10 * seed + int(Hs * 10))
            r = run_cell(seed, Hs, cell_rng)
            rows.append(r)
            print(f"Hs {Hs} seed {seed} [{time.time()-t0:5.0f}s] "
                  f"T {r['T']:+.3f} L {r['L']:+.3f} "
                  f"d {r['T']-r['L']:+.3f}", flush=True)
            merge_out('hos', {'rows': rows, 'complete': False})
    d = np.array([r['T'] - r['L'] for r in rows])
    per_hs = {}
    for Hs in HS_LIST:
        dh = np.array([r['T'] - r['L'] for r in rows if r['Hs'] == Hs])
        per_hs[str(Hs)] = {'mean_d': float(dh.mean()),
                           'sign': f"{int((dh > 0).sum())}/{len(dh)}"}
    ok = d.mean() >= 0 and all(
        np.mean([r['T'] - r['L'] for r in rows if r['Hs'] == h]) >= 0
        for h in HS_LIST)
    payload = {
        'pooled_d': float(d.mean()), 'cells_gt0': f"{int((d > 0).sum())}/{len(d)}",
        'T_mean': float(np.mean([r['T'] for r in rows])),
        'L_mean': float(np.mean([r['L'] for r in rows])),
        'per_Hs': per_hs, 'direction_pass': bool(ok), 'rows': rows,
    }
    merge_out('hos', payload)
    print(f"\nHOS T vs L: pooled d {d.mean():+.4f} cells>0 "
          f"{int((d > 0).sum())}/{len(d)}  T {payload['T_mean']:+.3f} "
          f"L {payload['L_mean']:+.3f}  direction {'PASS' if ok else 'FAIL'}")
    print(f"wrote {OUT}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True,
                    choices=['decomp_uq', 'seed3', 'hos'])
    a = ap.parse_args()
    {'decomp_uq': task_decomp_uq, 'seed3': task_seed3,
     'hos': task_hos}[a.task]()
    return 0


if __name__ == '__main__':
    sys.exit(main())
