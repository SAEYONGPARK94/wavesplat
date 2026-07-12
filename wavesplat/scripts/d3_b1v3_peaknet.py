"""B-1 v3 -- operator-matched training data + matched-filter recovery.
Frozen prereg: docs/B1B2_DEEP_prereg_2026-07-12.md ("B-1 v3" section).

Why v3 (from the B-1 심화 3-way autopsy):
  B-1 심화 trained the peak identifier on ANALYTIC scenes (clean single wave +
  2 cm white noise). It FAILED on BF2 (1/5) because the real BF2 collapse is
  driven by OPERATOR CLUTTER outside that generative model: the residual of the
  stored LiDAR z vs the true single wave has std 1.5-3.4x (measured here 3.2-4.8x)
  the 2 cm noise -- COMPARABLE to the wave amplitude itself -- so the dominant
  beamform lobe is made by structured (partially dispersion-coherent) clutter,
  not the sampling PSF + white noise. The independent recovery diagnostic showed
  the single-component MATCHED FILTER recovers 3/5 at oracle-k (0.97-1.00), while
  the fine-bank path was capped at 1/5.

v3 = two replacements (design FROZEN):
  1. TRAIN DATA -> wavesim operator-passed distribution.  Scenes are built from
     the REAL LiDAR observations of NON-BF2 1comp fixedT episodes (BF3/BF4, seeds
     0-4 = 10 episodes), decomposed per point into true-wave signal s and the
     measured operator clutter c = z - s.  A training scene is
         z' = a*s + b*c + 2cm,   a~logU(0.25,2.0),  b~logU(0.35,1.3) (or 0),
     RIGIDLY ROTATED by a random dtheta (exact: k'.x' = k.x under rotation, so
     the same per-point (s,c) values are a valid plane-wave-plus-clutter scene at
     the rotated angle) -> full angle coverage with the real, aligned operator
     clutter.  a spans the buried regime (a small, b large -> SNR~1, the BF2 case
     that analytic white noise cannot reproduce).  A monochromaticity confidence
     head is trained with multi-component (BF3/BF4 5comp+15comp) scenes as conf=0
     negatives so the fire gate can fire on a BURIED-but-1comp map while rejecting
     broadband.  NO BF2 data touched (seed->angle is shared across BF levels, so
     BF3/BF4 already covers BF2's angles legitimately).
  2. RECOVERY -> single-component matched filter (the 심화 oracle 3/5 path):
     learned k_hat -> a7v2 continuous-k refine -> single (cos,sin) LS on ALL obs
     -> forecast at the query points.  Replaces the fine-bank solve.

Gates (FROZEN, information-bearing seed scope s0,s1,s4; s2,s3 = information limit,
prereg-excluded, oracle-MF confirmed here -0.13 / -1.00):
  G-B1v3-1 identification : |dk|/k_true <= 0.05 in >= 2/3 of {s0,s1,s4}.
  G-B1v3-2 recovery       : matched-filter field corr(lead0) & encounter rho
                            >= 0.5 in >= 2/3 of {s0,s1,s4}. (s2,s3 measured, report.)
  G-B1v3-3 harmless       : confidence gate -> 6 synth dfield >= -0.01 AND YS17
                            12 windows dencounter >= -0.02.

python3 scripts/d3_b1v3_peaknet.py               # train (if no ckpt) + eval
python3 scripts/d3_b1v3_peaknet.py --smoke       # tiny pool/steps, skip real
python3 scripts/d3_b1v3_peaknet.py --eval_only   # load ckpt, run gates
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from d3_gpu_mmse import mmse_gpu, DEV, G
from d3_a7_finebank import (load_synth, build_queries, field_track_corr,
                            hand_predict, APP_EPISODES, LEADS)
# reuse the B-1 심화 model + low-level beamform/identify plumbing verbatim
from d3_b1deep_peaknet import (power_map, to_input, argmax_nk, PeakNet,
                               true_cell_index, n_params, converged,
                               peaknet_identify, single_forecast, calibrate_conf,
                               NGRID, B_HI, NOISE, CONF_TOL, _KXn, _KYn, _INBAND)

OUT = 'results/direction3/b1v3_eval.json'
CKPT = 'results/direction3/b1v3_peaknet.pt'
DATE = '2026-07-12'
K_LO, K_HI = 0.05, 1.2         # (reported; radius stays ~1 for 1comp regime)
SIG_LO, SIG_HI = 0.25, 2.0     # signal-amplitude scale a (low-SNR emphasis)
CLU_LO, CLU_HI = 0.35, 1.3     # clutter-amplitude scale b
CLEAN_FRAC = 0.15              # fraction of 1comp scenes with b=0 (ideal PSF anchor)
P_MULTI = 0.20                 # fraction of scenes that are multi-comp (conf=0)
N_TRAIN = 40000                # >= 30k (frozen)
N_VAL = 4000
STEPS = 20000                  # >= 20k (frozen); extended if not converged
BATCH = 128
LR = 1.5e-3
WIN_STARTS = list(range(0, 91, 10))   # window-start sweep (frames)

# NON-BF2 pools (identification: 1comp; confidence negatives: multi-comp)
POOL_1COMP = ['data/episode_BF3_1comp_seed%d_fixedT.h5' % s for s in range(5)] \
    + ['data/episode_BF4_1comp_seed%d_fixedT.h5' % s for s in range(5)]
POOL_MULTI = ['data/episode_BF3_5comp_seed%d_fixedT.h5' % s for s in range(5)] \
    + ['data/episode_BF4_5comp_seed%d_fixedT.h5' % s for s in range(5)] \
    + ['data/episode_BF4_15comp_seed%d_fixedT.h5' % s for s in range(5)]

SCOPE = [0, 1, 4]              # information-bearing seeds (prereg; oracle-confirmed)


# ============================================================ operator scenes
def load_window(path, start, rng, multi=False):
    """Stratified LiDAR sample of ONE window (frames start..start+200) of a
    fixedT episode. For 1comp: also decompose z = true-wave s + operator clutter c."""
    with h5py.File(path, 'r') as f:
        comp = f['config/wave_components'][:]
        depth = float(f['config'].attrs['depth'])
        ts = f['frames/timestamps'][:]
        pts = f['frames/lidar_points']
        nfr = len(ts)
        ai = int(np.argmax(comp[:, 0]))
        kp = float(np.hypot(comp[ai, 2], comp[ai, 3]))
        end = min(start + 200, nfr - 1)
        obs_i = np.linspace(start, end, 12).astype(int)
        P, zv = [], []
        for i in obs_i:
            p = pts[i][:]
            P.append(np.column_stack([p[:, 0], p[:, 1], np.full(len(p), ts[i])]))
            zv.append(p[:, 2])
        P = np.vstack(P).astype(np.float64)
        zv = np.concatenate(zv).astype(np.float64)
    rr = np.hypot(P[:, 0], P[:, 1])
    edges = np.linspace(rr.min(), rr.max() + 1e-6, 13)
    b = np.clip(np.digitize(rr, edges) - 1, 0, 11)
    take = []
    for j in range(12):
        idx = np.where(b == j)[0]
        k = min(len(idx), 167)
        if k:
            take.append(rng.choice(idx, k, replace=False))
    sel = np.concatenate(take)
    P, zv = P[sel], zv[sel]
    out = dict(P=P.astype(np.float32), z=zv.astype(np.float32),
               kp=kp, depth=depth)
    if not multi:
        A, kx, ky, phi = (float(comp[ai, 0]), float(comp[ai, 2]),
                          float(comp[ai, 3]), float(comp[ai, 4]))
        k = math.hypot(kx, ky)
        om = math.sqrt(G * k * math.tanh(k * depth) + 1e-12)
        s = A * np.cos(kx * P[:, 0] + ky * P[:, 1] - om * P[:, 2] + phi)
        out['s'] = s.astype(np.float32)
        out['c'] = (zv - s).astype(np.float32)
        out['nk'] = (np.array([kx, ky], np.float32) / kp)
    return out


def build_bases(paths, starts, multi, log=print):
    """Precompute base windows -> GPU tensors (P,z[,s,c,nk],kp,depth)."""
    rng = np.random.default_rng(2026)
    bases = []
    for p in paths:
        for st in starts:
            d = load_window(p, st, rng, multi=multi)
            b = dict(P=torch.tensor(d['P'], device=DEV),
                     z=torch.tensor(d['z'], device=DEV),
                     kp=d['kp'], depth=d['depth'])
            if not multi:
                b['s'] = torch.tensor(d['s'], device=DEV)
                b['c'] = torch.tensor(d['c'], device=DEV)
                b['nk'] = torch.tensor(d['nk'], device=DEV)
            bases.append(b)
    log(f"  built {len(bases)} {'multi' if multi else '1comp'} base windows "
        f"({len(paths)} eps x {len(starts)} starts)")
    return bases


def _rotate_xy(P, dth):
    c, s = math.cos(dth), math.sin(dth)
    x, y = P[:, 0], P[:, 1]
    return torch.stack([c * x - s * y, s * x + c * y, P[:, 2]], 1)


def gen_op_scene(bases_1c, bases_mc, rng):
    """One operator-passed scene (GPU). Returns (input(2,n,n), nk(2,), conf, reg)."""
    dth = float(rng.uniform(0, 2 * math.pi))
    if rng.random() < P_MULTI:                       # multi-comp -> conf=0
        b = bases_mc[rng.integers(len(bases_mc))]
        amp = float(math.exp(rng.uniform(math.log(0.4), math.log(1.6))))
        z = amp * b['z'] + NOISE * torch.randn(len(b['z']), device=DEV)
        Prot = _rotate_xy(b['P'], dth)
        pw = power_map(Prot, z, b['kp'], b['depth'])
        return to_input(pw), torch.zeros(2, device=DEV), 0.0, 0.0
    # 1comp -> conf=1
    b = bases_1c[rng.integers(len(bases_1c))]
    a = float(math.exp(rng.uniform(math.log(SIG_LO), math.log(SIG_HI))))
    bb = 0.0 if rng.random() < CLEAN_FRAC else \
        float(math.exp(rng.uniform(math.log(CLU_LO), math.log(CLU_HI))))
    z = a * b['s'] + bb * b['c'] + NOISE * torch.randn(len(b['s']), device=DEV)
    Prot = _rotate_xy(b['P'], dth)
    pw = power_map(Prot, z, b['kp'], b['depth'])
    c, s = math.cos(dth), math.sin(dth)
    nk0 = b['nk']
    nk = torch.stack([c * nk0[0] - s * nk0[1], s * nk0[0] + c * nk0[1]])
    return to_input(pw), nk, 1.0, 1.0


def build_dataset(bases_1c, bases_mc, n, seed, tag):
    rng = np.random.default_rng(seed)
    X = torch.empty((n, 2, NGRID, NGRID), dtype=torch.float16)
    Y = torch.empty((n, 2), dtype=torch.float32)
    C = torch.empty((n,), dtype=torch.float32)
    Rg = torch.empty((n,), dtype=torch.float32)
    t0 = time.time()
    n_multi = 0
    for i in range(n):
        inp, nk, conf, reg = gen_op_scene(bases_1c, bases_mc, rng)
        X[i] = inp.half().cpu()
        Y[i] = nk.cpu()
        C[i] = conf
        Rg[i] = reg
        n_multi += (reg < 0.5)
        if (i + 1) % 10000 == 0:
            print(f"  [{tag}] {i+1}/{n}  ({(time.time()-t0):.0f}s, "
                  f"multi {int(n_multi)}/{i+1})", flush=True)
    return X, Y, C, Rg, dict(n_multi=int(n_multi), n_1comp=int(n - n_multi))


# ==================================================================== train
def train(bases_1c, bases_mc, steps=STEPS, log=print):
    log(f"building train ({N_TRAIN}) + val ({N_VAL}) operator-passed scenes ...")
    Xtr, Ytr, Ctr, Rtr, comp_tr = build_dataset(bases_1c, bases_mc, N_TRAIN,
                                                 100, 'train')
    Xva, Yva, Cva, Rva, comp_va = build_dataset(bases_1c, bases_mc, N_VAL,
                                                 999, 'val')
    Yva_d, Rva_d = Yva.to(DEV), Rva.to(DEV)
    net = PeakNet().to(DEV)
    log(f"PeakNet params = {n_params(net):,} (<=1M gate "
        f"{'OK' if n_params(net) <= 1e6 else 'FAIL'})")
    log(f"train composition: 1comp {comp_tr['n_1comp']}  multi {comp_tr['n_multi']}")
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    rng = np.random.default_rng(7)

    def evaluate():
        net.eval()
        errs, confok = [], []
        with torch.no_grad():
            for i in range(0, N_VAL, 512):
                xb = Xva[i:i + 512].float().to(DEV)
                pnk, _, conf = net(xb)
                rm = Rva_d[i:i + 512] > 0.5              # id metrics on 1comp only
                e = torch.hypot(pnk[:, 0] - Yva_d[i:i + 512, 0],
                                pnk[:, 1] - Yva_d[i:i + 512, 1])[rm]
                errs.append(e.cpu())
                confok.append(((conf > 0) == (Cva[i:i + 512].to(DEV) > 0.5)).cpu())
        net.train()
        errs = torch.cat(errs)
        return dict(med_dnk=float(errs.median()),
                    id05=float((errs <= 0.05).float().mean()),
                    id10=float((errs <= 0.10).float().mean()),
                    conf_acc=float(torch.cat(confok).float().mean()))

    curve, best, t0 = [], None, time.time()
    for s in range(1, steps + 1):
        idx = rng.integers(0, N_TRAIN, BATCH)
        xb = Xtr[idx].float().to(DEV)
        yb = Ytr[idx].to(DEV)
        cb = Ctr[idx].to(DEV)
        rb = Rtr[idx].to(DEV)                            # reg-active mask
        pnk, logit, conf = net(xb)
        tcell = true_cell_index(yb)
        denom = rb.sum().clamp_min(1.0)
        l_reg = (rb * ((pnk - yb) ** 2).sum(1)).sum() / denom
        l_heat = (rb * F.cross_entropy(logit.reshape(BATCH, -1), tcell,
                                       reduction='none')).sum() / denom
        l_conf = F.binary_cross_entropy_with_logits(conf, cb)
        loss = l_reg + 0.2 * l_heat + 0.3 * l_conf
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if s % 500 == 0 or s == 1:
            m = evaluate()
            curve.append(dict(step=s, id05=m['id05'], id10=m['id10'],
                              med_dnk=m['med_dnk'], conf_acc=m['conf_acc'],
                              loss=float(loss.detach())))
            if best is None or m['id05'] > best[0]:
                best = (m['id05'], {k: v.detach().cpu().clone()
                                    for k, v in net.state_dict().items()})
            log(f"  step {s:5d}  loss {float(loss):.4f}  val id@.05 "
                f"{m['id05']:.3f} id@.10 {m['id10']:.3f} med|dnk| "
                f"{m['med_dnk']:.3f} confacc {m['conf_acc']:.3f} "
                f"({(time.time()-t0)/60:.1f}m)", flush=True)
    net.load_state_dict(best[1])
    return net, curve, dict(best_id05=best[0], composition=comp_tr)


# ============================================================= matched-filter path
def peaknet_predict_mf(net, P, z, depth, Q, sigma0, kp, theta, force=False):
    """identify -> refine -> (fire) single-component MATCHED FILTER forecast /
    (else) hand path (mmse_gpu, delta=0 harmless)."""
    idd = peaknet_identify(net, P, z, kp, depth)
    fired = force or (idd['prob'] >= theta)
    kx0, ky0 = idd['khat']
    from d3_a7v2_refine import _refine_single
    rx, ry = _refine_single(kx0, ky0, P, z, depth)
    idd['refined'] = (rx, ry)
    if not fired:
        Pt = torch.tensor(P, device=DEV, dtype=torch.float32)
        zt = torch.tensor(z, device=DEV, dtype=torch.float32)
        Qt = torch.tensor(Q, device=DEV, dtype=torch.float32)
        pred = mmse_gpu(Pt, zt, depth, Qt, sigma0, kp_hint=kp).cpu().numpy()
        return pred, idd, False, dict(mode='hand')
    pred = single_forecast(rx, ry, P, z, depth, Q)       # matched filter
    return pred, idd, True, dict(mode='matched_filter')


# ================================================================= gate 1 / 2
def gate_bf2(net, theta):
    rows = []
    for seed in range(5):
        ep = load_synth('data/episode_BF2_1comp_seed%d_fixedT.h5' % seed,
                        np.random.default_rng(11))
        tk = ep['kp'] * ep['khat']
        Qf, Qt, npix, truth_tr = build_queries(ep)
        Q = np.vstack([Qf, Qt]); nf = len(Qf)
        sig0 = 0.05 * ep['Hs']
        ph = hand_predict(ep['P'], ep['z'], ep['depth'], Q, sig0, ep['kp'])
        # gated deployment + forced (isolates recovery-given-identification)
        pn, idd, fired, info = peaknet_predict_mf(net, ep['P'], ep['z'],
                                                  ep['depth'], Q, sig0, ep['kp'],
                                                  theta)
        pf, idf, _, _ = peaknet_predict_mf(net, ep['P'], ep['z'], ep['depth'],
                                           Q, sig0, ep['kp'], theta, force=True)
        rx, ry = idd['refined']
        nkx, nky = idd['nk']
        dk_refined = float(math.hypot(rx - tk[0], ry - tk[1]) / ep['kp'])
        dk_cnn = float(math.hypot(nkx * ep['kp'] - tk[0],
                                  nky * ep['kp'] - tk[1]) / ep['kp'])
        dk_argmax = float(math.hypot(idd['argmax_nk'][0] * ep['kp'] - tk[0],
                                     idd['argmax_nk'][1] * ep['kp'] - tk[1])
                          / ep['kp'])
        ch, rh, _ = field_track_corr(ph, ep, npix, truth_tr, nf)
        cn, rn, _ = field_track_corr(pn, ep, npix, truth_tr, nf)   # gated
        cf, rf, _ = field_track_corr(pf, ep, npix, truth_tr, nf)   # forced MF
        identified = bool(dk_refined <= 0.05)
        rows.append(dict(
            seed=seed, in_scope=(seed in SCOPE), true_kabs=float(ep['kp']),
            true_nk=ep['khat'].tolist(), prob=idd['prob'], fired=fired,
            argmax_kabs=idd['argmax_kabs'], dk_argmax=dk_argmax,
            cnn_nk=[nkx, nky], dk_cnn=dk_cnn,
            refined=[rx, ry], dk_refined=dk_refined, identified=identified,
            hand_f0=ch[0], hand_rho=rh,
            gated_f0=cn[0], gated_rho=rn,
            mf_f0=cf[0], mf_rho=rf,                       # matched filter (forced)
            recovered_mf=bool(cf[0] >= 0.5 and rf >= 0.5),
            recovered_gated=bool(cn[0] >= 0.5 and rn >= 0.5),
            hand_field=ch, mf_field=cf))
        tag = 'SCOPE' if seed in SCOPE else 'excl '
        print(f"[BF2 s{seed} {tag}] prob {idd['prob']:.3f} fire {fired}  "
              f"argmax|k| {idd['argmax_kabs']:.2f} CNNdk {dk_cnn:.3f} -> refdk "
              f"{dk_refined:.3f} {'ID' if identified else '  '}  "
              f"hand {ch[0]:+.2f}/{rh:+.2f}  MF(forced) {cf[0]:+.2f}/{rf:+.2f}  "
              f"gated {cn[0]:+.2f}/{rn:+.2f}", flush=True)
    sc = [r for r in rows if r['seed'] in SCOPE]
    n_id = int(sum(r['identified'] for r in sc))
    n_rec_mf = int(sum(r['recovered_mf'] for r in sc))
    n_rec_gated = int(sum(r['recovered_gated'] for r in sc))
    return rows, dict(
        scope=SCOPE, n_scope=len(sc),
        n_identified=n_id, id_pass=n_id >= 2,
        n_recover_mf=n_rec_mf, n_recover_gated=n_rec_gated,
        recover_pass=n_rec_mf >= 2)


# ==================================================================== gate 3
def gate_harmless_synth(net, theta):
    rows = []
    for fname in APP_EPISODES:
        ep = load_synth('data/' + fname, np.random.default_rng(11))
        Qf, Qt, npix, truth_tr = build_queries(ep)
        Q = np.vstack([Qf, Qt]); nf = len(Qf)
        sig0 = 0.05 * ep['Hs']
        ph = hand_predict(ep['P'], ep['z'], ep['depth'], Q, sig0, ep['kp'])
        pn, idd, fired, info = peaknet_predict_mf(net, ep['P'], ep['z'],
                                                  ep['depth'], Q, sig0, ep['kp'],
                                                  theta)
        ch, _, _ = field_track_corr(ph, ep, npix, truth_tr, nf)
        cn, _, _ = field_track_corr(pn, ep, npix, truth_tr, nf)
        dfield = [cn[k] - ch[k] for k in range(len(LEADS))]
        rows.append(dict(episode=fname, sea=ep['sea'], ncomp=ep['ncomp'],
                         prob=idd['prob'], fired=fired, dfield=dfield,
                         dfield_min=float(min(dfield))))
        print(f"[harm synth {fname}] prob {idd['prob']:.3f} fire {fired}  "
              f"dfield {[f'{d:+.3f}' for d in dfield]}", flush=True)
    return rows


def gate_harmless_real(net, theta, smoke=False):
    if smoke:
        return [], dict(skipped=True)
    from d3_sa_multirecord import MRecord
    from d3_gate_promotion_eval import window_data
    from d3_falcon_real_operator import RECORDS
    name = 'YS17'
    cfg = RECORDS[name]
    print(f"[harm real] loading {name} ...", flush=True)
    rec = MRecord(cfg['path'], stride=cfg['stride'])
    nfw = int(round(40.0 / rec.dt))
    n_lead = int(round(cfg['lead_max'] / rec.dt))
    i0s = np.linspace(nfw + 1, rec.T_all - n_lead - 2, 12).astype(int)
    rows = []
    for wi, i0 in enumerate(i0s):
        P, z, Pq, gt_q, leads, Hs, kp, _ = window_data(rec, cfg, int(i0),
                                                       nfw, n_lead, wi)
        v = np.isfinite(gt_q) & (leads <= 10.0)
        if v.sum() < 5:
            continue
        sig0 = 0.05 * Hs
        ph = hand_predict(P, z, cfg['depth'], Pq, sig0, kp)
        pn, idd, fired, info = peaknet_predict_mf(net, P, z, cfg['depth'], Pq,
                                                  sig0, kp, theta)
        ch = float(np.corrcoef(ph[v], gt_q[v])[0, 1])
        cn = float(np.corrcoef(pn[v], gt_q[v])[0, 1])
        rows.append(dict(record=name, wi=wi, prob=idd['prob'], fired=fired,
                         hand_enc=ch, fire_enc=cn, denc=cn - ch))
        print(f"[harm real {name} w{wi}] prob {idd['prob']:.3f} fire {fired}  "
              f"hand {ch:+.2f} fire {cn:+.2f}  d {cn-ch:+.3f}", flush=True)
    d = np.array([r['denc'] for r in rows])
    return rows, dict(n=len(rows), min_denc=float(d.min()) if len(d) else None,
                      mean_denc=float(d.mean()) if len(d) else None)


# ======================================================================= main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--eval_only', action='store_true')
    ap.add_argument('--skip_real', action='store_true')
    a = ap.parse_args()
    global N_TRAIN, N_VAL, STEPS, POOL_1COMP, POOL_MULTI, WIN_STARTS
    if a.smoke:
        N_TRAIN, N_VAL, STEPS = 2500, 400, 400
        POOL_1COMP = POOL_1COMP[:2]
        POOL_MULTI = POOL_MULTI[:2]
        WIN_STARTS = [0, 40, 80]
    t0 = time.time()
    print(f"device {DEV}  smoke={a.smoke}", flush=True)

    if a.eval_only and os.path.exists(CKPT):
        net = PeakNet().to(DEV)
        net.load_state_dict(torch.load(CKPT, map_location=DEV))
        curve, tinfo, converged_flag = [], {'loaded': True}, True
        print(f"loaded ckpt {CKPT}", flush=True)
    else:
        print("building operator-passed base windows ...", flush=True)
        bases_1c = build_bases(POOL_1COMP, WIN_STARTS, multi=False)
        bases_mc = build_bases(POOL_MULTI, WIN_STARTS, multi=True)
        net, curve, tinfo = train(bases_1c, bases_mc, STEPS)
        converged_flag = converged(curve)
        print(f"convergence (last 3k val id@.05 gain <2%): {converged_flag}",
              flush=True)
        if not converged_flag and not a.smoke:
            print("--- not converged: extending +10k steps (prereg) ---",
                  flush=True)
            net2, curve2, tinfo2 = train(bases_1c, bases_mc, 10000)
            if tinfo2['best_id05'] >= tinfo['best_id05']:
                net = net2
            curve = curve + [{**c, 'step': c['step'] + STEPS} for c in curve2]
            converged_flag = converged(curve)
            tinfo['best_id05'] = max(tinfo['best_id05'], tinfo2['best_id05'])
        torch.save(net.state_dict(), CKPT)
        print(f"saved ckpt {CKPT}", flush=True)

    print("\n--- confidence-gate calibration (1comp vs 5comp, non-BF2) ---",
          flush=True)
    cal = calibrate_conf(net)
    theta = cal['theta']
    print(f"pos(1comp) prob {['%.2f' % v for v in cal['pos']]}")
    print(f"neg(5comp) prob {['%.2f' % v for v in cal['neg']]}")
    print(f"theta {theta:.4f}  Youden {cal['youden_j']:.2f}  TPR "
          f"{cal['tpr_1comp']:.2f} FPR {cal['fpr_5comp']:.2f}", flush=True)

    print("\n--- G-B1v3-1 identification / G-B1v3-2 recovery (BF2 x5, "
          "scope s0/s1/s4) ---", flush=True)
    g1_rows, g1 = gate_bf2(net, theta)
    fire_rate = float(np.mean([r['fired'] for r in g1_rows]))
    print(f"G-B1v3-1 identification: {g1['n_identified']}/{g1['n_scope']} "
          f"(scope) |dk|/k<=0.05  ({'PASS' if g1['id_pass'] else 'FAIL'})",
          flush=True)
    print(f"G-B1v3-2 recovery(MF): {g1['n_recover_mf']}/{g1['n_scope']} "
          f"(scope; gated {g1['n_recover_gated']}/{g1['n_scope']})  "
          f"{'PASS' if g1['recover_pass'] else 'FAIL'}", flush=True)

    print("\n--- G-B1v3-3 harmless: synth 6 ---", flush=True)
    g3s = gate_harmless_synth(net, theta)
    g3s_min = float(min(d for r in g3s for d in r['dfield']))
    print(f"synth min dfield {g3s_min:+.4f}  "
          f"({'PASS' if g3s_min >= -0.01 else 'FAIL'} @ -0.01)", flush=True)

    print("\n--- G-B1v3-3 harmless: real YS17 12 windows ---", flush=True)
    smoke_real = a.smoke or a.skip_real
    g3r, g3r_sum = gate_harmless_real(net, theta, smoke=smoke_real)
    if not smoke_real:
        real_min = g3r_sum['min_denc']
        g3_pass = (g3s_min >= -0.01) and (real_min is not None
                                          and real_min >= -0.02)
        print(f"real min denc {real_min}  "
              f"({'PASS' if real_min is not None and real_min >= -0.02 else 'FAIL'}"
              f" @ -0.02)", flush=True)
    else:
        real_min = None
        g3_pass = g3s_min >= -0.01

    out = dict(
        date=DATE, prereg='docs/B1B2_DEEP_prereg_2026-07-12.md (B-1 v3)',
        method='operator-matched training + single-component matched-filter recovery',
        design=dict(
            ngrid=NGRID, b_hi=B_HI, noise=NOISE,
            sig_scale=[SIG_LO, SIG_HI], clu_scale=[CLU_LO, CLU_HI],
            clean_frac=CLEAN_FRAC, p_multi=P_MULTI,
            n_train=N_TRAIN, steps_planned=STEPS, batch=BATCH,
            win_starts=WIN_STARTS, n_params=n_params(net),
            pool_1comp=POOL_1COMP, pool_multi=POOL_MULTI, scope=SCOPE,
            note='training = wavesim operator-passed z decomposed to signal s + '
                 'clutter c; scene z=a*s+b*c+2cm, rigid-rotated (exact) for angle '
                 'coverage; multi-comp scenes = conf=0 negatives (monochromaticity '
                 'fire gate). recovery = k_hat -> a7v2 refine -> single (cos,sin) '
                 'LS matched filter -> query forecast.'),
        training=dict(best_id05=tinfo.get('best_id05'),
                      converged=converged_flag,
                      composition=tinfo.get('composition'), curve=curve),
        conf_calibration=cal,
        gate1_bf2=dict(summary=g1, fire_rate=fire_rate, seeds=g1_rows),
        gate3_harmless=dict(synth=g3s, synth_min_dfield=g3s_min,
                            real=g3r, real_summary=g3r_sum, passed=g3_pass),
        gates=dict(
            G_B1v3_1_identification=dict(
                passed=g1['id_pass'], n_identified=g1['n_identified'],
                scope=SCOPE, criterion='|dk|/k_true<=0.05 in >=2/3 of {s0,s1,s4}'),
            G_B1v3_2_recovery=dict(
                passed=g1['recover_pass'], n_recover_mf=g1['n_recover_mf'],
                n_recover_gated=g1['n_recover_gated'], scope=SCOPE,
                criterion='matched-filter field0>=0.5 AND rho>=0.5 in >=2/3 '
                          'of {s0,s1,s4}'),
            G_B1v3_3_harmless=dict(passed=g3_pass, synth_min_dfield=g3s_min,
                                   real_min_denc=real_min)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\nwrote {OUT}  ({(time.time()-t0)/60:.1f} min)", flush=True)

    print("\n================= GATES (scope s0/s1/s4) =================",
          flush=True)
    print(f"G-B1v3-1 identification : {'PASS' if g1['id_pass'] else 'FAIL'}  "
          f"({g1['n_identified']}/3 |dk|/k<=0.05; v2 was 1/5-all, 심화 1/5-all)")
    print(f"G-B1v3-2 recovery       : {'PASS' if g1['recover_pass'] else 'FAIL'}"
          f"  ({g1['n_recover_mf']}/3 matched-filter; gated "
          f"{g1['n_recover_gated']}/3)")
    print(f"G-B1v3-3 harmless       : {'PASS' if g3_pass else 'FAIL'}  "
          f"(synth min dfield {g3s_min:+.4f}"
          f"{'' if smoke_real else f', real min denc {real_min}'})")
    return 0


if __name__ == '__main__':
    sys.exit(main())
