"""A-7 — quasi-monochromatic local fine-bank hybrid (frozen prereg
docs/A7_D20_prereg_2026-07-12.md, A-7 section).

Hypothesis: BF2-1comp collapse is bank quantisation (coarse dk = kp/10 >
required 2*pi/L ~ 0.026). On a quasi-monochromatic detection, splice a
local fine bank (dk = kp/40) around the beamform peak; broadband seas do
not trigger, so the treatment is harmless.

Design (FROZEN):
  detect   : beamform power top-1 cell fraction >= theta (theta tuned on
             1comp-vs-5comp synthetic, NON-BF2 -- outside the judgment set).
  finebank : around detected peak k0, |k-k0|/k0 <= 0.25 and angle +-15 deg,
             add a dk = kp/40 grid (coarse bank kept); the added atoms'
             prior INHERITS the parent (nearest coarse cell) beamform energy.
  arms     : hand      = mmse_gpu (existing explicit bank, dk=kp/10 loop)
             finebank  = hand, plus the local fine bank when detection fires.
  mmse_gpu internals are NOT modified; the augmented solve is a separate
  faithful copy (solve_mmse_bank), verified to reproduce mmse_gpu on the
  coarse bank.

Gates (FROZEN):
  G-A7-1 recovery : BF2-1comp fixedT (all 5 seeds) field corr(lead0) >= 0.5
                    AND encounter rho >= 0.5 -- in the MAJORITY of episodes.
  G-A7-2 harmless : 6 appendix synth episodes  dfield >= -0.01 AND YS17 12
                    real windows  dencounter >= -0.02.
  FAIL branch     : "bank-quantisation prescription insufficient (phase
                    requirement stricter than detection/alignment error)".

python3 scripts/d3_a7_finebank.py            # full run
python3 scripts/d3_a7_finebank.py --smoke    # BF2 only (skip real spot)
"""
import argparse
import glob
import json
import os
import sys
import time

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from d3_gpu_mmse import (mmse_gpu, DEV, build_bank, beamform_prior, design, G)

OUT = 'results/direction3/a7_finebank_eval.json'
DATE = '2026-07-12'
U_SHIP = 11.3                      # C-8 design speed [m/s] (encounter track)
LEADS = [0.0, 2.0, 5.0, 8.0]      # field-corr leads (matches appendix)
TRACK_LEADS = np.round(np.arange(0.0, 8.01, 0.2), 3)
TRACK_YOFF = np.linspace(-60.0, 60.0, 13)
KREL, ANGDEG, FINE_DIV = 0.25, 15.0, 40    # frozen fine-bank geometry

# ---- 6 appendix representative episodes (same files as appendix_synth) ----
APP_EPISODES = [
    'episode_BF2_1comp_seed0_fixedT.h5',
    'episode_BF3_5comp_seed0_fixedT.h5',
    'episode_BF4_15comp_seed0_fixedT.h5',
    'episode_BF5_15comp_seed0_fixedT.h5',
    'episode_BF4_15comp_seed0_fixedT_dir180.h5',
    'episode_BF5_15comp_seed5_fixedT_dir180.h5',
]


# ====================================================================== solve
def solve_mmse_bank(P, z, depth, Pq, sigma0, kxb, kyb, om, pw,
                    seed=0, val_frac=0.25, max_em=6, no_em=False):
    """Faithful copy of mmse_gpu's (pexp,smult)-select + SBL-EM + solve for an
    EXPLICIT bank (kxb,kyb,om) and an EXPLICIT raw prior pw (B,). mmse_gpu is
    not touched. Passing the coarse bank + its beamform prior reproduces
    mmse_gpu exactly (max-norm of pw cancels under the pexp renormalisation)."""
    dev = P.device
    n = len(z)
    g = np.random.default_rng(seed)
    vi = g.choice(n, max(8, int(val_frac * n)), replace=False)
    vmask = torch.zeros(n, dtype=torch.bool, device=dev)
    vmask[torch.tensor(vi, device=dev)] = True
    Ptr, ztr = P[~vmask], z[~vmask]
    Pv, zv = P[vmask], z[vmask]
    B = len(kxb)

    Phi_tr = design(Ptr, kxb, kyb, om)
    Phi_v = design(Pv, kxb, kyb, om)
    m = len(ztr)
    I = torch.eye(m, device=dev, dtype=P.dtype)
    varz = z.var(unbiased=False)

    best = None
    for pexp in (0.5, 1.0, 2.0):
        gm = pw ** pexp
        gm = gm / gm.sum()
        gam = torch.cat([gm, gm]) * varz
        for smult in (1.0, 3.0, 10.0):
            s2 = (smult * sigma0) ** 2
            K = (Phi_tr * gam[None]) @ Phi_tr.T + s2 * I
            fv = (Phi_v * gam[None]) @ (Phi_tr.T @ torch.linalg.solve(K, ztr))
            vr = torch.sqrt(((fv - zv) ** 2).mean())
            if best is None or float(vr) < best[0]:
                best = (float(vr), pexp, smult, gam)
    vr_best, pexp, smult, gam = best
    s2 = (smult * sigma0) ** 2

    best_em = (vr_best, gam.clone())
    for _ in range(0 if no_em else max_em):
        K = (Phi_tr * gam[None]) @ Phi_tr.T + s2 * I
        Ki_Phi = torch.linalg.solve(K, Phi_tr)
        mu = gam * (Phi_tr.T @ torch.linalg.solve(K, ztr))
        sig = gam - gam ** 2 * (Phi_tr * Ki_Phi).sum(0)
        gb = 0.5 * (mu[:B] ** 2 + mu[B:] ** 2
                    + sig[:B].clamp_min(0) + sig[B:].clamp_min(0))
        gam = torch.cat([gb, gb])
        K2 = (Phi_tr * gam[None]) @ Phi_tr.T + s2 * I
        fv = (Phi_v * gam[None]) @ (Phi_tr.T @ torch.linalg.solve(K2, ztr))
        vr = float(torch.sqrt(((fv - zv) ** 2).mean()))
        if vr < best_em[0] - 1e-5:
            best_em = (vr, gam.clone())
        else:
            break
    gam = best_em[1]

    Phi_all = design(P, kxb, kyb, om)
    K = (Phi_all * gam[None]) @ Phi_all.T + s2 * torch.eye(
        n, device=dev, dtype=P.dtype)
    w_pred = Phi_all.T @ torch.linalg.solve(K, z)
    out = torch.empty(len(Pq), device=dev, dtype=P.dtype)
    for i in range(0, len(Pq), 512):
        Phi_q = design(Pq[i:i + 512], kxb, kyb, om)
        out[i:i + 512] = (Phi_q * gam[None]) @ w_pred
    return out


# =============================================================== detect / bank
def coarse_and_beamform(Pt, zt, kp, depth):
    kxb, kyb, om = build_bank(kp, (0.35, 2.5), kp / 10.0, depth, DEV,
                              torch.float32)
    pw = beamform_prior(Pt, zt, kxb, kyb, om)          # normalised to max 1
    return kxb, kyb, om, pw


def detect_top1(pw):
    """peak concentration = top-1 cell energy fraction (frozen statistic)."""
    return float(pw.max() / pw.sum())


def augment_finebank(kxb, kyb, om, pw, kp, depth):
    """Splice a dk=kp/40 grid around the beamform peak k0; the added atoms'
    prior inherits the nearest coarse cell's beamform energy."""
    kxa, kya, pwa = (kxb.cpu().numpy(), kyb.cpu().numpy(), pw.cpu().numpy())
    ip = int(np.argmax(pwa))
    k0x, k0y = float(kxa[ip]), float(kya[ip])
    k0 = float(np.hypot(k0x, k0y))
    dk = kp / FINE_DIV
    half = KREL * k0
    axf = np.arange(-half, half + dk / 2, dk)
    FX, FY = np.meshgrid(axf, axf)
    fkx, fky = k0x + FX.ravel(), k0y + FY.ravel()
    fk = np.hypot(fkx, fky)
    ang0 = np.arctan2(k0y, k0x)
    dang = np.abs(np.arctan2(np.sin(np.arctan2(fky, fkx) - ang0),
                             np.cos(np.arctan2(fky, fkx) - ang0)))
    keep = (np.abs(fk - k0) / k0 <= KREL) & (dang <= np.deg2rad(ANGDEG)) \
        & (fk > 1e-6)
    fkx, fky = fkx[keep], fky[keep]
    fk = np.hypot(fkx, fky)
    fom = np.sqrt(G * fk * np.tanh(fk * depth))
    # inherit parent (nearest coarse atom) beamform energy
    d2 = ((fkx[:, None] - kxa[None]) ** 2 + (fky[:, None] - kya[None]) ** 2)
    parent = np.argmin(d2, axis=1)
    fpw = pwa[parent]
    kxb2 = torch.cat([kxb, torch.tensor(fkx, device=DEV, dtype=torch.float32)])
    kyb2 = torch.cat([kyb, torch.tensor(fky, device=DEV, dtype=torch.float32)])
    om2 = torch.cat([om, torch.tensor(fom, device=DEV, dtype=torch.float32)])
    pw2 = torch.cat([pw, torch.tensor(fpw, device=DEV, dtype=torch.float32)])
    return kxb2, kyb2, om2, pw2, dict(k0=[k0x, k0y], k0abs=k0, n_fine=int(keep.sum()))


def finebank_predict(P, z, depth, Q, sigma0, kp, theta, force=False):
    """finebank arm: fire on detection (or force), else identical to mmse_gpu."""
    Pt = torch.tensor(P, device=DEV, dtype=torch.float32)
    zt = torch.tensor(z, device=DEV, dtype=torch.float32)
    Qt = torch.tensor(Q, device=DEV, dtype=torch.float32)
    kxb, kyb, om, pw = coarse_and_beamform(Pt, zt, kp, depth)
    top1 = detect_top1(pw)
    fired = force or (top1 >= theta)
    if not fired:
        pred = mmse_gpu(Pt, zt, depth, Qt, sigma0, kp_hint=kp).cpu().numpy()
        return pred, top1, False, dict(n_fine=0)
    kxb2, kyb2, om2, pw2, info = augment_finebank(kxb, kyb, om, pw, kp, depth)
    pred = solve_mmse_bank(Pt, zt, depth, Qt, sigma0, kxb2, kyb2, om2,
                           pw2).cpu().numpy()
    info['top1'] = top1
    return pred, top1, True, info


def hand_predict(P, z, depth, Q, sigma0, kp):
    Pt = torch.tensor(P, device=DEV, dtype=torch.float32)
    zt = torch.tensor(z, device=DEV, dtype=torch.float32)
    Qt = torch.tensor(Q, device=DEV, dtype=torch.float32)
    return mmse_gpu(Pt, zt, depth, Qt, sigma0, kp_hint=kp).cpu().numpy()


# ================================================================== synth load
def load_synth(path, rng):
    with h5py.File(path, 'r') as f:
        comp = f['config/wave_components'][:]
        Hs = float(f['config'].attrs['Hs'])
        depth = float(f['config'].attrs['depth'])
        ncomp = int(f['config'].attrs['n_components'])
        sea = str(f['config'].attrs['sea_state'])
        ts = f['frames/timestamps'][:]
        pts = f['frames/lidar_points']
        gx = f['frames/gt_grid_x'][:]
        gy = f['frames/gt_grid_y'][:]
        gz = f['frames/gt_wave_field'][:]
        ai = int(np.argmax(comp[:, 0]))
        kp = float(np.hypot(comp[ai, 2], comp[ai, 3]))
        khat = np.array([comp[ai, 2], comp[ai, 3]]) / kp
        obs_i = np.linspace(20, 220, 12).astype(int)
        t_end = float(ts[obs_i[-1]])
        dt = float(ts[1] - ts[0])
        P, zv = [], []
        for i in obs_i:
            p = pts[i]
            P.append(np.column_stack([p[:, 0], p[:, 1], np.full(len(p), ts[i])]))
            zv.append(p[:, 2])
        P, zv = np.vstack(P), np.concatenate(zv)
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
    return dict(P=P[sel], z=zv[sel], Hs=Hs, depth=depth, ncomp=ncomp, sea=sea,
                kp=kp, khat=khat, gx=gx, gy=gy, gz=gz, ts=ts, dt=dt,
                t_end=t_end)


def bilinear_frame(field, gx, gy, xq, yq):
    xq = np.clip(xq, gx[0], gx[-1]); yq = np.clip(yq, gy[0], gy[-1])
    ix = np.clip(np.searchsorted(gx, xq) - 1, 0, len(gx) - 2)
    iy = np.clip(np.searchsorted(gy, yq) - 1, 0, len(gy) - 2)
    tx = (xq - gx[ix]) / (gx[ix + 1] - gx[ix])
    ty = (yq - gy[iy]) / (gy[iy + 1] - gy[iy])
    return ((field[iy, ix] * (1 - tx) + field[iy, ix + 1] * tx) * (1 - ty)
            + (field[iy + 1, ix] * (1 - tx) + field[iy + 1, ix + 1] * tx) * ty)


def build_queries(ep):
    """field grid at all LEADS + encounter-track diagonals (head sea, h=-khat)."""
    gx, gy, t_end = ep['gx'], ep['gy'], ep['t_end']
    X, Y = np.meshgrid(gx, gy)
    npix = X.size
    Qf = np.vstack([np.column_stack([X.ravel(), Y.ravel(),
                                     np.full(npix, t_end + L)]) for L in LEADS])
    h = -ep['khat']                                  # ship into the wave
    p = np.array([-h[1], h[0]])
    cx, cy = 0.5 * (gx.min() + gx.max()), 0.5 * (gy.min() + gy.max())
    start = np.array([cx, cy]) - 0.5 * U_SHIP * TRACK_LEADS[-1] * h
    Qt_list, truth = [], []
    for off in TRACK_YOFF:
        base = start + off * p
        xy = base[None] + U_SHIP * TRACK_LEADS[:, None] * h[None]
        tq = t_end + TRACK_LEADS
        Qt_list.append(np.column_stack([xy[:, 0], xy[:, 1], tq]))
        gt = np.array([bilinear_frame(
            ep['gz'][min(int(round(t / ep['dt'])), len(ep['ts']) - 1)],
            gx, gy, np.array([x]), np.array([y]))[0]
            for x, y, t in zip(xy[:, 0], xy[:, 1], tq)])
        truth.append(gt)
    return Qf, np.vstack(Qt_list), npix, np.array(truth)


def field_track_corr(pred, ep, Qf_npix, truth_tr, nf):
    """split a stacked prediction into per-lead field corr + encounter rho."""
    corr = []
    for k, L in enumerate(LEADS):
        idx = min(int(round((ep['t_end'] + L) / ep['dt'])), len(ep['ts']) - 1)
        g = ep['gz'][idx].ravel()
        pk = pred[k * Qf_npix:(k + 1) * Qf_npix]
        corr.append(float(np.corrcoef(pk, g)[0, 1]))
    tr = pred[nf:].reshape(truth_tr.shape)
    per = [float(np.corrcoef(tr[i], truth_tr[i])[0, 1])
           for i in range(len(truth_tr)) if np.std(truth_tr[i]) > 1e-9]
    rho = float(np.mean(per))
    rho_pool = float(np.corrcoef(tr.ravel(), truth_tr.ravel())[0, 1])
    return corr, rho, rho_pool


# =================================================================== theta cal
CAL_POS = ['data/episode_BF3_1comp_seed%d_fixedT.h5',
           'data/episode_BF4_1comp_seed%d_fixedT.h5']
CAL_NEG = ['data/episode_BF3_5comp_seed%d_fixedT.h5',
           'data/episode_BF4_5comp_seed%d_fixedT.h5']


def calibrate_theta():
    """theta on 1comp (fire) vs 5comp (no fire), NON-BF2. Youden-J max, ties
    -> largest theta (fewest false positives = harmless-favouring)."""
    def top1_of(path):
        rng = np.random.default_rng(11)
        ep = load_synth(path, rng)
        Pt = torch.tensor(ep['P'], device=DEV, dtype=torch.float32)
        zt = torch.tensor(ep['z'], device=DEV, dtype=torch.float32)
        _, _, _, pw = coarse_and_beamform(Pt, zt, ep['kp'], ep['depth'])
        return detect_top1(pw)
    pos = np.array([top1_of(p % s) for p in CAL_POS for s in range(5)])
    neg = np.array([top1_of(p % s) for p in CAL_NEG for s in range(5)])
    cands = np.unique(np.concatenate([pos, neg]))
    mids = (cands[:-1] + cands[1:]) / 2
    best = None
    for th in mids:
        tpr = (pos >= th).mean(); fpr = (neg >= th).mean()
        j = tpr - fpr
        if best is None or j > best[0] + 1e-12 or (abs(j - best[0]) < 1e-12
                                                   and th > best[1]):
            best = (j, float(th), tpr, fpr)
    j, theta, tpr, fpr = best
    return dict(theta=theta, youden_j=j, tpr_1comp=float(tpr),
                fpr_5comp=float(fpr), pos=pos.tolist(), neg=neg.tolist())


# ====================================================================== gate 1
def gate1(theta):
    rows = []
    for seed in range(5):
        path = 'data/episode_BF2_1comp_seed%d_fixedT.h5' % seed
        rng = np.random.default_rng(11)
        ep = load_synth(path, rng)
        Qf, Qt, npix, truth_tr = build_queries(ep)
        Q = np.vstack([Qf, Qt]); nf = len(Qf)
        sig0 = 0.05 * ep['Hs']
        ph = hand_predict(ep['P'], ep['z'], ep['depth'], Q, sig0, ep['kp'])
        pn, top1, fired, info = finebank_predict(ep['P'], ep['z'], ep['depth'],
                                                 Q, sig0, ep['kp'], theta)
        pf, _, _, finfo = finebank_predict(ep['P'], ep['z'], ep['depth'], Q,
                                           sig0, ep['kp'], theta, force=True)
        ch, rh, rhp = field_track_corr(ph, ep, npix, truth_tr, nf)
        cn, rn, rnp = field_track_corr(pn, ep, npix, truth_tr, nf)
        cf, rf, rfp = field_track_corr(pf, ep, npix, truth_tr, nf)
        rows.append(dict(
            seed=seed, kp=ep['kp'], top1=top1, fired=fired, n_fine=info['n_fine'],
            hand_field0=ch[0], hand_rho=rh,
            finebank_field0=cn[0], finebank_rho=rn,
            forced_field0=cf[0], forced_rho=rf, forced_k0abs=finfo.get('k0abs'),
            hand_field=ch, finebank_field=cn, forced_field=cf))
        print(f"[G1 BF2 s{seed}] top1 {top1:.4f} fire {fired}  "
              f"hand f0/rho {ch[0]:+.2f}/{rh:+.2f}  "
              f"fine {cn[0]:+.2f}/{rn:+.2f}  "
              f"forced(k0={finfo.get('k0abs'):.2f},n{finfo['n_fine']}) "
              f"{cf[0]:+.2f}/{rf:+.2f}", flush=True)
    fb0 = np.array([r['finebank_field0'] for r in rows])
    fbr = np.array([r['finebank_rho'] for r in rows])
    n_pass = int(((fb0 >= 0.5) & (fbr >= 0.5)).sum())
    passed = n_pass >= 3
    return rows, dict(n_pass=n_pass, n_total=5, passed=passed,
                      criterion='field0>=0.5 AND rho>=0.5 in majority (>=3/5)')


# ====================================================================== gate 2
def gate2_synth(theta):
    rows = []
    for fname in APP_EPISODES:
        rng = np.random.default_rng(11)
        ep = load_synth('data/' + fname, rng)
        Qf, Qt, npix, truth_tr = build_queries(ep)
        Q = np.vstack([Qf, Qt]); nf = len(Qf)
        sig0 = 0.05 * ep['Hs']
        ph = hand_predict(ep['P'], ep['z'], ep['depth'], Q, sig0, ep['kp'])
        pn, top1, fired, info = finebank_predict(ep['P'], ep['z'], ep['depth'],
                                                 Q, sig0, ep['kp'], theta)
        ch, rh, _ = field_track_corr(ph, ep, npix, truth_tr, nf)
        cn, rn, _ = field_track_corr(pn, ep, npix, truth_tr, nf)
        dfield = [cn[k] - ch[k] for k in range(len(LEADS))]
        rows.append(dict(episode=fname, sea=ep['sea'], ncomp=ep['ncomp'],
                         kp=ep['kp'], top1=top1, fired=fired,
                         hand_field=ch, finebank_field=cn, dfield=dfield,
                         dfield0=dfield[0], dfield_min=float(min(dfield))))
        print(f"[G2 synth {fname}] top1 {top1:.4f} fire {fired}  "
              f"dfield {[f'{d:+.3f}' for d in dfield]}", flush=True)
    return rows


def gate2_real(theta, smoke=False):
    if smoke:
        return [], dict(skipped=True)
    from d3_sa_multirecord import MRecord
    from d3_gate_promotion_eval import window_data
    from d3_falcon_real_operator import RECORDS, bilinear
    name = 'YS17'
    cfg = RECORDS[name]
    print(f"[G2 real] loading {name} record ...", flush=True)
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
        pn, top1, fired, info = finebank_predict(P, z, cfg['depth'], Pq, sig0,
                                                 kp, theta)
        ch = float(np.corrcoef(ph[v], gt_q[v])[0, 1])
        cn = float(np.corrcoef(pn[v], gt_q[v])[0, 1])
        rows.append(dict(record=name, wi=wi, kp=kp, top1=top1, fired=fired,
                         hand_enc=ch, finebank_enc=cn, denc=cn - ch))
        print(f"[G2 real {name} w{wi}] top1 {top1:.4f} fire {fired}  "
              f"hand {ch:+.2f} fine {cn:+.2f}  d {cn-ch:+.3f}", flush=True)
    d = np.array([r['denc'] for r in rows])
    return rows, dict(n=len(rows), min_denc=float(d.min()) if len(d) else None,
                      mean_denc=float(d.mean()) if len(d) else None)


# ========================================================================= verify
def verify_solve(ep):
    """solve_mmse_bank on the coarse bank must reproduce mmse_gpu."""
    Q = np.column_stack([np.linspace(20, 240, 200),
                         np.linspace(-50, 50, 200),
                         np.full(200, ep['t_end'])]).astype(np.float32)
    Pt = torch.tensor(ep['P'], device=DEV, dtype=torch.float32)
    zt = torch.tensor(ep['z'], device=DEV, dtype=torch.float32)
    Qt = torch.tensor(Q, device=DEV, dtype=torch.float32)
    sig0 = 0.05 * ep['Hs']
    a = mmse_gpu(Pt, zt, ep['depth'], Qt, sig0, kp_hint=ep['kp']).cpu().numpy()
    kxb, kyb, om, pw = coarse_and_beamform(Pt, zt, ep['kp'], ep['depth'])
    b = solve_mmse_bank(Pt, zt, ep['depth'], Qt, sig0, kxb, kyb, om,
                        pw).cpu().numpy()
    return float(np.max(np.abs(a - b)))


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    t0 = time.time()
    print(f"device {DEV}", flush=True)

    # verify the augmented solver reproduces mmse_gpu on the coarse bank
    rng = np.random.default_rng(11)
    ep0 = load_synth('data/episode_BF2_1comp_seed0_fixedT.h5', rng)
    vdiff = verify_solve(ep0)
    print(f"verify solve_mmse_bank == mmse_gpu (coarse): max|diff| {vdiff:.2e} "
          f"({'OK' if vdiff < 1e-3 else 'MISMATCH'})", flush=True)

    print("\n--- theta calibration (1comp vs 5comp, non-BF2) ---", flush=True)
    cal = calibrate_theta()
    theta = cal['theta']
    print(f"pos(1comp) top1: {['%.3f' % v for v in cal['pos']]}")
    print(f"neg(5comp) top1: {['%.3f' % v for v in cal['neg']]}")
    print(f"theta = {theta:.4f}  (Youden J {cal['youden_j']:.2f}, "
          f"TPR {cal['tpr_1comp']:.2f}, FPR {cal['fpr_5comp']:.2f})",
          flush=True)

    print("\n--- G-A7-1 recovery (BF2 1comp x5) ---", flush=True)
    g1_rows, g1 = gate1(theta)
    fire_rate = float(np.mean([r['fired'] for r in g1_rows]))
    print(f"G-A7-1 {'PASS' if g1['passed'] else 'FAIL'}  "
          f"({g1['n_pass']}/5 episodes recover; BF2 fire-rate {fire_rate:.0%})",
          flush=True)

    print("\n--- G-A7-2 harmless: synth 6 ---", flush=True)
    g2s = gate2_synth(theta)
    dfield_all = [d for r in g2s for d in r['dfield']]
    g2s_min = float(min(dfield_all))
    print(f"synth min dfield {g2s_min:+.4f}  "
          f"({'PASS' if g2s_min >= -0.01 else 'FAIL'} @ -0.01)", flush=True)

    print("\n--- G-A7-2 harmless: real YS17 12 windows ---", flush=True)
    g2r, g2r_sum = gate2_real(theta, smoke=a.smoke)
    if not a.smoke:
        real_min = g2r_sum['min_denc']
        g2_passed = (g2s_min >= -0.01) and (real_min is not None
                                            and real_min >= -0.02)
        print(f"real min denc {real_min:+.4f}  "
              f"({'PASS' if real_min >= -0.02 else 'FAIL'} @ -0.02)", flush=True)
    else:
        g2_passed = g2s_min >= -0.01
        real_min = None

    out = dict(
        date=DATE, prereg='docs/A7_D20_prereg_2026-07-12.md (A-7)',
        design=dict(detect='beamform top-1 cell fraction >= theta',
                    finebank=f'dk=kp/{FINE_DIV}, |k-k0|/k0<={KREL}, '
                             f'+-{ANGDEG}deg, inherited prior',
                    U_ship=U_SHIP, note='mmse_gpu unmodified; augmented solve '
                                        'is solve_mmse_bank (verified equal on '
                                        'coarse bank)'),
        verify_solve_maxdiff=vdiff,
        theta_calibration=cal,
        gate1=dict(summary=g1, bf2_fire_rate=fire_rate, episodes=g1_rows),
        gate2=dict(synth=g2s, synth_min_dfield=g2s_min,
                   real=g2r, real_summary=g2r_sum if not a.smoke else None,
                   passed=g2_passed),
        gates=dict(
            G_A7_1_recovery=dict(passed=g1['passed'], n_pass=g1['n_pass'],
                                 bf2_fire_rate=fire_rate),
            G_A7_2_harmless=dict(passed=g2_passed, synth_min_dfield=g2s_min,
                                 real_min_denc=real_min)))
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\nwrote {OUT}  ({(time.time()-t0)/60:.1f} min)", flush=True)

    print("\n================= GATES =================", flush=True)
    print(f"G-A7-1 recovery : {'PASS' if g1['passed'] else 'FAIL'}  "
          f"({g1['n_pass']}/5 recover, fire-rate {fire_rate:.0%})")
    print(f"G-A7-2 harmless : {'PASS' if g2_passed else 'FAIL'}  "
          f"(synth min dfield {g2s_min:+.4f}"
          f"{'' if a.smoke else f', real min denc {real_min:+.4f}'})")
    return 0


if __name__ == '__main__':
    sys.exit(main())
