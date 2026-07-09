"""E2 — predictable-zone split: is B2's gap to the linear floor (O1p) estimation- or
information-limited?

E1 (d3_e1_oracle_fix) showed the TRUE linear floor is O1p_32=0.128 / O1p_64=0.097 vs
B2=0.253 — ~0.12 nRMSE headroom. But O1p is fit to the FULL GT grid (optimistic): a real
estimator only sees sparse obs in a limited footprint. E2 splits the grid into an
INFORMED (predictable) zone vs an UNINFORMED zone and recomputes (nRMSE, rho) for B2 and
O1p inside/outside, per window and pooled.

INFORMED-ZONE DEFINITION (physics cg-cone, NOT a convex hull):
  A grid point g is informed at nowcast time t* iff wave energy observed by some LiDAR
  sample (x_i, y_i, t_i) can have propagated to g by t*:
      || g - [p_i + v_g(k_m) * (t* - t_i)] || <= r_tol   for some obs i and mode m,
  where v_g(k) = cg * k/|k|,  cg = 0.5*(om/k)*(1 + 2kd/sinh(2kd)),  om = sqrt(gk tanh(kd)),
  over the true top-32 init modes' (kx, ky) (energy-dominant directions/band — same GT
  knowledge the O1p oracle uses; this is a diagnostic, not an estimator).
  r_tol = lambda_p/4 (quarter peak wavelength — the resolution scale over which a point
  sample constrains the phase of the dominant wave). Sensitivity to r_tol in
  {lambda_p/8, lambda_p/4, lambda_p/2} is reported so the split is not a tuned knob.
  Rationale for union-of-disks over convex hull: the pool is heavily concentrated near
  the sensor (median x ~ 16 m) with sparse far outliers (x up to ~230 m) — a hull is
  blown up by outliers to ~the whole grid and carries no information density. The
  cg-cone construction is the geometric predictable zone already validated against the
  pointwise CRB in Direction I (d1_predictable_zone.py).

GAP SPLIT: fraction of the (B2 - O1p_32) squared-error gap that lies INSIDE the zone:
      frac_in = sum_in[(B2-gt)^2 - (O1p-gt)^2] / sum_all[(B2-gt)^2 - (O1p-gt)^2].
  frac_in high  -> real ESTIMATION headroom inside the informed region (improve method).
  frac_in low   -> gap is INFORMATION/aperture-limited (outside) -> aperture is the lever.

Cache: extends tmp/e1_windows/*.npz with P, z (full obs stacks; backward-compatible —
E1 keeps reading gt, t). build_window is deterministic, asserted vs the cached gt.

Run:  python3 scripts/d3_e2_zone_split.py --smoke            (BF3_s0/BF4_s0/BF5_s0)
      python3 scripts/d3_e2_zone_split.py                     (full 15 windows)
      python3 scripts/d3_e2_zone_split.py --cache_only BF3_s0 (pre-warm one window cache)
"""
import os, sys, json, time, argparse
os.environ['CUDA_VISIBLE_DEVICES'] = ''   # CPU only (GPU reserved; no checkpoint edits)
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

import torch
import d44_hos_nonlinear as H
from d3_r5_oracle import init_modes
from d3_e1_oracle_fix import ls_refit_floor, GX, GY, CACHE
from d3_ceiling_gif import b2_components, render_bma, PRIOR
from d3_setdiff_prior import SetDenoiser

G = 9.81
DEPTH = H.DEPTH
SEAS = [('BF3', 0.6, 5.5), ('BF4', 1.2, 7.0), ('BF5', 2.0, 9.0)]
SEEDS = [0, 1, 2, 3, 4]
R_FRACS = (0.125, 0.25, 0.5)   # r_tol = frac * lambda_p; PRIMARY = 0.25
R_PRIMARY = 0.25
OUT_JSON = 'results/direction3/e2_zone_split.json'


# ── window data (cache extension, backward-compatible with E1) ──────────────
def window_data(tag, Hs, Tp, seed):
    os.makedirs(CACHE, exist_ok=True)
    fp = f'{CACHE}/{tag}_s{seed}.npz'
    gt_old = None
    if os.path.exists(fp):
        d = np.load(fp)
        if 'P' in d.files:
            return d['gt'], float(d['t']), d['P'], d['z']
        gt_old = d['gt']
    n_back = int(round(2.2 * Tp / 0.1))
    fr = H.build_window(Hs, Tp, seed, n_back, GX, GY, M=3)
    i_star = len(fr) - 2
    gt = fr[i_star]['gt']; t = float(fr[i_star]['t'])
    P = np.vstack([f['P'] for f in fr[:i_star + 1]]).astype(np.float32)
    z = np.concatenate([f['z'] for f in fr[:i_star + 1]]).astype(np.float32)
    if gt_old is not None:
        assert np.allclose(gt, gt_old, atol=1e-5), \
            f'{fp}: rebuilt GT != cached GT (build_window must be deterministic)'
    np.savez_compressed(fp, gt=gt, t=t, P=P, z=z)
    return gt, t, P, z


# ── physics ──────────────────────────────────────────────────────────────────
def peak_wavelength(Tp, d=DEPTH):
    om = 2 * np.pi / Tp
    k = om ** 2 / G
    for _ in range(60):                      # fixed point of om^2 = gk tanh(kd)
        k = om ** 2 / (G * np.tanh(k * d))
    return 2 * np.pi / k


def group_velocity_vec(kx, ky, d=DEPTH):
    k = np.hypot(kx, ky)
    om = np.sqrt(G * k * np.tanh(k * d))
    cg = 0.5 * (om / k) * (1 + 2 * k * d / np.sinh(2 * k * d))
    return cg * kx / k, cg * ky / k, cg


def zone_mindist(Ps, t_star, kx, ky, advect=True):
    """Min distance from every grid point to the cg-advected obs cloud (m).
    advect=False -> raw footprint (no physics buffer)."""
    if advect:
        vgx, vgy, _ = group_velocity_vec(kx, ky)
        dt = (t_star - Ps[:, 2])[:, None]                    # (n,1) >= 0
        qx = (Ps[:, 0][:, None] + vgx[None, :] * dt).ravel()
        qy = (Ps[:, 1][:, None] + vgy[None, :] * dt).ravel()
        q = np.column_stack([qx, qy])                        # (n*M, 2)
    else:
        q = Ps[:, :2].astype(np.float64)
    Xg, Yg = np.meshgrid(GX, GY)
    g = np.column_stack([Xg.ravel(), Yg.ravel()])
    md2 = np.full(len(g), np.inf)
    for s in range(0, len(q), 2000):                         # chunked pairwise
        d2 = ((g[:, None, :] - q[None, s:s + 2000, :]) ** 2).sum(-1)
        md2 = np.minimum(md2, d2.min(1))
    return np.sqrt(md2).reshape(Xg.shape)                    # (Ny, Nx)


# ── metrics ──────────────────────────────────────────────────────────────────
def zone_metrics(f, gt, Hs_t, mask):
    out = {}
    for name, m in (('in', mask), ('out', ~mask)):
        if m.sum() < 10:
            out[name] = dict(nrmse=None, rho=None, n=int(m.sum())); continue
        e = f[m] - gt[m]
        out[name] = dict(nrmse=float(np.sqrt(np.mean(e ** 2)) / max(Hs_t, 1e-6)),
                         rho=float(np.corrcoef(f[m], gt[m])[0, 1]), n=int(m.sum()))
    e = f - gt
    out['all'] = dict(nrmse=float(np.sqrt(np.mean(e ** 2)) / max(Hs_t, 1e-6)),
                      rho=float(np.corrcoef(f.ravel(), gt.ravel())[0, 1]), n=int(gt.size))
    return out


def gap_frac_inside(f_b2, f_o, gt, mask, Hs_t):
    """Fraction of the (B2 - O1p) normalized squared-error gap lying inside the zone.
    Also returns the normalized SSE gap terms for cross-window pooling."""
    d = ((f_b2 - gt) ** 2 - (f_o - gt) ** 2) / max(Hs_t, 1e-6) ** 2
    g_in, g_all = float(d[mask].sum()), float(d.sum())
    return (g_in / g_all if abs(g_all) > 1e-12 else None), g_in, g_all


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true', help='seeds [0] only (3 windows)')
    ap.add_argument('--cache_only', default=None, metavar='TAG_sSEED',
                    help='build/extend one window cache and exit (e.g. BF4_s2)')
    args = ap.parse_args()

    if args.cache_only:
        tag, s = args.cache_only.split('_s')
        Hs, Tp = {t: (h, p) for t, h, p in SEAS}[tag]
        t0 = time.time(); window_data(tag, Hs, Tp, int(s))
        print(f'cached {args.cache_only} ({time.time()-t0:.0f}s)'); return

    seeds = [0] if args.smoke else SEEDS
    ck = torch.load(PRIOR, map_location='cpu')
    stats = ck['stats']
    prior = SetDenoiser(**ck.get('arch', {}))
    prior.load_state_dict(ck['model']); prior.eval()
    abar = ck['abar'].cpu(); betas = ck['betas'].cpu()
    ceil = json.load(open('results/direction3/ceiling_metrics.json'))
    e1 = json.load(open('results/direction3/e1_oracle_fix.json'))['per_window']

    rows = {}; t0 = time.time()
    # pooled accumulators: normalized SSE + n per sea x method x region; gap SSE terms
    P_ACC = {s[0]: {m: {r: [0.0, 0] for r in ('in', 'out', 'all')}
                    for m in ('b2', 'o1p32', 'o1p64')} for s in SEAS}
    GAP = [0.0, 0.0]                                       # [sum g_in, sum g_all]
    print('E2 zone split — informed (cg-cone) vs uninformed, B2 vs O1p'
          f'  [r_tol = {R_PRIMARY}*lambda_p; CPU]', flush=True)
    hdr = (f"{'window':8s} {'cov%':>5} {'fp%':>4} | {'B2in':>5} {'B2out':>5} {'B2all':>5} | "
           f"{'O32in':>5} {'O32out':>6} {'O32all':>6} | {'O64in':>5} {'O64out':>6} | "
           f"{'rB2in':>5} {'rB2out':>6} | {'gapIn%':>6}")
    print(hdr, flush=True)

    for tag, Hs, Tp in SEAS:
        lam = peak_wavelength(Tp)
        for seed in seeds:
            gt, t, P, z = window_data(tag, Hs, Tp, seed)
            Hs_t = float(gt.std() * 4)
            idx = np.random.default_rng(seed).choice(len(z), min(400, len(z)), replace=False)
            Ps, zs = P[idx].astype(np.float32), z[idx].astype(np.float32)

            # B2 (ceiling-gif pipeline: DPS sample + polish + LS fit -> BMA render at t*)
            comps = b2_components(prior, stats, abar, betas, Ps, zs, DEPTH)
            f_b2 = render_bma(comps, DEPTH, t)

            # O1p (E1 corrected linear floor: true (kx,ky), A/phi LS-refit on full GT grid)
            kx, ky, _, _ = init_modes(Hs, Tp, seed)
            o32, f_o32 = ls_refit_floor(kx[:32], ky[:32], gt, t, Hs_t)
            o64, f_o64 = ls_refit_floor(kx[:64], ky[:64], gt, t, Hs_t)

            # informed zone: cg-cone mindist + primary/sensitivity masks
            md = zone_mindist(Ps, t, kx[:32], ky[:32], advect=True)
            md_fp = zone_mindist(Ps, t, kx[:32], ky[:32], advect=False)
            masks = {fr: md <= fr * lam for fr in R_FRACS}
            mask = masks[R_PRIMARY]
            cov = float(mask.mean()); cov_fp = float((md_fp <= R_PRIMARY * lam).mean())

            m_b2 = zone_metrics(f_b2, gt, Hs_t, mask)
            m_o32 = zone_metrics(f_o32, gt, Hs_t, mask)
            m_o64 = zone_metrics(f_o64, gt, Hs_t, mask)
            gfrac, g_in, g_all = gap_frac_inside(f_b2, f_o32, gt, mask, Hs_t)
            GAP[0] += g_in; GAP[1] += g_all
            for mname, mm in (('b2', m_b2), ('o1p32', m_o32), ('o1p64', m_o64)):
                for r in ('in', 'out', 'all'):
                    if mm[r]['nrmse'] is not None:
                        P_ACC[tag][mname][r][0] += mm[r]['nrmse'] ** 2 * mm[r]['n']
                        P_ACC[tag][mname][r][1] += mm[r]['n']

            sens = {}
            for frc in R_FRACS:                              # r_tol sensitivity
                mk = masks[frc]
                sb, so = zone_metrics(f_b2, gt, Hs_t, mk), zone_metrics(f_o32, gt, Hs_t, mk)
                gf, _, _ = gap_frac_inside(f_b2, f_o32, gt, mk, Hs_t)
                sens[frc] = dict(coverage=float(mk.mean()), b2_in=sb['in']['nrmse'],
                                 o1p32_in=so['in']['nrmse'], gap_frac_in=gf)

            key = f'{tag}_s{seed}'
            rows[key] = dict(coverage=cov, coverage_footprint_only=cov_fp,
                             lambda_p=float(lam), r_tol=float(R_PRIMARY * lam),
                             b2=m_b2, o1p32=m_o32, o1p64=m_o64,
                             gap_frac_inside=gfrac, sensitivity=sens,
                             b2_all_ref_gpu=ceil[key]['b2']['nrmse'],
                             o1p32_all_ref_e1=e1[key]['O1p_32'])
            print(f"{key:8s} {100*cov:5.1f} {100*cov_fp:4.1f} | "
                  f"{m_b2['in']['nrmse']:5.3f} {m_b2['out']['nrmse']:5.3f} {m_b2['all']['nrmse']:5.3f} | "
                  f"{m_o32['in']['nrmse']:5.3f} {m_o32['out']['nrmse']:6.3f} {m_o32['all']['nrmse']:6.3f} | "
                  f"{m_o64['in']['nrmse']:5.3f} {m_o64['out']['nrmse']:6.3f} | "
                  f"{m_b2['in']['rho']:5.2f} {m_b2['out']['rho']:6.2f} | "
                  f"{100*gfrac:6.1f}  ({time.time()-t0:.0f}s)", flush=True)

    # ── pooled summary (pixel-pooled, normalized by per-window Hs) ──
    def _pool(acc):                                        # {method: {region: nrmse}}
        return {m: {r: (float(np.sqrt(a[0] / a[1])) if a[1] else None)
                    for r, a in acc[m].items()} for m in acc}
    per_sea = {s: _pool(P_ACC[s]) for s in P_ACC if any(
        P_ACC[s]['b2'][r][1] for r in ('in', 'out', 'all'))}
    tot = {m: {r: [sum(P_ACC[s][m][r][0] for s in P_ACC),
                   sum(P_ACC[s][m][r][1] for s in P_ACC)] for r in ('in', 'out', 'all')}
           for m in ('b2', 'o1p32', 'o1p64')}
    pooled = _pool(tot)
    pooled_gap = GAP[0] / GAP[1] if abs(GAP[1]) > 1e-12 else None
    covs = np.array([r['coverage'] for r in rows.values()])
    print('\n=== E2 pooled (pixel-pooled over windows) ===', flush=True)
    print(f"  zone coverage: mean {100*covs.mean():.1f}%  range "
          f"[{100*covs.min():.1f}, {100*covs.max():.1f}]%", flush=True)
    for m in ('b2', 'o1p32', 'o1p64'):
        p = pooled[m]
        print(f"  {m:6s} nRMSE  in {p['in']:.3f}  out {p['out']:.3f}  all {p['all']:.3f}", flush=True)
    gi, go = pooled['b2']['in'] - pooled['o1p32']['in'], pooled['b2']['out'] - pooled['o1p32']['out']
    print(f"  B2-O1p32 gap:  in {gi:+.3f}  out {go:+.3f}  |  "
          f"share of squared-error gap INSIDE zone = {100*pooled_gap:.1f}%"
          f"  (coverage-confounded; per-sea in-gap below is the discriminating stat)", flush=True)
    print('\n=== per-sea read (pixel-pooled in-zone gap to linear floor) ===', flush=True)
    sea_read = {}
    for s in per_sea:
        p = per_sea[s]
        ig = p['b2']['in'] - p['o1p32']['in']
        og = (p['b2']['out'] - p['o1p32']['out']) if p['b2']['out'] is not None else None
        cv = float(np.mean([r['coverage'] for k, r in rows.items() if k.startswith(s)]))
        lab = 'INFO-limited (near floor inside)' if ig < 0.05 else 'ESTIMATION headroom inside'
        sea_read[s] = dict(coverage=cv, in_gap=float(ig),
                           out_gap=(float(og) if og is not None else None), read=lab)
        print(f"  {s}: cov {100*cv:5.1f}%  in-gap {ig:+.3f}  "
              f"out-gap {og:+.3f}  -> {lab}", flush=True)

    # ── sanity gates ──
    bad_cov = [k for k, r in rows.items() if not (0.05 < r['coverage'] < 0.95)]
    bad_ord = [k for k, r in rows.items()
               if r['b2']['in']['nrmse'] > r['b2']['out']['nrmse'] + 1e-6]
    o1p_dev = max(abs(r['o1p32']['all']['nrmse'] - r['o1p32_all_ref_e1']) for r in rows.values())
    b2_dev = max(abs(r['b2']['all']['nrmse'] - r['b2_all_ref_gpu']) for r in rows.values())
    print('\n=== sanity gates ===', flush=True)
    print(f"  coverage in (5%,95%):            violations = {bad_cov or 'none'}", flush=True)
    print(f"  B2 in-nRMSE <= out-nRMSE:        violations = {bad_ord or 'none'}", flush=True)
    print(f"  O1p32(all) == E1 (deterministic): max|diff| = {o1p_dev:.4f}  "
          f"({'OK' if o1p_dev < 1e-3 else 'MISMATCH'})", flush=True)
    print(f"  B2(all) vs ceiling ref (CPU-RNG != GPU-RNG draw, loose): max|diff| = {b2_dev:.3f}  "
          f"({'OK' if b2_dev < 0.08 else 'CHECK'})", flush=True)

    n_est = sum(1 for s in sea_read if sea_read[s]['in_gap'] >= 0.05)
    verdict = (f"{n_est}/{len(sea_read)} seas show ESTIMATION headroom inside the informed "
               f"zone (in-gap >= 0.05); remaining seas are near the linear floor inside "
               f"-> INFORMATION/aperture-limited there")
    print(f"\nREAD: {verdict}", flush=True)

    out = dict(zone=dict(rule='cg-cone: union of r_tol disks around obs advected by '
                              'group velocity of true top-32 modes over (t*-t_i)',
                         r_tol_frac_of_lambda_p=R_PRIMARY, sensitivity_fracs=list(R_FRACS)),
               pooled=pooled, per_sea=per_sea, sea_read=sea_read,
               pooled_gap_frac_inside=pooled_gap, per_window=rows,
               smoke=args.smoke, verdict=verdict)
    path = OUT_JSON if not args.smoke else OUT_JSON.replace('.json', '_smoke.json')
    json.dump(out, open(path, 'w'), indent=2)
    print(f'saved {path}', flush=True)


if __name__ == '__main__':
    main()
