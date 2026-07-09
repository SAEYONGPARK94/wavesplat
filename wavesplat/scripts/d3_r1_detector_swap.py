"""R1 — detector swap: is a BETTER DETECTOR the lever for B2's mode-geometry error?

E3 (shape diagnosis) split B2's gap to the linear floor (0.253 -> 0.128) into
d_fit = 0.052 (A,phi fitting) + d_geom = 0.073 (mode geometry, 58% share), with B2's
detected (kx,ky) sitting Chamfer/k_p ~ 0.88 off the true top-32 modes. R0 (physics
polish) barely moved the geometry (chamfer 0.878 -> 0.868, modes_GT 0.201 -> 0.197)
-> local refinement cannot fix ~1*k_p misses. HYPOTHESIS under test: the beamform-
detection CNN (learned CLEAN in the k-space matched-filter domain) finds BETTER mode
geometry than the diffusion+polish detector.

Per window (same 15 HOS M=3 windows, cache tmp/e1_windows; 400-obs subsample protocol
identical to E1-E3/R0):
  B2        : diffusion+polish detector (d3_ceiling_gif.b2_components, BMA of 2,
              <=30 modes/sample), A,phi LS to the 400 obs.  [E3 ref ~0.253]
  R1_off    : DetectOffsetCNN (results/direction3/beamform_offset_ds.pt, trained on the
              200k fixed dataset incl HOS) on the beamform power map -> off-grid (kx,ky)
              via extract(refine=True); A,phi LS to the SAME 400 obs, ridge 0.5.
              (aux: refine=False grid variant isolates the offset head's contribution;
              NK=64 grid cell = 0.019 rad/m = 0.14/0.23/0.38 k_p for BF3/4/5.)
  R1_off+R0 : R0 physics polish (d3_r0_physics_polish.r0_polish, identical hp/FIT-VAL
              protocol) on top of R1_off.
  R1_net    : plain DetectCNN (results/direction3/beamform_net.pt) second variant (+R0).
  O1p_true  : E1 corrected linear floor (must reproduce e1_oracle_fix exactly).
  CTRL      : capacity control — R1_off's detected peak(s) + Gaussian-jittered copies to
              30 modes (sigma in {0.3, 0.6}*k_p, seeded rng, NO extra information), A,phi
              LS to obs. If CTRL matches B2's modes_GT, B2's geometry advantage is
              prior-driven SPREAD CAPACITY around the peak, not mode identification.

EXTRACTION-THRESHOLD NOTE (probed on the 3 smoke windows, tmp/r1_probe_extract.py):
  on HOS continuous spectra the nets output ~ONE valid blob at the spectral peak;
  lowering thresh 0.15 -> 0.005 only adds spurious high-|k| peaks (chamfer/k_p 4-12,
  field nRMSE worsens 0.31 -> 0.35-0.53). thresh=0.15 (source default) is kept — the
  1-2-mode detection is a property of the heatmap, not of the threshold.

DECISIVE metrics:
  - Chamfer/k_p of each detector's (kx,ky) vs true top-32 (one-sided det->true, mean +
    fitted-A-weighted; plus reverse true->det "coverage" so few-mode detectors cannot
    cheat the one-sided metric). Is beamform-detect < B2's 0.88?
  - Field nRMSE pooled/per-sea + E2 cg-cone in/out zone split.
  - Ladder modes_GT (detector's (kx,ky), A,phi refit to full GT grid): does R1 beat
    B2's 0.201 and approach O1p 0.128?  Capacity note: modes_GT improves with mode
    COUNT regardless of quality, so a top-32 (by fitted |A|) variant is reported
    whenever a detector returns >32 modes.

Run:  python3 scripts/d3_r1_detector_swap.py --smoke   (BF3_s0/BF4_s0/BF5_s0, GPU)
      python3 scripts/d3_r1_detector_swap.py           (full 15 windows)
"""
import os, sys, json, time, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

import torch
import d44_hos_nonlinear as H
import d3_setdiff_posterior as DP                      # sets DEVICE (GPU if free)
from d3_setdiff_prior import SetDenoiser
from d3_r5_oracle import init_modes
from d3_e1_oracle_fix import ls_refit_floor, nrmse, GX, GY
from d3_ceiling_gif import b2_components, render_bma, PRIOR
from d3_e3_shape_diagnosis import window_data, peak_wavenumber, chamfer_to_true, rho
from wavesplat.model.tnse import fit_amplitudes_phases
# R0 module handles the E2 CUDA_VISIBLE_DEVICES import quirk at ITS import.
import d3_r0_physics_polish as R0
E2 = R0.E2
import d3_beamform_net as BN                           # module-level k-grid on BN.DEVICE
from d3_beamform_net import beamform, DetectCNN
from d3_beamform_net import extract as extract_net
from d3_beamform_offset import DetectOffsetCNN
from d3_beamform_offset import extract as extract_off

G = 9.81
DEPTH = H.DEPTH
SEAS = [('BF3', 0.6, 5.5), ('BF4', 1.2, 7.0), ('BF5', 2.0, 9.0)]
SEEDS = [0, 1, 2, 3, 4]
OUT_JSON = 'results/direction3/r1_detector_swap.json'
CKPT_OFF = 'results/direction3/beamform_offset_ds.pt'
CKPT_NET = 'results/direction3/beamform_net.pt'
# R0 polish hyper-params == d3_r0_physics_polish main() defaults (r0_physics_polish.json)
HP = dict(rounds=5, inner=40, lr_frac=0.02, clip_frac=0.10, ridge=0.5)
HOLDOUT = 0.15


def ls_comp(kx, ky, Ps, zs, depth):
    """LS A,phi fit to the 400-obs subsample — identical protocol to b2_components."""
    obs = np.column_stack([Ps[:, 0], Ps[:, 1], Ps[:, 2], zs]).astype(np.float32)
    A, phi = fit_amplitudes_phases(kx.astype(np.float32), ky.astype(np.float32),
                                   obs, depth, lambda_reg=0.5)
    return (kx.astype(np.float32), ky.astype(np.float32), A, phi)


def chamfer_from_true(comps, kx_true, ky_true, A_true):
    """REVERSE chamfer (coverage): true top-32 -> pooled detected set, mean +
    true-energy-weighted. High = detector MISSES energetic true modes (a few-mode
    detector can score well on the one-sided det->true metric while covering nothing)."""
    kxd = np.concatenate([c[0] for c in comps]); kyd = np.concatenate([c[1] for c in comps])
    if len(kxd) == 0:
        return None, None
    d = np.sqrt((kx_true[:, None] - kxd[None, :]) ** 2 +
                (ky_true[:, None] - kyd[None, :]) ** 2).min(axis=1)
    w = A_true.astype(np.float64) ** 2
    return float(d.mean()), float((d * w).sum() / max(w.sum(), 1e-12))


def top32(comp):
    """Capacity-matched subset: top-32 modes by fitted |A| (no-op if <=32)."""
    kx, ky, A, phi = comp
    if len(kx) <= 32:
        return comp
    o = np.argsort(-np.abs(A))[:32]
    return (kx[o], ky[o], A[o], phi[o])


def ctrl_spread(kx0, ky0, kp, seed, sigma_frac, n=30):
    """Capacity control: detected peak(s) + Gaussian-jittered copies to n modes.
    Carries NO information beyond the detected peak location(s)."""
    rng = np.random.default_rng(10_000 * int(round(100 * sigma_frac)) + seed)
    anc = rng.integers(0, len(kx0), n - len(kx0))
    kx = np.concatenate([kx0, kx0[anc] + rng.normal(0, sigma_frac * kp, n - len(kx0))])
    ky = np.concatenate([ky0, ky0[anc] + rng.normal(0, sigma_frac * kp, n - len(kx0))])
    return kx.astype(np.float32), ky.astype(np.float32)


@torch.no_grad()
def beamform_map(Ps, zs):
    """Power map input. t-origin shifted to min like the source evaluate() protocols
    (|resp|^2 is time-shift-invariant, so this is belt-and-braces only)."""
    Psh = Ps.copy(); Psh[:, 2] -= Psh[:, 2].min()
    return beamform(torch.tensor(Psh[None], device=BN.DEVICE, dtype=torch.float32),
                    torch.tensor(zs[None], device=BN.DEVICE, dtype=torch.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true', help='seeds [0] only (3 windows)')
    ap.add_argument('--thresh', type=float, default=0.15,
                    help='peak-extraction threshold (source-script default, untuned)')
    args = ap.parse_args()
    seeds = [0] if args.smoke else SEEDS

    # diffusion prior (B2 baseline)
    ck = torch.load(PRIOR, map_location=DP.DEVICE)
    stats = ck['stats']
    prior = SetDenoiser(**ck.get('arch', {})).to(DP.DEVICE)
    prior.load_state_dict(ck['model']); prior.eval()
    abar = ck['abar'].to(DP.DEVICE); betas = ck['betas'].to(DP.DEVICE)
    # beamform detectors
    net_off = DetectOffsetCNN().to(BN.DEVICE)
    net_off.load_state_dict(torch.load(CKPT_OFF, map_location=BN.DEVICE)); net_off.eval()
    net_pl = DetectCNN().to(BN.DEVICE)
    net_pl.load_state_dict(torch.load(CKPT_NET, map_location=BN.DEVICE)); net_pl.eval()
    # references
    e1 = json.load(open('results/direction3/e1_oracle_fix.json'))['per_window']
    e3 = json.load(open('results/direction3/e3_shape_diagnosis.json'))['per_window']
    r0ref = json.load(open('results/direction3/r0_physics_polish.json'))['per_window']

    rows = {}; t0 = time.time()
    print(f'R1 detector swap — B2 (diffusion+polish) vs beamform-detect CNNs  '
          f'[DPS dev={DP.DEVICE}, beamform dev={BN.DEVICE}]\n'
          f'  off ckpt={CKPT_OFF}\n  net ckpt={CKPT_NET}\n'
          f'  R0 polish hp={HP} holdout={HOLDOUT}  thresh={args.thresh}', flush=True)
    hdr = (f"{'window':8s} {'nB2/nOf/nNe':>11} | {'B2':>5} {'Off':>5} {'Of+R0':>5} "
           f"{'Net':>5} {'Ne+R0':>5} {'O1p':>5} | {'mGT:B2':>6} {'Off':>5} {'Net':>5} "
           f"{'C.3':>5} {'C.6':>5} | "
           f"{'ch:B2':>5} {'ofG':>4} {'Off':>4} {'Net':>4} {'R0':>4} | {'cov':>4} | "
           f"{'zin B2':>6} {'Off':>5} {'zout B2':>7} {'Off':>5}")
    print(hdr, flush=True)

    for tag, Hs, Tp in SEAS:
        kp = peak_wavenumber(Tp)
        lam = E2.peak_wavelength(Tp)
        for seed in seeds:
            gt, t, P, z = window_data(tag, seed)
            Hs_t = float(gt.std() * 4)
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(z), min(400, len(z)), replace=False)
            Ps, zs = P[idx].astype(np.float32), z[idx].astype(np.float32)
            # FIT/VAL split of the FULL obs stack — identical protocol to R0 (rng
            # continues after the 400-draw so the baseline subsample is untouched)
            perm = rng.permutation(len(z))
            n_val = int(round(HOLDOUT * len(z)))
            vi, fi = perm[:n_val], perm[n_val:]
            Pf, zf = P[fi].astype(np.float64), z[fi].astype(np.float64)
            Pv, zv = P[vi].astype(np.float64), z[vi].astype(np.float64)

            kxT, kyT, AT, _ = init_modes(Hs, Tp, seed)
            kxT32, kyT32, AT32 = kxT[:32], kyT[:32], AT[:32] / 2.0   # factor-2-fixed A

            # ── 1) B2 baseline ──
            comps_b2 = b2_components(prior, stats, abar, betas, Ps, zs, DEPTH)
            if not comps_b2:
                print(f'{tag}_s{seed}: EMPTY B2 set — skipped', flush=True); continue
            f_b2 = render_bma(comps_b2, DEPTH, t)
            nr_b2 = nrmse(f_b2, gt, Hs_t)
            mgt_b2 = R0.modes_gt_refit(comps_b2, gt, t, Hs_t)
            ch_b2, chw_b2 = chamfer_to_true(comps_b2, kxT32, kyT32)
            cov_b2, covw_b2 = chamfer_from_true(comps_b2, kxT32, kyT32, AT32)

            # ── 2) beamform detectors (one map, two nets) ──
            Bmap = beamform_map(Ps, zs)
            with torch.no_grad():
                Hm, oxm, oym = net_off(Bmap)
                Hp = net_pl(Bmap)[0]
            kxo, kyo = extract_off(Hm[0], oxm[0], oym[0], thresh=args.thresh, refine=True)
            kxg, kyg = extract_off(Hm[0], oxm[0], oym[0], thresh=args.thresh, refine=False)
            kxn, kyn = extract_net(Hp, thresh=args.thresh)

            dets = {}
            for name, kxd, kyd in (('off', kxo, kyo), ('off_grid', kxg, kyg),
                                   ('net', kxn, kyn)):
                if len(kxd) == 0:
                    dets[name] = None; continue
                comp = ls_comp(kxd, kyd, Ps, zs, DEPTH)
                f = render_bma([comp], DEPTH, t)
                d = dict(comp=comp, f=f, n=int(len(kxd)),
                         nrmse=nrmse(f, gt, Hs_t), rho=rho(f, gt))
                d['chamfer'], d['chamfer_w'] = chamfer_to_true([comp], kxT32, kyT32)
                d['cov'], d['cov_w'] = chamfer_from_true([comp], kxT32, kyT32, AT32)
                if name != 'off_grid':                       # ladder rungs
                    d['modes_gt'] = R0.modes_gt_refit([comp], gt, t, Hs_t)
                    d['modes_gt_top32'] = (R0.modes_gt_refit([top32(comp)], gt, t, Hs_t)
                                           if len(kxd) > 32 else d['modes_gt'])
                dets[name] = d
            if dets['off'] is None or dets['net'] is None:
                print(f'{tag}_s{seed}: EMPTY beamform detection — skipped', flush=True)
                continue

            # ── 3) R0 physics polish on top of each R1 variant ──
            for name in ('off', 'net'):
                kxd, kyd, Ad, phid = dets[name]['comp']
                best, _, val_hist, _, acc = R0.r0_polish(
                    kxd.astype(np.float64), kyd.astype(np.float64),
                    Ad.astype(np.float64), phid.astype(np.float64),
                    Pf, zf, Pv, zv, DEPTH, kp, **HP)
                f_r = render_bma([best], DEPTH, t)
                ch_r, chw_r = chamfer_to_true([best], kxT32, kyT32)
                dets[name + '_r0'] = dict(
                    nrmse=nrmse(f_r, gt, Hs_t), rho=rho(f_r, gt), f=f_r,
                    chamfer=ch_r, chamfer_w=chw_r, accepted=int(acc),
                    modes_gt=R0.modes_gt_refit([best], gt, t, Hs_t),
                    val_rms=float(val_hist[-1]))

            # ── 3b) capacity CONTROL: detected peak + blind jitter to 30 modes ──
            ctrl = {}
            for sf in (0.3, 0.6):
                kxc, kyc = ctrl_spread(kxo, kyo, kp, seed, sf)
                compc = ls_comp(kxc, kyc, Ps, zs, DEPTH)
                fc = render_bma([compc], DEPTH, t)
                chc, _ = chamfer_to_true([compc], kxT32, kyT32)
                covc, _ = chamfer_from_true([compc], kxT32, kyT32, AT32)
                ctrl[str(sf)] = dict(nrmse=nrmse(fc, gt, Hs_t),
                                     modes_gt=R0.modes_gt_refit([compc], gt, t, Hs_t),
                                     chamfer_over_kp=chc / kp, cov_over_kp=covc / kp)

            # ── 4) O1p floor + E2 zone split ──
            nr_o1p, f_o1p = ls_refit_floor(kxT32, kyT32, gt, t, Hs_t)
            md = E2.zone_mindist(Ps, t, kxT32, kyT32, advect=True)
            mask = md <= E2.R_PRIMARY * lam
            zones = {n: E2.zone_metrics(f, gt, Hs_t, mask)
                     for n, f in (('b2', f_b2), ('off', dets['off']['f']),
                                  ('off_r0', dets['off_r0']['f']),
                                  ('net', dets['net']['f']),
                                  ('net_r0', dets['net_r0']['f']), ('o1p', f_o1p))}

            key = f'{tag}_s{seed}'
            n_b2 = [int(len(c[0])) for c in comps_b2]

            def det_row(name):
                d = dets[name]
                r = dict(n_modes=d['n'], nrmse=d['nrmse'], rho=d['rho'],
                         chamfer=dict(mean=d['chamfer'], wmean=d['chamfer_w'],
                                      over_kp=d['chamfer'] / kp),
                         cov_from_true=dict(mean=d['cov'], wmean=d['cov_w'],
                                            over_kp=d['cov'] / kp))
                if 'modes_gt' in d:
                    r['modes_gt'] = d['modes_gt']; r['modes_gt_top32'] = d['modes_gt_top32']
                if name in zones:
                    r['zone'] = {k: zones[name][k]['nrmse'] for k in ('in', 'out')}
                return r

            rows[key] = dict(
                b2=dict(n_modes=n_b2, nrmse=nr_b2, rho=rho(f_b2, gt), modes_gt=mgt_b2,
                        chamfer=dict(mean=ch_b2, wmean=chw_b2, over_kp=ch_b2 / kp),
                        cov_from_true=dict(mean=cov_b2, wmean=covw_b2, over_kp=cov_b2 / kp),
                        zone={k: zones['b2'][k]['nrmse'] for k in ('in', 'out')}),
                off=det_row('off'), off_grid=det_row('off_grid'), net=det_row('net'),
                off_r0=dict(nrmse=dets['off_r0']['nrmse'], rho=dets['off_r0']['rho'],
                            chamfer_over_kp=dets['off_r0']['chamfer'] / kp,
                            modes_gt=dets['off_r0']['modes_gt'],
                            accepted=dets['off_r0']['accepted'],
                            zone={k: zones['off_r0'][k]['nrmse'] for k in ('in', 'out')}),
                net_r0=dict(nrmse=dets['net_r0']['nrmse'],
                            chamfer_over_kp=dets['net_r0']['chamfer'] / kp,
                            modes_gt=dets['net_r0']['modes_gt'],
                            accepted=dets['net_r0']['accepted'],
                            zone={k: zones['net_r0'][k]['nrmse'] for k in ('in', 'out')}),
                ctrl=ctrl,
                o1p_true=dict(nrmse=nr_o1p,
                              zone={k: zones['o1p'][k]['nrmse'] for k in ('in', 'out')}),
                k_p=float(kp), zone_coverage=float(mask.mean()),
                o1p_ref_e1=e1[key]['O1p_32'],
                b2_ref_e3=e3.get(key, {}).get('b2_obs', {}).get('nrmse'),
                b2mgt_ref_e3=e3.get(key, {}).get('b2modes_gt', {}).get('nrmse'),
                b2r0_ref=r0ref.get(key, {}).get('r0', {}).get('nrmse'))

            r = rows[key]
            print(f"{key:8s} {'+'.join(map(str, n_b2)) + '/' + str(r['off']['n_modes']) + '/' + str(r['net']['n_modes']):>11} | "
                  f"{nr_b2:5.3f} {r['off']['nrmse']:5.3f} {r['off_r0']['nrmse']:5.3f} "
                  f"{r['net']['nrmse']:5.3f} {r['net_r0']['nrmse']:5.3f} {nr_o1p:5.3f} | "
                  f"{mgt_b2:6.3f} {r['off']['modes_gt']:5.3f} {r['net']['modes_gt']:5.3f} "
                  f"{ctrl['0.3']['modes_gt']:5.3f} {ctrl['0.6']['modes_gt']:5.3f} | "
                  f"{ch_b2 / kp:5.2f} {r['off_grid']['chamfer']['over_kp']:4.2f} "
                  f"{r['off']['chamfer']['over_kp']:4.2f} {r['net']['chamfer']['over_kp']:4.2f} "
                  f"{r['off_r0']['chamfer_over_kp']:4.2f} | "
                  f"{r['off']['cov_from_true']['over_kp']:4.2f} | "
                  f"{r['b2']['zone']['in']:6.3f} {r['off_r0']['zone']['in']:5.3f} "
                  f"{r['b2']['zone']['out']:7.3f} {r['off_r0']['zone']['out']:5.3f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)

    # ── summary (window-mean, pooled + per-sea) ──
    tags = np.array([k.split('_')[0] for k in rows])

    def agg(fn):
        arr = np.array([fn(r) for r in rows.values()], dtype=np.float64)
        ps = {s: float(arr[tags == s].mean()) for s in ('BF3', 'BF4', 'BF5')
              if (tags == s).any()}
        return dict(mean=float(arr.mean()), per_sea=ps)

    summary = dict(
        b2=agg(lambda r: r['b2']['nrmse']),
        off_grid=agg(lambda r: r['off_grid']['nrmse']),
        off=agg(lambda r: r['off']['nrmse']),
        off_r0=agg(lambda r: r['off_r0']['nrmse']),
        net=agg(lambda r: r['net']['nrmse']),
        net_r0=agg(lambda r: r['net_r0']['nrmse']),
        o1p_true=agg(lambda r: r['o1p_true']['nrmse']),
        modes_gt_b2=agg(lambda r: r['b2']['modes_gt']),
        modes_gt_off=agg(lambda r: r['off']['modes_gt']),
        modes_gt_off_top32=agg(lambda r: r['off']['modes_gt_top32']),
        modes_gt_off_r0=agg(lambda r: r['off_r0']['modes_gt']),
        modes_gt_net=agg(lambda r: r['net']['modes_gt']),
        modes_gt_net_top32=agg(lambda r: r['net']['modes_gt_top32']),
        chamfer_over_kp_b2=agg(lambda r: r['b2']['chamfer']['over_kp']),
        chamfer_over_kp_off_grid=agg(lambda r: r['off_grid']['chamfer']['over_kp']),
        chamfer_over_kp_off=agg(lambda r: r['off']['chamfer']['over_kp']),
        chamfer_over_kp_off_r0=agg(lambda r: r['off_r0']['chamfer_over_kp']),
        chamfer_over_kp_net=agg(lambda r: r['net']['chamfer']['over_kp']),
        cov_over_kp_b2=agg(lambda r: r['b2']['cov_from_true']['over_kp']),
        cov_over_kp_off=agg(lambda r: r['off']['cov_from_true']['over_kp']),
        cov_over_kp_net=agg(lambda r: r['net']['cov_from_true']['over_kp']),
        n_modes_off=agg(lambda r: r['off']['n_modes']),
        n_modes_net=agg(lambda r: r['net']['n_modes']),
        ctrl03_nrmse=agg(lambda r: r['ctrl']['0.3']['nrmse']),
        ctrl03_modes_gt=agg(lambda r: r['ctrl']['0.3']['modes_gt']),
        ctrl03_chamfer=agg(lambda r: r['ctrl']['0.3']['chamfer_over_kp']),
        ctrl06_nrmse=agg(lambda r: r['ctrl']['0.6']['nrmse']),
        ctrl06_modes_gt=agg(lambda r: r['ctrl']['0.6']['modes_gt']),
        ctrl06_chamfer=agg(lambda r: r['ctrl']['0.6']['chamfer_over_kp']),
        zone_in_b2=agg(lambda r: r['b2']['zone']['in']),
        zone_in_off=agg(lambda r: r['off']['zone']['in']),
        zone_in_off_r0=agg(lambda r: r['off_r0']['zone']['in']),
        zone_out_b2=agg(lambda r: r['b2']['zone']['out']),
        zone_out_off=agg(lambda r: r['off']['zone']['out']),
        zone_out_off_r0=agg(lambda r: r['off_r0']['zone']['out']))

    m = {k: summary[k]['mean'] for k in summary}
    print('\n=== R1 summary (window-mean) ===', flush=True)
    for k in ('b2', 'off_grid', 'off', 'off_r0', 'net', 'net_r0', 'o1p_true'):
        s = summary[k]
        print(f"  nRMSE {k:10s} {s['mean']:.3f}  |  " +
              '  '.join(f'{t_} {v:.3f}' for t_, v in s['per_sea'].items()), flush=True)
    print(f"  modes_GT ladder      B2 {m['modes_gt_b2']:.3f}  Off {m['modes_gt_off']:.3f} "
          f"(top32 {m['modes_gt_off_top32']:.3f}, +R0 {m['modes_gt_off_r0']:.3f})  "
          f"Net {m['modes_gt_net']:.3f} (top32 {m['modes_gt_net_top32']:.3f})  "
          f"| O1p {m['o1p_true']:.3f}", flush=True)
    print(f"  chamfer/k_p det->true  B2 {m['chamfer_over_kp_b2']:.2f}  "
          f"OffGrid {m['chamfer_over_kp_off_grid']:.2f}  Off {m['chamfer_over_kp_off']:.2f}  "
          f"Off+R0 {m['chamfer_over_kp_off_r0']:.2f}  Net {m['chamfer_over_kp_net']:.2f}",
          flush=True)
    print(f"  coverage/k_p true->det B2 {m['cov_over_kp_b2']:.2f}  "
          f"Off {m['cov_over_kp_off']:.2f}  Net {m['cov_over_kp_net']:.2f}   "
          f"(#modes Off {m['n_modes_off']:.1f}, Net {m['n_modes_net']:.1f}, B2 <=30x2)",
          flush=True)
    print(f"  zone in/out          B2 {m['zone_in_b2']:.3f}/{m['zone_out_b2']:.3f}  "
          f"Off {m['zone_in_off']:.3f}/{m['zone_out_off']:.3f}  "
          f"Off+R0 {m['zone_in_off_r0']:.3f}/{m['zone_out_off_r0']:.3f}", flush=True)
    print(f"  CTRL peak+jitter30   sig0.3: nRMSE {m['ctrl03_nrmse']:.3f} "
          f"mGT {m['ctrl03_modes_gt']:.3f} ch {m['ctrl03_chamfer']:.2f}   "
          f"sig0.6: nRMSE {m['ctrl06_nrmse']:.3f} mGT {m['ctrl06_modes_gt']:.3f} "
          f"ch {m['ctrl06_chamfer']:.2f}   (vs B2 mGT {m['modes_gt_b2']:.3f})", flush=True)

    # ── sanity gates ──
    o1p_dev = max(abs(r['o1p_true']['nrmse'] - r['o1p_ref_e1']) for r in rows.values())
    b2_dev = max(abs(r['b2']['nrmse'] - r['b2_ref_e3'])
                 for r in rows.values() if r['b2_ref_e3'] is not None)
    mgt_dev = max(abs(r['b2']['modes_gt'] - r['b2mgt_ref_e3'])
                  for r in rows.values() if r['b2mgt_ref_e3'] is not None)
    print('\n=== sanity gates ===', flush=True)
    print(f"  O1p_true == E1 O1p_32 (deterministic): max|diff| = {o1p_dev:.2e}  "
          f"({'OK' if o1p_dev < 1e-6 else 'MISMATCH'})", flush=True)
    print(f"  B2_obs vs E3 ref (same seed/device):   max|diff| = {b2_dev:.3f}  "
          f"({'OK' if b2_dev < 0.02 else 'CHECK'})", flush=True)
    print(f"  B2 modes_GT vs E3 ref:                 max|diff| = {mgt_dev:.3f}  "
          f"({'OK' if mgt_dev < 0.02 else 'CHECK'})", flush=True)

    # ── verdict (honest, either direction) ──
    best_field = min(m['off'], m['off_r0'], m['net'], m['net_r0'])
    best_name = min((('off', m['off']), ('off_r0', m['off_r0']),
                     ('net', m['net']), ('net_r0', m['net_r0'])), key=lambda x: x[1])[0]
    ch_best = min(m['chamfer_over_kp_off'], m['chamfer_over_kp_net'])
    mgt_best = min(m['modes_gt_off_top32'], m['modes_gt_net_top32'])
    ctrl_mgt = min(m['ctrl03_modes_gt'], m['ctrl06_modes_gt'])
    few = m['n_modes_off'] < 8 and m['n_modes_net'] < 8       # can't resolve the cluster
    geom_better = ch_best < m['chamfer_over_kp_b2'] - 0.10
    ladder_better = mgt_best < m['modes_gt_b2'] - 0.01
    field_better = best_field < m['b2'] - 0.01
    ctrl_matches_b2 = ctrl_mgt < m['modes_gt_b2'] + 0.02
    verdict = (f"beamform-detect: n_modes off {m['n_modes_off']:.1f} / net "
               f"{m['n_modes_net']:.1f}; chamfer/k_p det->true {ch_best:.2f} vs B2 "
               f"{m['chamfer_over_kp_b2']:.2f}, coverage true->det "
               f"{min(m['cov_over_kp_off'], m['cov_over_kp_net']):.2f} vs B2 "
               f"{m['cov_over_kp_b2']:.2f}; modes_GT {mgt_best:.3f} vs B2 "
               f"{m['modes_gt_b2']:.3f} (O1p {m['o1p_true']:.3f}); best field "
               f"{best_name} {best_field:.3f} vs B2 {m['b2']:.3f}. ")
    if few and not (ladder_better and field_better):
        verdict += ('CAPACITY-NULL: the CNN localizes the dominant spectral peak '
                    f'precisely (per-mode chamfer {ch_best:.2f} k_p << B2 '
                    f"{m['chamfer_over_kp_b2']:.2f}) but resolves only ~1-2 of the 32 "
                    'clustered true modes from these obs; the extraction-threshold '
                    'probe shows no further real peaks exist in the heatmap. ')
        if ctrl_matches_b2:
            verdict += (f'CTRL (detected peak + blind jitter to 30 modes, no extra '
                        f'information) reaches modes_GT {ctrl_mgt:.3f} vs B2 '
                        f"{m['modes_gt_b2']:.3f} (matches or beats) -> B2's geometry "
                        'advantage is '
                        'prior-driven SPREAD CAPACITY around the peak, not mode '
                        "identification. E3's d_geom is therefore largely an "
                        'obs-RESOLUTION (information) limit, not a fixable detector '
                        'deficiency — detector-swap is NOT the lever.')
        else:
            verdict += (f'However CTRL spread does NOT reach B2 (modes_GT {ctrl_mgt:.3f} '
                        f"vs {m['modes_gt_b2']:.3f}) -> B2's mode geometry carries real "
                        'information beyond peak+spread; a detector with more capacity '
                        'may still be a lever.')
    elif geom_better and field_better:
        verdict += ('DETECTION CONFIRMED AS THE LEVER: the beamform CNN finds genuinely '
                    'better mode geometry AND a better field.')
    elif geom_better or ladder_better:
        verdict += ('MIXED: geometry improves but the field gain is small — better '
                    '(kx,ky) alone does not convert to field nRMSE without better A,phi '
                    'fitting (d_fit still binding).')
    elif ch_best > 0.75:
        verdict += ('NULL: beamform-detect geometry is NOT better (both detectors '
                    '~0.8-0.9 k_p off) -> mode identification from 400 sparse HOS obs '
                    'is hard for BOTH detector families -> E3\'s d_geom is more '
                    'INFORMATION-limited than the "estimation" framing implied.')
    else:
        verdict += 'PARTIAL: geometry mildly better but below the decision threshold.'
    print(f'\nREAD: {verdict}', flush=True)

    # strip fields (non-serializable) before saving
    out = dict(summary=summary, per_window=rows, smoke=args.smoke,
               hp=dict(**HP, holdout=HOLDOUT, thresh=args.thresh),
               ckpts=dict(off=CKPT_OFF, net=CKPT_NET), verdict=verdict,
               gates=dict(o1p_vs_e1_maxdiff=o1p_dev, b2_vs_e3_maxdiff=b2_dev,
                          b2_modes_gt_vs_e3_maxdiff=mgt_dev))
    path = OUT_JSON if not args.smoke else OUT_JSON.replace('.json', '_smoke.json')
    json.dump(out, open(path, 'w'), indent=2)
    print(f'saved {path}', flush=True)


if __name__ == '__main__':
    main()
