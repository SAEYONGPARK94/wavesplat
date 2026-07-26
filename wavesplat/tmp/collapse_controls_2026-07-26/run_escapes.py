"""W8 second half — NON-LEARNED ("cheap analytic") ESCAPE CONTROLS on the 18 harvested
collapse windows. Protocol frozen in tmp/collapse_controls_2026-07-26/PREREG.md.

CPU ONLY: mmse_bank_estimate is pure numpy when kp_hint is given, and the second-peak
beamform map below is pure numpy. No torch import, no diffusion model, no GPU.

  nice -n 10 python3 tmp/collapse_controls_2026-07-26/run_escapes.py
"""
import os, sys, json, time, hashlib, traceback
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import warnings; warnings.filterwarnings('ignore')
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.join(ROOT, 'tmp', 'heldout_real_2026-07-26'))

OUT = os.path.join(ROOT, 'tmp/collapse_controls_2026-07-26/escapes.json')
PREREG = os.path.join(ROOT, 'tmp/collapse_controls_2026-07-26/PREREG.md')
G = 9.81

# ---- frozen target set (PREREG s1) -----------------------------------------
HELDOUT = [('BS11b', 4), ('BS11c', 3), ('BS11c', 4),
           ('BS13a', 0), ('BS13a', 1), ('BS13a', 2), ('BS13a', 4), ('BS13a', 5),
           ('BS13c', 2), ('BS13c', 3),
           ('LJ18n', 0), ('LJ18n', 1), ('LJ18n', 3), ('LJ18n', 4), ('LJ18n', 5)]
DESIGN = [('AA14', 0), ('AA15a', 3), ('LJ18g', 3)]
TARGETS = HELDOUT + DESIGN

# ---- frozen escape grids (PREREG s2) ---------------------------------------
E1_SEED_OFFSETS = (1000, 2000, 3000, 4000, 5000)
E2_KP_SCALES = (0.5, 2.0)          # plus the second-beamform-peak variant
E3_SIG_SCALES = (0.5, 2.0)
BF_N = 121                          # power-map resolution
BF_KLIM_MULT = 3.5                  # = mmse_bank_estimate's own zoom half-range
BF_SUPPRESS = 0.5                   # suppression box half-width, in units of kp_base


def second_beamform_peak(P, z, depth, kp_base):
    """FROZEN (PREREG s2 / E2-peak2). Pure-numpy dispersion-constrained power map,
    DC 3x3 block zeroed, 0.5*kp_base box suppression around peak 1, return peak 2."""
    klim = BF_KLIM_MULT * kp_base
    ax = np.linspace(-klim, klim, BF_N)
    KX, KY = np.meshgrid(ax, ax)                      # KX varies along axis 1
    kx = KX.ravel(); ky = KY.ravel()
    k = np.sqrt(kx ** 2 + ky ** 2 + 1e-12)
    om = np.sqrt(G * k * np.tanh(k * depth) + 1e-12)
    pw = np.empty(kx.size)
    for i in range(0, kx.size, 512):
        psi = (kx[i:i + 512, None] * P[None, :, 0]
               + ky[i:i + 512, None] * P[None, :, 1]
               - om[i:i + 512, None] * P[None, :, 2])
        pw[i:i + 512] = ((z[None] * np.cos(psi)).sum(1) ** 2
                         + (z[None] * np.sin(psi)).sum(1) ** 2)
    Pmap = pw.reshape(BF_N, BF_N)                     # [iy, ix]
    c = int(np.argmin(np.abs(ax)))
    Pmap[max(0, c - 1):c + 2, max(0, c - 1):c + 2] = 0.0
    iy1, ix1 = np.unravel_index(int(np.argmax(Pmap)), Pmap.shape)
    p1 = float(Pmap[iy1, ix1])
    dk = ax[1] - ax[0]
    sup = int(BF_SUPPRESS * kp_base / dk) + 1
    Q = Pmap.copy()
    Q[max(0, iy1 - sup):iy1 + sup + 1, max(0, ix1 - sup):ix1 + sup + 1] = 0.0
    iy2, ix2 = np.unravel_index(int(np.argmax(Q)), Q.shape)
    p2 = float(Q[iy2, ix2])
    if not (p2 > 0):
        return None
    return dict(kx1=float(ax[ix1]), ky1=float(ax[iy1]), k1=float(np.hypot(ax[ix1], ax[iy1])),
                kx2=float(ax[ix2]), ky2=float(ax[iy2]), k2=float(np.hypot(ax[ix2], ax[iy2])),
                p2_over_p1=(p2 / p1) if p1 > 0 else float('nan'))


def window_job(args):
    rec, win = args
    import heldout_cfg
    heldout_cfg.install()
    import d3_compare_gifs as cg
    from d3_sa_nomp import mmse_bank_estimate
    t0 = time.time()
    sc = cg.scene_real(rec, win=win)
    P, z, Pq, depth = sc['P'], sc['z'], sc['Pq'], sc['depth']
    npx = sc['npx']; nl = len(sc['leads'])
    kp_base = float(sc['kp']); sig_base = 0.05 * float(sc['Hs']); s_base = int(sc['seed'])
    gt = sc['maps_gt']; sg = float(np.std(gt[np.isfinite(gt)]))

    def score(fq):
        m0 = fq[:npx * nl].reshape(nl, *sc['Xq'].shape)[0]
        m = np.isfinite(gt[0]) & np.isfinite(m0)
        if m.sum() < 8:
            return None
        return float(np.sqrt(np.mean((m0[m] - gt[0][m]) ** 2)) / sg)

    def run(seed=None, kp=None, sig=None):
        return score(mmse_bank_estimate(P, z, depth, Pq,
                                        sigma0=sig_base if sig is None else sig,
                                        kp_hint=kp_base if kp is None else kp,
                                        seed=s_base if seed is None else seed))

    out = dict(record=rec, win=int(win), kp_base=kp_base, sigma0_base=sig_base,
               seed_base=s_base, Hs=float(sc['Hs']), n_obs=int(len(z)), npx=int(npx),
               depth=float(depth), arena=('design' if (rec, win) in DESIGN else 'heldout'))
    out['baseline_recheck'] = run()                                   # bit-identity control
    out['E1'] = {str(s_base + off): run(seed=s_base + off) for off in E1_SEED_OFFSETS}
    pk = second_beamform_peak(np.asarray(P, float), np.asarray(z, float), depth, kp_base)
    out['bf_peak2'] = pk
    e2 = {}
    e2['peak2'] = run(kp=max(pk['k2'], 1e-3)) if pk is not None else None
    for s in E2_KP_SCALES:
        e2[f'kp_x{s}'] = run(kp=s * kp_base)
    out['E2'] = e2
    out['E3'] = {f'sig_x{s}': run(sig=s * sig_base) for s in E3_SIG_SCALES}
    out['sec'] = time.time() - t0
    return out


def main():
    with open(PREREG, 'rb') as f:
        ph = hashlib.sha256(f.read()).hexdigest()
    print(f"# PREREG sha256 {ph[:16]}  ({len(TARGETS)} target windows)", flush=True)
    t0 = time.time(); rows = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(window_job, j): j for j in TARGETS}
        for fu in as_completed(futs):
            j = futs[fu]
            try:
                r = fu.result()
            except Exception as e:
                print(f"  {j[0]}_w{j[1]}: FAIL {e!r}", flush=True)
                traceback.print_exc()
                rows.append(dict(record=j[0], win=j[1], error=repr(e)))
                continue
            rows.append(r)
            e1 = list(r['E1'].values())
            print(f"  {r['record']}_w{r['win']} base={r['baseline_recheck']:.3f} "
                  f"E1med={np.median(e1):.3f} E1best={min(e1):.3f} "
                  f"E2peak2={r['E2']['peak2'] if r['E2']['peak2'] is None else round(r['E2']['peak2'],3)} "
                  f"E2h={r['E2']['kp_x0.5']:.3f} E2d={r['E2']['kp_x2.0']:.3f} "
                  f"E3h={r['E3']['sig_x0.5']:.3f} E3d={r['E3']['sig_x2.0']:.3f} "
                  f"({r['sec']:.0f}s)", flush=True)
    rows.sort(key=lambda r: (r.get('arena', ''), r['record'], r['win']))
    json.dump(dict(rows=rows, prereg=PREREG, prereg_sha256=ph,
                   e1_seed_offsets=list(E1_SEED_OFFSETS),
                   e2_kp_scales=list(E2_KP_SCALES), e3_sig_scales=list(E3_SIG_SCALES),
                   bf=dict(n=BF_N, klim_mult=BF_KLIM_MULT, suppress_mult=BF_SUPPRESS),
                   targets=[list(t) for t in TARGETS],
                   elapsed_s=time.time() - t0), open(OUT, 'w'), indent=1)
    print(f"\nwrote {OUT}: {len(rows)} rows, {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
