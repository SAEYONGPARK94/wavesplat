"""How exposed is the 8-record learning headline to the patch-floored kp?

The pipeline (d3_sa_uspeed.run_window:205) calls characterize() on every window.
characterize() -> directional_spread() takes kp from the SPATIAL spectrum of one
frame on the patch. A patch of side L cannot represent any wavelength > L, so its
lowest non-zero wavenumber is k_min = 2*pi/L. If the sea's true kp < k_min, the
returned kp is FLOORED and everything downstream is wrong.

Downstream of kp:
  k_cut = 1.5*kp      -> the band-limited evaluation target gt_bl (ALL arms)
  cp    = om_p/kp     -> U = ratio*cp, the ship speeds of the moving conditions
  lam_p = 2*pi/kp     -> D_eff/lam_p, the aperture-law x-axis
  kp_hint=kp          -> mmse_bank_estimate AND net_gamma_mmse   (NOT RELAX)

So we print, per record, exactly what the pipeline sees vs the truth.
"""
import sys
import json
import numpy as np

sys.path.insert(0, 'scripts')
from d3_sa_gonogo import characterize  # noqa: E402
from d3_sa_multirecord import MRecord, RECORDS  # noqa: E402

G = 9.81
TRUE = {r['name']: r for r in json.load(open('tmp/record_wavelength.json'))}


def cp_of(k, d):
    om = np.sqrt(G * k * np.tanh(k * d))
    return om / k, 2 * np.pi / om


print(f"{'rec':<7}{'L':>5}{'k_min':>7}{'kp_pipe':>9}{'kp_true':>9}{'FLOOR?':>8}"
      f"{'kp err':>8}{'cp_pipe':>9}{'cp_true':>9}{'U err':>8}"
      f"{'lam_pipe':>10}{'lam_true':>10}")
print('-' * 100)
rows = []
for key, cfg in RECORDS.items():
    t = TRUE.get(key)
    if t is None:
        continue
    rec = MRecord(cfg['path'], stride=cfg.get('stride', 1))
    nf = int(round(cfg['T'] / rec.dt))
    # same window the pipeline would take (first of the linspace starts)
    Zwin = rec.window(200, nf)
    Hs_w, lam_p, kp, pdir, spread, Tp, cp = characterize(Zwin, rec, cfg['depth'])
    rec.f.close()

    L = min(np.ptp(rec.gx), np.ptp(rec.gy))
    k_min = 2 * np.pi / L                       # patch fundamental
    kp_true = 2 * np.pi / t['lam_temporal']
    cp_true, Tp_true = cp_of(kp_true, cfg['depth'])
    floored = kp_true < k_min

    rows.append(dict(rec=key, L=L, k_min=k_min, kp_pipe=kp, kp_true=kp_true,
                     floored=bool(floored), kp_err=kp / kp_true,
                     cp_pipe=cp, cp_true=cp_true, U_err=cp / cp_true,
                     lam_pipe=lam_p, lam_true=t['lam_temporal'],
                     k_cut_pipe=1.5 * kp, k_cut_true=1.5 * kp_true))
    r = rows[-1]
    print(f"{key:<7}{L:5.0f}{k_min:7.3f}{kp:9.3f}{kp_true:9.3f}"
          f"{('YES' if floored else '-'):>8}"
          f"{r['kp_err']:7.2f}x{cp:9.2f}{cp_true:9.2f}{r['U_err']:7.2f}x"
          f"{lam_p:10.1f}{t['lam_temporal']:10.1f}")

json.dump(rows, open('tmp/kp_exposure.json', 'w'), indent=1)

print("\nk_min = 2*pi/L is the LOWEST wavenumber the patch can represent.")
print("FLOOR? = YES means the sea's true peak lies BELOW the patch fundamental,")
print("         i.e. the spatial spectrum cannot see it at all -> kp is floored.")
print("\nexposure of the 3 arms to a wrong kp:")
print("  RELAX      : iterative_relax_cv(P, z, depth, target)      -> NO kp   (immune)")
print("  hand-MMSE  : mmse_bank_estimate(..., kp_hint=kp)          -> USES kp")
print("  net-MMSE   : net_gamma_mmse(..., kp_hint=kp)              -> USES kp")
print("  metric     : k_cut = 1.5*kp defines gt_bl                 -> ALL arms, paired")
print("  moving cond: U = ratio * cp                               -> ALL arms, paired")
