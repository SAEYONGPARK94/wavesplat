"""Per-frequency sigma_1(f) for the records whose aperture can actually resolve it."""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from report import load_generic_calib, invert_generic


def main(fn='records_all.json'):
    R = json.load(open(os.path.join(HERE, fn)))['records']
    TAB = load_generic_calib()
    for r in R:
        sp = r.get('spec')
        if not sp:
            continue
        f = np.array(sp['f']); sig = np.array(sp['sigma1_deg'])
        S = np.array(sp['S']); kr = np.array(sp['ridge_k']); on = np.array(sp['onshell'])
        L = min(r['Lx_used_m'], r['Ly_used_m'])
        Etot = np.nansum(S)
        print(f"\n=== {r['site']}/{r['file']}  L={L:.0f} m  Tp={r['shell']['Tp']:.2f}s "
              f"lam_p={r.get('lambda_p_m', float('nan')):.0f} m ===")
        print('    f(Hz)  T(s)  lam(m)  L/lam   sig1raw  sig1corr  E%   onshell')
        ok = np.isfinite(sig) & np.isfinite(kr) & (kr > 0)
        idx = np.argsort(-np.where(ok, S, -np.inf))[:8]
        for i in sorted(idx):
            if not ok[i]:
                continue
            lam = 2 * np.pi / kr[i]
            c = invert_generic(TAB, L / lam, sig[i])
            print('  %7.3f %5.2f %7.1f %6.2f %8.1f %9s %5.1f %7.2f'
                  % (f[i], 1 / f[i], lam, L / lam, sig[i],
                     ('%.1f' % c) if np.isfinite(c) else '  --',
                     100 * S[i] / Etot, on[i]))


if __name__ == '__main__':
    main(*sys.argv[1:])
