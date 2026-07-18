"""e56 — close E-55's pessimism gap: does external Uc pay MORE on a realistic
net base than on the oracle-partition base? E-55 audit flagged the oracle base
may ABSORB the Uc phase error (net/flat base -> more data-dependent -> Uc may
pay). This re-runs value_marginal|kp_hint with base = incumbent NET gamma (at
true kp), alongside the oracle base (must reproduce E-55).

If net-base value_marginal @ radar grades (4-7 cm/s) STILL < 0.03 -> pessimism
gap CLOSED, VALUE-death robust to base. If >= 0.03 -> oracle base hid value ->
current pays on realistic base -> NOT value-death.

Reuses the E-55 (audited) build_sea_uc + solver; only the base grid changes.
Usage: python3 tmp/e56_current_netbase.py [--smoke] [--n 48]
Out:   results/direction3/e56_current_netbase.json
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from e29_gennet_C import DEVICE, DEPTH, _MASK, corr, KXN_NP, KYN_NP, design_t
from e37_bimodal_ladder import net_gamma_map
from d18_relax_estimator import omega
from e55_current_channel import build_sea_uc, oracle_grid, GRADES

OUT = 'results/direction3/e56_current_netbase.json'


def net_grid(S, kp):
    """Incumbent net gamma at a given kp (10^Y, floored). Realistic base."""
    S2 = dict(S); S2['kp'] = kp
    return np.maximum(10.0 ** net_gamma_map(S2), 1e-8)


def solve_uc_base(S, kp, Ucx, Ucy, g_free):
    """E-55 solve_uc with a caller-supplied base grid (om Doppler-shifted by Uc)."""
    gb = np.zeros_like(g_free)
    g = np.concatenate([g_free[_MASK], gb[_MASK]])
    if g.sum() <= 0:
        return float('nan')
    g = g / g.sum()
    kxb = KXN_NP * kp; kyb = KYN_NP * kp
    om = omega(np.hypot(kxb, kyb) + 1e-12, DEPTH) + kxb * Ucx + kyb * Ucy
    kxb2 = np.concatenate([kxb, 2 * kxb]); kyb2 = np.concatenate([kyb, 2 * kyb])
    om2 = np.concatenate([om, 2 * om])
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=DEVICE)
    Phi = design_t(t(kxb2), t(kyb2), t(om2), t(S['P']))
    Phi_q = design_t(t(kxb2), t(kyb2), t(om2), t(S['Pq']))
    gam = t(np.concatenate([g, g]) * float(np.var(S['z'])))
    K = (Phi * gam[None]) @ Phi.T + (3.0 * S['sigma0']) ** 2 * torch.eye(
        len(S['z']), device=DEVICE)
    w = Phi.T @ torch.linalg.solve(K, t(S['z']))
    return float(corr(((Phi_q * gam[None]) @ w).cpu().numpy(), S['gt_u']))


def paired(rows, key):
    d = np.array([r[key] for r in rows if np.isfinite(r[key])])
    n = len(d); se = d.std(ddof=1) / np.sqrt(n)
    lo, hi = stats.t.interval(0.95, n - 1, loc=d.mean(), scale=se)
    return dict(mean=float(d.mean()), se=float(se), lo=float(lo), hi=float(hi),
                med=float(np.median(d)), win=int((d > 0).sum()), n=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--n', type=int, default=48)
    a = ap.parse_args()
    n = 3 if a.smoke else a.n
    grades = [0.0, 0.07] if a.smoke else GRADES
    t0 = time.time()

    per_grade = {g: [] for g in grades}
    print(f"--- current NET-BASE, si=800..{800+n-1}, grades {grades} ---", flush=True)
    for si in range(800, 800 + n):
        ang = np.random.default_rng(5000 + si).uniform(0, 2 * np.pi)
        for gmag in grades:
            Ucx, Ucy = gmag * np.cos(ang), gmag * np.sin(ang)
            try:
                S = build_sea_uc(si, Ucx, Ucy)
                kp = S['kp_true']
                go = oracle_grid(kp, S)
                gn = net_grid(S, kp)
                vm_o = solve_uc_base(S, kp, Ucx, Ucy, go) - solve_uc_base(S, kp, 0, 0, go)
                base_n0 = solve_uc_base(S, kp, 0, 0, gn)
                vm_n = solve_uc_base(S, kp, Ucx, Ucy, gn) - base_n0
            except Exception as e:
                print(f"  si{si} g{gmag} EXCLUDED ({type(e).__name__})", flush=True)
                continue
            per_grade[gmag].append(dict(si=si, vm_oracle=vm_o, vm_net=vm_n,
                                        net_base_corr=base_n0))
        if (si - 800) % 10 == 0:
            print(f"  si{si} done [{time.time()-t0:.0f}s]", flush=True)

    res = {}
    print(f"\n=== value_marginal|kp_hint: ORACLE base vs NET base ===")
    print(f"{'|Uc|':>6} {'vm_oracle':>24} {'vm_NET':>24} {'net_base_corr':>14}")
    for g in grades:
        rows = per_grade[g]
        vo = paired(rows, 'vm_oracle'); vn = paired(rows, 'vm_net')
        res[f'{g}'] = dict(vm_oracle=vo, vm_net=vn,
                           net_base_corr=float(np.mean([r['net_base_corr'] for r in rows])))
        so = '*' if (vo['lo'] > 0 or vo['hi'] < 0) else ' '
        sn = '*' if (vn['lo'] > 0 or vn['hi'] < 0) else ' '
        print(f"{g*100:>5.0f}c {vo['mean']:+.4f}[{vo['lo']:+.4f},{vo['hi']:+.4f}]{so}"
              f" {vn['mean']:+.4f}[{vn['lo']:+.4f},{vn['hi']:+.4f}]{sn}"
              f" {res[f'{g}']['net_base_corr']:>13.3f}")

    verdict = 'SMOKE'
    if not a.smoke:
        vn4 = res['0.04']['vm_net']; vn7 = res['0.07']['vm_net']
        closed = (vn4['hi'] < 0.03 and vn7['hi'] < 0.03)
        print(f"\n=== VERDICT (pessimism gap) ===")
        if closed:
            verdict = ("GAP CLOSED: net-base value_marginal @ radar grades still < 0.03 "
                       "=> VALUE-death robust to base choice (not an oracle-base artifact).")
        elif (vn4['lo'] > 0 and vn4['mean'] >= 0.03) or (vn7['lo'] > 0 and vn7['mean'] >= 0.03):
            verdict = ("REOPENED: net-base value_marginal @ radar >= 0.03 => oracle base HID "
                       "value; current pays on realistic base => NOT value-death.")
        else:
            verdict = "PARTIAL: net-base larger than oracle but not decisively >=0.03; report CIs."
        print(f"  -> {verdict}")

    json.dump(dict(per_grade={str(g): per_grade[g] for g in grades},
                   contrasts=res, verdict=verdict), open(OUT, 'w'), indent=1,
              default=float)
    print(f"saved -> {OUT}  ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
