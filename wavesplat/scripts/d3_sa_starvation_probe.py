"""
d3_sa_starvation_probe.py — What predicts fixed-sensor starvation (A_abs) and
the motion gain across records? Feature extraction + shallow-learnability test.

Context (2026-07-04): multi-record collapse falsified the universal D_eff/lam
law; the dominant structure is corr(gain, A_abs) ~ -0.5 ("motion pays only
where staying still is information-starved"), and the AA14 drill-down showed
the moving-track failure is an IDENTIFIABILITY breakdown (forcing more RELAX
components makes the field worse — components are not identifiable from
moving samples on broadband seas), not a CV bug.

Features per window (all computable from a fixed point sensor + depth chart,
i.e., operationally available BEFORE deciding to move):
  nu       — temporal spectral width sqrt(m0 m2 / m1^2 - 1) (broadbandedness)
  qp       — Goda peakedness of the point spectrum
  ipr_k    — effective number of occupied wavenumber cells of the 2D spatial
             spectrum, inverse participation ratio (identifiability proxy;
             needs an array/stereo in practice — flagged as such)
  ipr_dir  — effective number of occupied 22.5-deg direction bins
  steep    — kp Hs / 2
  kd       — kp * depth
  tcoh_tp, spread_deg, patch_lam — from the run JSON

Targets (from results/direction3/sa_multirecord.json):
  A_abs (RELAX/RIDGE), supra-gain (mean gain of conditions with D_eff>0.7 lam),
  swept-gain.

Learnability probe: record-level leave-one-record-out OLS on the top features
-> can a shallow rule predict where motion pays? (n=10 records — this is a
hypothesis probe, not a validated model.)

Usage: python3 scripts/d3_sa_starvation_probe.py
"""
import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from d3_sa_multirecord import MRecord, RECORDS

JSON = 'results/direction3/sa_multirecord.json'
OUT = 'results/direction3/sa_starvation_probe.json'


def window_features(Zwin, dt, dx, dy, kp, depth, rng, n_pix=100):
    nf = Zwin.shape[0]
    flat = Zwin.reshape(nf, -1)
    ok = np.where(np.isfinite(flat).all(axis=0))[0]
    sel = rng.choice(ok, size=min(n_pix, len(ok)), replace=False)
    S = flat[:, sel].astype(np.float64)
    S -= S.mean(axis=0)
    # mean point PSD (rfft)
    psd = np.mean(np.abs(np.fft.rfft(S, axis=0)) ** 2, axis=1)
    f = np.fft.rfftfreq(nf, dt)
    psd[0] = 0.0
    # restrict to a sane band (0.02 - 2 Hz) to keep moments off the noise floor
    band = (f > 0.02) & (f < 2.0)
    fb, pb = f[band], psd[band]
    m0 = np.trapezoid(pb, fb)
    m1 = np.trapezoid(fb * pb, fb)
    m2 = np.trapezoid(fb ** 2 * pb, fb)
    nu = float(np.sqrt(max(m0 * m2 / max(m1 ** 2, 1e-30) - 1.0, 0.0)))
    qp = float(2.0 / max(m0 ** 2, 1e-30) * np.trapezoid(fb * pb ** 2, fb))
    # 2D spatial spectrum, averaged over frames for stability
    n0, n1 = Zwin.shape[1:]
    acc = np.zeros((n0, n1))
    for fi in np.linspace(0, nf - 1, 16).astype(int):
        acc += np.abs(np.fft.fft2(np.nan_to_num(Zwin[fi]))) ** 2
    kx = 2 * np.pi * np.fft.fftfreq(n1, dx)
    ky = 2 * np.pi * np.fft.fftfreq(n0, dy)
    KX, KY = np.meshgrid(kx, ky)
    K = np.hypot(KX, KY)
    sea = (K > 0.3 * kp) & (K < 3.0 * kp)          # exclude DC/bias and noise floor
    E = acc[sea]
    ipr_k = float((E.sum() ** 2) / max((E ** 2).sum(), 1e-30))
    ang = np.arctan2(KY, KX)[sea]
    bins = np.linspace(-np.pi, np.pi, 17)
    Eb = np.array([E[(ang >= a) & (ang < b)].sum()
                   for a, b in zip(bins[:-1], bins[1:])])
    p = Eb / max(Eb.sum(), 1e-30)
    ipr_dir = float(1.0 / max((p ** 2).sum(), 1e-30))
    return dict(nu=nu, qp=qp, ipr_k=ipr_k, ipr_dir=ipr_dir)


def targets_from_json(store):
    rows = []
    for key, rec in store.items():
        lam_ref = rec.get('lam_ref')
        for wi, w in enumerate(rec['windows']):
            row = dict(record=key, wi=wi, i0=w['i0'],
                       spread=w['spread_deg'], tcoh_tp=w['t_coh_s'] / w['Tp'],
                       patch_lam=w['L_diag'] / w['lam_p'],
                       steep=w['kp'] * w['Hs_win'] / 2.0,
                       kd=w['kp'] * rec['depth'],
                       lam_p=w['lam_p'], kp=w['kp'])
            for which, tag in (('mean', 'rx'), ('mean_ridge', 'rd')):
                A = w['conditions']['A'][which]['common_unobs']['corr_bl']
                sup, swp = [], []
                for name, e in w['conditions'].items():
                    if not name.startswith('B_r') or '_L' in name:
                        continue
                    lam = lam_ref if lam_ref else w['lam_p']
                    if e['D_eff_m'] / lam > 0.7:
                        sup.append(e[which]['common_unobs']['corr_bl'] - A)
                    if 'swept' in e[which]:
                        swp.append(e[which]['swept']['corr_bl']
                                   - w['conditions']['A'][which]['swept']['corr_bl'])
                row[f'A_abs_{tag}'] = A
                row[f'gain_sup_{tag}'] = float(np.mean(sup)) if sup else np.nan
                row[f'gain_swp_{tag}'] = float(np.mean(swp)) if swp else np.nan
            rows.append(row)
    return rows


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4:
        return np.nan
    rx = np.argsort(np.argsort(x[m])).astype(float)
    ry = np.argsort(np.argsort(y[m])).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    store = json.load(open(JSON))
    rows = targets_from_json(store)

    # feature extraction (window reload)
    cache = {}
    for key, cfg in RECORDS.items():
        if key not in store:
            continue
        rec = MRecord(cfg['path'], stride=cfg.get('stride', 1))
        nf = int(round(cfg['T'] / rec.dt))
        for row in rows:
            if row['record'] != key:
                continue
            Zwin = rec.window(row['i0'], nf)
            row.update(window_features(Zwin, rec.dt, rec.dx, rec.dy, row['kp'],
                                       cfg['depth'],
                                       np.random.default_rng(row['i0'])))
            del Zwin
        rec.f.close()
        print(f"features done: {key}")

    feats = ['nu', 'qp', 'ipr_k', 'ipr_dir', 'spread', 'tcoh_tp', 'patch_lam',
             'steep', 'kd']
    tgts = ['A_abs_rx', 'A_abs_rd', 'gain_sup_rx', 'gain_sup_rd', 'gain_swp_rx']

    print("\n=== window-level Spearman (n=56) ===")
    print(f"{'feature':10s}" + "".join(f"{t:>13s}" for t in tgts))
    for ftr in feats:
        line = f"{ftr:10s}"
        for t in tgts:
            line += f"{spearman([r[ftr] for r in rows], [r[t] for r in rows]):>13.2f}"
        print(line)

    # record-level (n = #records)
    recs = sorted({r['record'] for r in rows})
    rl = {k: {ftr: float(np.nanmean([r[ftr] for r in rows if r['record'] == k]))
              for ftr in feats + tgts} for k in recs}
    print(f"\n=== record-level Spearman (n={len(recs)}) ===")
    print(f"{'feature':10s}" + "".join(f"{t:>13s}" for t in tgts))
    for ftr in feats:
        line = f"{ftr:10s}"
        for t in tgts:
            line += (f"{spearman([rl[k][ftr] for k in recs], [rl[k][t] for k in recs]):>13.2f}")
        print(line)

    # LORO shallow learnability: predict record-level gain_sup_rx from top-2
    # features by |record-level spearman| (chosen per fold on train records)
    print("\n=== LORO shallow rule (record-level, target gain_sup_rx) ===")
    y_true, y_pred = [], []
    for held in recs:
        tr = [k for k in recs if k != held]
        scores = {f: abs(spearman([rl[k][f] for k in tr],
                                  [rl[k]['gain_sup_rx'] for k in tr]))
                  for f in feats}
        top2 = sorted(scores, key=scores.get, reverse=True)[:2]
        X = np.array([[rl[k][f] for f in top2] + [1.0] for k in tr])
        yv = np.array([rl[k]['gain_sup_rx'] for k in tr])
        m = np.isfinite(yv)
        beta, *_ = np.linalg.lstsq(X[m], yv[m], rcond=None)
        xh = np.array([rl[held][f] for f in top2] + [1.0])
        yh = rl[held]['gain_sup_rx']
        if np.isfinite(yh):
            y_true.append(yh)
            y_pred.append(float(xh @ beta))
            print(f"  {held:7s} top2={top2} pred={y_pred[-1]:+.3f} true={yh:+.3f}")
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    sign_acc = float(np.mean(np.sign(y_pred) == np.sign(y_true)))
    ss = 1 - np.sum((y_pred - y_true) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
    print(f"  LORO sign accuracy {sign_acc:.2f}, R^2 {ss:+.2f} "
          f"(baseline sign acc of always-negative: "
          f"{float(np.mean(y_true < 0)):.2f})")

    json.dump(dict(rows=rows, record_level=rl,
                   loro=dict(y_true=list(y_true), y_pred=list(y_pred),
                             sign_acc=sign_acc, r2=float(ss))),
              open(OUT, 'w'), indent=1, default=float)
    print(f"saved {OUT}")


if __name__ == '__main__':
    main()
