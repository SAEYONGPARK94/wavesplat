"""P2 (2026-07-08): standardized KPI report — replaces "0-10s pooled" single
numbers (verdict of GAP_RMSE_persona_debate_2026-07-08.md).

For every falcon_real_operator_<REC>.json (and its _damp twin when present)
emits, per record and per memory W:
  - lead-resolved nRMSE/corr (fixed per-record normalizer),
  - band KPIs: 0-2s (FF requirement), 0-5s, 0-10s, 10-20s,
    with window-bootstrap CIs and the persistence baseline,
  - damped-vs-raw comparison when pred_damp exists (P1).
Writes results/direction3/kpi_report.json + docs/KPI_REPORT_2026-07-08.md.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

RES = 'results/direction3'
BANDS = [(0, 2, '0-2s'), (0, 5, '0-5s'), (0, 10, '0-10s'), (10, 20, '10-20s')]
RECS = ['YS17', 'AA15a', 'AA14', 'AA15b', 'LJ18g']


def pool(windows, leads, W, lo, hi, key='pred'):
    ps, gs = [], []
    for w in windows:
        gt = np.asarray(w['gt'], float)
        src = w['by_W'][W].get(key)
        if src is None:
            return None, None
        pr = np.asarray(src, float) if key != 'persist' else \
            np.full_like(gt, gt[0])
        n = min(len(pr), len(gt), len(leads))
        m = (leads[:n] >= lo) & (leads[:n] <= hi) & \
            np.isfinite(gt[:n]) & np.isfinite(pr[:n])
        ps.append(pr[:n][m]); gs.append(gt[:n][m])
    return np.concatenate(ps), np.concatenate(gs)


def persist_pool(windows, leads, lo, hi):
    ps, gs = [], []
    for w in windows:
        gt = np.asarray(w['gt'], float)
        pr = np.full_like(gt, gt[0])
        n = min(len(gt), len(leads))
        m = (leads[:n] >= lo) & (leads[:n] <= hi) & \
            np.isfinite(gt[:n]) & np.isfinite(pr[:n])
        ps.append(pr[:n][m]); gs.append(gt[:n][m])
    return np.concatenate(ps), np.concatenate(gs)


def nrmse_corr(p, g):
    if p is None or len(g) < 4 or np.std(g) == 0:
        return None
    return dict(nrmse=float(np.sqrt(np.mean((p - g) ** 2)) / np.std(g)),
                corr=float(np.corrcoef(p, g)[0, 1]), n=int(len(g)))


def boot_ci(windows, leads, W, lo, hi, key='pred', n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        sel = rng.choice(len(windows), len(windows), replace=True)
        p, g = pool([windows[i] for i in sel], leads, W, lo, hi, key)
        if p is None or len(g) < 4 or np.std(g) == 0:
            continue
        vals.append(np.sqrt(np.mean((p - g) ** 2)) / np.std(g))
    if not vals:
        return None
    return [float(np.percentile(vals, q)) for q in (2.5, 97.5)]


def lead_curve(windows, leads, W, key='pred'):
    gall = np.concatenate([np.asarray(w['gt'], float)[
        np.isfinite(w['gt'])] for w in windows])
    sg = np.std(gall)
    row = []
    for li in range(len(leads)):
        es = []
        for w in windows:
            gt = np.asarray(w['gt'], float)
            src = w['by_W'][W].get(key)
            if src is None:
                return None
            pr = np.asarray(src, float)
            n = min(len(gt), len(pr))
            if li < n and np.isfinite(gt[li]) and np.isfinite(pr[li]):
                es.append(pr[li] - gt[li])
        row.append(float(np.sqrt(np.mean(np.square(es)))) / sg
                   if len(es) >= 4 else None)
    return row


def main():
    report = {}
    md = ["# KPI 리포트 — 리드-분해 표준 보고 (2026-07-08)",
          "",
          "생성: `scripts/d3_report_kpi.py` (P2 처방 — pooled 단일수치 폐기).",
          "nRMSE 정규화 = 레코드-pooled GT std. persist = 관성 베이스라인.",
          ""]
    for rec in RECS:
        path = f'{RES}/falcon_real_operator_{rec}.json'
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        dp = None
        dpath = f'{RES}/falcon_real_operator_{rec}_damp.json'
        if os.path.exists(dpath):
            dp = json.load(open(dpath))
        leads = np.asarray(d['leads'], float)
        wins = d['windows']
        ent = {'Hs': float(np.mean([w['Hs'] for w in wins])),
               'n_win': len(wins), 'bands': {}, 'lead_curves': {}}
        md.append(f"## {rec} (Hs {ent['Hs']:.2f} m, {len(wins)} windows)")
        md.append("")
        md.append("| W | band | nRMSE [95% CI] | corr | persist "
                  "| damp nRMSE | T_c med |")
        md.append("|---|---|---|---|---|---|---|")
        for W in [str(float(x)) for x in d['w_mems']]:
            ent['lead_curves'][W] = lead_curve(wins, leads, W)
            for lo, hi, bl in BANDS:
                p, g = pool(wins, leads, W, lo, hi)
                s = nrmse_corr(p, g)
                if s is None:
                    continue
                ci = boot_ci(wins, leads, W, lo, hi)
                pp, gg = persist_pool(wins, leads, lo, hi)
                sp = nrmse_corr(pp, gg)
                dmp, tcm = '', ''
                if dp is not None:
                    dwins = dp['windows']
                    dleads = np.asarray(dp['leads'], float)
                    if W in dwins[0]['by_W'] and \
                            'pred_damp' in dwins[0]['by_W'][W]:
                        pd_, gd_ = pool(dwins, dleads, W, lo, hi,
                                        'pred_damp')
                        sd = nrmse_corr(pd_, gd_)
                        if sd:
                            dmp = f"{sd['nrmse']:.2f}"
                        tcs = [w['by_W'][W].get('T_c') for w in dwins]
                        tcs = [t for t in tcs if t]
                        if tcs:
                            tcm = f"{np.median(tcs):.0f}s"
                s['ci'] = ci
                s['persist_nrmse'] = sp['nrmse'] if sp else None
                ent['bands'].setdefault(bl, {})[W] = s
                ci_s = f"[{ci[0]:.2f}-{ci[1]:.2f}]" if ci else ""
                md.append(
                    f"| {float(W):.0f} | {bl} | {s['nrmse']:.2f} {ci_s} "
                    f"| {s['corr']:+.2f} | {s['persist_nrmse']:.2f} "
                    f"| {dmp} | {tcm} |")
        md.append("")
        report[rec] = ent
    json.dump(report, open(f'{RES}/kpi_report.json', 'w'), indent=1)
    open('docs/KPI_REPORT_2026-07-08.md', 'w').write("\n".join(md) + "\n")
    print("wrote", f'{RES}/kpi_report.json', "and docs/KPI_REPORT_2026-07-08.md")
    # console: requirement-band summary
    print("\n요구대역(0-2s) 요약 (mid W):")
    for rec, ent in report.items():
        Ws = sorted(ent['bands'].get('0-2s', {}).keys(), key=float)
        if not Ws:
            continue
        W = Ws[len(Ws) // 2]
        s = ent['bands']['0-2s'][W]
        print(f"  {rec:6} W={float(W):.0f}: nRMSE {s['nrmse']:.2f} "
              f"{s['ci']} corr {s['corr']:+.2f} (persist "
              f"{s['persist_nrmse']:.2f})")


if __name__ == '__main__':
    main()
