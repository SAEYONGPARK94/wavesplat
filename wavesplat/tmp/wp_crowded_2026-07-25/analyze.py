"""Assemble the paired head-to-head per PREREG.md. Prints the prereg verdict, the full paired
table, by-regime splits, stochasticity accounting, sparse control, and the flagged extra.
Frozen decision rule: PASS iff (one-sided Wilcoxon p<0.05, cond<ws_best) AND (median Δ<0).
"""
import json, math
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon

WS = 'tmp/wp_crowded_2026-07-25/ws_runs.jsonl'
COND = 'tmp/wp_crowded_2026-07-25/cond_runs.jsonl'

CROWDED = [f'episode_BF{bf}_{nc}comp_seed{s}_{arm}.h5'
           for bf in (4, 5) for nc in (15, 30) for s in range(5)
           for arm in ('uniform', 'slope')]
SPARSE = [
    'episode_BF2_1comp_seed0_uniform.h5', 'episode_BF2_1comp_seed0_slope.h5',
    'episode_BF3_1comp_seed0_uniform.h5', 'episode_BF3_2comp_seed0_uniform.h5',
    'episode_BF3_2comp_seed0_slope.h5',  'episode_BF3_5comp_seed0_uniform.h5',
    'episode_BF3_5comp_seed1_uniform.h5', 'episode_BF3_5comp_seed0_slope.h5',
    'episode_BF4_1comp_seed0_uniform.h5', 'episode_BF4_5comp_seed0_uniform.h5',
    'episode_BF4_5comp_seed0_slope.h5',  'episode_BF4_5comp_seed1_uniform.h5',
]
EXTRA = ['episode_BF4_20comp_seed7_uniform.h5']


def load():
    ws = defaultdict(lambda: {'init': [], 'track': []})
    ws_err = []
    for line in open(WS):
        d = json.loads(line)
        if d['err']:
            ws_err.append((d['episode'], d['seed'], d['err'])); continue
        ws[d['episode']]['init'].append(d['init_nrmse'])
        ws[d['episode']]['track'].append(d['avg_nRMSE'])
    cond = defaultdict(list)
    cond_err = []
    for line in open(COND):
        d = json.loads(line)
        if d['err']:
            cond_err.append((d['episode'], d['seed'], d['err'])); continue
        cond[d['episode']].append(d['cond_polish_nrmse_Hs'])
    return ws, cond, ws_err, cond_err


def med_sd(xs):
    a = np.array(xs, float)
    return float(np.median(a)), float(np.std(a)), len(a)


def parse(ep):
    bf = int(ep.split('_')[1][2:])
    nc = int(ep.split('comp')[0].split('_')[-1])
    arm = 'slope' if 'slope' in ep else 'uniform'
    return bf, nc, arm


def build_rows(eplist, ws, cond):
    rows = []
    for ep in eplist:
        wi, wisd, ni = med_sd(ws[ep]['init'])
        wt, wtsd, nt = med_sd(ws[ep]['track'])
        cm, csd, ncn = med_sd(cond[ep])
        wsbest = min(wi, wt)
        bf, nc, arm = parse(ep)
        rows.append(dict(ep=ep, bf=bf, nc=nc, arm=arm,
                         ws_init=wi, ws_init_sd=wisd, ws_track=wt, ws_track_sd=wtsd,
                         ws_best=wsbest, cond=cm, cond_sd=csd,
                         d_best=cm - wsbest, d_track=cm - wt,
                         n_ws=ni, n_cond=ncn))
    return rows


def wilcox_report(rows, key='d_best'):
    cond = np.array([r['cond'] for r in rows])
    ref = np.array([r['ws_best'] if key == 'd_best' else r['ws_track'] for r in rows])
    d = cond - ref
    # one-sided: cond < ref
    try:
        stat, p = wilcoxon(cond, ref, alternative='less', zero_method='wilcox')
    except Exception as e:
        stat, p = float('nan'), float('nan')
    # rank-biserial from signed ranks
    nz = d[d != 0]
    ranks = np.argsort(np.argsort(np.abs(nz))) + 1
    wp = ranks[nz > 0].sum(); wm = ranks[nz < 0].sum()
    tot = wp + wm
    rrb = (wm - wp) / tot if tot else float('nan')   # positive => favors cond (cond smaller)
    return dict(p=float(p), med_d=float(np.median(d)), mean_d=float(np.mean(d)),
                wins=int((d < 0).sum()), n=len(d), rrb=float(rrb),
                cond_mean=float(cond.mean()), ref_mean=float(ref.mean()))


def fmt_split(name, rows):
    r = wilcox_report(rows, 'd_best')
    return (f"  {name:20s} n={r['n']:2d}  cond {r['cond_mean']:.3f}  ws_best {r['ref_mean']:.3f}  "
            f"medΔ {r['med_d']:+.3f}  wins {r['wins']}/{r['n']}  rrb {r['rrb']:+.2f}  p={r['p']:.2e}")


def main():
    ws, cond, ws_err, cond_err = load()
    print("=== data integrity ===")
    print(f"  WS errored runs: {len(ws_err)}; cond errored runs: {len(cond_err)}")
    for ep in CROWDED:
        if len(ws[ep]['init']) != 5 or len(cond[ep]) != 5:
            print(f"  INCOMPLETE {ep}: ws={len(ws[ep]['init'])} cond={len(cond[ep])}")

    crow = build_rows(CROWDED, ws, cond)

    print("\n=== PRIMARY: 40 crowded, cond vs ws_best=min(WSinit,WStracked) ===")
    prim = wilcox_report(crow, 'd_best')
    P1 = prim['p'] < 0.05
    P2 = prim['med_d'] < 0
    verdict = 'PASS' if (P1 and P2) else 'FAIL'
    print(f"  cond mean {prim['cond_mean']:.4f}  ws_best mean {prim['ref_mean']:.4f}")
    print(f"  median Δ(cond−ws_best) {prim['med_d']:+.4f}  mean Δ {prim['mean_d']:+.4f}")
    print(f"  wins(cond better) {prim['wins']}/{prim['n']}  rank-biserial {prim['rrb']:+.3f}")
    print(f"  one-sided Wilcoxon p = {prim['p']:.3e}")
    print(f"  P1 (p<0.05): {P1}   P2 (medΔ<0): {P2}   ==> PRIMARY {verdict}")

    print("\n=== SECONDARY: cond vs WS tracked (paper product) ===")
    sec = wilcox_report(crow, 'd_track')
    print(f"  cond mean {sec['cond_mean']:.4f}  ws_track mean {sec['ref_mean']:.4f}  "
          f"medΔ {sec['med_d']:+.4f}  wins {sec['wins']}/{sec['n']}  p={sec['p']:.2e}")

    print("\n=== by-comp / by-BF / by-arm splits (cond vs ws_best) ===")
    print(fmt_split('15comp', [r for r in crow if r['nc'] == 15]))
    print(fmt_split('30comp', [r for r in crow if r['nc'] == 30]))
    print(fmt_split('BF4', [r for r in crow if r['bf'] == 4]))
    print(fmt_split('BF5', [r for r in crow if r['bf'] == 5]))
    print(fmt_split('uniform', [r for r in crow if r['arm'] == 'uniform']))
    print(fmt_split('slope', [r for r in crow if r['arm'] == 'slope']))

    print("\n=== full paired table (crowded) ===")
    print(f"{'episode':42s}{'nc':>3s}{'cond(sd)':>14s}{'WSinit(sd)':>14s}"
          f"{'WStrk(sd)':>14s}{'ws_best':>9s}{'Δbest':>8s}")
    for r in sorted(crow, key=lambda r: (r['nc'], r['bf'], r['arm'], r['ep'])):
        print(f"{r['ep']:42s}{r['nc']:3d}"
              f"{r['cond']:8.3f}({r['cond_sd']:.3f}){r['ws_init']:8.3f}({r['ws_init_sd']:.3f})"
              f"{r['ws_track']:8.3f}({r['ws_track_sd']:.3f}){r['ws_best']:9.3f}{r['d_best']:+8.3f}")

    print("\n=== SPARSE CONTROL (expected WS-wins) ===")
    sp = build_rows(SPARSE, ws, cond)
    for ep in SPARSE:
        if len(ws[ep]['init']) != 5 or len(cond[ep]) != 5:
            print(f"  INCOMPLETE {ep}: ws={len(ws[ep]['init'])} cond={len(cond[ep])}")
    spr = wilcox_report(sp, 'd_best')
    print(f"  cond mean {spr['cond_mean']:.4f}  ws_best mean {spr['ref_mean']:.4f}  "
          f"medΔ {spr['med_d']:+.4f}  cond-wins {spr['wins']}/{spr['n']}  p(one-sided cond<)={spr['p']:.2e}")
    print(f"{'episode':42s}{'nc':>3s}{'cond(sd)':>14s}{'WSinit(sd)':>14s}{'WStrk(sd)':>14s}{'ws_best':>9s}{'Δbest':>8s}")
    for r in sorted(sp, key=lambda r: (r['nc'], r['bf'], r['ep'])):
        print(f"{r['ep']:42s}{r['nc']:3d}"
              f"{r['cond']:8.3f}({r['cond_sd']:.3f}){r['ws_init']:8.3f}({r['ws_init_sd']:.3f})"
              f"{r['ws_track']:8.3f}({r['ws_track_sd']:.3f}){r['ws_best']:9.3f}{r['d_best']:+8.3f}")

    print("\n=== FLAGGED EXTRA (BF4_20comp, DEGRADED set, secondary only) ===")
    ex = build_rows(EXTRA, ws, cond)
    for r in ex:
        print(f"  {r['ep']}: cond {r['cond']:.3f}(sd{r['cond_sd']:.3f}) "
              f"WSinit {r['ws_init']:.3f} WStrk {r['ws_track']:.3f} ws_best {r['ws_best']:.3f} "
              f"Δbest {r['d_best']:+.3f}")

    print("\n=== stochasticity accounting (mean per-episode sd) ===")
    print(f"  crowded: WS_init sd {np.mean([r['ws_init_sd'] for r in crow]):.4f}  "
          f"WS_track sd {np.mean([r['ws_track_sd'] for r in crow]):.4f}  "
          f"cond sd {np.mean([r['cond_sd'] for r in crow]):.4f}")

    # dump machine-readable
    json.dump({'primary': prim, 'secondary_track': sec,
               'crowded_rows': crow, 'sparse_rows': sp, 'extra_rows': ex,
               'verdict': verdict, 'P1': P1, 'P2': P2},
              open('tmp/wp_crowded_2026-07-25/results.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
