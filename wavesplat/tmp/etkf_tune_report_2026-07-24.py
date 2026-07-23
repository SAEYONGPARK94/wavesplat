"""Render the tuning-stage JSONs into the tables used by ETKF_FINEMESH_2026-07-24.md."""
import json, os, sys
import numpy as np

ROOT = '/home/vexit/wavesplat/wavesplat'
short = lambda e: e.replace('episode_', '').replace('.h5', '') \
                   .replace('_seed0', '').replace('_seed1', '') \
                   .replace('_uniform', ' U').replace('_slope', ' S')


def L(stage):
    return json.load(open(os.path.join(ROOT, 'tmp',
                                       f'etkf_tune_stage{stage}_2026-07-24.json')))


def table(summary, title, keys=None):
    rows = sorted(summary.items(), key=lambda kv: kv[1]['mean'])
    print(f'\n### {title}  (mean nRMSE over 12 TUNING episodes, median of 3 filter seeds)')
    print('| rank | config | mean nRMSE | mean rho | ms/frame |')
    print('|---|---|---|---|---|')
    for i, (k, v) in enumerate(rows, 1):
        c = json.loads(k)
        s = ', '.join(f'{a}={c[a]}' for a in (keys or sorted(c)))
        print(f"| {i} | {s} | {v['mean']:.4f} | {v['mean_rho']:.3f} | {v['ms']:.1f} |")


if __name__ == '__main__':
    which = sys.argv[1:] or ['A1', 'A2', 'A3', 'C']
    for st in which:
        if st == 'C':
            continue
        try:
            d = L(st)
        except FileNotFoundError:
            print(f'(stage {st} not present yet)'); continue
        table(d['summary'], f'Stage {st}')
    if 'C' in which:
        try:
            d = L('C')
        except FileNotFoundError:
            print('(stage C not present yet)'); raise SystemExit
        fin = json.dumps(d['final_cfg'], sort_keys=True)
        shp = json.dumps(d['shipped_cfg'], sort_keys=True)
        pap = json.dumps(d['paper_tuned_cfg'], sort_keys=True)
        NEW, OLD = 'data_finemesh_2026-07-23', 'data'
        eps = list(d[NEW][fin]['per_episode'].keys())
        print('\n### Stage C — REPORT set (seed 0), median of 7 filter seeds\n')
        print('| episode | ETKF as-published (fine) | ETKF paper-tuned (fine) | '
              '**ETKF re-tuned (fine)** | rho re-tuned | ETKF re-tuned (old) | '
              'ETKF paper-tuned (old) |')
        print('|---|---|---|---|---|---|---|')
        acc = {k: [] for k in ('fs', 'fp', 'ff', 'os', 'op', 'of')}
        for e in eps:
            fs = d[NEW][shp]['per_episode'][e]; fp = d[NEW][pap]['per_episode'][e]
            ff = d[NEW][fin]['per_episode'][e]; rr = d[NEW][fin]['per_episode_rho'][e]
            of = d[OLD][fin]['per_episode'][e]; op = d[OLD][pap]['per_episode'][e]
            os_ = d[OLD][shp]['per_episode'][e]
            sp = d[NEW][fin]['spread'][e]
            for k, v in zip(('fs', 'fp', 'ff', 'os', 'op', 'of'),
                            (fs, fp, ff, os_, op, of)):
                acc[k].append(v)
            print(f"| {short(e)} | {fs:.4f} | {fp:.4f} | **{ff:.4f}** "
                  f"[{sp[0]:.4f},{sp[1]:.4f}] | {rr:.3f} | {of:.4f} | {op:.4f} |")
        print(f"| **mean** | **{np.mean(acc['fs']):.4f}** | **{np.mean(acc['fp']):.4f}** "
              f"| **{np.mean(acc['ff']):.4f}** | "
              f"{np.mean([d[NEW][fin]['per_episode_rho'][e] for e in eps]):.3f} | "
              f"**{np.mean(acc['of']):.4f}** | **{np.mean(acc['op']):.4f}** |")
        print('\nfinal cfg:', d['final_cfg'])
        print('shipped  :', d['shipped_cfg'])
        print('paper    :', d['paper_tuned_cfg'])
        print('final cfg on TUNING set mean nRMSE:',
              round(d['final_cfg_on_tuning_set']['mean'], 4))
        json.dump({'per_episode_finemesh_retuned':
                   {short(e): d[NEW][fin]['per_episode'][e] for e in eps},
                   'per_episode_finemesh_retuned_spread':
                   {short(e): d[NEW][fin]['spread'][e] for e in eps},
                   'per_episode_finemesh_retuned_rho':
                   {short(e): d[NEW][fin]['per_episode_rho'][e] for e in eps},
                   'per_episode_finemesh_aspublished':
                   {short(e): d[NEW][shp]['per_episode'][e] for e in eps},
                   'per_episode_finemesh_papertuned':
                   {short(e): d[NEW][pap]['per_episode'][e] for e in eps},
                   'per_episode_old_retuned':
                   {short(e): d[OLD][fin]['per_episode'][e] for e in eps},
                   'per_episode_old_papertuned':
                   {short(e): d[OLD][pap]['per_episode'][e] for e in eps},
                   'final_cfg': d['final_cfg'], 'shipped_cfg': d['shipped_cfg'],
                   'paper_tuned_cfg': d['paper_tuned_cfg'],
                   'tuning_set': d['tuning_set'], 'report_set': d['report_set']},
                  open(os.path.join(ROOT, 'tmp',
                                    'ETKF_FINEMESH_2026-07-24_numbers.json'), 'w'),
                  indent=1)
