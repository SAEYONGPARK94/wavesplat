"""B-16 러너 — 표현 절제(파수 격자 스냅). 샤딩: `python3 run_b16.py <shard> <nshards>`

★배포 진입점 `wavesplat.run.run_estimation` 을 `DEFAULT_CONFIG` 그대로 부르고
`k_snap_dk` **하나만** 바꾼다. 개입 계수기(`k_snap_fired`·`k_snap_cells`)를 행에 싣는다.
"""
import glob
import json
import os
import re
import sys
import time
import warnings

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for p in (HERE, ROOT, os.path.join(ROOT, 'tmp', 'B11_HOS_BASELINE_2026-08-23'),
          os.path.join(ROOT, 'tmp', 'B12_HOS_EXTEND_2026-08-23')):
    sys.path.insert(0, p)
os.chdir(ROOT)
import gates_b16 as G                                       # noqa: E402


def episodes():
    out = []
    for p in sorted(glob.glob(os.path.join(G.HOS, '*.h5'))):
        m = re.match(r'episode_(BF\d)_(\d+)comp_seed(\d+)_(.+)\.h5', os.path.basename(p))
        if m and int(m.group(2)) >= 15 and int(m.group(3)) in G.ALL_SEEDS:
            out.append(dict(path=p, episode=os.path.basename(p)[len('episode_'):-3],
                            bf=m.group(1), ncomp=int(m.group(2)), gen_seed=int(m.group(3))))
    return out


def run_one(ep, arm, est_seed):
    from wavesplat.run import run_estimation, DEFAULT_CONFIG
    cfg = dict(DEFAULT_CONFIG)
    cfg['seed'] = est_seed
    cfg.update(G.ARMS[arm])
    t0 = time.time()
    r = run_estimation(ep['path'], config=cfg, verbose=False)
    return dict(arm=arm, est_seed=est_seed, episode=ep['episode'], bf=ep['bf'],
                ncomp=ep['ncomp'], gen_seed=ep['gen_seed'],
                avg_nRMSE=float(r['avg_nRMSE']),
                zero_nRMSE=(float(r['zero_nRMSE']) if r.get('zero_nRMSE') is not None else None),
                k_snap_dk=r.get('k_snap_dk'), k_snap_fired=bool(r.get('k_snap_fired')),
                k_snap_cells=r.get('k_snap_cells'),
                k_snap_shift_median=r.get('k_snap_shift_median'),
                varpro_steps=cfg.get('varpro_steps'),
                learned_init_iters=cfg.get('learned_init_iters'),
                auto_switch_thresh=cfg.get('auto_switch_thresh'),
                lambda_l1=cfg.get('lambda_l1'), final_n=r.get('final_n'),
                wall_s=time.time() - t0)


def main():
    shard = int(sys.argv[1]) if len(sys.argv) > 2 else 0
    nsh = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    out = os.path.join(HERE, f'results_b16_{shard}.jsonl')
    done = set()
    if os.path.exists(out):
        for l in open(out):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            if 'error' not in r:
                done.add((r['arm'], r['episode'], r['est_seed']))
    eps = [e for i, e in enumerate(episodes()) if i % nsh == shard]
    print(f'shard {shard}/{nsh}: 에피소드 {len(eps)}개', flush=True)
    with open(out, 'a') as fh:
        for ep in eps:
            for arm in G.ARM_ORDER:
                for s in G.EST_SEEDS:
                    if (arm, ep['episode'], s) in done:
                        continue
                    try:
                        row = run_one(ep, arm, s)
                    except Exception as e:                  # noqa: BLE001
                        import traceback
                        row = dict(arm=arm, episode=ep['episode'], est_seed=s,
                                   error=repr(e), tb=traceback.format_exc()[-400:])
                    fh.write(json.dumps(row) + '\n')
                    fh.flush()
                    print(f"{arm:8s} {ep['episode'][:38]:38s} s{s} "
                          f"{row.get('error') or format(row['avg_nRMSE'], '.4f')}"
                          f"{'' if row.get('error') else ' snap=' + str(row['k_snap_fired'])}",
                          flush=True)


if __name__ == '__main__':
    main()
