"""B-15 러너 — 같은 파랑장 위에서 **센서 모형만** 바꿔 헤드라인을 다시 잰다.

★기준선 정의를 손으로 옮기지 않는다 — `S16` 의 `run_s16.run_letkf` 를 그대로 import.
★`SHIP-NOW` 는 배포 진입점 `wavesplat.run.run_estimation` 을 `DEFAULT_CONFIG` 그대로.
샤딩: `python3 run_b15.py <shard> <nshards>` → `results_b15_<shard>.jsonl`
"""
import glob
import json
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')
import numpy as np                                        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for p in (HERE, ROOT, os.path.join(ROOT, 'tmp', 'S16_BASELINE_2026-08-14'),
          os.path.join(ROOT, 'tmp', 'B11_HOS_BASELINE_2026-08-23')):
    sys.path.insert(0, p)
os.chdir(ROOT)
import gates_b15 as G                                     # noqa: E402


def episodes():
    out = []
    for bf in G.BFS:
        for nc in G.NCOMPS:
            for sd in G.GEN_SEEDS:
                for rd in G.RAYDROPS:
                    name = f'episode_{bf}_{nc}comp_seed{sd}_{rd}'
                    p = os.path.join(ROOT, 'data', name + '.h5')
                    if os.path.exists(p):
                        out.append(dict(path=p, episode=name, bf=bf, ncomp=nc,
                                        gen_seed=sd, raydrop=rd,
                                        pair=f'{bf}_{nc}comp_seed{sd}'))
    return out


def obs_stats(path):
    """개입 계수기 — 반환점 수와 반환된 점의 평균 입사각(도)."""
    import h5py
    with h5py.File(path, 'r') as f:
        v = f['frames/lidar_valid_mask'][...].astype(bool)
        a = f['frames/lidar_incidence_angles'][...]
        gt = f['frames/gt_wave_field'][...]
        return dict(pts_per_frame=float(v.sum(1).mean()),
                    mean_inc_deg=float(np.degrees(a[v]).mean()) if v.any() else None,
                    gt_hash=float(np.nansum(np.abs(gt))))


def run_ship(ep, est_seed):
    from wavesplat.run import run_estimation, DEFAULT_CONFIG
    cfg = dict(DEFAULT_CONFIG)
    cfg['seed'] = est_seed
    t0 = time.time()
    r = run_estimation(ep['path'], config=cfg, verbose=False)
    return dict(arm='SHIP-NOW', est_seed=est_seed,
                avg_nRMSE=float(r['avg_nRMSE']),
                zero_nRMSE=(float(r['zero_nRMSE']) if r.get('zero_nRMSE') is not None else None),
                varpro_steps=cfg.get('varpro_steps'),
                learned_init_iters=cfg.get('learned_init_iters'),
                auto_switch_thresh=cfg.get('auto_switch_thresh'),
                lambda_l1=cfg.get('lambda_l1'), wall_s=time.time() - t0)


def run_spec(ep, filter_seed):
    import run_s16
    t0 = time.time()
    row = run_s16.run_letkf(ep['path'], 'SPEC25', filter_seed=filter_seed)
    return dict(arm='SPEC25', filter_seed=filter_seed,
                avg_nRMSE=float(row['avg_nRMSE']), zero_nRMSE=float(row['zero_nRMSE']),
                loc_radius=row.get('loc_radius'), n_ensemble=row.get('n_ensemble'),
                background=row.get('background'), wall_s=time.time() - t0)


def main():
    shard = int(sys.argv[1]) if len(sys.argv) > 2 else 0
    nsh = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    out = os.path.join(HERE, f'results_b15_{shard}.jsonl')
    done = set()
    if os.path.exists(out):
        for l in open(out):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            if 'error' not in r:
                done.add((r['arm'], r['episode'], r.get('est_seed') or r.get('filter_seed')))
    eps = [e for i, e in enumerate(episodes()) if i % nsh == shard]
    print(f'shard {shard}/{nsh}: 에피소드 {len(eps)}개', flush=True)
    with open(out, 'a') as fh:
        for ep in eps:
            st = obs_stats(ep['path'])
            for arm, seeds, fn in (('SHIP-NOW', G.EST_SEEDS, run_ship),
                                   ('SPEC25', G.FILTER_SEEDS, run_spec)):
                for s in seeds:
                    if (arm, ep['episode'], s) in done:
                        continue
                    try:
                        row = fn(ep, s)
                    except Exception as e:                # noqa: BLE001
                        import traceback
                        row = dict(arm=arm, error=repr(e), tb=traceback.format_exc()[-400:])
                    row.update({k: ep[k] for k in ('episode', 'bf', 'ncomp',
                                                   'gen_seed', 'raydrop', 'pair')})
                    row.update(st)
                    fh.write(json.dumps(row) + '\n')
                    fh.flush()
                    print(f"{arm:8s} {ep['episode'][:36]:36s} s{s} "
                          f"{row.get('error') or format(row['avg_nRMSE'], '.4f')}", flush=True)


if __name__ == '__main__':
    main()
