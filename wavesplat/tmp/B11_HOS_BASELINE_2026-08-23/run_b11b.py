"""B-11 러너 — 비선형(HOS)·신선 시드 표본에서 헤드라인을 다시 잰다.

★**기준선 정의를 손으로 옮기지 않는다** — `S16` 의 `run_s16.run_letkf` 를 **그대로 import** 한다
(그 함수가 `letkf_spec.LETKFGridSpec` 을 만들고 `SPEC25` 를 정의한다).
★`SHIP` 은 배포 진입점 `wavesplat.run.run_estimation` 을 **DEFAULT_CONFIG 그대로** 부른다.
"""
import glob, json, os, sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for p in (HERE, ROOT, os.path.join(ROOT, 'tmp', 'S16_BASELINE_2026-08-14')):
    sys.path.insert(0, p)
os.chdir(ROOT)
import gates_b11b as G                                   # noqa: E402
OUT = os.path.join(HERE, 'results_b11b.jsonl')


def crowded_episodes():   # ★B-11b: 이름은 유지하되 희소를 고른다(러너 구조 동일)
    """혼잡 = ncomp >= 15 (HOSGATE 의 정의와 동일)."""
    out = []
    import re
    for p in sorted(glob.glob(os.path.join(G.HOS, '*.h5'))):
        m = re.match(r'episode_(BF\d)_(\d+)comp_seed(\d+)_(.+)\.h5', os.path.basename(p))
        if m and int(m.group(2)) < 15 and int(m.group(3)) in G.SPARSE_SEEDS:
            out.append(dict(path=p, episode=os.path.basename(p)[len('episode_'):-3],
                            bf=m.group(1), ncomp=int(m.group(2)), seed=int(m.group(3))))
    return out


def run_ship(ep, est_seed):
    from wavesplat.run import run_estimation, DEFAULT_CONFIG
    cfg = dict(DEFAULT_CONFIG); cfg['seed'] = est_seed
    t0 = time.time()
    r = run_estimation(ep['path'], config=cfg, verbose=False)
    return dict(arm='SHIP-NOW', episode=ep['episode'], bf=ep['bf'], ncomp=ep['ncomp'],
                gen_seed=ep['seed'], est_seed=est_seed,
                avg_nRMSE=float(r['avg_nRMSE']),
                zero_nRMSE=float(r.get('zero_nRMSE')) if r.get('zero_nRMSE') is not None else None,
                varpro_steps=cfg.get('varpro_steps'),
                learned_init_iters=cfg.get('learned_init_iters'),
                auto_switch_thresh=cfg.get('auto_switch_thresh'),
                bank_refit_every_s=cfg.get('bank_refit_every_s'),
                wall_s=time.time() - t0)


def run_spec(ep, filter_seed):
    import run_s16                                        # ★기준선 정의를 그대로 쓴다
    t0 = time.time()
    row = run_s16.run_letkf(ep['path'], 'SPEC25', filter_seed=filter_seed)
    return dict(arm='SPEC25', episode=ep['episode'], bf=ep['bf'], ncomp=ep['ncomp'],
                gen_seed=ep['seed'], filter_seed=filter_seed,
                avg_nRMSE=float(row['avg_nRMSE']),
                zero_nRMSE=float(row['zero_nRMSE']),
                loc_radius=row.get('loc_radius'), n_ensemble=row.get('n_ensemble'),
                background=row.get('background'), wall_s=time.time() - t0)


def main():
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            try: r = json.loads(l)
            except json.JSONDecodeError: continue
            if 'error' not in r:
                done.add((r['arm'], r['episode'], r.get('est_seed') or r.get('filter_seed')))
    eps = crowded_episodes()
    print(f'혼잡 HOS 에피소드 {len(eps)}개')
    with open(OUT, 'a') as fh:
        for ep in eps:
            for s in G.EST_SEEDS:
                if ('SHIP-NOW', ep['episode'], s) in done: continue
                try: row = run_ship(ep, s)
                except Exception as e:
                    import traceback
                    row = dict(arm='SHIP-NOW', episode=ep['episode'], est_seed=s,
                               error=repr(e), tb=traceback.format_exc()[-500:])
                fh.write(json.dumps(row) + '\n'); fh.flush()
                print(f"SHIP-NOW {ep['episode'][:34]:34s} s{s} " +
                      (f"nRMSE={row['avg_nRMSE']:.4f} {row['wall_s']:.1f}s"
                       if 'error' not in row else str(row['error'])[:60]), flush=True)
            for s in G.FILTER_SEEDS:
                if ('SPEC25', ep['episode'], s) in done: continue
                try: row = run_spec(ep, s)
                except Exception as e:
                    import traceback
                    row = dict(arm='SPEC25', episode=ep['episode'], filter_seed=s,
                               error=repr(e), tb=traceback.format_exc()[-500:])
                fh.write(json.dumps(row) + '\n'); fh.flush()
                print(f"SPEC25   {ep['episode'][:34]:34s} f{s} " +
                      (f"nRMSE={row['avg_nRMSE']:.4f} zero={row['zero_nRMSE']:.4f} {row['wall_s']:.1f}s"
                       if 'error' not in row else str(row['error'])[:60]), flush=True)


if __name__ == '__main__':
    main()
