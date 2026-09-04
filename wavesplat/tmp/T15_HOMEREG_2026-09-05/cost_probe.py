"""T15 배관조사 — **단가 실측 프로브** (⛔러너·게이트·사전등록 아니다).

에피소드 **1개**에 배포 팔(`SHIP` = `dict(DEFAULT_CONFIG)`, 무처치)을 한 번 돌려
벽시계를 잰다. `PLUMBING.md` §5 의 유일 출처.

⛔반드시 메모리 캡 안에서: `systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 -- python3 -u cost_probe.py <ep>`
"""
import os
import sys

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '1'

import json    # noqa: E402
import time    # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, ROOT)

import torch                                                # noqa: E402
torch.set_num_threads(1)
from wavesplat.run import run_estimation, DEFAULT_CONFIG    # noqa: E402


def main():
    ep = sys.argv[1] if len(sys.argv) > 1 else 'data/episode_BF4_30comp_seed0_uniform.h5'
    path = os.path.join(ROOT, ep)
    cfg = dict(DEFAULT_CONFIG)          # ⛔팔 정의는 되읽기로 남긴다(빈 dict 금지 규율)
    t0 = time.time()
    r = run_estimation(path, cfg)
    wall = time.time() - t0
    out = dict(
        episode=ep, wall_s=round(wall, 3),
        cuda=torch.cuda.is_available(), device_cfg=cfg['device'],
        omp=os.environ.get('OMP_NUM_THREADS'),
        init_time_s=r.get('init_time_s'), init_iters=r.get('init_iters'),
        varpro_fired=r.get('varpro_fired'), varpro_wall_s=r.get('varpro_wall_s'),
        auto_selected=r.get('auto_selected'), auto_n_selected=r.get('auto_n_selected'),
        auto_real_gate=r.get('auto_real_gate'), ensemble_k=r.get('ensemble_k'),
        n_gaussians=r.get('n_gaussians'),
        tracking_frames=r.get('tracking_frames'),
        episode_total_frames=r.get('episode_total_frames'),
        scored_t_first=r.get('scored_t_first'), scored_t_last=r.get('scored_t_last'),
        avg_inference_ms=r.get('avg_inference_ms'),
        avg_nRMSE=r.get('avg_nRMSE'), avg_RMSE_z=r.get('avg_RMSE_z'),
        avg_RMSE_near=r.get('avg_RMSE_near'),
        shell_keys=sorted((r.get('avg_shell_rmse') or {}).keys()),
        Hs_true=r.get('Hs_true'), Tp_true=r.get('Tp_true'),
        lambda_l1=cfg['lambda_l1'], varpro_steps_cfg=cfg['varpro_steps'],
        learned_init_iters=cfg['learned_init_iters'],
        auto_switch_thresh=cfg['auto_switch_thresh'],
        init_method_cfg=cfg['init_method'],
    )
    print(json.dumps(out, ensure_ascii=False, indent=1))
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'cost_probe.json'), 'a') as fh:
        fh.write(json.dumps(out, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
