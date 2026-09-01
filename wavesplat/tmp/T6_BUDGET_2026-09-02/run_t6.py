#!/usr/bin/env python3
"""`T6-BUDGET` 러너 — 한 장치 = 한 프로세스 = 한 샤드.

용법: python3 run_t6.py CPU   |   python3 run_t6.py GPU
⛔`CUDA_VISIBLE_DEVICES` 는 **torch import 전에** 설정해야 하므로 장치를 인자로 받아
이 파일 맨 위에서 처리한다.
"""
import os
import sys

DEVICE = sys.argv[1] if len(sys.argv) > 1 else 'CPU'
assert DEVICE in ('CPU', 'GPU')
if DEVICE == 'CPU':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
# 1 스레드 고정 --- PREREG `G-1`. ⛔torch import 전에.
for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
          'NUMEXPR_NUM_THREADS'):
    os.environ[v] = '1'

import json                                                        # noqa: E402
import time                                                        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch                                                       # noqa: E402
torch.set_num_threads(1)

from wavesplat.run import DEFAULT_CONFIG                           # noqa: E402
from wavesplat.data_loader import EpisodeDataLoader                # noqa: E402
from wavesplat.optimizer.sequential_update import make_updater     # noqa: E402

SEED = 42
CROWDED_DIR = 'data_s8_crowded_2026-08-10'
SPARSE_DIR = 'data'


def episodes():
    """혼잡 8 + 희소 8. ⛔이름순 앞 8개 --- 고르지 않는다."""
    cr = sorted(f for f in os.listdir(os.path.join(ROOT, CROWDED_DIR))
                if f.endswith('.h5'))[:8]
    sp = sorted(f for f in os.listdir(os.path.join(ROOT, SPARSE_DIR))
                if f.startswith('episode_BF4_1comp') and f.endswith('_uniform.h5'))[:8]
    if len(sp) < 8:
        sp = sorted(f for f in os.listdir(os.path.join(ROOT, SPARSE_DIR))
                    if '_1comp_' in f and f.endswith('_uniform.h5'))[:8]
    return ([(os.path.join(CROWDED_DIR, f), 'crowded') for f in cr]
            + [(os.path.join(SPARSE_DIR, f), 'sparse') for f in sp])


def calibrate():
    """`G-5` --- 단조 시계와 부하 확인."""
    t = time.perf_counter()
    time.sleep(0.05)
    return time.perf_counter() - t


def main():
    out = os.path.join(HERE, f'results_t6_{DEVICE.lower()}.jsonl')
    rows = []
    for path, stratum in episodes():
        loader = EpisodeDataLoader(os.path.join(ROOT, path))
        init_w = loader.get_init_window(tp_factor=2.0)
        track = loader.get_tracking_frames(tp_factor=2.0)
        for arm, init_method in (('LEARN', 'auto'), ('ANA', 'v2_konly')):
            # ⛔`{}` 로 팔을 정의하지 않는다 --- 배포 기본값에서 시작하고 한 키만 바꾼다.
            cfg = dict(DEFAULT_CONFIG)
            cfg['depth'] = float(loader.depth)
            cfg['seed'] = SEED
            cfg['init_method'] = init_method
            torch.manual_seed(SEED)
            upd = make_updater(cfg)
            t0 = time.perf_counter()
            rec = upd.initialize(init_w)
            init_s = time.perf_counter() - t0
            prim = getattr(upd, '_primary', upd)
            fms = []
            for fr in track:
                a = time.perf_counter()
                upd.update(fr)
                fms.append((time.perf_counter() - a) * 1e3)
            # ⚠**되읽기** --- 입력 dict 복사(=의도표)가 아니다(S13b 교훈).
            #   ⛔단 여섯 중 **넷만** 객체 속성이다(AMENDMENT_0): `varpro_steps` 와
            #   `learned_init_iters` 는 `updater.config` 안에만 있다. 출처를 함께 적는다.
            ATTR_KEYS = ('lambda_l1', 'tracking_steps', 'tracking_window',
                         'tracking_mode')
            CFG_KEYS = ('varpro_steps', 'learned_init_iters')
            eff = {k: getattr(prim, k) for k in ATTR_KEYS}
            eff.update({k: prim.config[k] for k in CFG_KEYS})
            eff_source = {k: 'attr' for k in ATTR_KEYS}
            eff_source.update({k: 'config' for k in CFG_KEYS})
            rows.append(dict(
                device=DEVICE, arm=arm, stratum=stratum, episode=os.path.basename(path),
                route_effective=(getattr(prim, '_auto_selected', None)
                                 or prim.init_method),
                threads=torch.get_num_threads(),
                cuda_available=torch.cuda.is_available(),
                pid=os.getpid(), seed=SEED,
                init_s=init_s, frame_ms=fms, n_frames=len(fms),
                final_n=int(prim.model.n_gaussians),
                effective=eff, eff_source=eff_source,
                varpro=bool(rec.get('varpro')),
            ))
            print(f'{DEVICE} {arm:5s} {stratum:8s} {os.path.basename(path)[:44]:44s} '
                  f'init {init_s:6.2f}s  frame_med '
                  f'{sorted(fms)[len(fms) // 2]:6.2f}ms  n={len(fms)}', flush=True)
    with open(out, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    with open(os.path.join(HERE, f'calib_{DEVICE.lower()}.json'), 'w') as f:
        json.dump({DEVICE: calibrate()}, f)
    print(f'wrote {out}  ({len(rows)} rows)')


if __name__ == '__main__':
    main()
