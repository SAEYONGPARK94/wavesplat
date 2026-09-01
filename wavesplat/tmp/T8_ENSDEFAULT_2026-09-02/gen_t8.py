#!/usr/bin/env python3
"""`T8` 신선 희소 코퍼스 생성 — (BF4,BF5) x (2,5)성분 x 시드 1704-1719 = 64 에피소드.

## 왜 신선한가 (PREREG §2)
`S-ENS` 의 희소 64 에피소드에서 나는 **중앙값(-1.56%)과 승수(35/64)를 이미 보았다**.
꼬리 개수는 아직 안 뽑았지만 뽑을 수 **있는** 자료로 비열등성을 판정하면 노출이다.
⟹ 코퍼스를 새로 만든다.

## 시드
저장소 전수 스캔(에피소드 파일명 + 결과 jsonl) 결과 **사용된 최대 시드 = 1703**.
⟹ **1704 이상은 구조적으로 충돌 0**(이 스크립트가 실행 전에 다시 확인한다).

## 절차
`tmp/S9_lr_x_l1_2026-08-11/gen_s9.py` 와 **동일**: 메시 200x167, duration/dt/depth/n_pts 를
`data_finemesh_2026-07-23` 의 대응 원본에서 읽음, GPU raycast, 손잡이 3종 전부 ON.
⛔`wavesim/**` 은 수정하지 않는다.

★생성은 추정을 돌리지 않으므로 **결과를 볼 수 없다** — 사전등록 동결과 병행해도
편의가 생기지 않는다(`gen_s9.py` 의 같은 논거).

사용:
    python3 gen_t8.py --smoke        # 1 에피소드로 비용·mesh 확인
    python3 gen_t8.py --worker 0 --nworker 4
"""
import argparse
import json
import os
import sys
import time
import traceback

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '2')

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
os.chdir(REPO)

import h5py                                    # noqa: E402
from wavesim.config import WaveSimConfig       # noqa: E402
from wavesim.pipeline import run_episode       # noqa: E402

SRC = 'data_finemesh_2026-07-23'
OUT = 'data_t8_sparse_2026-09-02'
NX, NY = 200, 167
SEEDS = range(1704, 1720)                      # 16 -> n=64
JOBS = [(bf, nc, s) for bf in ('BF4', 'BF5') for nc in (2, 5) for s in SEEDS]

# `gen_s9.py` 와 **같은** 대체표(해당 (bf,nc) 원본이 없을 때).
SRC_FALLBACK = {
    ('BF4', 2): 'episode_BF4_5comp_seed0_uniform.h5',
    ('BF4', 5): 'episode_BF4_5comp_seed0_uniform.h5',
    ('BF5', 2): 'episode_BF5_15comp_seed0_uniform.h5',
    ('BF5', 5): 'episode_BF5_15comp_seed0_uniform.h5',
}


def assert_seeds_free():
    """⛔실행 전에 시드 충돌을 **다시** 확인한다 — 사전등록의 주장을 코드가 붙든다."""
    import glob
    import re
    used = set()
    for d in glob.glob(os.path.join(REPO, 'data*')) + glob.glob(os.path.join(REPO, 'tmp', '*', 'data*')):
        if not os.path.isdir(d) or os.path.basename(d) == OUT:
            continue
        for f in os.listdir(d):
            m = re.search(r'seed(\d+)', f)
            if m:
                used.add(int(m.group(1)))
    for f in glob.glob(os.path.join(REPO, 'tmp', '*', '*.jsonl')):
        for m in re.finditer(r'seed(\d+)', open(f, errors='replace').read()):
            used.add(int(m.group(1)))
    clash = sorted(set(SEEDS) & used)
    assert not clash, f'⛔시드 충돌: {clash} — 사전등록 §2 의 주장이 깨졌다'
    return max(x for x in used if x < 10_000)


def src_for(bf, nc):
    return os.path.join(SRC, SRC_FALLBACK.get((bf, nc),
                                              f'episode_{bf}_{nc}comp_seed0_uniform.h5'))


def log(msg, fh=None):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    if fh:
        fh.write(line + '\n')
        fh.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--worker', type=int, default=0)
    ap.add_argument('--nworker', type=int, default=1)
    a = ap.parse_args()

    hi = assert_seeds_free()
    os.makedirs(OUT, exist_ok=True)
    cfg = WaveSimConfig()
    cfg.mesh.nx, cfg.mesh.ny = NX, NY

    jobs = [j for i, j in enumerate(JOBS) if i % a.nworker == a.worker]
    if a.smoke:
        jobs = jobs[:1]
    tag = 'smoke' if a.smoke else f'w{a.worker}'
    fh = open(os.path.join(HERE, f'gen_{tag}.log'), 'a')
    mpath = os.path.join(OUT, f'_manifest_{tag}.json')
    manifest = json.load(open(mpath)) if os.path.exists(mpath) else {}
    log(f'=== T8 생성 {tag}: {len(jobs)} 에피소드 -> {OUT}/ '
        f'(저장소 최대 사용 시드 {hi} < {min(SEEDS)})', fh)

    for i, (bf, nc, seed) in enumerate(jobs, 1):
        name = f'episode_{bf}_{nc}comp_seed{seed}_uniform_ra_rf_ds.h5'
        path = os.path.join(OUT, name)
        if os.path.exists(path):
            log(f'[{i}/{len(jobs)}] skip (있음) {name}', fh)
            continue
        with h5py.File(src_for(bf, nc), 'r') as f:
            depth = float(f['config'].attrs['depth'])
            n_pts = int(f['config/lidar_config'].attrs.get('n_pts', 0)) or 50
            ts = f['frames/timestamps'][:]
        t0 = time.time()
        try:
            p = run_episode(sea_state=bf, n_components=nc, n_pts=n_pts, seed=seed,
                            raydrop_model='uniform',
                            duration=float(ts[-1] - ts[0]), dt=float(ts[1] - ts[0]),
                            depth=depth, output_dir=OUT, verbose=False, cfg=cfg,
                            use_gpu_raycast=True, random_amplitudes=True,
                            decorrelate_stream=True, random_frequencies=True)
            el = time.time() - t0
            with h5py.File(p, 'r') as f:
                nfr = f['frames/gt_wave_field'].shape[0]
                at = dict(f['config'].attrs)
            manifest[os.path.basename(p)] = dict(
                status='ok', elapsed_s=el, n_frames=nfr,
                random_amplitudes=bool(at.get('random_amplitudes', False)),
                random_frequencies=bool(at.get('random_frequencies', False)),
                decorrelate_stream=bool(at.get('decorrelate_stream', False)),
                mesh_verdict=str(at.get('mesh_verdict', '?')))
            log(f'[{i}/{len(jobs)}] ok {os.path.basename(p)} {el:.0f}s '
                f'frames={nfr} mesh={at.get("mesh_verdict", "?")}', fh)
        except Exception as e:                                  # noqa: BLE001
            manifest[name] = dict(status='error', error=str(e))
            log(f'[{i}/{len(jobs)}] ERROR {name}: {e}', fh)
            traceback.print_exc()
        json.dump(manifest, open(mpath, 'w'), indent=1)

    ok = sum(1 for v in manifest.values() if v.get('status') == 'ok')
    log(f'{tag} 완료: ok {ok} / {len(manifest)}', fh)
    return 0


if __name__ == '__main__':
    sys.exit(main())
