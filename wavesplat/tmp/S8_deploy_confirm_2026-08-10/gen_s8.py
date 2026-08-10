"""S8 신선 확증 표본 생성 — CROWDED32 + SPARSE32 (2026-08-10).

왜 또 만드는가: S6 의 PRIMARY 는 **n=16 에서 13/16 을 찍고 Holm-3 에서 0.064 로 죽었다**.
n=16 · 3대조 구조에서 13/16 은 **원리적으로 통과 불가**였다(원 p=0.0213 × 3 = 0.064).
즉 결과가 없었던 게 아니라 **설계가 못 잡았다**. 그래서 n 을 32 로 올린다.

⛔기존 32ep(시드 20-23)은 S1-S6 에서 **여섯 번** 채점됐다. 거기에 16 을 덧붙여 n=32 를 만들면
표본의 절반이 재사용이라 확증이라 부를 수 없다. ⟹ **완전히 새 시드 블록 32-39** 로 두 레짐 각각 32.

    CROWDED32 = (BF4, BF5) x (15, 30) x seed 32-39   -> data_s8_crowded_2026-08-10/
    SPARSE32  = (BF4, BF5) x ( 2,  5) x seed 32-39   -> data_s8_sparse_2026-08-10/

두 레짐은 해상상태·시드 블록·생성 절차·duration·dt·depth·n_pts 를 공유한다.
⛔**"성분수만 다르다" 라고 쓰지 않는다**(S7 AMENDMENT A12 승계): 레짐을 바꾸면 최소 넷이 함께
움직인다 — 성분수 · MIN_DK 억제반경 발동 여부 · 사다리 대 진실 N · 기준선 18-20배.
생성 절차는 `tmp/S1_indep_sparse_2026-08-08/gen_{crowded,sparse}.py` 와 **동일**:
mesh 200x167, duration/dt/depth/n_pts 는 대응 원본에서 읽음, GPU raycast,
세 손잡이(random_amplitudes / random_frequencies / decorrelate_stream) 전부 ON.

시드 32-39 는 기존 어느 집합과도 겹치지 않는다(확인: 0-4, 8-11, 20-31, 42).
`data/`, `data_finemesh_2026-07-23/` 는 **읽기 전용**으로만 연다. 재개 가능(있는 파일은 skip).
"""
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
os.chdir(REPO)

import h5py  # noqa: E402

from wavesim.config import WaveSimConfig  # noqa: E402
from wavesim.pipeline import run_episode  # noqa: E402

SRC = 'data_finemesh_2026-07-23'
NX, NY = 200, 167
SEEDS = range(32, 40)

REGIMES = {
    'crowded': dict(
        out='data_s8_crowded_2026-08-10',
        jobs=[(bf, nc, s) for bf in ('BF4', 'BF5') for nc in (15, 30) for s in SEEDS],
    ),
    'sparse': dict(
        out='data_s8_sparse_2026-08-10',
        jobs=[(bf, nc, s) for bf in ('BF4', 'BF5') for nc in (2, 5) for s in SEEDS],
    ),
}

# duration/dt/depth/n_pts 를 읽어올 원본. 해당 (bf, nc) 원본이 없으면 S1 과 **같은** 대체를 쓴다.
SRC_FALLBACK = {
    ('BF4', 2): 'episode_BF4_5comp_seed0_uniform.h5',
    ('BF4', 5): 'episode_BF4_5comp_seed0_uniform.h5',
    ('BF5', 2): 'episode_BF5_15comp_seed0_uniform.h5',
    ('BF5', 5): 'episode_BF5_15comp_seed0_uniform.h5',
}


def log(msg, fh=None):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    if fh:
        fh.write(line + '\n')
        fh.flush()


def src_for(bf, nc):
    return os.path.join(SRC, SRC_FALLBACK.get((bf, nc),
                                              f'episode_{bf}_{nc}comp_seed0_uniform.h5'))


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    names = list(REGIMES) if which == 'all' else [which]
    cfg = WaveSimConfig()
    cfg.mesh.nx, cfg.mesh.ny = NX, NY
    fh = open(os.path.join(HERE, 'gen.log'), 'a')

    for name in names:
        out = REGIMES[name]['out']
        jobs = REGIMES[name]['jobs']
        os.makedirs(out, exist_ok=True)
        manifest_path = os.path.join(out, '_manifest.json')
        manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
        log(f'=== {name}: 생성 대상 {len(jobs)} 에피소드 -> {out}/', fh)

        for i, (bf, nc, seed) in enumerate(jobs, 1):
            out_name = f'episode_{bf}_{nc}comp_seed{seed}_uniform_ra_rf_ds.h5'
            out_path = os.path.join(out, out_name)
            if os.path.exists(out_path):
                log(f'[{name} {i}/{len(jobs)}] skip (있음) {out_name}', fh)
                continue

            with h5py.File(src_for(bf, nc), 'r') as f:
                depth = float(f['config'].attrs['depth'])
                n_pts = int(f['config/lidar_config'].attrs.get('n_pts', 0)) or 50
                ts = f['frames/timestamps'][:]
            duration = float(ts[-1] - ts[0])
            dt = float(ts[1] - ts[0])

            t0 = time.time()
            try:
                path = run_episode(
                    sea_state=bf, n_components=nc, n_pts=n_pts, seed=seed,
                    raydrop_model='uniform', duration=duration, dt=dt, depth=depth,
                    output_dir=out, verbose=False, cfg=cfg,
                    use_gpu_raycast=True,
                    random_amplitudes=True, decorrelate_stream=True,
                    random_frequencies=True,
                )
                el = time.time() - t0
                with h5py.File(path, 'r') as f:
                    nfr = f['frames/gt_wave_field'].shape[0]
                    a = dict(f['config'].attrs)
                manifest[os.path.basename(path)] = dict(
                    status='ok', elapsed_s=el, n_frames=nfr,
                    random_amplitudes=bool(a.get('random_amplitudes', False)),
                    random_frequencies=bool(a.get('random_frequencies', False)),
                    decorrelate_stream=bool(a.get('decorrelate_stream', False)),
                    mesh_verdict=str(a.get('mesh_verdict', '?')))
                log(f'[{name} {i}/{len(jobs)}] ok {os.path.basename(path)} '
                    f'{el:.0f}s frames={nfr} mesh={a.get("mesh_verdict", "?")}', fh)
            except Exception as e:
                manifest[out_name] = dict(status='error', error=str(e))
                log(f'[{name} {i}/{len(jobs)}] ERROR {out_name}: {e}', fh)
                traceback.print_exc()
            json.dump(manifest, open(manifest_path, 'w'), indent=1)

        ok = sum(1 for v in manifest.values() if v.get('status') == 'ok')
        log(f'{name} 완료: ok {ok} / {len(manifest)}', fh)
    fh.close()


if __name__ == '__main__':
    main()
