"""S9 신선 표본 생성 — SPARSE64 + CROWDED32 (2026-08-11).

## 왜 비대칭인가
S8 은 혼잡을 사전 검정력 **0.937** 로 통과시켰고(27/32), 희소는 **0.589** 에서 22/32 로
문턱(23/32)에 **하나 차이**로 미달했다. 즉 희소의 결말은 효과 부재가 아니라 **검정력 부족**이다.
⟹ 표본을 검정력이 부족한 쪽에만 쓴다: **희소 n=64, 혼잡 n=32**.

## 시드
시드 **48-63**. `wavesplat/data*/` 전수 스캔 결과 기존 사용 시드는
`0-11, 20-39, 42, 777, 888, 999, 1001, 1111, 1212` 이므로 48-63 은 **충돌 0**.
희소는 48-63(16개) 전부, 혼잡은 **48-55(8개)** 를 쓴다 — 혼잡 시드 집합은 희소의 **접두**라서
설계가 n 을 올리는 쪽으로 바뀌어도 이미 만든 파일이 버려지지 않는다.

    SPARSE64  = (BF4, BF5) x ( 2,  5)comp x seed 48-63  -> data_s9_sparse_2026-08-11/
    CROWDED32 = (BF4, BF5) x (15, 30)comp x seed 48-55  -> data_s9_crowded_2026-08-11/

## 절차
`gen_s8.py` 와 **동일**: mesh 200x167, duration/dt/depth/n_pts 는 대응 원본에서 읽음,
GPU raycast, 손잡이 3종(random_amplitudes / random_frequencies / decorrelate_stream) 전부 ON.
⛔레짐 차이를 "성분수만 다르다" 라고 쓰지 않는다(S7 A12 승계): 레짐을 바꾸면 최소 넷이 함께
움직인다 — 성분수 · MIN_DK 억제반경 발동 여부 · 사다리 대 진실 N · 기준선 18-20배.

`data/`, `data_finemesh_2026-07-23/` 는 **읽기 전용**으로만 연다. 재개 가능(있는 파일은 skip).
★생성은 추정을 돌리지 않으므로 결과를 볼 수 없다 — 사전등록 감사와 병행해도 편의가 생기지 않는다.
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

SPARSE_SEEDS = range(48, 64)     # 16 -> n=64
CROWDED_SEEDS = range(48, 56)    # 8  -> n=32  (희소 시드의 접두)

REGIMES = {
    # 희소를 먼저 만든다: 검정력이 부족한 쪽이고, 어느 설계에서도 반드시 필요하다.
    'sparse': dict(
        out='data_s9_sparse_2026-08-11',
        jobs=[(bf, nc, s) for bf in ('BF4', 'BF5') for nc in (2, 5) for s in SPARSE_SEEDS],
    ),
    'crowded': dict(
        out='data_s9_crowded_2026-08-11',
        jobs=[(bf, nc, s) for bf in ('BF4', 'BF5') for nc in (15, 30) for s in CROWDED_SEEDS],
    ),
}

# duration/dt/depth/n_pts 를 읽어올 원본. 해당 (bf, nc) 원본이 없으면 S1/S8 과 **같은** 대체를 쓴다.
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
