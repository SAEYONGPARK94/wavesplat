"""신선 독립 CROWDED16 생성 (2026-08-08).

왜 또 만드는가: 독립 혼잡 48개(`data_indep2/3/4_2026-08-07`, 시드 0-3 / 4-7 / 10-13)는
**이미 네 번**(Rx-7 / E-B / E-C / E-D) 채점됐다. 그 위에서 G1P1 을 다섯 번째로 재면
"독립 표본"이라고는 부를 수 있어도 "신선"이라고는 부를 수 없다.

이 파일은 `gen_sparse.py` 와 **정확히 같은 절차·같은 시드 블록(20-23)** 으로 혼잡 셀을 뽑는다:

    SPARSE16  = (BF4, BF5) x ( 2,  5) x seed 20-23   [gen_sparse.py]
    CROWDED16 = (BF4, BF5) x (15, 30) x seed 20-23   [이 파일]

⟹ 두 레짐이 **해상상태·시드 블록·생성 절차를 전부 공유**하고 성분수만 다르다.
S1 의 레짐 대조는 이 32 에피소드 위에서만 이뤄진다.

나머지는 `tmp/indep_data_2026-08-07/gen_indep4.py` 와 동일.
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
OUT = 'data_indep_crowded_2026-08-08'
MANIFEST = os.path.join(OUT, '_manifest.json')
NX, NY = 200, 167

CROWDED16 = [(bf, nc, s) for bf in ('BF4', 'BF5') for nc in (15, 30) for s in range(20, 24)]


def log(msg, fh=None):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    if fh:
        fh.write(line + '\n')
        fh.flush()


def main():
    only = int(sys.argv[1]) if len(sys.argv) > 1 else None
    os.makedirs(OUT, exist_ok=True)
    cfg = WaveSimConfig()
    cfg.mesh.nx, cfg.mesh.ny = NX, NY

    manifest = {}
    if os.path.exists(MANIFEST):
        manifest = json.load(open(MANIFEST))

    jobs = CROWDED16[:only] if only else CROWDED16
    fh = open(os.path.join(HERE, 'gen.log'), 'a')
    log(f'생성 대상 {len(jobs)} 에피소드 -> {OUT}/', fh)

    for i, (bf, nc, seed) in enumerate(jobs, 1):
        src = os.path.join(SRC, f'episode_{bf}_{nc}comp_seed0_uniform.h5')
        out_name = f'episode_{bf}_{nc}comp_seed{seed}_uniform_ra_rf_ds.h5'
        out_path = os.path.join(OUT, out_name)
        if os.path.exists(out_path):
            log(f'[{i}/{len(jobs)}] skip (있음) {out_name}', fh)
            continue

        with h5py.File(src, 'r') as f:
            c = f['config']
            depth = float(c.attrs['depth'])
            n_pts = int(f['config/lidar_config'].attrs.get('n_pts', 0)) or None
            ts = f['frames/timestamps'][:]
        duration = float(ts[-1] - ts[0])
        dt = float(ts[1] - ts[0])
        if n_pts is None:
            n_pts = 50

        t0 = time.time()
        try:
            path = run_episode(
                sea_state=bf, n_components=nc, n_pts=n_pts, seed=seed,
                raydrop_model='uniform', duration=duration, dt=dt, depth=depth,
                output_dir=OUT, verbose=False, cfg=cfg,
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
            log(f'[{i}/{len(jobs)}] ok {os.path.basename(path)} '
                f'{el:.0f}s frames={nfr} mesh={a.get("mesh_verdict", "?")}', fh)
        except Exception as e:
            manifest[out_name] = dict(status='error', error=str(e))
            log(f'[{i}/{len(jobs)}] ERROR {out_name}: {e}', fh)
            traceback.print_exc()
        json.dump(manifest, open(MANIFEST, 'w'), indent=1)

    ok = sum(1 for v in manifest.values() if v.get('status') == 'ok')
    log(f'완료: ok {ok} / {len(manifest)}', fh)
    fh.close()


if __name__ == '__main__':
    main()
