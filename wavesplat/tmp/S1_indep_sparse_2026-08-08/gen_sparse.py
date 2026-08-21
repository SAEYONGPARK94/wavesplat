"""독립 표본 SPARSE16 생성 (2026-08-08).

왜: `tmp/audit_independence_2026-08-07/` 이후 만들어진 독립 집합
(`data_indep2/3/4_2026-08-07`, 48 에피소드)은 **전부 혼잡**이다
(BF4/BF5 x 15/30comp x seed). 희소 레짐에는 독립 에피소드가 **0개**다.
기존 `SPARSE8`(Rx-4 블록 S)은 진폭 6 · 위상 3 · 방향 3 종뿐이라 바닥 p = 0.25 다.

그래서 B2 가 내린 판정 *"G1P1 이득은 혼잡 전용"* 은 지금까지 **독립 표본에서 시험된 적이 없다**.
이 스크립트는 혼잡16 과 **해상상태를 동일하게 고정**하고 성분수만 낮춘 희소16 을 뽑는다:

    CROWDED16 = (BF4, BF5) x (15, 30) x seed 0-3 / 4-7 / 10-13   [이미 있음]
    SPARSE16  = (BF4, BF5) x ( 2,  5) x seed 20-23               [이 파일]

⟹ 레짐 축(성분수)이 해상상태와 **교락되지 않는다**. 시드 20-23 은 기존 독립 집합
(0-3, 4-7, 8-11)과 겹치지 않는다.

세 손잡이(`random_amplitudes` / `random_frequencies` / `decorrelate_stream`)를 전부 켠다
= `data_indep2/3/4` 와 동일한 절차. 그 외 생성 절차는 `tmp/indep_data_2026-08-07/gen_indep4.py`
와 **동일**: mesh 200x167, duration/dt/depth/n_pts 는 대응 기존 파일에서 읽어옴, GPU raycast.

출력 파일명 태그 `_ra_rf_ds` ⟹ 기존 파일과 충돌 불가.
`data/` 및 `data_finemesh_2026-07-23/` 는 **읽기 전용**으로만 연다.
재개 가능: 이미 있는 출력 파일은 건너뛴다.
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
OUT = 'data_indep_sparse_2026-08-08'
MANIFEST = os.path.join(OUT, '_manifest.json')
NX, NY = 200, 167

# 해상상태는 혼잡16 과 동일하게 고정, 성분수만 낮춘다 (레짐 축 비교락)
SPARSE16 = [(bf, nc, s) for bf in ('BF4', 'BF5') for nc in (2, 5) for s in range(20, 24)]

# duration/dt/depth/n_pts 를 읽어올 원본 (해당 (bf, nc) 파일이 없으면 같은 bf 의 것을 쓴다)
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


def main():
    only = int(sys.argv[1]) if len(sys.argv) > 1 else None
    os.makedirs(OUT, exist_ok=True)
    cfg = WaveSimConfig()
    cfg.mesh.nx, cfg.mesh.ny = NX, NY

    manifest = {}
    if os.path.exists(MANIFEST):
        manifest = json.load(open(MANIFEST))

    jobs = SPARSE16[:only] if only else SPARSE16
    fh = open(os.path.join(HERE, 'gen.log'), 'a')
    log(f'생성 대상 {len(jobs)} 에피소드 -> {OUT}/', fh)

    for i, (bf, nc, seed) in enumerate(jobs, 1):
        src = os.path.join(SRC, SRC_FALLBACK[(bf, nc)])
        out_name = f'episode_{bf}_{nc}comp_seed{seed}_uniform_ra_rf_ds.h5'
        out_path = os.path.join(OUT, out_name)
        if os.path.exists(out_path):
            log(f'[{i}/{len(jobs)}] skip (있음) {out_name}', fh)
            continue

        # 관측 예산과 시간축은 기존 파일에서 그대로 읽어온다 (성분수만 바뀌도록)
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
