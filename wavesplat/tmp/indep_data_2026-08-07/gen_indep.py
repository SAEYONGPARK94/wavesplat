"""독립 표본 혼잡16 생성 (2026-08-07).

왜: `tmp/audit_independence_2026-08-07/` 가 잰 대로, 기존 `data_finemesh_2026-07-23/`
의 혼잡16 은 **진폭/omega 벡터 4종 · 위상 벡터 8종**뿐이다. 진폭·omega 에는 rng draw 가
아예 없고(= 시드를 늘려도 새 스펙트럼이 안 생긴다), 위상·방향 스트림은 sea_state 와
n_components 를 무시한다(= BF4/BF5 가 같은 위상, 15comp 이 30comp 의 prefix).
그래서 그 집합 위의 어떤 클러스터-강건 검정도 p=0.125 밑으로 못 내려간다.

이 스크립트는 `random_amplitudes=True` + `decorrelate_stream=True` 로 같은 16 셀을
다시 뽑는다. 두 손잡이 외에는 2026-07-23 재생성 절차와 **동일**:
  * mesh 200x167 (dx 1.206 m) — `tmp/regen_gap_fixedT.py` / REGEN_SPEC Sec 5
  * duration / dt / depth / n_pts 는 대응하는 기존 파일에서 **읽어온다**
  * GPU raycast
출력 파일명에는 `_ra_ds` 태그가 붙으므로 기존 파일과 **충돌하지 않는다**.
`data/` 및 `data_finemesh_2026-07-23/` 는 열지 않는다(읽기 전용).

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
OUT = 'data_indep2_2026-08-07'
MANIFEST = os.path.join(OUT, '_manifest.json')
NX, NY = 200, 167

CROWDED16 = [(bf, nc, s) for bf in ('BF4', 'BF5') for nc in (15, 30) for s in range(4)]


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
        src = os.path.join(SRC, f'episode_{bf}_{nc}comp_seed{seed}_uniform.h5')
        out_name = f'episode_{bf}_{nc}comp_seed{seed}_uniform_ra_rf_ds.h5'
        out_path = os.path.join(OUT, out_name)
        if os.path.exists(out_path):
            log(f'[{i}/{len(jobs)}] skip (있음) {out_name}', fh)
            continue

        # 관측 예산과 시간축은 기존 파일에서 그대로 읽어온다 (두 손잡이만 바뀌도록)
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
