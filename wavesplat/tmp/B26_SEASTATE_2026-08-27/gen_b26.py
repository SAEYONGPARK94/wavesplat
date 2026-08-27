"""B-26 생성기 — 2×2 해상상태 코너 96 에피소드 (+ PC-1 비트동일 양성대조).

⛔**`wavesim/**` 무수정 · 몽키패치 0개.** 새 해상상태는 이미 노출된 배포 노브
`cfg.sea_states` 로 넣는다(`wavesim/pipeline.py:113` 이 그것을 읽는다; BF6/BF7 이 그 경로로 존재).
⛔`gen_hos` 의 스냅 몽키패치는 **쓰지 않는다** — HOS 짝을 위한 장치이고 짧은 주기에서는
나이퀴스트를 넘겨 **조용히 에일리어싱**한다(PREREG §전제 2).

호출 인자는 `tmp/S8_deploy_confirm_2026-08-10/gen_s8.py:107-114` 와 같다.
그 스크립트가 만든 `data_s8_crowded_2026-08-10` + `data_s9_crowded_2026-08-11` 이
**`B-1` 헤드라인의 혼잡 64 에피소드**다. 같음의 **증명**은 산문이 아니라 `pc1()` 이다:
동결 에피소드 하나를 이 파일의 `gen_one()` 으로 재생성해 **비트 동일**을 요구한다.

    python3 tmp/B26_SEASTATE_2026-08-27/gen_b26.py pc1              # 양성대조만 (≈3분)
    python3 tmp/B26_SEASTATE_2026-08-27/gen_b26.py smoke            # 1셀 스모크 (지운다)
    python3 tmp/B26_SEASTATE_2026-08-27/gen_b26.py all              # 96 에피소드
    python3 tmp/B26_SEASTATE_2026-08-27/gen_b26.py all 0 4          # 샤드 0/4
"""
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import h5py                                                    # noqa: E402
import numpy as np                                             # noqa: E402

import gates_b26 as G                                          # noqa: E402
from wavesim.pipeline import run_episode                       # noqa: E402

MANIFEST = os.path.join(HERE, 'gen_manifest_b26.json')


def log(msg, fh=None):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    if fh:
        fh.write(line + '\n')
        fh.flush()


def gen_one(sea_state, ncomp, seed, out_dir):
    """★단일 생성 진입점. **모든 팔·모든 코너·PC-1 이 이 함수 하나를 쓴다.**

    인자는 `gen_s8.py:107-114` 와 동일하고, duration/dt/depth/n_pts 는
    `G.gen_params()` 가 **동결 원본 에피소드에서 읽는다**(손복사 0)."""
    p = G.gen_params()
    os.makedirs(out_dir, exist_ok=True)
    return run_episode(
        sea_state=sea_state, n_components=ncomp, n_pts=p['n_pts'], seed=seed,
        raydrop_model=G.RAYDROP, duration=p['duration'], dt=p['dt'], depth=p['depth'],
        output_dir=out_dir, verbose=False, cfg=G._cfg(),
        use_gpu_raycast=True,
        random_amplitudes=True, random_frequencies=True, decorrelate_stream=True,
    )


def _diag(path, expect_Hs):
    with h5py.File(path, 'r') as f:
        gt = f['frames/gt_wave_field'][:]
        a = dict(f['config'].attrs)
    return dict(n_frames=int(gt.shape[0]), gt_Hs_4std=float(4 * np.std(gt)),
                gt_Hs_rel_err=float(abs(4 * np.std(gt) - expect_Hs) / expect_Hs),
                gt_absmax=float(np.abs(gt).max()),
                attr_Hs=float(a['Hs']), attr_Tp=float(a['Tp']),
                mesh_verdict=str(a.get('mesh_verdict')),
                var_under_resolved=float(a.get('var_under_resolved', np.nan)),
                lambda_min=float(a.get('lambda_min', np.nan)),
                lambda_peak=float(a.get('lambda_peak', np.nan)))


def pc1(out_dir=None):
    """★PC-1 — 동결 `B-1` 코퍼스 에피소드를 `gen_one()` 으로 재생성해 **비트 동일** 요구.

    ⟹ *"우리 생성기가 헤드라인 코퍼스의 생성기와 같은 호출인가"* 를 산문이 아니라
    비트로 답한다. `cfg.sea_states` 에 코너를 **추가**한 것이 BF4 를 건드리지 않는다는
    증명이기도 하다."""
    out_dir = out_dir or os.path.join(HERE, 'pc1')
    dst = os.path.join(out_dir, os.path.basename(G.PC1_PATH))
    if not os.path.exists(dst):
        t0 = time.time()
        gen_one(G.PC1['sea_state'], G.PC1['n_components'], G.PC1['seed'], out_dir)
        log(f'PC-1 재생성 {time.time()-t0:.0f}s')
    ident, maxabs = {}, {}
    with h5py.File(G.PC1_PATH, 'r') as a, h5py.File(dst, 'r') as b:
        for k in G.PC1_DATASETS:
            x, y = a[f'frames/{k}'][:], b[f'frames/{k}'][:]
            ident[k] = bool(x.shape == y.shape and np.array_equal(
                np.nan_to_num(x, nan=-1e30), np.nan_to_num(y, nan=-1e30)))
            maxabs[k] = (float(np.nanmax(np.abs(x.astype(float) - y.astype(float))))
                         if x.shape == y.shape else None)
    r = dict(frozen=G.PC1_PATH, regenerated=dst, identical=ident, maxabs=maxabs)
    json.dump(r, open(os.path.join(HERE, 'pc1_b26.json'), 'w'), indent=1)
    for name, ok, msg in G.gates_pc1(r):
        log(f'  [{"PASS" if ok else "FAIL"}] {name}: {msg}')
    return r


def main():
    which = (sys.argv[1:] or ['all'])[0]
    shard, nshard = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 1)
    if which == 'pc1':
        pc1()
        return
    eps, _ = G.select()
    if which == 'smoke':
        eps = [e for e in eps if e['corner'] == G.PRIMARY_CORNER][:1]
        out_dir = G.SMOKE
    else:
        out_dir = G.DATA
    jobs = [e for i, e in enumerate(eps) if i % nshard == shard]
    fh = open(os.path.join(HERE, f'gen_b26_{shard}.log'), 'a')
    log(f'=== B-26 gen {which} shard {shard}/{nshard}: {len(jobs)} 에피소드 -> {out_dir}', fh)
    manifest = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else {}
    t_start = time.time()
    for i, e in enumerate(jobs, 1):
        name = G.fname(e['corner'], e['ncomp'], e['seed'])
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            log(f'[{i}/{len(jobs)}] skip {name}', fh)
            continue
        t0 = time.time()
        try:
            gen_one(e['corner'], e['ncomp'], e['seed'], out_dir)
            rec = dict(status='ok', elapsed_s=time.time() - t0, corner=e['corner'],
                       ncomp=e['ncomp'], seed=e['seed'],
                       screen_verdict=e['mesh_verdict'], **_diag(path, e['Hs']))
            log(f'[{i}/{len(jobs)}] ok {name} {rec["elapsed_s"]:.0f}s '
                f'Hs4std={rec["gt_Hs_4std"]:.3f} (rel {rec["gt_Hs_rel_err"]:.1e}) '
                f'mesh={rec["mesh_verdict"]}', fh)
        except Exception as exc:                               # noqa: BLE001
            rec = dict(status='error', error=repr(exc), tb=traceback.format_exc()[-500:])
            log(f'[{i}/{len(jobs)}] ERROR {name}: {exc!r}', fh)
        if which != 'smoke':
            manifest[name] = rec
            json.dump(manifest, open(MANIFEST, 'w'), indent=1)
        el = time.time() - t_start
        log(f'    누적 {el/60:.1f} min · ETA {el/i*(len(jobs)-i)/60:.0f} min', fh)
    fh.close()


if __name__ == '__main__':
    main()
