"""M3 생성기 — 지지집합-밖 3셀 48 에피소드 (+ PC-1 비트동일 양성대조).

⛔**`wavesim/**` 무수정 · 몽키패치 0개.** 세 좌표는 전부 **이미 노출된 인자**로 넣는다:
`run_episode(direction_deg=…, direction_spread_deg=…, depth=…)` (`wavesim/pipeline.py:73,77,78`).
호출 계보는 `B-26 gen_b26.gen_one` == `gen_s8.py:107-114` (= `B-1` 헤드라인 코퍼스의 생성기).
같음의 **증명**은 산문이 아니라 `pc1()` — 좌표를 배포값(0°/30°/50 m)으로 두고 동결
에피소드를 재생성해 **비트 동일**을 요구한다(G-22).

⚠파일명 함정(사전등록 §7): `run_episode` 는 방향에만 `_dir` 태그를 붙이고 **방향폭·수심은
파일명에 없다** ⟹ 셀마다 **하위 디렉터리**로 분리하고, 처치는 파일명이 아니라
**h5 attrs + 성분 파생측정**(G-7a–d·G-9)으로 검증한다.

    python3 tmp/M3_SUPPORT_2026-08-28/gen_m3.py pc1              # 양성대조만 (≈3분)
    python3 tmp/M3_SUPPORT_2026-08-28/gen_m3.py smoke            # 1셀 스모크 (지운다)
    python3 tmp/M3_SUPPORT_2026-08-28/gen_m3.py all              # 48 에피소드, ⛔직렬
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

import gates_m3 as G                                           # noqa: E402
from wavesim.pipeline import run_episode                       # noqa: E402

MANIFEST = os.path.join(HERE, 'gen_manifest_m3.json')


def log(msg, fh=None):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    if fh:
        fh.write(line + '\n')
        fh.flush()


def gen_one(ncomp, seed, out_dir, direction_deg, spread_deg, depth):
    """★단일 생성 진입점 — **모든 셀·PC-1 이 이 함수 하나를 쓴다.**
    좌표 셋은 **키워드 필수**(⛔기본값 없음 — 빈 호출로 팔이 정의되는 부류를 막는다)."""
    p = G.gen_params()
    os.makedirs(out_dir, exist_ok=True)
    return run_episode(
        sea_state=G.SEA_STATE, n_components=ncomp, n_pts=p['n_pts'], seed=seed,
        raydrop_model=G.RAYDROP, duration=p['duration'], dt=p['dt'],
        depth=depth, direction_deg=direction_deg, direction_spread_deg=spread_deg,
        output_dir=out_dir, verbose=False, cfg=G._cfg(),
        use_gpu_raycast=True,
        random_amplitudes=True, random_frequencies=True, decorrelate_stream=True,
    )


def _diag(path):
    with h5py.File(path, 'r') as f:
        gt = f['frames/gt_wave_field'][:]
        a = dict(f['config'].attrs)
    d = G.derived_coords(G._read_ep(path))
    return dict(n_frames=int(gt.shape[0]), gt_Hs_4std=float(4 * np.std(gt)),
                attr_depth=float(a['depth']),
                attr_spread=float(a['direction_spread_deg']),
                mesh_verdict=str(a.get('mesh_verdict')),
                var_under=float(a.get('var_under_resolved', np.nan)), **d)


def pc1(out_dir=None):
    """★PC-1 — 좌표=배포값으로 동결 `B-1` 코퍼스 에피소드를 재생성, **비트 동일** 요구."""
    out_dir = out_dir or os.path.join(HERE, 'pc1')
    dst = os.path.join(out_dir, os.path.basename(G.PC1_PATH))
    if not os.path.exists(dst):
        t0 = time.time()
        c = G.DEPLOY_COORD
        gen_one(G.PC1['n_components'], G.PC1['seed'], out_dir,
                direction_deg=c['direction_deg'], spread_deg=c['spread_deg'],
                depth=c['depth'])
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
    json.dump(r, open(os.path.join(HERE, 'pc1_m3.json'), 'w'), indent=1)
    for name, ok, msg in G.gates_pc1(r):
        log(f'  [{"PASS" if ok else "FAIL"}] {name}: {msg}')
    return r


def main():
    which = (sys.argv[1:] or ['all'])[0]
    if which == 'pc1':
        pc1()
        return
    # ★AMENDMENT_1: 직렬 방벽 — gen_m1.py 와 같은 기계 (종전에는 독스트링 산문뿐이었다).
    #   이 생성기는 샤딩을 아예 받지 않으므로, 샤드 인자를 주는 호출 자체를 병렬 시도로 본다.
    nshard = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    if nshard > 1 and '--allow-parallel' not in sys.argv:
        print('⛔사전등록이 **생성 직렬**을 명시한다 — GPU 가 직렬화해 병렬 소득이 0 이다\n'
              '  (B-26 실측: 4-way 병렬 695 s/4ep == 직렬 4×172.6 s).\n'
              '  2026-08-27 에 오케스트레이터가 이 규칙을 어겨 34배 느려진 전례가 있다.', flush=True)
        sys.exit(1)
    eps, _ = G.select()
    base = G.DATA
    if which == 'smoke':
        eps = [e for e in eps if e['cell'] == G.PRIMARY_CELL][:1]
        base = G.SMOKE
    fh = open(os.path.join(HERE, 'gen_m3.log'), 'a')
    log(f'=== M3 gen {which}: {len(eps)} 에피소드 -> {base} (⛔직렬 — GPU 가 직렬화한다)', fh)
    manifest = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else {}
    t_start = time.time()
    for i, e in enumerate(eps, 1):
        out_dir = G.cell_dir(e['cell'], base)
        path = os.path.join(out_dir, G.fname(e['cell'], e['ncomp'], e['seed']))
        if os.path.exists(path):
            log(f'[{i}/{len(eps)}] skip {e["episode"]}', fh)
            continue
        t0 = time.time()
        try:
            gen_one(e['ncomp'], e['seed'], out_dir,
                    direction_deg=e['direction_deg'], spread_deg=e['spread_deg'],
                    depth=e['depth'])
            rec = dict(status='ok', elapsed_s=time.time() - t0, cell=e['cell'],
                       ncomp=e['ncomp'], seed=e['seed'],
                       screen_verdict=e['mesh_verdict'], **_diag(path))
            log(f'[{i}/{len(eps)}] ok {e["episode"]} {rec["elapsed_s"]:.0f}s '
                f'Hs4std={rec["gt_Hs_4std"]:.3f} cos={rec["mean_cos"]:+.2f} '
                f'spr={rec["circ_std_deg"]:.1f} d50resid={rec["disp_resid_50"]:.2e} '
                f'mesh={rec["mesh_verdict"]}', fh)
        except Exception as exc:                               # noqa: BLE001
            rec = dict(status='error', error=repr(exc), tb=traceback.format_exc()[-500:])
            log(f'[{i}/{len(eps)}] ERROR {e["episode"]}: {exc!r}', fh)
        if which != 'smoke':
            manifest[e['episode']] = rec
            json.dump(manifest, open(MANIFEST, 'w'), indent=1)
        el = time.time() - t_start
        log(f'    누적 {el/60:.1f} min · ETA {el/i*(len(eps)-i)/60:.0f} min', fh)
    fh.close()


if __name__ == '__main__':
    main()
