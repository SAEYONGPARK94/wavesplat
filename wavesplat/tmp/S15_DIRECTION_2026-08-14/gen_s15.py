#!/usr/bin/env python3
"""S15 생성기 — `direction_deg=180` 쌍둥이 32 에피소드만 만든다.

사전등록 `PREREG.md` (md5 는 `FROZEN_md5.txt`) §2 를 그대로 구현.

    python3 tmp/S15_DIRECTION_2026-08-14/gen_s15.py            # 전량 32
    python3 tmp/S15_DIRECTION_2026-08-14/gen_s15.py --only 2   # 앞 2개만 (스모크용)
    python3 tmp/S15_DIRECTION_2026-08-14/gen_s15.py --check    # 생성 0회, 짝 검사만

★`dir0` 팔은 **만들지 않는다.** 기존
    data_indep_crowded_2026-08-08/  (혼잡 16, S1/S4 코퍼스)
    data_indep_sparse_2026-08-08/   (희소 16)
를 그대로 쓴다 — `probe_dir.py` 가 오늘 코드로 그 32 파일의 성분을 `max|Δ|=0` 으로
재현함을 이미 확인했다(PREREG §0). 그래서 dir0·λ=1e-3 팔은 **S4 의 재실행**이 되고
재현 게이트 G-0 이 공짜로 붙는다.

생성 상수는 `gen_crowded.py` / `gen_sparse.py` (2026-08-08) 와 **동일**하다:
    sea_state, n_components, seed 20-23, raydrop 'uniform', duration 30.0, dt 0.1,
    depth 50.0, n_pts 50, mesh 200x167, GPU raycast,
    random_amplitudes / random_frequencies / decorrelate_stream = True
바뀌는 것은 `direction_deg = 0.0 -> 180.0` **하나뿐**이다.
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # .../wavesplat/wavesplat
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import h5py                                                    # noqa: E402
from wavesim.config import WaveSimConfig                       # noqa: E402
from wavesim.pipeline import run_episode                       # noqa: E402

OUT = os.path.join(HERE, 'data_dir180')
MANIFEST = os.path.join(OUT, '_manifest.json')
LOG = os.path.join(HERE, 'gen.log')

# PREREG §2 동결 상수
NX, NY = 200, 167
DURATION, DT, DEPTH, N_PTS = 30.0, 0.1, 50.0, 50
SPREAD = 30.0
DIRECTION = 180.0
EP_SEEDS = (20, 21, 22, 23)

DIR0 = {'crowded': 'data_indep_crowded_2026-08-08',
        'sparse': 'data_indep_sparse_2026-08-08'}

# (stratum, bf, ncomp)
CELLS = ([('crowded', bf, nc) for bf in ('BF4', 'BF5') for nc in (15, 30)]
         + [('sparse', bf, nc) for bf in ('BF4', 'BF5') for nc in (2, 5)])
JOBS = [(st, bf, nc, s) for (st, bf, nc) in CELLS for s in EP_SEEDS]


def dir0_path(st, bf, nc, s):
    return os.path.join(ROOT, DIR0[st],
                        f'episode_{bf}_{nc}comp_seed{s}_uniform_ra_rf_ds.h5')


def dir180_path(bf, nc, s):
    # pipeline.py:257-269 의 명명 규약: raydrop 뒤에 dtag, 그 뒤 ra/rf/ds
    return os.path.join(OUT, f'episode_{bf}_{nc}comp_seed{s}_uniform_dir180_ra_rf_ds.h5')


def log(msg, fh=None):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    if fh:
        fh.write(line + '\n')
        fh.flush()


def check_pairs(verbose=True):
    """PREREG G-2/G-3 의 생성기 쪽 절반: 성분 수준 짝 검사. 추정기 0회 호출."""
    ok, rows = True, []
    for st, bf, nc, s in JOBS:
        p0, p1 = dir0_path(st, bf, nc, s), dir180_path(bf, nc, s)
        if not os.path.exists(p0):
            log(f'  MISSING dir0   {p0}')
            ok = False
            continue
        if not os.path.exists(p1):
            if verbose:
                log(f'  (아직 없음) dir180 {os.path.basename(p1)}')
            ok = False
            continue
        with h5py.File(p0, 'r') as f:
            c0 = np.asarray(f['config/wave_components'])
            t0 = f['frames/timestamps'][:]
            n0 = f['frames/gt_wave_field'].shape[0]
        with h5py.File(p1, 'r') as f:
            c1 = np.asarray(f['config/wave_components'])
            t1 = f['frames/timestamps'][:]
            n1 = f['frames/gt_wave_field'].shape[0]
        # 열 규약: A, omega, kx, ky, phi (, Q)
        dA = float(np.abs(c0[:, 0] - c1[:, 0]).max())
        dw = float(np.abs(c0[:, 1] - c1[:, 1]).max())
        dphi = float(np.abs(c0[:, 4] - c1[:, 4]).max())
        dk = float(np.abs(c0[:, 2:4] + c1[:, 2:4]).max())
        same_t = (n0 == n1) and t0.shape == t1.shape and np.allclose(t0, t1)
        kx0, kx1 = float(c0[:, 2].mean()), float(c1[:, 2].mean())
        flipped = (kx1 < 0.0 < kx0)
        good = (dA <= 1e-12 and dw <= 1e-12 and dphi <= 1e-12 and dk <= 1e-12
                and same_t and flipped)
        ok &= good
        rows.append((f'{bf}_{nc}comp_seed{s}', dA, dw, dphi, dk, same_t, kx0, kx1, good))
        if verbose:
            print(f'  {bf}_{nc}comp_seed{s:<3d} dA={dA:.1e} dw={dw:.1e} dphi={dphi:.1e} '
                  f'|k180+k0|={dk:.1e} frames={n0}/{n1} '
                  f'mean_kx {kx0:+.4f}->{kx1:+.4f}  {"OK" if good else "FAIL"}')
    print(f'  짝 검사: {sum(1 for r in rows if r[-1])}/{len(JOBS)} 통과')
    return ok, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', type=int, default=None, help='앞 N 개만 생성 (스모크)')
    ap.add_argument('--keys', default=None,
                    help='쉼표구분 pair 키만 생성, 예 BF4_15comp_seed20,BF4_2comp_seed20')
    ap.add_argument('--check', action='store_true', help='생성 0회, 짝 검사만')
    args = ap.parse_args()

    if args.check:
        ok, _ = check_pairs()
        sys.exit(0 if ok else 1)

    os.makedirs(OUT, exist_ok=True)
    cfg = WaveSimConfig()
    cfg.mesh.nx, cfg.mesh.ny = NX, NY
    assert cfg.hos.enabled is False, 'HOS 는 비활성이어야 한다 (PREREG §2)'

    manifest = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else {}
    jobs = JOBS
    if args.keys:
        want = set(args.keys.split(','))
        jobs = [j for j in JOBS if f'{j[1]}_{j[2]}comp_seed{j[3]}' in want]
        assert jobs, f'키가 아무 셀에도 안 맞는다: {sorted(want)}'
    elif args.only:
        jobs = JOBS[:args.only]
    fh = open(LOG, 'a')
    log(f'S15 dir180 생성 {len(jobs)} 에피소드 -> {OUT}/  (direction_deg={DIRECTION})', fh)

    for i, (st, bf, nc, s) in enumerate(jobs, 1):
        out_path = dir180_path(bf, nc, s)
        name = os.path.basename(out_path)
        if os.path.exists(out_path):
            log(f'[{i}/{len(jobs)}] skip (있음) {name}', fh)
            continue
        p0 = dir0_path(st, bf, nc, s)
        if not os.path.exists(p0):
            log(f'[{i}/{len(jobs)}] ERROR dir0 짝이 없다: {p0}', fh)
            manifest[name] = dict(status='error', error='dir0 pair missing')
            continue
        # 짝의 시간축/깊이를 dir0 파일에서 직접 읽어 상수 표류를 원천 차단한다
        with h5py.File(p0, 'r') as f:
            ts = f['frames/timestamps'][:]
            depth = float(f['config'].attrs['depth'])
        duration = float(ts[-1] - ts[0])
        dt = float(ts[1] - ts[0])
        assert abs(duration - DURATION) < 1e-9 and abs(dt - DT) < 1e-9, (duration, dt)
        assert abs(depth - DEPTH) < 1e-9, depth

        t0 = time.time()
        try:
            path = run_episode(
                sea_state=bf, n_components=nc, n_pts=N_PTS, seed=s,
                raydrop_model='uniform', duration=duration, dt=dt, depth=depth,
                output_dir=OUT, verbose=False, cfg=cfg,
                direction_deg=DIRECTION, direction_spread_deg=SPREAD,
                use_gpu_raycast=True,
                random_amplitudes=True, decorrelate_stream=True,
                random_frequencies=True,
            )
            el = time.time() - t0
            assert os.path.abspath(path) == os.path.abspath(out_path), (path, out_path)
            with h5py.File(path, 'r') as f:
                nfr = f['frames/gt_wave_field'].shape[0]
                a = dict(f['config'].attrs)
                comp = np.asarray(f['config/wave_components'])
            manifest[name] = dict(
                status='ok', elapsed_s=el, n_frames=nfr, stratum=st,
                bf=bf, ncomp=nc, ep_seed=s, direction_deg=DIRECTION,
                mean_kx=float(comp[:, 2].mean()),
                random_amplitudes=bool(a.get('random_amplitudes', False)),
                random_frequencies=bool(a.get('random_frequencies', False)),
                decorrelate_stream=bool(a.get('decorrelate_stream', False)),
                mesh_verdict=str(a.get('mesh_verdict', '?')))
            log(f'[{i}/{len(jobs)}] ok {name} {el:.0f}s frames={nfr} '
                f'mean_kx={comp[:, 2].mean():+.4f} mesh={a.get("mesh_verdict", "?")}', fh)
        except Exception as e:
            manifest[name] = dict(status='error', error=str(e))
            log(f'[{i}/{len(jobs)}] ERROR {name}: {e}', fh)
            traceback.print_exc()
        json.dump(manifest, open(MANIFEST, 'w'), indent=1)

    ok = sum(1 for v in manifest.values() if v.get('status') == 'ok')
    log(f'생성 완료: ok {ok} / {len(manifest)}', fh)
    fh.close()
    check_pairs(verbose=False)


if __name__ == '__main__':
    main()
