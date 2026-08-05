"""확산 교차설계 — 처치팔(명목 60°) 10 에피소드 생성 + 대조팔 비트동일 검증.

사전등록: tmp/spread_xdesign_2026-08-05/PREREG.md
설정은 tmp/regen_run_finemesh.py 와 **완전히 동일**하고 바꾸는 것은
`direction_spread_deg` 하나뿐이다(기본 30.0 -> 60.0).

Step 0 은 승계 가정을 **검증**한다: 대조팔 1개를 30.0 으로 재생성해 기존 파일과
성분/GT 가 바이트 동일인지 확인. 불일치면 전체를 다시 생성해야 하므로 여기서 멈춘다.
"""
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import h5py
import numpy as np

from wavesim.config import WaveSimConfig
from wavesim.pipeline import run_episode

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
CTRL_DIR = os.path.join(REPO, 'data_finemesh_2026-07-23')     # 확산 30 = 기존 세트
TREAT_DIR = os.path.join(REPO, 'data_spread60_2026-08-05')    # 확산 60 = 신규
VERIFY_DIR = os.path.join(HERE, 'verify_ctrl')

NCOMPS = [5, 30]
SEEDS = [0, 1, 2, 3, 4]
SEA = 'BF4'
RAYDROP = 'uniform'
NOMINAL_TREAT = 60.0
NOMINAL_CTRL = 30.0

CFG_KW = dict(n_pts=50, duration=30.0, dt=0.1, depth=50.0,
              verbose=False, use_gpu_raycast=True)


def cfg():
    c = WaveSimConfig()
    c.mesh.nx, c.mesh.ny = 200, 167
    return c


def digest(path):
    """성분 + GT 필드의 해시. 렌더링 이전 물리와 이후 GT 를 따로 본다."""
    with h5py.File(path, 'r') as f:
        comp = np.asarray(f['config/wave_components'][:])   # [A, omega, kx, ky, phi, Q]
        gt = np.asarray(f['frames/gt_wave_field'][:])
    h_comp = hashlib.md5(np.ascontiguousarray(comp).tobytes()).hexdigest()
    h_gt = hashlib.md5(np.ascontiguousarray(gt).tobytes()).hexdigest()
    return h_comp, h_gt, comp


def realised_spread_deg(comp):
    """이산 성분집합의 에너지가중 1차 원형모멘트 확산 (사전등록 §5)."""
    A, kx, ky = comp[:, 0], comp[:, 2], comp[:, 3]
    th = np.arctan2(ky, kx)
    w = A ** 2
    r1 = np.abs((w * np.exp(1j * th)).sum() / w.sum())
    return float(np.degrees(np.sqrt(2 * (1 - r1))))


def step0_verify():
    """대조팔 승계 가능성 검증 — 1 에피소드 재생성 후 비트동일 확인."""
    os.makedirs(VERIFY_DIR, exist_ok=True)
    ss, nc, sd = SEA, NCOMPS[0], SEEDS[0]
    ref = os.path.join(CTRL_DIR, f'episode_{ss}_{nc}comp_seed{sd}_{RAYDROP}.h5')
    print(f'[step0] 대조팔 승계 검증 — {os.path.basename(ref)} 재생성')
    t0 = time.time()
    new = run_episode(sea_state=ss, n_components=nc, seed=sd, raydrop_model=RAYDROP,
                      output_dir=VERIFY_DIR, cfg=cfg(),
                      direction_spread_deg=NOMINAL_CTRL, **CFG_KW)
    hr_c, hr_g, comp_r = digest(ref)
    hn_c, hn_g, comp_n = digest(new)
    ok_c, ok_g = hr_c == hn_c, hr_g == hn_g
    print(f'  성분 md5  기존 {hr_c[:12]}  신규 {hn_c[:12]}  -> {"동일" if ok_c else "★불일치"}')
    print(f'  GT   md5  기존 {hr_g[:12]}  신규 {hn_g[:12]}  -> {"동일" if ok_g else "★불일치"}')
    print(f'  실현 확산(기존) {realised_spread_deg(comp_r):.2f} deg   ({time.time()-t0:.0f}s)')
    return ok_c, ok_g


def main():
    ok_c, ok_g = step0_verify()
    if not ok_c:
        print('\n★ 성분이 재현되지 않는다 — 승계 불가. 20개 전부 생성해야 한다. 중단.')
        return
    if not ok_g:
        print('\n⚠ 성분은 동일하나 GT 가 다르다 — 렌더링 비결정성. 기록하고 계속하되 '
              '결과 해석에 반드시 병기할 것.')

    os.makedirs(TREAT_DIR, exist_ok=True)
    man_path = os.path.join(TREAT_DIR, '_manifest.json')
    man = json.load(open(man_path)) if os.path.exists(man_path) else {}
    man.setdefault('_ctrl_inherit_verified', {'components_bit_identical': ok_c,
                                              'gt_bit_identical': ok_g})

    jobs = [(nc, sd) for nc in NCOMPS for sd in SEEDS]
    t_start = time.time()
    for i, (nc, sd) in enumerate(jobs):
        fname = f'episode_{SEA}_{nc}comp_seed{sd}_{RAYDROP}.h5'
        fp = os.path.join(TREAT_DIR, fname)
        if man.get(fname, {}).get('status') == 'ok' and os.path.exists(fp):
            print(f'[{i+1}/{len(jobs)}] SKIP {fname}')
            continue
        print(f'[{i+1}/{len(jobs)}] {fname} (nominal {NOMINAL_TREAT} deg) ...', flush=True)
        t0 = time.time()
        try:
            out = run_episode(sea_state=SEA, n_components=nc, seed=sd,
                              raydrop_model=RAYDROP, output_dir=TREAT_DIR, cfg=cfg(),
                              direction_spread_deg=NOMINAL_TREAT, **CFG_KW)
            with h5py.File(out, 'r') as f:
                nfr = f['frames/lidar_points'].shape[0]
                verdict = str(f['config'].attrs.get('mesh_verdict', 'UNKNOWN'))
                nan_gt = bool(np.isnan(f['frames/gt_wave_field'][:]).any())
            _, _, comp = digest(out)
            sp = realised_spread_deg(comp)
            man[fname] = dict(status='ok' if (nfr > 0 and not nan_gt) else 'SUSPECT',
                              nominal_spread_deg=NOMINAL_TREAT, realised_spread_deg=sp,
                              n_frames=int(nfr), mesh_verdict=verdict, nan_in_gt=nan_gt,
                              elapsed_s=time.time() - t0)
            print(f'    OK  실현확산 {sp:.2f} deg  frames={nfr}  mesh={verdict} '
                  f'[{time.time()-t0:.0f}s]')
        except Exception as e:
            import traceback
            man[fname] = dict(status='FAILED', error=str(e),
                              traceback=traceback.format_exc())
            print(f'    FAILED: {e}')
        json.dump(man, open(man_path, 'w'), indent=2, default=float)

    # 대조팔 실현확산도 같은 함수로 측정 (조작 점검용)
    ctrl = {}
    for nc in NCOMPS:
        for sd in SEEDS:
            p = os.path.join(CTRL_DIR, f'episode_{SEA}_{nc}comp_seed{sd}_{RAYDROP}.h5')
            if os.path.exists(p):
                _, _, comp = digest(p)
                ctrl[f'{nc}comp_seed{sd}'] = realised_spread_deg(comp)
    man['_ctrl_realised_spread_deg'] = ctrl
    json.dump(man, open(man_path, 'w'), indent=2, default=float)

    print(f'\n조작 점검 — 실현 확산 [deg]')
    for nc in NCOMPS:
        c = [ctrl[f'{nc}comp_seed{s}'] for s in SEEDS if f'{nc}comp_seed{s}' in ctrl]
        t = [man[f'episode_{SEA}_{nc}comp_seed{s}_{RAYDROP}.h5']['realised_spread_deg']
             for s in SEEDS
             if man.get(f'episode_{SEA}_{nc}comp_seed{s}_{RAYDROP}.h5', {}).get('status') == 'ok']
        if c and t:
            print(f'  {nc:2d}comp  30deg팔 {np.median(c):6.2f} (범위 {min(c):.2f}-{max(c):.2f})  '
                  f'->  60deg팔 {np.median(t):6.2f} (범위 {min(t):.2f}-{max(t):.2f})')
    print(f'\n총 {time.time()-t_start:.0f}s')


if __name__ == '__main__':
    main()
