"""M1 생성기 — `SS-REAL` 해상상태에서 **짝(LIN, HOS M=3)** 48 에피소드 + PC-1.

★몽키패치 3종은 `HOSGATE`(`tmp/HOSGATE_2026-08-13/gen_hos.py`) 의 것을 **그대로** 쓴다.
  ⛔`wavesim/**` 무수정. 스냅 함수는 `gates_m1.snap_components` **하나**가 정의한다(단일출처).
★짝의 정의: 같은 (ncomp, seed) 에서 **같은 스냅 성분**으로 LIN / HOS 두 팔을 만든다.
  LiDAR·잡음·레이드롭·메시·격자·duration·dt·depth 는 전부 같다. 다른 것은
  **파면을 시간전개시키는 방식 하나**뿐이다 (`G-25` 가 성분 해시로 비트 증명).

    python3 tmp/M1_NONLIN_REAL_2026-08-28/gen_m1.py pc1      # 양성대조 (패치 OFF, ≈3분)
    python3 tmp/M1_NONLIN_REAL_2026-08-28/gen_m1.py smoke    # 1짝 스모크 (지운다)
    python3 tmp/M1_NONLIN_REAL_2026-08-28/gen_m1.py all      # 48 에피소드 — ⛔**직렬 전용**
"""
import hashlib
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

import gates_m1 as G                                           # noqa: E402
import wavesim.pipeline as PL                                  # noqa: E402
from wavesim import mesh_adequacy                              # noqa: E402
from wavesim.hos_simulator import HOSSimulator                 # noqa: E402

MANIFEST = G.MANIFEST
_LAST = {}
_ORIG_JC = PL.jonswap_components
_ORIG_HOS = PL.HOSSimulator
_PATCHED = {'on': False}


def log(msg, fh=None):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    if fh:
        fh.write(line + '\n')
        fh.flush()


# ── 몽키패치 1: 성분을 HOS 스펙트럼 격자에 스냅 (양 팔 공통) ───────────────────
def _patched_jc(*a, **kw):
    comps = _ORIG_JC(*a, **kw)
    sc, diag = G.snap_components(comps, kw.get('depth', G.G26.gen_params()['depth']))
    _LAST['components'] = sc
    _LAST['diag'] = diag
    return sc


# ── 몽키패치 2+3: 짝 초기화 + 분리가능 evaluate_at (gen_hos.py 축자) ───────────
class PairedHOS(HOSSimulator):
    def initialize_jonswap(self, Hs, Tp, **kw):                # noqa: D102
        self.initialize_from_components(_LAST['components'])

    def evaluate_at(self, X, Y):
        """`HOSSimulator.evaluate_at` 와 **같은 합**을 인수분해로 계산 (`gen_hos.py:102` 축자).

        make_grid 규약: 배열 shape (ny,nx), X[i,j]=x[j], Y[i,j]=y[i].
        ⚠아래 두 assert 는 **원본의 가드**다. 2026-08-28 적대적 관문이 이 이식본에서
        빠진 것을 잡았다 — *"축자"* 라 적었으면 가드도 축자여야 한다(결과 0행에서 복원).
        """
        assert X.ndim == 2 and X.shape == Y.shape
        x, y = X[0, :], Y[:, 0]
        assert np.allclose(X, x[None, :]) and np.allclose(Y, y[:, None]), \
            'separable evaluate_at requires a regular meshgrid'
        Ex = np.exp(1j * np.outer(x, self.KX[:, 0]))
        Ey = np.exp(1j * np.outer(y, self.KY[0, :]))
        a = self.eta_hat
        Z = np.real(Ey @ a.T @ Ex.T)
        Zx = np.real(Ey @ (1j * self.KX * a).T @ Ex.T)
        Zy = np.real(Ey @ (1j * self.KY * a).T @ Ex.T)
        mag = np.sqrt(Zx ** 2 + Zy ** 2 + 1.0)
        return Z, -Zx / mag, -Zy / mag, 1.0 / mag


def install():
    PL.jonswap_components = _patched_jc
    PL.HOSSimulator = PairedHOS
    _PATCHED['on'] = True


def uninstall():
    PL.jonswap_components = _ORIG_JC
    PL.HOSSimulator = _ORIG_HOS
    _PATCHED['on'] = False


# ── 생성 ──────────────────────────────────────────────────────────────────────
def comp_md5(comps):
    arr = np.array([[c.A, c.kx, c.ky, c.omega, c.phi, c.Q] for c in comps], float)
    return hashlib.md5(arr.tobytes()).hexdigest()


def gen_one(gt_arm, ncomp, seed, out_dir):
    """★단일 생성 진입점. 두 GT 팔·스모크가 이 함수 하나를 쓴다."""
    p = G.G26.gen_params()
    os.makedirs(out_dir, exist_ok=True)
    return PL.run_episode(
        sea_state=G.CORNER, n_components=ncomp, n_pts=p['n_pts'], seed=seed,
        raydrop_model=G.G26.RAYDROP, duration=p['duration'], dt=p['dt'], depth=p['depth'],
        output_dir=out_dir, verbose=False, cfg=G.cfg_m1(gt_arm == 'HOS'),
        use_gpu_raycast=True,
        random_amplitudes=True, random_frequencies=True, decorrelate_stream=True,
    )


def _skew(a):
    a = np.asarray(a, float).ravel()
    m, s = a.mean(), a.std()
    return float(((a - m) ** 3).mean() / s ** 3) if s > 0 else 0.0


def _diag(path, gt_arm, ncomp, seed, pair_path=None):
    with h5py.File(path, 'r') as f:
        gt = f['frames/gt_wave_field'][:]
        a = dict(f['config'].attrs)
    rec = dict(status='ok', gt_arm=gt_arm, ncomp=ncomp, seed=seed,
               n_frames=int(gt.shape[0]), gt_Hs_4std=float(4 * np.std(gt)),
               gt_absmax=float(np.abs(gt).max()), gt_skew=_skew(gt),
               attr_Hs=float(a['Hs']), attr_Tp=float(a['Tp']),
               attr_mesh_verdict=str(a.get('mesh_verdict')),
               attr_var_under=float(a.get('var_under_resolved', np.nan)),
               n_wrapped=_LAST['diag']['n_wrapped'],
               max_snap_dk=_LAST['diag']['max_snap_dk'],
               comp_md5=comp_md5(_LAST['components']))
    # ★스냅 **후** 성분으로 메시 판정을 다시 본다 (화면은 스냅 전에 봤다) — G-27
    info = mesh_adequacy.check(_LAST['components'], G.cfg_m1(False), hos_order=G.HOS_M)
    rec['mesh_verdict_snapped'] = info['mesh_verdict']
    rec['var_under_snapped'] = float(info['var_under_resolved'])
    if gt_arm == 'HOS' and pair_path and os.path.exists(pair_path):
        with h5py.File(pair_path, 'r') as f:
            lin = f['frames/gt_wave_field'][:]
        rec['gt_absdiff_vs_pair'] = float(np.abs(gt - lin).max()) if gt.shape == lin.shape else None
        rec['gt_skew_pair_lin'] = _skew(lin)
    return rec


def pc1(out_dir=None):
    """★PC-1 — B-26 의 **`SS-REAL` 선형** 에피소드를 이 배치의 호출로 재생성해 비트 동일 요구.

    ⛔패치 OFF · 메시는 B-26 값(200×167) · `hos.enabled=False`.
    ⟹ *"우리 생성기가 B-26 의 생성기와 같은 호출인가"* 를 산문이 아니라 **비트로** 답한다."""
    assert not _PATCHED['on'], 'PC-1 은 몽키패치 OFF 에서 돌아야 한다'
    out_dir = out_dir or G.PC1_DIR
    dst = os.path.join(out_dir, os.path.basename(G.PC1_PATH))
    if not os.path.exists(dst):
        p = G.G26.gen_params()
        os.makedirs(out_dir, exist_ok=True)
        t0 = time.time()
        PL.run_episode(sea_state=G.CORNER, n_components=G.PC1['n_components'],
                       n_pts=p['n_pts'], seed=G.PC1['seed'], raydrop_model=G.G26.RAYDROP,
                       duration=p['duration'], dt=p['dt'], depth=p['depth'],
                       output_dir=out_dir, verbose=False, cfg=G.G26._cfg(),
                       use_gpu_raycast=True, random_amplitudes=True,
                       random_frequencies=True, decorrelate_stream=True)
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
    json.dump(r, open(os.path.join(HERE, 'pc1_m1.json'), 'w'), indent=1)
    for name, ok, msg in G.gates_pc1(r):
        log(f'  [{"PASS" if ok else "FAIL"}] {name}: {msg}')
    return r


def main():
    which = (sys.argv[1:] or ['all'])[0]
    if which == 'pc1':
        pc1()
        return
    nshard = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    if nshard > 1 and '--allow-parallel' not in sys.argv:
        print('⛔사전등록이 **생성 직렬**을 명시한다 — GPU 가 직렬화해 병렬 소득이 0 이다\n'
              '  (B-26 실측: 4-way 병렬 695 s/4ep == 직렬 4×172.6 s).\n'
              '  2026-08-27 에 오케스트레이터가 이 규칙을 어겨 34배 느려진 전례가 있다.', flush=True)
        sys.exit(1)
    install()
    eps, _ = G.select()
    pairs = sorted({(e['ncomp'], e['seed']) for e in eps})
    if which == 'smoke':
        pairs, out_root = pairs[:1], G.SMOKE
    else:
        out_root = G.DATA
    manifest = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else {}
    fh = open(os.path.join(HERE, 'gen_m1.log'), 'a')
    log(f'=== M1 gen {which}: {len(pairs)} 짝 (LIN+HOS) -> {out_root} · '
        f'HOS {G.HOS_M}차 {G.HOS_NX}×{G.HOS_NY} L=({G.HOS_LX:.0f},{G.HOS_LY:.0f}) dt={G.HOS_DT} · '
        f'메시 {G.MESH_NX}×{G.MESH_NY}', fh)
    t_start = time.time()
    for i, (ncomp, seed) in enumerate(pairs, 1):
        lin_path = None
        for gt_arm in G.GT_ARMS:              # ⛔LIN 먼저 — HOS 가 짝을 되읽는다
            out_dir = os.path.join(out_root, gt_arm)
            path = os.path.join(out_dir, G.fname(ncomp, seed))
            key = f'{gt_arm}/{G.fname(ncomp, seed)}'
            if gt_arm == 'LIN':
                lin_path = path
            if os.path.exists(path):
                log(f'[{i}/{len(pairs)}] skip {key}', fh)
                continue
            t0 = time.time()
            try:
                gen_one(gt_arm, ncomp, seed, out_dir)
                rec = _diag(path, gt_arm, ncomp, seed, pair_path=lin_path)
                rec['elapsed_s'] = time.time() - t0
                log(f'[{i}/{len(pairs)}] ok {key} {rec["elapsed_s"]:.0f}s '
                    f'Hs4std={rec["gt_Hs_4std"]:.3f} skew={rec["gt_skew"]:+.4f} '
                    f'wrap={rec["n_wrapped"]} dk={rec["max_snap_dk"]:.5f} '
                    f'mesh(snap,M3)={rec["mesh_verdict_snapped"]}', fh)
            except Exception as exc:                            # noqa: BLE001
                rec = dict(status='error', gt_arm=gt_arm, ncomp=ncomp, seed=seed,
                           error=repr(exc), tb=traceback.format_exc()[-600:])
                log(f'[{i}/{len(pairs)}] ERROR {key}: {exc!r}', fh)
            if which != 'smoke':
                manifest[key] = rec
                json.dump(manifest, open(MANIFEST, 'w'), indent=1)
        el = time.time() - t_start
        log(f'    누적 {el/60:.1f} min · ETA {el/i*(len(pairs)-i)/60:.0f} min', fh)
    fh.close()


if __name__ == '__main__':
    main()
