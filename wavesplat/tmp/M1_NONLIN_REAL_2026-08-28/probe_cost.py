"""M1 비용 탐침 — 격자 선택을 **산술이 아니라 실측**으로 고른다 (2026-08-27 교훈).

  python3 probe_cost.py <mesh_nx> <mesh_ny> <hos:0|1> [hos_Nx] [hos_Ny] [duration]

`run_episode` 를 짧은 duration 으로 돌려 프레임당 비용을 재고 301 프레임으로 외삽한다.
HOS 는 프레임당 고정 5 스텝(dt 0.1 / hos.dt 0.02)이므로 외삽이 성립한다.
⛔ 산출 .h5 는 지운다. 이 파일은 배치의 결과가 아니라 **설계 근거**다.
"""
import os, sys, time, json, shutil
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import h5py
import wavesim.pipeline as PL
from wavesim.config import WaveSimConfig
from wavesim.hos_simulator import HOSSimulator

G = 9.81
DEPTH, N_PTS, DT = 50.0, 50, 0.1
SS_REAL = {'Hs': 0.725, 'Tp': 3.49}

# ── gen_hos.py 의 몽키패치 3종을 그대로 (HOS 팔에서만 설치) ────────────────
_LAST = {}
_ORIG_JC = PL.jonswap_components


def install_hos_patches(NX, NY, LX, LY):
    dkx, dky = 2 * np.pi / LX, 2 * np.pi / LY
    KX1 = np.fft.fftfreq(NX, d=1.0 / (NX * dkx))
    KY1 = np.fft.fftfreq(NY, d=1.0 / (NY * dky))

    def snap(components, depth):
        out, moved, wrapped = [], [], 0
        for c in components:
            rx, ry = c.kx / dkx, c.ky / dky
            ix, iy = int(round(rx)) % NX, int(round(ry)) % NY
            if abs(round(rx)) > NX // 2 or abs(round(ry)) > NY // 2:
                wrapped += 1
            kx, ky = float(KX1[ix]), float(KY1[iy])
            k = float(np.hypot(kx, ky))
            w = float(np.sqrt(G * k * np.tanh(np.clip(k * depth, 0, 20)))) if k > 0 else 0.0
            moved.append(float(np.hypot(kx - c.kx, ky - c.ky)))
            out.append(type(c)(A=c.A, omega=w, kx=kx, ky=ky, phi=c.phi, Q=c.Q))
        _LAST['max_snap_dk'] = max(moved) if moved else 0.0
        _LAST['n_wrapped'] = wrapped
        return out

    def patched(*a, **kw):
        comps = _ORIG_JC(*a, **kw)
        sc = snap(comps, kw.get('depth', DEPTH))
        _LAST['components'] = sc
        return sc

    class PairedHOS(HOSSimulator):
        def initialize_jonswap(self, Hs, Tp, **kw):
            self.initialize_from_components(_LAST['components'])

        def evaluate_at(self, X, Y):
            x, y = X[0, :], Y[:, 0]
            Ex = np.exp(1j * np.outer(x, self.KX[:, 0]))
            Ey = np.exp(1j * np.outer(y, self.KY[0, :]))
            a = self.eta_hat
            Z = np.real(Ey @ a.T @ Ex.T)
            Zx = np.real(Ey @ (1j * self.KX * a).T @ Ex.T)
            Zy = np.real(Ey @ (1j * self.KY * a).T @ Ex.T)
            mag = np.sqrt(Zx ** 2 + Zy ** 2 + 1.0)
            return Z, -Zx / mag, -Zy / mag, 1.0 / mag

    PL.jonswap_components = patched
    PL.HOSSimulator = PairedHOS


def main():
    mnx, mny, hos = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    HNX = int(sys.argv[4]) if len(sys.argv) > 4 else 192
    HNY = int(sys.argv[5]) if len(sys.argv) > 5 else 128
    dur = float(sys.argv[6]) if len(sys.argv) > 6 else 2.0
    nc, seed = 30, 9001                      # 최악(성분 30) · 배치 밖 시드
    LX, LY = 600.0, 400.0

    cfg = WaveSimConfig()
    cfg.sea_states = dict(getattr(cfg, 'sea_states', {}) or {})
    cfg.sea_states['SS-REAL'] = dict(SS_REAL)
    cfg.mesh.nx, cfg.mesh.ny = mnx, mny
    cfg.hos.enabled = bool(hos)
    if hos:
        cfg.hos.M, cfg.hos.Nx, cfg.hos.Ny = 3, HNX, HNY
        cfg.hos.Lx, cfg.hos.Ly, cfg.hos.dt = LX, LY, 0.02
        install_hos_patches(HNX, HNY, LX, LY)

    out = os.path.join('/tmp/claude-1000', 'm1probe')
    shutil.rmtree(out, ignore_errors=True); os.makedirs(out, exist_ok=True)
    t0 = time.time()
    try:
        path = PL.run_episode(sea_state='SS-REAL', n_components=nc, n_pts=N_PTS,
                              seed=seed, raydrop_model='uniform', duration=dur,
                              dt=DT, depth=DEPTH, output_dir=out, verbose=False,
                              cfg=cfg, use_gpu_raycast=True, random_amplitudes=True,
                              random_frequencies=True, decorrelate_stream=True)
        el = time.time() - t0
        with h5py.File(path, 'r') as f:
            nfr = f['frames/gt_wave_field'].shape[0]
            attrs = dict(f['config'].attrs)
        rec = dict(ok=True, mesh=[mnx, mny], hos=bool(hos), hos_grid=[HNX, HNY],
                   n_tri=2 * (mnx - 1) * (mny - 1), duration=dur, frames=nfr,
                   elapsed_s=round(el, 1), s_per_frame=round(el / max(nfr, 1), 3),
                   est_301_s=round(el / max(nfr, 1) * 301, 0),
                   mesh_verdict=str(attrs.get('mesh_verdict', '?')),
                   var_under=float(attrs.get('var_under_resolved', np.nan)),
                   n_wrapped=_LAST.get('n_wrapped'), max_snap_dk=_LAST.get('max_snap_dk'))
    except Exception as e:
        rec = dict(ok=False, mesh=[mnx, mny], hos=bool(hos), hos_grid=[HNX, HNY],
                   n_tri=2 * (mnx - 1) * (mny - 1), duration=dur,
                   elapsed_s=round(time.time() - t0, 1),
                   error=f'{type(e).__name__}: {str(e)[:300]}')
    shutil.rmtree(out, ignore_errors=True)
    print(json.dumps(rec))
    with open(os.path.join(HERE, 'probe_cost.jsonl'), 'a') as fh:
        fh.write(json.dumps(rec) + '\n')


if __name__ == '__main__':
    main()
