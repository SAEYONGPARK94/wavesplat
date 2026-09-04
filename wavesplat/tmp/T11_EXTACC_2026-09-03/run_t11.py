"""`T11-EXTACC` v2 + **개정문 2** 러너 — 남의 참값 위에서 우리 팔을 돌리고 **귀속**한다.

⛔이 파일은 **게이트·분석기 다음에** 쓰였다(2026-09-04 규율). 행 스키마는
`analyze_t11.py` 독스트링이 정의한 것을 그대로 따른다.

★★2026-09-04 수리 1 (스모크 전, 결과 0행): 초판이 `gt_std=0.0`·`t_origin=0.0`·
`disp_relerr=0.0`·`n_consumed=min(제공, 캡)` 을 **리터럴로 적었다** ⟹ 그 네 게이트가
전부 **항진**이었다. 이제 넷 다 **실제 객체에서 되읽는다**.

★★★2026-09-04 수리 2 (`AMENDMENT_2_PREDATA.md`, **결과 0행**): 5축 다자 감사가
BLOCKER 5 · MAJOR 12 를 냈다. 이 파일에 반영된 것:
  · `predict_field` 에 **축 벡터**를 넘긴다(초판은 평탄 격자 ⟹ 16384² = 2.68억 점 ⟹ **OOM**)
  · `ORACLE` 재정의 — 켤레쌍 정준키 + **±ω 4열**(초판은 네 빈을 접고 +ω 만 ⟹ **하한**이었다)
  · `proposal_md5`(raw 제안) 되읽기 — 초판 `atoms_md5` 는 VARPRO 재적합 뒤라 **둔감**했다
  · `vp_clamp_hits`·`gt_std_after`·`proposal_device`·`dt`·`tq`·`e_shape` 되읽기 추가
  · `threads` 되읽기 · md5 를 런 **시작 전에도** 잰다 · `G_ACC` 는 배포 상수를 임포트
  · `n_consumed` → `n_consumed_formula` (산술 항등식임을 이름에 박는다)
"""
import os

# ★BLAS 스레드는 **numpy 임포트 전에** 못 박아야 한다.
#   ⚠`torch.set_num_threads(1)` 은 numpy/BLAS 를 제약하지 **않는다**(감사 지적).
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import hashlib, io as _io, json, re, sys, time, zipfile          # noqa: E402

import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
import gates_t11 as G                                                # noqa: E402
from wavesplat.optimizer import learned_init as LI                   # noqa: E402
from wavesplat.optimizer.sequential_update import FrameData, make_updater  # noqa: E402
from wavesplat.run import DEFAULT_CONFIG                             # noqa: E402

OUT = os.path.join(HERE, 'results_t11.jsonl'); META = os.path.join(HERE, 'meta_t11.json')
G_ACC = G.deployed_g()      # ★배포 단일출처(`model/dispersion.py`). ⛔초판은 9.80665 손복사


def parse(n):
    m = re.search(r'angle(\d+)h([\d.]+)\.mat', n)
    return int(m.group(1)), float(m.group(2))


def sample_obs(E, gx, gy, t, npts, rng):
    """창의 각 프레임에서 균등 무작위 `npts` 점. ★t 원점을 0 으로 옮긴다.

    ⚠축 규약: `E[j]` 를 `[iy, ix]` 로 읽는다(감사가 축교환 차이 무의미를 확인).
    """
    X, Y = np.meshgrid(gx, gy, indexing='xy')
    xs, ys, ts, zs = [], [], [], []
    t0 = float(t[0])
    for j in range(E.shape[0]):
        idx = rng.choice(E[j].size, npts, replace=False)
        iy, ix = np.unravel_index(idx, E[j].shape)
        xs.append(X[iy, ix]); ys.append(Y[iy, ix])
        ts.append(np.full(npts, float(t[j]) - t0)); zs.append(E[j][iy, ix])
    return (np.concatenate(xs), np.concatenate(ys),
            np.concatenate(ts), np.concatenate(zs))


def build_frames(x, y, tt, z, gt_const):
    """⛔`gt_field` 는 **상수** — 라우팅용이고 참 표면 정보를 0비트 담는다."""
    out = []
    for tv in np.unique(tt):
        m = tt == tv
        n = int(m.sum())
        out.append(FrameData(
            t=float(tv),
            x_obs=torch.tensor(x[m], dtype=torch.float32),
            y_obs=torch.tensor(y[m], dtype=torch.float32),
            z_obs=torch.tensor(z[m], dtype=torch.float32),
            confidence=torch.ones(n, dtype=torch.float32),
            incidence_angles=torch.zeros(n, dtype=torch.float32),
            gt_field=torch.full((8, 8), float(gt_const), dtype=torch.float32)))
    return out


def readback_frames(fr):
    """★게이트가 쓰는 양을 **만들어진 프레임에서 되읽는다** (⛔러너 리터럴 금지)."""
    gt = torch.cat([f.gt_field.reshape(-1) for f in fr])
    return dict(n_offered=int(sum(len(f.x_obs) for f in fr)),
                t_origin=float(min(float(f.t) for f in fr)),
                gt_std=float(gt.std(unbiased=False)),
                gt_value=float(gt.mean()),
                n_frames=len(fr))


def measured_scales(x_used, x_orig, t_used, t_orig):
    """★재척도가 **실제로 적용된 배율**을 나눠서 되읽고 분산관계 보존을 잰다.

    `x' = a·x` ⟹ `k' = k/a` · `t' = b·t` ⟹ `ω' = ω/b`.
    `ω=√(gk)` 가 보존되려면 `ω/b = √(g k/a)` ⟹ **`b = √a`**.
    ⛔해석적 항등식이 아니다 — 지수를 잘못 쓰면(예: `t·√s`) 이 값이 커진다.
    """
    mx = np.abs(x_orig) > 1e-9; mt = np.abs(t_orig) > 1e-9
    a = float(np.median(x_used[mx] / x_orig[mx])) if mx.any() else 1.0
    b = float(np.median(t_used[mt] / t_orig[mt])) if mt.any() else 1.0
    return a, b, float(abs(b - np.sqrt(a)) / np.sqrt(a))


def nrmse(pred, gt, s4):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(gt)) ** 2)) / s4)


def oracle(x, y, tt, z, gx, gy, tq, target, s4, nmode=None):
    """참 FFT 파수에서 `A,φ` 만 LS. ⛔추정기가 아니다 — **정보 존재의 상한**.

    ★★`AMENDMENT_2 §5` 재정의 (결과 0행 시점):
    ⛔초판 정준키 `(min(i,-i%N0), min(j,-j%N1))` 는 **네 빈**을 한 부류로 접었다 —
      `(i,j)`·`(N0-i,N1-j)`(켤레쌍) **그리고** `(i,N1-j)`·`(N0-i,j)`(±ky 미러 =
      **서로 다른 물리 모드**). 실측 확인(32×32, 두 모드 합성): 하나만 살아남았다.
    ⛔그리고 기저가 `cos(k·r − ωt)` 로 ω 부호 고정이라, 파워가 정확히 같은 켤레 빈에서
      `argsort` 타이브레이크가 반대 부호를 뽑으면 그 기저는 **역진행파** = 죽은 기저였다.
    ⟹ 초판은 상한이 아니라 **하한**이었다. 이제 (a) 켤레쌍만 접고 (b) 모드마다
      `±ω` **네 열**을 넣어 진행 방향을 자료가 고르게 한다. 여전히 "참 파수에서 A,φ 만 LS".
    """
    nmode = G.N_MODE if nmode is None else nmode
    F = np.fft.fft2(target)
    P = np.abs(F) ** 2
    kx = 2 * np.pi * np.fft.fftfreq(target.shape[1], gx[1] - gx[0])
    ky = 2 * np.pi * np.fft.fftfreq(target.shape[0], gy[1] - gy[0])
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    N0, N1 = P.shape
    order = np.argsort(P.ravel())[::-1]
    seen, ks = set(), []
    for o in order:
        i, j = np.unravel_index(o, P.shape)
        if i == 0 and j == 0:
            continue
        key = min((int(i), int(j)), (int((N0 - i) % N0), int((N1 - j) % N1)))
        if key in seen:
            continue
        seen.add(key); ks.append((float(KX[i, j]), float(KY[i, j])))
        if len(ks) == nmode:
            break
    k = np.array(ks)
    om = np.sqrt(G_ACC * np.sqrt(k[:, 0] ** 2 + k[:, 1] ** 2))
    kr = x[:, None] * k[None, :, 0] + y[:, None] * k[None, :, 1]
    wt = tt[:, None] * om[None, :]
    B = np.hstack([np.cos(kr - wt), np.sin(kr - wt),
                   np.cos(kr + wt), np.sin(kr + wt)])
    c, *_ = np.linalg.lstsq(B, z, rcond=None)
    X, Y = np.meshgrid(gx, gy, indexing='xy')
    krq = X.ravel()[:, None] * k[None, :, 0] + Y.ravel()[:, None] * k[None, :, 1]
    wtq = tq * om[None, :]
    Bq = np.hstack([np.cos(krq - wtq), np.sin(krq - wtq),
                    np.cos(krq + wtq), np.sin(krq + wtq)])
    return nrmse((Bq @ c).reshape(target.shape), target, s4)


def proposal_fingerprint(cfg):
    """★`G-ATOMS` 의 되읽기 — **raw 제안** 원자(`sequential_update.py:1122` 가 호출자
    dict 에 심는다)를 해싱한다.

    ⛔초판은 **설치된** 원자를 해싱했다. 그 사이 VARPRO 400스텝(`varpro.py:228-237`)이
    `kx1,ky1` 을 파일별 관측으로 **재적합해 덮어쓰고** `_adaptive_prune_grow` 가 원자를
    더하므로, 제안 캐시가 **완전히 새더라도** 지문은 전부 고유해졌다 ⟹ **항상 통과**.
    """
    la = cfg.get('learned_atoms')
    if not la:
        return None
    arr = np.c_[np.asarray(la['kx'], float).ravel(),
                np.asarray(la['ky'], float).ravel(),
                np.asarray(la['energy'], float).ravel()]
    return hashlib.md5(np.round(arr, 9).tobytes()).hexdigest()


def atoms_fingerprint(upd):
    """설치된 원자의 지문 — 참고용(⛔게이트가 읽지 않는다: `proposal_md5` 를 읽는다)."""
    for attr in ('model', '_primary'):
        m = getattr(upd, attr, None)
        if m is not None and hasattr(m, 'kx'):
            kx = np.asarray(m.kx.detach().cpu()); ky = np.asarray(m.ky.detach().cpu())
            return hashlib.md5(np.round(np.c_[kx, ky], 6).tobytes()).hexdigest()
        if m is not None and hasattr(m, 'model') and getattr(m.model, 'kx', None) is not None:
            kx = np.asarray(m.model.kx.detach().cpu()); ky = np.asarray(m.model.ky.detach().cpu())
            return hashlib.md5(np.round(np.c_[kx, ky], 6).tobytes()).hexdigest()
    return None


class ObsProbe:
    """★`G-CAP` 의 되읽기 — 제안 단계가 **실제로 모은** 관측 수를 센다.
    ★개정문 2: 제안이 **실제로 쓴 device** 도 되읽는다(⛔config 가 정하지 않는다 —
    `propose_atoms` 가 `cuda if available` 로 자기가 고른다).
    """
    def __enter__(self):
        self.n = None; self.device = None
        self._orig = LI._gather_init_obs
        self._orig_load = LI._load_cond

        def wrapped(frames):
            out = self._orig(frames)
            self.n = 0 if out is None else int(len(out[0]))
            return out

        def wrapped_load(ckpt, device):
            self.device = str(device)
            return self._orig_load(ckpt, device)

        LI._gather_init_obs = wrapped
        LI._load_cond = wrapped_load
        return self

    def __exit__(self, *a):
        LI._gather_init_obs = self._orig
        LI._load_cond = self._orig_load
        return False


def _clamp_hits(info):
    """`initialize()` 반환에서 `vp_clamp_hits` 를 캔다(⛔초판은 반환값을 버렸다)."""
    if not isinstance(info, dict):
        return None
    vp = info.get('varpro')
    if isinstance(vp, dict) and 'vp_clamp_hits' in vp:
        return int(vp['vp_clamp_hits'])
    return 0 if vp is None else None


def run_estimator(x, y, tt, z, gx, gy, tq, target, s4, seed):
    """⛔**config 를 매번 새로 만든다** — v1 BLOCKER-2(원자 캐시 누출) 차단."""
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(depth=G.DEPTH, seed=seed, device='cpu', init_method='auto')
    assert 'learned_atoms' not in cfg, '제안 캐시가 새 config 에 이미 있다 (BLOCKER-2)'
    upd = make_updater(cfg)
    fr = build_frames(x, y, tt, z, G.GT_CONST)
    rb = readback_frames(fr)
    with ObsProbe() as probe:
        info = upd.initialize(fr, x_range=(float(gx.min()), float(gx.max())),
                              y_range=(float(gy.min()), float(gy.max())), verbose=False)
    # ★`initialize()` **뒤에도** gt 를 되읽는다 (추정기가 본 뒤의 상태)
    gt_after = float(torch.cat([f.gt_field.reshape(-1) for f in fr]).std(unbiased=False))
    # ★2026-09-04 수리: `predict_field` 는 **축 벡터** `(Nx,)`,`(Ny,)` 를 받아 **내부에서**
    #   meshgrid 한다 — 두 경로 다 그렇다(`model/spectral_gaussian.py:145-162 predict_grid` ·
    #   `optimizer/sequential_update.py:1460 _bank_predict_grid` 가 축자로 *"same grid
    #   convention"*). 초판은 **이미 평탄화한 격자**(각 16384)를 넘겨 16384² = 2.68억 점
    #   (텐서당 1 GB)을 만들었다 ⟹ RSS 23.7 GB 로 WSL 24 GB 천장을 치고 OOM killer 가
    #   `init.scope` 를 잡아 **배포판 전체가 죽었다**(2026-09-04 두 번).
    #   ⛔램 문제가 아니다 — 반환이 2.68억 원소라 `reshape(target.shape)` 도 못 넘는다.
    pred = upd.predict_field(torch.tensor(gx, dtype=torch.float32),
                             torch.tensor(gy, dtype=torch.float32), float(tq))
    pred = np.asarray(pred.detach().cpu()).reshape(target.shape)   # (Ny, Nx) ≡ target
    rb.update(n_gathered=probe.n, cap_proposal=int(LI.M_OBS_CAP),
              n_consumed_formula=(None if probe.n is None
                                  else int(min(probe.n, LI.M_OBS_CAP))),
              proposal_device=probe.device, gt_std_after=gt_after,
              proposal_md5=proposal_fingerprint(cfg),
              vp_clamp_hits=_clamp_hits(info),
              atoms_md5=atoms_fingerprint(upd), prov=upd.arm_provenance(),
              n_atoms_installed=int(getattr(getattr(upd, 'model', None),
                                            'n_gaussians', 0) or 0))
    return nrmse(pred, target, s4), rb


def read_mat(z, name):
    import h5py
    with h5py.File(_io.BytesIO(z.read(name)), 'r') as f:
        E = np.asarray(f['eta']).squeeze()
        x = np.asarray(f['x']).squeeze(); y = np.asarray(f['y']).squeeze()
        t = np.asarray(f['t']).squeeze()
    return E, x - x.mean(), y - y.mean(), t          # ★재중심화(규약 고정)


# ★스모크 전용 파일 — **이미 노출된** 사전감사 표본에서 두 개만 쓴다.
#   ⛔프리즌 표본(`sample_t11.json`)은 스모크로 건드리지 않는다.
#   ★개정문 2: 이 목록은 `gates_t11.SMOKE_FILES` 로 **단일출처화**됐다 —
#   러너가 따로 들고 있으면 `G-SAMPLE` 이 그 배제를 기계로 확인할 수 없다.
SMOKE_FILES = list(G.SMOKE_FILES)


def main(limit=None, smoke=False):
    global OUT, META
    torch.set_num_threads(1)
    md5_start = G.freeze_map()          # ★런 **시작 전** (초판은 끝에만 쟀다)
    if smoke:
        files = SMOKE_FILES[:(limit or None)]
        OUT = OUT.replace('.jsonl', '_SMOKE.jsonl')
        META = META.replace('.json', '_SMOKE.json')
        print(f'⚠SMOKE — 노출된 감사 파일 {len(files)}개, PRIMARY 표본 미접촉')
    else:
        files = G.sample()[:(limit or None)]
    nf = int(G.WIN_S)
    rows = []
    with zipfile.ZipFile(G.ZIP) as zf:
        for fi, name in enumerate(files, 1):
            t0 = time.time()
            E, gx, gy, tarr = read_mat(zf, name)
            j0 = G.J0                      # 결과 전 고정: 창 시작 프레임
            Ew, tw = E[j0:j0 + nf], tarr[j0:j0 + nf]
            target = Ew[-1]
            s4 = 4.0 * float(np.std(Ew))
            ang, hs = parse(name)
            rng = np.random.default_rng(G.SEED + fi)
            x, y, tt, zz = sample_obs(Ew, gx, gy, tw, G.DENSITY, rng)
            tq = float(tw[-1] - tw[0])
            base = dict(block=name, angle=ang, hs=hs, sigma4=s4,
                        frame_index=j0 + nf - 1, seed=G.SEED + fi,
                        n_frames=nf, density=G.DENSITY,
                        dt=float(np.median(np.diff(tarr))), tq=tq,
                        e_shape=[int(s) for s in E.shape],
                        mean_convention='raw (⛔demean 하지 않는다 — 세 팔 동일)',
                        axis_convention='E[j][iy, ix]')
            none_rb = dict(n_gathered=None, n_consumed_formula=None,
                           t_origin=float(tt.min()),   # ★되읽기(관측 배열에서)
                           cap_proposal=int(LI.M_OBS_CAP), gt_std=None, gt_value=None,
                           gt_std_after=None, proposal_md5=None, proposal_device=None,
                           vp_clamp_hits=None, atoms_md5=None, prov={},
                           n_atoms_installed=0, rescale_s=None,
                           x_scale=None, t_scale=None, disp_relerr=None)

            # ── ZERO ──
            rows.append(dict(base, arm='ZERO',
                             nrmse=nrmse(np.zeros_like(target), target, s4),
                             n_offered=int(len(zz)), wall_s=0.0, **none_rb))
            # ── ORACLE ──
            ta = time.time()
            rows.append(dict(base, arm='ORACLE',
                             nrmse=oracle(x, y, tt, zz, gx, gy, tq, target, s4),
                             n_offered=int(len(zz)),
                             wall_s=round(time.time() - ta, 1), **none_rb))
            # ── SHIP ──
            ta = time.time()
            v, rb = run_estimator(x, y, tt, zz, gx, gy, tq, target, s4, G.SEED + fi)
            a, b, rel = measured_scales(x, x, tt, tt)
            rows.append(dict(base, arm='SHIP', nrmse=v, **rb,
                             rescale_s=None, x_scale=a, t_scale=b, disp_relerr=rel,
                             wall_s=round(time.time() - ta, 1)))
            # ── RESCALE: x'=x/s, t'=t/√s ⟹ k'=s·k 이고 ω'=√(gk') 가 보존된다 ──
            s = G.RESCALE_S
            xr, yr, ttr = x / s, y / s, tt / np.sqrt(s)
            ta = time.time()
            v2, rb2 = run_estimator(xr, yr, ttr, zz, gx / s, gy / s,
                                    tq / np.sqrt(s), target, s4, G.SEED + fi)
            a2, b2, rel2 = measured_scales(xr, x, ttr, tt)
            rows.append(dict(base, arm='RESCALE', nrmse=v2, **rb2,
                             rescale_s=s, x_scale=a2, t_scale=b2, disp_relerr=rel2,
                             wall_s=round(time.time() - ta, 1)))
            print(f"  [{fi:2d}/{len(files)}] {os.path.basename(name):20s} "
                  f"ZERO {rows[-4]['nrmse']:.4f} · ORACLE {rows[-3]['nrmse']:.4f} · "
                  f"SHIP {rows[-2]['nrmse']:.4f} · RESCALE {rows[-1]['nrmse']:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            with open(OUT, 'w', encoding='utf-8') as fh:   # ★부분 결과도 남긴다
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    md5_end = G.freeze_map()
    json.dump(dict(n_runs=sum(1 for r in rows if r['arm'] in ('SHIP', 'RESCALE')),
                   threads=int(torch.get_num_threads()),          # ★되읽기
                   threads_readback=int(torch.get_num_threads()),  # ★런 종료 후 재측정
                   blas_env={v: os.environ.get(v) for v in
                             ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                              'MKL_NUM_THREADS')},
                   seed=G.SEED, code_md5=md5_end, code_md5_start=md5_start,
                   md5_stable=(md5_start == md5_end),
                   density=G.DENSITY, win_s=G.WIN_S, depth=G.DEPTH, j0=G.J0,
                   n_mode=G.N_MODE, g_acc=G_ACC,
                   cuda_available=bool(torch.cuda.is_available()),
                   files=len(files)),
              open(META, 'w'), ensure_ascii=False, indent=1)
    print(f'✅{len(rows)} 행 · md5 안정 {md5_start == md5_end}')


if __name__ == '__main__':
    args = sys.argv[1:]
    sm = '--smoke' in args
    args = [a for a in args if a != '--smoke']
    main(int(args[0]) if args else None, smoke=sm)
