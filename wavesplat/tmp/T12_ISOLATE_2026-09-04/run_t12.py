"""`T12-ISOLATE` 러너 — ⛔`consts_t12.py` → `analyze_t12.py`/`gates_t12.py` **다음에** 쓰였다.

행 스키마는 `analyze_t12.py` 독스트링이 정의한 것을 그대로 따른다.

★**관측 표집·프레임·오라클·재척도는 `T11 run_t11.py` 를 임포트해 재사용**한다 —
  시드 규약(`SEED+fi`)과 표집을 **비트 보존**해야 `G-ANCHOR` 가 성립한다(PREREG §2.1).

★**개입은 컨텍스트 매니저로 걸고 반드시 되돌린다**(`G-KNOB`/`G-RESTORE`).
⛔`REACHABILITY.md §5` 항목 5: `varpro.VP_LR_DEFAULT` 재대입은 **안 먹는다**(기본 인자가
  `def` 시점에 바인딩) ⟹ **함수 객체 자체**를 갈아끼운다. 개입 계수기 없이 이 축을
  처치하면 **조용한 no-op 이 「널」로 오보고**된다.
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import hashlib, json, sys, time, zipfile                          # noqa: E402

import numpy as np                                                # noqa: E402
import torch                                                      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import consts_t12 as C                                            # noqa: E402
sys.path.insert(0, C.ROOT)
sys.path.insert(0, os.path.join(C.ROOT, 'scripts'))
sys.path.insert(0, os.path.abspath(C.T11))
import run_t11 as R11                                             # noqa: E402  ★T11 재사용
from wavesplat.optimizer import learned_init as LI                # noqa: E402
from wavesplat.optimizer import initializer as INIT               # noqa: E402
from wavesplat.optimizer import varpro as VP                      # noqa: E402
from wavesplat.model import spectral_gaussian as SG               # noqa: E402
from wavesplat.optimizer.sequential_update import make_updater    # noqa: E402
from wavesplat.run import DEFAULT_CONFIG                          # noqa: E402
import d3_beamform_net as BN                                      # noqa: E402  ⛔bare name 필수

OUT = os.path.join(HERE, 'results_t12.jsonl'); META = os.path.join(HERE, 'meta_t12.json')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def rng_md5():
    """전역 numpy 상태 지문 (G-RNGSTREAM)."""
    st = np.random.get_state()
    return hashlib.md5(np.asarray(st[1]).tobytes() + str(st[2]).encode()).hexdigest()


def read_knobs(cfg=None, upd=None):
    """★**살아있는 모듈·객체 상태**를 되읽는다 (G-KNOB/G-RESTORE, PREREG §6).

    ⛔러너가 처치값을 그대로 적는 것이 아니다 — 개입이 조용히 실패하면 여기서 드러난다.
    """
    k = dict(KLIM=float(BN.KLIM), KX_G00=float(BN.KX_G[0, 0].item()),
             OM_Gmd5=hashlib.md5(BN.OM_G.detach().cpu().numpy().tobytes()).hexdigest(),
             MAP_DEPTH=float(BN.DEPTH),
             MIN_DK=float(INIT.MIN_DK),
             vp_lr=float(getattr(VP.apply_varpro, '_t12_lr', VP.VP_LR_DEFAULT)),
             grow_krange=float(getattr(INIT._beamform_coarse, '_t12_kr', 4.0)))
    try:                                    # ★K_LIM 은 (래핑됐을 수도 있는) 로더에서 되읽는다
        k['K_LIM'] = float(LI._load_cond(LI.DEFAULT_CKPT, DEV)[2]['K_LIM'])
    except Exception:                                             # noqa: BLE001
        k['K_LIM'] = None
    src = (upd.config if upd is not None and hasattr(upd, 'config') else (cfg or {}))
    k['max_steepness'] = float(src.get('max_steepness', 0.15))
    k['learned_dedup_dk'] = float(src.get('learned_dedup_dk', 0.03))
    return k


class Arm:
    """팔 개입을 걸고 **반드시 되돌린다**. ⛔되돌리기 실패 = 프로세스 전역 오염."""

    def __init__(self, name):
        self.name = name
        self.spec = dict(C.TREAT.get(name, {}))
        self.grow_calls = 0
        self.grow_atoms = 0
        self.n_raw = None
        self.n_sel = None
        self.kmed_raw = None
        self.kmed_sel = None
        self.device = None

    def cfg_overrides(self):
        return {k: v for k, v in self.spec.items()
                if k in ('learned_dedup_dk', 'max_steepness')}

    def __enter__(self):
        self._undo = []
        need_grid = False
        if 'KLIM' in self.spec:
            old = BN.KLIM; BN.KLIM = float(self.spec['KLIM']); need_grid = True
            self._undo.append(lambda: setattr(BN, 'KLIM', old))
        if 'MAP_DEPTH' in self.spec:
            old = BN.DEPTH; BN.DEPTH = float(self.spec['MAP_DEPTH']); need_grid = True
            self._undo.append(lambda: setattr(BN, 'DEPTH', old))
        if need_grid:
            og = (BN.KX_G, BN.KY_G, BN.OM_G)
            BN.KX_G, BN.KY_G, BN.OM_G = BN.kgrid()   # ★재대입만으로는 0회 발화한다
            def _rg(og=og):
                BN.KX_G, BN.KY_G, BN.OM_G = og
            self._undo.append(_rg)
        if 'K_LIM' in self.spec:
            orig, newk = LI._load_cond, float(self.spec['K_LIM'])
            def wrapped(ckpt, device, _o=orig, _k=newk):
                cd, model, stats, abar, betas, den = _o(ckpt, device)
                st = dict(stats); st['K_LIM'] = _k    # ⛔캐시 원본을 변형하지 않는다
                return (cd, model, st, abar, betas, den)
            LI._load_cond = wrapped
            self._undo.append(lambda: setattr(LI, '_load_cond', orig))
        if 'MIN_DK' in self.spec:
            old = INIT.MIN_DK; INIT.MIN_DK = float(self.spec['MIN_DK'])
            self._undo.append(lambda: setattr(INIT, 'MIN_DK', old))
        if 'grow_krange' in self.spec:
            orig, kr = INIT._beamform_coarse, float(self.spec['grow_krange'])
            def wc(x, y, z, t, depth, kx_range=None, ky_range=None,
                   n_kx=81, n_ky=81, _o=orig, _kr=kr):
                return _o(x, y, z, t, depth, kx_range=(-_kr, _kr),
                          ky_range=(-_kr, _kr), n_kx=n_kx, n_ky=n_ky)
            wc._t12_kr = kr
            INIT._beamform_coarse = wc
            self._undo.append(lambda: setattr(INIT, '_beamform_coarse', orig))
        if 'vp_lr' in self.spec:
            orig, lr = VP.apply_varpro, float(self.spec['vp_lr'])
            def wv(updater, frames, _o=orig, _lr=lr, **kw):
                kw['lr'] = _lr                # ⛔모듈 상수 재대입은 안 먹는다(기본 인자 바인딩)
                return _o(updater, frames, **kw)
            wv._t12_lr = lr
            VP.apply_varpro = wv
            self._undo.append(lambda: setattr(VP, 'apply_varpro', orig))

        # ── 계측 (개입 아님 — 되읽기 계수기) ──────────────────────────────
        og = SG.SpectralGaussianModel.add_gaussians
        def wg(slf, n_new, *a, _o=og, _s=self, **kw):
            _s.grow_calls += 1; _s.grow_atoms += int(n_new)
            return _o(slf, n_new, *a, **kw)
        SG.SpectralGaussianModel.add_gaussians = wg
        self._undo.append(lambda: setattr(SG.SpectralGaussianModel, 'add_gaussians', og))

        ops = LI.pool_and_select
        def wp(kx, ky, en, *a, _o=ops, _s=self, **kw):
            _s.n_raw = int(len(kx))
            _s.kmed_raw = float(np.median(np.hypot(np.asarray(kx), np.asarray(ky)))) \
                if len(kx) else None
            out = _o(kx, ky, en, *a, **kw)
            try:                     # ★반환은 (kx, ky, en) **배열**이다 (learned_init.py:358)
                skx, sky = np.asarray(out[0]), np.asarray(out[1])
                _s.n_sel = int(len(skx))
                _s.kmed_sel = float(np.median(np.hypot(skx, sky))) if len(skx) else None
            except Exception:                                     # noqa: BLE001
                pass
            return out
        LI.pool_and_select = wp
        self._undo.append(lambda: setattr(LI, 'pool_and_select', ops))

        olc = LI._load_cond
        def wl(ckpt, device, _o=olc, _s=self):
            _s.device = str(device)
            return _o(ckpt, device)
        LI._load_cond = wl
        self._undo.append(lambda: setattr(LI, '_load_cond', olc))

        ogi = LI._gather_init_obs
        def wgi(frames, _o=ogi, _s=self):
            out = _o(frames)
            _s.n_gathered = 0 if out is None else int(len(out[0]))
            return out
        self.n_gathered = None
        LI._gather_init_obs = wgi
        self._undo.append(lambda: setattr(LI, '_gather_init_obs', ogi))
        return self

    def __exit__(self, *a):
        for f in reversed(self._undo):
            f()
        return False


def estimate(arm, x, y, tt, z, gx, gy, tq, target, s4, seed):
    """⛔config 를 매번 새로 만든다 (v1 BLOCKER-2: 원자 캐시 누출 차단)."""
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(depth=C.DEPTH, seed=seed, device='cpu', init_method='auto')
    cfg.update(arm.cfg_overrides())
    assert 'learned_atoms' not in cfg, '제안 캐시가 새 config 에 이미 있다 (BLOCKER-2)'
    upd = make_updater(cfg)
    fr = R11.build_frames(x, y, tt, z, C.GT_CONST)
    rb = R11.readback_frames(fr)
    kb = read_knobs(cfg, upd)
    info = upd.initialize(fr, x_range=(float(gx.min()), float(gx.max())),
                          y_range=(float(gy.min()), float(gy.max())), verbose=False)
    gt_after = float(torch.cat([f.gt_field.reshape(-1) for f in fr]).std(unbiased=False))
    pred = upd.predict_field(torch.tensor(gx, dtype=torch.float32),
                             torch.tensor(gy, dtype=torch.float32), float(tq))
    pred = np.asarray(pred.detach().cpu()).reshape(target.shape)
    m = getattr(upd, 'model', None)
    kmed_final = None
    if m is not None and getattr(m, 'kx', None) is not None:
        kx = np.asarray(m.kx.detach().cpu()); ky = np.asarray(m.ky.detach().cpu())
        kmed_final = float(np.median(np.hypot(kx, ky))) if len(kx) else None
    la = cfg.get('learned_atoms')
    pmd5 = None
    if la:
        arr = np.c_[np.asarray(la['kx'], float).ravel(), np.asarray(la['ky'], float).ravel(),
                    np.asarray(la['energy'], float).ravel()]
        pmd5 = hashlib.md5(np.round(arr, 9).tobytes()).hexdigest()
    vp = (info or {}).get('varpro') or {}
    rb.update(knobs_before=kb, gt_std_after=gt_after, proposal_md5=pmd5,
              proposal_device=arm.device, n_gathered=arm.n_gathered,
              cap_proposal=int(LI.M_OBS_CAP), n_raw=arm.n_raw, n_sel=arm.n_sel,
              kmed_raw=arm.kmed_raw, kmed_sel=arm.kmed_sel, kmed_final=kmed_final,
              n_grow_calls=arm.grow_calls, n_grow_atoms=arm.grow_atoms,
              vp_lr=vp.get('vp_lr'), vp_k_shift=vp.get('vp_k_shift'),
              vp_clamp_hits=vp.get('vp_clamp_hits'), prov=upd.arm_provenance(),
              n_atoms_installed=int(getattr(m, 'n_gaussians', 0) or 0))
    return R11.nrmse(pred, target, s4), rb


def main(files, layer, arms, out, meta_out, limit=None):
    torch.set_num_threads(1)
    md5_start = C.freeze_map()
    rows = []
    files = files[:(limit or None)]
    with zipfile.ZipFile(C.ZIP) as zf:
        for fi, name in enumerate(files, 1):
            t0 = time.time()
            E, gx, gy, tarr = R11.read_mat(zf, name)
            nf = int(C.WIN_S)
            Ew, tw = E[C.J0:C.J0 + nf], tarr[C.J0:C.J0 + nf]
            target = Ew[-1]; s4 = 4.0 * float(np.std(Ew))
            ang, hs = R11.parse(name)
            seed = C.SEED + fi
            rng = np.random.default_rng(seed)
            x, y, tt, zz = R11.sample_obs(Ew, gx, gy, tw, C.DENSITY, rng)
            tq = float(tw[-1] - tw[0])
            F = np.fft.fft2(target); Pw = np.abs(F) ** 2; Pw[0, 0] = 0
            i0, j0_ = np.unravel_index(np.argmax(Pw), Pw.shape)
            kxa = 2 * np.pi * np.fft.fftfreq(target.shape[1], gx[1] - gx[0])
            kya = 2 * np.pi * np.fft.fftfreq(target.shape[0], gy[1] - gy[0])
            kp_true = float(np.hypot(kxa[j0_], kya[i0]))
            base = dict(layer=layer, block=name, angle=ang, hs=hs, sigma4=s4,
                        frame_index=C.J0 + nf - 1, seed=seed, n_frames=nf,
                        density=C.DENSITY, dt=float(np.median(np.diff(tarr))), tq=tq,
                        e_shape=[int(v) for v in E.shape], kp_true=kp_true)
            dep_knobs = read_knobs()
            none_rb = dict(n_gathered=None, cap_proposal=int(LI.M_OBS_CAP),
                           t_origin=float(tt.min()), gt_std=None, gt_value=None,
                           gt_std_after=None, knobs_before=dep_knobs,
                           proposal_md5=None, proposal_device=None, n_raw=None, n_sel=None,
                           kmed_raw=None, kmed_sel=None, kmed_final=None,
                           n_grow_calls=None, n_grow_atoms=None, vp_lr=None,
                           vp_k_shift=None, vp_clamp_hits=None, prov={},
                           n_atoms_installed=0, rescale_s=None, x_scale=None,
                           t_scale=None, disp_relerr=None)
            got = {}
            for arm_name in arms:
                r0 = rng_md5(); ta = time.time()
                if arm_name == 'ZERO':
                    val, rb = R11.nrmse(np.zeros_like(target), target, s4), dict(none_rb)
                    rb['n_offered'] = int(len(zz))
                elif arm_name == 'ORACLE':
                    val, rb = R11.oracle(x, y, tt, zz, gx, gy, tq, target, s4,
                                         C.N_MODE), dict(none_rb)
                    rb['n_offered'] = int(len(zz))
                else:
                    resc = arm_name == 'RESCALE'
                    s = C.RESCALE_S
                    ax, ay, at = (x / s, y / s, tt / np.sqrt(s)) if resc else (x, y, tt)
                    agx, agy = (gx / s, gy / s) if resc else (gx, gy)
                    atq = tq / np.sqrt(s) if resc else tq
                    with Arm(arm_name) as arm:
                        val, rb = estimate(arm, ax, ay, at, zz, agx, agy, atq,
                                           target, s4, seed)
                    a_, b_, rel = R11.measured_scales(ax, x, at, tt)
                    rb.update(rescale_s=(s if resc else None), x_scale=a_,
                              t_scale=b_, disp_relerr=rel)
                rb['knobs_after'] = read_knobs()
                km = rb.get('kmed_sel')
                rows.append(dict(base, arm=arm_name, nrmse=val, **rb,
                                 ratio=(km / kp_true if km and kp_true else None),
                                 rng_md5_before=r0, rng_md5_after=rng_md5(),
                                 wall_s=round(time.time() - ta, 1)))
                got[arm_name] = val
            print(f"  [{fi:2d}/{len(files)}] {os.path.basename(name):18s} "
                  + ' · '.join(f'{a.split("-")[-1][:7]} {got[a]:.4f}' for a in arms),
                  f'({time.time()-t0:.0f}s)', flush=True)
            with open(out, 'w', encoding='utf-8') as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    md5_end = C.freeze_map()
    json.dump(dict(layer=layer, n_runs=sum(1 for r in rows
                                           if r['arm'] not in ('ZERO', 'ORACLE')),
                   threads=int(torch.get_num_threads()),
                   threads_readback=int(torch.get_num_threads()),
                   blas_env={v: os.environ.get(v) for v in
                             ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
                   seed=C.SEED, code_md5=md5_end, code_md5_start=md5_start,
                   md5_stable=(md5_start == md5_end), propid_blocking=True,
                   cuda_available=bool(torch.cuda.is_available()),
                   files=len(files), arms=list(arms)),
              open(meta_out, 'w'), ensure_ascii=False, indent=1)
    print(f'✅{len(rows)} 행 · md5 안정 {md5_start == md5_end}')
    return rows


if __name__ == '__main__':
    a = sys.argv[1:]
    mode = a[0] if a else 'primary'
    lim = int(a[1]) if len(a) > 1 else None
    if mode == 'smoke':
        print(f'⚠SMOKE — 노출된 감사 파일 {len(C.SMOKE_FILES)}개, 프리즌 표본(24·28) 미접촉')
        main(list(C.SMOKE_FILES), 'SMOKE', C.ARMS,
             OUT.replace('.jsonl', '_SMOKE.jsonl'), META.replace('.json', '_SMOKE.json'), lim)
    elif mode == 'secr':
        main(C.heldout_blocks(), 'SECR', C.SECR_ARMS,
             OUT.replace('.jsonl', '_SECR.jsonl'), META.replace('.json', '_SECR.json'), lim)
    else:
        main(C.sample(), 'PRIMARY', C.ARMS, OUT, META, lim)
