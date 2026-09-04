"""`T14-AUTOSCALE` 러너 — ⛔`gates_t14`·`analyze_t14` **다음에** 쓰였다.
★관측·프레임·오라클은 `T11 run_t11` 을 임포트해 **비트 재사용**(앵커 전제)."""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import hashlib, json, subprocess, sys, tempfile, time, zipfile      # noqa: E402

import numpy as np                                                 # noqa: E402
import torch                                                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_t14 as G                                              # noqa: E402
import scale_rules_t14 as SR                                       # noqa: E402
sys.path.insert(0, G.T11)
import run_t11 as R11                                              # noqa: E402
# ⛔cwd 를 되돌리지 마라 — d3_beamform_cond_diffusion.py:25 의 chdir 은 의존성이다
#   (:40 PRIOR_CKPT 가 상대 경로). T13 AMENDMENT_1 §1 이 그 대가를 기록했다.
from wavesplat.optimizer import learned_init as LI                 # noqa: E402
from wavesplat.optimizer.sequential_update import make_updater     # noqa: E402
from wavesplat.model.tnse import render_wave_field                 # noqa: E402
from wavesplat.run import DEFAULT_CONFIG                           # noqa: E402

OUT = os.path.join(HERE, 'results_t14.jsonl'); META = os.path.join(HERE, 'meta_t14.json')
IMPORTS = ('wavesplat/optimizer/sequential_update.py', 'wavesplat/optimizer/learned_init.py',
           'wavesplat/optimizer/bank_mmse.py', 'wavesplat/optimizer/varpro.py',
           'wavesplat/model/spectral_gaussian.py', 'wavesplat/model/tnse.py',
           'wavesplat/model/dispersion.py', 'wavesplat/run.py',
           'scripts/d3_beamform_net.py', 'scripts/d3_beamform_cond_diffusion.py')
CKPT = 'results/direction3/beamform_cond_diffusion_mixed_v1.pt'


def _md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def freeze_map():
    out = {}
    for f in ('gates_t14.py', 'analyze_t14.py', 'run_t14.py', 'scale_rules_t14.py',
              'blind_worker_t14.py', 'PREREG.md', 'SCALE_ESTIMATORS.md'):
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            out[f] = _md5(p)
    for f in IMPORTS:
        p = os.path.join(G.ROOT, f)
        if os.path.exists(p):
            out[f] = _md5(p)
    for f in ('sample_t11.json', 'results_t11_SMOKE.jsonl', 'gates_t11.py', 'run_t11.py'):
        p = os.path.join(G.T11, f)
        if os.path.exists(p):
            out['T11/' + f] = _md5(p)
    h = hashlib.sha256()
    with open(os.path.join(G.ROOT, CKPT), 'rb') as fh:
        for c in iter(lambda: fh.read(1 << 20), b''):
            h.update(c)
    out['sha256:' + CKPT] = h.hexdigest()
    return out


def blind_s(x, y, t, z, depth):
    """★별도 프로세스에서 규칙을 재실행 — 관측 5배열만 넘긴다."""
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        np.savez(f, x=x, y=y, t=t, z=z, depth=np.float64(depth)); p = f.name
    try:
        r = subprocess.run([sys.executable,
                            os.path.join(HERE, 'blind_worker_t14.py'), p],
                           capture_output=True, text=True, timeout=600)
        return json.loads(r.stdout.strip().splitlines()[-1])
    finally:
        os.unlink(p)


class Probe:
    def __enter__(self):
        self.device = None; self.n_gathered = None
        ol, og = LI._load_cond, LI._gather_init_obs

        def wl(c, d, _o=ol, _s=self):
            _s.device = str(d); return _o(c, d)

        def wg(fr, _o=og, _s=self):
            o = _o(fr); _s.n_gathered = 0 if o is None else int(len(o[0])); return o
        LI._load_cond, LI._gather_init_obs = wl, wg
        self._u = [lambda: setattr(LI, '_load_cond', ol),
                   lambda: setattr(LI, '_gather_init_obs', og)]
        return self

    def __exit__(self, *a):
        for f in self._u:
            f()
        return False


def estimate(s, x, y, tt, z, gx, gy, tq, target, s4, seed):
    """재라벨 `x/s, y/s, t/√s` 로 추정하고 **두 역변환 경로를 다 계산**한다(`G-INVERT`)."""
    rs = float(np.sqrt(s))
    ax, ay, at = x / s, y / s, tt / rs
    agx, agy, atq = gx / s, gy / s, tq / rs
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(depth=G.DEPTH, seed=seed, device='cpu', init_method='auto')
    assert 'learned_atoms' not in cfg
    upd = make_updater(cfg)
    fr = R11.build_frames(ax, ay, at, z, G.GT_CONST)
    rb = R11.readback_frames(fr)
    with Probe() as pr:
        info = upd.initialize(fr, x_range=(float(agx.min()), float(agx.max())),
                              y_range=(float(agy.min()), float(agy.max())), verbose=False)
    gt_after = float(torch.cat([f.gt_field.reshape(-1) for f in fr]).std(unbiased=False))
    # ── P-implicit (T11 경로: 원 격자의 상에서 질의) ──
    pi = upd.predict_field(torch.tensor(agx, dtype=torch.float32),
                           torch.tensor(agy, dtype=torch.float32), float(atq))
    pi = np.asarray(pi.detach().cpu()).reshape(target.shape)
    # ── ★P-explicit (배포 가능 경로: 원자를 원 단위로 되돌려 원 격자에 렌더) ──
    m = getattr(upd, 'model', None)
    pe, st2, nnc = None, None, None
    if m is not None and getattr(m, 'kx', None) is not None:
        kxp = np.asarray(m.kx.detach().cpu(), float) / s
        kyp = np.asarray(m.ky.detach().cpu(), float) / s
        Aa = np.asarray(m.w.detach().cpu(), float)
        ph = np.asarray(m.phi.detach().cpu(), float)
        pe = np.asarray(render_wave_field(kxp, kyp, Aa, ph, G.DEPTH, gx, gy, float(tq)))
        st2 = bool(getattr(m, 'use_stokes2', False))
        nnc = bool(getattr(m, 'use_neural_corrector', False))
    vp = (info or {}).get('varpro') or {}
    la = cfg.get('learned_atoms'); pmd5 = None
    if la:
        arr = np.c_[np.asarray(la['kx'], float).ravel(), np.asarray(la['ky'], float).ravel(),
                    np.asarray(la['energy'], float).ravel()]
        pmd5 = hashlib.md5(np.round(arr, 9).tobytes()).hexdigest()
    a_, b_, rel = R11.measured_scales(ax, x, at, tt)
    rb.update(x_scale=a_, t_scale=b_, disp_relerr=rel, proposal_md5=pmd5,
              atoms_md5=(hashlib.md5(np.round(np.c_[np.asarray(m.kx.detach().cpu()),
                                                    np.asarray(m.ky.detach().cpu())], 6).tobytes()
                                     ).hexdigest() if m is not None
                         and getattr(m, 'kx', None) is not None else None),
              proposal_device=pr.device, n_gathered=pr.n_gathered,
              cap_proposal=int(LI.M_OBS_CAP), gt_std_after=gt_after,
              vp_clamp_hits=vp.get('vp_clamp_hits'), prov=upd.arm_provenance(),
              n_atoms_installed=int(getattr(m, 'n_gaussians', 0) or 0),
              use_stokes2=st2, use_neural_corrector=nnc,
              invert_maxdiff=(float(np.max(np.abs(pi - pe))) if pe is not None else None),
              nrmse_explicit=(R11.nrmse(pe, target, s4) if pe is not None else None),
              deploy_consts={k: cfg.get(k) for k in
                             ('lambda_l1', 'varpro_steps', 'learned_init_iters',
                              'auto_switch_thresh', 'init_method', 'tracking_mode',
                              'analytic_ensemble_k', 'learned_ensemble_k')})
    return R11.nrmse(pi, target, s4), rb


def main(mode='primary'):
    torch.set_num_threads(1)
    md5_start = freeze_map()
    t1, t2, t3, anc = G.tiers()
    plan = ([(b, 'ANCHOR', G.ANCHOR_ARMS) for b in anc] if mode == 'smoke'
            else [(b, 'T1', G.T1_ARMS) for b in t1]
            + [(b, 'T2', G.T2_ARMS) for b in t2]
            + [(b, 'T3', G.T1_ARMS) for b in t3]
            + [(b, 'ANCHOR', G.ANCHOR_ARMS) for b in anc])
    if mode == 'smoke':
        plan = [(b, 'ANCHOR', G.T1_ARMS + G.T2_ARMS) for b in anc]
        print(f'⚠SMOKE — 앵커 2파일(사전노출) · 프리즌 표본 34 미접촉')
    out = OUT.replace('.jsonl', '_SMOKE.jsonl') if mode == 'smoke' else OUT
    meta_o = META.replace('.json', '_SMOKE.json') if mode == 'smoke' else META
    rows, cache = [], {}
    with zipfile.ZipFile(G.ZIP) as zf:
        for fi, (name, tier, arms) in enumerate(plan, 1):
            t0 = time.time()
            if name not in cache:
                E, gx, gy, tarr = R11.read_mat(zf, name)
                nf = int(G.WIN_S)
                Ew, tw = E[G.J0:G.J0 + nf], tarr[G.J0:G.J0 + nf]
                tgt = Ew[-1]; s4 = 4.0 * float(np.std(Ew))
                sd = G.seed_of(name, tier)
                x, y, tt, zz = R11.sample_obs(Ew, gx, gy, tw, G.DENSITY,
                                              np.random.default_rng(sd))
                tq = float(tw[-1] - tw[0])
                kp1 = None
                from wavesplat.optimizer.bank_mmse import _beamform_top
                P = np.column_stack([x, y, tt]).astype(np.float32)
                bx, by = _beamform_top(P, zz.astype(np.float32), G.DEPTH, klim=2.0, n=120)
                kp1 = max(float(np.hypot(bx, by)), 1e-3)
                kph = SR.kp_hat(x, y, tt, zz, G.DEPTH)
                srule = {a: SR.RULES[a](x, y, tt, zz, G.DEPTH) for a in G.AUTO_ARMS}
                sbl = blind_s(x, y, tt, zz, G.DEPTH)
                # ★등변성은 **비클리핑** 규칙으로 잰다 — 레일이 등변성을 깬다(AMENDMENT_1 §2)
                sraw = {a: SR.RULES_RAW[a](x, y, tt, zz, G.DEPTH) for a in G.AUTO_ARMS}
                sc05 = {a: SR.RULES_RAW[a](x / 0.5, y / 0.5, tt / np.sqrt(0.5), zz, G.DEPTH)
                        for a in G.AUTO_ARMS}
                sc20 = {a: SR.RULES_RAW[a](x / 2.0, y / 2.0, tt / np.sqrt(2.0), zz, G.DEPTH)
                        for a in G.AUTO_ARMS}
                cache[name] = (E, gx, gy, tarr, Ew, tw, tgt, s4, sd, x, y, tt, zz, tq,
                               kp1, kph, srule, sbl, sc05, sc20, sraw)
            (E, gx, gy, tarr, Ew, tw, tgt, s4, sd, x, y, tt, zz, tq,
             kp1, kph, srule, sbl, sc05, sc20, sraw) = cache[name]
            ang, hs = R11.parse(name)
            base = dict(block=name, tier=tier, angle=ang, hs=hs, sigma4=s4, seed=sd,
                        frame_index=G.J0 + int(G.WIN_S) - 1, n_frames=int(G.WIN_S),
                        density=G.DENSITY, dt=float(np.median(np.diff(tarr))), tq=tq,
                        e_shape=[int(v) for v in E.shape], mean_convention='raw',
                        axis_convention='E[j][iy, ix]', kp_hat=kph, kp_hat_stage1=kp1,
                        L_obs_x=float(np.ptp(x)), L_obs_y=float(np.ptp(y)),
                        L_TRAIN=SR.L_TRAIN, K_TRAIN=SR.K_TRAIN,
                        kd_min=float(kph * G.DEPTH))
            got = {}
            for a in arms:
                ta = time.time()
                if a == 'ZERO':
                    val = R11.nrmse(np.zeros_like(tgt), tgt, s4)
                    rb = dict(t_origin=float(tt.min()), n_offered=int(len(zz)),
                              gt_std=None, gt_value=None, gt_std_after=None,
                              proposal_md5=None, atoms_md5=None, proposal_device=None,
                              n_gathered=None, cap_proposal=None, vp_clamp_hits=None,
                              prov={}, n_atoms_installed=0, use_stokes2=None,
                              use_neural_corrector=None, invert_maxdiff=None,
                              nrmse_explicit=None, deploy_consts={},
                              x_scale=None, t_scale=None, disp_relerr=None)
                    s_used, sb, c05, c20, s_raw = None, None, None, None, None
                else:
                    s_used = srule[a] if a in G.AUTO_ARMS else G.S_FIXED[a]
                    sb = sbl.get(a) if a in G.AUTO_ARMS else None
                    c05 = sc05.get(a) if a in G.AUTO_ARMS else None
                    c20 = sc20.get(a) if a in G.AUTO_ARMS else None
                    s_raw = sraw.get(a) if a in G.AUTO_ARMS else None
                    val, rb = estimate(s_used, x, y, tt, zz, gx, gy, tq, tgt, s4, sd)
                rows.append(dict(base, arm=a, nrmse=val, s_rule=s_used, s_blind=sb,
                                 s_raw=s_raw, s_c05=c05, s_c20=c20,
                                 s_clipped=(s_used is not None
                                            and (abs(s_used - SR.S_MIN) < 1e-12
                                                 or abs(s_used - SR.S_MAX) < 1e-12)),
                                 s_readback=(None if rb.get('x_scale') in (None, 0)
                                             else 1.0 / rb['x_scale']),
                                 wall_s=round(time.time() - ta, 1), **rb))
                got[a] = val
            print(f"  [{fi:2d}/{len(plan)}] {tier:6s} {os.path.basename(name):18s} "
                  f"s(ap {srule['AUTO-AP']:.3f} band {srule['AUTO-BAND']:.3f}) "
                  + ' · '.join(f'{k[:8]} {v:.4f}' for k, v in got.items())
                  + f' ({time.time()-t0:.0f}s)', flush=True)
            with open(out, 'w', encoding='utf-8') as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    md5_end = freeze_map()
    json.dump(dict(mode=mode, n_runs=sum(1 for r in rows if r['arm'] != 'ZERO'),
                   threads=int(torch.get_num_threads()),
                   threads_readback=int(torch.get_num_threads()),
                   blas_env={v: os.environ.get(v) for v in
                             ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
                   seed14=G.SEED14, code_md5=md5_end, code_md5_start=md5_start,
                   code_md5_now=md5_end, md5_stable=(md5_start == md5_end),
                   cuda_available=bool(torch.cuda.is_available()), files=len(cache)),
              open(meta_o, 'w'), ensure_ascii=False, indent=1)
    print(f'✅{len(rows)} 행 · md5 안정 {md5_start == md5_end}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'primary')
