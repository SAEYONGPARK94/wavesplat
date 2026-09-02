"""`R4-DISPERSION` 러너 — ⛔**추정기 런 0회**. 두 코퍼스를 **같은 자**로 잰다."""
import glob, hashlib, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for p in (HERE, ROOT, os.path.join(ROOT, 'scripts'),
          os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23')):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(ROOT)
import gates_r4 as G                                                # noqa: E402
import scene_station as SS                                          # noqa: E402
from d3_sa_multirecord import MRecord                               # noqa: E402
from agulhas_record import AgulhasRecord                            # noqa: E402

OUT = os.path.join(HERE, 'results_r4.jsonl')
AGDIR = os.path.join(ROOT, 'data', 'agulhas_wass_2017')
METHOD = 'wk_sheetfrac_v1'      # G-SAME 이 이 문자열의 유일성을 검사한다
_MD5 = [os.path.join(HERE, x) for x in ('gates_r4.py', 'wk_common.py', 'agulhas_record.py')]


def code_md5():
    return {os.path.basename(p): hashlib.md5(open(p, 'rb').read()).hexdigest()[:12] for p in _MD5}


def sheet(Z, dx, dy, dt):
    """링별 **분산면 ±3빈** 에너지 비율. `wk_common.wk_report` 와 같은 규약."""
    mu = np.nanmean(Z, axis=0)
    Zf = np.nan_to_num(np.where(np.isfinite(Z), Z, mu[None]) - mu[None])
    nf, ny, nx = Zf.shape
    W = (np.hanning(nf)[:, None, None] * np.hanning(ny)[None, :, None]
         * np.hanning(nx)[None, None, :])
    P = np.abs(np.fft.fftn(Zf * W)) ** 2
    om = 2 * np.pi * np.fft.fftfreq(nf, dt); dom = om[1] - om[0]
    ky = 2 * np.pi * np.fft.fftfreq(ny, dy); kx = 2 * np.pi * np.fft.fftfreq(nx, dx)
    KX, KY = np.meshgrid(kx, ky); K = np.hypot(KX, KY)
    kfund = max(abs(kx[1]), abs(ky[1])); knyq = np.pi / max(dx, dy)
    rings = np.exp(np.linspace(np.log(kfund * 1.6), np.log(knyq * 0.5), 6))
    out = []
    for kr in rings:
        m = (K > kr * 0.85) & (K <= kr * 1.18)
        if m.sum() < 8:
            continue
        spec = P[:, m].sum(axis=1)
        kb = K[m]; jl = np.round(np.sqrt(9.81 * kb) / dom).astype(int)
        Pm = P[:, m]; idx = np.arange(m.sum()); e = 0.0
        for d in range(-3, 4):
            e += Pm[(jl + d) % nf, idx].sum() + Pm[(-jl + d) % nf, idx].sum()
        if spec.sum() > 0:
            out.append(float(e / spec.sum()))
    return out, float(kfund), float(knyq)


def load(group, key):
    if group == 'control':
        cfg = SS.RECORDS_12[key]
        r = MRecord(os.path.join(ROOT, cfg['path']), stride=cfg['stride'])
        i0 = max(0, r.T_all // 2 - G.NT // 2)
        Z = r.window(i0, G.NT)
        s = max(1, min(Z.shape[1], Z.shape[2]) // 48)
        return Z[:, ::s, ::s], r.dx * s, r.dy * s, r.dt
    p = glob.glob(os.path.join(AGDIR, f'wass__20170704_{key}.nc'))
    if not p:
        return None
    r = AgulhasRecord(p[0])
    i0 = max(0, r.T_all // 2 - G.NT // 2)
    Z = r.window(i0, G.NT)
    s = max(1, min(Z.shape[1], Z.shape[2]) // 48)
    return Z[:, ::s, ::s], r.dx * s, r.dy * s, r.dt


def run(group, key):
    t0 = time.time()
    got = load(group, key)
    if got is None:
        return dict(group=group, record=key, error='자료 없음')
    Z, dx, dy, dt = got
    vf = float(np.isfinite(Z).mean())
    if vf < G.MIN_VALID_FRAC:
        return dict(group=group, record=key, valid_frac=vf, void=True,
                    note=f'유효율 {vf:.3f} < {G.MIN_VALID_FRAC} — 결과 전 규칙으로 VOID')
    sf, kf, kn = sheet(Z, dx, dy, dt)
    sig = float(np.nanstd(Z - np.nanmean(Z, axis=0, keepdims=True)))
    return dict(group=group, record=key, nt=int(Z.shape[0]), method=METHOD,
                sheetfrac=sf, sheetfrac_mean=(float(np.mean(sf)) if sf else float('nan')),
                n_rings=len(sf), valid_frac=vf, sigma=sig, Hs=4 * sig,
                dx=dx, dy=dy, dt=dt, box=[Z.shape[2] * dx, Z.shape[1] * dy],
                k_fund=kf, k_nyq=kn, code_md5=code_md5(), pid=os.getpid(),
                wall_s=round(time.time() - t0, 1))


def main():
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            if l.strip():
                r = json.loads(l); done.add((r['group'], r['record']))
    jobs = [('control', k) for k in G.CONTROL] + [('agulhas', k) for k in G.AGULHAS]
    with open(OUT, 'a') as fh:
        for grp, key in jobs:
            if (grp, key) in done:
                continue
            r = run(grp, key)
            fh.write(json.dumps(r, ensure_ascii=False) + '\n'); fh.flush()
            sfm = r.get('sheetfrac_mean')
            print(f"  {grp:8s} {key:6s} sheetfrac={('%.3f' % sfm) if sfm==sfm else '--':>6s} "
                  f"링{r.get('n_rings','-')} 유효{r.get('valid_frac',0):.2f} "
                  f"{r.get('wall_s','-')}s", flush=True)
    print('✅ 완료'); return 0


if __name__ == '__main__':
    sys.exit(main())
