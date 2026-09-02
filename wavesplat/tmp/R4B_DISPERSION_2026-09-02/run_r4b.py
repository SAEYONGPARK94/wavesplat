"""`R4b-DISPERSION` 러너 — ⛔**추정기 런 0회**.

관측 `sheetfrac` 은 `R4` 와 **같은 계산**이고, 여기에 **분산곡선을 옮긴 귀무**를 더한다.
⛔IAAFT 는 쓰지 않는다 — `sheetfrac` 이 `|FFT|²` 만의 함수라 위상무작위화는 검정력 0 이다
(`PREREG §1`).
"""
import glob, hashlib, inspect, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for p in (HERE, ROOT, os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23')):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(ROOT)
import gates_r4b as G                                                # noqa: E402
import scene_station as SS                                           # noqa: E402
from d3_sa_multirecord import MRecord                                # noqa: E402
from agulhas_record import AgulhasRecord                             # noqa: E402

OUT = os.path.join(HERE, 'results_r4b.jsonl')
AGDIR = os.path.join(ROOT, 'data', 'agulhas_wass_2017')
R4_ROWS = os.path.join(ROOT, 'tmp', 'R4_DISPERSION_2026-09-02', 'results_r4.jsonl')
_MD5 = [os.path.join(HERE, x) for x in ('gates_r4b.py', 'wk_common.py', 'agulhas_record.py')]


def code_md5():
    return {os.path.basename(p): hashlib.md5(open(p, 'rb').read()).hexdigest()[:12] for p in _MD5}


def sheet_ext(Z, dx, dy, dt, rng):
    """링별 `sheetfrac` 관측 + **분산곡선을 옮긴** 귀무 `B` 회.

    ★귀무는 자료를 건드리지 않는다 — **같은 계수기하(±3 빈, ±ω 대칭)를 틀린 위치**에서
    적용한다 ⟹ 상자·유효율·표본율·대역폭이 관측과 귀무에 **똑같이** 들어가 상쇄된다.
    """
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

    # 옮김량은 참 밴드(±3)와 겹치지 않게 |s| ≥ 8, 그리고 레코드마다 **같은 draw** 를 쓴다
    hi = max(G.NULL_MIN_SHIFT + 1, nf // 2)
    mag = rng.integers(G.NULL_MIN_SHIFT, hi, size=G.N_NULL)
    shifts = mag * rng.choice([-1, 1], size=G.N_NULL)

    obs, nulls = [], []
    for kr in rings:
        m = (K > kr * 0.85) & (K <= kr * 1.18)
        if m.sum() < 8:
            continue
        # ⚠`tot` 의 **합산 순서**는 `R4` 와 한 글자도 달라선 안 된다 —
        #   `Pm.sum()` 으로 바꿨더니 앵커가 즉시 깨졌다(부동소수점 결합순서).
        Pm = P[:, m]; spec = Pm.sum(axis=1); tot = spec.sum()
        if tot <= 0:
            continue
        kb = K[m]; jl = np.round(np.sqrt(9.81 * kb) / dom).astype(int)
        idx = np.arange(m.sum())

        def frac(s):
            e = 0.0
            for d in range(-3, 4):
                e += Pm[(jl + s + d) % nf, idx].sum() + Pm[(-(jl + s) + d) % nf, idx].sum()
            return float(e / tot)

        obs.append(frac(0))
        nulls.append([frac(int(s)) for s in shifts])
    return obs, np.array(nulls), float(kfund), float(knyq), nf


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


def r4_anchor():
    """★선행 배치의 **동결 행**. ⛔손복사 수치가 아니다."""
    out = {}
    for l in open(R4_ROWS):
        if l.strip():
            r = json.loads(l)
            if 'sheetfrac' in r:
                out[(r['group'], r['record'])] = r['sheetfrac']
    return out


def run(group, key, anchor):
    t0 = time.time()
    got = load(group, key)
    if got is None:
        return dict(group=group, record=key, error='자료 없음')
    Z, dx, dy, dt = got
    vf = float(np.isfinite(Z).mean())
    if vf < G.MIN_VALID_FRAC:
        return dict(group=group, record=key, valid_frac=vf, void=True,
                    note=f'유효율 {vf:.3f} < {G.MIN_VALID_FRAC} — 결과 전 규칙으로 VOID')
    rng = np.random.default_rng(G.NULL_SEED)
    obs, nl, kf, kn, nf = sheet_ext(Z, dx, dy, dt, rng)
    om = float(np.mean(obs)) if obs else float('nan')
    nm = nl.mean(axis=0) if len(nl) else np.array([np.nan])
    thr = float(np.percentile(nm, G.PCTL))
    a = anchor.get((group, key))
    return dict(
        group=group, record=key, nt=int(Z.shape[0]), n_rings=len(obs),
        sheetfrac=obs, sheetfrac_mean=om,
        null_p95=thr, null_median=float(np.median(nm)),
        null_min=float(nm.min()), null_max=float(nm.max()),
        passes=bool(om > thr), ratio_obs_null=(om / float(np.median(nm))),
        chance=14.0 / nf, n_null=G.N_NULL, null_seed=G.NULL_SEED,
        valid_frac=vf, dx=dx, dy=dy, dt=dt, k_fund=kf, k_nyq=kn,
        anchor_r4=(None if a is None else bool(np.allclose(obs, a, rtol=0, atol=0))),
        sheet_md5=hashlib.md5(inspect.getsource(sheet_ext).encode()).hexdigest()[:12],
        code_md5=code_md5(), pid=os.getpid(), wall_s=round(time.time() - t0, 1))


def main():
    anchor = r4_anchor()
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
            r = run(grp, key, anchor)
            fh.write(json.dumps(r, ensure_ascii=False) + '\n'); fh.flush()
            print(f"  {grp:8s} {key:6s} obs={r.get('sheetfrac_mean', float('nan')):.4f} "
                  f"귀무p95={r.get('null_p95', float('nan')):.4f} "
                  f"{'통과' if r.get('passes') else '  — '} "
                  f"비{r.get('ratio_obs_null', float('nan')):.1f}× "
                  f"앵커{'OK' if r.get('anchor_r4') else 'X'} {r.get('wall_s', '-')}s", flush=True)
    print('✅ 완료'); return 0


if __name__ == '__main__':
    sys.exit(main())
