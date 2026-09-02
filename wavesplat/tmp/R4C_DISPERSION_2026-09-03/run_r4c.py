"""`R4c-DISPERSION` 러너 — ⛔**추정기 런 0회**.

`R4b` 와 **한 곳만** 다르다: 두 ±3 밴드를 **집합 합집합**으로 세어 각 빈을 한 번만 계수한다.
⟹ `2(jl+s) ≡ 0 (mod nf)` 부근에서 일어나던 **이중 계수**가 사라진다.
관측(`s=0`)은 `2·jl` 이 0 에서 멀어 영향받지 않으며, `G-OBS` 가 그것을 자료로 증명한다.
"""
import glob, hashlib, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for p in (HERE, ROOT, os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23')):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(ROOT)
import gates_r4c as G                                                # noqa: E402
import scene_station as SS                                           # noqa: E402
from d3_sa_multirecord import MRecord                                # noqa: E402
from agulhas_record import AgulhasRecord                             # noqa: E402

OUT = os.path.join(HERE, 'results_r4c.jsonl')
AGDIR = os.path.join(ROOT, 'data', 'agulhas_wass_2017')
_MD5 = [os.path.join(HERE, x) for x in ('gates_r4c.py', 'agulhas_record.py')]


def code_md5():
    return {os.path.basename(p): hashlib.md5(open(p, 'rb').read()).hexdigest()[:12] for p in _MD5}


def sheet_dedup(Z, dx, dy, dt, rng):
    """관측 + 귀무 `B` 회. ★두 밴드를 **합집합**으로 세어 이중 계수를 없앤다."""
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

    hi = max(G.NULL_MIN_SHIFT + 1, nf // 2)
    mag = rng.integers(G.NULL_MIN_SHIFT, hi, size=G.N_NULL)
    shifts = mag * rng.choice([-1, 1], size=G.N_NULL)     # ⛔`R4b` 와 같은 draw

    obs, nulls, ovl = [], [], []
    for kr in rings:
        m = (K > kr * 0.85) & (K <= kr * 1.18)
        if m.sum() < 8:
            continue
        Pm = P[:, m]; spec = Pm.sum(axis=1); tot = spec.sum()   # ⛔합산 순서 보존
        if tot <= 0:
            continue
        kb = K[m]; jl = np.round(np.sqrt(9.81 * kb) / dom).astype(int)
        idx = np.arange(m.sum())

        def frac(s):
            """★`R4b` 의 **산술을 한 글자도 바꾸지 않고**, 이중 계수된 몫만 뺀다.

            ⛔합집합으로 다시 짜면 합산 순서가 달라져 관측이 비트 단위로 흔들린다
            (동결 전 스모크가 즉시 잡았다 — `R4b` 에서도 같은 부류가 있었다).
            중복 조건: `(c+d1) ≡ (-c+d2) (mod nf)` ⟺ `2c ≡ d2-d1`, `|d2-d1| ≤ 2·BAND`.
            """
            c = jl + s
            e = 0.0
            for d in range(-G.BAND, G.BAND + 1):
                e += Pm[(c + d) % nf, idx].sum() + Pm[(-c + d) % nf, idx].sum()
            # 이중 계수분 차감 — 겹치는 픽셀이 없으면 정확히 0.0 이 되어 비트 동일하다.
            dup = 0.0
            w = (2 * c) % nf
            for e_ in range(-2 * G.BAND, 2 * G.BAND + 1):
                hit = (w == (e_ % nf))
                if not hit.any():
                    continue
                # d2 - d1 = e_ 인 (d1,d2) 쌍의 수만큼 그 빈이 중복된다
                for d1 in range(-G.BAND, G.BAND + 1):
                    d2 = d1 + e_
                    if -G.BAND <= d2 <= G.BAND:
                        dup += Pm[(c[hit] + d1) % nf, idx[hit]].sum()
            return float((e - dup) / tot)

        def overlaps(s):
            w = (2 * (jl + s)) % nf
            return bool(np.any(np.minimum(w, nf - w) <= G.OVERLAP_TOL))

        obs.append(frac(0))
        nulls.append([frac(int(s)) for s in shifts])
        ovl.append(np.mean([overlaps(int(s)) for s in shifts]))
    return obs, np.array(nulls), float(np.mean(ovl)) if ovl else float('nan'), float(kfund), nf


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


def r4b_rows():
    out = {}
    for l in open(G.R4B_ROWS):
        if l.strip():
            r = json.loads(l)
            if 'sheetfrac' in r:
                out[(r['group'], r['record'])] = r
    return out


def run(group, key, prev):
    t0 = time.time()
    got = load(group, key)
    if got is None:
        return dict(group=group, record=key, error='자료 없음')
    Z, dx, dy, dt = got
    vf = float(np.isfinite(Z).mean())
    if vf < G.MIN_VALID_FRAC:
        return dict(group=group, record=key, valid_frac=vf, void=True)
    rng = np.random.default_rng(G.NULL_SEED)
    obs, nl, ovl, kf, nf = sheet_dedup(Z, dx, dy, dt, rng)
    om = float(np.mean(obs)) if obs else float('nan')
    nm = nl.mean(axis=0) if len(nl) else np.array([np.nan])
    thr = float(np.percentile(nm, G.PCTL))
    p = prev.get((group, key), {})
    return dict(
        group=group, record=key, nt=int(Z.shape[0]), n_rings=len(obs),
        sheetfrac=obs, sheetfrac_mean=om,
        null_p95=thr, null_median=float(np.median(nm)),
        null_draws=[float(x) for x in nm],          # ★G-DRAWS — 전부 저장
        passes=bool(om > thr),
        null_p95_r4b=p.get('null_p95'), null_median_r4b=p.get('null_median'),
        obs_identical=(None if 'sheetfrac' not in p
                       else bool(np.allclose(obs, p['sheetfrac'], rtol=0, atol=0))),
        overlap_frac=ovl, chance=14.0 / nf, dt=dt,
        n_null=G.N_NULL, null_seed=G.NULL_SEED, pctl=G.PCTL,
        valid_frac=vf, code_md5=code_md5(), pid=os.getpid(),
        wall_s=round(time.time() - t0, 1))


def main():
    prev = r4b_rows()
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            if l.strip():
                r = json.loads(l); done.add((r['group'], r['record']))
    with open(OUT, 'a') as fh:
        for grp, key in ([('control', k) for k in G.CONTROL]
                         + [('agulhas', k) for k in G.AGULHAS]):
            if (grp, key) in done:
                continue
            r = run(grp, key, prev)
            fh.write(json.dumps(r, ensure_ascii=False) + '\n'); fh.flush()
            print(f"  {grp:8s} {key:6s} obs={r.get('sheetfrac_mean', float('nan')):.4f} "
                  f"p95 {r.get('null_p95_r4b', float('nan')):.4f}→{r.get('null_p95', float('nan')):.4f} "
                  f"{'통과' if r.get('passes') else '  — '} "
                  f"중첩{r.get('overlap_frac', float('nan')):.3f} "
                  f"앵커{'OK' if r.get('obs_identical') else 'X'} {r.get('wall_s', '-')}s",
                  flush=True)
    print('✅ 완료'); return 0


if __name__ == '__main__':
    sys.exit(main())
