"""`E2-EXTSYNTH` 러너 — ⛔우리 코드·저들 코드 **둘 다 무수정**. 어댑터는 `E1` 에서 임포트.

⛔`external/tfenv/bin/python` 으로 실행한다.
"""
import glob, json, os, sys, time, warnings
warnings.filterwarnings('ignore')
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np, h5py                                             # noqa: E402
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_e2 as G                                                 # noqa: E402
sys.path.insert(0, G.E1)
import extarm_e1 as EX                                               # noqa: E402  ⛔단일출처
import run_e1 as R1                                                  # noqa: E402  (양성대조·kgrid)
OUT = os.path.join(HERE, 'results_e2.jsonl'); META = os.path.join(HERE, 'meta_e2.json')


def run(path, model):
    t0 = time.time()
    with h5py.File(path, 'r') as f:
        P = np.asarray(f['frames/lidar_points']); V = np.asarray(f['frames/lidar_valid_mask'])
        gx = np.asarray(f['frames/gt_grid_x']); gy = np.asarray(f['frames/gt_grid_y'])
        Z = np.asarray(f['frames/gt_wave_field']); t = np.asarray(f['frames/timestamps'])
    dt = float(np.median(np.diff(t)))
    j0 = Z.shape[0] // 2                                  # 창 중앙(결과 전 고정)
    XX, YY = np.meshgrid(gx, gy)
    g = Z[j0]; sg = float(np.std(Z))
    zero = float(np.sqrt(np.mean(g ** 2)) / sg)
    n = G.CNN_N
    xmin, xmax, ymin, ymax = gx.min(), gx.max(), gy.min(), gy.max()
    KX, KY = EX.kgrid(n, (xmax - xmin) / (n - 1), (ymax - ymin) / (n - 1))

    # ── 배포 밀도: 저장된 LiDAR 반환 **그대로**
    B = np.zeros((1, n, n, 3), np.float32); n_in = 0; held = []
    for c, jj in enumerate((j0 - 1, j0, j0 + 1)):
        m = V[jj]
        px, py, pz = P[jj, m, 0], P[jj, m, 1], P[jj, m, 2]
        if c == 1:                                        # 방향 선택용 20% 유보
            k = max(1, int(round(len(pz) * G.DIR_HOLDOUT)))
            sel = np.random.default_rng(G.SAMPLE_SEED).permutation(len(pz))
            held = (px[sel[:k]], py[sel[:k]], pz[sel[:k]])
            px, py, pz = px[sel[k:]], py[sel[k:]], pz[sel[k:]]
        Zr, ng = EX.rasterise(px, py, pz, n, xmin, xmax, ymin, ymax)
        B[0, :, :, c] = Zr; n_in += ng
    # GT-free 방향: 유보점 잔차 최소
    std = float(np.nanstd(B)); bmin, bmax = ((-3 * std, 3 * std) if std > 0 else (-.01, .01))
    Xn = (B - bmin) / (bmax - bmin)
    best, bres, bZ = None, np.inf, None
    for sgn in G.SIGNS:
        PH = EX.phase_mat(KX, KY, sgn[0], sgn[1], dt, 50.0)
        Y = np.squeeze(model.predict([Xn, PH], verbose=0)) * (bmax - bmin) + bmin
        hx, hy, hz = held
        ix = np.clip(np.floor((hx - xmin) / (xmax - xmin) * (n - 1) + .5).astype(int), 0, n - 1)
        iy = np.clip(np.floor((hy - ymin) / (ymax - ymin) * (n - 1) + .5).astype(int), 0, n - 1)
        r = float(np.sqrt(np.mean((Y[iy, ix] - hz) ** 2)))
        if r < bres:
            best, bres, bZ = sgn, r, Y
    Zq = EX.to_query(bZ, gx, gy, gx, gy, n=n)
    row = dict(episode=os.path.basename(path), crowded=bool(G.is_crowded(os.path.basename(path))),
               zero=zero, dt=dt, j0=int(j0), dir_sign=list(best),
               n_obs_deploy=int(V[j0 - 1:j0 + 2].sum()), n_in_deploy=int(n_in),
               deploy_is_stored=True, gt_free_deploy=bool(n_in <= int(V[j0-1:j0+2].sum())),
               cnn_deploy=float(np.sqrt(np.mean((Zq - g) ** 2)) / sg))
    # ── 사다리: ⛔**참값 표본**(실현 가능한 관측이 아니다)
    for m_pts in G.LADDER:
        rng = np.random.default_rng(G.SAMPLE_SEED + m_pts)
        Bl = np.zeros((1, n, n, 3), np.float32)
        for c, jj in enumerate((j0 - 1, j0, j0 + 1)):
            iy = rng.integers(0, len(gy), m_pts); ix = rng.integers(0, len(gx), m_pts)
            Zr, _ = EX.rasterise(gx[ix], gy[iy], Z[jj][iy, ix], n, xmin, xmax, ymin, ymax)
            Bl[0, :, :, c] = Zr
        std = float(np.nanstd(Bl)); bmin, bmax = ((-3*std, 3*std) if std > 0 else (-.01, .01))
        PH = EX.phase_mat(KX, KY, best[0], best[1], dt, 50.0)
        Y = np.squeeze(model.predict([(Bl - bmin) / (bmax - bmin), PH], verbose=0)) * (bmax - bmin) + bmin
        row[f'cnn_{m_pts}'] = float(np.sqrt(np.mean((EX.to_query(Y, gx, gy, gx, gy, n=n) - g) ** 2)) / sg)
    row['wall_s'] = round(time.time() - t0, 1)
    return row


def main():
    model = EX.build_model(G.CNN_N)
    pos = R1.positive_control(model); kok, krel = R1.kgrid_check()
    eok, emsg = G.ext_clean()
    print(f'  G-POS {pos:.4f} · G-KGRID {krel:.2e} · G-NOMOD {eok}', flush=True)
    files = sorted(glob.glob(os.path.join(G.HOS, '*.h5')))
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT, encoding='utf-8'):
            if l.strip():
                done.add(json.loads(l).get('episode'))
    with open(OUT, 'a', encoding='utf-8') as fh:
        for i, p in enumerate(files, 1):
            if os.path.basename(p) in done:
                continue
            try:
                r = run(p, model)
            except Exception as e:                                   # noqa: BLE001
                r = dict(episode=os.path.basename(p),
                         crowded=bool(G.is_crowded(os.path.basename(p))),
                         error=f'{type(e).__name__}: {e}')
            fh.write(json.dumps(r, ensure_ascii=False) + '\n'); fh.flush()
            if i % 8 == 0 or 'error' in r:
                if 'error' in r:
                    msg = '⛔' + r['error'][:60]
                else:
                    msg = 'zero %.3f deploy %.3f' % (r['zero'], r['cnn_deploy'])
                print('  [%d/%d] %-42s %s' % (i, len(files), r['episode'][:40], msg), flush=True)
    json.dump(dict(pos_nrmse=pos, kgrid_ok=kok, kgrid_relerr=krel,
                   ext_clean=eok, ext_msg=emsg),
              open(META, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('✅ 완료')
    return 0


if __name__ == '__main__':
    sys.exit(main())
