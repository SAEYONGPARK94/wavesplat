"""`R3-GEOMETRY` 러너 — ⛔**추정기 런 0회**. `iR/jR` 로 카메라 자세를 자료에서 푼다."""
import glob
import hashlib
import json
import os
import re
import sys
import time

import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for p in (HERE, ROOT, os.path.join(ROOT, 'scripts'),
          os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23')):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(ROOT)
import gates_r3 as G                                                # noqa: E402
import scene_station as SS                                          # noqa: E402
from realdata_analysis import _unpack                               # noqa: E402

OUT = os.path.join(HERE, 'results_r3.jsonl')


def read_K(cfg_dir):
    """OpenCV FileStorage XML 에서 3x3 K."""
    p = os.path.join(cfg_dir, 'intrinsics_01.xml')
    if not os.path.exists(p):
        return None
    txt = open(p, encoding='utf-8', errors='replace').read()
    m = re.search(r'<data>(.*?)</data>', txt, re.S)
    if not m:
        return None
    v = [float(x) for x in m.group(1).split()]
    return np.array(v[:9]).reshape(3, 3) if len(v) >= 9 else None


PIX_OFFSET = 1500.0      # ⭐`iR`/`jR` 의 저장 오프셋 — 이것이 없으면 DLT 가 깨진다


def dlt(P3, p2):
    """★검증된 구현(2026-09-02 감사) — 2D·3D 양쪽 Hartley 정규화."""
    c3 = P3.mean(0); s3 = np.sqrt(3) / np.abs(P3 - c3).std()
    T3 = np.eye(4); T3[:3, :3] *= s3; T3[:3, 3] = -s3 * c3
    c2 = p2.mean(0); s2 = np.sqrt(2) / np.abs(p2 - c2).std()
    T2 = np.array([[s2, 0, -s2 * c2[0]], [0, s2, -s2 * c2[1]], [0, 0, 1]])
    Q = (P3 - c3) * s3; q = (p2 - c2) * s2; n = len(Q)
    Qh = np.hstack([Q, np.ones((n, 1))]); A = np.zeros((2 * n, 12))
    A[0::2, 0:4] = Qh; A[0::2, 8:12] = -q[:, 0:1] * Qh
    A[1::2, 4:8] = Qh; A[1::2, 8:12] = -q[:, 1:2] * Qh
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    return np.linalg.inv(T2) @ Vt[-1].reshape(3, 4) @ T3


def reproj(M, P3, p2):
    Ph = np.hstack([P3, np.ones((len(P3), 1))]); u = Ph @ M.T
    uv = u[:, :2] / u[:, 2:3]
    return uv, np.linalg.norm(uv - p2, axis=1)


def decomp(M, K):
    RT = np.linalg.inv(K) @ M
    U, S, Vt = np.linalg.svd(RT[:, :3]); sc = S.mean(); R = U @ Vt
    if np.linalg.det(R) < 0:
        R = -R; sc = -sc
    T = RT[:, 3] / sc
    return R, -R.T @ T, float(S.max() - S.min()) / float(S.mean())


def run(prefix, record):
    t0w = time.time()
    fp = glob.glob(os.path.join(G.FULL, prefix + '__*.nc'))
    if not fp:
        return dict(record=record, error='full nc 없음')
    cfg_dir = os.path.join(G.FULL, 'config_' + prefix)
    K = read_K(cfg_dir)
    short = os.path.join(ROOT, SS.RECORDS_12[record]['path'])

    with h5py.File(fp[0], 'r') as ff, h5py.File(short, 'r') as fs:
        n0, n1 = ff['Z'].shape[1], ff['Z'].shape[2]
        # ★G-1 — Z 바이트 동일 (표본 프레임 12개)
        idx = np.linspace(0, ff['Z'].shape[0] - 1, 12).astype(int)
        za = np.asarray(ff['Z'][idx]); zb = np.asarray(fs['Z'][idx])
        z_id = bool(np.array_equal(za, zb))
        # iR/jR
        iR0 = np.asarray(ff['iR'][0]).astype(float)
        jR0 = np.asarray(ff['jR'][0]).astype(float)
        iR1 = np.asarray(ff['iR'][1]).astype(float)
        chg = float(np.mean(iR0 != iR1))
        fill = int((iR0 == 0).sum() + (jR0 == 0).sum() == iR0.size * 2)
        X2 = _unpack(ff['_nc4_non_coord_X'], np.s_[:])
        Y2 = _unpack(ff['Y'], np.s_[:])
        Z0 = _unpack(ff['Z'], np.s_[0])

    m = np.isfinite(Z0) & np.isfinite(iR0) & np.isfinite(jR0)
    ii, jj = np.nonzero(m)
    take = np.linspace(0, len(ii) - 1, min(4000, len(ii))).astype(int)
    P3 = np.column_stack([X2[ii[take], jj[take]], Y2[ii[take], jj[take]],
                          Z0[ii[take], jj[take]]])
    uv = np.column_stack([iR0[ii[take], jj[take]] + PIX_OFFSET,
                          jR0[ii[take], jj[take]] + PIX_OFFSET])

    row = dict(record=record, full=os.path.basename(fp[0]), z_identical=z_id,
               irjr_change=chg, irjr_fill=fill, K_ok=bool(K is not None),
               n_corr=int(len(take)))
    if K is None:
        row.update(pose_ok=False, note='intrinsics 없음')
    else:
        M = dlt(P3, uv)
        uvp, err = reproj(M, P3, uv)
        R, C, spread = decomp(M, K)
        # 방사 잔차 평탄성
        c = np.array([n1 / 2, n0 / 2])
        rad = np.linalg.norm(uv - c, axis=1)
        q = np.quantile(rad, np.linspace(0, 1, 7))
        flat = float(np.std([np.median(err[(rad >= q[k]) & (rad < q[k + 1])])
                             for k in range(6)]))
        rng = np.linalg.norm(P3 - C, axis=1)
        nz = np.array([0.0, 0.0, 1.0])
        inc = np.degrees(np.arccos(np.clip(
            np.abs(((C - P3) / rng[:, None]) @ nz), 0, 1)))
        h_ok = abs(C[2] - G.SENSOR_H_DECLARED) <= G.SENSOR_H_TOL * G.SENSOR_H_DECLARED
        row.update(
            reproj_median=float(np.median(err)), reproj_p95=float(np.quantile(err, .95)),
            rot_spread=spread, radial_flat=flat,
            cam_center=[float(v) for v in C],
            range_min=float(rng.min()), range_max=float(rng.max()),
            incidence_min=float(inc.min()), incidence_max=float(inc.max()),
            incidence_median=float(np.median(inc)),
            height_ok=bool(h_ok),
            pose_ok=bool(np.median(err) <= G.REPROJ_MED_MAX
                         and spread <= G.ROT_SPREAD_MAX and h_ok))
    row.update(pid=os.getpid(), wall_s=round(time.time() - t0w, 1))
    return row


def main():
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            if l.strip():
                done.add(json.loads(l).get('record'))
    with open(OUT, 'a') as fh:
        for pre, rec in G.PAIRS:
            if rec in done:
                continue
            r = run(pre, rec)
            fh.write(json.dumps(r, ensure_ascii=False) + '\n'); fh.flush()
            if 'error' in r:
                print(f"  {rec:6s} ⛔{r['error']}", flush=True)
            else:
                print(f"  {rec:6s} Z동일={r['z_identical']} reproj={r.get('reproj_median',float('nan')):.3f}px "
                      f"rot={r.get('rot_spread',float('nan')):.4f} C={['%.2f'%v for v in r.get('cam_center',[0,0,0])]} "
                      f"pose_ok={r.get('pose_ok')}", flush=True)
    print('✅ 완료')
    return 0


if __name__ == '__main__':
    sys.exit(main())
