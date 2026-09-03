"""`E2` 개정 1 의 측정 — **발자국 대 채점영역**. ⛔추정기 런 0회, 판정 무수정.

적대적 재검이 지적한 것: `E2` 의 사다리는 **한 요인(밀도) 축이 아니다**. 사다리 칸은
채점영역 **전체**에 균등 표본을 뿌리는데, 배포 팔의 반환은 영역의 일부에만 있다.
⟹ 배포 칸이 지는 것이 「밀도가 낮아서」인지 「덮이지 않아서」인지 이 배치는 못 가른다.
"""
import glob, json, os
import numpy as np, h5py

HERE = os.path.dirname(os.path.abspath(__file__))
HOS = os.path.join(HERE, '..', 'HOSGATE_2026-08-13', 'data_hos')
RUNG = 200                       # 비교 대상 사다리 칸 (E2 에서 48/48 통과)


def main():
    rows = []
    for p in sorted(glob.glob(os.path.join(HOS, '*.h5'))):
        with h5py.File(p, 'r') as f:
            P = np.asarray(f['frames/lidar_points'])
            V = np.asarray(f['frames/lidar_valid_mask'])
            gx = np.asarray(f['frames/gt_grid_x']); gy = np.asarray(f['frames/gt_grid_y'])
        j = P.shape[0] // 2
        pts = P[j][V[j].astype(bool)]
        if pts.size == 0:
            continue
        bx = float(pts[:, 0].max() - pts[:, 0].min())
        by = float(pts[:, 1].max() - pts[:, 1].min())
        dom = float((gx.max() - gx.min()) * (gy.max() - gy.min()))
        rows.append(dict(episode=os.path.basename(p), n_pts=int(len(pts)),
                         foot_area=bx * by, domain=dom, frac=bx * by / dom,
                         local_density=len(pts) / (bx * by),
                         rung_global_density=RUNG / dom))
    frac = np.array([r['frac'] for r in rows])
    ratio = np.array([r['local_density'] / r['rung_global_density'] for r in rows])
    out = dict(n=len(rows), rung=RUNG,
               frac_median=float(np.median(frac)),
               frac_min=float(frac.min()), frac_max=float(frac.max()),
               ratio_median=float(np.median(ratio)),
               ratio_min=float(ratio.min()), ratio_max=float(ratio.max()))
    json.dump(dict(summary=out, rows=rows),
              open(os.path.join(HERE, 'footprint_e2.json'), 'w'),
              ensure_ascii=False, indent=1)
    print(f"n={out['n']}  발자국/채점영역 중앙 {out['frac_median']*100:.1f}% "
          f"({out['frac_min']*100:.1f}–{out['frac_max']*100:.1f}%)")
    print(f"  국소밀도 / {RUNG}칸 전역밀도 = 중앙 {out['ratio_median']:.2f}배 "
          f"({out['ratio_min']:.2f}–{out['ratio_max']:.2f})")


if __name__ == '__main__':
    main()
