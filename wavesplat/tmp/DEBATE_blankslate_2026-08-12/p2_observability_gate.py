#!/usr/bin/env python3
"""P2 (제어 실무자) R2 — **재실행 가능한 관측성 게이트**.

R1 에서 내가 주장한 "FF 미리보기 16점 중 15점이 0/488 프레임 관측, U=11.3 에서 8/8 전부 0" 을
메인 세션이 `[미검증]` 으로 표시했다. R1 의 임시 스크립트는 세션 스크래치패드에 있었고
**저장되지 않았다** ⟹ 이 파일은 **처음부터 다시 구현한 것**이고, 아래 수치가 권위다.

## 입력 (전부 저장소 안, 임의로 고른 값 0개)
- 프리뷰점 정의 = `wavesplat/run.py:319-338` (`S2_ALPHA=1` 블록, 출하 코드).
    foils `x_foil ∈ {2.0, -2.5}` m  ← `hydrofoil/config.py:67,79` (`x_pos`)
    leads `L ∈ {0.1, 0.2, 0.3, 0.5}` s
    speed `U = float(os.environ.get('S2_SPEED', '11.3'))` ← `hydrofoil/config.py:41` `speed=11.3`
    preview point `x_p = x_foil + U * L`, `y = 0`
  ⟹ **좌표는 내가 고른 것이 아니라 출하 계측 코드가 계산하는 바로 그 점이다.**
- 논문이 요구하는 영역 = `thesis/thesis_main.tex:3014` "각 foil 전방 ~3 m 이내",
  `:3005` "head sea 0.1 s 최적 (~1.1 m 전방)".
- 센서 = `configs/default.yaml` `lidar.height=5.0`, `fov_v_min_deg=-25.0`, `range_min=1.5`.
- 에피소드 = BS-3 이 쓴 24개 (`tmp/BLANKSLATE_2026-08-12/bs3_ship_rings.py:99-104` 그대로 복사).
- 실측 선속 창 1.2-2.4 m/s = `memory/reference_agreed_operating_point.md` (합의 작동점 11.3 m/s).

## 모드
    python3 p2_observability_gate.py gate     # 순간 관측 커버리지 (원 주장 재현)
    python3 p2_observability_gate.py sens     # 민감도: U x lead x foil x 게이트반경 x 센서기하
    python3 p2_observability_gate.py cone     # 반대가설: 배가 움직이면/dir=180 이면 열리는가
    python3 p2_observability_gate.py rulers   # P1 대 P2: 셸 vs 배-고리 교차예측 판정
    python3 p2_observability_gate.py all
출력: `p2_observability_gate.json` (같은 폴더). 콘솔에는 표만 인쇄한다.
"""
import json
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))          # .../wavesplat/wavesplat
OUT = os.path.join(HERE, 'p2_observability_gate.json')
JSONL = os.path.join(REPO, 'tmp', 'BLANKSLATE_2026-08-12', 'ship_rings.jsonl')

# ── 에피소드: bs3_ship_rings.py:99-104 과 글자 그대로 동일 ────────────────────
SPARSE = [f'data_s9_sparse_2026-08-11/episode_{bf}_{nc}comp_seed{s}_uniform_ra_rf_ds.h5'
          for bf in ('BF4', 'BF5') for nc in (2, 5) for s in (56, 57, 58)]
CROWDED = [f'data_s9_crowded_2026-08-11/episode_{bf}_{nc}comp_seed{s}_uniform_ra_rf_ds.h5'
           for bf in ('BF4', 'BF5') for nc in (15, 30) for s in (52, 53, 54)]
EPISODES = SPARSE + CROWDED
STRIDE = 5                       # 매 5프레임 (301 프레임 -> 61/에피소드)

# ── 출하 코드에서 그대로 가져온 상수 ─────────────────────────────────────────
FOILS = (2.0, -2.5)              # hydrofoil/config.py:67,79
LEADS = (0.1, 0.2, 0.3, 0.5)     # run.py:325
U_SHIP = 11.3                    # hydrofoil/config.py:41  (run.py:323 기본값)
U_ENC = 19.1                     # thesis_main.tex:638  V + c_p cos(beta), head sea
LIDAR_H = 5.0                    # configs/default.yaml
FOV_V_MIN = -25.0                # configs/default.yaml
GATE_R = 2.0                     # 프리뷰점 반경 R 안에 반환이 하나라도 있으면 '관측됨'
GRID_XMIN = 10.0                 # configs/default.yaml gt_grid.x_range[0]

SHIP_EDGES = [10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0, np.inf]
SHIP_KEYS = [f'{int(a)}_{"inf" if b == np.inf else int(b)}'
             for a, b in zip(SHIP_EDGES[:-1], SHIP_EDGES[1:])]
OBS_EDGES = [0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, np.inf]
OBS_KEYS = [f'{int(a)}_{"inf" if b == np.inf else int(b)}'
            for a, b in zip(OBS_EDGES[:-1], OBS_EDGES[1:])]

RESULT = {}


def r_min_geom(h=LIDAR_H, fov_v_min_deg=FOV_V_MIN):
    """FOV 하단 광선이 정지수면과 만나는 수평거리 = 사각지대 반경."""
    return h / np.tan(np.deg2rad(abs(fov_v_min_deg)))


def load_returns(path, stride=STRIDE):
    """유효 LiDAR 반환점 (x, y) 를 프레임 리스트로."""
    with h5py.File(path, 'r') as f:
        pts = f['frames/lidar_points'][::stride]          # (F, P, 3)
        msk = f['frames/lidar_valid_mask'][::stride]      # (F, P)
        comp = f['config/wave_components'][:]             # [A, omega, kx, ky, phi, Q]
        gx = f['frames/gt_grid_x'][:]
        gy = f['frames/gt_grid_y'][:]
    out = [(pts[i][msk[i]][:, 0], pts[i][msk[i]][:, 1]) for i in range(len(pts))]
    return out, comp, gx, gy


# ═══════════════════════════════════════════════════════════════════════════
# 1. GATE — 원 주장의 재현
# ═══════════════════════════════════════════════════════════════════════════
def preview_points(U, foils=FOILS, leads=LEADS):
    return [(f'{xf}|{L}', xf + U * L) for xf in foils for L in leads]


def mode_gate():
    pts_U = {'11.3': preview_points(U_SHIP), '19.1': preview_points(U_ENC)}
    tally = {u: {k: 0 for k, _ in pl} for u, pl in pts_U.items()}
    mindist = {u: {k: [] for k, _ in pl} for u, pl in pts_U.items()}
    nframes = 0
    rstats = []
    per_ep = {}
    for ep in EPISODES:
        frames, _, _, _ = load_returns(os.path.join(REPO, ep))
        ep_tally = {u: {k: 0 for k, _ in pl} for u, pl in pts_U.items()}
        ep_n = 0
        for ox, oy in frames:
            if len(ox) == 0:
                continue
            ep_n += 1
            nframes += 1
            r = np.hypot(ox, oy)
            rstats.append((float(r.min()), float(np.median(r)),
                           float(np.percentile(r, 95)), len(ox)))
            for u, pl in pts_U.items():
                for k, xp in pl:
                    d = float(np.min(np.hypot(ox - xp, oy)))
                    mindist[u][k].append(d)
                    if d < GATE_R:
                        tally[u][k] += 1
                        ep_tally[u][k] += 1
        per_ep[ep] = {u: {k: [v, ep_n] for k, v in d.items()} for u, d in ep_tally.items()}

    rs = np.array(rstats)
    res = {
        'n_frames': nframes, 'n_episodes': len(EPISODES), 'stride': STRIDE,
        'gate_radius_m': GATE_R,
        'return_radius': {'min': float(rs[:, 0].min()), 'median_of_frame_min': float(np.median(rs[:, 0])),
                          'median': float(np.median(rs[:, 1])), 'p95': float(np.median(rs[:, 2])),
                          'mean_n_returns': float(rs[:, 3].mean())},
        'r_min_geom_m': float(r_min_geom()),
        'points': {}, 'per_episode': per_ep,
    }
    print(f"\n=== GATE  ({len(EPISODES)} ep x stride {STRIDE} = {nframes} frames, "
          f"gate radius {GATE_R} m) ===")
    print(f"LiDAR 반환 반경: min {rs[:,0].min():.2f} m · 프레임최소 중앙 "
          f"{np.median(rs[:,0]):.2f} m · 중앙 {np.median(rs[:,1]):.1f} m · p95 "
          f"{np.median(rs[:,2]):.1f} m · 기하 사각지대 {r_min_geom():.2f} m")
    print(f"{'U':>6} {'foil':>6} {'lead':>5} {'x_p':>7} {'관측프레임':>12} {'비율':>7} "
          f"{'프리뷰점~최근접반환 중앙':>22}")
    for u, pl in pts_U.items():
        for k, xp in pl:
            c = tally[u][k]
            md = float(np.median(mindist[u][k]))
            xf, L = k.split('|')
            print(f"{u:>6} {xf:>6} {L:>5} {xp:7.2f} {c:6d}/{nframes:<5d} "
                  f"{c/nframes:7.3f} {md:22.2f}")
            res['points'][f'{u}|{k}'] = {'x_p': xp, 'observed_frames': c,
                                         'n_frames': nframes, 'frac': c / nframes,
                                         'median_dist_to_nearest_return_m': md}
    n_zero = sum(1 for v in res['points'].values() if v['observed_frames'] == 0)
    print(f"\n16점 중 0/{nframes} 프레임 관측 = **{n_zero}점**  "
          f"(U=11.3 만 보면 "
          f"{sum(1 for k,v in res['points'].items() if k.startswith('11.3') and v['observed_frames']==0)}/8)")

    # 논문이 요구하는 영역 자체 (프리뷰점이 아니라 구간)
    print("\n--- thesis_main.tex:3014 '각 foil 전방 ~3 m 이내' 구간 커버리지 ---")
    box = {}
    for xf in FOILS:
        lo, hi = xf, xf + 3.0
        hit = 0
        for ep in EPISODES:
            frames, _, _, _ = load_returns(os.path.join(REPO, ep))
            for ox, oy in frames:
                if len(ox) and np.any((ox >= lo) & (ox <= hi) & (np.abs(oy) <= 5.0)):
                    hit += 1
        box[f'{xf}'] = {'x_range': [lo, hi], 'y_abs_max': 5.0,
                        'frames_with_return': hit, 'n_frames': nframes}
        print(f"  foil x={xf:+.1f}  구간 x∈[{lo:.1f},{hi:.1f}], |y|≤5 : "
              f"{hit}/{nframes} 프레임에 반환 존재")
    res['thesis_3m_box'] = box
    RESULT['gate'] = res
    return res


# ═══════════════════════════════════════════════════════════════════════════
# 2. SENS — 민감도
# ═══════════════════════════════════════════════════════════════════════════
def mode_sens(gate_res=None):
    """게이트는 **선체고정 기하**이므로 U 는 오직 x_p 를 통해서만 들어온다.
    따라서 관측가능 조건은 닫힌 형태로 x_p >= r_obs_min 이다. 경험적 r_obs_min 을
    쓰고, 표는 (U, lead, foil) 격자에서 x_p 와 판정을 낸다."""
    if gate_res is None:
        gate_res = RESULT.get('gate') or mode_gate()
    r_emp = gate_res['return_radius']['min']              # 전 프레임 통틀어 최소 반환 반경
    r_emp_typ = gate_res['return_radius']['median_of_frame_min']
    r_geo = gate_res['r_min_geom_m']

    speeds = [1.2, 2.4, 5.0, 11.3, 15.0, 19.1, 25.0]
    leads = [0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0]
    rows = []
    print(f"\n=== SENS-A: x_p = x_foil + U*L  vs  사각지대 (기하 {r_geo:.2f} m, "
          f"경험적 최소 {r_emp:.2f} m, 프레임최소 중앙 {r_emp_typ:.2f} m) ===")
    print("판정: ○ = x_p >= 프레임최소중앙(신뢰) · △ = x_p >= 경험적최소(간헐) · × = 사각지대")
    for xf in (2.0, -2.5):
        print(f"\n  foil x_foil = {xf:+.1f} m       " +
              "".join(f"{L:>8.1f}s" for L in leads))
        for U in speeds:
            cells = []
            for L in leads:
                xp = xf + U * L
                v = '○' if xp >= r_emp_typ else ('△' if xp >= r_emp else '×')
                cells.append(f"{xp:7.1f}{v}")
                rows.append({'x_foil': xf, 'U': U, 'lead': L, 'x_p': xp, 'verdict': v})
            print(f"    U={U:5.1f} m/s   " + "".join(cells))
    # 필요 조건 닫힌 형태
    req = {}
    print("\n=== SENS-B: FF 대역(L<=0.5 s)을 열려면 얼마가 필요한가 ===")
    for xf in (2.0, -2.5):
        for L in (0.1, 0.3, 0.5):
            U_need = (r_emp_typ - xf) / L
            req[f'{xf}|{L}'] = U_need
            print(f"  foil {xf:+.1f}, lead {L:.1f} s -> 필요 선속 U >= "
                  f"{U_need:7.1f} m/s  ({U_need*1.944:6.0f} kt)"
                  + ("   <= 설계 11.3 m/s 로 충족" if U_need <= U_SHIP else ""))
    # 센서 기하를 바꾸면?
    print("\n=== SENS-C: 사각지대 반경 r_min = h / tan|fov_v_min|  (센서 재거치) ===")
    sensor = {}
    for h in (2.0, 3.0, 5.0, 8.0):
        for fov in (-25.0, -45.0, -60.0, -75.0):
            r = r_min_geom(h, fov)
            sensor[f'h{h}|fov{fov}'] = r
        print(f"  h={h:4.1f} m : " + "  ".join(
            f"fov {fov:+.0f}° -> {r_min_geom(h,fov):5.2f} m" for fov in (-25., -45., -60., -75.)))
    print(f"  ※ 출하 구성 h=5.0, fov=-25° -> {r_geo:.2f} m."
          f"  FF 최근점 x_p={2.0+U_SHIP*0.1:.2f} m 를 덮으려면 "
          f"|fov| >= {np.rad2deg(np.arctan(LIDAR_H/(2.0+U_SHIP*0.1))):.0f}° 또는 "
          f"h <= {(2.0+U_SHIP*0.1)*np.tan(np.deg2rad(25)):.2f} m 가 필요하다.")
    RESULT['sens'] = {'r_geom': r_geo, 'r_emp_min': r_emp, 'r_emp_frame_min_median': r_emp_typ,
                      'grid': rows, 'required_speed': req, 'sensor_geometry': sensor}
    return RESULT['sens']


# ═══════════════════════════════════════════════════════════════════════════
# 3. CONE — 반대가설: 움직이는 배 / dir=180
# ═══════════════════════════════════════════════════════════════════════════
def mode_cone(gate_res=None):
    """순간 커버리지가 아니라 **인과 원뿔(causal cone)** 을 묻는다.
    선체고정 좌표에서 파 특징이 프리뷰점 x_p 로 접근하는 속도:
        following (dir=0,  파 +x):  v_close = U - c_p     (U<c_p 면 뒤에서 추월 -> 영원히 미관측)
        head     (dir=180, 파 -x):  v_close = U + c_p
    현재 시각 t 에 x_p 에 있는 특징은 t-tau 에 x_p + v_close*tau 에 있었다.
    관측되려면 x_p + v_close*tau >= r_obs  ==>  tau_fill = (r_obs - x_p)/v_close.
    tau_fill 은 **요구되는 순수 예보 리드**다."""
    if gate_res is None:
        gate_res = RESULT.get('gate') or mode_gate()
    r_obs = gate_res['return_radius']['median_of_frame_min']

    # 에피소드별 에너지가중 위상/군속도 (실제 성분에서)
    cps, cgs = [], []
    for ep in EPISODES:
        with h5py.File(os.path.join(REPO, ep), 'r') as f:
            c = f['config/wave_components'][:]           # A, omega, kx, ky, phi, Q
        A, om, kx, ky = c[:, 0], c[:, 1], c[:, 2], c[:, 3]
        k = np.hypot(kx, ky)
        w = A ** 2
        d = 50.0                                            # configs/default.yaml episode.depth
        cp_c = om / k
        cg_c = 0.5 * cp_c * (1.0 + 2.0 * k * d / np.sinh(np.clip(2.0 * k * d, 0, 700)))
        cps.append(float(np.sum(w * cp_c) / np.sum(w)))
        cgs.append(float(np.sum(w * cg_c) / np.sum(w)))
    cp_med, cg_med = float(np.median(cps)), float(np.median(cgs))
    print(f"\n=== CONE  (r_obs = 프레임최소반환 중앙 {r_obs:.2f} m; 에너지가중 "
          f"c_p 중앙 {cp_med:.2f} [{min(cps):.2f},{max(cps):.2f}] · "
          f"c_g 중앙 {cg_med:.2f} [{min(cgs):.2f},{max(cgs):.2f}] m/s) ===")

    scen = []
    xps = {f'{xf}|{L}': xf + U_SHIP * L for xf in FOILS for L in LEADS}
    xp_near = min(xps.values()); xp_far = max(xps.values())
    print(f"프리뷰점(U=11.3) x_p ∈ [{xp_near:.2f}, {xp_far:.2f}] m")
    print("c_g = 보수적(결정론적 예보가능영역 이론) · c_p = 낙관적(단일 crest 추적)\n")
    hdr = (f"{'시나리오':<32}" + f"{'v(c_g)':>8}{'τ_fill 최원~최근 (c_g)':>26}"
           + f"{'v(c_p)':>8}{'τ_fill (c_p)':>20}  판정(c_g 기준)")
    print(hdr)
    for name, direc, U in [
        ('현행 코퍼스: dir=0, 정지',            0,   0.0),
        ('dir=0, 배 U=1.2 (실측창)',            0,   1.2),
        ('dir=0, 배 U=2.4 (실측창)',            0,   2.4),
        ('dir=0, 배 U=11.3 (설계, 서프라이딩)', 0,  11.3),
        ('dir=180, 정지 (스칼라 1개만 수정)',   180, 0.0),
        ('dir=180, 배 U=1.2',                   180, 1.2),
        ('dir=180, 배 U=2.4',                   180, 2.4),
        ('dir=180, 배 U=11.3 (설계 작동점)',    180, 11.3),
    ]:
        row = {'name': name, 'dir': direc, 'U': U}
        for tag, c in (('cg', cg_med), ('cp', cp_med)):
            v = (U - c) if direc == 0 else (U + c)
            if v <= 0:
                tn = tf = float('inf')
            else:
                tn = (r_obs - xp_near) / v
                tf = (r_obs - xp_far) / v
            row[f'v_close_{tag}'] = v
            row[f'tau_fill_near_{tag}'] = tn
            row[f'tau_fill_far_{tag}'] = tf
        t = row['tau_fill_near_cg']
        row['verdict'] = ('영원히 미관측' if not np.isfinite(t) else
                          ('열림 · E-D 측정지평 L<=2 s 안' if t <= 2.0 else
                           '원리상 열리나 리드 > 2 s = 미측정'))
        scen.append(row)
        f2 = lambda x: '∞' if not np.isfinite(x) else f'{x:.2f}'
        print(f"{name:<32}{row['v_close_cg']:8.2f}"
              f"{f2(row['tau_fill_near_cg'])+' ~ '+f2(row['tau_fill_far_cg'])+' s':>26}"
              f"{row['v_close_cp']:8.2f}"
              f"{f2(row['tau_fill_near_cp'])+' ~ '+f2(row['tau_fill_far_cp'])+' s':>20}"
              f"  {row['verdict']}")

    # 고리 -> 리드타임 번역 (dir=180 에서만 의미가 있다)
    mids = [15., 25., 40., 62.5, 87.5, 125., 175., 235.]
    print("\n--- 배-고리 -> 리드타임 번역 (dir=180 일 때만 정의됨) ---")
    print(f"{'고리':>10}{'r̄ [m]':>8}" + "".join(
        f"{n:>16}" for n in ('U=0 (c_g)', 'U=11.3 (c_g)', 'U=11.3 (c_p)')))
    ring_lead = {}
    for k, r in zip(SHIP_KEYS, mids):
        a, b, c_ = r / cg_med, r / (11.3 + cg_med), r / (11.3 + cp_med)
        ring_lead[k] = {'r_mid': r, 'lead_U0_cg': a, 'lead_U113_cg': b, 'lead_U113_cp': c_}
        print(f"{k:>10}{r:8.0f}{a:16.2f}{b:16.2f}{c_:16.2f}")

    print("\n※ **순간 커버리지는 U 에 불변이다** — 센서·포일이 같은 강체에 붙어 있어 배가"
          "\n   움직여도 사각지대가 함께 움직인다. 합성개구가 여는 것은 커버리지가 아니라 "
          "**인과 원뿔**이다.")
    RESULT['cone'] = {'r_obs': r_obs, 'cp_median': cp_med, 'cg_median': cg_med,
                      'cp_all': cps, 'cg_all': cgs, 'scenarios': scen,
                      'ring_lead': ring_lead}
    return RESULT['cone']


# ═══════════════════════════════════════════════════════════════════════════
# 4. RULERS — P1 대 P2 판정
# ═══════════════════════════════════════════════════════════════════════════
def mode_rulers():
    """두 자(관측셸 · 배-고리)의 결합질량표를 기하만으로 계산하고,
    `ship_rings.jsonl` 의 고리·셸 MSE 로 **교차예측**을 시험한다.
      A) 셸 MSE 를 고리 MSE 로 예측  (셸이 고리축의 거친 빈이면 잘 맞아야 한다)
      B) 고리 MSE 를 셸 MSE 로 예측  (고리가 셸축의 거친 빈이면 잘 맞아야 한다)
    A 만 맞고 B 가 틀리면 = **셸이 고리축을 미분해**(P1 의 주장). 둘 다 틀리면 = 서로 다른 축."""
    # ── 결합질량 (기하만) ──
    joint = np.zeros((len(SHIP_KEYS), len(OBS_KEYS)))
    nfr = 0
    for ep in EPISODES:
        frames, _, gx, gy = load_returns(os.path.join(REPO, ep))
        Xg, Yg = np.meshgrid(gx, gy)
        dship = np.hypot(Xg, Yg).ravel()
        si = np.digitize(dship, SHIP_EDGES[1:-1])
        for ox, oy in frames:
            if len(ox) == 0:
                continue
            nfr += 1
            d = np.min(np.hypot(Xg.ravel()[:, None] - ox[None, :],
                                Yg.ravel()[:, None] - oy[None, :]), axis=1)
            oi = np.digitize(d, OBS_EDGES[1:-1])
            np.add.at(joint, (si, oi), 1.0)
    joint /= joint.sum()
    P_obs_given_ship = joint / joint.sum(axis=1, keepdims=True)
    P_ship_given_obs = joint / joint.sum(axis=0, keepdims=True)

    print(f"\n=== RULERS: 결합질량표 ({nfr} 프레임, 격자 4800셀) ===")
    print("P(관측셸 | 배-고리)  — 행=배-고리")
    print(f"{'':>10}" + "".join(f"{k:>9}" for k in OBS_KEYS))
    for i, sk in enumerate(SHIP_KEYS):
        print(f"{sk:>10}" + "".join(f"{100*P_obs_given_ship[i,j]:9.1f}" for j in range(len(OBS_KEYS))))
    print("\nP(배-고리 | 관측셸)  — 행=관측셸")
    print(f"{'':>10}" + "".join(f"{k:>9}" for k in SHIP_KEYS))
    for j, ok in enumerate(OBS_KEYS):
        print(f"{ok:>10}" + "".join(f"{100*P_ship_given_obs[i,j]:9.1f}" for i in range(len(SHIP_KEYS))))

    # ── MSE 원자료 ──
    rows = [json.loads(l) for l in open(JSONL)]
    out = {'n_frames': nfr,
           'P_obs_given_ship': P_obs_given_ship.tolist(),
           'P_ship_given_obs': P_ship_given_obs.tolist(),
           'ship_keys': SHIP_KEYS, 'obs_keys': OBS_KEYS,
           'p1_check': {
               'P(shell 0_5 | ring 10_20)': float(P_obs_given_ship[0, 0]),
               'P(ring<=50 | shell 0_5)': float(P_ship_given_obs[:3, 0].sum()),
               'P(ring 10_20 | shell 0_5)': float(P_ship_given_obs[0, 0]),
           },
           'cross_prediction': {}}
    print(f"\nP1 의 두 수 재현: P(셸 0_5 | 고리 10_20) = "
          f"{100*P_obs_given_ship[0,0]:.1f}%   (P1: 98.3-98.7%)")
    print(f"                  P(배-거리<=50 | 셸 0_5) = "
          f"{100*P_ship_given_obs[:3,0].sum():.1f}%   (P1: 89.4-91.0%)")
    print(f"                  P(고리 10_20 | 셸 0_5) = {100*P_ship_given_obs[0,0]:.1f}%")

    print("\n--- 교차예측 (팔별, 에피소드 평균 MSE) ---")
    for cell in ('SHIP', 'NT'):
        for reg in ('sparse', 'crowded'):
            sel = [r for r in rows if r['cell'] == cell and r['regime'] == reg]
            if not sel:
                continue
            # jsonl 은 고리 RMSE 를 담는다 (bs3_ship_rings.py:129) -> MSE 로 되돌린다
            mse_ship = np.array([np.nanmean([r['ship'][k]['rmse'] ** 2 for r in sel])
                                 for k in SHIP_KEYS])
            mse_obs = np.array([np.nanmean([r['obs'][k]['rmse'] ** 2 for r in sel])
                                for k in OBS_KEYS])
            pred_obs = P_ship_given_obs.T @ np.nan_to_num(mse_ship)     # 셸 <- 고리
            pred_ship = P_obs_given_ship @ np.nan_to_num(mse_obs)       # 고리 <- 셸
            eA = np.abs(np.log(pred_obs[:4] / mse_obs[:4]))             # 안쪽 4 셸
            eB = np.abs(np.log(pred_ship[:4] / mse_ship[:4]))           # 안쪽 4 고리
            out['cross_prediction'][f'{cell}|{reg}'] = {
                'mse_ship_rings': mse_ship.tolist(), 'mse_obs_shells': mse_obs.tolist(),
                'pred_shell_from_rings': pred_obs.tolist(),
                'pred_ring_from_shells': pred_ship.tolist(),
                'A_mean_abs_log_err_inner4': float(eA.mean()),
                'B_mean_abs_log_err_inner4': float(eB.mean()),
                'A_ratio_shell_0_5': float(pred_obs[0] / mse_obs[0]),
                'B_ratio_ring_10_20': float(pred_ship[0] / mse_ship[0]),
            }
            print(f"  {cell:>4} {reg:>8}   A(셸<-고리) 안쪽4 평균|log err| = {eA.mean():.3f} "
                  f"[셸 0_5 예측/실제 = {pred_obs[0]/mse_obs[0]:.2f}x]   "
                  f"B(고리<-셸) = {eB.mean():.3f} "
                  f"[고리 10_20 예측/실제 = {pred_ship[0]/mse_ship[0]:.2f}x]")
    print("\n판정 규칙(결과 전 선언): A 오차 < B 오차 이면 '셸이 고리축을 미분해'(P1),"
          "\n                        A ≈ B 이면 '두 자는 대칭적으로 서로를 못 예측'(서로 다른 축).")
    RESULT['rulers'] = out
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 5. DOMAIN — ★게이트 결과의 교락 감사 (내 R1 주장을 내가 깎는 절)
# ═══════════════════════════════════════════════════════════════════════════
def mode_domain():
    """0/1464 는 **세 원인이 같은 반경에서 겹쳐** 과결정돼 있다.
      (a) 레이캐스트 메시가 x>=10 m 에서만 존재한다 (`wavesim/pipeline.py:160-161` 이
          gt_grid 와 **같은** x_range 로 메시를 만든다) ⟹ x<10 m 에는 물 자체가 없다.
      (b) GT 격자도 x>=10 (`configs/default.yaml gt_grid.x_range`).
      (c) FOV 기하: 평수면 r_min = h/tan25 = 10.72 m.
    (a),(b) 는 **시뮬레이터 성질**이고 (c) 만 센서 성질이다. 게다가 (c) 는 파고에 따라
    r_min = (h - eta)/tan25 로 **줄어든다** — 마루 위에서는 훨씬 가까이 맞는다."""
    t = np.tan(np.deg2rad(abs(FOV_V_MIN)))
    rows = []
    for ep in EPISODES:
        with h5py.File(os.path.join(REPO, ep), 'r') as f:
            A = f['config/wave_components'][:, 0]
            eta_max = float(f['frames/gt_wave_field'][:].max())
            x = f['frames/lidar_points'][:, :, 0][f['frames/lidar_valid_mask'][:]]
        rows.append({'ep': ep, 'sumA': float(A.sum()), 'eta_max_grid': eta_max,
                     'r_min_at_sumA': (LIDAR_H - float(A.sum())) / t,
                     'r_min_at_eta_max': (LIDAR_H - eta_max) / t,
                     'x_min_returns': float(x.min()),
                     'frac_returns_below_10_05': float((x < 10.05).mean())})
    a = np.array([r['r_min_at_sumA'] for r in rows])
    b = np.array([r['r_min_at_eta_max'] for r in rows])
    xm = np.array([r['x_min_returns'] for r in rows])
    print("\n=== DOMAIN (교락 감사) ===")
    print(f"레이캐스트 메시 x_range = gt_grid.x_range = [{GRID_XMIN}, 250]  "
          f"(pipeline.py:160-161)  ⟹ x<{GRID_XMIN} m 에 **수면이 존재하지 않는다**")
    print(f"관측된 반환의 x 최소: 전 에피소드 최소 {xm.min():.3f} m, 중앙 {np.median(xm):.3f} m "
          f"⟹ 메시 경계 {GRID_XMIN} m 에 **정확히 붙어 있다**")
    print(f"FOV 기하 (평수면)                r_min = {LIDAR_H}/tan25 = {LIDAR_H/t:.2f} m")
    print(f"FOV 기하 (마루 = 격자 max eta)   r_min = {b.min():.2f} ~ {b.max():.2f} m "
          f"(중앙 {np.median(b):.2f})")
    print(f"FOV 기하 (마루 = sum|A| 상한)    r_min = {a.min():.2f} ~ {a.max():.2f} m "
          f"(중앙 {np.median(a):.2f})")
    ff = [2.0 + U_SHIP * L for L in LEADS]
    print(f"\nU=11.3 전방 포일 프리뷰점: {['%.2f' % v for v in ff]}")
    for xp in ff:
        n_ep = int((b <= xp).sum())
        print(f"  x_p={xp:5.2f} m : 마루 기준 FOV 로 **도달 가능한 에피소드** {n_ep}/24"
              + ("   ⟹ 이 코퍼스에서는 메시 경계 때문에 시험 불가" if n_ep else ""))
    print("\n⟹ 판정: 0/1464 는 참이지만 **과결정**이다. FF 관측성은 이 코퍼스에서"
          "\n   `UNANSWERABLE` (B-3 와 같은 판정). 답하려면 생성 스칼라 2개를 고쳐야 한다:"
          "\n   `gt_grid.x_range[0]: 10 -> -10` 과 `mesh.nx` 를 같은 간격 유지하도록 재계산.")
    RESULT['domain'] = {'rows': rows, 'r_min_flat': LIDAR_H / t,
                        'r_min_at_eta_max': {'min': float(b.min()), 'median': float(np.median(b)),
                                             'max': float(b.max())},
                        'ff_points_U113_front': ff,
                        'episodes_reachable_by_crest': {f'{xp:.2f}': int((b <= xp).sum())
                                                        for xp in ff}}
    return RESULT['domain']


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
    g = None
    if mode in ('domain', 'all'):
        mode_domain()
    if mode in ('gate', 'all'):
        g = mode_gate()
    if mode in ('sens', 'all'):
        mode_sens(g)
    if mode in ('cone', 'all'):
        mode_cone(g)
    if mode in ('rulers', 'all'):
        mode_rulers()
    if os.path.exists(OUT):
        RESULT = {**json.load(open(OUT)), **RESULT}
    json.dump(RESULT, open(OUT, 'w'), indent=1)
    print(f"\n[saved] {OUT}")
