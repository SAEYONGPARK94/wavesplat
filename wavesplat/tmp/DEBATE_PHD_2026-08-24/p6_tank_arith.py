#!/usr/bin/env python3
"""P-6 / P-6b 산술 단일출처 (2026-08-26, 런 0회).

제5장에 들어가는 모든 수치를 여기서 낸다. 손복사 금지 규율.
  python3 tmp/DEBATE_PHD_2026-08-24/p6_tank_arith.py

출력 (A) 분산관계  — 논문 표 tab:tank_phase1 검증 + 본문 4572 의 |k| 정정
출력 (B) Nyquist   — P-6b 파고계 배열 제원
출력 (C) 관측 SNR  — P-6 정지규칙의 셀별 sigma_z/sigma_eta 투영

⚠(C) 의 sigma_obs = 0.01 m 는 **가정**이다(thesis_main.tex:4598 "LiDAR 거리
해상도 ~1cm at 50m"). 실측값은 계획 #31--32 가 낸다. 그것이 P-6 의 요지다.
"""
import math

G, D = 9.81, 3.5          # 중력, 수조 수심 (thesis_main.tex:4561)
PITCH_DEG = 10.0          # LiDAR 하방 피치 (thesis_main.tex:4590)
SIGMA_OBS_ASSUMED = 0.01  # m — 가정. (thesis_main.tex:4598)

# 실측 12 레코드의 sigma_noise/sigma 관측 범위 (thesis tab:real_noise_ratio,
# 원천 = tmp/DEBATE_PHD_2026-08-24/p1_noise_table.py). 여기서는 상수로 두되
# 문서에는 "표 3.3 에서 읽은 값" 이라 적는다.
REAL_RATIO_MIN, REAL_RATIO_MAX = 0.008, 0.218


def wavenumber(T, d=D, g=G):
    """유한 수심 분산관계 omega^2 = g k tanh(k d) 를 고정점 반복으로 푼다."""
    w = 2 * math.pi / T
    k = w * w / g
    for _ in range(300):
        k = w * w / (g * math.tanh(k * d))
    return k


def main():
    print("=" * 68)
    print("(A) 분산관계  d = %.1f m" % D)
    print("=" * 68)
    print(f"{'T [s]':>6} {'|k| [rad/m]':>12} {'lambda [m]':>11} {'|k|d':>7}")
    for T in [0.5, 0.7, 0.8, 0.85, 1.0, 1.15, 1.2, 1.4, 1.5, 1.8, 2.0, 3.5]:
        k = wavenumber(T)
        print(f"{T:6.2f} {k:12.4f} {2*math.pi/k:11.4f} {k*D:7.2f}")
    k12 = wavenumber(1.2)
    print(f"\n  -> 본문 4572 정정: |k|(T=1.2) = {k12:.4f} rad/m (논문 2.91 은 4.1% 초과),")
    print(f"     |k|d = {k12*D:.2f} (논문 10.2 는 과대). 같은 장 표 4659 의 2.81 이 옳다.")

    print()
    print("=" * 68)
    print("(B) Nyquist — 파고계 배열 (k <= pi/dx  =>  dx <= lambda/2)")
    print("=" * 68)
    for n_g, dx in [(11, 0.5), (5, 1.0)]:
        span = (n_g - 1) * dx
        print(f"  {n_g:2d}기 x {dx:.1f} m : 스팬 {span:.1f} m, "
              f"분해 대역 lambda in [{2*dx:.1f}, {span:.1f}] m")
    print("\n  계획 조건 중 권장안(11x0.5)이 덮지 못하는 것:")
    for tag, T in [("Phase2 #17", 0.7), ("Phase2 #19", 2.0),
                   ("Phase1 #3/#6", 1.8), ("시설 하한", 0.5)]:
        lam = 2 * math.pi / wavenumber(T)
        need_dx, need_span = lam / 2, lam
        bad = []
        if need_dx < 0.5:
            bad.append(f"dx <= {need_dx:.3f} m 필요(앨리어싱)")
        if need_span > 5.0:
            bad.append(f"스팬 >= {need_span:.2f} m 필요")
        print(f"    {tag:14s} T={T:4.2f}  lambda={lam:6.3f} m  "
              + ("; ".join(bad) if bad else "OK"))

    print()
    print("=" * 68)
    print("(C) 관측 SNR 투영  (가정 sigma_obs = %.3f m)" % SIGMA_OBS_ASSUMED)
    print("=" * 68)
    s = math.sin(math.radians(PITCH_DEG))
    sz = SIGMA_OBS_ASSUMED * s
    print(f"  sin({PITCH_DEG:.0f} deg) = {s:.4f}  =>  sigma_z = {sz:.6f} m")
    print(f"  (eta = h - r sin(theta)  =>  d eta = -sin(theta) dr; 수평 수면 주항)")
    print(f"\n  {'셀':>16} {'sigma_eta [m]':>14} {'sigma_z/sigma_eta':>18}")
    rows = []
    for H in [2, 5, 10]:                       # 규칙파, sigma_eta = H/(2 sqrt2)
        se = (H / 100) / (2 * math.sqrt(2))
        rows.append((f"규칙파 H={H} cm", se, sz / se))
    for Hs in [3, 5, 8, 12]:                   # 불규칙파, sigma_eta = Hs/4
        se = (Hs / 100) / 4
        rows.append((f"불규칙 Hs={Hs} cm", se, sz / se))
    for name, se, r in rows:
        flag = "  <-- 실측 최악(%.3f) 초과" % REAL_RATIO_MAX if r > REAL_RATIO_MAX else ""
        print(f"  {name:>16} {se:14.5f} {r:18.4f}{flag}")
    lo, hi = min(r for _, _, r in rows), max(r for _, _, r in rows)
    print(f"\n  계획 셀 투영 범위 = [{lo:.3f}, {hi:.3f}]")
    print(f"  실측 12 레코드 관측 범위 = [{REAL_RATIO_MIN:.3f}, {REAL_RATIO_MAX:.3f}]")
    print(f"  -> 유리한 쪽 끝({REAL_RATIO_MIN:.3f})에 닿는 계획 셀은 없다."
          f" 최선이 {lo:.3f} 다.")
    print(f"  -> {sum(1 for _,_,r in rows if r > REAL_RATIO_MAX)} 개 셀이 실측 최악보다 불리하다.")


if __name__ == "__main__":
    main()
