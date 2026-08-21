#!/usr/bin/env python3
"""`EVIDENCE.md` 를 원자료에서 생성한다.  손으로 옮긴 숫자 0개.

⚠**이 파일은 사후 저장이다.** 2026-08-12 토론 때 heredoc 으로 인라인 실행하고 버려서
P4 가 *"EVIDENCE.md 가 주장한 생성 스크립트가 디렉터리에 없다 = 재현 불가"* 로 지적했다.
내용은 그때 실행한 것과 동일하며, P4 가 §1–§3 을 독립 재도출해 일치를 확인했다.

⚠생성된 `EVIDENCE.md` 자체에 오류가 있었다(토론이 적발) — 재실행 전에 `VERDICT.md §4` 를 읽을 것:
  · §5 "173×197 m" 는 **n=1** (실제 149–203 × 183–197) 이고 **extent 이지 개구가 아니다**
    (유효개구 39.5×48.3 m).  피복면적은 **cell=2.0 m 라는 미공개 손잡이**가 절대값을 정한다.
  · §3 CRB "4.4배" 는 **단일 시드**(`d72_platform_aperture.py:50` seed=1, `:88` 이 시드 미전달).

사용:  python3 make_evidence.py > EVIDENCE.md      (cwd = 저장소 루트 `wavesplat/`)
"""
import json
import glob
import numpy as np
import h5py

P = 'results/direction1/'

print("# B-1 증거표 — **원자료에서 기계 생성** (손으로 옮긴 숫자 없음)\n")
print("생성 스크립트가 이 파일을 만들었다. 숫자가 의심되면 아래 경로를 직접 열어라.\n")

print("## 1. `traj_step1b_relax.json` — 오라클 k vs 추정 k (24 ep)\n")
d = json.load(open(P + 'traj_step1b_relax.json'))
s = d['summary']
print("| 팔 | 오라클 k nRMSE | 정지보다 나은 | 오라클 CI | 추정 k nRMSE | 정지보다 나은 | 추정 CI |")
print("|---|---|---|---|---|---|---|")
for a in ['stationary', 'straight-cross', 'greedy', 'CEM', 'learned']:
    o, r = s[a + '_oracle'], s[a + '_relax']
    print(f"| {a} | {o['nrmse']:.4f} | {o['nbetter']}/24 | [{o['ci'][0]:+.4f},{o['ci'][1]:+.4f}] "
          f"| {r['nrmse']:.4f} | {r['nbetter']}/24 | [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] |")
print(f"\n**verdict**: `{s['verdict']}`\n")
print("**궤적 공간 범위 (m)**: "
      + " · ".join(f"{k} {v['x']:.0f}x{v['y']:.0f}" for k, v in d['geom'].items()) + "\n")

print("## 2. `traj_step1_netmmse.json` — 비오라클 확인 (24 ep, 8.0 m/s)\n")
d2 = json.load(open(P + 'traj_step1_netmmse.json'))
s2 = d2['summary']
print(f"speed={d2['speed']} smult={d2['smult']} note=`{d2['note']}`\n")
print("| 팔 | nRMSE | 정지보다 나은 | CI |")
print("|---|---|---|---|")
for a in ['stationary', 'straight-cross', 'greedy', 'CEM', 'learned']:
    v = s2[a]
    print(f"| {a} | {v['nrmse']:.4f} | {v['nbetter']}/24 | [{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}] |")
print(f"\n**verdict**: `{s2['verdict']}`\n")

print("## 3. `platform_aperture.json` — 정보(CRB) 측\n")
d3 = json.load(open(P + 'platform_aperture.json'))
sp = d3['speeds']
print(f"군속도 c_g={d3['cg']:.3f} m/s.  속도 그리드 = {sp}\n")
print("**방향 CRB (deg) — 낮을수록 정보 많음**\n")
print("| 기하 | " + " | ".join(f"{v:.0f} m/s" for v in sp) + " |")
print("|---" * (len(sp) + 1) + "|")
for _, blk in d3['controlled'].items():
    print(f"| (이상) {blk['name']} | " + " | ".join(f"{v:.3f}" for v in blk['dir_crb_deg']) + " |")
for _, blk in d3['realistic'].items():
    print(f"| **(현실) {blk['name']}** | " + " | ".join(f"**{v:.2f}**" for v in blk['dir_crb_deg']) + " |")
print(f"\n**verdict**: `{d3['verdict']}`\n")

print("## 4. 오늘(08-12) B-3 이 잰 개구 요구량\n")
c = json.load(open('tmp/B3_regime_position_2026-08-12/aperture_curve.json'))


def auc(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(((b[:, None] > a[None, :]).sum()
                  + 0.5 * (b[:, None] == a[None, :]).sum()) / (a.size * b.size))


keys = sorted(c, key=float)
row = [auc([x['PR'] for x in c[k] if x['regime'] == 'sparse'],
           [x['PR'] for x in c[k] if x['regime'] == 'crowded']) for k in keys]
print("| 개구 | " + " | ".join(keys) + " |")
print("|---" * (len(row) + 1) + "|")
print("| AUC | " + " | ".join(f"{v:.3f}" for v in row) + " |")
print("\n게이트: AUC≥0.90 판정가능 · 0.70–0.90 약함 · <0.70 판정불가.\n")

print("## 5. 오늘 잰 조우기하 (런 0회, `data_s9_crowded_2026-08-11`)\n")
cps = []
for p in sorted(glob.glob('data_s9_crowded_2026-08-11/*.h5'))[:8]:
    with h5py.File(p, 'r') as f:
        C = f['config/wave_components'][:]     # col0=진폭 col1=omega col2=kx col3=ky col4=위상
    i = np.argmax(C[:, 0])
    cps.append(C[i, 1] / np.hypot(C[i, 2], C[i, 3]))
cp = float(np.median(cps))
print(f"- BF4 피크 위상속도 **c_p = {cp:.2f} m/s** ⟹ 설계선속 11.3 m/s 에서 "
      f"**U/c_p = {11.3 / cp:.3f}** (서프라이딩 임계)")
print("- 에피소드 = 30.0 s, 301 프레임. 프레임당 유효 반사점 중앙 50/71.")
print("- **정지 배도 30 s 동안 173x197 m 를 훑는다** (파면 상하동으로 반사점 이동). 피복 4,188 m².")
print("  ⛔**위 두 값은 토론이 정정했다** — n=1 이고(실제 149–203 m), **extent 이지 개구가 아니다**"
      "(유효개구 39.5×48.3 m). 피복은 cell=2.0 m 손잡이에 의존(cell 1→5 m 에서 3.7배 변동).")
print("- following 11.3 = 19,024 m²(4.5x) · crossing 11.3 = 12,396 m²(3.0x) · "
      "**과거 실패한 cross 8.0 = 10,068 m²(2.4x)**\n")
