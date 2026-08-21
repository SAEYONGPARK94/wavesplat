"""AUDIT-C 부속 — `platform_aperture.json` 의 CRB 헤드라인 시드 복제.

발표값: 정지 9.61° -> cross-track 10 m/s 2.16° = **4.4배**, verdict `CONFIRMED`.
원 스크립트 `scripts/d72_platform_aperture.py` 의 `realistic_sea(...)` 는 **seed=1 고정**이고
전 속도가 같은 시드를 쓴다 ⟹ 표는 **바다 1개 실현**이다. 여기서 시드 1..30 으로 복제한다.
"""
import os, sys, math, json
import numpy as np
ROOT = '/home/vexit/wavesplat/wavesplat'
os.chdir(ROOT); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import torch
from d1_crb import DEVICE, DT, fisher, direction_crb_deg
SIGMA = 0.02; DEPTH = 50.0


def realistic_sea(speed, heading_deg, N=8, M=600, L=60.0, T_obs=20.0, seed=1):
    rng = np.random.default_rng(seed)
    ang = np.radians(rng.uniform(-25, 25, N)); kmag = rng.uniform(0.04, 0.12, N)
    A = 0.2 * np.exp(-rng.uniform(0, 1.2, N))
    theta = torch.tensor(np.stack([A, kmag*np.cos(ang), kmag*np.sin(ang), rng.uniform(0, 6.28, N)], 1),
                         dtype=DT, device=DEVICE)
    x0 = rng.uniform(0, L, M); y0 = rng.uniform(-L/2, L/2, M); t = rng.uniform(0, T_obs, M)
    vx = speed*math.cos(math.radians(heading_deg)); vy = speed*math.sin(math.radians(heading_deg))
    x = torch.tensor(x0+vx*t, dtype=DT, device=DEVICE)
    y = torch.tensor(y0+vy*t, dtype=DT, device=DEVICE)
    tt = torch.tensor(t, dtype=DT, device=DEVICE)
    F = fisher(theta, x, y, tt, DEPTH, SIGMA); n = F.shape[0]
    Finv = torch.linalg.inv(F + 1e-10*torch.trace(F)/n*torch.eye(n, dtype=DT, device=F.device))
    return float(np.median(direction_crb_deg(theta, Finv, N)))


if __name__ == '__main__':
    SEEDS = list(range(1, 31))
    rows = []
    for s in SEEDS:
        c0 = realistic_sea(0.0, 90, seed=s)
        c10 = realistic_sea(10.0, 90, seed=s)
        c5 = realistic_sea(5.0, 90, seed=s)
        a5 = realistic_sea(5.0, 0, seed=s)
        rows.append(dict(seed=s, c0=c0, c5=c5, c10=c10, a5=a5,
                         r10=c0/c10, r5=c0/c5,
                         verdict=bool(c5 < 0.6*c0 and a5 > 0.9*c0)))
    for r in rows[:8]:
        print(f"  seed{r['seed']:>3}  c0 {r['c0']:7.2f}  c5 {r['c5']:6.2f}  c10 {r['c10']:6.2f}  "
              f"x10 {r['r10']:5.2f}  verdict {'FIRE' if r['verdict'] else '  . '}")
    import statistics as st
    for k in ('c0', 'c5', 'c10', 'r10', 'r5'):
        v = [r[k] for r in rows]
        print(f"  {k:>4}: median {st.median(v):8.3f}  [{min(v):.3f}, {max(v):.3f}]")
    for n in (8, 30):
        sub = rows[:n]
        print(f"  seeds 1-{n}: median c0 {st.median([r['c0'] for r in sub]):.2f} -> "
              f"c10 {st.median([r['c10'] for r in sub]):.2f} = "
              f"x{st.median([r['c0'] for r in sub])/st.median([r['c10'] for r in sub]):.2f}  |  "
              f"median ratio x{st.median([r['r10'] for r in sub]):.2f}  |  "
              f"verdict fires {sum(r['verdict'] for r in sub)}/{n}")
    json.dump(rows, open('/home/vexit/wavesplat/wavesplat/tmp/AUDIT_full_2026-08-12/crb_seeds.json', 'w'), indent=1)
