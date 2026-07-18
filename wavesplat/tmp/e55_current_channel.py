"""e55 — CURRENT (Uc) channel dose ladder. E-54 다채널 이식 조건: 5요소 회계를
regime 밖 두 번째 채널(해류)로 이식. 이식 기준 = 채널 생존이 아니라 5요소가 검정력
있게 판정 완료(E-54 Q5). prereg = tmp/E54_multichannel_transplant_draft.md §E-55.

물리: 해류 Uc가 있으면 관측 주파수가 Doppler 이동 omega_obs = omega_intrinsic(k) + k.Uc.
아레나가 z/GT를 omega_true로 렌더하고, 추정기가 Uc를 모르면(omega_intrinsic) 부정합=해류 해악.
07-05 발견: 주 해악통로는 kp/대역배치(beamform Doppler 취약)지 전파식 자체 아님 — kp_known이면
해류손실 미미(-0.002~-0.04). => value_marginal|kp_hint(kp 고정 후 외부 Uc의 잔여 가치)가
일차 위협가설. radar 등급(4-7 cm/s)서 <0.03이면 "current=value-사망 채널"(프로토콜 성공 계상).

설계 (frozen):
  ARENA build_sea_uc(si, Uc): build_sea_crosstp 사본 + omega_true=omega+k.Uc로 z/GT 렌더.
        rng 동일(si) => grade간 modes/phases/noise 동일, Uc만 다름 (within-sea paired).
        Uc=0이면 build_sea_crosstp와 동일(sanity).
  GRADES |Uc| in {0, 0.04, 0.07, 0.15} m/s (X-band 4-7, HF ~7, kill-dose 15), 방향 per-sea 무작위.
  BASE GRID  참-파티션 oracle_reg 격자 (E-52와 동일 구성) — Uc/kp 축을 격리하려 고정.
  SOLVER  solve_uc(S, kp, Uc): solve_with_grid 사본, bank_atoms om을 k.Uc로 이동 + 주어진 kp.
  ARMS (per sea, per grade):
    naive     = solve_uc(est_kp,  Uc=0)      추정 kp, Uc 무시 (배포 naive)
    kp_only   = solve_uc(true_kp, Uc=0)      kp 힌트, Uc 무시
    kp_Uc     = solve_uc(true_kp, Uc=true)   kp 힌트 + Uc 오라클
    Uc_only   = solve_uc(est_kp,  Uc=true)   Uc 오라클, 추정 kp
  CONTRASTS
    ★value_marginal|kp_hint = kp_Uc - kp_only   ← 일차 판정 (kp 고정 후 Uc 가치)
    value_raw               = Uc_only - naive   (추정 kp서 Uc 가치)
    kp_value                = kp_only - naive   (kp 힌트만의 가치)
    total                   = kp_Uc - naive
  POWER  n=48 x >=... blocks; per-grade paired CI. 규율1: SE<=0.010 목표(n=48 paired면 근접).
  DECISION (frozen)
    value_marginal|kp_hint @ (4-7 cm/s):  < 0.03 CI => "current=VALUE-사망 채널"(프로토콜 성공)
                                          >= 0.03 CI excl0 => 외부 Uc 배포가치 실재(regime 못 한 것)
    @ 15 cm/s(kill): 대조 — 여기선 커야 정합(Doppler 크면 kp고정으로도 잔여 큼).
  CAVEAT 합성 스코프; 실측 anchor(측정 Uc 주입) 미실행(regime 4중 사망 자리). Gaussian solver.

Usage: python3 tmp/e55_current_channel.py [--smoke] [--n 48]
Out:   results/direction3/e55_current_channel.json
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from wavesim.wave_generator import jonswap_components
from d3_sa_fim_identifiability import omega_of
from d3_sa_achievability_gap import (dispersion_lam, sample_obs,
                                     masks_common_unobs, N_OBS)
from e29_gennet_C import (DEVICE, DEPTH, _MASK, corr, two_stage_kp,
                          KXN_NP, KYN_NP, design_t)
from e41_regime_prize_ladder import lobe_grid
from e42_detector_v2 import GAMMA, SPREAD
from d18_relax_estimator import omega

OUT = 'results/direction3/e55_current_channel.json'
GRADES = [0.0, 0.04, 0.07, 0.15]          # |Uc| m/s
SPREAD_DEG = 30.0


def build_sea_uc(si, Ucx, Ucy):
    """build_sea_crosstp with ocean current Uc=(Ucx,Ucy) Doppler-shifting the
    observed frequency omega_obs = omega_intrinsic(k) + k.Uc. rng identical to
    build_sea_crosstp(si) so grades share modes/phases/noise/geometry."""
    rng = np.random.default_rng(9700 + 31 * si)
    Tp1 = 7.0
    Hs = rng.uniform(1.5, 3.0)
    r2 = rng.uniform(0.2, 0.5)
    d1 = rng.uniform(0, 360)
    sign = 1.0 if rng.uniform() < 0.5 else -1.0
    d2 = d1 + sign * rng.uniform(60, 150)
    Tp2 = float(Tp1 * rng.uniform(0.75, 1.4))
    c = list(jonswap_components(Hs=Hs * np.sqrt(1 - r2), Tp=Tp1, n_components=100,
                               depth=DEPTH, gamma=3.3, direction_deg=d1,
                               direction_spread_deg=SPREAD_DEG, seed=4000 + si))
    c += list(jonswap_components(Hs=Hs * np.sqrt(r2), Tp=Tp2, n_components=100,
                                depth=DEPTH, gamma=3.3, direction_deg=d2,
                                direction_spread_deg=SPREAD_DEG, seed=5000 + si))
    kx = np.array([x.kx for x in c]); ky = np.array([x.ky for x in c])
    Am = np.array([x.A for x in c]); ph = np.array([x.phi for x in c])
    om_true = omega_of(kx, ky, DEPTH) + kx * Ucx + ky * Ucy       # Doppler
    lam = dispersion_lam(Tp1, DEPTH); cp = lam / Tp1; T = 10 * Tp1
    fp = 0.18 * lam; r_fp = fp / 2
    side = 2.0 * lam
    gx = np.linspace(0, side, 96); gy = np.linspace(0, side, 96)
    Xg, Yg = np.meshgrid(gx, gy)
    mu = masks_common_unobs(gx, gy, r_fp, 0.6 * fp)
    Xs, Ys = Xg[mu], Yg[mu]
    take = np.random.default_rng(0).choice(len(Xs), min(1200, len(Xs)), False)
    Xs, Ys = Xs[take], Ys[take]
    gt_u = (Am[None] * np.cos(kx[None] * Xs[:, None] + ky[None] * Ys[:, None]
                              - om_true[None] * T + ph[None])).sum(1)   # GT w/ Uc
    Pq = np.column_stack([Xs, Ys, np.full(len(Xs), T)])
    U = rng.uniform(0.0, 1.5) * cp
    nfrac = rng.uniform(0.02, 0.09)
    P = sample_obs('fixed' if U < 1e-6 else 'moving', U, lam, cp, T, gx, gy,
                   r_fp, rng)
    psi = (kx[None] * P[:, 0:1] + ky[None] * P[:, 1:2]
           - om_true[None] * P[:, 2:3] + ph[None])
    z = (Am[None] * np.cos(psi)).sum(1) + rng.normal(0, nfrac * Hs, N_OBS)
    kp_est = two_stage_kp(P, z)
    kp_true = 2 * np.pi / lam
    return dict(P=P, z=z, Pq=Pq, gt_u=gt_u, sigma0=float(nfrac * Hs),
                kp_est=float(kp_est), kp_true=float(kp_true),
                d1=float(np.deg2rad(d1)), d2=float(np.deg2rad(d2)),
                r2=float(r2), Tp1=Tp1, Tp2=Tp2, Ucx=float(Ucx), Ucy=float(Ucy))


def oracle_grid(kp, S):
    r2 = S['r2']
    g = (1 - r2) * lobe_grid(kp, [(S['Tp1'], GAMMA, 1.0)], [(S['d1'], SPREAD, 1.0)])
    g = g + r2 * lobe_grid(kp, [(S['Tp2'], GAMMA, 1.0)], [(S['d2'], SPREAD, 1.0)])
    return g


def solve_uc(S, kp, Ucx, Ucy):
    """solve_with_grid with bank atoms Doppler-shifted by Uc and given kp."""
    g_free = oracle_grid(kp, S)
    gb = np.zeros_like(g_free)
    g = np.concatenate([g_free[_MASK], gb[_MASK]])
    if g.sum() <= 0:
        return float('nan')
    g = g / g.sum()
    kxb = KXN_NP * kp; kyb = KYN_NP * kp
    om = omega(np.hypot(kxb, kyb) + 1e-12, DEPTH) + kxb * Ucx + kyb * Ucy
    kxb2 = np.concatenate([kxb, 2 * kxb]); kyb2 = np.concatenate([kyb, 2 * kyb])
    om2 = np.concatenate([om, 2 * om])
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=DEVICE)
    Phi = design_t(t(kxb2), t(kyb2), t(om2), t(S['P']))
    Phi_q = design_t(t(kxb2), t(kyb2), t(om2), t(S['Pq']))
    gam = t(np.concatenate([g, g]) * float(np.var(S['z'])))
    K = (Phi * gam[None]) @ Phi.T + (3.0 * S['sigma0']) ** 2 * torch.eye(
        len(S['z']), device=DEVICE)
    w = Phi.T @ torch.linalg.solve(K, t(S['z']))
    return float(corr(((Phi_q * gam[None]) @ w).cpu().numpy(), S['gt_u']))


def paired(rows, key):
    d = np.array([r[key] for r in rows if np.isfinite(r[key])])
    n = len(d); se = d.std(ddof=1) / np.sqrt(n)
    lo, hi = stats.t.interval(0.95, n - 1, loc=d.mean(), scale=se)
    return dict(mean=float(d.mean()), se=float(se), lo=float(lo), hi=float(hi),
                med=float(np.median(d)), win=int((d > 0).sum()), n=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--n', type=int, default=48)
    a = ap.parse_args()
    n = 3 if a.smoke else a.n
    grades = [0.0, 0.07] if a.smoke else GRADES
    t0 = time.time()

    per_grade = {g: [] for g in grades}
    print(f"--- CURRENT channel, si=800..{800+n-1}, grades {grades} ---", flush=True)
    for si in range(800, 800 + n):
        # random current direction per sea (fixed across grades)
        ang = np.random.default_rng(5000 + si).uniform(0, 2 * np.pi)
        for gmag in grades:
            Ucx, Ucy = gmag * np.cos(ang), gmag * np.sin(ang)
            try:
                S = build_sea_uc(si, Ucx, Ucy)
                naive = solve_uc(S, S['kp_est'], 0.0, 0.0)
                kp_only = solve_uc(S, S['kp_true'], 0.0, 0.0)
                kp_Uc = solve_uc(S, S['kp_true'], Ucx, Ucy)
                Uc_only = solve_uc(S, S['kp_est'], Ucx, Ucy)
            except Exception as e:
                print(f"  si{si} g{gmag} EXCLUDED ({type(e).__name__})", flush=True)
                continue
            per_grade[gmag].append(dict(
                si=si, naive=naive, kp_only=kp_only, kp_Uc=kp_Uc, Uc_only=Uc_only,
                value_marginal=kp_Uc - kp_only, value_raw=Uc_only - naive,
                kp_value=kp_only - naive, total=kp_Uc - naive))
        if (si - 800) % 10 == 0:
            print(f"  si{si} done [{time.time()-t0:.0f}s]", flush=True)

    res = {}
    print(f"\n=== per-grade contrasts (n per grade) ===")
    print(f"{'|Uc|':>6} {'val_marg|kp':>22} {'val_raw':>18} {'kp_value':>18}")
    for g in grades:
        rows = per_grade[g]
        vm = paired(rows, 'value_marginal')
        vr = paired(rows, 'value_raw')
        kv = paired(rows, 'kp_value')
        res[f'{g}'] = dict(value_marginal=vm, value_raw=vr, kp_value=kv,
                           naive_mean=float(np.mean([r['naive'] for r in rows])))
        sm = '*' if (vm['lo'] > 0 or vm['hi'] < 0) else ' '
        print(f"{g*100:>5.0f}c {vm['mean']:+.4f}[{vm['lo']:+.4f},{vm['hi']:+.4f}]{sm}"
              f" {vr['mean']:+.4f}[{vr['lo']:+.4f},{vr['hi']:+.4f}]"
              f" {kv['mean']:+.4f}[{kv['lo']:+.4f},{kv['hi']:+.4f}] (n{vm['n']} SE{vm['se']:.4f})")

    # pre-registered verdict on value_marginal|kp_hint at radar grades 4-7 cm/s
    verdict = 'SMOKE'
    if not a.smoke:
        vm4 = res['0.04']['value_marginal']; vm7 = res['0.07']['value_marginal']
        vm15 = res['0.15']['value_marginal']
        radar_dead = (vm4['hi'] < 0.03 and vm7['hi'] < 0.03)
        radar_pays = (vm4['lo'] > 0 and vm4['mean'] >= 0.03) or (vm7['lo'] > 0 and vm7['mean'] >= 0.03)
        print(f"\n=== PRE-REGISTERED VERDICT (value_marginal|kp_hint @ radar 4-7 cm/s) ===")
        print(f"  4cm hi {vm4['hi']:+.4f} | 7cm hi {vm7['hi']:+.4f} | 15cm mean {vm15['mean']:+.4f}")
        if radar_dead:
            verdict = ("VALUE-DEATH CHANNEL (protocol SUCCESS): kp_hint 후 radar급 외부 Uc의 "
                       "잔여 가치 < 0.03 => 배포지침 'radar 말고 kp 고쳐라'. current dies at VALUE "
                       "(regime dies at achievability) = 프레임워크가 두 채널서 다른 요인 판별.")
        elif radar_pays:
            verdict = ("DEPLOYABLE SUPPLY CHANNEL: radar급 Uc가 kp 고정 후에도 >=0.03 가치 "
                       "=> regime이 못 한 '배포 가능 외부-공급 채널'의 실례.")
        else:
            verdict = "INCONCLUSIVE: report CIs (규율6). 검정력/부호 미결."
        print(f"  -> {verdict}")

    json.dump(dict(per_grade={str(g): per_grade[g] for g in grades},
                   contrasts=res, verdict=verdict, grades=grades),
              open(OUT, 'w'), indent=1, default=float)
    print(f"saved -> {OUT}  ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
