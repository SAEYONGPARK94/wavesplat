"""S2 — 제어 물리량 alpha_wave = -slope - w/U 의 GT 계산과 nRMSE 조립.

`PREREG.md` §3 "alpha_wave 계산 규약" 의 구현. 규약은 채점 전 동결됐다.

핵심 규약 하나만 반복한다: ★**GT 와 추정에 동일한 중심차분 스텐실을 쓴다.**
2026-07-09 캐비엇(`docs/REQUIRED_HORIZON_persona_debate_2026-07-09.md`)이 지적한 실패는
"해석적 GT 미분 vs 평활화된 추정"을 비교한 것이었다. 여기서는 두 팔 모두
공간 ±2.0 m · 시간 ±0.05 s 스텐실을 통과한다.

GT 성분 배열 `config/wave_components` 열 규약(= `run_rx2b.spectral_accuracy` 와 동일):
    [:,0]=A  [:,1]=omega  [:,2]=kx  [:,3]=ky  [:,4]=phi
    eta(x,y,t) = sum_i A_i cos(kx_i x + ky_i y - omega_i t + phi_i)

⚠**PREREG §3 의 정규화 문구 해소**: "에피소드별 GT alpha_wave 의 표준편차(전 프레임·양 포일 풀)"
가 리드를 풀에 넣는지 명시하지 않는다. 규약이 판정을 만드는 사고(S1 에서 실제로 겪었다)를
피하기 위해 **두 해석을 모두 계산해 둘 다 출력**한다:
    norm='episode' : 프레임 x 포일 x 리드 전부 풀 -> 에피소드당 1개  (**PRIMARY**)
    norm='lead'    : 프레임 x 포일 풀, 리드별      -> 에피소드 x 리드
"""
import numpy as np

FOILS = (2.0, -2.5)                      # hydrofoil/config.py:67,79
LEADS = (0.1, 0.2, 0.3, 0.5)             # docs/STUDY_GUIDE_2026-07-06.md:135
DX, DT = 2.0, 0.05
U_DESIGN = 11.3                          # reference_agreed_operating_point


def gt_eta(comp, x, t):
    """eta(x, y=0, t). x, t 는 스칼라 또는 같은 shape 배열."""
    A, om, kx, _ky, phi = comp[:, 0], comp[:, 1], comp[:, 2], comp[:, 3], comp[:, 4]
    x = np.atleast_1d(np.asarray(x, float))
    t = np.atleast_1d(np.asarray(t, float))
    ph = kx[None, :] * x[:, None] - om[None, :] * t[:, None] + phi[None, :]
    return (A[None, :] * np.cos(ph)).sum(axis=1)


def gt_alpha_exact(comp, x, t, U=U_DESIGN):
    """해석적 미분 (스텐실 오차 측정용 기준)."""
    A, om, kx, _ky, phi = comp[:, 0], comp[:, 1], comp[:, 2], comp[:, 3], comp[:, 4]
    x = np.atleast_1d(np.asarray(x, float))
    t = np.atleast_1d(np.asarray(t, float))
    ph = kx[None, :] * x[:, None] - om[None, :] * t[:, None] + phi[None, :]
    slope = (-A[None, :] * kx[None, :] * np.sin(ph)).sum(axis=1)
    w = (A[None, :] * om[None, :] * np.sin(ph)).sum(axis=1)
    return -slope - w / U


def alpha_from_legs(legs, U=U_DESIGN, dx=DX, dt=DT):
    """저장된 4개 스텐실 다리 [x-dx, x+dx, t-dt, t+dt] -> alpha_wave."""
    e_xm, e_xp, e_tm, e_tp = legs
    slope = (e_xp - e_xm) / (2.0 * dx)
    w = (e_tp - e_tm) / (2.0 * dt)
    return -slope - w / U


def gt_alpha_stencil(comp, xp, t, U=U_DESIGN, dx=DX, dt=DT):
    """추정과 **동일한** 스텐실을 GT 에 적용."""
    return alpha_from_legs((gt_eta(comp, xp - dx, t), gt_eta(comp, xp + dx, t),
                            gt_eta(comp, xp, t - dt), gt_eta(comp, xp, t + dt)), U, dx, dt)


def episode_alpha(frames, comp, U=U_DESIGN):
    """한 에피소드의 alpha_wave nRMSE.

    frames : run_estimation 의 frame_results 중 'alpha_legs' 와 't' 를 가진 것들
    comp   : (N,6) GT 성분
    반환   : dict(per_lead=..., episode=..., stencil_err=...)
    """
    est, gt, keys = {}, {}, []
    ts = np.array([f['t'] for f in frames], float)
    for xf in FOILS:
        for L in LEADS:
            k = f'{xf}|{L}'
            keys.append((xf, L, k))
            legs = np.array([f['alpha_legs'][k] for f in frames], float)   # (F,4)
            est[k] = alpha_from_legs([legs[:, 0], legs[:, 1], legs[:, 2], legs[:, 3]], U)
            xp = xf + U * L
            gt[k] = gt_alpha_stencil(comp, xp, ts, U)

    # --- 스텐실 자체 오차 (PREREG §4.10): 동일 GT 를 해석 미분 vs 스텐실로 ---
    ex, sx = [], []
    for xf, L, k in keys:
        ex.append(gt_alpha_exact(comp, xf + U * L, ts, U))
        sx.append(gt[k])
    ex, sx = np.concatenate(ex), np.concatenate(sx)
    stencil_err = float(np.sqrt(np.mean((sx - ex) ** 2)) / max(np.std(ex), 1e-12))

    all_gt = np.concatenate([gt[k] for _, _, k in keys])
    norm_ep = float(np.std(all_gt))

    out_lead, out_ep = {}, {}
    for L in LEADS:
        e = np.concatenate([est[f'{xf}|{L}'] for xf in FOILS])
        g = np.concatenate([gt[f'{xf}|{L}'] for xf in FOILS])
        rmse = float(np.sqrt(np.mean((e - g) ** 2)))
        out_lead[str(L)] = rmse / max(float(np.std(g)), 1e-12)   # norm='lead'
        out_ep[str(L)] = rmse / max(norm_ep, 1e-12)              # norm='episode' (PRIMARY)
    return dict(episode=out_ep, per_lead=out_lead,
                stencil_err=stencil_err, norm_episode=norm_ep)
