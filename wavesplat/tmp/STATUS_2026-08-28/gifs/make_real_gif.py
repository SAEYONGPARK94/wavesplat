"""실측 추정 결과 GIF — **고정 관측소** 스테레오 레코드, 배포 팔 + 제안 표현 팔.

## 자료·팔·채점 (전부 캡션에 박는다)
- 자료 = `data/real_bergamasco/` 의 **12 독립 레코드**(고정 관측소 4곳, **스테레오** 재구성면).
  ⛔*"선박 탑재"* 도 ⛔*"LiDAR"* 도 아니다 — 둘 다 실증 0건이다.
  장면 빌더 = `tmp/B2_REAL_2026-08-23/scene_station.py`(팬 없음, 유효 격자 균등추출, m_cap=600).
- 레코드 선택 = **결과에서 고르지 않는다**: `gates_b2.LARGE_BOX`(결과 전에 기하로 선언된 5개,
  4 m 질의 격자가 장으로 렌더되는 유일한 층) 중 배포 팔 nowcast / **참 제로필드** 비의 **중앙**.
  ⟹ `YS17`. ⚠최고(AA14 0.424)도 최악(LJ18a 1.415)도 아니다.
  ★대조로 **최악 레코드 `LJ18a`** 도 같은 규약으로 따로 낸다(`--record LJ18a`).
- 팔 두 개:
  · `SHIP` = 배포 기본값 그대로 ⟹ **실측 게이트가 `bank_mmse` 로 보낸다**(제안 표현 아님).
  · `PROP` = `real_domain_gate=False, init_method='learned'` ⟹ **제안 표현(WavePrior) 강제**.
- 채점 = **`nRMSE_σ = RMSE / σ`**, σ = 40 s 기억창의 표준편차(실측 규약).
  ⛔합성의 `RMSE/Hs` 와 **직접 비교 금지**(`metrics.py:32` 의 cross-domain trap, 약 4배 차이).
  기준선 = ★**참 제로필드** — 레코드·lead 마다 같은 채점기에 `Z=0` 을 통과시켜 산출한다.
  ⛔`ZERO_REF = 1.0` 은 **정의상 참이 아니다**(2026-08-23 `AMENDMENT_ZEROREF.md`).

## ⛔이 GIF 로 주장할 수 없는 것
- 외부 기준선이 **0개**다. *"경쟁자보다 낫다/못하다"* 를 주장하지 않는다.
- 한 레코드다. 사이트 수준 일반화 불가(사이트 4곳, 58%가 한 사이트).
- `B-2` 동결 배치는 **수심 17(오기) 구성**으로 돌았다. 이 스크립트는 **정정된 수심 30**
  (`scene_station.py` 현행)으로 돌므로 `B-2` 표의 수와 **비트동일하지 않다**.
"""
import argparse
import json
import os
import statistics as st
import sys
import warnings

warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = ''          # ⛔GPU 금지 (M1 생성이 점유 중)
os.environ.setdefault('OMP_NUM_THREADS', '4')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib import font_manager                              # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))     # .../wavesplat/wavesplat
B2 = os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23')
for p in (ROOT, os.path.join(ROOT, 'scripts'), B2,
          os.path.join(ROOT, 'tmp', 'bankmmse_port_2026-07-29')):
    sys.path.insert(0, p)
os.chdir(ROOT)

for _f in ('/home/vexit/.local/share/fonts/malgun.ttf',):
    if os.path.exists(_f):
        font_manager.fontManager.addfont(_f)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=_f).get_name()
plt.rcParams['axes.unicode_minus'] = False

N_GIF_FRAMES, FPS, DPI = 56, 7, 78
SITE = {'YS17': 'Yellow Sea 2017-05-13 (고정 관측소)',
        'LJ18a': 'La Jument 2018-01-03 (고정 관측소)',
        'AA14': 'Acqua Alta 2014-03-27 (고정 관측소)',
        'AA15a': 'Acqua Alta 2015-03-05 (고정 관측소)',
        'AA15b': 'Acqua Alta 2015-05-15 (고정 관측소)'}


def pick_record():
    """결과 전 선언된 LARGE_BOX 층에서 배포 팔 skill 비의 **중앙** 레코드."""
    import gates_b2 as G
    tz = json.load(open(os.path.join(B2, 'true_zero_field.json')))
    rows = [json.loads(l) for l in open(os.path.join(B2, 'results.jsonl'))]
    r = {x['record']: x['nowcast'] / tz[x['record']]
         for x in rows if x['cell'] == 'SHIP' and 'error' not in x
         and x['record'] in G.LARGE_BOX}
    m = st.median(r.values())
    return sorted(r, key=lambda k: (abs(r[k] - m), k))[0], r, m


def dense_gt(record, win, lm, sq_leads):
    """`scene_station` 과 **같은 호출 사슬**로 레코드 원래 프레임률의 GT 를 낸다.
    반환값이 protocol 의 `maps_gt` 를 lead 시각에서 재현하는지 호출부가 assert 한다."""
    from d3_sa_multirecord import MRecord
    from scene_station import RECORDS_12
    cfg = RECORDS_12[record]
    rec = MRecord(os.path.join(ROOT, cfg['path']), stride=cfg['stride'])
    nfw = int(round(40.0 / rec.dt))
    n_lead = int(round(lm / rec.dt))
    lo, hi = nfw + 1, rec.T_all - n_lead - 2
    i0 = int(np.linspace(lo, hi, 6).astype(int)[win])
    Zfut = rec.window(i0, n_lead + 1)
    sq = max(1, int(round(4.0 / rec.dx)))
    return Zfut[:, ::sq, ::sq], float(rec.dt)


def build(record, win, arm_cfg, seed=42):
    from wavesplat.optimizer.sequential_update import make_updater
    from wavesplat.run import DEFAULT_CONFIG, set_seed
    from scene_station import scene_station
    import make_gif_bank_deployed as M
    sc = scene_station(record, win=win)
    t_shift = float(np.asarray(sc['P'])[:, 2].min())
    t0 = float(sc['Pq'][0, 2]) - t_shift
    frames = M.frames_from_scene(sc, t_shift, dtype=torch.float32)
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(depth=float(sc['depth']), seed=seed, device='cpu')
    cfg.update(arm_cfg)
    set_seed(seed)
    upd = make_updater(cfg)
    upd.initialize(frames,
                   x_range=(float(sc['gxq'].min()), float(sc['gxq'].max())),
                   y_range=(float(sc['gyq'].min()), float(sc['gyq'].max())),
                   verbose=False)
    return sc, upd, t0, cfg


def render(upd, sc, t0, times):
    gxq = torch.tensor(np.asarray(sc['gxq']), dtype=torch.float32)
    gyq = torch.tensor(np.asarray(sc['gyq']), dtype=torch.float32)
    out = []
    for tt in times:
        Z = upd.predict_field(gxq, gyq, t0 + float(tt)).cpu().numpy()
        out.append(Z.reshape(len(sc['gyq']), len(sc['gxq'])))
    return np.array(out)


def score(pred, gt, sg):
    m = np.isfinite(gt)
    if not m.any():
        return float('nan')
    return float(np.sqrt(np.mean((pred[m] - gt[m]) ** 2)) / sg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--record', default='')
    ap.add_argument('--win', type=int, default=3)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    med_rec, ratios, med = pick_record()
    record = a.record or med_rec
    tag = a.tag or ('median' if record == med_rec else 'other')
    out_gif = os.path.join(HERE, f'real_station_{record}_{tag}.gif')
    out_json = out_gif[:-4] + '.json'
    print(f'record {record} (LARGE_BOX skill ratios '
          f'{ {k: round(v, 3) for k, v in sorted(ratios.items(), key=lambda kv: kv[1])} }, '
          f'median {med:.3f})', flush=True)

    sc, upd_s, t0, cfg_s = build(record, a.win, {})                       # 배포
    _, upd_p, _, cfg_p = build(record, a.win,
                               {'real_domain_gate': False,
                                'init_method': 'learned'})                # 제안 표현
    leads = np.asarray(sc['leads'], float)
    sg = float(sc['sg'])
    G, dt = dense_gt(record, a.win, float(leads[-1]), None)
    for li, L in enumerate(leads):                # ★protocol 재현 assert (repo 관례)
        j = int(round(float(L) / dt))
        assert np.allclose(G[j], sc['maps_gt'][li], equal_nan=True), li
    step = max(1, int(round(G.shape[0] / N_GIF_FRAMES)))
    jj = list(range(0, G.shape[0], step))
    times = [j * dt for j in jj]

    Ps = render(upd_s, sc, t0, times)
    Pp = render(upd_p, sc, t0, times)
    ns = np.array([score(Ps[k], G[j], sg) for k, j in enumerate(jj)])
    npp = np.array([score(Pp[k], G[j], sg) for k, j in enumerate(jj)])
    nz = np.array([score(np.zeros_like(G[j]), G[j], sg) for j in jj])     # ★참 제로필드
    # lead 그리드에서의 사전등록 값(대조용)
    Ls = render(upd_s, sc, t0, leads)
    Lp = render(upd_p, sc, t0, leads)
    ls = np.array([score(Ls[i], sc['maps_gt'][i], sg) for i in range(len(leads))])
    lp = np.array([score(Lp[i], sc['maps_gt'][i], sg) for i in range(len(leads))])
    lz = np.array([score(np.zeros_like(sc['maps_gt'][i]), sc['maps_gt'][i], sg)
                   for i in range(len(leads))])
    tzf = json.load(open(os.path.join(B2, 'true_zero_field.json')))
    print(f'  nowcast  SHIP {ls[0]:.4f} | PROP {lp[0]:.4f} | 참 제로필드 {lz[0]:.4f} '
          f'(B-2 동결표 {tzf[record]:.4f}, 수심 정정으로 다를 수 있다)', flush=True)
    print(f'  lead 평균 SHIP {np.nanmean(ls):.4f} | PROP {np.nanmean(lp):.4f} | '
          f'제로 {np.nanmean(lz):.4f}', flush=True)

    gx, gy = np.asarray(sc['gxq']), np.asarray(sc['gyq'])
    ext = [gx[0], gx[-1], gy[0], gy[-1]]
    lim = float(np.nanpercentile(np.abs(G), 99.0))
    elim = lim
    # ⚠`make_gif_bank_deployed` 가 import 시점에 rcParams['font.family'] 를 덮는다 —
    #   한글이 두부로 나온다. 그리기 직전에 다시 건다.
    plt.rcParams['font.family'] = font_manager.FontProperties(
        fname='/home/vexit/.local/share/fonts/malgun.ttf').get_name()
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure(figsize=(16.6, 4.3), facecolor='white')
    gs = fig.add_gridspec(1, 5, left=.032, right=.988, top=.685, bottom=.205,
                          wspace=.30, width_ratios=[1, 1, 1, 1, 1.30])
    ax = [fig.add_subplot(g) for g in gs]
    kw = dict(extent=ext, origin='lower', aspect='equal', interpolation='nearest')
    im0 = ax[0].imshow(G[0], cmap='RdBu_r', vmin=-lim, vmax=lim, **kw)
    im1 = ax[1].imshow(Ps[0], cmap='RdBu_r', vmin=-lim, vmax=lim, **kw)
    im2 = ax[2].imshow(Ps[0] - G[0], cmap='PuOr_r', vmin=-elim, vmax=elim, **kw)
    im3 = ax[3].imshow(Pp[0], cmap='RdBu_r', vmin=-lim, vmax=lim, **kw)
    for a_ in ax[:4]:
        a_.set_xlabel('x [m]', fontsize=8)
        a_.tick_params(labelsize=6.5)
    ax[0].set_ylabel('y [m]', fontsize=8)
    fig.colorbar(im1, ax=ax[1], fraction=.036, pad=.02).ax.tick_params(labelsize=6)
    fig.colorbar(im2, ax=ax[2], fraction=.036, pad=.02).ax.tick_params(labelsize=6)

    ax[4].plot(times, nz, color='#8a8a8a', lw=1.6,
               label='참 제로필드 (Z=0 을 같은 채점기에)')
    ax[4].plot(times, ns, color='#1b5e9e', lw=1.8, label='배포 팔 (게이트 -> bank_mmse)')
    ax[4].plot(times, npp, color='#c1440e', lw=1.8, label='제안 표현 강제 (WavePrior)')
    mk, = ax[4].plot([times[0]], [ns[0]], 'o', color='#1b5e9e', ms=5.5)
    mk2, = ax[4].plot([times[0]], [npp[0]], 'o', color='#c1440e', ms=5.5)
    ax[4].set_ylim(0, float(np.nanmax([nz.max(), ns.max(), npp.max()])) * 1.42)
    ax[4].set_xlabel('예보 시간 lead [s]   (t=0 이후 새 관측 없음)', fontsize=8)
    ax[4].set_ylabel('nRMSE_sigma  (RMSE / sigma)', fontsize=8)
    ax[4].set_title('채점: 실측 규약 = RMSE / sigma', fontsize=9.5, pad=4)
    ax[4].tick_params(labelsize=7)
    ax[4].legend(fontsize=6.4, loc='upper left', framealpha=.95)
    ax[4].grid(alpha=.25, lw=.5)

    fig.text(.032, .965, '', fontsize=11.0, ha='left', va='top').set_text(
        f'실측 (고정 관측소 · 스테레오 재구성면)  [{record}]   -   lead 0-14 s 평균 nRMSE_sigma:  '
        f'배포 팔 {np.nanmean(ls):.2f}  ·  제안 표현 강제 {np.nanmean(lp):.2f}  ·  '
        f'참 제로필드 {np.nanmean(lz):.2f}')
    fig.text(.032, .905, '', fontsize=7.4, ha='left', va='top', color='#333333').set_text(
        f'자료: {SITE.get(record, record)} · Bergamasco 스테레오 12 레코드 중 하나.  '
        f'선박 탑재 아님 · LiDAR 아님 (둘 다 실증 0건).  '
        f'관측 = 40 s 기억창에서 균등추출 {int(len(sc["z"]))}점 + N(0, 0.02 m) 잡음, 전방팬 없음.  '
        f'질의 격자 {len(gy)}x{len(gx)} (4 m).  Hs={float(sc["Hs"]):.2f} m, sigma={sg:.3f} m, '
        f'depth={float(sc["depth"]):.0f} m, win={a.win}\n'
        f'팔: (파랑) 배포 기본값 DEFAULT_CONFIG - 실측 게이트가 bank_mmse 로 보낸다 = '
        f'제안 표현이 아니다.   (주황) real_domain_gate=False, init_method=learned = 제안 표현 강제.  '
        f'추정 시드 42, CPU.   세 지도는 같은 색척도.\n'
        f'채점: nRMSE_sigma = RMSE / sigma (실측 규약).  합성의 RMSE/Hs 와 직접 비교 금지 - '
        f'같은 물리 오차에서 약 4배 차이다 (metrics.py:32).  '
        f'기준선 = 참 제로필드: 상수 1.0 이 아니라 lead 마다 Z=0 을 채점기에 통과시킨 값.')
    fig.text(.032, .008, '', fontsize=6.3, ha='left', va='bottom', color='#666666').set_text(
        f'선택 사유: 결과 전에 기하로 선언된 LARGE_BOX 층 5개(4 m 질의 격자가 장으로 렌더되는 유일한 층) 중 '
        f'배포 팔 nowcast/참제로 비의 중앙 = {med_rec}.  '
        f'비: ' + ', '.join(f'{k} {v:.2f}' for k, v in sorted(ratios.items(), key=lambda kv: kv[1])) +
        f'.  최고도 최악도 아니다.\n'
        f'12 레코드 전체(B-2, 참 제로필드 기준, 수심 정정 후; 재현 = tmp/B2_REAL_2026-08-23/depth_recheck.py, 런 0회): '
        f'배포 팔이 참 제로필드를 이기는 것은 nowcast 10/12, p=0.0193 - 16팔 Holm 보정 후 생존 0.  '
        f'제안 표현 강제는 7/12, p=0.387 = 무기술.  짝 비교로는 배포 팔이 제안 표현을 10/12 로 이긴다.\n'
        f'주장할 수 없는 것: 외부 기준선이 0개다 - "경쟁자보다 낫다/못하다" 가 아니다.  '
        f'한 레코드이므로 사이트 수준 일반화 불가(사이트 4곳, 58%가 한 사이트).  '
        f'B-2 동결 배치는 BS 레코드 수심 17(오기)로 돌았다 - 이 그림은 현행 코드(30)다.')

    def upd_f(k):
        im0.set_data(G[jj[k]])
        im1.set_data(Ps[k])
        im2.set_data(Ps[k] - G[jj[k]])
        im3.set_data(Pp[k])
        mk.set_data([times[k]], [ns[k]])
        mk2.set_data([times[k]], [npp[k]])
        ax[0].set_title(f'참값 = 스테레오 재구성면   lead={times[k]:.1f} s\n'
                        f'참 제로필드 nRMSE={nz[k]:.3f}', fontsize=9.3, pad=4)
        ax[1].set_title(f'배포 팔 (게이트 -> bank_mmse)\nnRMSE={ns[k]:.3f}',
                        fontsize=9.3, pad=4)
        ax[2].set_title(f'배포 팔 오차 = 추정 - 참값\n', fontsize=9.3, pad=4)
        ax[3].set_title(f'제안 표현 강제 (WavePrior)\nnRMSE={npp[k]:.3f}',
                        fontsize=9.3, pad=4)
        return im0, im1, im2, im3, mk, mk2

    ani = FuncAnimation(fig, upd_f, frames=len(jj), blit=False)
    ani.save(out_gif, writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)
    json.dump(dict(record=record, win=a.win, site=SITE.get(record, record),
                   selection='median of pre-declared LARGE_BOX by SHIP nowcast / true-zero',
                   large_box_ratios=ratios, large_box_median=med,
                   n_obs=int(len(sc['z'])), npx=int(sc['npx']),
                   Hs=float(sc['Hs']), sigma=sg, depth=float(sc['depth']),
                   leads=list(map(float, leads)),
                   ship_lead_curve=list(map(float, ls)),
                   prop_lead_curve=list(map(float, lp)),
                   true_zero_lead_curve=list(map(float, lz)),
                   true_zero_nowcast_b2_frozen=tzf[record],
                   scoring='RMSE / sigma (real convention)',
                   ship_effective=dict(
                       varpro_steps=cfg_s.get('varpro_steps'),
                       learned_init_iters=cfg_s.get('learned_init_iters'),
                       auto_switch_thresh=cfg_s.get('auto_switch_thresh'),
                       init_method=cfg_s.get('init_method'),
                       real_domain_gate=cfg_s.get('real_domain_gate')),
                   prop_effective=dict(init_method=cfg_p.get('init_method'),
                                       real_domain_gate=cfg_p.get('real_domain_gate')),
                   n_gif_frames=len(jj), dt_dense=dt),
              open(out_json, 'w'), indent=1)
    print('wrote', out_gif, os.path.getsize(out_gif) // 1024, 'KB')


if __name__ == '__main__':
    main()
