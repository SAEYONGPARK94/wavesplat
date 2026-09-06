#!/usr/bin/env python3
"""★**실측 개구**를 배포 체크포인트의 **학습 지지집합**과 나란히 놓는다.

⛔**판정 도구가 아니다.** 추정기 런 0회 · GPU 0회 · torch 0회.
어떤 배치의 verdict 도 바꾸지 않고, 동결 파일은 **읽기만** 한다.
⛔`run_all.sh` 에 등록하지 않는다(donor 적재 + 장면 표집으로 수 분 걸린다).

무엇을 재는가
-------------
`derive_train_support.py`(2026-09-05)가 배포 사전 `beamform_cond_diffusion_mixed_v1.pt` 의
학습 개구를 **[72.3, 238.8] m** 로 재도출했다. 이 스크립트는 그 분포를 **다시 뽑고**
(⛔손복사 금지 — 임포트해서 같은 함수로 표집한다) 실측 12 레코드의 개구를 그 위에 얹는다.

실측 개구의 출처는 **동결된 `T17` 결과 행**이다:
    `s_raw_ap` = `scale_rules_t14.s_ap_raw` = `max(ptp x, ptp y) / L_TRAIN`
⟹ `L_obs = s_raw_ap * L_TRAIN` 이고 `L_TRAIN` 은 **임포트**한다(손복사 0).
`x_scale == 1.0` 행만 쓴다 — 재척도된 팔의 기하는 그 레코드의 것이 아니다.

정직한 한계
-----------
- **런 0회이므로 이것은 개입이 아니라 「범위 대조」다.** 겹치지 않는다는 사실만으로
  *"개구가 원인이다"* 를 말할 수 없다. 인과 주장은 개입(재척도)이 실측에서 **라벨을 낼 때**만
  가능하고, `T17` 은 `BLOCKED` 로 끝났다(`AMENDMENT_2_POSTRESULT.md`).
- 학습 지지집합은 **분포의 추정**이다(학습 스트림 96만 장면의 비트 재현이 아니다).
- 실측 `k̂_p`(`kp2_hat`)는 레코드 여럿에서 **2단 빔포머 코너**에 닿는다
  (`T17/AMENDMENT_2_POSTRESULT.md` C1). ⟹ `L/λ` 열은 **참고**이고, 하중은 `L_obs` 에 있다
  (`L_obs` 는 빔포머를 **전혀 쓰지 않는다** — `ptp` 뿐이다).
- 실측 개구는 우리가 자른 것이 아니라 **WASS 재구성 상자의 크기**다.
  `tmp/B2_REAL_2026-08-23/scene_station.py:36` 이 2026-08-23 에 축자로 적었다:
  *"BS 7개는 박스가 13–19 m 로 개구/λ≈1 이라 훨씬 어렵다. 사후배제 금지."*

사용
----
    cd /home/vexit/wavesplat/wavesplat
    systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 \
        python3 tmp/_audit/aperture_gap_real_vs_train.py --nscene 600
"""
import argparse
import collections
import json
import os
import statistics as st
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
T17 = os.path.join(ROOT, 'tmp/T17_REALSCALE_2026-09-05/results_t17.jsonl')


def real_apertures():
    """동결 `T17` 행에서 레코드별 개구를 되읽는다. ⛔손복사 0 — `L_TRAIN` 은 임포트."""
    sys.path.insert(0, os.path.join(ROOT, 'tmp/T14_AUTOSCALE_2026-09-04'))
    import scale_rules_t14 as SR
    L_TRAIN = float(SR.L_TRAIN)

    rows = [json.loads(l) for l in open(T17, encoding='utf-8')]
    assert len(rows) == 432, f'★전제 파손: T17 결과가 432행이 아니다 ({len(rows)})'
    base = [r for r in rows if float(r.get('x_scale', 1.0)) == 1.0]
    assert len(base) == 144, f'★전제 파손: x_scale==1 행이 144가 아니다 ({len(base)})'
    arms = sorted({r['arm'] for r in base})
    assert arms == ['BANK', 'PROP'], f'★전제 파손: 미재척도 팔이 예상과 다르다 {arms}'

    per = collections.defaultdict(list)
    for r in base:
        per[r['record']].append(r)
    assert len(per) == 12, f'★전제 파손: 레코드가 12개가 아니다 ({len(per)})'

    out = {}
    for rec, v in per.items():
        L = [float(r['s_raw_ap']) * L_TRAIN for r in v]
        # 같은 레코드 안에서 개구는 창에 따라 흔들릴 수 있다 — 폭을 함께 보고한다.
        out[rec] = dict(
            L_med=st.median(L), L_min=min(L), L_max=max(L), n=len(v),
            kp_med=st.median([float(r['kp2_hat']) for r in v]),
            kp_min=min(float(r['kp2_hat']) for r in v),
            kp_max=max(float(r['kp2_hat']) for r in v),
            edge=sum(1 for r in v if r.get('edge_hit')),
            depth=float(v[0].get('depth_used')),
            Hs=st.median([float(r['Hs']) for r in v]),
        )
    return L_TRAIN, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nscene', type=int, default=600)
    ap.add_argument('--seed', type=int, default=20260726)   # train_mixed.py 기본값
    ap.add_argument('--no-ruler', action='store_true')
    a = ap.parse_args()

    os.chdir(ROOT)
    for p in (ROOT, os.path.join(ROOT, 'scripts'),
              os.path.join(ROOT, 'tmp/mixed_retrain_2026-07-26'),
              os.path.join(ROOT, 'tmp/_audit')):
        sys.path.insert(0, p)
    warnings.filterwarnings('ignore')

    import derive_train_support as D          # ★손복사 0 — 같은 계측기를 임포트한다
    import gen_mixed as gm

    L_TRAIN, real = real_apertures()

    # ── 학습 지지집합 재표집 ────────────────────────────────────────────────
    bank = gm.GeometryBank()
    rng = np.random.default_rng(a.seed)
    L, kn, kb = [], [], []
    for i in range(a.nscene):
        s = gm.sample_scene_ok(rng, bank)
        x, y, t = s['P'][:, 0], s['P'][:, 1], s['P'][:, 2]
        L.append(max(float(np.ptp(x)), float(np.ptp(y))))
        kn.append(D.kp_from_Tp(s['Tp']))
        if not a.no_ruler:
            kb.append(D.kp_hat_abs(x, y, t, s['z'], D.DEPTH))
        if (i + 1) % 200 == 0:
            print(f'   train scene {i+1}/{a.nscene}', flush=True)
    L = np.array(L); kn = np.array(kn); kb = np.array(kb) if kb else None

    hdr = ' '.join(f'{h:>9s}' for h in ('min', 'p5', 'p25', 'med', 'p75', 'p95', 'max'))
    print(f'\n== 배포 체크포인트의 학습 지지집합 (n={a.nscene}, 재표집) ==\n{"":26s}{hdr}')
    print(f'{"L_obs [m]":26s}' + ' '.join(f'{v:9.4g}' for v in D.pct(L)))
    print(f'{"L/lambda nominal":26s}' + ' '.join(f'{v:9.4g}' for v in D.pct(L * kn / (2 * np.pi))))
    if kb is not None:
        print(f'{"L/lambda beamformer":26s}'
              + ' '.join(f'{v:9.4g}' for v in D.pct(L * kb / (2 * np.pi))))

    tmin, tp5, tmed = float(L.min()), float(np.percentile(L, 5)), float(np.median(L))

    # ── 실측 12 레코드 ──────────────────────────────────────────────────────
    print(f'\n== 실측 12 레코드 (T17 동결 행, x_scale==1) ==')
    print(f'{"record":8s}{"L_obs[m]":>10s}{"(min–max)":>16s}  {"vs train":>9s}'
          f'{"kp2 med":>10s}{"edge":>6s}{"Hs":>7s}{"depth":>7s}')
    below_min = below_p5 = 0
    for rec in sorted(real):
        d = real[rec]
        tag = 'OUT<min' if d['L_med'] < tmin else ('OUT<p5' if d['L_med'] < tp5 else 'in')
        below_min += d['L_med'] < tmin
        below_p5 += d['L_med'] < tp5
        span = '({:.1f}–{:.1f})'.format(d['L_min'], d['L_max'])
        print('{:8s}{:10.2f}{:>16s}  {:>9s}{:10.3f}{:6d}{:7.2f}{:7.1f}'.format(
            rec, d['L_med'], span, tag, d['kp_med'], d['edge'], d['Hs'], d['depth']))

    Lr = [real[r]['L_med'] for r in real]
    print(f'\n실측 L_obs : min {min(Lr):.2f} · med {st.median(Lr):.2f} · max {max(Lr):.2f} [m]')
    print(f'학습 L_obs : min {tmin:.2f} · p5 {tp5:.2f} · med {tmed:.2f} [m]')
    print(f'★**학습 최소({tmin:.1f} m)보다 작은 실측 레코드 = {below_min}/12** '
          f'· p5({tp5:.1f} m) 아래 = {below_p5}/12')
    print(f'★중앙 대비 배율 = 학습 {tmed:.1f} / 실측 {st.median(Lr):.1f} = **{tmed/st.median(Lr):.2f}배**')

    ncorner = sum(1 for r in real if real[r]['kp_max'] > 5.0)
    print(f'\n⚠빔포머 코너 오염 의심 레코드(kp2 max > 5 rad/m) = {ncorner}/12 '
          f'⟹ `L/λ` 대조는 **참고**이고 하중은 `L_obs` 에 있다.')
    print('\n⛔이 출력은 어떤 배치의 판정도 바꾸지 않는다 — **범위 대조**일 뿐이고,'
          ' 인과 주장은 실측 개입이 라벨을 낼 때만 가능하다(`T17` = BLOCKED).')


if __name__ == '__main__':
    main()
