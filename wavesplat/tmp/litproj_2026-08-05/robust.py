"""대칭감사: 문헌 D(theta) 를 CDIP 의 **대역분해** 확산에도 투사한다.

왜 필요한가: `spread_total_deg` 는 0.04-0.5 Hz 를 통째로 적분하므로 실측 쪽에는
**주파수별 평균방향의 변동**이 섞여 들어간다. 논문의 단일 D(theta) 에는 그 항이 없다.
따라서 total 로만 비교하면 문헌이 부당하게 좁아 보일 수 있다.
`spread_band_deg`(대역분해)가 단일 D(theta) 와 같은 피적분 대상이다.
"""
import json
import numpy as np

raw = json.load(open('tmp/cdip_support_2026-08-03/cdip_support_raw.json'))
band, total = [], []
for st in raw['real']:
    band += [v for v in st['spread_band_deg'] if v is not None]
    total += [v for v in st['spread_total_deg'] if v is not None]
band = np.asarray(band, float); band = band[np.isfinite(band)]
total = np.asarray(total, float); total = total[np.isfinite(total)]

lit = json.load(open('tmp/litproj_2026-08-05/lit_projection.json'))
S = lit['spread_sigma1_deg']

print(f"CDIP band-resolved (n={len(band)}): min {band.min():.2f}  p01 {np.percentile(band,1):.2f}  "
      f"p05 {np.percentile(band,5):.2f}  median {np.median(band):.2f}  p95 {np.percentile(band,95):.2f}")
print(f"CDIP total        (n={len(total)}): min {total.min():.2f}  p01 {np.percentile(total,1):.2f}  "
      f"p05 {np.percentile(total,5):.2f}  median {np.median(total):.2f}  p95 {np.percentile(total,95):.2f}")
print()
print(f"{'spreading spec':44s} {'sigma1':>7s} {'pct(band)':>10s} {'pct(total)':>11s}")
for k, v in S.items():
    pb = float((band < v).mean() * 100)
    pt = float((total < v).mean() * 100)
    print(f'  {k:42s} {v:7.2f} {pb:9.2f}% {pt:10.2f}%')

print()
print('한 문장: 문헌의 가장 넓은 방향 설정(Desmars2018, sigma=pi/4)조차 대역분해 기준 '
      f'{float((band < S["Desmars2018 cos^2, sigma=pi/4"]).mean()*100):.1f}% 백분위,'
      f' total 기준 {float((total < S["Desmars2018 cos^2, sigma=pi/4"]).mean()*100):.1f}% 백분위다.')
json.dump({'band': {'n': len(band), 'min': float(band.min()),
                    'p01': float(np.percentile(band, 1)), 'p05': float(np.percentile(band, 5)),
                    'median': float(np.median(band))},
           'pct_band': {k: float((band < v).mean() * 100) for k, v in S.items()},
           'pct_total': {k: float((total < v).mean() * 100) for k, v in S.items()}},
          open('tmp/litproj_2026-08-05/robust.json', 'w'), indent=2)
