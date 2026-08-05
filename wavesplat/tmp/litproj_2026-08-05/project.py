"""문헌 시험 해상상태 → CDIP 지지집합 투사.

R5 다음실행 #2. 목적: 우리 생성기의 배포-support 불일치가 **우리 결함**인지
**분야 전체의 성질**인지 가른다. 자기감사 → 분야감사 승격 여부가 여기서 결정된다.

양팔 추정기 동일: sigma_1 = sqrt(2(1 - sqrt(a1^2+b1^2)))  [CDIP 와 같은 식]
문헌 쪽은 논문이 **연속 D(theta) 를 명시**하므로 CDIP 와 같은 피적분 대상이다
(⛔#23 의 경고는 우리 이산 합성 팔에만 해당 — 이 비교에는 해당하지 않는다).

모든 값의 출처는 tmp/litproj_2026-08-05/excerpts/ 의 발췌문이며 LIT 표에 인용해 둔다.
"""
import json
import numpy as np

G = 9.81


def sigma1_from_D(D, lo, hi, n=200001):
    """CDIP 와 동일한 1차 모멘트 확산. D 는 정규화됐다고 가정하지 않는다."""
    th = np.linspace(lo, hi, n)
    d = D(th)
    m0 = np.trapezoid(d, th)
    a1 = np.trapezoid(d * np.cos(th), th) / m0
    b1 = np.trapezoid(d * np.sin(th), th) / m0
    r1 = np.hypot(a1, b1)
    return float(np.degrees(np.sqrt(2 * (1 - r1))))


# ---- 문헌이 실제로 쓴 방향분포 -------------------------------------------------
def cos2_halfwidth(sig):
    """D(th) = (1/sig) cos^2(pi*th/(2 sig)),  |th| <= sig   (Desmars 2018 식 (10))"""
    return (lambda th: np.cos(np.pi * th / (2 * sig)) ** 2 / sig), -sig, sig


def cos2_beta(beta):
    """D(th) = (2/beta) cos^2(pi*th/beta), |th| < beta/2   (Wang & Pan 식 (3.5)/(3.2))"""
    return (lambda th: (2 / beta) * np.cos(np.pi * th / beta) ** 2), -beta / 2, beta / 2


SPREAD_SPECS = {
    'unidirectional (long-crested)': ('delta', 0.0),
    'Wang&Pan cos^2, beta=pi/6 (3-D case)': ('fn',) + cos2_beta(np.pi / 6),
    'Desmars2018 cos^2, sigma=pi/4': ('fn',) + cos2_halfwidth(np.pi / 4),
    '[sensitivity] cos^2, sigma=pi/2': ('fn',) + cos2_halfwidth(np.pi / 2),
    '[reference] cos^2s, s=10': ('fn', lambda th: np.cos(th / 2) ** 20, -np.pi, np.pi),
    '[reference] cos^2s, s=4': ('fn', lambda th: np.cos(th / 2) ** 8, -np.pi, np.pi),
    '[reference] cos^2s, s=1': ('fn', lambda th: np.cos(th / 2) ** 2, -np.pi, np.pi),
}

spread_sigma1 = {}
for name, spec in SPREAD_SPECS.items():
    if spec[0] == 'delta':
        spread_sigma1[name] = 0.0
    else:
        spread_sigma1[name] = sigma1_from_D(spec[1], spec[2], spec[3])

# ---- 문헌 시험 해상상태 원장 ---------------------------------------------------
# dirn: 'uni' = long-crested/1D, 'dir' = 실제 방향분포, 'mono' = 규칙파, 'real' = 실측
LIT = [
    dict(key='Desmars2020 (OE)', dirn='uni', spread=None,
         Hs=(0.89, 8.02), Tp=(10.0, 10.0), depth='deep (tank 1:50, d*=250 m)',
         note='JONSWAP gamma=3.3, long-crested tank + HOS-NWT, 8 sea states'),
    dict(key='Desmars2023 (JFM) main', dirn='uni', spread=None,
         Hs=(0.5, 5.0), Tp=(10.0, 10.0), depth='deep',
         note='JONSWAP gamma=3.3, Hs/lam_p up to 3.2% -> Hs<=5.0 m at lam_p=156 m'),
    dict(key='Desmars2023 (JFM) tank case', dirn='uni', spread=None,
         Hs=(11.92, 11.92), Tp=(16.7, 16.7), depth='70 m (k_p h ~ 1.2)',
         note='gamma=1; the single deep-swell case in the whole corpus'),
    dict(key='Desmars2018 (HAL/ISOPE)', dirn='dir', spread='Desmars2018 cos^2, sigma=pi/4',
         Hs=(3.0, 3.0), Tp=(10.0, 10.0), depth='deep',
         note='directional JONSWAP, gamma=3.3, sigma=pi/4'),
    dict(key='Klein2020 (Fluids)', dirn='uni', spread=None,
         Hs=(1.83, 5.48), Tp=(8.58, 8.58), depth='k_p d = 4.1 (tank d=1 m)',
         note='JONSWAP gamma in {1,3,6}, eps in {0.05,0.10,0.15}, TUB basin'),
    dict(key='Ehlers2023 (OE) / Ehlers2023 (FNO)', dirn='uni', spread=None,
         Hs=(0.25, 6.37), Tp=(7.16, 11.32), depth='500 m',
         note='1-D eta(x,t); L_p 80-200 m, eps 0.01-0.10, gamma=3; Hs=2eps/k_p, Tp=sqrt(2 pi L_p/g)'),
    dict(key='Ehlers2025 (PINO)', dirn='uni', spread=None,
         Hs=(None, None), Tp=(1.2, 5.24), depth='200 m / tank',
         note='"demonstrated on long-crested, unidirectional (1D+t) wave surfaces"'),
    dict(key='WangPan2021 (JFM) 2-D', dirn='uni', spread=None,
         Hs=(1.70, 1.70), Tp=(None, None), depth='deep',
         note='main idealistic case unidirectional, k_p=16k_0'),
    dict(key='WangPan2021 (JFM) 3-D', dirn='dir', spread='Wang&Pan cos^2, beta=pi/6 (3-D case)',
         Hs=(1.70, 1.70), Tp=(None, None), depth='deep', note='beta=pi/6 spreading angle'),
    dict(key='WangPan2022 (JFM) 3-D', dirn='dir', spread='Wang&Pan cos^2, beta=pi/6 (3-D case)',
         Hs=(1.50, 1.50), Tp=(11.28, 11.28), depth='deep', note='k_p=6k_0, beta=pi/6'),
    dict(key='Zhang2022 (Appl.Energy)', dirn='uni', spread=None,
         Hs=(None, None), Tp=(None, None), depth='EMEC 1:50 tank',
         note='"the wave tank experiments in the current work consider the uni-directional waves"'),
    dict(key='VOF-EnKF2025', dirn='mono', spread=None,
         Hs=(0.075, 0.12), Tp=(2.0, 5.0), depth='0.4-0.6 m (lab)',
         note='regular cosine waves + one irregular case; surf-zone breaking'),
    dict(key='Golden Arrow sea trial (radar)', dirn='real', spread=None,
         Hs=(None, None), Tp=(11.0, 13.0), depth='open ocean (W of Scotland)',
         note='REAL ocean, two wave systems (W + SW) -> genuinely multidirectional; Hs not stated in text'),
    dict(key='WangPan2025 (EnKF-HOS-CMI, ship)', dirn='uni', spread=None,
         Hs=(None, None), Tp=(None, None), depth='deep',
         note='2-D (long-crested) synthetic; JONSWAP k_p=16k_0, k_p Hs/2 = 0.11, gamma=3.3'),
    dict(key='FengWang2025 (Coastal Eng, Wave-ETKF)', dirn='real', spread=None,
         Hs=(None, None), Tp=(None, None), depth='Acqua Alta platform (~16 m)',
         note='REAL: public Acqua Alta stereo AA03 + own field data; same WASS family as our data/real_bergamasco/; code public'),
    dict(key='Liu/Benetazzo2025 (camera, 6 cond.)', dirn='real', spread=None,
         Hs=(0.36, 0.66), Tp=(2.63, 5.27), depth='coastal platform',
         note='REAL stereo, A1-A3/B1-B3; Hs 360-660 mm, Tp 2.63-5.27 s; "strong directional scattering"'),
    dict(key='THIS WORK (synthetic 100 ep)', dirn='dir', spread=None,
         Hs=(0.3, 2.0), Tp=(4.0, 9.0), depth='deep',
         note='sigma_1 = 0 - 41.7 deg (median 16.1); discrete N-component, no continuous D(theta;f)'),
]

# ---- CDIP 지지집합 -----------------------------------------------------------
summ = json.load(open('tmp/cdip_support_2026-08-03/support_summary.json'))
raw = json.load(open('tmp/cdip_support_2026-08-03/cdip_support_raw.json'))


REAL = {}
_st = raw['real']
for f, key in [('Hs', 'Hs'), ('Tp', 'Tp'), ('spread', 'spread_total_deg')]:
    vals = []
    for st in _st:
        vals += [v for v in st[key] if v is not None]
    REAL[f] = np.asarray(vals, float)
    REAL[f] = REAL[f][np.isfinite(REAL[f])]
print({k: len(v) for k, v in REAL.items()})

print('=' * 78)
print('A. 문헌 방향분포 -> sigma_1 (CDIP 와 동일 추정기)')
print('=' * 78)
cd = summ['marginals']['spread_total_deg']['real']
for name, s in spread_sigma1.items():
    if 'spread' in REAL:
        pct = float((REAL['spread'] < s).mean() * 100)
        tag = f'CDIP 백분위 {pct:5.2f}%'
    else:
        tag = ('CDIP min 미만' if s < cd['min'] else
               'CDIP p05 미만' if s < cd['p05'] else 'CDIP p05 이상')
    print(f'  {name:44s} sigma_1 = {s:6.2f} deg   {tag}')
print(f"\n  CDIP 실측(n={cd['n']}): min {cd['min']:.2f}  p05 {cd['p05']:.2f}  "
      f"median {cd['median']:.2f}  p95 {cd['p95']:.2f}  max {cd['max']:.2f}")

print()
print('=' * 78)
print('B. 문헌 시험 해상상태 원장 (dirn: uni=long-crested, dir=방향분포, mono=규칙파, real=실측)')
print('=' * 78)
hdr = f"{'paper':38s} {'dirn':5s} {'sigma1':>7s} {'Hs [m]':>13s} {'Tp [s]':>13s}  depth"
print(hdr)
rows = []
for r in LIT:
    s = 0.0 if r['dirn'] == 'uni' else (spread_sigma1[r['spread']] if r['spread'] else None)
    sstr = f'{s:7.2f}' if s is not None else '      ?'
    hs = ('%.2f-%.2f' % r['Hs']) if r['Hs'][0] is not None else '?'
    tp = ('%.2f-%.2f' % r['Tp']) if r['Tp'][0] is not None else '?'
    print(f"{r['key']:38s} {r['dirn']:5s} {sstr} {hs:>13s} {tp:>13s}  {r['depth']}")
    rows.append({**r, 'sigma1_deg': s})

n_uni = sum(1 for r in LIT if r['dirn'] == 'uni')
n_dir = sum(1 for r in LIT if r['dirn'] == 'dir' and 'THIS WORK' not in r['key'])
n_mono = sum(1 for r in LIT if r['dirn'] == 'mono')
n_real = sum(1 for r in LIT if r['dirn'] == 'real')
tot = n_uni + n_dir + n_mono + n_real
n_spec = n_uni + n_dir + n_mono          # 확산이 **명시된** 설정 = 투사 가능한 분모
n_zero = n_uni + n_mono + sum(1 for r in LIT if r['dirn'] == 'dir' and r['spread']
                              and spread_sigma1[r['spread']] < cd['min'])
print(f"\n  집계: long-crested/1D {n_uni} · 방향분포 {n_dir} · 규칙파 {n_mono} · 실측(확산 미명시) {n_real}"
      f"  (총 {tot} 시험설정)")
print(f"  ⟹ 확산이 명시된 {n_spec}개 중 **{n_zero}개가 CDIP 관측 최소(12.18 deg) 미만 = 0.00 백분위**")
print(f"  ⟹ 실측 설정 {n_real}개는 확산이 자연이 준 값이며 본문에 미기재 (투사 불가, 분모에서 제외)")

print()
print('=' * 78)
print('C. Hs / Tp 투사 (CDIP 주변분포 백분위)')
print('=' * 78)


def pct_of(field, val):
    if field in REAL and val is not None:
        return float((REAL[field] < val).mean() * 100)
    return None


for r in LIT:
    parts = []
    for f in ('Hs', 'Tp'):
        lo, hi = r[f]
        if lo is None:
            parts.append(f'{f}: ?')
            continue
        plo, phi = pct_of(f, lo), pct_of(f, hi)
        if plo is None:
            m = summ['marginals'][f]['real']
            inside = m['p05'] <= lo <= m['p95']
            parts.append(f'{f}: {lo}-{hi} ({"in" if inside else "out"} p05-p95)')
        else:
            parts.append(f'{f}: p{plo:.1f}-p{phi:.1f}')
    print(f"  {r['key']:38s} " + '  |  '.join(parts))

out = {'spread_sigma1_deg': spread_sigma1, 'literature': rows,
       'cdip_spread_total_deg': cd,
       'counts': {'long_crested_1D': n_uni, 'directional': n_dir,
                  'monochromatic': n_mono, 'real_ocean': n_real, 'total': tot}}
json.dump(out, open('tmp/litproj_2026-08-05/lit_projection.json', 'w'), indent=2)
print('\n-> tmp/litproj_2026-08-05/lit_projection.json')
