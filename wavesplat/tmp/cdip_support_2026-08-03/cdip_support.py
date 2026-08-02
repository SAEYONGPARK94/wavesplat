"""CDIP 지지집합(support) 측정 — 페르소나 토론 ① C3 스코프.

재는 것 : 실측 해상상태의 (Hs, Tp, 방향확산) 분포 vs 우리 합성 생성기의 지지집합 피복률
⛔재지 않는 것: 성능. 이 스크립트에서 어떤 nRMSE 도 나오지 않는다.
⛔ n_components 재정의 불가 — 부이 방향확산 != 분리가능 성분 수.

**팔-정체성**: 방향확산은 양쪽 모두 *같은 추정기*로 계산한다 —
    sigma_1(f) = sqrt(2 * (1 - sqrt(a1^2 + b1^2)))   [rad]
CDIP 쪽은 배포된 waveSpread 를 쓰지 않고 a1/b1 원본에서 직접 재도출하고,
합성 쪽은 wave_components 의 (A, omega, kx, ky) 에서 같은 식으로 만든다.

표본 규칙(사전 고정): 아카이브 스테이션 목록을 정렬 -> STRIDE 간격으로 뽑고 -> 각 스테이션의
**마지막 배치**를 쓰고 -> 시간축은 레코드가 TARGET_REC 이하가 되도록 stride.
QC: waveFlagPrimary == 1 (good) 인 레코드만.
"""
import json
import os
import re
import sys
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
ARCHIVE = 'https://thredds.cdip.ucsd.edu/thredds/catalog/cdip/archive/catalog.xml'
DODS = 'https://thredds.cdip.ucsd.edu/thredds/dodsC/'

STRIDE_STATION = 8      # 194개 중 매 8번째
TARGET_REC = 1000       # 스테이션당 목표 레코드 수
FMIN, FMAX = 0.04, 0.50  # sea-swell 대역 (CDIP 관례)


def get(url, tries=4, timeout=120):
    last = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read().decode('utf-8', 'replace')
        except Exception as e:
            last = e
            time.sleep(2 * (i + 1))
    raise last


HDR = re.compile(r'^([A-Za-z_][\w.]*)((?:\[\d+\])+)$')


def parse_ascii(txt):
    """OPeNDAP .ascii -> {varname: np.ndarray}.

    헤더는 `name[N]` 또는 Grid 안의 `parent.child[N][M]`. 2D 는 행마다 `[i], v, v, ...`.
    점 표기의 **마지막 성분**을 키로 쓴다(`waveEnergyDensity.waveEnergyDensity` -> `waveEnergyDensity`).
    """
    out, dims = {}, {}
    body = txt.split('-' * 20, 1)[-1]
    cur, rows, twod = None, [], False

    def flush():
        if cur is None or not rows:
            return
        arr = np.array(rows, dtype=float)
        if not twod:
            arr = arr.reshape(-1)
        out[cur] = arr

    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        m = HDR.match(line)
        if m:
            flush()
            cur = m.group(1).split('.')[-1]
            shp = [int(s) for s in re.findall(r'\[(\d+)\]', m.group(2))]
            dims[cur] = shp
            twod = len(shp) == 2
            rows = []
            continue
        if cur is None:
            continue
        if line.startswith('['):                       # 2D row: "[3], 1.0, 2.0"
            vals = line.split(']', 1)[1].lstrip(', ')
            rows.append([float(v) for v in vals.split(',') if v.strip()])
        else:                                          # 1D chunk
            rows.append([float(v) for v in line.split(',') if v.strip()])
    flush()
    return out


def spread_from_moments(a1, b1):
    """sigma_1 in degrees. a1,b1 shape (..., nfreq)."""
    r1 = np.sqrt(np.clip(a1 ** 2 + b1 ** 2, 0, 1))
    return np.degrees(np.sqrt(np.clip(2.0 * (1.0 - r1), 0, None)))


def energy_weighted(sig, E, fmask):
    """band-energy-weighted mean over the sea-swell band."""
    w = np.where(fmask[None, :], E, 0.0)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    tot = w.sum(axis=1)
    ok = tot > 0
    out = np.full(sig.shape[0], np.nan)
    out[ok] = (w[ok] * np.nan_to_num(sig[ok])).sum(axis=1) / tot[ok]
    return out


# --------------------------------------------------------------- real (CDIP)
def stations():
    x = get(ARCHIVE)
    st = [t for t in re.findall(r'xlink:title="([^"]+)"', x) if re.fullmatch(r'\d+p\d+', t)]
    return sorted(set(st))


def last_deployment(st):
    x = get(f'https://thredds.cdip.ucsd.edu/thredds/catalog/cdip/archive/{st}/catalog.xml')
    ds = sorted(set(re.findall(rf'({st}_d\d+\.nc)', x)))
    return ds[-1] if ds else None


def fetch_station(st, dep):
    base = f'{DODS}cdip/archive/{st}/{dep}'
    dds = get(base + '.dds')
    m = re.search(r'waveEnergyDensity\[waveTime = (\d+)\]\[waveFrequency = (\d+)\]', dds)
    if not m:
        return None
    nt, nf = int(m.group(1)), int(m.group(2))
    stride = max(1, nt // TARGET_REC)
    sl = f'[0:{stride}:{nt - 1}]'
    q = (f'waveFrequency[0:1:{nf - 1}],waveHs{sl},waveTp{sl},waveFlagPrimary{sl},'
         f'waveEnergyDensity{sl}[0:1:{nf - 1}],waveA1Value{sl}[0:1:{nf - 1}],'
         f'waveB1Value{sl}[0:1:{nf - 1}]')
    d = parse_ascii(get(base + '.ascii?' + q, timeout=300))
    f = d['waveFrequency']
    E = np.atleast_2d(d['waveEnergyDensity'])
    a1 = np.atleast_2d(d['waveA1Value'])
    b1 = np.atleast_2d(d['waveB1Value'])
    hs, tp, fl = d['waveHs'].ravel(), d['waveTp'].ravel(), d['waveFlagPrimary'].ravel()
    for arr in (E, a1, b1):
        arr[arr <= -999] = np.nan
    hs = np.where(hs <= -999, np.nan, hs)
    tp = np.where(tp <= -999, np.nan, tp)
    fmask = (f >= FMIN) & (f <= FMAX)
    sig = spread_from_moments(a1, b1)
    sw = energy_weighted(sig, E, fmask)                       # (i) 대역내(within-band)
    # (ii) 전대역(total-band): 대역 전체를 하나의 a1/b1 로 합쳐 계산 — 이산 성분 모형과
    #      비교 가능한 유일한 형태. (i) 은 우리 생성기에서 구조적으로 0 이 나오므로 병기한다.
    w = np.where(fmask[None, :] & np.isfinite(E) & (E > 0), E, 0.0)
    tot = w.sum(axis=1)
    A1 = np.where(tot > 0, (w * np.nan_to_num(a1)).sum(axis=1) / np.where(tot > 0, tot, 1), np.nan)
    B1 = np.where(tot > 0, (w * np.nan_to_num(b1)).sum(axis=1) / np.where(tot > 0, tot, 1), np.nan)
    sw_tot = spread_from_moments(A1, B1)
    good = (fl == 1) & np.isfinite(hs) & np.isfinite(tp) & np.isfinite(sw) & (tp > 0)
    return {'station': st, 'deployment': dep, 'n_raw': len(hs), 'n_good': int(good.sum()),
            'Hs': hs[good].tolist(), 'Tp': tp[good].tolist(),
            'spread_band_deg': sw[good].tolist(), 'spread_total_deg': sw_tot[good].tolist()}


# --------------------------------------------------- synthetic (our generator)
def synth_records(data_dir):
    import h5py
    recs = []
    for fn in sorted(os.listdir(data_dir)):
        if not fn.endswith('.h5'):
            continue
        with h5py.File(os.path.join(data_dir, fn), 'r') as f:
            if 'config/wave_components' not in f:
                continue
            wc = f['config/wave_components'][:]
            att = dict(f['config'].attrs)
        A, om, kx, ky = wc[:, 0], wc[:, 1], wc[:, 2], wc[:, 3]
        fr = om / (2 * np.pi)
        th = np.arctan2(ky, kx)
        E = 0.5 * A ** 2
        band = (fr >= FMIN) & (fr <= FMAX)
        if not band.any() or E[band].sum() <= 0:
            continue
        # 같은 추정기: 대역 단위 a1/b1 -> sigma_1 -> 대역에너지 가중
        edges = np.arange(FMIN, FMAX + 1e-9, 0.005)
        sig_b, w_b = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = band & (fr >= lo) & (fr < hi)
            if not s.any():
                continue
            e = E[s].sum()
            a1 = (E[s] * np.cos(th[s])).sum() / e
            b1 = (E[s] * np.sin(th[s])).sum() / e
            sig_b.append(float(spread_from_moments(np.array([a1]), np.array([b1]))[0]))
            w_b.append(e)
        if not w_b:
            continue
        sig_b, w_b = np.array(sig_b), np.array(w_b)
        # (ii) 전대역: 실측 팔과 같은 식으로, 대역 전체 에너지 가중 a1/b1
        e_all = E[band].sum()
        A1 = (E[band] * np.cos(th[band])).sum() / e_all
        B1 = (E[band] * np.sin(th[band])).sum() / e_all
        recs.append({
            'episode': fn[len('episode_'):-3],
            'sea_state': str(att.get('sea_state')),
            'n_components': int(att.get('n_components', len(A))),
            'n_comp_in_band': int(band.sum()),
            'Hs': float(att.get('Hs', np.nan)),
            'Tp': float(att.get('Tp', np.nan)),
            'spread_band_deg': float((sig_b * w_b).sum() / w_b.sum()),
            'spread_total_deg': float(spread_from_moments(np.array([A1]), np.array([B1]))[0]),
        })
    return recs


def main():
    out = {'scope': 'descriptive support comparison only; no performance claim',
           'estimator': 'sigma_1 = sqrt(2(1-sqrt(a1^2+b1^2))), identical both arms',
           'band_hz': [FMIN, FMAX], 'stride_station': STRIDE_STATION,
           'target_records_per_station': TARGET_REC}

    print('--- synthetic ---', flush=True)
    syn = synth_records(os.path.join(ROOT, 'data_finemesh_2026-07-23'))
    out['synthetic'] = syn
    print(f'  {len(syn)} episodes', flush=True)

    print('--- CDIP ---', flush=True)
    st = stations()
    pick = st[::STRIDE_STATION]
    out['stations_total'] = len(st)
    out['stations_picked'] = pick
    real, fails = [], []
    for s in pick:
        try:
            dep = last_deployment(s)
            if dep is None:
                fails.append((s, 'no deployment')); continue
            r = fetch_station(s, dep)
            if r is None:
                fails.append((s, 'no waveEnergyDensity')); continue
            real.append(r)
            print(f'  {s}/{r["deployment"]}: {r["n_good"]}/{r["n_raw"]} good', flush=True)
        except Exception as e:
            fails.append((s, f'{type(e).__name__}: {e}'))
            print(f'  {s}: FAIL {type(e).__name__}', flush=True)
    out['real'] = real
    out['failures'] = fails
    with open(os.path.join(HERE, 'cdip_support_raw.json'), 'w') as f:
        json.dump(out, f)
    print(f'stations ok={len(real)} fail={len(fails)} '
          f'records={sum(r["n_good"] for r in real)}', flush=True)


if __name__ == '__main__':
    main()
