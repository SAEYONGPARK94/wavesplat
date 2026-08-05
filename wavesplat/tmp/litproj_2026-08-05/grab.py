import os, re, sys
TXT = 'tmp/litproj_2026-08-05/text'
targets = {
 'ROOT__Sea_trial_on_deterministic_sea_waves_prediction_using_wave-profiling_radar.txt':
   ['sea state', 'significant', 'Hs', 'trial', 'North Sea', 'metocean', 'buoy', 'Tp', 'period'],
 'wang_pan_2021_jfm.txt': ['kp = ', 'k0', 'steepness', 'Hs', 'JONSWAP', 'global steepness'],
 'vof_enkf_2025.txt': ['significant wave height', 'peak period', 'spreading', 'depth', 'JONSWAP'],
 '07_Wave_competitors__ehlers_2025b_pino.txt': ['water depth', 'steepness', 'peak wavelength', 'JONSWAP', 'Lp'],
 '12_learning_strategy__arxiv_2501.08430.txt': ['water depth', 'steepness', 'peak wavelength', 'JONSWAP', 'Lp ∈'],
 '08_UQ_wave__zhang2022_varBayes_UQ_appliedenergy.txt': ['significant wave height', 'peak period', 'water depth', 'JONSWAP'],
 '07_Wave_competitors__swh_sparse_2025.txt': ['significant wave height', 'spreading', 'buoy', 'dataset'],
 'feng_wang_2025.txt': ['significant wave height', 'spreading', 'peak period', 'JONSWAP'],
 '08_UQ_wave__da_wave_ship_2504.13943.txt': ['significant wave height', 'spreading', 'peak period', 'JONSWAP'],
}
for fn, keys in targets.items():
    p = os.path.join(TXT, fn)
    if not os.path.exists(p):
        print('MISSING', fn); continue
    txt = re.sub(r'-\n', '', open(p, encoding='utf-8').read())
    flat = re.sub(r'\s+', ' ', txt)
    print(f"\n════════════════ {fn}")
    seen = set()
    for k in keys:
        for m in re.finditer(re.escape(k), flat):
            s = flat[max(0, m.start()-160): m.start()+240]
            if s[:80] in seen:
                continue
            seen.add(s[:80])
            if re.search(r'\d', s):
                print(f"  [{k}] …{s}…")
            if len(seen) > 60:
                break
