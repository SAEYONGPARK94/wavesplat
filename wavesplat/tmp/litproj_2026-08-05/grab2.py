import os, re
TXT = 'tmp/litproj_2026-08-05/text'
FILES = ['ROOT__Sea_trial_on_deterministic_sea_waves_prediction_using_wave-profiling_radar.txt',
         'wang_pan_2021_jfm.txt', 'wang_pan_2022_jfm.txt', 'vof_enkf_2025.txt',
         '07_Wave_competitors__ehlers_2025b_pino.txt', '12_learning_strategy__arxiv_2501.08430.txt',
         '08_UQ_wave__zhang2022_varBayes_UQ_appliedenergy.txt',
         '07_Wave_competitors__swh_sparse_2025.txt', 'feng_wang_2025.txt',
         '08_UQ_wave__da_wave_ship_2504.13943.txt', '02_INR__stride_tong_2026.txt',
         '12_learning_strategy__arxiv_2512.06024.txt', '12_learning_strategy__arxiv_2204.06843.txt']
PATS = [
 r'[Ss]ignificant wave height[^.]{0,90}?\d[^.]{0,60}',
 r'H_?s\s*[=≈]\s*[\d.]+\s*(?:m|meters)?',
 r'T_?p\s*[=≈]\s*[\d.]+\s*s?',
 r'peak period[^.]{0,60}?\d[^.]{0,40}',
 r'water depth[^.]{0,60}?\d[^.]{0,40}',
 r'spreading (?:angle|parameter|coefficient)[^.]{0,60}',
 r'[βσ]\s*=\s*[^.,;]{1,25}',
 r'k_?p\s*=\s*[^.,;]{1,25}',
 r'steepness[^.]{0,20}?[=∈][^.]{0,45}',
 r'sea state[s]?\s*(?:of|with)[^.]{0,70}',
]
RX = [re.compile(p) for p in PATS]
for fn in FILES:
    p = os.path.join(TXT, fn)
    if not os.path.exists(p):
        print('MISSING', fn); continue
    flat = re.sub(r'\s+', ' ', re.sub(r'-\n', '', open(p, encoding='utf-8').read()))
    outs, seen = [], set()
    for rx in RX:
        for m in rx.finditer(flat):
            s = m.group(0).strip()
            if s[:50] in seen: continue
            seen.add(s[:50]); outs.append(s)
    print(f"\n════ {fn}  ({len(outs)} hits)")
    for s in outs[:28]:
        print('   ·', s[:190])
