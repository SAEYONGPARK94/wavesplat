import re
for fn, keys in [
  ('ROOT__Sea_trial_on_deterministic_sea_waves_prediction_using_wave-profiling_radar.txt',
   ['wave height', 'Hs', 'metre', ' m ', 'period', 'trial', 'Atlantic', 'sea state 4', 'sea state 5']),
  ('desmars2023_jfm.txt', ['Hs/λp', 'Tp =', 'Hs =', 'peak period', 'water depth', 'tank', 'spreading']),
  ('vof_enkf_2025.txt', ['d = 0', 'depth', 'basin', 'scale']),
]:
    flat = re.sub(r'\s+', ' ', re.sub(r'-\n', '', open('tmp/litproj_2026-08-05/text/'+fn, encoding='utf-8').read()))
    print(f"\n════ {fn}")
    seen = set()
    for k in keys:
        for m in re.finditer(re.escape(k), flat):
            s = flat[max(0, m.start()-120): m.start()+180]
            if not re.search(r'\d', s) or s[:60] in seen: continue
            seen.add(s[:60]); print('  ·', s.strip()[:290])
            if len(seen) > 22: break
        if len(seen) > 22: break
