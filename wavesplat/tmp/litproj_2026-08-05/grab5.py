import re
for fn, keys in [
 ('feng_wang_2025.txt', ['numerical experiment','synthetic','stereo','case 1','Case 1','Hs','significant','spreading','JONSWAP','directional','real','field experiment','Tp','peak period','validation']),
 ('12_learning_strategy__arxiv_2512.06024.txt', ['A1','experimental condition','Hs','significant wave height','field','Acqua Alta','tower','directional','spreading','Tp','peak period']),
 ('08_UQ_wave__da_wave_ship_2504.13943.txt', ['2D wave','directional','spreading','Hs','significant','JONSWAP','kp']),
]:
    flat = re.sub(r'\s+',' ', re.sub(r'-\n','', open('tmp/litproj_2026-08-05/text/'+fn, encoding='utf-8').read()))
    print(f"\n════════ {fn}")
    seen=set()
    for k in keys:
        for m in re.finditer(re.escape(k), flat):
            s = flat[max(0,m.start()-140): m.start()+220]
            if not re.search(r'\d', s) or s[:70] in seen: continue
            seen.add(s[:70]); print(f'  [{k}] …{s.strip()[:300]}…')
            break
