import re
for fn in ['feng_wang_2025.txt', '12_learning_strategy__arxiv_2512.06024.txt',
           '02_INR__stride_tong_2026.txt', '07_Wave_competitors__swh_sparse_2025.txt',
           '12_learning_strategy__arxiv_2204.06843.txt',
           '08_UQ_wave__da_wave_ship_2504.13943.txt']:
    flat = re.sub(r'\s+', ' ', re.sub(r'-\n', '', open('tmp/litproj_2026-08-05/text/'+fn, encoding='utf-8').read()))
    print(f"\n════ {fn}")
    print('  TITLE:', flat[:170])
    seen=set()
    for k in ['long-crested','unidirectional','uni-directional','multidirectional','multi-directional',
              'short-crested','directional spread','spreading','2D','3D','buoy','tank','basin','flume','field data']:
        i = flat.lower().find(k.lower())
        if i>=0 and k[:6] not in seen:
            seen.add(k[:6])
            print(f'   [{k}] …{flat[max(0,i-110):i+150]}…')
