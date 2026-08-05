import re
flat = re.sub(r'\s+',' ', re.sub(r'-\n','', open('tmp/litproj_2026-08-05/text/feng_wang_2025.txt', encoding='utf-8').read()))
for k in ['Acqua Alta','oceanographic','platform','experiment','dataset','sequence','frames','m/s','wind',
          'Hs','significant wave height of','wave height','peak','period','depth','Adriatic','Black Sea','synthetic case','numerical']:
    for m in re.finditer(re.escape(k), flat):
        s = flat[max(0,m.start()-150): m.start()+250]
        if re.search(r'\d', s):
            print(f'[{k}] …{s.strip()[:330]}…\n')
            break
