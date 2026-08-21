import json,os,glob
from collections import Counter
tot=0; per={}
for f in sorted(glob.glob('tmp/*/results*.jsonl')):
    n=sum(1 for _ in open(f)); per[f]=n; tot+=n
print('총 런수', tot)
audited={'tmp/S7_confirm_n32_2026-08-10/results.jsonl','tmp/S8_deploy_confirm_2026-08-10/results.jsonl','tmp/S9_lr_x_l1_2026-08-11/results.jsonl','tmp/X2_no_learned_2026-08-11/results.jsonl'}
a=sum(v for k,v in per.items() if k in audited)
print(f'AUDIT-C 가 원자료로 연 파일 {len(audited)}개 = {a} 런 ({a/tot*100:.1f}%)')
print(f'한 번도 안 연 파일 {len(per)-len(audited)}개 = {tot-a} 런 ({(tot-a)/tot*100:.1f}%)')
print()
for k,v in sorted(per.items(), key=lambda x:-x[1]):
    print(f'   {"AUDITED" if k in audited else "       "}  {v:6d}  {k}')
