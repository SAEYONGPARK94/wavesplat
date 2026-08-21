"""§4 후보를 동결 채점기(analyze_s9.py)에 먹여 보는 가짜 배치 생성기 (감사용, 런 0)."""
import json, os, sys
sys.path.insert(0, '/home/vexit/wavesplat/wavesplat/tmp/S11_null_corrected_2026-08-12')
import neutral_rederive as N

D, reg, err = N.load()
OUT = sys.argv[1]; MODE = sys.argv[2]

def emit(fh, cellname, srccell, ep):
    for s, r in D[srccell][ep].items():
        q = dict(r); q['cell'] = cellname; q['job_id'] = f'{cellname}|{ep}|{s}'
        fh.write(json.dumps(q) + '\n')

with open(OUT, 'w') as fh:
    for ep in sorted(reg):
        if MODE == 'A':          # §4-A: I900 == V2K 가정, 기준선 = AUTO-L1
            emit(fh, 'AUTO', 'AUTO-L1', ep)          # 기준선
            emit(fh, 'AUTO-L1', 'V2K', ep)           # 처치 I900
            emit(fh, 'AUTO-SEED-NULL', 'AUTO-SEED-NULL', ep)
        else:                     # §4-B: 선택기 대 시드-널 선택기
            _, vr = N.selector(D, ['AUTO-L1', 'V2K'], 'window_resid')
            _, vn = N.selector(D, ['AUTO', 'AUTO-SEED-NULL'], 'window_resid')
            pr, _ = N.selector(D, ['AUTO-L1', 'V2K'], 'window_resid')
            pn, _ = N.selector(D, ['AUTO', 'AUTO-SEED-NULL'], 'window_resid')
            emit(fh, 'AUTO', pn(ep), ep)             # 기준선 = 널 선택기
            emit(fh, 'AUTO-L1', pr(ep), ep)          # 처치 = 후보 선택기
            emit(fh, 'AUTO-SEED-NULL', 'AUTO-SEED-NULL', ep)
        emit(fh, 'AUTO-DUP', 'AUTO-DUP', ep)
        for c in ('AUTO-QTR', 'AUTO-QTR-L1', 'AUTO-NT'):
            emit(fh, c, c, ep)
print('wrote', OUT)
