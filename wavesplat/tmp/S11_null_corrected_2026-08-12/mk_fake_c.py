"""시험 C: §4-B 의 **실제 구조**(두 선택기가 팔 하나를 공유) 를 동결 채점기에 먹인다."""
import json, sys
sys.path.insert(0, '/home/vexit/wavesplat/wavesplat/tmp/S11_null_corrected_2026-08-12')
import neutral_rederive as N
D, reg, err = N.load()
pr, _ = N.selector(D, ['AUTO', 'V2K'], 'window_resid')            # 처치 선택기
pn, _ = N.selector(D, ['AUTO', 'AUTO-SEED-NULL'], 'window_resid') # 널 선택기 (팔 A 공유)
def emit(fh, cellname, src, ep):
    for s, r in D[src][ep].items():
        q = dict(r); q['cell'] = cellname; q['job_id'] = f'{cellname}|{ep}|{s}'
        fh.write(json.dumps(q) + '\n')
with open(sys.argv[1], 'w') as fh:
    for ep in sorted(reg):
        emit(fh, 'AUTO', pn(ep), ep)
        emit(fh, 'AUTO-L1', pr(ep), ep)
        emit(fh, 'AUTO-SEED-NULL', 'AUTO-SEED-NULL', ep)
        emit(fh, 'AUTO-DUP', pn(ep), ep)
        for c in ('AUTO-QTR', 'AUTO-QTR-L1', 'AUTO-NT'):
            emit(fh, c, c, ep)
