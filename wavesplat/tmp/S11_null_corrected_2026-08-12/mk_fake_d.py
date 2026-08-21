"""시험 D: S10 원 후보 SEL{AUTO-L1,V2K} vs AUTO-L1 (혼잡 유효n=1) 를 동결 채점기에."""
import json, sys
sys.path.insert(0, '/home/vexit/wavesplat/wavesplat/tmp/S11_null_corrected_2026-08-12')
import neutral_rederive as N
D, reg, err = N.load()
pr, _ = N.selector(D, ['AUTO-L1', 'V2K'], 'window_resid')
def emit(fh, c, src, ep):
    for s, r in D[src][ep].items():
        q = dict(r); q['cell'] = c; q['job_id'] = f'{c}|{ep}|{s}'; fh.write(json.dumps(q) + '\n')
with open(sys.argv[1], 'w') as fh:
    for ep in sorted(reg):
        emit(fh, 'AUTO', 'AUTO-L1', ep)
        emit(fh, 'AUTO-L1', pr(ep), ep)
        emit(fh, 'AUTO-SEED-NULL', 'AUTO-SEED-NULL', ep)
        emit(fh, 'AUTO-DUP', 'AUTO-L1', ep)
        for c in ('AUTO-QTR', 'AUTO-QTR-L1', 'AUTO-NT'):
            emit(fh, c, c, ep)
