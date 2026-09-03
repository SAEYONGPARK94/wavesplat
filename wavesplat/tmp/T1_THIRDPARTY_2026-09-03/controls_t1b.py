"""자 검증 2차 — **프레임을 모아** 통계량을 잰다(유효 표본 확대).

1차(`controls_t1.py`)에서 한 프레임(60×80)짜리 모멘트 검정은 **양성 대조 0/48** 로
검정력이 없었다. 귀무 대체표본의 왜도 산포(q95 0.43)가 신호(중앙 0.085)보다 5배 컸다.
여기서는 에피소드의 프레임 전부를 이어 붙여 한 통계량을 만들고, 대체표본도 **프레임마다
독립으로 위상무작위화한 뒤 이어 붙인다**(같은 귀무).
"""
import glob, json, os, sys
import numpy as np, h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '_shared'))
sys.path.insert(0, os.path.join(HERE, '..', 'C1_MODELGAP_2026-09-03'))
import surrogate as S                                                # noqa: E402
import gates_c1 as C                                                 # noqa: E402

NSUR, SEED, NFRAME = 32, 20260903, 24


def episode_test(Z, idx, rng):
    pool = np.concatenate([Z[j].ravel() for j in idx])
    out = {}
    for nm, fn in (('skew', S.skewness), ('ct', S.crest_trough)):
        v = fn(pool)
        null = []
        for _ in range(NSUR):
            sp = np.concatenate([S.ft_surrogate2d(Z[j], rng).ravel() for j in idx])
            null.append(fn(sp))
        null = np.sort(np.array(null))
        q = float(null[int(np.ceil(0.95 * NSUR)) - 1])
        out[nm] = float(v); out[nm + '_q95'] = q; out[nm + '_exceed'] = bool(v > q)
        out[nm + '_null_sd'] = float(null.std())
    return out


def main():
    rows = []
    for grp, paths in (('hos', sorted(glob.glob(os.path.join(C.HOS, '*.h5')))[:8]),
                       ('linear', sorted(glob.glob(os.path.join(
                           C.LIN, 'episode_BF*comp_seed*.h5')))[:8])):
        for p in paths:
            with h5py.File(p, 'r') as f:
                Z = np.asarray(f['frames/gt_wave_field'])
            idx = np.linspace(0, Z.shape[0] - 1, min(NFRAME, Z.shape[0])).astype(int)
            r = episode_test(Z, idx, np.random.default_rng(SEED))
            r.update(episode=os.path.basename(p), group=grp, n_frames=len(idx),
                     n_pts=int(len(idx) * Z.shape[1] * Z.shape[2]))
            rows.append(r); 
    with open(os.path.join(HERE, 'controls_t1b.jsonl'), 'w', encoding='utf-8') as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    for g in ('hos', 'linear'):
        sub = [r for r in rows if r['group'] == g]
        for nm in ('skew', 'ct'):
            ex = sum(r[nm + '_exceed'] for r in sub)
            v = np.array([r[nm] for r in sub]); q = np.array([r[nm + '_q95'] for r in sub])
            print(f'{g:7s} {nm:5s} 초과 {ex}/{len(sub)}  값중앙 {np.median(v):+.4f}  '
                  f'귀무q95중앙 {np.median(q):+.4f}  귀무sd중앙 '
                  f'{np.median([r[nm+"_null_sd"] for r in sub]):.4f}')
    print('점 수/에피소드', rows[0]['n_pts'])


if __name__ == '__main__':
    main()
