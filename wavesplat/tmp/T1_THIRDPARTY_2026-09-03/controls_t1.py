"""`T1` v2 의 **자 검증** — 제3자 자료를 열기 **전에**, 우리 동결 코퍼스로 검정력을 보인다.

⛔이것은 처치가 아니다. 우리 자료만 쓴다(제3자 아카이브 접촉 0). 여기서 자가 이빨을
못 보이면 사전등록을 동결하지 않는다 — 그것이 v1 이 죽은 이유다.
"""
import glob, json, os, sys
import numpy as np, h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '_shared'))
sys.path.insert(0, os.path.join(HERE, '..', 'C1_MODELGAP_2026-09-03'))
import surrogate as S                                                # noqa: E402
import gates_c1 as C                                                 # noqa: E402

NSUR, SEED, NFRAME = 32, 20260903, 8


def stats_and_p(z, rng):
    """왜도·마루골비를 대체표본 귀무에 대해 검정. 반환 = (값, 초과여부) 짝."""
    z = np.asarray(z, float)
    out = {}
    for nm, fn in (('skew', S.skewness), ('ct', S.crest_trough)):
        v = fn(z)
        null = np.array([fn(S.ft_surrogate2d(z, rng)) for _ in range(NSUR)])
        # nearest-rank 95 백분위, strict 초과 (결과 전 고정)
        q = float(np.sort(null)[int(np.ceil(0.95 * NSUR)) - 1])
        out[nm] = v
        out[nm + '_q95'] = q
        out[nm + '_exceed'] = bool(v > q)
    return out


def scan(paths, group, limit):
    rows = []
    for p in paths[:limit]:
        with h5py.File(p, 'r') as f:
            Z = np.asarray(f['frames/gt_wave_field'])
        rng = np.random.default_rng(SEED)
        idx = np.linspace(0, Z.shape[0] - 1, NFRAME).astype(int)
        for j in idx:
            r = stats_and_p(Z[j], rng)
            r.update(episode=os.path.basename(p), group=group, frame=int(j),
                     shape=list(Z[j].shape))
            rows.append(r)
    return rows


def main():
    hos = sorted(glob.glob(os.path.join(C.HOS, '*.h5')))
    lin = sorted(glob.glob(os.path.join(C.LIN, 'episode_BF*comp_seed*.h5')))
    rows = scan(hos, 'hos', 6) + scan(lin, 'linear', 6)
    with open(os.path.join(HERE, 'controls_t1.jsonl'), 'w', encoding='utf-8') as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    for g in ('hos', 'linear'):
        sub = [r for r in rows if r['group'] == g]
        for nm in ('skew', 'ct'):
            ex = sum(r[nm + '_exceed'] for r in sub)
            vs = np.array([r[nm] for r in sub])
            print(f'{g:7s} {nm:5s} 초과 {ex:3d}/{len(sub):3d}   '
                  f'값 {vs.min():+.4f}/{np.median(vs):+.4f}/{vs.max():+.4f}')
    print('shape', rows[0]['shape'], '| 에피소드', len({r["episode"] for r in rows}))


if __name__ == '__main__':
    main()
