"""`T1b` 설계용 — **처치 코퍼스의 귀무 폭이 프레임 수에 따라 어떻게 좁아지는가**.

★`T1` 의 교훈: 검정력 표를 **라벨 확률로만** 적고 처치 자료의 귀무 폭으로는 안 적었다.
여기서는 그것을 **먼저 잰다**. ⛔이 스크립트가 여는 파일은 `T1` 이 **이미 연** 블록뿐이다
(블라인드가 이미 풀린 자료) — `T1b` 의 PRIMARY 가 될 **새 파일 12개는 열지 않는다**.

⚠프레임은 dt=1 s 로 상관돼 있으므로 `1/√N` 스케일링이 그대로 성립하지 않는다.
그래서 **가정하지 않고 잰다.**
"""
import io, json, os, sys, zipfile
import numpy as np, h5py

HERE = os.path.dirname(os.path.abspath(__file__))
T1 = os.path.join(HERE, '..', 'T1_THIRDPARTY_2026-09-03')
sys.path.insert(0, T1)
import gates_t1 as G1                                                # noqa: E402
sys.path.insert(0, G1.SHARED)
import surrogate as S                                                # noqa: E402

SEEN = ['Data/mixed_data/angle10h0.5.mat', 'Data/mixed_data/angle10h2.0.mat',
        'Data/mixed_data/angle10h5.0.mat']      # ⛔전부 T1 이 이미 연 블록
NF = [24, 48, 96, 192, 384, 502]
NSUR = 32


def main():
    out = []
    with zipfile.ZipFile(G1.ZIP) as z:
        for n in SEEN:
            with h5py.File(io.BytesIO(z.read(n)), 'r') as f:
                E = np.asarray(f['eta']).squeeze()
            for nf in NF:
                idx = np.linspace(0, E.shape[0] - 1, min(nf, E.shape[0])).astype(int)
                fr = [E[j] for j in idx]
                pool = np.concatenate([a.ravel() for a in fr])
                v = float(S.skewness(pool))
                rng = np.random.default_rng(G1.SEED)
                null = [float(S.skewness(np.concatenate(
                    [S.ft_surrogate2d(a, rng).ravel() for a in fr]))) for _ in range(NSUR)]
                q = float(np.sort(null)[int(np.ceil(0.95 * NSUR)) - 1])
                out.append(dict(block=os.path.basename(n), n_frames=len(idx),
                                skew=v, q95=q, null_sd=float(np.std(null)),
                                exceed=bool(v > q)))
                print(f'  {os.path.basename(n):18s} nf={len(idx):3d}  skew {v:+.4f}  '
                      f'q95 {q:.4f}  sd {np.std(null):.4f}  '
                      f"{'초과' if v > q else ''}")
    json.dump(out, open(os.path.join(HERE, 'power_t1b.json'), 'w'), indent=1)
    print('\n★ 프레임 수 대 귀무 폭 (같은 블록 안에서):')
    for b in {r['block'] for r in out}:
        s = sorted((r for r in out if r['block'] == b), key=lambda r: r['n_frames'])
        r0 = s[0]
        print(f'  {b:18s} ' + ' '.join(
            f"{r['n_frames']}:{r['q95']/r0['q95']:.2f}x" for r in s))


if __name__ == '__main__':
    main()
