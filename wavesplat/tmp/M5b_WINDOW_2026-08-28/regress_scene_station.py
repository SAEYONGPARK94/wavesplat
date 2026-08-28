"""★`scene_station.py` **비트동일 회귀 시험** — 2026-08-28 `cap_mode` 추가분.

    python3 tmp/M5b_WINDOW_2026-08-28/regress_scene_station.py          # 전건(12×3)
    python3 tmp/M5b_WINDOW_2026-08-28/regress_scene_station.py --quick  # 3 레코드

⛔**추정기 0회 · GPU 0회 · 자료 추가 0건.** 장면 빌더만 돌린다.

## 왜 이 시험이 필요한가
`tmp/B2_REAL_2026-08-23/scene_station.py` 는 **약 30곳이 임포트**한다
(`B-2`·`B-3`·`B-5`·`B-6`·`B-7`·`B-9`·`B-10`·`B-14`·`B-17`·`B-18`·`B-26`·`M4`·`M6`
그리고 배포 테스트 `wavesplat/tests/test_config_exposure.py`).
`M5b` 가 그 파일에 `cap_mode` 를 넣었으므로, **기본값 경로가 한 비트도 안 움직였다**는 것을
기계가 매번 다시 증명해야 한다. 선례 = 같은 파일의 `with_future_obs`
(*"기본 False 이고 그때는 반환 dict 이 비트 동일"*).

## 네 겹
  **L1 (전건)** 수정 전 소스(git blob `27c35dc…`, md5 대조)와 수정 후 소스가
        **기본 인자**에서 반환 dict 의 **모든 키**를 비트동일하게 낸다 — 12 레코드 × 3 창.
  **L2 (변형 표본)** 기본값이 아닌 인자에서도 비트동일 — `W`·`m_cap`·`with_future_obs`·
        `lead_max`. ⟹ *"기본값만 안 건드렸다"* 가 아니라 **`cap_mode` 를 안 주면 전부 그대로**.
  **L3 (외부 앵커)** 수정 후 기본 경로의 `obs_hash` 가 `B-2` **동결 결과 행**과 12/12 일치.
        ⟹ 두 소스를 다 고친 실수까지 잡는다(L1·L2 는 상대 비교라 그 경우 조용히 통과한다).
  **L4 (처치가 실제로 다르다)** `cap_mode='end_anchored'` 는
        (a) `n_obs` 를 정확히 `m_cap` 으로 유지하고,
        (b) 끝 시각을 **cap 이전 pool 의 끝 시각**과 같게 만들며,
        (c) `'uniform'` 과 **적어도 한 셀에서 다르다**.
        ⛔(c) 가 없으면 이 배치의 처치는 **no-op** 이다(오늘 `M4 G-13` 의 부류).

⚠**L1·L2 는 `cap_mode` 를 넘기지 않는다** — 넘기면 수정 전 소스가 `TypeError` 를 낸다.
"""
import importlib.util
import os
import sys
import warnings

warnings.filterwarnings('ignore')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for _p in (HERE, ROOT, os.path.join(ROOT, 'scripts'),
           os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23'),
           os.path.join(ROOT, 'tmp', 'B3_REFIT_2026-08-23')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(ROOT)

import numpy as np                                            # noqa: E402
import gates_m5b as G                                         # noqa: E402
from scene_station import scene_station as post               # noqa: E402


def load_pre():
    """수정 전 판본을 git blob 에서 꺼내 **깊이가 같은 경로**에 임시로 쓰고 임포트한다.

    ⛔사본을 저장소에 남기지 않는다(같은 것이 N곳 금지)."""
    src = G.scene_pre_source()
    assert 'cap_mode' not in src, 'blob 핀이 틀렸다'
    path = os.path.join(HERE, '_scene_station_pre_regress.py')
    with open(path, 'w') as fh:
        fh.write(src)
    try:
        spec = importlib.util.spec_from_file_location('scene_station_pre_regress', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        os.remove(path)
    return mod.scene_station


def diff_dict(a, b):
    """두 장면 dict 이 **비트동일**한지. 다르면 (키, 사유) 목록을 낸다."""
    bad = []
    if set(a) != set(b):
        bad.append(('<keys>', f'{sorted(set(a) ^ set(b))}'))
        return bad
    for k in sorted(a):
        x, y = a[k], b[k]
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            xa, ya = np.asarray(x), np.asarray(y)
            if xa.shape != ya.shape or xa.dtype != ya.dtype:
                bad.append((k, f'shape/dtype {xa.shape}{xa.dtype} vs {ya.shape}{ya.dtype}'))
            elif not np.array_equal(xa, ya, equal_nan=np.issubdtype(xa.dtype, np.floating)):
                bad.append((k, 'array bits differ'))
        elif isinstance(x, float) and isinstance(y, float):
            if np.float64(x).tobytes() != np.float64(y).tobytes():
                bad.append((k, f'{x!r} vs {y!r}'))
        elif x != y:
            bad.append((k, f'{x!r} vs {y!r}'))
    return bad


def main():
    quick = '--quick' in sys.argv
    recs = G.RECORDS[:3] if quick else G.RECORDS
    wins = [G.PRIMARY_WIN] if quick else G.WINDOWS
    pre = load_pre()
    fails, n = [], 0

    print(f'수정 전 blob={G.SCENE_PRE_BLOB[:12]} md5={G.SCENE_PRE_MD5}  '
          f'(레코드 {len(recs)} · 창 {wins})')

    # ── L1 전건: 기본 인자에서 반환 dict 전 키 비트동일 ─────────────────────
    print('L1 기본 인자 비트동일 ...', flush=True)
    for rec in recs:
        for w in wins:
            a, b = pre(rec, win=w), post(rec, win=w)
            d = diff_dict(a, b)
            n += 1
            if d:
                fails.append(f'L1 {rec} w{w}: {d[:3]}')
            print(f'  {rec:7s} w{w} {"OK" if not d else "⛔ " + str(d[:2])}', flush=True)

    # ── L2 변형 표본: 기본값이 아닌 인자에서도 비트동일 ─────────────────────
    print('L2 변형 인자 비트동일 ...', flush=True)
    variants = [dict(W=8.0), dict(W=4.0), dict(m_cap=300), dict(m_cap=G.POOL_NO_CAP),
                dict(with_future_obs=True), dict(lead_max=6.0), dict(m_scan=6)]
    for rec in recs[:3]:
        for kw in variants:
            a, b = pre(rec, win=G.PRIMARY_WIN, **kw), post(rec, win=G.PRIMARY_WIN, **kw)
            d = diff_dict(a, b)
            n += 1
            if d:
                fails.append(f'L2 {rec} {kw}: {d[:3]}')
            print(f'  {rec:7s} {str(kw):28s} {"OK" if not d else "⛔ " + str(d[:2])}',
                  flush=True)

    # ── L3 외부 앵커: 수정 후 기본 경로 == B-2 동결 행 ──────────────────────
    print('L3 외부 앵커 (B-2 동결 obs_hash) ...', flush=True)
    ref = G.b2_scene_ref()
    if not ref or len(ref) != len(G.RECORDS):
        fails.append(f'L3 외부 참조가 비었다: {len(ref)}/{len(G.RECORDS)}')
    neq = 0
    for rec in recs:
        sc = post(rec, win=G.FROZEN_B2_WIN)
        h = G.hash_obs(sc['P'], sc['z'])
        eq = ref.get(rec) == h
        neq += int(bool(eq))
        n += 1
        if not eq:
            fails.append(f'L3 {rec}: b2={ref.get(rec)} mine={h}')
    print(f'  {neq}/{len(recs)} 일치')

    # ── L4 처치가 실제로 다르다 + 불변식 ────────────────────────────────────
    print("L4 cap_mode='end_anchored' 불변식 + 실제 차이 ...", flush=True)
    n_diff = 0
    for rec in recs:
        for w in wins:
            u = post(rec, win=w)                                  # 기본 = uniform
            e = post(rec, win=w, cap_mode='end_anchored')
            n += 1
            pl = post(rec, win=w, m_cap=G.POOL_NO_CAP)            # cap 이전 pool
            te_pool = float(np.asarray(pl['P'], float)[:, 2].max())
            te_e = float(np.asarray(e['P'], float)[:, 2].max())
            te_u = float(np.asarray(u['P'], float)[:, 2].max())
            if len(e['z']) != len(u['z']):
                fails.append(f'L4 {rec} w{w}: n_obs {len(e["z"])} != {len(u["z"])}')
            if abs(te_e - te_pool) > G.T_EQ_TOL:
                fails.append(f'L4 {rec} w{w}: end_anchored 끝시각 {te_e} != pool {te_pool}')
            if abs(te_e - float(e['t0'])) > G.T_EQ_TOL:
                fails.append(f'L4 {rec} w{w}: end_anchored 끝시각 != t0 (최신 프레임 비었나)')
            if G.hash_obs(e['P'], e['z']) != G.hash_obs(u['P'], u['z']):
                n_diff += 1
            tag = ('★uniform 과 끝시각 다름' if abs(te_u - te_pool) > G.T_EQ_TOL else '')
            print(f'  {rec:7s} w{w} n={len(e["z"])} end={te_e:.4f} pool={te_pool:.4f} '
                  f'{tag}', flush=True)
    if n_diff == 0:
        fails.append("L4 ⛔end_anchored 가 uniform 과 **한 셀도 다르지 않다** = 처치 no-op")

    print(f'\n비교 {n} 건 · 차이 있는 셀 {n_diff} (L4)')
    print(f'regress_scene_station: {"PASS" if not fails else "FAIL"} ({len(fails)} 실패)')
    for f in fails:
        print('  -', f)
    return 0 if not fails else 1


if __name__ == '__main__':
    sys.exit(main())
