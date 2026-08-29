"""M3 `G-8` FAIL 진단 — ★**추정 런 0회 · 결과 파일 0회 열람.**

읽는 것: `data_m3/**/*.h5`(GT 필드·성분) · `gen_manifest_m3.json` · `gates_m3.py` 상수.
⛔읽지 않는 것: `results_m3_*.jsonl`(존재하지 않는다) · `verdict_m3.json`(존재하지 않는다).

`AMENDMENT_1.md` 부록 A 의 수치를 **여기서 다시 만들고**, 문서와 어긋나면 `AssertionError`.
(본보기 = `tmp/B2_REAL_2026-08-23/depth_recheck.py` — 저장소 규약 "처방도 검증하라".)

⛔이 파일은 **판정 코드가 아니다**. 게이트·문턱·선별 규칙에 손대지 않는다.

★**AMENDMENT_1 부록 B (2026-08-29) 이후의 두 가지 변경** — ⛔수치·assert 는 **무변**이다:
 ① 해석 분해를 자기 사본에서 **`gates_m3` import 로 전환**했다. 부록 B 가 그 계산을
    선별 술어 ③ 으로 승격시켰으므로, 사본을 남기면 저장소가 이름 붙인 반복 결함
    (*"같은 기능이 두 파일에 중복 구현돼 한쪽만 계측"*)을 이 진단기가 스스로 저지른다.
    ⟹ 이제 `[B]` 의 `max|해석 − 실측| ≤ 4.44e-16` 은 **선별 술어 자체의 검산**이다.
 ② 표본을 `DIAG_SEEDS` **명시 상수**로 고정했다. 이 진단이 기술하는 것은 **진단 시점의
    48 에피소드(1695–1702)** 이고, 부록 B 이후의 `gates_m3.FROZEN_SEEDS` 와는 다르다.
    거부된 `SPR5|30|1702` 는 삭제되지 않고 `REJECTED_seed1702/` 로 옮겨졌으며(증거 보존),
    `_diag_path()` 가 두 위치를 모두 찾는다.

    python3 tmp/M3_SUPPORT_2026-08-28/diag_g8_2026-08-29.py
"""
import glob
import json
import math
import os
import statistics as st
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import gates_m3 as G                                           # noqa: E402

HS = 1.2                                                       # BF4 공칭 Hs
G8_THRESH = G.HS4STD_MAX_DEV                                   # ⛔손복사 금지 — 단일 심볼
assert G8_THRESH == 0.25, G8_THRESH                            # 부록 B 에서도 **불변**

# ★진단 시점(2026-08-29, 부록 A)의 표본 — ⛔`G.FROZEN_SEEDS` 가 아니다.
#   부록 B 가 `SPR5|30` 의 8번째를 1702→1703 으로 바꿨으므로, 이 진단이 기술하는 코퍼스는
#   **여섯 칸 전부 1695–1702** 인 부록 B **이전**의 48 에피소드다.
DIAG_SEEDS = {f'{c}|{n}': tuple(range(1695, 1703))
              for c in G.CELL_ORDER for n in G.NCOMPS}
REJECTED_DIR = os.path.join(HERE, 'REJECTED_seed1702')


def _diag_path(cell, nc, sd):
    """거부분은 `REJECTED_seed1702/` 로 **이동**됐다(삭제 아님) — 두 위치를 다 찾는다."""
    fn = G.fname(cell, nc, sd)
    p = os.path.join(G.cell_dir(cell), fn)
    if os.path.exists(p):
        return p
    alt = glob.glob(os.path.join(REJECTED_DIR, '**', fn), recursive=True)
    assert alt, f'증거 파일이 두 위치 어디에도 없다: {fn}'
    return alt[0]


# ── 0. blind 확인 ────────────────────────────────────────────────────────────
def assert_blind():
    res = glob.glob(os.path.join(HERE, 'results_m3_*.jsonl'))
    ver = glob.glob(os.path.join(HERE, 'verdict_m3.json'))
    assert not res and not ver, f'결과가 존재한다 — 이 진단은 blind 가 아니다: {res + ver}'
    return len(res) + len(ver)


# ── 1. 해석 분해 — ★**`gates_m3` 에서 가져온다**(부록 B: 중복 구현 금지) ──────
#    이 계산은 이제 선별 술어 ③ 의 본체이고, 단일출처는 `gates_m3.hs4std_decompose` 다.
_window = G._gt_window
decompose = G.hs4std_decompose


def from_components(cell, ncomp, seed, win=None):
    """★**생성 전** 경로 — `gates_m3.components()` 만 쓴다. 에피소드 파일을 안 읽는다."""
    c = G.components(cell, ncomp, seed)
    return decompose(np.array([x.A for x in c]), np.array([x.omega for x in c]),
                     np.array([x.kx for x in c]), np.array([x.ky for x in c]),
                     np.array([x.phi for x in c]), win)


def from_file(path, win=None):
    """생성 **후** 경로 — 저장 h5 의 성분으로 같은 분해 + 실측 4·std."""
    import h5py
    with h5py.File(path, 'r') as h:
        C = h['config/wave_components'][:]
        s4 = 4 * float(np.std(h['frames/gt_wave_field'][:]))
    d = decompose(C[:, 0], C[:, 1], C[:, 2], C[:, 3], C[:, 4], win)
    d['s4_measured'] = s4
    return d


def main():
    print(f'[0] blind 확인 — 결과/판정 파일 {assert_blind()}개')
    man = json.load(open(os.path.join(HERE, 'gen_manifest_m3.json')))
    meas = {(v['cell'], v['ncomp'], v['seed']): v['gt_Hs_4std'] for v in man.values()
            if v['seed'] in DIAG_SEEDS[f"{v['cell']}|{v['ncomp']}"]}
    assert len(meas) == G.N_EP, len(meas)   # ★진단 시점 표본(1695–1702)만 센다
    win = _window()

    # ── A. 게이트 재현 ───────────────────────────────────────────────────────
    dev = {k: abs(v / HS - 1) for k, v in meas.items()}
    worst = max(dev, key=dev.get)
    top3 = sorted(dev, key=dev.get, reverse=True)[:3]
    print(f'[A] G-8 max|4·std/Hs − 1| = {dev[worst]:.4f}  @ {worst}')
    assert abs(dev[worst] - 0.2595) < 5e-5 and worst == ('SPR5', 30, 1702)
    assert [round(dev[k], 4) for k in top3] == [0.2595, 0.1217, 0.0925]
    assert top3[1] == ('DEP12', 30, 1695) and top3[2] == ('DIR180', 30, 1699)
    for c, m in (('DIR180', 0.0925), ('SPR5', 0.2595), ('DEP12', 0.1217)):
        assert abs(max(v for k, v in dev.items() if k[0] == c) - m) < 5e-5, c
    print(f'    셀별 최대 = DIR180 0.0925 · SPR5 0.2595 · DEP12 0.1217 '
          f'(문턱 {G8_THRESH} 초과분 {dev[worst] - G8_THRESH:.4f} '
          f'= {100 * (dev[worst] - G8_THRESH) / G8_THRESH:.1f}%)')

    # ── B. 편차의 출처: 진폭 draw 가 아니라 **간섭 잔차** ───────────────────
    mx_an, mx_sum, mx_diag = 0.0, 0.0, 0.0
    for cell in G.CELL_ORDER:
        for nc in G.NCOMPS:
            for sd in DIAG_SEEDS[f'{cell}|{nc}']:
                d = from_file(_diag_path(cell, nc, sd), win)
                mx_an = max(mx_an, abs(d['s4'] - d['s4_measured']))
                mx_sum = max(mx_sum, abs(d['V_sum']))
                mx_diag = max(mx_diag, abs(4 * math.sqrt(d['V_diag']) - HS))
    print(f'[B] 분해가 실측 4·std 를 재현: max|해석 − 실측| = {mx_an:.3e}')
    print(f'    ΣA²/2 는 **정확히** 공칭 Hs: max|4·√(ΣA²/2) − 1.2| = {mx_diag:.3e}')
    print(f'    쌍-합 항은 무시가능: max|V_sum| = {mx_sum:.3e}')
    assert mx_an < 1e-12 and mx_diag < 1e-12 and mx_sum < 1e-3
    print('    ⟹ ★편차의 100%가 V_diff(쌍-차 간섭 잔차)다 — 진폭 draw 는 기여 0.')

    # ── C. 왜 SPR5 인가 — 성분이 **비트 동일**한 짝 비교 ────────────────────
    import h5py
    pair_ok, rows = True, []
    for nc in G.NCOMPS:
        for sd in DIAG_SEEDS[f'SPR5|{nc}']:
            cs = {}
            for cell in G.CELL_ORDER:
                with h5py.File(_diag_path(cell, nc, sd), 'r') as h:
                    cs[cell] = h['config/wave_components'][:]
            a = cs['SPR5']
            for cell in ('DIR180', 'DEP12'):
                b = cs[cell]
                pair_ok &= bool(np.allclose(a[:, 0], b[:, 0])      # A
                                and np.allclose(a[:, 1], b[:, 1])  # ω
                                and np.allclose(a[:, 4], b[:, 4]))  # φ
            rows.append((nc, sd, dev[('SPR5', nc, sd)], dev[('DIR180', nc, sd)]))
    print(f'[C] 세 셀의 (A, ω, φ) 가 (ncomp, seed) 마다 비트 동일: {pair_ok} '
          '⟹ 셀 간 차이는 **방향 draw 뿐**')
    assert pair_ok
    k = ('SPR5', 30, 1702)
    print(f"    seed1702·30comp: 5°={dev[k]:.4f} vs 30°(DIR180)={dev[('DIR180', 30, 1702)]:.4f}"
          f" vs 30°(DEP12)={dev[('DEP12', 30, 1702)]:.4f}  → {dev[k] / dev[('DIR180', 30, 1702)]:.1f}배")
    assert abs(dev[('DIR180', 30, 1702)] - 0.0448) < 5e-5
    assert abs(dev[('DEP12', 30, 1702)] - 0.0084) < 5e-5
    w = sum(1 for _, _, s, d in rows if s > d)
    print(f'    짝 |dev|: SPR5 > DIR180 {w}/16 '
          f'(부호검정 p={sum(math.comb(16, i) for i in range(w, 17)) / 2 ** 16:.4f} — ⚠위치가 아니라 산포다)')
    assert w == 11

    # ── D. 기제: 좁은 방향폭이 **y 방향 평균화를 없앤다** ────────────────────
    print('[D] 쌍-차 창 평균 인자 (진폭가중 평균):')
    for nc in G.NCOMPS:
        for cell in G.CELL_ORDER:
            ds = [from_components(cell, nc, s, win)
                  for s in DIAG_SEEDS[f'{cell}|{nc}']]
            r = [meas[(cell, nc, s)] / HS for s in DIAG_SEEDS[f'{cell}|{nc}']]
            print(f'    {cell:7s} nc={nc:2d}  <|Dy|>={st.mean(x["Dy"] for x in ds):.4f}'
                  f'  <|Dx·Dy·Dt|>={st.mean(x["DxDyDt"] for x in ds):.4f}'
                  f'  sd(비)={st.stdev(r):.4f}  평균(비)={st.mean(r):.4f}')

    # ── E. 전수 census — 선언 풀 [1695,1757] 6칸 × 63시드 ───────────────────
    t0 = time.time()
    census, n_bad = {}, 0
    for cell in G.CELL_ORDER:
        for nc in G.NCOMPS:
            rs = [from_components(cell, nc, s, win)['s4'] / HS for s in G.SEED_POOL]
            bad = [s for s, r in zip(G.SEED_POOL, rs) if abs(r - 1) > G8_THRESH]
            n_bad += len(bad)
            census[(cell, nc)] = (st.stdev(rs), max(abs(r - 1) for r in rs), bad)
    el = time.time() - t0
    print(f'[E] 전수 census {len(G.CELL_ORDER) * len(G.NCOMPS) * len(G.SEED_POOL)}개 '
          f'후보를 {el:.2f} s 에 판정 (생성 137.5 분 대비 {137.5 * 60 / el:.0f}배)')
    for (cell, nc), (sd_, mx, bad) in census.items():
        print(f'    {cell:7s} nc={nc:2d}  sd={sd_:.4f}  max|dev|={mx:.4f}'
              f'  0.25={G8_THRESH / sd_:.2f}σ  초과={bad}')
    assert n_bad == 1 and census[('SPR5', 30)][2] == [1702], (n_bad, census[('SPR5', 30)][2])
    print(f'    ⟹ ★풀 전체 378 후보 중 초과 **1개**(0.26%) — SPR5|30|1702 뿐.')

    # ── F. 대체안 검증 (⛔미적용 — 제안의 근거만 만든다) ────────────────────
    print('[F] 대체 후보 (풀 순서상 1702 다음 = 1703):')
    for sd in (1702, 1703, 1704):
        d = from_components('SPR5', 30, sd, win)
        s = G.screen('SPR5', 30, sd)
        a = dict(A=np.array([x.A for x in G.components('SPR5', 30, sd)]),
                 omega=np.array([x.omega for x in G.components('SPR5', 30, sd)]),
                 kx=np.array([x.kx for x in G.components('SPR5', 30, sd)]),
                 ky=np.array([x.ky for x in G.components('SPR5', 30, sd)]),
                 depth=G.CELLS['SPR5']['depth'])
        dc = G.derived_coords(a)
        print(f'    seed{sd}  |dev|={abs(d["s4"] / HS - 1):.4f}  '
              f'circ_std={dc["circ_std_deg"]:.2f}°  mesh={s["mesh_verdict"]}  q_ok={s["q_ok"]}')
    r1703 = from_components('SPR5', 30, 1703, win)['s4'] / HS
    assert abs(abs(r1703 - 1) - 0.0938) < 5e-4
    new = {k: v for k, v in meas.items() if k != ('SPR5', 30, 1702)}
    new[('SPR5', 30, 1703)] = r1703 * HS
    print(f'    대체 후 G-8 max|dev| = {max(abs(v / HS - 1) for v in new.values()):.4f} '
          f'(≤ {G8_THRESH})')
    assert max(abs(v / HS - 1) for v in new.values()) < G8_THRESH

    # ── G. 동결 md5 대조 (부수 발견) ────────────────────────────────────────
    import hashlib
    # ★규약(`tmp/_audit/check_batch_artifacts.py:225-231`): 같은 경로가 여러 번 적히면
    #   **마지막 줄이 현재 동결**이고 앞엣것은 이력이다(재동결 사실을 지우지 않기 위함).
    #   ⟹ 여기서도 마지막 줄로 대조하고, 이력 줄 수를 따로 보고한다.
    latest, hist = {}, {}
    for line in open(os.path.join(HERE, 'FROZEN_md5.txt')):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        h, rel = line.split(None, 1)
        if rel in latest:
            hist.setdefault(rel, []).append(latest[rel])
        latest[rel] = h
    bad = [rel for rel, h in latest.items()
           if os.path.exists(os.path.join(ROOT, rel))
           and hashlib.md5(open(os.path.join(ROOT, rel), 'rb').read()).hexdigest() != h]
    print(f'[G] FROZEN_md5 대조: 불일치 {len(bad)}개 {bad}')
    print(f'    재동결 이력 {len(hist)}개 (⛔지우지 않는다): {sorted(hist)}')

    print('\n★전건 통과 — `AMENDMENT_1.md` 부록 A 의 수치가 재현된다. 추정 런 0회.')


if __name__ == '__main__':
    main()
