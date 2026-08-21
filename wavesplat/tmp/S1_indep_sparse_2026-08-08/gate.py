"""S1 실행 전 게이트 (PREREG §4).

G-1 독립성 · G-2 메시 · G-3 충돌없음 을 신규 두 집합에 대해 확인한다.
독립성 규약은 `tmp/K6_run_2026-08-07/independence_gate.py` 의 `report()` 를
**import 해서 그대로** 쓴다(복사 금지 — 규약이 표류하면 게이트가 의미를 잃는다).

⛔5축(진폭·omega·위상·방향·진실면)이 **전부 16종**이 아니면 S1 을 돌리지 않는다.
이 스크립트는 S1 결과 파일을 읽지 않는다(존재하지 않는다).
"""
import os
import sys

import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
K6 = os.path.abspath(os.path.join(HERE, '..', 'K6_run_2026-08-07'))
sys.path.insert(0, K6)

from independence_gate import report  # noqa: E402  (frozen convention, imported)

import numpy as np  # noqa: E402
import itertools  # noqa: E402

TOL = 1e-4

CROWDED_DIR = 'data_indep_crowded_2026-08-08'
SPARSE_DIR = 'data_indep_sparse_2026-08-08'
CROWDED16 = [f'episode_{bf}_{nc}comp_seed{s}_uniform_ra_rf_ds.h5'
             for bf in ('BF4', 'BF5') for nc in (15, 30) for s in range(20, 24)]
SPARSE16 = [f'episode_{bf}_{nc}comp_seed{s}_uniform_ra_rf_ds.h5'
            for bf in ('BF4', 'BF5') for nc in (2, 5) for s in range(20, 24)]


def blocked_draws(vecs, labels):
    """AMENDMENT 1 F8 — **길이 블록 안에서** draw 를 센다 (NEUTRAL 지적).

    frozen `independence_gate.n_draws` 는 길이가 다르면 무조건 다른 draw 로 센다.
    혼잡16 = 15comp 8 + 30comp 8 이라 두 블록은 **원리적으로 충돌 불가** ⟹ "16종 통과" 가
    실제로는 블록당 8종만 보증한다. 여기서는 블록별로 세고, 블록마다 그 블록의 에피소드 수와
    같은지 요구한다. (⚠블록 간 독립성은 여전히 **측정되지 않는다** — 길이가 달라 비교 자체가
    정의되지 않기 때문. 그래서 '보증 = 블록당' 이라고 정확히 적는다.)
    """
    blocks = {}
    for v, lab in zip(vecs, labels):
        blocks.setdefault(len(v), []).append((v, lab))
    out = {}
    for L, items in sorted(blocks.items()):
        seen = []
        for v, _ in items:
            if not any(np.allclose(v, s, atol=TOL, rtol=0) for s in seen):
                seen.append(v)
        out[L] = (len(seen), len(items))
    return out


def block_report(data_dir, eps, label):
    import h5py as _h5
    comps, flds = [], []
    for ep in eps:
        with _h5.File(os.path.join(REPO, data_dir, ep), 'r') as f:
            comps.append(np.array(f['config/wave_components']))
            flds.append(np.array(f['frames/gt_wave_field']))
    axes = {
        '진폭 A': [c[:, 0] for c in comps],
        'omega': [c[:, 1] for c in comps],
        '위상 phi': [c[:, 4] for c in comps],
        '방향 theta': [np.arctan2(c[:, 3], c[:, 2]) for c in comps],
    }
    print(f'\n### G-1b 길이블록 내 draw  {label}')
    ok = True
    for name, vecs in axes.items():
        b = blocked_draws(vecs, eps)
        txt = '  '.join(f'len{L}: {d}/{n}' for L, (d, n) in b.items())
        good = all(d == n for d, n in b.values())
        ok &= good
        print(f'  {name:12s} {txt}{"  ✅" if good else "  ⛔"}')
    fl = blocked_draws([f.ravel() for f in flds], eps)
    good = all(d == n for d, n in fl.values())
    ok &= good
    print(f'  {"진실면":12s} ' + '  '.join(f'len{L}: {d}/{n}' for L, (d, n) in fl.items())
          + ('  ✅' if good else '  ⛔'))
    phi_blocks = blocked_draws(axes['위상 phi'], eps)
    per_block = min(n for _, n in phi_blocks.values())
    print(f'  ⟹ 보증 = **블록당 {per_block} 종** (블록 간 독립성은 길이가 달라 측정 불가)')
    return ok


def mesh_and_provenance(data_dir, eps, label):
    """G-2 + 손잡이 출처: 세 손잡이가 정말 켜진 채로 저장됐는지 파일에서 확인한다."""
    print(f'\n### G-2/G-3  {label}  ({data_dir})')
    bad_mesh, bad_knob, lam = [], [], []
    for ep in eps:
        p = os.path.join(REPO, data_dir, ep)
        with h5py.File(p, 'r') as f:
            a = dict(f['config'].attrs)
            if str(a.get('mesh_verdict', '?')) != 'ADEQUATE':
                bad_mesh.append((ep, a.get('mesh_verdict')))
            knobs = tuple(bool(a.get(k, False)) for k in
                          ('random_amplitudes', 'random_frequencies', 'decorrelate_stream'))
            if knobs != (True, True, True):
                bad_knob.append((ep, knobs))
            if 'lambda_min' in a:
                lam.append(float(a['lambda_min']))
            if 'dirs_per_freq' in a and int(a['dirs_per_freq']) != 1:
                bad_knob.append((ep, f"dirs_per_freq={a['dirs_per_freq']}"))
    print(f'  mesh_verdict ADEQUATE : {len(eps)-len(bad_mesh)}/{len(eps)}'
          + ('  ✅' if not bad_mesh else f'  ⛔ {bad_mesh[:5]}'))
    print(f'  세 손잡이 ON (ra/rf/ds) : {len(eps)-len(bad_knob)}/{len(eps)}'
          + ('  ✅' if not bad_knob else f'  ⛔ {bad_knob[:5]}'))
    if lam:
        print(f'  lambda_min            : min {min(lam):.3f} m  max {max(lam):.3f} m')
    return not bad_mesh and not bad_knob


def main():
    print('S1 실행 전 게이트 (PREREG §4)')
    ok = True
    res = {}
    for data_dir, eps, label in ((CROWDED_DIR, CROWDED16, '신규 혼잡16 (시드 20-23)'),
                                 (SPARSE_DIR, SPARSE16, '신규 희소16 (시드 20-23)')):
        r = report(data_dir, eps, label)
        if r is None:
            print(f'  ⛔ {label}: 파일 결측')
            ok = False
            continue
        axes, lo, floor = r
        res[label] = (lo, floor)
        if lo != len(eps):
            ok = False
        if not block_report(data_dir, eps, label):
            ok = False
        if not mesh_and_provenance(data_dir, eps, label):
            ok = False

    # AMENDMENT 1 F8 — H3 은 두 집합을 **가로질러** 비교하므로 32 pooled 게이트도 낸다.
    pooled = [(CROWDED_DIR, e) for e in CROWDED16] + [(SPARSE_DIR, e) for e in SPARSE16]
    if all(os.path.exists(os.path.join(REPO, d, e)) for d, e in pooled):
        import h5py as _h5
        pc, pf = [], []
        for d, e in pooled:
            with _h5.File(os.path.join(REPO, d, e), 'r') as f:
                pc.append(np.array(f['config/wave_components']))
                pf.append(np.array(f['frames/gt_wave_field']))
        print('\n### G-1c pooled 32 (H3 이 가로지르는 집합)')
        pax = {'진폭 A': [c[:, 0] for c in pc], 'omega': [c[:, 1] for c in pc],
               '위상 phi': [c[:, 4] for c in pc],
               '방향 theta': [np.arctan2(c[:, 3], c[:, 2]) for c in pc],
               '진실면': [f.ravel() for f in pf]}
        for name, vecs in pax.items():
            b = blocked_draws(vecs, [e for _, e in pooled])
            print(f'  {name:12s} ' + '  '.join(f'len{L}: {d}/{n}' for L, (d, n) in b.items())
                  + ('  ✅' if all(d == n for d, n in b.values()) else '  ⛔'))
            if not all(d == n for d, n in b.values()):
                ok = False

    # 대조: 원 G1P1 이 측정된 집합
    report('data_finemesh_2026-07-23',
           [f'episode_{bf}_{nc}comp_seed0_uniform.h5'
            for bf, nc in (('BF4', 15), ('BF4', 30), ('BF5', 15), ('BF5', 30))],
           '(대조) 원 CROWDED4 — G1P1 -0.0701 이 측정된 집합')

    print('\n' + '=' * 70)
    for k, (lo, floor) in res.items():
        print(f'  {k}: 가장 좁은 축 {lo}, 바닥 p = {floor:.2e}')
    print(f'GATE: {"PASS ✅ — S1 실행 허가" if ok else "FAIL ⛔ — S1 을 돌리지 않는다"}')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
