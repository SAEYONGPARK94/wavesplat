"""S7 실행 전 게이트 (PREREG §7, AMENDMENT_1).

⛔하나라도 FAIL 이면 `run_s8.py` 를 **돌리지 않는다.** 결과를 본 뒤 완화하지 않는다.

G-1 독립성 (5축 32종 + 길이블록 내 16/16)   ← S1 `gate.py` 를 **import 해서 규약 그대로**
G-2 메시 ADEQUATE 64/64
G-3 손잡이 3종 ON 64/64 (+ dirs_per_freq==1)
G-4 충돌 0 (기존 데이터와 파일명 교집합 0, 시드 32-39)
G-5 코드 출처 (미커밋 9파일 518줄 — A14): diff md5 를 **기록**하고 런 전후 동일함을 요구

이 스크립트는 S8 결과 파일을 읽지 않는다(존재하지 않는다).
"""
import glob
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
S1DIR = os.path.abspath(os.path.join(HERE, '..', 'S1_indep_sparse_2026-08-08'))
K6 = os.path.abspath(os.path.join(HERE, '..', 'K6_run_2026-08-07'))
for p in (K6, S1DIR):
    sys.path.insert(0, p)

import gate as S1GATE                      # noqa: E402  S1 게이트 (복사 금지, import)
from independence_gate import report       # noqa: E402  frozen 독립성 규약

CROWDED_DIR = 'data_s8_crowded_2026-08-10'
SPARSE_DIR = 'data_s8_sparse_2026-08-10'
SEEDS = range(32, 40)
CROWDED32 = [f'episode_{bf}_{nc}comp_seed{s}_uniform_ra_rf_ds.h5'
             for bf in ('BF4', 'BF5') for nc in (15, 30) for s in SEEDS]
SPARSE32 = [f'episode_{bf}_{nc}comp_seed{s}_uniform_ra_rf_ds.h5'
            for bf in ('BF4', 'BF5') for nc in (2, 5) for s in SEEDS]

CODE_FILES = ['wavesplat/run.py', 'wavesplat/evaluation/metrics.py',
              'wavesplat/optimizer/initializer.py', 'wavesplat/optimizer/sequential_update.py',
              'wavesim/wave_generator.py', 'wavesim/pipeline.py', 'wavesim/config.py',
              'wavesim/data_writer.py', 'generate_dataset.py', 'configs/estimation.yaml']


def g4_collisions():
    """기존 어느 데이터 집합과도 파일명이 겹치지 않는지."""
    print('\n### G-4 충돌 0 (기존 집합과 파일명 교집합)')
    new = set(CROWDED32) | set(SPARSE32)
    others = [d for d in glob.glob(os.path.join(REPO, 'data*'))
              if os.path.isdir(d) and os.path.basename(d) not in (CROWDED_DIR, SPARSE_DIR)]
    bad = []
    for d in others:
        inter = new & {os.path.basename(p) for p in glob.glob(os.path.join(d, '*.h5'))}
        if inter:
            bad.append((os.path.basename(d), sorted(inter)[:3]))
    # ⚠AMENDMENT 2: 이 검사는 S7 에서 상속될 때 시드 구간이 `24 <= s <= 31` 로 **굳어 있었다**.
    # 상수를 다시 적지 않고 **이 배치의 `SEEDS` 에서 유도**한다 — 같은 방식으로 다시 낡을 수 없다.
    got = {int(e.split('seed')[1].split('_')[0]) for e in new}
    want = set(SEEDS)
    seeds_ok = (got == want)
    print(f'  기존 {len(others)} 집합과 교집합: {"0  ✅" if not bad else f"⛔ {bad}"}')
    print(f'  시드 전부 {min(want)}-{max(want)} : {"✅" if seeds_ok else f"⛔ 실측 {sorted(got)}"}')
    return not bad and seeds_ok


def g5_code_provenance():
    """A14 — 미커밋 코드 위에서 돈다는 사실을 **기록**한다(숨기지 않는다)."""
    print('\n### G-5 코드 출처 (A14: 미커밋 변경 위에서 실행 중)')
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO,
                          capture_output=True, text=True).stdout.strip()
    stat = subprocess.run(['git', 'diff', '--stat'] + CODE_FILES, cwd=REPO,
                          capture_output=True, text=True).stdout.strip()
    diff = subprocess.run(['git', 'diff'] + CODE_FILES, cwd=REPO,
                          capture_output=True, text=True).stdout
    md5 = hashlib.md5(diff.encode()).hexdigest()
    print(f'  HEAD = {head[:12]}')
    for line in stat.splitlines():
        print(f'  {line}')
    print(f'  ★ 미커밋 diff md5 = {md5}')
    print('  ⚠이 배치의 데이터와 추정기는 위 미커밋 변경 위에서 돈다. 런 종료 후 같은 md5 를'
          ' 재확인하고, HEAD 대비 대표 셀 bit-identity 를 별도 측정한다(§5).')
    json.dump({'head': head, 'diff_md5': md5, 'stat': stat},
              open(os.path.join(HERE, 'G5_code_provenance.json'), 'w'), indent=1)
    return True                      # 기록 의무이지 차단 조건이 아니다


def main():
    print('S7 실행 전 게이트 (PREREG §3.1 / AMENDMENT_1 A5)')
    missing = [os.path.join(d, e) for d, eps in
               ((CROWDED_DIR, CROWDED32), (SPARSE_DIR, SPARSE32)) for e in eps
               if not os.path.exists(os.path.join(REPO, d, e))]
    if missing:
        print(f'⛔ 에피소드 결측 {len(missing)}개 — 생성이 끝나지 않았다. 예: {missing[:2]}')
        sys.exit(2)

    ok = True
    summary = {}
    for data_dir, eps, label in ((CROWDED_DIR, CROWDED32, '신규 혼잡32 (시드 32-39)'),
                                 (SPARSE_DIR, SPARSE32, '신규 희소32 (시드 32-39)')):
        r = report(data_dir, eps, label)
        if r is None:
            print(f'  ⛔ {label}: 파일 결측')
            ok = False
            continue
        _axes, lo, floor = r
        summary[label] = (lo, floor)
        if lo != len(eps):
            ok = False
        if not S1GATE.block_report(data_dir, eps, label):
            ok = False
        if not S1GATE.mesh_and_provenance(data_dir, eps, label):
            ok = False

    # pooled 64 — 레짐을 가로지르는 대조(희소 게이트)가 이 집합 위에서 이뤄진다
    import h5py
    pooled = [(CROWDED_DIR, e) for e in CROWDED32] + [(SPARSE_DIR, e) for e in SPARSE32]
    pc, pf = [], []
    for d, e in pooled:
        with h5py.File(os.path.join(REPO, d, e), 'r') as f:
            pc.append(np.array(f['config/wave_components']))
            pf.append(np.array(f['frames/gt_wave_field']))
    print('\n### G-1c pooled 64 (레짐 대조가 가로지르는 집합)')
    pax = {'진폭 A': [c[:, 0] for c in pc], 'omega': [c[:, 1] for c in pc],
           '위상 phi': [c[:, 4] for c in pc],
           '방향 theta': [np.arctan2(c[:, 3], c[:, 2]) for c in pc],
           '진실면': [f.ravel() for f in pf]}
    for name, vecs in pax.items():
        b = S1GATE.blocked_draws(vecs, [e for _, e in pooled])
        good = all(d == n for d, n in b.values())
        print(f'  {name:12s} ' + '  '.join(f'len{L}: {d}/{n}' for L, (d, n) in b.items())
              + ('  ✅' if good else '  ⛔'))
        ok &= good

    ok &= g4_collisions()
    g5_code_provenance()

    # G-6 신규 게이트 결백 (PREREG §7) — `bitcheck_s8.py` 가 측정한 결과를 **재수록**한다.
    print('\n### G-6 신규 게이트 결백 (S8_L1_PHASE1_ONLY / S8_L1_TRACK_ONLY)')
    bc = os.path.join(HERE, 'bitcheck.txt')
    if os.path.exists(bc):
        body = open(bc).read().strip().splitlines()
        for line in body:
            print('  ' + line)
        ok &= ('G-6: PASS' in open(bc).read())
    else:
        print('  ⛔ bitcheck.txt 없음 — `python3 bitcheck_s8.py` 를 먼저 돌려라')
        ok = False

    print('\n' + '=' * 70)
    for k, (lo, floor) in summary.items():
        print(f'  {k}: 가장 좁은 축 {lo}종, 바닥 p = {floor:.2e}')
    print('  ⚠보증 = **길이 블록당 16종**(블록 간 독립성은 길이가 달라 측정 불가) — S1 F8 승계')
    print('  ⚠조건 단위 자유도 = 4 (BF×성분수), 조건 라벨 바닥 p = 0.125 — A8')
    print(f'GATE: {"PASS ✅ — S8 실행 허가" if ok else "FAIL ⛔ — S8 을 돌리지 않는다"}')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
