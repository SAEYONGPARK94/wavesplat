"""`AMENDMENT_1` **결과-무관 증명기** — 판정 표면(judgment surface)의 비트 대조.

    python3 tmp/M10_GEOMETRY_2026-08-30/amend1_surface.py --snapshot   # 개정 **전** 1회
    python3 tmp/M10_GEOMETRY_2026-08-30/amend1_surface.py              # 개정 **후** 대조

★무엇을 증명하는가 — `AMENDMENT_1` 이 **판정을 바꿀 수 없다**는 것을 산문이 아니라 기계로.
  「판정 표면」 = 자료를 라벨로 옮기는 모든 것:
  문턱 다섯 수(n=6…48) · 라벨 사상(w=0…48 × cmp∈{T,F}) · 접기 공역 · `READING` 축자 ·
  밴드 정의 · 셸 경계 · 코퍼스 48 · 시드 · 허용오차 상수 · `band_rmse`/`band_frac` 의 탐침값 ·
  ★**깨끗한 픽스처(스레드 고정 · `cuda_available=False`)에서의 게이트 전 출력**.
  이 전부가 **비트 동일**이면 개정은 어떤 자료에서도 같은 라벨을 낸다.

★그리고 **바뀐 것 하나**를 따로 증명한다(`--snapshot` 이후에만 의미가 있다):
  `cuda_available=True` 픽스처에서 `G-10` 의 판정이 FAIL → PASS 로 **딱 그 축에서만** 뒤집힌다.
  ⛔`device`·`omp_threads`·`code_md5` 축은 **그대로 FAIL** 이어야 한다(완화가 새는지 시험).

⛔이 파일은 판정하지 않는다. 결과 행을 읽지 않는다(`results_m10_*.jsonl` 미참조).
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for _p in (HERE, ROOT,
           os.path.join(ROOT, 'tmp', 'B11_HOS_BASELINE_2026-08-23'),
           os.path.join(ROOT, 'tmp', 'B12_HOS_EXTEND_2026-08-23'),
           os.path.join(ROOT, 'tmp', 'S16_BASELINE_2026-08-14')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gates_m10 as G                                              # noqa: E402

SNAP = os.path.join(HERE, 'amend1_surface.json')
PROBE_SH = {'obs:0_5': 2.0, 'obs:5_10': 4.0, 'obs:10_20': 7.0}
PROBE_FR = {'obs:0_5': 0.25, 'obs:5_10': 0.75, 'obs:10_20': 0.0}


def _fixture(**over):
    fx = G._fixture()
    if over:
        fx['rows'] = [dict(r, **over) for r in fx['rows']]
    return fx


def _gates_repr(**over):
    fx = _fixture(**over)
    return [[n, bool(o), str(d), bool(b)] for n, o, d, b in G.gates(fx['rows'], fx['frozen'])]


def surface():
    """★자료를 라벨로 옮기는 전부. 하나라도 움직이면 개정이 판정을 건드린 것이다."""
    # ⚠키는 **문자열**로 고정한다 — JSON 왕복에서 int 키가 str 로 바뀌면 `sort_keys` 순서가
    #   달라져 *"바뀌지 않았는데 달라 보이는"* 오탐이 난다(초판 실행에서 실제로 났다).
    thr = {str(n): G.thresholds(n) for n in range(6, G.N_EP + 1)}
    lab = {'%d|%s' % (w, c): G.label(w, G.N_EP, c)
           for w in range(G.N_EP + 1) for c in (True, False)}
    return dict(
        thresholds=thr,
        labels=lab,
        label_names=list(G.LABELS),
        fold={L: G.fold(L) for L in G.LABELS},
        reading={L: G.READING[L] for L in G.LABELS},
        bands={k: list(v) for k, v in G.BANDS.items()},
        primary_band=G.PRIMARY_BAND,
        shell_keys=list(G.SHELL_KEYS),
        edges=[('inf' if e == float('inf') else e) for e in G.EDGES],
        consts=dict(ALPHA=G.ALPHA, N_FAMILY=G.N_FAMILY, FAMILY_ALPHA=G.FAMILY_ALPHA,
                    REPRO_RTOL=G.REPRO_RTOL, FRAC_RTOL=G.FRAC_RTOL,
                    MIN_BAND_CELLS=G.MIN_BAND_CELLS, N_EP=G.N_EP,
                    CROWDED_MIN_NCOMP=G.CROWDED_MIN_NCOMP,
                    EST_SEEDS=list(G.EST_SEEDS), FILTER_SEEDS=list(G.FILTER_SEEDS),
                    ARM_ORDER=list(G.ARM_ORDER)),
        probe=dict(band_rmse={b: G.band_rmse(PROBE_SH, PROBE_FR, G.BANDS[b])
                              for b in G.BANDS},
                   band_frac={b: G.band_frac(PROBE_FR, G.BANDS[b]) for b in G.BANDS}),
        episodes=list(G.episodes()),
        # ★깨끗한 픽스처(=사전등록이 상정한 CPU·스레드 고정 상태)에서의 게이트 전 출력
        gates_clean=_gates_repr(),
    )


def _g10(rows_over):
    return [ok for n, ok, d, b in _gates_repr(**rows_over) if n.startswith('G-10')][0]


def main():
    s = surface()
    blob = json.dumps(s, sort_keys=True, ensure_ascii=False)
    dig = hashlib.md5(blob.encode()).hexdigest()
    if '--snapshot' in sys.argv:
        with open(SNAP, 'w') as fh:
            json.dump(dict(md5=dig, surface=s), fh, indent=1, ensure_ascii=False)
        print('판정 표면 스냅샷 기록 md5=%s -> %s' % (dig, os.path.basename(SNAP)))
        print('  (개정 전 상태. ⛔이 파일을 지우지 마라 — 결과-무관 증명의 기준선이다.)')
        return 0
    if not os.path.exists(SNAP):
        print('⛔스냅샷이 없다 — 먼저 `--snapshot` 을 개정 **전에** 돌렸어야 한다.')
        return 1
    ref = json.load(open(SNAP))
    fails = []
    for k in sorted(set(ref['surface']) | set(s)):
        if k == 'gates_clean':
            continue                        # ↓ 아래에서 게이트 단위로 더 엄하게 본다
        a = json.dumps(ref['surface'].get(k), sort_keys=True, ensure_ascii=False)
        b = json.dumps(s.get(k), sort_keys=True, ensure_ascii=False)
        same = (a == b)
        print('  %s  %s' % ('동일' if same else '⛔달라짐', k))
        if not same:
            fails.append(k)
            print('     전 %s' % a[:200])
            print('     후 %s' % b[:200])

    # ── gates_clean — ★게이트 **단위**로. 통째 비교보다 엄하다. ──────────────────
    #   허용되는 것은 딱 둘이고 **둘 다 개정문에 적힌 것**이다:
    ALLOWED_DELTA = {'G-10': 'AMENDMENT_1: cuda_available 값 고정 절 완화(이름·detail 포함)'}
    ALLOWED_NEW = {'G-15': 'AMENDMENT_1: 사전 장치 되읽기 (⛔반드시 범위축소)'}
    before = {e[0].split()[0]: e for e in ref['surface']['gates_clean']}
    after = {e[0].split()[0]: e for e in s['gates_clean']}
    blk_b = {k for k, e in before.items() if e[3]}
    blk_a = {k for k, e in after.items() if e[3]}
    ok = (blk_b == blk_a)
    print('  %s  gates_clean: **차단** 게이트 집합 불변 (%d개)'
          % ('동일' if ok else '⛔달라짐', len(blk_b)))
    if not ok:
        fails.append('차단 게이트 집합이 바뀌었다 %r' % ((blk_b ^ blk_a),))
    for gid in sorted(set(before) | set(after)):
        if gid in before and gid in after:
            same = before[gid] == after[gid]
            if same:
                continue
            if gid in ALLOWED_DELTA:
                print('  허용  gates_clean[%s] 변경 — %s' % (gid, ALLOWED_DELTA[gid]))
                if not after[gid][1] or not after[gid][3]:
                    fails.append('%s 가 깨끗한 픽스처에서 PASS·차단이 아니다' % gid)
                continue
            fails.append('gates_clean[%s] 가 허가 없이 바뀌었다' % gid)
            print('  ⛔달라짐  gates_clean[%s]\n     전 %r\n     후 %r'
                  % (gid, before[gid], after[gid]))
        elif gid in after:
            if gid in ALLOWED_NEW and after[gid][3] is False:
                print('  허용  gates_clean[%s] 신설 (범위축소) — %s'
                      % (gid, ALLOWED_NEW[gid]))
            else:
                fails.append('허가되지 않은 신설 게이트 %s (blocking=%r)'
                             % (gid, after[gid][3]))
        else:
            fails.append('게이트 %s 가 사라졌다' % gid)
    # ★핵심 한 줄 — `gates_clean` 을 뺀 **판정 표면 본체**는 비트 동일해야 한다.
    core_b = {k: v for k, v in ref['surface'].items() if k != 'gates_clean'}
    core_a = {k: v for k, v in s.items() if k != 'gates_clean'}
    hb = hashlib.md5(json.dumps(core_b, sort_keys=True,
                                ensure_ascii=False).encode()).hexdigest()
    ha = hashlib.md5(json.dumps(core_a, sort_keys=True,
                                ensure_ascii=False).encode()).hexdigest()
    print('  %s  ★판정 표면 본체(gates_clean 제외) md5  전 %s  후 %s'
          % ('동일' if hb == ha else '⛔달라짐', hb, ha))
    if hb != ha:
        fails.append('판정 표면 본체가 바뀌었다')
    print('  (참고) 전체 표면 md5  전 %s  후 %s — 차이는 위 gates_clean 규칙이 전부 설명한다'
          % (ref['md5'], dig))
    # ★의도된 유일한 변화 — `cuda_available` 축에서만 G-10 이 열린다.
    intended = [
        ('cuda_available=True (개정이 허용해야 한다)', dict(cuda_available=True), True),
        ('device="cuda" (⛔여전히 막아야 한다)', dict(device='cuda'), False),
        ('omp_threads="8" (⛔여전히 막아야 한다)', dict(omp_threads='8'), False),
    ]
    for name, over, want in intended:
        got = _g10(over)
        print('  %s  G-10 %s  %s' % ('OK' if got == want else '⛔FAIL',
                                     'PASS' if got else 'FAIL', name))
        if got != want:
            fails.append('의도된 변화: %s' % name)
    # ★혼합 스탬프는 완화 뒤에도 반드시 막힌다(= 반쪽 GPU 배치 탐지).
    fx = G._fixture()
    half = [dict(r, cuda_available=(i % 2 == 0)) for i, r in enumerate(fx['rows'])]
    mixed = [ok for n, ok, d, b in G.gates(half, fx['frozen']) if n.startswith('G-10')][0]
    print('  %s  G-10 %s  혼합 cuda_available (⛔반드시 FAIL — 단일 스탬프 요구)'
          % ('OK' if not mixed else '⛔FAIL', 'PASS' if mixed else 'FAIL'))
    if mixed:
        fails.append('혼합 스탬프가 통과한다')
    print('\n%s (%d 결함)' % ('결과-무관 증명 PASS' if not fails else '⛔FAIL', len(fails)))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
