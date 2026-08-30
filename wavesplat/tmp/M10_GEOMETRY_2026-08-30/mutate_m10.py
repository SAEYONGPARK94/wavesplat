"""M-10 **변형 하네스** — 게이트가 *"자료가 무엇이든 같은 답"* 을 내지 않는지 시험한다.
런 0회 · 자료 0행 · GPU 0회.

    python3 tmp/M10_GEOMETRY_2026-08-30/mutate_m10.py

★**러너를 모형한다, 게이트를 모형하지 않는다.** 합성 행은 `jobs_m10.expected_jobs()` 와
`jobs_m10.stamp()` — **러너가 실제로 쓰는 그 함수** — 로 찍는다. ⟹ 러너가 찍는 스탬프가
게이트의 필터를 항진시키는 부류(`M4 G-13`)를 결과 전에 잡는다.
그리고 `smoke_m10_*.jsonl` 이 있으면 **합성 행의 키 집합이 실제 행과 일치하는지** 대조한다
(⟹ *"픽스처만 갖고 있는 키로 게이트가 통과"* 를 막는다 — `M2 G-13` 이 물린 부류).

시험 항목
  A. 깨끗한 합성 288행에서 **차단 게이트 전건 PASS** + `G-CMP` PASS + 사전등록 라벨
  B. **커버리지 0 게이트 0개** (`G-8` 이 실제로 그것을 잡는지 포함)
  C. **삭제 민감도** — 팔·시드·에피소드를 하나씩 지우면 반드시 어떤 차단 게이트가 FAIL
  D. **항등 게이트 0** — 어떤 축의 행을 통째로 지웠는데 `(ok, detail)` 이 **비트 동일**한
     게이트가 있으면 FAIL(`check_gate_blindspots` 규약을 배치 안에서 미리 돌리는 것)
  E. **필드 민감도** — `shell_frac`·`zero_shell`·`avg_nRMSE`·스탬프·팔 되읽기를 흔들면
     **이름이 그것을 주장하는 게이트**가 FAIL
  F. **공허 참조 방어** — 동결 원천을 비우면 `G-4` 가 FAIL(0짝을 OK 로 찍지 않는다)
  G. **라벨 만족가능성** — 다섯 라벨이 전부 **도달 가능**하고 접기 공역이 규율을 지킨다
  H. **스키마 대조** — 합성 행 키 == 실제(스모크) 행 키
"""
import glob
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
import jobs_m10 as J                                               # noqa: E402

# ★AMENDMENT_1 (2026-08-30): 러너의 재현 통화가 「스레드 고정 · GPU 허용」으로 바뀌었다
#   ⟹ 합성 스탬프도 그 모양이어야 **러너를 모형한다**(⛔게이트를 모형하는 것이 아니다).
STAMP = dict(device='cpu', cuda_available=True, omp_threads='1', torch_threads=1,
             cuda_off_policy=False, code_md5='synthetic')


def synthetic_rows(ship_beats=None, spec_beats_zero=True):
    """★러너를 모형한 합성 행 288개. `(rows, frozen)`.

    `ship_beats` = SHIP 이 `BAND_REAL` 에서 이기는 에피소드 수(None = 전건).
    ⛔값은 전부 지어낸 것이고 **판정 경로에 들어가지 않는다** — 게이트 논리만 시험한다.
    """
    jobs = J.expected_jobs()
    eps = [e['episode'] for e in J.episodes()]
    nb = len(eps) if ship_beats is None else int(ship_beats)
    keys = G.SHELL_KEYS
    frac = {k: 1.0 / len(keys) for k in keys}
    rows, frozen = [], {}
    for arm, ep, seed in jobs:
        i = eps.index(ep)
        zero = {k: 0.50 + 0.001 * i + 0.01 * j for j, k in enumerate(keys)}
        if arm == 'SHIP':
            base = 0.10 + 0.001 * i
            sh = {k: base * (1.0 if i < nb else 1.60) for k in keys}
        else:
            base = 0.10 + 0.001 * i
            sh = {k: base * 1.30 if spec_beats_zero else base * 9.0 for k in keys}
        val = float(sh[keys[0]])
        frozen[(arm, ep, seed)] = val
        m = dict(avg_nRMSE=val, zero_nRMSE=0.25 + 0.001 * i,
                 shell_rmse=dict(sh), shell_frac=dict(frac),
                 zero_shell_batch=dict(zero),
                 zero_shell_ref=(dict(zero) if arm == 'SPEC25' else None),
                 n_scored=160, t_first=14.1, t_last=30.0,
                 bf='BF4', ncomp=15, gen_seed=1581 + i, mesh_verdict='DEGRADED',
                 Hs=1.2, Tp=7.0, depth=50.0, n_grid=4800, wall_s=1.0,
                 arm_cfg_nonempty=True, arm_cfg_nkeys=(24 if arm == 'SHIP' else None))
        if arm == 'SHIP':
            m.update(obj_init_method='auto', obj_tracking_mode='w_phi',
                     obj_lambda_l1=1e-4, obj_device='cpu',
                     cap_varpro_steps=400, cap_varpro_declared=400,
                     cap_varpro_fired=True, cap_init_iters=0,
                     cap_varpro_obj_before=1.0, cap_varpro_obj_after=0.5,
                     cap_ensemble_member_n=1,
                     prior_device='cuda',        # ★AMENDMENT_1 (G-15) — SHIP 행에만
                     auto_selected='learned')
        else:
            m.update(loc_radius=25.0, n_ensemble=600, background='jonswap',
                     filter_class='LETKFGridSpec')
        for b in G.BANDS:
            m['band_' + b] = G.band_rmse(sh, frac, G.BANDS[b])
            m['zero_' + b] = G.band_rmse(zero, frac, G.BANDS[b])
        m['band_cells'] = {b: int(round(G.band_frac(frac, G.BANDS[b]) * 4800))
                           for b in G.BANDS}
        m['band_frac'] = {b: G.band_frac(frac, G.BANDS[b]) for b in G.BANDS}
        rows.append(J.stamp((arm, ep, seed), m, **STAMP))
    return rows, frozen


# ── 도구 ──────────────────────────────────────────────────────────────────────
def _blocking_fails(gs):
    return [n.split()[0] for n, o, d, b in gs if b and not o]


def _sig(gs):
    """게이트 서명 — `(전체 이름, ok, detail)`. ⚠**이름까지** 본다:
    `gates_m10` 은 대조 수를 **이름 안에** 넣는다(`G-4 … (%d짝)`·`G-6 … (%d 대조)`).
    `(ok, detail)` 만 보면 그 게이트들이 **항등처럼 보이는 오탐**이 난다(초판이 그랬다)."""
    return {n.split()[0]: (n, o, d) for n, o, d, b in gs}


def _drop(rows, key, value):
    return [r for r in rows if r.get(key) != value]


def _mutate(rows, pred, **kw):
    return [dict(r, **kw) if pred(r) else r for r in rows]


def run():
    fails, notes = [], []

    def chk(name, cond, det=''):
        if not cond:
            fails.append('%s  %s' % (name, det))
        notes.append((name, bool(cond), det))

    rows, frozen = synthetic_rows()
    base = G.gates(rows, frozen)
    sig0 = _sig(base)

    # A ── 깨끗한 합성 행에서 차단 전건 PASS
    chk('A 차단 게이트 전건 PASS', not _blocking_fails(base), repr(_blocking_fails(base)))
    cmp_ok, cmp_d = G.cmp_ok_from(rows)
    chk('A2 G-CMP PASS', cmp_ok, repr(cmp_d))
    n = G.paired_wins(G.med_by_episode(rows, 'SHIP', 'band_' + G.PRIMARY_BAND),
                      G.med_by_episode(rows, 'SPEC25', 'band_' + G.PRIMARY_BAND))
    chk('A3 전건 승리 합성에서 라벨 = NEARFIELD-HOLDS',
        G.label(n[0], n[1], cmp_ok) == 'NEARFIELD-HOLDS',
        '%r -> %s' % (n[:2], G.label(n[0], n[1], cmp_ok)))

    # B ── 커버리지 0 게이트 0개 (G-8 이 그것을 잡는지도 함께)
    g8 = [(o, d) for nm, o, d, b in base if nm.startswith('G-8')]
    chk('B G-8 커버리지 0 게이트 0개', g8 and g8[0][0], repr(g8))

    # C ── 삭제 민감도
    for arm in J.ARMS:
        g = G.gates(_drop(rows, 'arm', arm), frozen)
        chk('C 팔 %s 삭제 ⟹ 차단 FAIL' % arm, bool(_blocking_fails(g)),
            repr(_blocking_fails(g)))
    for s in J.SEEDS['SHIP']:
        g = G.gates(_drop(rows, 'seed', s), frozen)
        chk('C 시드 %s 삭제 ⟹ 차단 FAIL' % s, bool(_blocking_fails(g)),
            repr(_blocking_fails(g)))
    ep0 = rows[0]['episode']
    g = G.gates(_drop(rows, 'episode', ep0), frozen)
    chk('C 에피소드 1개 삭제 ⟹ 차단 FAIL', bool(_blocking_fails(g)),
        repr(_blocking_fails(g)))
    chk('C 행 0개 ⟹ 차단 FAIL', bool(_blocking_fails(G.gates([], {}))))

    # D ── 항등 게이트 0 (`check_gate_blindspots` 규약을 배치 안에서 미리)
    ident = []
    for key in ('arm', 'seed', 'episode'):
        for v in sorted({r.get(key) for r in rows}, key=str):
            sig = _sig(G.gates(_drop(rows, key, v), frozen))
            for gid, pair in sig.items():
                if gid not in sig0:
                    continue
                toks = set(str(v).split()) | {str(v)}
                nm = [nm for nm, o, d, b in base if nm.startswith(gid)][0]
                named = bool(toks & set(nm.replace('/', ' ').split()))
                if named and pair == sig0[gid]:
                    ident.append((gid, key, v))
            if key != 'arm':
                break        # 시드·에피소드는 대표 1값이면 충분(비용)
    chk('D 이름이 축을 주장하는데 그 축의 행을 안 보는 게이트 0개', not ident, repr(ident[:5]))

    # E ── 필드 민감도 (이름이 주장하는 것을 실제로 본다)
    cases = [
        ('G-4', _mutate(rows, lambda r: True, avg_nRMSE=0.999), '동결에서 벗어난 값'),
        ('G-6', [dict(r, shell_frac={k: v * 2 for k, v in r['shell_frac'].items()})
                 if r['arm'] == 'SPEC25' else r for r in rows], '팔 간 마스크 불일치'),
        ('G-7', [dict(r, zero_shell_ref={k: v + 1e-9 for k, v in r['zero_shell_ref'].items()})
                 if r.get('zero_shell_ref') else r for r in rows], '교차구현 제로필드 불일치'),
        ('G-5', _mutate(rows, lambda r: r['arm'] == 'SPEC25', n_scored=159), '채점 창 불일치'),
        ('G-9', _mutate(rows, lambda r: r['arm'] == 'SPEC25', zero_nRMSE=0.3), 'GT 불일치'),
        ('G-3', _mutate(rows, lambda r: r['arm'] == 'SHIP', obj_init_method='v2_konly'),
         '배포 상수 이탈'),
        ('G-3', _mutate(rows, lambda r: r['arm'] == 'SHIP', cap_varpro_steps=None),
         '개입 계수기 부재'),
        ('G-3', _mutate(rows, lambda r: r['arm'] == 'SHIP', arm_cfg_nonempty=False),
         '⛔빈 dict 팔'),
        # ⛔★AMENDMENT_1: `cuda_available=True` **전 행**은 이제 의도된 통과다(BLOCKER-D1).
        #   ⟹ 같은 축을 **행마다 다르게** 흔들어 시험을 유지한다 — 완화는 값의 고정만 뺐고
        #   `len(stamps) == 1` 은 남았으므로 이쪽은 여전히 FAIL 이어야 한다.
        ('G-10', [dict(r, cuda_available=(i % 2 == 0)) for i, r in enumerate(rows)],
         '배치 중간 GPU 가시성 변화'),
        ('G-10', _mutate(rows, lambda r: True, device='cuda'), '선언 장치 이탈'),
        ('G-10', _mutate(rows, lambda r: True, omp_threads='8'), '스레드 미고정'),
        ('G-10', _mutate(rows, lambda r: r['arm'] == 'SHIP', code_md5='other'),
         'code_md5 분열'),
        # ★AMENDMENT_1 신설 (범위축소 게이트도 항진하면 안 된다)
        ('G-15', [{k: v for k, v in r.items() if k != 'prior_device'} for r in rows],
         '사전 장치 되읽기 부재'),
        ('G-15', [dict(r, prior_device=('cuda' if i % 2 else 'cpu'))
                  if r['arm'] == 'SHIP' else r for i, r in enumerate(rows)],
         '사전 장치 분열'),
        ('G-2', _mutate(rows, lambda r: r is rows[0], error='boom'), '에러 행'),
    ]
    for gid, mut, why in cases:
        g = G.gates(mut, frozen)
        # ⚠이름 기준이다 — `G-15` 는 **범위축소**라 `_blocking_fails` 에 안 잡힌다.
        hit = [nm.split()[0] for nm, o, d, b in g if not o and nm.startswith(gid)]
        chk('E %s FAIL (%s)' % (gid, why), bool(hit),
            '실제 FAIL(차단) = %r · 전체 FAIL = %r'
            % (_blocking_fails(g), [nm.split()[0] for nm, o, d, b in g if not o]))

    # ★D2 — **항진 탐지**: 모든 게이트가 삭제 또는 내용 변형 **어느 하나에는** 반응해야 한다.
    #   ⚠삭제만으로는 부족하다 — `G-2`(에러 수)·`G-5`·`G-9`·`G-10` 은 행을 지워도 안 움직이고
    #     **내용**에만 반응한다. 그 둘을 합친 것이 정직한 판별이다.
    #   ⛔`G-11` 만 면제 — 그것은 **소스 텍스트** 게이트라 자료에 구조적으로 무관하다(설계).
    D2_EXEMPT = {'G-11'}
    perturbed = [_drop(rows, 'arm', 'SHIP'), _drop(rows, 'arm', 'SPEC25'),
                 _drop(rows, 'seed', J.SEEDS['SHIP'][0]), _drop(rows, 'episode', ep0),
                 _mutate(rows, lambda r: True, mesh_verdict='ADEQUATE'),
                 _mutate(rows, lambda r: True, band_cells={b: 1 for b in G.BANDS}),
                 _mutate(rows, lambda r: r['arm'] == 'SHIP', avg_nRMSE=0.777)] \
        + [m for _g, m, _w in cases]
    sigs = [_sig(G.gates(p, frozen)) for p in perturbed]
    inert = [gid for gid in sig0
             if gid not in D2_EXEMPT and all(s.get(gid) == sig0[gid] for s in sigs)]
    chk('D2 삭제·변형 어느 것에도 서명이 안 움직이는 게이트 0개 (G-11 면제=소스 게이트)',
        not inert, repr(inert))

    # ★E2 — 범위축소 게이트가 **보고를 실제로 바꾸는가**(항상 True 를 찍는 게이트의 무용 시험)
    small = _mutate(rows, lambda r: True,
                    band_cells={b: 1 for b in G.BANDS})
    d13 = [d for nm, o, d, b in G.gates(small, frozen) if nm.startswith('G-13')]
    d13b = [d for nm, o, d, b in base if nm.startswith('G-13')]
    chk('E2 G-13 격자점 하한 미달이 보고에 나타난다', d13 != d13b, '%r vs %r' % (d13, d13b))

    # F ── 공허 참조 방어
    g = G.gates(rows, {})
    chk('F 동결 원천이 비면 G-4 FAIL',
        any((not o) for nm, o, d, b in g if nm.startswith('G-4')),
        repr(_blocking_fails(g)))

    # G ── 라벨 만족가능성 + 접기 규율
    reach = {}
    for w in range(G.N_EP + 1):
        reach.setdefault(G.label(w, G.N_EP, True), []).append(w)
    reach.setdefault(G.label(0, G.N_EP, False), []).append('cmp=False')
    chk('G 다섯 라벨 전부 도달 가능', set(reach) == set(G.LABELS), repr(sorted(reach)))
    chk('G2 break 로 접히는 라벨은 정확히 하나',
        sum(1 for L in G.LABELS if G.fold(L) == 'break') == 1)
    chk('G3 VOID 는 어떤 승수에서도 VOID',
        {G.label(w, G.N_EP, False) for w in (0, 17, 31, 32, 48)} == {'NEARFIELD-VOID'})
    # ★BREAKS 가 실제 자료 모양에서 도달 가능한가 (라벨이 장식이 아니라는 증거)
    rows_b, frozen_b = synthetic_rows(ship_beats=0)
    ok_b, _ = G.cmp_ok_from(rows_b)
    nb = G.paired_wins(G.med_by_episode(rows_b, 'SHIP', 'band_' + G.PRIMARY_BAND),
                       G.med_by_episode(rows_b, 'SPEC25', 'band_' + G.PRIMARY_BAND))
    chk('G4 SHIP 전패 합성 ⟹ NEARFIELD-BREAKS',
        G.label(nb[0], nb[1], ok_b) == 'NEARFIELD-BREAKS',
        '%r -> %s' % (nb[:2], G.label(nb[0], nb[1], ok_b)))
    rows_v, frozen_v = synthetic_rows(spec_beats_zero=False)
    ok_v, dv = G.cmp_ok_from(rows_v)
    nv = G.paired_wins(G.med_by_episode(rows_v, 'SHIP', 'band_' + G.PRIMARY_BAND),
                       G.med_by_episode(rows_v, 'SPEC25', 'band_' + G.PRIMARY_BAND))
    chk('G5 비교자 무기술 합성 ⟹ NEARFIELD-VOID (승수 %d/%d 여도)' % nv[:2],
        (not ok_v) and G.label(nv[0], nv[1], ok_v) == 'NEARFIELD-VOID', repr(dv))

    # H ── 스키마 대조 (실제 행이 있으면)
    real = []
    for p in sorted(glob.glob(os.path.join(HERE, 'smoke_m10_*.jsonl'))
                    + glob.glob(os.path.join(HERE, 'results_m10_*.jsonl'))):
        for line in open(p):
            line = line.strip()
            if line:
                try:
                    real.append(json.loads(line))
                except ValueError:
                    pass
    if real:
        for arm in J.ARMS:
            rr = [r for r in real if r.get('arm') == arm and 'error' not in r]
            ss = [r for r in rows if r['arm'] == arm]
            if not rr:
                notes.append(('H %s 실제 행 0 — 대조 생략' % arm, True, ''))
                continue
            miss = sorted(set(rr[0]) - set(ss[0]))
            extra = sorted(set(ss[0]) - set(rr[0]))
            chk('H %s 합성 행 키 == 실제 행 키' % arm, not miss and not extra,
                '실제에만 %r · 합성에만 %r' % (miss, extra))
    else:
        notes.append(('H 실제 행 0 — 스키마 대조는 스모크 뒤에 돈다', True, ''))

    print('── mutate_m10 변형 하네스 (런 0회 · 자료 0행) ──')
    for name, ok, det in notes:
        print('  %s  %s%s' % ('PASS' if ok else '⛔FAIL', name,
                              ('   %s' % det) if (det and not ok) else ''))
    print('  %d/%d PASS' % (len(notes) - len(fails), len(notes)))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(run())
