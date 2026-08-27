"""B-21b — **등가한계 Δ 민감도**를 동결 자료에서 재현한다 (2026-08-27, 런 0회).

★**이 스크립트는 `B-21` 의 동결 판정을 바꾸지 않는다.**
`verdict_b21.json` 을 읽지도 쓰지도 않고(D-2 만 대조용으로 읽는다), `PREREG.md`·
`gates_b21.py`·결과 jsonl 을 **한 바이트도 건드리지 않는다.** 만드는 것은 **공시**뿐이다.

무엇을 재현하는가 (전부 `AMENDMENT_DELTA_2026-08-27.md` 의 표와 대조된다):
  D-0  동결 입력의 md5 가 `FROZEN_md5.txt` 와 일치한다
  D-1  `B-21` 결과 jsonl 6개의 md5 가 이 스크립트가 본 것과 같다
  D-2  재집계가 `verdict_b21.json` 의 PRIMARY 를 그대로 낸다 (판정 무손상)
  D-3  희소 짝 상대차의 순서통계량 `d(5)`·`d(12)` = 소수의견 CI `[-13.4%, +11.7%]`
  D-4  같은 두 값이 **교차배치 동결 원본**(B-19 × B-11b)에서도 나온다 (B-21 재실행과 무관)
  D-5  부호-TOST 의 Δ 경계 = `max(d(k), -d(n-k+1))` — 층별 실측
  D-6  Δ 에 대해 TOST 는 단조 ⟹ 층마다 뒤집힘이 **정확히 한 번**
  D-7  라벨 스윕: Δ=10.0(채택) 양 층 UNDECIDED · Δ=10.8 혼잡 뒤집힘 · Δ=13.4 양 층 뒤집힘
  D-8  ★`PREREG.md:302-303` 의 산술 주장(*"13.4% 를 고르면 아슬아슬하게 **실패**"*)이 거짓
  D-9  교락 TOST(Δ_conf=5%)는 Δ 민감하지 **않다** (경계가 5% 의 1/9 이하)
  D-10 도출 규칙의 제수(1/4)에 대한 민감도: 1/4 → 양 층 UNDECIDED, 1/3 → 양 층 CONFIRMED
  D-11 문서(`AMENDMENT_DELTA_2026-08-27.md`)에 위 수치가 **축자로** 실려 있다

⛔새 통계량을 만들지 않는다 — 집계·부호검정·TOST·라벨은 전부 `gates_b21` 를 **호출만** 한다.

    python3 tmp/B21_MEDIATION_2026-08-27/delta_sensitivity.py
"""
import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for _p in (HERE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import gates_b21 as G                                        # noqa: E402

DOC = os.path.join(HERE, 'AMENDMENT_DELTA_2026-08-27.md')

# ── ★문서의 표를 여기에 **핀** 한다. 어긋나면 이 스크립트가 실패한다 ───────────
#    (저장소 규율: 인용 가능한 수치를 낸 스크립트가 없으면 안 된다.)
EXPECTED = {
    'sparse': dict(
        n=16, k=12, better=8, worse=8,
        d_lo=-13.369027553844626,      # d(5)  = 소수의견 CI 하단
        d_hi=11.691690061852167,       # d(12) = 소수의견 CI 상단
        boundary=13.369027553844626,   # PRIMARY 뒤집힘 경계 (열린 구간: Δ > 경계)
        conf_boundary=0.4698022315945287,
        label_at_10='MEDIATION-UNDECIDED',
        label_at_10_8='MEDIATION-UNDECIDED',
        label_at_13_4='MEDIATION-CONFIRMED'),
    'crowded': dict(
        n=48, k=31, better=21, worse=27,
        d_lo=-3.7265537047507102,      # d(18)
        d_hi=10.739270453766956,       # d(31)
        boundary=10.739270453766956,
        conf_boundary=0.512074127515037,
        label_at_10='MEDIATION-UNDECIDED',
        label_at_10_8='MEDIATION-CONFIRMED',
        label_at_13_4='MEDIATION-CONFIRMED'),
}
DELTA_ADOPTED = 10.0                   # 사전등록 §8 이 채택한 값
DELTA_REJECTED = 13.4                  # `PREREG.md:302` 가 거부한 값
DELTA_CONF = 5.0                       # 사전등록 §8 의 Δ_conf
FROZEN_MARGIN_SMALLER = 40.19          # `B-12` 혼잡 마진 (= Δ 도출의 분자)
DIVISOR_FLIP = {'sparse': 3.0062021966917314, 'crowded': 3.7423398705731232}
RTOL = 1e-9

# `B-21` 결과 jsonl — 이 스크립트가 읽은 바이트를 못박는다 (동결 목록에는 없다: 결과다)
RESULT_MD5 = {
    'results_b21_0.jsonl': 'f4659f54a316b67d1b4ef936ff438d2b',
    'results_b21_1.jsonl': '6490b27ca2dc2027a77b38bfc7f6d1da',
    'results_b21_2.jsonl': 'edd42a0a9f479e1b624b19815aa8f252',
    'results_b21_3.jsonl': '09c8fdf4aa1eeb262b02a54d2648262b',
    'results_b21_4.jsonl': 'fcd3aad4fa96b5b334a3ab53e6f79652',
    'results_b21_5.jsonl': '6d8caa16edb637a3aa5449c10a2f65ad',
}

# ★`PREREG.md` 의 두 줄을 **줄번호로** 고정한다 (문서가 움직이면 D-8 이 죽는다)
PREREG_ANCHORS = [
    (37, '[−13.4%, +11.7%]'),      # 소수의견 CI 가 사전등록 본문에 실려 있다
    (302, '13.4% 를 고르면'),
    (303, '아슬아슬하게 실패'),
]

# ★문서에 축자로 있어야 하는 문자열 (손복사 표류 방지)
DOC_MUST_CONTAIN = [
    '13.369028', '10.739270', '11.691690', '3.726554',
    '0.469802', '0.512074', '10.0475', '3.0062', '3.7423',
    'MEDIATION-UNDECIDED', 'MEDIATION-CONFIRMED',
]


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_rows():
    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, 'results_b21*.jsonl'))):
        if os.path.basename(f).startswith('SMOKE'):
            continue
        with open(f) as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def close(a, b, rtol=RTOL):
    if a is None or b is None:
        return False
    return abs(a - b) <= rtol * max(1.0, abs(b))


def boundary_of(diffs, k):
    """부호-TOST 가 통과하는 최소 Δ (열린 경계): `max(d(k), -d(n-k+1))`."""
    s = sorted(diffs)
    n = len(s)
    return max(s[k - 1], -s[n - k])


def cross_batch_sparse_diffs():
    """★`B-21` 재실행을 **쓰지 않고** 교차배치 동결 원본에서 같은 짝 상대차를 만든다.

    `gates_b21.posctrl_ref` 가 권위다 (B-19 `KGRID1` × B-11b `SPEC25`).
    """
    import statistics as st
    k1 = G.posctrl_ref('sparse', 'KGRID1')
    sp = G.posctrl_ref('sparse', 'SPEC25')

    def by_ep(d):
        out = {}
        for (ep, _seed), v in d.items():
            out.setdefault(ep, []).append(v)
        return {ep: st.median(v) for ep, v in out.items()}

    a, b = by_ep(k1), by_ep(sp)
    eps = sorted(set(a) & set(b))
    return sorted((a[e] - b[e]) / b[e] * 100.0 for e in eps)


def main():
    checks = []

    def chk(name, ok, detail=''):
        checks.append((name, bool(ok), detail))

    # ── D-0 동결 입력 md5 ────────────────────────────────────────────────
    frozen = {}
    with open(os.path.join(HERE, 'FROZEN_md5.txt')) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            h, _, rel = line.partition('  ')
            frozen[rel.strip()] = h.strip()
    need = [r for r in frozen
            if r.endswith(('gates_b21.py', 'PREREG.md'))
            or 'B19_KGRID_SPARSE' in r or 'B11_HOS_BASELINE' in r
            or 'B16_KGRID' in r or 'B12_HOS_EXTEND' in r]
    bad = [r for r in sorted(need) if md5(os.path.join(ROOT, r)) != frozen[r]]
    chk('D-0 동결 입력 md5 일치', not bad and len(need) >= 10,
        f'{len(need) - len(bad)}/{len(need)} 일치' + (f' · 이탈 {bad}' if bad else ''))

    # ── D-1 결과 jsonl md5 ──────────────────────────────────────────────
    bad = [n for n, h in sorted(RESULT_MD5.items())
           if md5(os.path.join(HERE, n)) != h]
    chk('D-1 결과 jsonl md5 핀', not bad,
        f'{len(RESULT_MD5) - len(bad)}/{len(RESULT_MD5)} 일치'
        + (f' · 이탈 {bad}' if bad else ''))

    # ── 재집계 (⛔새 구현 금지 — `gates_b21` 호출) ──────────────────────
    rows = load_rows()
    agg = G.aggregate(rows)
    val = agg['val']
    prim, conf, neg = {}, {}, {}
    for st_name in G.STRATA:
        good = agg['good'][st_name]
        prim[st_name] = G.cmp_pair(val, good, st_name, 'KGRID1', G.BASELINE_ARM)
        conf[st_name] = G.cmp_pair(val, good, st_name, 'FITSWAP', 'SHIP')
        neg[st_name] = G.negative_ok(agg, st_name)

    # ── D-2 동결 판정 무손상 ────────────────────────────────────────────
    vp = os.path.join(HERE, 'verdict_b21.json')
    with open(vp) as fh:
        vd = json.load(fh)
    same = []
    for st_name in G.STRATA:
        v = vd['strata'][st_name]['primary']
        p = prim[st_name]
        same.append(v['n'] == p['n'] and v['better'] == p['better']
                    and v['worse'] == p['worse']
                    and close(p['rel_median_pct'], v['rel_median_pct']))
        t = G.sign_tost(p['diffs_pct'], G.TOST_DELTA_PCT)
        lab = G.label_primary(st_name, p, t, neg[st_name])[0]
        same.append(lab == vd['strata'][st_name]['primary_label']
                    == EXPECTED[st_name]['label_at_10'])
    chk('D-2 재집계가 동결 PRIMARY·라벨을 그대로 낸다 (판정 무손상)', all(same),
        ' · '.join(f'{s}:{vd["strata"][s]["primary_label"]}' for s in G.STRATA))

    # ── D-3 희소 순서통계량 = 소수의견 CI ───────────────────────────────
    e = EXPECTED['sparse']
    s = sorted(prim['sparse']['diffs_pct'])
    n, k = len(s), G.k_min(len(s), G.ALPHA)
    ok = (n == e['n'] and k == e['k']
          and close(s[n - k], e['d_lo']) and close(s[k - 1], e['d_hi'])
          and f'{s[n - k]:.1f}' == '-13.4' and f'{s[k - 1]:.1f}' == '11.7')
    chk('D-3 희소 d(5)·d(12) = 소수의견 CI [-13.4%, +11.7%]', ok,
        f'd({n - k + 1})={s[n - k]:.12f} · d({k})={s[k - 1]:.12f}')

    # ── D-4 교차배치 원본에서도 같은 두 값 ──────────────────────────────
    xs = cross_batch_sparse_diffs()
    ok = (len(xs) == e['n'] and abs(xs[e['n'] - e['k']] - e['d_lo']) < 1e-9
          and abs(xs[e['k'] - 1] - e['d_hi']) < 1e-9)
    chk('D-4 교차배치 동결 원본(B-19 × B-11b)이 같은 CI 를 낸다', ok,
        f'n={len(xs)} · d(5)={xs[4]:.12f} · d(12)={xs[11]:.12f} '
        f'(⟹ 그 CI 는 **이 자료의 순서통계량**이다)')

    # ── D-5 Δ 경계 ──────────────────────────────────────────────────────
    det = []
    ok = True
    for st_name in G.STRATA:
        p = prim[st_name]
        kk = G.k_min(p['n'], G.ALPHA)
        b = boundary_of(p['diffs_pct'], kk)
        ok &= close(b, EXPECTED[st_name]['boundary'])
        # 경계의 양쪽에서 실제로 갈리는가 (열린 경계)
        ok &= (not G.sign_tost(p['diffs_pct'], b)['passed'])
        ok &= G.sign_tost(p['diffs_pct'], b + 1e-9)['passed']
        det.append(f'{st_name} Δ*={b:.12f}')
    chk('D-5 Δ 경계 = max(d(k), -d(n-k+1)) · 열린 경계 확인', ok, ' · '.join(det))

    # ── D-6 단조 ⟹ 뒤집힘이 정확히 한 번 ────────────────────────────────
    ok, det = True, []
    for st_name in G.STRATA:
        p = prim[st_name]
        grid = [i * 0.005 for i in range(1, 20001)]      # 0.005 … 100.0
        seq = [G.sign_tost(p['diffs_pct'], d)['passed'] for d in grid]
        flips = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
        ok &= (flips == 1 and seq[0] is False and seq[-1] is True)
        det.append(f'{st_name} 뒤집힘 {flips}회')
    chk('D-6 TOST 는 Δ 에 단조 ⟹ 층마다 뒤집힘 정확히 1회', ok, ' · '.join(det))

    # ── D-7 라벨 스윕 ───────────────────────────────────────────────────
    ok, det = True, []
    for st_name in G.STRATA:
        p = prim[st_name]
        for d, key in ((DELTA_ADOPTED, 'label_at_10'), (10.8, 'label_at_10_8'),
                       (DELTA_REJECTED, 'label_at_13_4')):
            lab = G.label_primary(st_name, p, G.sign_tost(p['diffs_pct'], d),
                                  neg[st_name])[0]
            ok &= (lab == EXPECTED[st_name][key])
            det.append(f'{st_name}@{d}={lab.split("-")[-1]}')
    chk('D-7 라벨 스윕 (Δ=10.0 / 10.8 / 13.4)', ok, ' · '.join(det))

    # ── D-8 ★`PREREG.md:302-303` 의 산술 주장이 거짓이다 ────────────────
    lines = open(os.path.join(HERE, 'PREREG.md')).read().splitlines()
    anch = all(1 <= ln <= len(lines) and needle in lines[ln - 1]
               for ln, needle in PREREG_ANCHORS)
    passed_at_13_4 = G.sign_tost(prim['sparse']['diffs_pct'], DELTA_REJECTED)['passed']
    b = EXPECTED['sparse']['boundary']
    chk('D-8 ★PREREG:302-303 의 *"13.4% ⟹ 아슬아슬하게 실패"* 가 **거짓**',
        anch and passed_at_13_4 and DELTA_REJECTED > b,
        f'줄 고정 {anch} · 경계 {b:.6f} < 13.4 ⟹ 희소 TOST **통과** '
        f'({passed_at_13_4}) = 아슬아슬한 **성공**이다')

    # ── D-9 교락 TOST 는 Δ 민감하지 않다 ────────────────────────────────
    ok, det = True, []
    for st_name in G.STRATA:
        c = conf[st_name]
        kk = G.k_min(c['n'], G.ALPHA)
        b = boundary_of(c['diffs_pct'], kk)
        ok &= close(b, EXPECTED[st_name]['conf_boundary'])
        ok &= G.sign_tost(c['diffs_pct'], DELTA_CONF)['passed']
        ok &= (DELTA_CONF / b > 9.0)
        det.append(f'{st_name} Δ*_conf={b:.6f} (여유 {DELTA_CONF / b:.2f}배)')
    chk('D-9 교락 TOST(Δ_conf=5%)는 Δ 민감하지 않다 (여유 9배 이상)', ok,
        ' · '.join(det))

    # ── D-10 도출 규칙 제수 민감도 ──────────────────────────────────────
    ok, det = True, []
    for st_name in G.STRATA:
        p = prim[st_name]
        b = EXPECTED[st_name]['boundary']
        ok &= close(FROZEN_MARGIN_SMALLER / b, DIVISOR_FLIP[st_name], 1e-6)
        lab4 = G.label_primary(st_name, p,
                               G.sign_tost(p['diffs_pct'], FROZEN_MARGIN_SMALLER / 4),
                               neg[st_name])[0]
        lab3 = G.label_primary(st_name, p,
                               G.sign_tost(p['diffs_pct'], FROZEN_MARGIN_SMALLER / 3),
                               neg[st_name])[0]
        ok &= (lab4 == 'MEDIATION-UNDECIDED' and lab3 == 'MEDIATION-CONFIRMED')
        det.append(f'{st_name} 뒤집힘 제수={DIVISOR_FLIP[st_name]:.4f} '
                   f'(1/4→{lab4.split("-")[-1]}, 1/3→{lab3.split("-")[-1]})')
    chk('D-10 제수 민감도: 1/4 → UNDECIDED · 1/3 → CONFIRMED (양 층)', ok,
        ' · '.join(det))

    # ── D-11 문서 축자 대조 ─────────────────────────────────────────────
    if os.path.exists(DOC):
        text = open(DOC).read()
        miss = [t for t in DOC_MUST_CONTAIN if t not in text]
        chk('D-11 문서에 표의 수치가 축자로 실려 있다', not miss,
            f'{len(DOC_MUST_CONTAIN) - len(miss)}/{len(DOC_MUST_CONTAIN)}'
            + (f' · 누락 {miss}' if miss else ''))
    else:
        chk('D-11 문서에 표의 수치가 축자로 실려 있다', False,
            f'⛔문서 없음: {DOC}')

    # ── 출력 ────────────────────────────────────────────────────────────
    print('=== B-21b Δ 민감도 재현 (2026-08-27) ===')
    for name, good, detail in checks:
        print(f'  {"PASS" if good else "FAIL"}  {name}')
        if detail:
            print(f'        {detail}')

    print('\n── Δ 스윕 (PRIMARY: KGRID1 vs SPEC25) ' + '─' * 30)
    print(f'{"Δ(%)":>14} | {"희소 hi/lo/안":>13} {"라벨":>10} '
          f'| {"혼잡 hi/lo/안":>13} {"라벨":>10}')
    sweep = [5.0, 8.0, DELTA_ADOPTED, 10.0475, 10.5,
             EXPECTED['crowded']['boundary'], 10.74, 10.8, 11.7, 13.0,
             EXPECTED['sparse']['boundary'], 13.37, DELTA_REJECTED, 15.0,
             18.675, 20.0]
    for d in sweep:
        cols = []
        for st_name in G.STRATA:
            p = prim[st_name]
            t = G.sign_tost(p['diffs_pct'], d)
            lab = G.label_primary(st_name, p, t, neg[st_name])[0]
            cols.append((f'{t["below_hi"]}/{t["above_lo"]}/{t["n_inside"]}',
                         lab.split('-')[-1]))
        print(f'{d:>14.10g} | {cols[0][0]:>13} {cols[0][1]:>10} '
              f'| {cols[1][0]:>13} {cols[1][1]:>10}')

    n_fail = sum(1 for _n, g, _d in checks if not g)
    print(f'\n{len(checks) - n_fail}/{len(checks)} PASS')
    if n_fail:
        print('⛔문서와 어긋난다 — 표를 고치거나 스크립트를 고쳐라.')
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
