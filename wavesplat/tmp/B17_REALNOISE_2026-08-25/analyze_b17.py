"""B-17 분석기 — ★**판정 규칙·문턱·라벨은 전부 `gates_b17` 가 낸다**(여기서 재기재 없음).

여기가 하는 것은 자료 배관뿐이다: `results_b17_*.jsonl` 을 glob 으로 모아
`verdict_b17.json`(**`verdict.label` 단일 문자열 + `verdict.gates`**)을 쓴다.
`tmp/_audit/check_batch_artifacts.py` 가 그 형태를 읽는다.

⛔**결과 0행이면 `verdict_b17.json` 을 쓰지 않고 exit 1 한다** (M-4, 2026-08-26).
초판은 0행에서도 exit 0 으로 `label`·`gates` 를 갖춘 verdict 을 써서
`tmp/_audit/check_batch_artifacts.py` 의 *"인용 가능"* 판정을 통과시켰다 —
**아무것도 재지 않은 배치가 잰 배치처럼 보인다.** 게이트 출력은 계속 인쇄하되
(사전 스모크용) **판정 파일은 만들지 않는다.**
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
B2 = os.path.abspath(os.path.join(HERE, '..', 'B2_REAL_2026-08-23'))
B14 = os.path.abspath(os.path.join(HERE, '..', 'B14_OBSMODEL_2026-08-24'))
sys.path.insert(0, HERE)
import gates_b17 as G                                          # noqa: E402


def _load(pat):
    rows = []
    for p in sorted(glob.glob(pat)):
        for l in open(p):
            try:
                rows.append(json.loads(l))
            except json.JSONDecodeError:
                pass
    return rows


def main():
    rows = _load(os.path.join(HERE, 'results_b17_*.jsonl'))
    ok = [r for r in rows if 'error' not in r]
    if not ok:
        # ★M-4: 판정 파일을 만들지 않는다. 게이트만 찍고 실패로 끝낸다.
        for gid, g, m in G.gates(rows, sensor_origin=(
                json.load(open(os.path.join(HERE, 'sensor_origin.json')))
                if os.path.exists(os.path.join(HERE, 'sensor_origin.json')) else None)):
            print(f'  [{"PASS" if g else "FAIL"}] {gid}: {str(m)[:86]}')
        print(f'\n⛔유효 결과 0행 ({len(rows)} 행 중) — '
              f'verdict_b17.json 을 **쓰지 않는다**. 배치를 먼저 돌려라.')
        return 1
    b2 = _load(os.path.join(B2, 'results.jsonl'))
    b14b = _load(os.path.join(B14, 'results_b14b_*.jsonl'))
    so = None
    sp = os.path.join(HERE, 'sensor_origin.json')
    if os.path.exists(sp):
        so = json.load(open(sp))

    gs = G.gates(rows, sensor_origin=so, b2_rows=b2, b14b_rows=b14b)
    print('=' * 78)
    print('B-17 — 실측 잡음을 거리·입사각 의존으로 (외부 심사 P-5)')
    print('=' * 78)
    print('\n## 게이트')
    for gid, g, m in gs:
        print(f'  [{"PASS" if g else "FAIL"}] {gid}: {str(m)[:86]}')

    # ── 팔별 요약 ───────────────────────────────────────────────────────────
    summ, strat, pv = {}, {}, {}
    for a in G.ARM_ORDER:
        summ[a] = G.beats_zero(ok, a)
        strat[a] = G.stratum(ok, a)
        if a != G.REF_ARM:
            pv[a] = G.paired(ok, a)
    print(f'\n## 참 제로필드를 이기는 레코드 수 (n={G.N}, 명목 {G.NOMINAL} · '
          f'가족단위 {G.CONFIRM}, α={G.FAMILY_ALPHA:g})')
    for a in G.ARM_ORDER:
        s, st = summ[a], strat[a]
        mark = '★' if a == G.PRIMARY_ARM else ('·' if a in G.FAMILY else '~')
        med = 'n/a' if s['median'] is None else f"{s['median']:.4f}"
        p = 'n/a' if s['p_one_sided'] is None else f"{s['p_one_sided']:.5f}"
        print(f"  {mark}{a:13s} {s['wins']:2d}/{s['n']:<2d} 중앙 {med:>7s} p={p:>8s}   "
              f"대형 {st['large']['wins']}/{st['large']['n']} · "
              f"소형 {st['small']['wins']}/{st['small']['n']} · gap {st['gap']:+.3f}")

    print(f'\n## 짝 비교 vs {G.REF_ARM} (기준값 소거, n={G.N})')
    for a, d in pv.items():
        lab = G.paired_label(d)
        rm = 'n/a' if d['rel_median_pct'] is None else f"{d['rel_median_pct']:+7.2f}%"
        p = 'n/a' if d['p_one_sided'] is None else f"{d['p_one_sided']:.5f}"
        fam = '가족' if a in G.FAMILY else '탐색'
        print(f"  {a:13s} [{fam}] 개선 {d['better']:2d}/{d['n']:<2d} 상대중앙 {rm} "
              f"p={p:>8s}  → {lab[0]}")

    # ── ★PRIMARY = F-2 층 판정 ───────────────────────────────────────────────
    # ★m-10: PRIMARY 팔의 `frac_cos_clamped` 를 함께 넘긴다 ⟹ 붕괴 판정이 **공시된
    #   취약 레코드**(클램프 과다)에 기댔는지 `meaning` 에 자동으로 실린다.
    clamp = G.clamp_map(ok, G.PRIMARY_ARM)
    prim = G.stratum_label(strat[G.PRIMARY_ARM], strat[G.REF_ARM], clamp=clamp)
    print(f'\n## 층 대비 (⚠**검정력 없음** — Fisher 는 기술용, p 를 주장하지 않는다)')
    for a in G.ARM_ORDER:
        st = strat[a]
        fp = st['fisher_p_descriptive']
        print(f"  {a:13s} 대형 {st['frac_large']:.3f} · 소형 {st['frac_small']:.3f} · "
              f"gap {st['gap']:+.3f}  (Fisher descr. p="
              f"{'n/a' if fp is None else format(fp, '.4f')})")
    print('\n' + '=' * 78)
    print(f'## PRIMARY 판정 = {prim[0]}')
    print(f'##   {prim[1]}')
    print('=' * 78)

    v = {'batch': 'B-17', 'n_rows': len(rows), 'n_ok': len(ok),
         'verdict': {'label': prim[0], 'meaning': prim[1],
                     'primary_arm': G.PRIMARY_ARM, 'ref_arm': G.REF_ARM,
                     'gates': [{'id': g, 'ok': bool(o), 'msg': str(m)} for g, o, m in gs],
                     'gates_pass': sum(1 for g in gs if g[1]), 'gates_total': len(gs)},
         'label': prim[0],
         'gates': [{'id': g, 'ok': bool(o), 'msg': str(m)} for g, o, m in gs],
         'beats_zero': summ, 'stratum': strat, 'paired_vs_const': pv,
         'paired_labels': {a: G.paired_label(d)[0] for a, d in pv.items()},
         'rules': {'NOMINAL': G.NOMINAL, 'CONFIRM': G.CONFIRM,
                   'FAMILY': G.FAMILY, 'FAMILY_ALPHA': G.FAMILY_ALPHA,
                   'EXPLORATORY': G.EXPLORATORY,
                   'STRATUM_SURVIVE_RATIO': G.STRATUM_SURVIVE_RATIO,
                   'STRATUM_COLLAPSE_RATIO': G.STRATUM_COLLAPSE_RATIO,
                   'STRATUM_SMALL_CAP': G.STRATUM_SMALL_CAP},
         'sensor': {'xy': list(G.SENSOR_XY), 'h_declared': G.SENSOR_H,
                    'h_alt': G.SENSOR_H_ALT, 'cos_min': G.COS_MIN},
         'clamp_frac': {'thresh': G.CLAMP_FRAC_THRESH,
                        'primary_arm': clamp,
                        'ref_arm': G.clamp_map(ok, G.REF_ARM)},
         # ⛔줄번호를 손으로 적지 않는다 — `thesis_anchors()` 가 정규식으로 찾은
         #   **현재** 위치와 값을 그대로 굳힌다(m-5).
         'thesis_curve': {'sigma0': G.SIGMA0, 'sigma_r': G.SIGMA_R, 'd_ref': G.D_REF,
                          'sigma_r_src': 'wavesim/noise_model.py::add_noise(sigma_range=)',
                          'd_ref_src': ('wavesplat/model/confidence.py::'
                                        'observation_weights(d_ref=)'),
                          'anchors': G.thesis_anchors()}}
    json.dump(v, open(os.path.join(HERE, 'verdict_b17.json'), 'w'),
              ensure_ascii=False, indent=1)
    print('\n★verdict_b17.json 기록 — 동결 분석기 출력이다')


if __name__ == '__main__':
    sys.exit(main() or 0)
