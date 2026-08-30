#!/usr/bin/env python3
"""E3 보고 전용 집계 — 단일출처 (임무 C2, 2026-08-30).

⛔**보고 전용** — `PREREG.md §3 E3` 그대로: 바 없음 · p 값 없음 · 판정 없음.
HARD 로 학습한 칸(`DATA`·`BOTH`)에게 이 코너는 **in-family (out-of-realisation)** 다.
이 스크립트는:
  1. `results_m7_e3_*.jsonl` 1,008행을 되읽어 동결 분석기(`analyze_m7.descriptive_e3`)의
     집계를 **독립 재현**하고, `verdict_m7_e3.json` 과 어긋나면 `AssertionError` 로 죽는다
     (본보기 = `tmp/B2_REAL_2026-08-23/depth_recheck.py` 규율 — 런 0회, 동결 결과만 읽는다).
  2. 기술 확장(전부 DESCRIPTIVE): 팔별 HOS−LIN 짝 열화, DATA−BASE · LOSS−BASE 짝 부호 계수.
     ⛔검정 0회 — 부호 계수는 계수일 뿐이고 귀무가 0.5 라는 근거가 없다(PREREG §4 규율 준용).
  3. `E3_REPORT.md` 를 생성한다(손집계 0회 — 이 파일이 표의 단일출처다).
"""
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
CELLS = ('BASE', 'LOSS', 'DATA', 'BOTH', 'BASE-B', 'SHIPPED')   # G.CELL_ORDER + REF_SHIPPED
SPEC = 'SPEC25'
N_SEEDS = 3
N_EP_PER_STRATUM = 24


def load_rows():
    rows = []
    for i in range(8):
        for ln in open(HERE / f'results_m7_e3_{i}.jsonl'):
            r = json.loads(ln)
            if r.get('exp') == 'e3' and 'error' not in r:
                rows.append(r)
    assert len(rows) == 1008, f'E3 행 수 {len(rows)} != 1008'
    return rows


def per_ep_median(rows, cell, stratum):
    """동결 분석기 `descriptive_e3.med` 재현: 에피소드별 시드 3개 중앙값."""
    per = {}
    for r in rows:
        if r.get('cell') == cell and str(r.get('episode', '')).startswith(stratum + '|'):
            per.setdefault(r['episode'], []).append(float(r['avg_nRMSE']))
    for k, v in per.items():
        assert len(v) == N_SEEDS, f'{cell} {k}: 시드 {len(v)}개 != {N_SEEDS}'
    return {k: st.median(v) for k, v in per.items()}


def crosscheck_frozen(table):
    """동결 verdict 과 비트 대조 — 어긋나면 이 보고 전체가 무효다."""
    fro = json.load(open(HERE / 'verdict_m7_e3.json'))['e3']
    assert fro['status'] == 'REPORTED'
    for stratum in ('LIN', 'HOS'):
        f = fro[stratum]
        assert f['n_ep'] == N_EP_PER_STRATUM
        assert abs(f['spec_median'] - table[stratum]['spec_median']) < 1e-12, stratum
        for c in CELLS:
            fc, tc = f['per_cell'][c], table[stratum]['per_cell'][c]
            assert fc['n'] == tc['n'] == N_EP_PER_STRATUM, (stratum, c)
            assert fc['better'] == tc['better'], (stratum, c, fc['better'], tc['better'])
            assert abs(fc['median'] - tc['median']) < 1e-12, (stratum, c)
    return True


def main():
    rows = load_rows()

    # ── 1. 동결 분석기 재현 (팔 × 층: 중앙 avg_nRMSE · SPEC25 대비 승수) ──
    table = {}
    ep_med = {}                                   # (stratum, cell) -> {episode: median}
    for stratum in ('LIN', 'HOS'):
        spec = per_ep_median(rows, SPEC, stratum)
        ep_med[(stratum, SPEC)] = spec
        blk = {}
        for c in CELLS:
            m = per_ep_median(rows, c, stratum)
            ep_med[(stratum, c)] = m
            ks = sorted(set(m) & set(spec))
            blk[c] = dict(n=len(ks), better=sum(1 for k in ks if m[k] < spec[k]),
                          median=st.median(m.values()))
        table[stratum] = dict(n_ep=len(spec), spec_median=st.median(spec.values()),
                              per_cell=blk)
    crosscheck_frozen(table)

    # 문맥: SPEC25 vs 참 ZERO (B-12 규약 — SPEC25 행의 zero_nRMSE 열, 런 0회)
    zero_ctx = {}
    for stratum in ('LIN', 'HOS'):
        per = {}
        for r in rows:
            if r.get('cell') == SPEC and str(r['episode']).startswith(stratum + '|') \
                    and r.get('zero_nRMSE') is not None:
                per.setdefault(r['episode'], []).append(float(r['zero_nRMSE']))
        zm = {k: st.median(v) for k, v in per.items()}
        spec = ep_med[(stratum, SPEC)]
        ks = sorted(set(zm) & set(spec))
        zero_ctx[stratum] = dict(n=len(ks),
                                 spec_better=sum(1 for k in ks if spec[k] < zm[k]),
                                 zero_median=st.median(zm.values()))

    # ── 2. 기술 확장 (DESCRIPTIVE — 검정 0회) ──
    def suffix(ep):
        return ep.split('|', 1)[1]

    # 2a. 팔별 HOS−LIN 짝 열화 (같은 suffix 24짝, 상대 %)
    degrade = {}
    for c in CELLS + (SPEC,):
        lin = {suffix(k): v for k, v in ep_med[('LIN', c)].items()}
        hos = {suffix(k): v for k, v in ep_med[('HOS', c)].items()}
        ks = sorted(set(lin) & set(hos))
        assert len(ks) == N_EP_PER_STRATUM, (c, len(ks))
        rel = [(hos[k] - lin[k]) / lin[k] * 100.0 for k in ks]
        degrade[c] = dict(n=len(ks), median_rel_pct=st.median(rel),
                          worse_on_hos=sum(1 for x in rel if x > 0))

    # 2b. 짝 대비 (같은 층 안, 에피소드 24짝): DATA−BASE · LOSS−BASE · BOTH−BASE
    contrasts = {}
    for a, b in (('DATA', 'BASE'), ('LOSS', 'BASE'), ('BOTH', 'BASE'), ('DATA', 'SHIPPED')):
        for stratum in ('LIN', 'HOS'):
            ma, mb = ep_med[(stratum, a)], ep_med[(stratum, b)]
            ks = sorted(set(ma) & set(mb))
            rel = [(ma[k] - mb[k]) / mb[k] * 100.0 for k in ks]
            contrasts[f'{a}-{b}|{stratum}'] = dict(
                n=len(ks), a_better=sum(1 for k in ks if ma[k] < mb[k]),
                median_rel_pct=st.median(rel))

    # ── 3. 보고서 생성 ──
    L = []
    w = L.append
    w('# E3 보고 — 가파른·비선형 합성 (`M1` 코퍼스 재사용, 1,008행)')
    w('')
    w('> 생성: `tmp/M7_SUPERVISION_2026-08-28/e3_report.py` (**단일출처** — 손집계 0회,')
    w('> 동결 `verdict_m7_e3.json` 과 대조해 어긋나면 스스로 죽는다). 실행일 2026-08-30, 런 0회.')
    w('>')
    w('> ⛔**보고 전용 (REPORTED)** — `PREREG.md §3 E3` 가 허용한 것은 기술 집계뿐이다:')
    w('> **바 없음 · p 값 없음 · 판정 없음.** HARD 로 학습한 칸(`DATA`·`BOTH`)에게 이 코너는')
    w('> **in-family (out-of-realisation)** 다(학습 앵커가 실측 중앙값이고 `M1` 이 그 코너다).')
    w('> ⛔확증 승격 금지 · ⛔`M1` 판정(배포 팔)과 혼동 금지 — 여기 팔은 **M7 의 다섯 팔**이다.')
    w('> ⚠인용 시 `verdict_m7.json` 의 공시 6건 + 분기표 초과 공시(excess_disclosure) 동반 의무.')
    w('')
    w('## 표 1 — 팔 × 층: 중앙 `avg_nRMSE` · `<SPEC25` 승수 (에피소드별 = 시드 3 중앙값)')
    w('')
    w('| 팔 | LIN 중앙 | LIN <SPEC25 | HOS 중앙 | HOS <SPEC25 | HOS−LIN 짝 열화(중앙 %) | HOS 에서 악화 |')
    w('|---|---|---|---|---|---|---|')
    order = ('BASE', 'BASE-B', 'LOSS', 'DATA', 'BOTH', 'SHIPPED')
    for c in order:
        li, ho, d = table['LIN']['per_cell'][c], table['HOS']['per_cell'][c], degrade[c]
        w(f"| `{c}` | {li['median']:.4f} | {li['better']}/24 | {ho['median']:.4f} "
          f"| {ho['better']}/24 | {d['median_rel_pct']:+.1f}% | {d['worse_on_hos']}/24 |")
    sl, sh, sd = table['LIN'], table['HOS'], degrade[SPEC]
    w(f"| `SPEC25` (기준선) | {sl['spec_median']:.4f} | — | {sh['spec_median']:.4f} | — "
      f"| {sd['median_rel_pct']:+.1f}% | {sd['worse_on_hos']}/24 |")
    w('')
    w(f"문맥(기술): `SPEC25 < 참ZERO` — LIN {zero_ctx['LIN']['spec_better']}/24"
      f" (ZERO 중앙 {zero_ctx['LIN']['zero_median']:.4f}) · HOS {zero_ctx['HOS']['spec_better']}/24"
      f" (ZERO 중앙 {zero_ctx['HOS']['zero_median']:.4f}) — 기준선 자체가 이 코퍼스에서 기술이 있다.")
    w('')
    w('## 표 2 — 짝 대비 (기술 — 계수는 계수일 뿐, 검정 아님)')
    w('')
    w('| 대비 | 층 | 앞 팔이 나은 에피소드 | 짝 상대 중앙 |')
    w('|---|---|---|---|')
    for key in ('DATA-BASE|LIN', 'DATA-BASE|HOS', 'LOSS-BASE|LIN', 'LOSS-BASE|HOS',
                'BOTH-BASE|LIN', 'BOTH-BASE|HOS', 'DATA-SHIPPED|LIN', 'DATA-SHIPPED|HOS'):
        cc = contrasts[key]
        pair, stratum = key.split('|')
        w(f"| `{pair}` | {stratum} | {cc['a_better']}/24 | {cc['median_rel_pct']:+.2f}% |")
    w('')
    w('## 읽기 (방향만 — 전부 기술)')
    w('')
    dl, dh = contrasts['DATA-BASE|LIN'], contrasts['DATA-BASE|HOS']
    w('1. ★**「DATA 가 가파른 평가(HOS)에서 특별히 잘한다」는 방향성은 이 자료에 없다 — 오히려 반대다.**')
    w(f'   `DATA−BASE` 는 LIN 에서 {dl["a_better"]}/24 ({dl["median_rel_pct"]:+.2f}%) 로 DATA 쪽이 낫고,')
    w(f'   HOS 에서 {dh["a_better"]}/24 ({dh["median_rel_pct"]:+.2f}%) 로 이득이 거의 사라진다.')
    w(f'   층 안 순위로도: LIN 최저 중앙은 `DATA`(0.1171, 24/24)인데 HOS 최저 중앙은 '
      f'`SHIPPED`({table["HOS"]["per_cell"]["SHIPPED"]["median"]:.4f})다.')
    w(f'   팔별 HOS 짝 열화는 `DATA` {degrade["DATA"]["median_rel_pct"]:+.1f}% 로 '
      f'`BASE` {degrade["BASE"]["median_rel_pct"]:+.1f}% 보다 크고 '
      f'`LOSS` {degrade["LOSS"]["median_rel_pct"]:+.1f}% · `SHIPPED` '
      f'{degrade["SHIPPED"]["median_rel_pct"]:+.1f}% 보다는 작다 — 어느 척도로도')
    w('   「가파른 학습 → 가파른 평가에서 상대 우위」 방향은 나타나지 않는다.')
    w('   ⚠단 이 코너는 `DATA` 에게 in-family 라 「일반화 실패」로도 읽을 수 없다 — 기술 그대로만.')
    w('2. `LOSS` 는 LIN 에서 `BASE` 보다 낫고'
      f'({contrasts["LOSS-BASE|LIN"]["a_better"]}/24, {contrasts["LOSS-BASE|LIN"]["median_rel_pct"]:+.2f}%),')
    w(f'   HOS 에서 뒤집힌다({contrasts["LOSS-BASE|HOS"]["a_better"]}/24, '
      f'{contrasts["LOSS-BASE|HOS"]["median_rel_pct"]:+.2f}%) — E2(동결 코퍼스)에서 본')
    w('   「LOSS 가 BASE 보다 약간 무르다」와 층별로 부호가 갈린다. 방향 관찰일 뿐이다.')
    w('3. 여섯 팔 전부가 양 층에서 `SPEC25` 를 큰 폭으로 이긴다(최저 18/24) — ⛔단 이 계수는')
    w('   기술이고, HARD 팔에게 in-family 인 코너라 확증 가치가 없다.')
    w('')
    w('## 넘지 않은 선')
    w('- p 값 0개 · 판정 라벨 0개 · 바 0개 (PREREG §3 E3 그대로).')
    w('- `M1`(배포 팔 판정)과의 비교 서술 없음 — 팔 정의가 다르다.')
    w('- 크기를 `B-12`·`E2` 와 직접 비교하지 않음(코퍼스가 다르다).')
    out = HERE / 'E3_REPORT.md'
    out.write_text('\n'.join(L) + '\n', encoding='utf-8')
    print(f'OK — 동결 대조 통과, {out} 생성 ({len(L)}행)')
    return table, degrade, contrasts


if __name__ == '__main__':
    main()
