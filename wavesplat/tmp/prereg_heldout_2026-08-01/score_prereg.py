"""Scoring for the 2026-08-01 held-out prereg. Runs ONLY after all arms are in.

Implements docs/PREREG_REAL_HELDOUT_BANKMMSE_2026-08-01.md sections 3.1-3.4
exactly as frozen, and writes RESULT.md with the applicable §3.5 sentence copied
verbatim. No threshold is chosen here; every number is MEASURED.
"""
import os, sys, json, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
os.chdir(ROOT)
JSONL = 'tmp/prereg_heldout_2026-08-01/results.jsonl'
OUT = 'tmp/prereg_heldout_2026-08-01/RESULT.md'

RECORDS = ['AA14', 'LJ18g', 'AA15a', 'AA15b', 'YS17']
STRICT_RECORDS = ['AA14', 'LJ18g']
ARMS = ['A', 'B', 'C', 'D', 'E']
ARM_DESC = {
    'A': 'bank-MMSE + refit R=2 s (claim)',
    'B': 'bank-MMSE frozen (shipped default)',
    'C': 'v2_konly init + refit R=2 s (low-cost control 1)',
    'D': 'bank-MMSE frozen, memory 20 s (low-cost control 2)',
    'E': 'bank-MMSE frozen, kp_hint = 2 x kp_blind (low-cost control 3)',
}

# §3.5 pre-committed verdict sentences, copied verbatim from the prereg.
V_PASS = ("bank-MMSE + rolling refit은 신선 실측 창에서 학습팔과 동일한 바를 통과한 "
          "**이 프로그램 최초의 실측 정확도 양성**이다 — 단 5개 판정 단위 중 "
          "3(AA15a·AA15b·YS17)은 **창-신선·레코드-재사용** 단위이며, 완전 신선 "
          "레코드는 2개(STRICT 표 §3.4를 반드시 병기).")
V_CHEAP = ("refit은 실측에서 재현되나 **저가 대안이 같은 이득을 준다.** "
           "이는 설정 선택이지 novelty 자산이 아니다.")
V_NULL = ("이 검증셋이 검출할 수 있는 크기의 refit 이득은 실측에서 확인되지 않았다. "
          "검정력 상한은 데이터셋 구조에서 온다(레코드 5개).")
V_ALWAYS = ("어느 경우든: 07-29의 3레코드 결과는 **탐색적(exploratory)**으로 재라벨하고, "
            "원장 §2E의 \"novelty 자산\" 승격을 철회한다.")


def load():
    arms, meta, excl = {}, {}, {}
    with open(JSONL) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get('kind') == 'arm':
                arms[(r['window'], r['arm'])] = r
            elif r.get('kind') == 'window':
                (meta if r.get('status') == 'ok' else excl)[r['window']] = r
    return arms, meta, excl


def main():
    arms, meta, excl = load()
    windows = [w for w in meta]
    order = {w: i for i, w in enumerate(
        [f'AA14_w{i}' for i in range(6)] + [f'LJ18g_w{i}' for i in range(6)]
        + [f'{r}_w{i}' for r in ('AA15a', 'AA15b', 'YS17') for i in (0, 2, 4)])}
    windows.sort(key=lambda w: order.get(w, 99))
    missing = [(w, a) for w in windows for a in ARMS if (w, a) not in arms]
    if missing:
        print('INCOMPLETE, missing:', missing)
        sys.exit(1)

    S = {(w, a): arms[(w, a)]['sustained_mean'] for w in windows for a in ARMS}
    rec_of = {w: arms[(w, 'A')]['record'] for w in windows}

    # ---------------------------------------------------------------- §3.1
    rec_rows, signs = [], []
    for r in RECORDS:
        ws = [w for w in windows if rec_of[w] == r]
        d = [S[(w, 'B')] - S[(w, 'A')] for w in ws]
        med = float(np.median(d)) if d else float('nan')
        rec_rows.append((r, len(ws), med, [round(x, 4) for x in d]))
        signs.append(med > 0)
    n_pos = sum(signs)
    p31 = 0.5 ** len(RECORDS) if n_pos == len(RECORDS) else None
    pass31 = n_pos == len(RECORDS)

    # ---------------------------------------------------------------- §3.2
    def improved(a):
        return sum(1 for w in windows if S[(w, a)] < S[(w, 'B')])
    cnt = {a: improved(a) for a in ('A', 'C', 'D', 'E')}
    c_best_arm = max(('C', 'D', 'E'), key=lambda a: cnt[a])
    c_best = cnt[c_best_arm]
    bar = math.ceil(1.5 * c_best)
    unpassable = bar > len(windows)
    pass32 = (not unpassable) and cnt['A'] >= bar

    # ---------------------------------------------------------------- §3.3
    harm = [(w, S[(w, 'A')] - S[(w, 'B')]) for w in windows
            if S[(w, 'A')] - S[(w, 'B')] > 0.02]
    pass33 = len(harm) == 0

    # ---------------------------------------------------------------- §3.4
    strict = [w for w in windows if rec_of[w] in STRICT_RECORDS]

    # ------------------------------------------------------------- compute
    comp = {}
    for a in ARMS:
        m3 = [v for w in windows for v in arms[(w, a)]['fit_wall_s_m3']]
        mem = [v for w in windows for v in (arms[(w, a)]['fit_wall_s_member'] or [])]
        comp[a] = dict(n_fits_total=sum(arms[(w, a)]['n_fits'] for w in windows),
                       med_m3=float(np.median(m3)),
                       min_m3=float(np.min(m3)), max_m3=float(np.max(m3)),
                       med_member=float(np.median(mem)) if mem else None,
                       arm_wall_total=sum(arms[(w, a)]['arm_wall_s']
                                          for w in windows))

    L = []
    P = L.append
    P('# RESULT — 실측 held-out 사전등록 실행 (2026-08-01)')
    P('')
    P('사전등록: `docs/PREREG_REAL_HELDOUT_BANKMMSE_2026-08-01.md` (구속력 있음). '
      '하네스: `tmp/prereg_heldout_2026-08-01/run_prereg_heldout.py`, '
      '원자료: `results.jsonl`.')
    P('**모든 수치 MEASURED.** 실행 중 임계·창셋·시드·팔 정의 변경 없음.')
    P('')
    P('## 0. 무결성 통제')
    P('')
    P(f'- 재생 복제 assert: **{len(meta)}/{len(meta) + len(excl)} 창 통과** '
      f'(제외 {len(excl)}창{"" if not excl else ": " + ", ".join(excl)}).')
    P('- 하네스 대조(`verify_vs_0729.py`, PRIMARY 밖 소비창): AA15a_w1 / YS17_w1 의 '
      'frozen·refit2 곡선이 07-29 `rolling_refit.json`과 **bit-identical** '
      '(max|Δcurve| = 0.000e+00).')
    P('- arm C 3시드 병렬(spawn pool) = 인프로세스와 **bit-identical** '
      '(`test_parallel.py`, max|Δ| = 0.000e+00).')
    P('- 모든 팔에서 `init_method` 명시 (arm C: `v2_konly`; bank 팔: 포트된 '
      '`fit_bank_mmse` 직접 호출).')
    P('')
    P('## 1. 팔별 지속(sustained) nRMSE — PRIMARY 21창')
    P('')
    P('| window | zero | A refit2 | B frozen | C v2k+refit | D mem20 | E kp×2 | Δ(B−A) |')
    P('|---|---|---|---|---|---|---|---|')
    for w in windows:
        z = meta[w]['zero_sustained']
        P(f'| {w} | {z:.3f} | {S[(w,"A")]:.4f} | {S[(w,"B")]:.4f} | '
          f'{S[(w,"C")]:.4f} | {S[(w,"D")]:.4f} | {S[(w,"E")]:.4f} | '
          f'{S[(w,"B")]-S[(w,"A")]:+.4f} |')
    P('')
    P('`zero` = 같은 마스크·같은 정규화의 제로-예측자 선 (진단용, 판정 아님). '
      'nRMSE ≳ zero 는 그 창에서 스킬 없음을 뜻한다.')
    P('')
    P('## 2. §3.1 PRIMARY — 레코드-군집 일측 부호검정')
    P('')
    P('단위 = 레코드 5개. 각 레코드의 신선 창 중앙값 Δ = (B − A); 5/5 A우세만 통과.')
    P('')
    P('| record | n win | median Δ(B−A) | 부호 | 창별 Δ |')
    P('|---|---|---|---|---|')
    for r, n, med, ds in rec_rows:
        P(f'| {r} | {n} | {med:+.4f} | {"A 우세" if med > 0 else "B 우세"} | {ds} |')
    P('')
    P(f'**{n_pos}/5 레코드가 A 우세** ⟹ §3.1 '
      f'{"**PASS**" if pass31 else "**FAIL**"}'
      + (f' (일측 p = {p31:.5f})' if pass31 else ' (4/5 이하 ⟹ p ≥ 0.156)'))
    P('')
    P('## 3. §3.2 PRIMARY — 저가대조 마진 (1.5 × C_best)')
    P('')
    P('| arm | B 대비 개선 창 수 / 21 |')
    P('|---|---|')
    for a in ('A', 'C', 'D', 'E'):
        P(f'| {a} ({ARM_DESC[a]}) | {cnt[a]}/21 |')
    P('')
    P(f'C_best = **{c_best}/21** (arm {c_best_arm}). '
      f'바 = ceil(1.5 × {c_best}) = **{bar}**. A = **{cnt["A"]}/21**.')
    if unpassable:
        P('')
        P(f'⚠ 1.5 × C_best = {1.5*c_best:.1f} > 21 ⟹ §3.2는 **구조상 통과 불가** '
          '(사전등록 §7 R3에 명시된 경우). 판정은 §3.5 둘째 문구로 자동 귀결된다.')
    P('')
    P(f'§3.2 {"**PASS**" if pass32 else "**FAIL**"}')
    P('')
    P('## 4. §3.3 해악 게이트')
    P('')
    P(f'A가 B보다 +0.02 초과 악화된 창: **{len(harm)}개**.')
    if harm:
        P('')
        P('| window | A − B |')
        P('|---|---|')
        for w, d in sorted(harm, key=lambda t: -t[1]):
            P(f'| {w} | {d:+.4f} |')
    P('')
    P(f'§3.3 {"**PASS**" if pass33 else "**FAIL**"} '
      '(위반 창 수는 통과 여부와 무관하게 명시 보고 의무).')
    P('')
    P('## 5. §3.4 STRICT 표 — 완전 신선 레코드 2개 (12창), 서술적')
    P('')
    P('n=2 부호바닥 p=0.25 ⟹ **어떤 유의 주장도 하지 않는다.**')
    P('')
    P('| window | A | B | Δ(B−A) | C | D | E |')
    P('|---|---|---|---|---|---|---|')
    for w in strict:
        P(f'| {w} | {S[(w,"A")]:.4f} | {S[(w,"B")]:.4f} | '
          f'{S[(w,"B")]-S[(w,"A")]:+.4f} | {S[(w,"C")]:.4f} | '
          f'{S[(w,"D")]:.4f} | {S[(w,"E")]:.4f} |')
    sa = float(np.mean([S[(w, 'A')] for w in strict]))
    sb = float(np.mean([S[(w, 'B')] for w in strict]))
    n_imp = sum(1 for w in strict if S[(w, 'A')] < S[(w, 'B')])
    P('')
    P(f'STRICT 평균: A {sa:.4f} vs B {sb:.4f} (Δ {sb-sa:+.4f}), '
      f'A 개선 {n_imp}/12창. 레코드 중앙값 Δ: '
      + ', '.join(f'{r} {m:+.4f}' for r, n, m, _ in rec_rows
                  if r in STRICT_RECORDS) + '.')
    P('')
    P('## 6. Lead-resolved (§2.3 의무)')
    P('')
    P('창별 곡선 전체는 `results.jsonl`의 `curve` 필드. 아래는 창 평균 '
      '(리드 0-14 s = 전 21창, 15-20 s = LJ18g·YS17 9창만).')
    P('')
    P('| τ [s] | n win | A | B | C | D | E | zero |')
    P('|---|---|---|---|---|---|---|---|')
    for tau in range(0, 21):
        ws = [w for w in windows if tau <= arms[(w, 'A')]['lead_max']]
        if not ws:
            continue
        row = []
        for a in ARMS:
            row.append(np.mean([arms[(w, a)]['curve'][tau] for w in ws]))
        zz = np.mean([meta[w]['zero_curve'][tau] for w in ws])
        P(f'| {tau} | {len(ws)} | ' + ' | '.join(f'{v:.3f}' for v in row)
          + f' | {zz:.3f} |')
    P('')
    P('## 7. 계산 비용 (사전등록 §4 의무)')
    P('')
    P('| arm | 총 fit 수 | fit wall (median-of-3) med [min–max] s | 단일멤버 med s | 팔 총 wall s |')
    P('|---|---|---|---|---|')
    for a in ARMS:
        c = comp[a]
        mm = f'{c["med_member"]:.2f}' if c['med_member'] is not None else '—'
        P(f'| {a} | {c["n_fits_total"]} | {c["med_m3"]:.2f} '
          f'[{c["min_m3"]:.2f}–{c["max_m3"]:.2f}] | {mm} | {c["arm_wall_total"]:.0f} |')
    dutyA = comp['A']['med_m3'] / 2.0
    dutyA_par = (comp['A']['med_member'] / 2.0) if comp['A']['med_member'] else None
    dutyC = comp['C']['med_m3'] / 2.0
    P('')
    P(f'- **arm A duty cycle @ R=2 s = {dutyA:.2f}** (median-of-3 fit '
      f'{comp["A"]["med_m3"]:.2f} s / 2 s). 멤버 병렬 시 구속수치는 단일멤버 '
      f'{comp["A"]["med_member"]:.2f} s ⟹ duty **{dutyA_par:.2f}**.')
    P(f'- arm C(v2_konly) duty @ R=2 s = **{dutyC:.1f}** — 3시드를 3프로세스로 '
      '병렬 실행한 벽시계이며, 실시간과 무관하게 **재적합이 원리적으로 불가능**한 '
      '영역이다(참고: 단일 v2_konly 초기화 ≈ 50-60 s).')
    P('- 논문의 13.4 ms는 **추적 단계만**이며 이 fit 비용을 포함하지 않는다. '
      '이 수치 없이 "실시간"이라 쓰지 않는다(§4).')
    P('')
    P('## 8. 판정 (§3.5 사전 공약 문구)')
    P('')
    P(f'- §3.1 {"PASS" if pass31 else "FAIL"} ({n_pos}/5)')
    P(f'- §3.2 {"PASS" if pass32 else "FAIL"} (A {cnt["A"]}/21 vs 바 {bar}'
      + (', 구조상 통과 불가' if unpassable else '') + ')')
    P(f'- §3.3 {"PASS" if pass33 else "FAIL"} (위반 {len(harm)}창)')
    P('')
    if pass31 and pass32 and pass33:
        P('> ' + V_PASS)
        verdict = 'PASS'
    elif pass31 and not pass32:
        P('> ' + V_CHEAP)
        verdict = 'CHEAP-ALTERNATIVE'
    elif not pass31:
        P('> ' + V_NULL)
        verdict = 'NULL'
    else:
        P('§3.1·§3.2는 통과했으나 §3.3(해악 게이트)이 미통과다. §3.5의 통과 문구는 '
          '**세 게이트 모두**를 전제하므로 인용할 수 없고, 이 조합을 덮는 사전 공약 '
          '문구는 존재하지 않는다. §3.6("실행 중 어떤 임계도 조정하지 않는다")에 따라 '
          '사후 문구를 만들지 않고 수치만 보고한다: **통과 주장 불가.**')
        verdict = 'NO-PRECOMMITTED-SENTENCE'
    P('')
    P('> ' + V_ALWAYS)
    P('')
    P('### ⛔ 금지 (사전등록 §3.5·§7 R2)')
    P('- "LJ18g만 빼면 4/4" 류의 사후 완화 금지.')
    P('- §3.1 미통과 시 "효과 없음" 표현 금지 — 정확한 문장은 '
      '"이 데이터셋이 검출할 수 있는 크기의 효과 없음"이다.')
    P('- 5개 판정 단위 중 3개는 창-신선·**레코드-재사용**이다(§7 R1); '
      '어떤 인용에도 STRICT 표(§5)를 병기한다.')
    P('')
    open(OUT, 'w').write('\n'.join(L) + '\n')
    json.dump(dict(verdict=verdict, pass31=pass31, n_pos=n_pos,
                   record_medians={r: m for r, _, m, _ in rec_rows},
                   pass32=pass32, counts=cnt, c_best=c_best,
                   c_best_arm=c_best_arm, bar=bar, unpassable=unpassable,
                   pass33=pass33, harm=[[w, d] for w, d in harm],
                   strict_mean=dict(A=sa, B=sb, improved=n_imp),
                   compute=comp, duty_A_m3=dutyA, duty_A_member=dutyA_par),
              open('tmp/prereg_heldout_2026-08-01/verdict.json', 'w'), indent=1)
    print('\n'.join(L[-40:]))
    print('\nwrote', OUT)


if __name__ == '__main__':
    main()
