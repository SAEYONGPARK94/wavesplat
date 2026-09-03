"""`H3c-LAW` 분석기 — ⛔**결과 전에 작성·동결**. 판정은 여기서만 나온다."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gates_h3c as G                                                # noqa: E402

RES = os.path.join(HERE, 'results_h3c.jsonl')
META = os.path.join(HERE, 'meta_h3c.json')
OUT = os.path.join(HERE, 'verdict_h3c.json')


def main():
    rows = [json.loads(l) for l in open(RES, encoding='utf-8') if l.strip()]
    meta = json.load(open(META, encoding='utf-8'))
    r = rows[0]
    A, C = r['model_A'], r['model_C']
    gate_rows = G.gates(meta)
    blocked = G.blocked_by(gate_rows)
    if blocked:
        lab, msg = 'BLOCKED', '차단 게이트 실패: ' + ','.join(blocked)
    else:
        lab, msg = G.label(C['MRE'], C['has_signal'], A['MRE'])
    lab = 'H3c-LAW :: ' + lab

    v = dict(batch='H3c-LAW', kind='REANALYSIS-OF-FROZEN-DATA', new_runs=0,
             new_measurements=0, grade='EXPLORATORY', n_cells=r['n_cells'],
             gates=[dict(name=n, pass_=bool(o), detail=str(d)) for n, o, d in gate_rows],
             gates_pass=f'{sum(1 for _, o, _ in gate_rows if o)}/{len(gate_rows)}',
             blocked_by=blocked, threshold=G.MRE_THRESH,
             model_A=A, model_C=C, m_sweep=r['m_sweep'],
             improvement=dict(abs=A['MRE'] - C['MRE'],
                              rel_pct=100.0 * (C['MRE'] - A['MRE']) / A['MRE']),
             prediction=dict(written_before_results='LAW-FAMILY-MATTERS-NOT-ENOUGH',
                             stated_probability=0.60,
                             where='tmp/H3B_LAW_2026-09-03/PREREG.md §3 (결과 전)'),
             predecessor=dict(batch='H3b-LAW', status='BLOCKED (G-IMPORT, G-EXCL)',
                              why='내 게이트 명세 오류 둘 — 모듈 엡실론 · 줄바꿈',
                              exposure='차단 확인 중 PRIMARY 통계를 보았다'),
             label=lab, label_msg=msg,
             disclosures=[
                 '⛔H3 판정(OBSMODEL-NOT-FAITHFUL) 갱신 아님.',
                 '⛔배포 기본값 변경 아님 — wavesim/** 무수정, 코퍼스 재생성은 별도 사전등록.',
                 '⛔쇄파대 레짐 한정(FRF Duck) · 심해 미검 · 역방향 전이 교락(H3 상속).',
                 '⛔A(복사량)는 고정했다 — 이 배치는 법칙만 바꾼다. A 가 옳다는 주장 아님.',
                 '⛔Model B(raydrop_slope)는 결과 전 배제(정반사 게이트가 이 기하에서 발화 안 함).',
                 '⛔M 스윕은 보고 전용 — 판정은 M=1 로만 한다(모수 증가 금지).',
                 '⛔추정기 런 0회 · 신규 계측 0회.',
                 '⛔blind 아님 — 선행 H3b 가 차단되는 과정에서 PRIMARY 를 보았다 ⟹ 등급 EXPLORATORY 상한.',
                 '★판정표·문턱·적합 규칙은 H3b 에서 한 글자도 바꾸지 않았다(게이트 명세만 고쳤다).',
             ])
    json.dump(v, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'{lab}\n  {msg}')
    print(f"  gates {v['gates_pass']}")
    for n, o, d in gate_rows:
        print(f"   [{'OK ' if o else 'FAIL'}] {n:48s} {d}")
    print(f"  A: MRE {A['MRE']:.4f} (shuf p5 {A['shuf_p5']:.4f}, signal={A['has_signal']})")
    print(f"  C: MRE {C['MRE']:.4f} (shuf p5 {C['shuf_p5']:.4f}, signal={C['has_signal']})")
    print(f"  개선 {v['improvement']['rel_pct']:+.1f}%   M 스윕 " +
          ' '.join(f"M{k}={d['mre']:.4f}" for k, d in r['m_sweep'].items()))
    print(f"  예측 {v['prediction']['written_before_results']} (p={v['prediction']['stated_probability']})")
    return 0


if __name__ == '__main__':
    sys.exit(main())
