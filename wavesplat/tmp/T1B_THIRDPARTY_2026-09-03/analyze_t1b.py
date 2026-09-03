"""`T1b` 분석기 — 동결 규칙만 적용한다. ⛔런 0회(행을 읽기만 한다)."""
import hashlib, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_t1b as G                                                # noqa: E402

ROWS = os.path.join(HERE, 'results_t1b.jsonl'); META = os.path.join(HERE, 'meta_t1b.json')


def main():
    rows = [json.loads(x) for x in open(ROWS, encoding='utf-8') if x.strip()]
    meta = json.load(open(META, encoding='utf-8'))
    fr = [r for r in rows if r['group'] == 'fresh']
    sm = [r for r in rows if r['group'] == 'same']
    pos = [r for r in rows if r['group'] == 'ctrl_hos']
    neg = [r for r in rows if r['group'] == 'ctrl_linear']
    w = sum(r['skew_exceed'] for r in fr)
    ws = sum(r['skew_exceed'] for r in sm)
    wp = sum(r['skew_exceed'] for r in pos); wn = sum(r['skew_exceed'] for r in neg)

    A = []
    A.append((f'★G-SIZE 크기 == {G.WANT_BYTES}', meta['size_ok'], str(meta['zip_size'])))
    relerr = max(r['null_fft_relerr'] for r in rows)
    A.append(('★G-NULL 대체표본이 |FFT| 보존', relerr < G.NULL_TOL, f'max relerr {relerr:.2e}'))
    nm = max(abs(r['skew_null_mean']) for r in neg)
    A.append(('★★G-NULLMOMENT 귀무 왜도가 0 에 있다', nm < G.NULLMOMENT_TOL,
              f'|평균| max {nm:.4f}'))
    A.append((f'★★G-POWER 양성 ≥{G.POWER_POS_MIN}/8 · 음성 ≤{G.POWER_NEG_MAX}/8',
              wp >= G.POWER_POS_MIN and wn <= G.POWER_NEG_MAX,
              f'우리 HOS {wp}/{len(pos)} · 우리 선형 {wn}/{len(neg)}'))
    md5 = hashlib.md5(open(os.path.join(G.SHARED, 'surrogate.py'), 'rb').read()).hexdigest()
    A.append(('★G-SHARED 공유 모듈 md5 핀', md5 == meta['shared_md5'], md5[:12]))
    A.append((f'★G-DISJOINT FRESH ∩ T1 == ∅', not meta['disjoint'],
              f"교집합 {len(meta['disjoint'])}개"))
    A.append((f'G-N 블록 {G.N_BLOCK} · 프레임 {G.N_FRAME} · 대체표본 {G.N_SUR}',
              len(fr) == G.N_BLOCK and all(r['n_frames'] == G.N_FRAME for r in fr)
              and meta['n_sur'] == G.N_SUR,
              f'{len(fr)} 블록 · {sorted({r["n_frames"] for r in fr})}'))
    A.append(('G-SOLVER 생성 solver 명시 여부', True,
              '⛔미명시 — "HOS"·"완전 비선형" 이라 부르지 않는다'
              if not meta['solver_named'] else '명시됨'))
    A.append(('G-NORUN 추정기 임포트 0', meta['norun'], meta['norun_msg']))
    probe = G.blocked_by([('★G-SIZE 합성', False, ''), ('G-SOLVER 합성', True, '')])
    A.append(('G-BLOCK blocked_by 자가진단', probe == ['G-SIZE'], f'합성 → {probe}'))
    A.append(('G-COV 게이트가 자료를 봤다', len(rows) > 0, f'rows={len(rows)}'))

    blocked = G.blocked_by(A)
    lab = G.label(w) if not blocked else None
    v = {'batch': 'T1b-THIRDPARTY', 'kind': 'CONFIRMATORY', 'new_runs': 0,
         'n_rows': len(rows),
         'gates': [{'name': n, 'pass_': bool(o), 'detail': d} for n, o, d in A],
         'gates_pass': f'{sum(1 for _, o, _ in A if o)}/{len(A)}', 'blocked_by': blocked,
         'primary': {'stat': 'FRESH pooled skewness > surrogate q95',
                     'w': w, 'n': len(fr)},
         'secondary_same': {'w': ws, 'n': len(sm),
                            'note': '⛔T1 과 독립 아님 — 기제 확인 전용'},
         'controls': {'pos_hos': f'{wp}/{len(pos)}', 'neg_linear': f'{wn}/{len(neg)}'},
         'skew_fresh': {'min': min(r['skew'] for r in fr),
                        'median': float(np.median([r['skew'] for r in fr])),
                        'max': max(r['skew'] for r in fr)},
         'q95_fresh_median': float(np.median([r['skew_q95'] for r in fr])),
         'prediction': {'label': 'THIRDPARTY-PHASE-COUPLED', 'p': 0.55},
         'label': lab, 'label_msg': lab or 'BLOCKED',
         'disclosures': [
             '⛔solver 미명시 — "HOS"·"완전 비선형" 이라 부르지 않는다',
             '⛔라벨은 위상 결합에 대한 것이지 「모델류」에 대한 것이 아니다',
             '⚠FRESH 12블록 중 angle30h0.5 **한 블록이 부분 노출**(md5·σ 만; '
             'PRIMARY 통계량은 계산되지 않았다). 선택 규칙은 그 전에 정해졌다',
             '⛔SAME 팔로 T1 을 갱신하지 않는다 — 같은 자료라 독립이 아니다',
             '⛔우리 코퍼스와 검출률을 비교하지 않는다(문턱이 다르면 검정력이지 물리가 아니다)',
         ]}
    json.dump(v, open(os.path.join(HERE, 'verdict_t1b.json'), 'w'),
              ensure_ascii=False, indent=1)
    print(f"T1b-THIRDPARTY :: {v['label_msg']}")
    print(f"  PRIMARY(FRESH) {w}/{len(fr)}  (문턱 ≥{G.LABEL_HI} / {G.LABEL_LO}–7 / ≤3)")
    print(f"  SECONDARY(SAME, ⛔비독립) {ws}/{len(sm)}   ⟹ T1 은 24프레임에서 2/12 였다")
    print(f"  대조 양성 {wp}/{len(pos)} · 음성 {wn}/{len(neg)}")
    print(f"  FRESH 왜도 {v['skew_fresh']['min']:+.4f}/{v['skew_fresh']['median']:+.4f}"
          f"/{v['skew_fresh']['max']:+.4f}  vs q95 중앙 {v['q95_fresh_median']:.4f}")
    print(f"  gates {v['gates_pass']}")
    for n, o, d in A:
        print(f"   [{'OK ' if o else 'FAIL'}] {n:46s} {d}")


if __name__ == '__main__':
    main()
