"""`T1-THIRDPARTY` v2 분석기 — 동결 규칙만 적용한다. ⛔런 0회(행을 읽기만 한다)."""
import hashlib, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_t1 as G                                                 # noqa: E402

ROWS = os.path.join(HERE, 'results_t1.jsonl'); META = os.path.join(HERE, 'meta_t1.json')
VERD = os.path.join(HERE, 'verdict_t1.json')


def main():
    rows = [json.loads(x) for x in open(ROWS, encoding='utf-8') if x.strip()]
    meta = json.load(open(META, encoding='utf-8'))
    tp = [r for r in rows if r['group'] == 'tp']
    pos = [r for r in rows if r['group'] == 'ctrl_hos']
    neg = [r for r in rows if r['group'] == 'ctrl_linear']
    w = sum(r['skew_exceed'] for r in tp)
    wp = sum(r['skew_exceed'] for r in pos)
    wn = sum(r['skew_exceed'] for r in neg)

    A = []
    A.append((f'★G-SIZE 크기 == {G.WANT_BYTES}', meta['size_ok'], str(meta['zip_size'])))
    relerr = max(r['null_fft_relerr'] for r in rows)
    A.append(('★G-NULL 대체표본이 |FFT| 보존', relerr < G.NULL_TOL, f'max relerr {relerr:.2e}'))
    nm = max(abs(r['skew_null_mean']) for r in neg)
    A.append(('★★G-NULLMOMENT 귀무 왜도가 0 에 있다(선형 대조)',
              nm < G.NULLMOMENT_TOL, f'|평균| max {nm:.4f} < {G.NULLMOMENT_TOL}'))
    A.append((f'★★G-POWER 양성 {G.POWER_POS_MIN}/8 · 음성 ≤{G.POWER_NEG_MAX}/8',
              wp >= G.POWER_POS_MIN and wn <= G.POWER_NEG_MAX,
              f'우리 HOS {wp}/{len(pos)} · 우리 선형 {wn}/{len(neg)}'))
    md5 = hashlib.md5(open(os.path.join(G.SHARED, 'surrogate.py'), 'rb').read()).hexdigest()
    A.append(('★G-SHARED 공유 대체표본 모듈 md5 핀',
              md5 == meta['shared_md5'], md5[:12]))
    A.append((f'G-N 블록 {G.N_BLOCK} · 프레임 {G.N_FRAME} · 대체표본 {G.N_SUR}',
              len(tp) == G.N_BLOCK and all(r['n_frames'] == G.N_FRAME for r in tp)
              and meta['n_sur'] == G.N_SUR,
              f'{len(tp)} 블록 · {sorted({r["n_frames"] for r in tp})} 프레임'))
    A.append(('G-SOLVER 생성 solver 명시 여부를 기록', True,
              '⛔아카이브에 문서 없음 — solver 미명시 ⟹ "HOS" 라 부르지 않는다'
              if not meta['solver_named'] else f"명시됨: {meta['solver_hits'][:2]}"))
    A.append(('G-NORUN 추정기 임포트 0', meta['norun'], meta['norun_msg']))
    probe = G.blocked_by([('★G-SIZE 합성', False, ''), ('G-SOLVER 합성', True, '')])
    A.append(('G-BLOCK blocked_by 자가진단', probe == ['G-SIZE'], f'합성 → {probe}'))
    A.append(('G-COV 게이트가 자료를 봤다', len(rows) > 0, f'rows={len(rows)}'))

    blocked = G.blocked_by(A)
    lab = G.label(w) if not blocked else None
    v = {'batch': 'T1-THIRDPARTY', 'kind': 'CONFIRMATORY', 'new_runs': 0,
         'n_rows': len(rows), 'prereg': 'v2 (v1 은 자료 열기 전 폐기)',
         'gates': [{'name': n, 'pass_': bool(o), 'detail': d} for n, o, d in A],
         'gates_pass': f'{sum(1 for _, o, _ in A if o)}/{len(A)}',
         'blocked_by': blocked,
         'primary': {'stat': 'pooled skewness > surrogate q95', 'w': w, 'n': len(tp)},
         'controls': {'pos_hos': f'{wp}/{len(pos)}', 'neg_linear': f'{wn}/{len(neg)}'},
         'secondary_ct': {'w': sum(r['ct_exceed'] for r in tp), 'n': len(tp)},
         'skew_tp': {'min': min(r['skew'] for r in tp),
                     'median': float(np.median([r['skew'] for r in tp])),
                     'max': max(r['skew'] for r in tp)},
         'prediction': {'label': 'THIRDPARTY-PHASE-COUPLED', 'p': 0.75},
         'label': lab, 'label_msg': lab or 'BLOCKED',
         'disclosures': [
             '⛔solver 미명시 — 이 자료를 "HOS"·"완전 비선형" 이라 부르지 않는다',
             '⛔라벨은 **위상 결합**에 대한 것이지 「모델류」에 대한 것이 아니다',
             '⛔정확도 주장 아님 · 추정기 런 0회 · C1/C1c/C2 판정 갱신 아님',
             'v1 의 LS 잔차 자는 검정력 0 이라 PRIMARY 에서 뺐다(행에 relerr 만 남는다)',
         ]}
    json.dump(v, open(VERD, 'w'), ensure_ascii=False, indent=1)
    print(f"T1-THIRDPARTY :: {v['label_msg']}")
    print(f"  PRIMARY {w}/{len(tp)}  (문턱 ≥{G.LABEL_HI} / {G.LABEL_LO}–9 / ≤4)")
    print(f"  대조 양성 우리HOS {wp}/{len(pos)} · 음성 우리선형 {wn}/{len(neg)}")
    print(f"  SECONDARY 마루골비 {v['secondary_ct']['w']}/{len(tp)}")
    print(f"  제3자 왜도 {v['skew_tp']['min']:+.4f}/{v['skew_tp']['median']:+.4f}"
          f"/{v['skew_tp']['max']:+.4f}")
    print(f"  gates {v['gates_pass']}")
    for n, o, d in A:
        print(f"   [{'OK ' if o else 'FAIL'}] {n:44s} {d}")


if __name__ == '__main__':
    main()
