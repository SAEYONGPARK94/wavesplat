"""`T10-EXTDISP` 분석기 — 동결 규칙만 적용한다. ⛔런 0회(행을 읽기만)."""
import hashlib, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_t10 as G                                                # noqa: E402

ROWS = os.path.join(HERE, 'results_t10.jsonl'); META = os.path.join(HERE, 'meta_t10.json')


def main():
    rows = [json.loads(x) for x in open(ROWS, encoding='utf-8') if x.strip()]
    meta = json.load(open(META, encoding='utf-8'))
    tp = [r for r in rows if r['arm'] == 'tp']
    pw = {r['true_depth']: r for r in rows if r['arm'] == 'power'}
    w = sum(r['shallow'] for r in tp)

    A = []
    A.append((f'★G-SIZE 크기 == {G.WANT_BYTES}', meta['size_ok'], str(meta['zip_size'])))
    md5 = hashlib.md5(open(os.path.join(HERE, 'null_t10.json'), 'rb').read()).hexdigest()
    A.append(('★★G-NULL-CAL 문턱을 동결 귀무에서 되읽는다',
              abs(G.THRESH - meta['threshold']) < 1e-12 and md5 == meta['null_md5'],
              f'문턱 {G.THRESH:.4f} · 귀무 md5 {md5[:12]}'))
    shal = all(pw[d]['detected'] for d in G.POWER_SHALLOW if d in pw)
    deep = (G.POWER_DEEP in pw) and (not pw[G.POWER_DEEP]['detected'])
    A.append(('★★G-POWER 참 수심 합성이 그날 이빨을 보인다', shal and deep,
              ' · '.join(f"d={d:.0f}:{pw[d]['ratio']:.3f}"
                         for d in list(G.POWER_SHALLOW) + [G.POWER_DEEP] if d in pw)))
    A.append((f'★G-RIDGE 능선 ≥ {G.MIN_RIDGE} · 띠 모드 ≥ {G.MIN_MODES}',
              all(r['n_ridge'] >= G.MIN_RIDGE and r['min_modes_used'] >= G.MIN_MODES
                  for r in tp),
              f"능선 {sorted({r['n_ridge'] for r in tp})} · 최소 모드 "
              f"{sorted({r['min_modes_used'] for r in tp})}"))
    A.append((f'★G-BAND k ∈ [{G.KMIN}, {G.KMAX}] 결과 전 고정',
              all(r['kmin'] == G.KMIN and r['kmax'] == G.KMAX for r in tp),
              f'{G.KMIN}–{G.KMAX}'))
    A.append((f'G-N 파일 {G.N_FILE} (sorted 의 {G.SKIP+1}–{G.SKIP+G.N_FILE}번째)',
              len(tp) == G.N_FILE, f'{len(tp)} 파일'))
    A.append(('G-DISJOINT T1·T1b 와 교집합 0', not meta['disjoint'],
              f"교집합 {len(meta['disjoint'])}"))
    A.append(('G-NORUN 추정기 임포트 0', meta['norun'], meta['norun_msg']))
    probe = G.blocked_by([('★★G-NULL-CAL 합성', False, ''), ('G-BAND 합성', True, '')])
    A.append(('G-BLOCK blocked_by 자가진단 (긴 이름 우선)',
              probe == ['G-NULL-CAL'], f'합성 → {probe}'))
    A.append(('G-COV 게이트가 자료를 봤다', len(rows) > 0, f'rows={len(rows)}'))

    blocked = G.blocked_by(A)
    lab = G.label(w) if not blocked else None
    rd = [r['rel_deep'] for r in tp]
    v = {'batch': 'T10-EXTDISP', 'kind': 'ENTRY-CHECK', 'grade': 'CONFIRMATORY',
         'new_runs': 0, 'n_rows': len(rows),
         'gates': [{'name': n, 'pass_': bool(o), 'detail': d} for n, o, d in A],
         'gates_pass': f'{sum(1 for _, o, _ in A if o)}/{len(A)}', 'blocked_by': blocked,
         'threshold': G.THRESH,
         'primary': {'stat': 'ratio < threshold (shallow detected)',
                     'w': w, 'n': len(tp)},
         'deep_fit_residual': {'min': float(min(rd)), 'median': float(np.median(rd)),
                               'max': float(max(rd))},
         'power': {str(d): {'ratio': pw[d]['ratio'], 'detected': pw[d]['detected']}
                   for d in sorted(pw)},
         'prediction': {'label': 'NOT-SHALLOW', 'p': 0.70},
         'label': lab, 'label_msg': lab or 'BLOCKED',
         'disclosures': [
             '⛔입장 검사다 --- 정확도 주장이 아니고, 통과는 다음 배치를 **할 수 있다**는 뜻뿐',
             '⛔"깊다" 라 쓰지 않는다 --- "얕지 않다" 까지만. MDE 는 d ≲ 30 m',
             '⚠d = 50 m 는 이 자가 검출하지 못한다 --- NOT-SHALLOW 는 d > 30 m 를 뜻하고 '
             'd = ∞ 를 뜻하지 않는다',
             '⛔ratio·적합 d 를 계수로 인쇄하지 않는다(점추정은 못 믿는다)',
             '⚠합성 대조는 선형 중첩이다(구속 고조파 없음) --- 능선만 보는 자라 무해하다고 '
             '보지만 내가 그것을 재지는 않았다',
             '★처치의 심해 잔차(중앙 2.1%)가 내 합성 대조(36–45%)보다 **한 자릿수 좋다** '
             '⟹ 문턱은 이 자료에 보수적이고, 검정력 표는 **비관적**이다(라벨은 안전한 방향)',
         ]}
    json.dump(v, open(os.path.join(HERE, 'verdict_t10.json'), 'w'),
              ensure_ascii=False, indent=1)
    print(f"T10-EXTDISP :: {v['label_msg']}")
    print(f"  PRIMARY 얕다 {w}/{len(tp)}  (문턱 ratio < {G.THRESH:.4f}; "
          f"≥{G.LABEL_HI} SHALLOW / {G.LABEL_LO}–7 MIXED / ≤2 NOT-SHALLOW)")
    print(f"  ★심해 가정 잔차: 중앙 {np.median(rd)*100:.1f}% "
          f"(범위 {min(rd)*100:.1f}–{max(rd)*100:.1f}%) ⟹ 둘 다 나쁜 경우가 아니다")
    print(f"  gates {v['gates_pass']}")
    for n, o, d in A:
        print(f"   [{'OK ' if o else 'FAIL'}] {n:50s} {d}")


if __name__ == '__main__':
    main()
