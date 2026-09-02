"""`R3-GEOMETRY` 분석기 — ⛔런 0회. 규칙은 `gates_r3.py` 단일출처."""
import json, os, statistics as st, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_r3 as G                                                # noqa: E402
ROWS = os.path.join(HERE, 'results_r3.jsonl'); OUT = os.path.join(HERE, 'verdict_r3.json')


def main():
    rows = [json.loads(l) for l in open(ROWS) if l.strip()]
    ok = [r for r in rows if 'error' not in r]
    gts = G.gates(rows); blocked = G.blocked_by(gts)
    lab, msg = (G.label(ok) if not blocked else (None, 'BLOCKED'))
    f = lambda k: [r[k] for r in ok if k in r]
    v = dict(batch='R3-GEOMETRY', n_rows=len(rows),
             gates=[[a, bool(b), c] for a, b, c in gts],
             gates_pass=f'{sum(1 for _, b, _ in gts if b)}/{len(gts)}',
             blocked_by=blocked, grade=('BLOCKED' if blocked else 'EXPLORATORY'),
             label=(f'R3-GEOMETRY :: {lab}' if lab else None), label_msg=msg,
             primary=dict(pose_ok=sum(1 for r in ok if r.get('pose_ok')), n=len(ok),
                          reproj_median=dict(min=min(f('reproj_median'), default=None),
                                             max=max(f('reproj_median'), default=None)),
                          rot_spread_max=max(f('rot_spread'), default=None)),
             secondary=dict(
                 cam_center=[r['cam_center'] for r in ok if 'cam_center' in r],
                 range_m=dict(min=min(f('range_min'), default=None),
                              max=max(f('range_max'), default=None)),
                 incidence_deg=dict(min=min(f('incidence_min'), default=None),
                                    max=max(f('incidence_max'), default=None),
                                    median=(st.median(f('incidence_median'))
                                            if f('incidence_median') else None))),
             disclosure=[
                 '⛔blind 아님 — 동결 전 탐색에서 재투영 ≈0.400 px·카메라 중심을 이미 보았다.',
                 '⛔CONFIRMATORY 청구 안 함 — 기능은 발견이 아니라 **기록**이다.',
                 '⛔정확도 주장 아님 — 추정기 런 0회.',
                 '⛔`B-17` 판정(REALNOISE-NULL) 갱신 아님 — 재실행은 별도 사전등록(R3b).',
                 '⛔full 판이 있는 레코드에만 적용(BS 계열). AA·YS 원본에는 iR/jR 이 없다.',
                 '⚠자세가 프레임마다 움직인다 — ⛔플랫폼 진동인지 WASS 평면 재추정인지 **지목 금지**.',
                 '★결정적 세부: `iR`/`jR` 에 **+1500 저장 오프셋**이 있다(없으면 DLT 가 깨진다).',
             ])
    json.dump(v, open(OUT, 'w'), ensure_ascii=False, indent=1)
    print(f"게이트 {v['gates_pass']} · 차단 {blocked}")
    for a, b, c in gts:
        if not b: print(f'  FAIL {a}  {c}')
    print(f"PRIMARY pose_ok {v['primary']['pose_ok']}/{v['primary']['n']}")
    print(f"라벨: {v['label']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
