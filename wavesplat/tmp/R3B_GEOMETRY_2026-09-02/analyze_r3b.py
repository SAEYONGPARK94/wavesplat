"""`R3b-GEOMETRY` 분석기 — ⛔런 0회(추정기). 규칙은 `gates_r3b.py` 단일출처."""
import json, os, statistics as st, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_r3b as G                                               # noqa: E402
ROWS = os.path.join(HERE, 'results_r3b.jsonl'); OUT = os.path.join(HERE, 'verdict_r3b.json')


def rng(vals):
    v = [x for x in vals if x is not None]
    return dict(min=min(v), median=st.median(v), max=max(v)) if v else None


def main():
    rows = [json.loads(l) for l in open(ROWS) if l.strip()]
    ok = [r for r in rows if 'error' not in r]
    gts = G.gates(rows); blocked = G.blocked_by(gts)
    lab, msg = (G.label(ok) if not blocked else (None, 'BLOCKED'))
    pose = [r for r in ok if r.get('pose_ok')]

    v = dict(batch='R3b-GEOMETRY', n_rows=len(rows),
             gates=[[a, bool(b), c] for a, b, c in gts],
             gates_pass=f'{sum(1 for _, b, _ in gts if b)}/{len(gts)}',
             blocked_by=blocked, grade=('BLOCKED' if blocked else 'EXPLORATORY'),
             label=(f'R3b-GEOMETRY :: {lab}' if lab else None), label_msg=msg,
             primary=dict(pose_ok=len(pose), n=G.N,
                          records_ok=[r['record'] for r in pose],
                          records_fail=[r['record'] for r in ok if not r.get('pose_ok')],
                          reproj_median_px=rng([r.get('reproj_median') for r in ok]),
                          rot_spread=rng([r.get('rot_spread') for r in ok]),
                          radial_flat_px=rng([r.get('radial_flat') for r in ok])),
             channels=dict(iR_change=rng([r.get('iR_change') for r in ok]),
                           jR_change=rng([r.get('jR_change') for r in ok]),
                           note='★선행 R3 은 iR 만 재고 jR 의 실측치(0.786)로 문턱을 걸었다'),
             secondary=dict(
                 cam_center=[[r['record']] + [round(x, 3) for x in r.get('cam_center', [])]
                             for r in ok],
                 range_m=rng([r.get('range_min') for r in ok] + [r.get('range_max') for r in ok]),
                 incidence_deg=rng([r.get('incidence_min') for r in ok]
                                   + [r.get('incidence_max') for r in ok]),
                 declared=dict(SENSOR_H=G.SENSOR_H_DECLARED, SENSOR_XY=[0.0, 0.0],
                               source='tmp/B17_REALNOISE_2026-08-25/gates_b17.py:149,158')),
             disclosure=[
                 '⛔blind 아님 — 선행 탐색에서 reproj≈0.395 px, 중심≈(0.016, 19.77, 12.297) 를 보았다.',
                 '⛔CONFIRMATORY 아님 — 기능은 발견이 아니라 **동결·재현 가능한 기록**이다.',
                 '⛔정확도 주장 아님 — 추정기를 돌리지 않는다.',
                 '⛔B-17 판정(REALNOISE-NULL)의 갱신 아님 — 그 재실행은 별도 사전등록이다.',
                 '⛔카메라 움직임의 기제 지목 금지(플랫폼 진동 / WASS 프레임별 평면 재추정).',
                 '⛔사이트 일반화 금지 — BS 1 사이트 7 레코드.',
                 '⚠화소 오프셋 +1500 은 러너 상수이고 행에 기록된다(⛔조용히 쓰지 않는다).',
             ])
    json.dump(v, open(OUT, 'w'), ensure_ascii=False, indent=1)
    print(f"게이트 {v['gates_pass']} · 차단 {blocked}")
    for a, b, c in gts:
        if not b: print(f'  FAIL {a}  {c}')
    print(f"PRIMARY pose_ok {len(pose)}/{G.N} · reproj {v['primary']['reproj_median_px']}")
    print(f"채널 변화 iR {v['channels']['iR_change']} · jR {v['channels']['jR_change']}")
    print(f"라벨: {v['label']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
