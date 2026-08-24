"""B-14b — `B-14` 축 A **재실행** (난수 소비 결함 수정 후).

⛔`B-14` 축 A 는 무효다: `noise=0` 에서 `rng.normal` 을 건너뛰어 **잡음 수준마다 표본이
달라졌다** ⟹ *"잡음만 바꾼 개입"* 이 아니었다. `G-1` 양성대조가 그것을 잡았다.
★수정 후 확인: 같은 레코드에서 σ 를 바꿔도 `P` 가 **비트 동일**하다.

⚠**축 B(밀도)는 재실행하지 않는다** — 전부 σ=0 이고 밀도마다 표본이 다른 것이 설계이므로
그 결함의 영향을 받지 않는다(`B-14` 의 H2 판정은 유효하다).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_b14 as R                                        # noqa: E402
import gates_b14 as G                                      # noqa: E402

OUT = os.path.join(HERE, 'results_b14b.jsonl')


def main():
    shard = int(sys.argv[1]) if len(sys.argv) > 2 else 0
    nsh = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    out = OUT.replace('.jsonl', f'_{shard}.jsonl')
    done = set()
    if os.path.exists(out):
        for l in open(out):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            if 'error' not in r:
                done.add((r['record'], r['obs_model'], r['noise']))
    cells = [(rec, mdl, nz) for rec in G.RECORDS for mdl in G.OBS_MODELS
             for nz in G.NOISE_LEVELS]
    todo = [c for i, c in enumerate(cells) if i % nsh == shard and c not in done]
    print(f'B-14b shard {shard}/{nsh}: {len(todo)} 셀', flush=True)
    with open(out, 'a') as fh:
        for i, (rec, mdl, nz) in enumerate(todo, 1):
            try:
                row = R._one(rec, mdl, nz, G.M_SCAN_A, 'A')
            except Exception as e:                          # noqa: BLE001
                row = dict(axis='A', record=rec, obs_model=mdl, noise=nz,
                           m_scan=G.M_SCAN_A, error=f'{type(e).__name__}: {e}')
            fh.write(json.dumps(row, ensure_ascii=False) + '\n')
            fh.flush()
            print(f"[{i}/{len(todo)}] {rec:6s} {mdl:5s} σ={nz:<5} "
                  f"{row.get('error') or format(row['nowcast'], '.4f')}", flush=True)


if __name__ == '__main__':
    main()
