"""G-6 — 신규 게이트 2종이 **꺼져 있을 때 바이트 동일**함을 측정한다 (PREREG §7).

무엇을 재는가: S8 을 위해 `sequential_update.py` 에 두 개의 opt-in 게이트를 넣었다.
  `S8_L1_PHASE1_ONLY=1` — 추적 루프의 λ 를 출하 기본값에 고정 (update())
  `S8_L1_TRACK_ONLY=1`  — Phase-1 의 λ 를 출하 기본값에 고정 (initialize())
둘 다 미설정이면 아무것도 하지 않아야 한다. **논증이 아니라 측정으로 확인한다.**

방법: **S7 이 이미 채점한 (셀, 에피소드, 시드)** 를 지금 코드로 다시 돌려
`results.jsonl` 의 값과 비교한다. S7 은 이 편집 **이전에** 돌았으므로, 값이 12자리까지
같으면 편집이 기본 경로를 건드리지 않았다는 **직접 증거**다.
(S1 오류 #7 의 교훈: spawn 전파를 논증하다 측정으로 바꿨다.)

덤: 두 게이트를 **켠** 런도 한 번씩 돌려 (a) 에러 없이 돌고 (b) 기본과 **달라지는지**
확인한다 — 달라지지 않으면 게이트가 아무 일도 안 하는 것이므로 그것도 결함이다.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
S7 = os.path.abspath(os.path.join(HERE, '..', 'S7_confirm_n32_2026-08-10'))
sys.path.insert(0, S7)

import run_s7 as R7   # noqa: E402  S7 러너를 그대로 쓴다(복사 금지)

N_CHECK = int(os.environ.get('N_CHECK', '3'))
OUT = []


def p(s):
    print(s, flush=True)
    OUT.append(s)


def main():
    ref = {}
    for line in open(os.path.join(S7, 'results.jsonl')):
        r = json.loads(line)
        if 'error' not in r:
            ref[r['job_id']] = r

    # 혼잡·희소를 섞어 뽑되 결정론적으로 (⛔무작위 금지)
    picks = []
    for cell in ('AUTO', 'AUTO-L1-1e4', 'AUTO-NT'):
        for ep in (R7.CROWDED32[0], R7.SPARSE32[0]):
            jid = f'{cell}|{ep}|{R7.SEEDS[0]}'
            if jid in ref:
                picks.append((cell, ep, R7.SEEDS[0], jid))
    picks = picks[:N_CHECK]

    p('G-6 신규 게이트 결백 검사 — S7 기록값 대비 재현 (게이트 2종 모두 OFF)')
    ok = True
    for cell, ep, seed, jid in picks:
        for g in ('S8_L1_PHASE1_ONLY', 'S8_L1_TRACK_ONLY'):
            os.environ.pop(g, None)
        rec = R7.run_one({'job_id': jid, 'cell': cell, 'episode': ep, 'seed': seed})
        if 'error' in rec:
            p(f'  ⛔ {jid} 에러: {rec["error"][:160]}')
            ok = False
            continue
        a, b = ref[jid]['nRMSE'], rec['nRMSE']
        same = (a == b)
        ok &= same
        p(f'  {cell:14s} {"혼잡" if "crowded" in ep else "희소"}  S7 {a!r}')
        p(f'  {"":14s}      지금 {b!r}   {"✅ 비트 동일" if same else f"⛔ Δ={b - a:.3e}"}')

    # 게이트를 켜면 실제로 달라지는가 (켜진 셀이 무의미하지 않은지)
    p('\n게이트 ON 시 동작 확인 (달라져야 정상)')
    cell, ep, seed = 'AUTO-L1-1e4', R7.CROWDED32[0], R7.SEEDS[0]
    jid = f'{cell}|{ep}|{seed}'
    base = ref.get(jid, {}).get('nRMSE')
    for g in ('S8_L1_PHASE1_ONLY', 'S8_L1_TRACK_ONLY'):
        for g2 in ('S8_L1_PHASE1_ONLY', 'S8_L1_TRACK_ONLY'):
            os.environ.pop(g2, None)
        os.environ[g] = '1'
        rec = R7.run_one({'job_id': jid, 'cell': cell, 'episode': ep, 'seed': seed})
        os.environ.pop(g, None)
        if 'error' in rec:
            p(f'  ⛔ {g}: 에러 {rec["error"][:160]}')
            ok = False
            continue
        diff = rec['nRMSE'] - base if base is not None else float('nan')
        p(f'  {g:22s} nRMSE {rec["nRMSE"]:.9f}  (기본 {base:.9f}, Δ={diff:+.3e})'
          f'  {"✅ 발화" if diff != 0 else "⛔ 아무 일도 안 함"}')
        ok &= (diff != 0)

    p(f'\nG-6: {"PASS ✅" if ok else "FAIL ⛔"}')
    open(os.path.join(HERE, 'bitcheck.txt'), 'w').write('\n'.join(OUT) + '\n')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
