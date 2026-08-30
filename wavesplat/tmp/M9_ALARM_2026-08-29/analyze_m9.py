#!/usr/bin/env python3
"""M-9 판정 조립 — 산출물(rg1/rg0 pre/rg0 post/mutation)을 읽어 verdict_m9.json 을 쓴다.

라벨 규칙은 `gates_m9.label_final`(단일출처·결과 전 동결)이 전부다. 산출물이 없으면
그 절은 **None** 으로 기록한다(0 이 아니다 — 커버리지 부재를 통과로 세지 않는다).
런 0회 — 파일만 읽는다.

    python3 tmp/M9_ALARM_2026-08-29/analyze_m9.py
"""
import os

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'

import hashlib    # noqa: E402
import json       # noqa: E402
import sys        # noqa: E402
import time       # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gates_m9 as G                                       # noqa: E402


def _load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def main():
    rg1 = _load(G.RG1_REPORT)
    rg0_pre = _load(G.RG0_REPORT['pre'])
    rg0_post = _load(G.RG0_REPORT['post'])
    mut = _load(G.MUT_REPORT)
    label, why = G.label_final(rg1, rg0_pre, rg0_post, mut)

    def _gate_summary(rep):
        if rep is None:
            return None                                    # 부재 = None (0 아님)
        return {x['name']: x['ok'] for x in rep['gates']}

    verdict = dict(
        batch='M9_ALARM_2026-08-29',
        label=label, reason=why,
        ts=time.strftime('%Y-%m-%d %H:%M:%S'),
        rg1=_gate_summary(rg1),
        rg1_mode=(rg1 or {}).get('mode'),
        rg1_derived=(rg1 or {}).get('derived'),
        rg0_pre=_gate_summary(rg0_pre),
        rg0_post=_gate_summary(rg0_post),
        mutation=(None if mut is None else
                  dict(all_caught=mut.get('all_caught'),
                       baseline_pass=mut.get('baseline_pass'),
                       n_mutants=len(mut.get('mutants', {})))),
        md5_jsonl=(rg1 or {}).get('md5_jsonl'),
        md5_sequential_update_now=hashlib.md5(
            open(G.SEQ_UPD_PATH, 'rb').read()).hexdigest(),
        md5_sequential_update_pre_pin=G.SEQ_UPD_MD5_PRE,
        expected=G.EXPECTED,
    )
    json.dump(verdict, open(G.VERDICT, 'w'), ensure_ascii=False, indent=1)
    print(f'판정: {label}\n사유: {why}\n→ {G.VERDICT}')
    ok = label.startswith(('ALARM-FIXED (CONFIRMATORY)', 'HARNESS-READY'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
