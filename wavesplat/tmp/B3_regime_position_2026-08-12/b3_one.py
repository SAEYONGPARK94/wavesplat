#!/usr/bin/env python3
"""B-3 실측 레코드 1개를 **독립 프로세스**에서 잰다.

왜 프로세스를 나누는가: 한 프로세스에서 13 레코드를 연달아 열면 주소공간이 누적되어
(파일당 1.2~11 GB, 청크 = 프레임 1개) 뒤쪽 레코드가 16 MiB 할당에서 죽는다.
레코드마다 새 프로세스면 상한을 낮게 걸어도 안전하다.  통계·표본 규약은 b3_regime 그대로.

사용:  python3 b3_one.py <tier> <rel_path>   →  stdout 에 JSON 한 줄
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b3_regime import real_record                      # noqa: E402  ⛔재구현 금지

if __name__ == '__main__':
    tier, rel = sys.argv[1], sys.argv[2]
    try:
        a = real_record(rel, tier)
    except Exception as e:
        a = dict(error=f'{type(e).__name__}: {e}')
    print(json.dumps(a))
