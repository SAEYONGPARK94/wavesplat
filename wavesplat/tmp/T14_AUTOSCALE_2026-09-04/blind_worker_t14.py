"""★`G-S-BLIND` 의 **별도 프로세스** — 관측 5배열만 받아 규칙을 재실행한다.
⛔표적·메타는 이 프로세스에 **존재하지 않는다**. 그것이 이 검사의 하중이다."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                '..', '..')))
import scale_rules_t14 as SR

d = np.load(sys.argv[1])
out = {k: float(f(d['x'], d['y'], d['t'], d['z'], float(d['depth'])))
       for k, f in SR.RULES.items()}
print(json.dumps(out))
