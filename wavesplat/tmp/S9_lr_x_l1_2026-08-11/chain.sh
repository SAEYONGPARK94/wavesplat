#!/usr/bin/env bash
# S9 실행 체인 — 게이트가 PASS 해야만 런이 돈다. ⛔게이트 FAIL 시 채점하지 않는다.
#   1) 동결 md5 재검증 (사전등록 산출물이 런 직전에 바뀌지 않았는가)
#   2) gate.py  (G-0~G-7)
#   3) run_s9.py
#   4) analyze_s9.py
set -u
cd "$(dirname "$0")" || exit 9
export WORKERS=${WORKERS:-8}
unset S2_ALPHA                      # ★제어 축 종료 (PREREG G-7)

echo "=== 1) 동결 md5 재검증 ==="
if ! md5sum -c FROZEN_md5.txt; then
  echo "⛔ 동결 파일이 바뀌었다 — 중단"; exit 1
fi

echo; echo "=== 2) 실행 전 게이트 ==="
python3 gate.py 2>&1 | tee gate.txt
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  echo "⛔ GATE FAIL — S9 를 돌리지 않는다"; exit 1
fi

echo; echo "=== 3) 배치 (2016 런) ==="
python3 run_s9.py 2>&1 | tee -a run.log
rc=$?
if [ "$rc" -ne 0 ]; then echo "⛔ 런 실패 rc=$rc"; exit "$rc"; fi

echo; echo "=== 4) 채점 ==="
python3 analyze_s9.py 2>&1 | tee analysis.txt

echo; echo "=== 5) 런 후 코드 출처 재확인 (G-5) ==="
python3 - <<'PY'
import hashlib, json, subprocess, os
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath('.')), '..', '..'))
prev = json.load(open('G5_code_provenance.json'))
import gate as G
diff = subprocess.run(['git', 'diff'] + G.CODE_FILES, cwd=G.REPO,
                      capture_output=True, text=True).stdout
now = hashlib.md5(diff.encode()).hexdigest()
print(f"  런 전 {prev['diff_md5']}  런 후 {now}  "
      f"{'✅ 동일' if now == prev['diff_md5'] else '⛔ 코드가 런 도중 바뀌었다'}")
PY
echo "체인 완료"
