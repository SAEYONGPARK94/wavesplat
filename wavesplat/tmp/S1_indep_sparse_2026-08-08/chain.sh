#!/usr/bin/env bash
# S1 실행 체인: 생성 완료 대기 -> 게이트 -> (PASS 일 때만) 768 런
# 게이트가 FAIL(exit 1) 이면 본 배치를 돌리지 않는다 (PREREG §4).
set -u
cd /home/vexit/wavesplat/wavesplat
D=tmp/S1_indep_sparse_2026-08-08

# 프로세스 기반 대기는 생성기가 죽어도 통과해 버린다. **파일 개수**로 기다린다.
echo "[$(date +%H:%M:%S)] 생성 완료 대기 (파일 32개 기준, 최대 3시간)..."
for _ in $(seq 1 360); do
  NS=$(ls data_indep_sparse_2026-08-08/*.h5 2>/dev/null | wc -l)
  NC=$(ls data_indep_crowded_2026-08-08/*.h5 2>/dev/null | wc -l)
  [ "$NS" -eq 16 ] && [ "$NC" -eq 16 ] && break
  sleep 30
done
# 마지막 파일이 완전히 닫히도록 생성 프로세스 종료까지 확인
while pgrep -f "gen_sparse.py|gen_crowded.py" >/dev/null; do sleep 20; done

NS=$(ls data_indep_sparse_2026-08-08/*.h5 2>/dev/null | wc -l)
NC=$(ls data_indep_crowded_2026-08-08/*.h5 2>/dev/null | wc -l)
echo "[$(date +%H:%M:%S)] 생성 종료: 희소 $NS/16, 혼잡 $NC/16"
if [ "$NS" -ne 16 ] || [ "$NC" -ne 16 ]; then
  echo "⛔ 에피소드 부족 — 중단"; exit 1
fi

echo "[$(date +%H:%M:%S)] 게이트 실행 (PREREG §4)"
python3 $D/gate.py > $D/gate.txt 2>&1
GATE=$?
tail -20 $D/gate.txt
if [ $GATE -ne 0 ]; then
  echo "⛔ GATE FAIL — S1 을 돌리지 않는다 (PREREG §4)"; exit 1
fi

echo "[$(date +%H:%M:%S)] GATE PASS — 768 런 시작"
WORKERS=8 python3 $D/run_s1.py 2>&1 | tail -40
echo "[$(date +%H:%M:%S)] 런 종료"
