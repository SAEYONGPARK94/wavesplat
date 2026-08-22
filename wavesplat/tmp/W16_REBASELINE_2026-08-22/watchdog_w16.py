"""W15 감시자 — 러너와 **분리**된 프로세스. 저장소 상시지시(감시/실행 분리) 준수.

러너를 신뢰하지 않는다: 결과 파일만 읽고, **러너가 스스로 보고하지 않는 것**을 본다.
  · 진행 정지(stall)
  · 게이트가 사후에 깨지는지 (러너는 끝에서 한 번만 본다)
  · ★러너가 인쇄하지 않는 **경로 이탈** — POST 팔이 learned 가 아닌 런
  · ★자기매칭 방지: 자기 자신을 프로세스 목록에서 세지 않는다 (W3 에서 8시간 잃은 결함)
"""
import json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gates_w16 as G  # noqa: E402

OUT = os.path.join(HERE, 'results.jsonl')
LOG = os.path.join(HERE, 'watchdog.log')
EXPECT = len(G.ARMS) * len(G.EPISODES) * len(G.SEEDS)
STALL_S = int(os.environ.get('W15_STALL_S', '900'))


def runner_alive():
    # ★자기 자신을 세지 않는다 — 이름이 겹치면 감시자가 감시자를 보고 살아있다고 판단한다
    me = str(os.getpid())
    try:
        out = subprocess.run(['pgrep', '-af', 'run_w16.py'], capture_output=True,
                             text=True, timeout=10).stdout
    except Exception:
        return True
    return any(l.split()[0] not in (me,) and 'watchdog' not in l
               for l in out.strip().splitlines() if l.strip())


def log(msg):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')


def main():
    last_n, last_change = -1, time.time()
    log(f'감시 시작 · 기대 {EXPECT}런 · stall 문턱 {STALL_S}s')
    while True:
        n = sum(1 for _ in open(OUT)) if os.path.exists(OUT) else 0
        rows = [json.loads(l) for l in open(OUT)] if n else []
        err = [r for r in rows if 'error' in r]
        post = [r for r in rows if r.get('cell') == 'POST' and 'error' not in r]
        stray = [r['job_id'] for r in post if r.get('auto_selected') != 'learned']
        if n != last_n:
            last_n, last_change = n, time.time()
            pct = 100 * n / EXPECT
            log(f'{n}/{EXPECT} ({pct:.0f}%) · 에러 {len(err)} · POST 경로이탈 {len(stray)}')
            if err:
                log(f'  ⛔에러 발생: {err[-1].get("error","")[:120]}')
            if stray:
                log(f'  ⛔★POST 팔이 learned 가 아니다 = 통일이 안 먹었다: {stray[:3]}')
        elif time.time() - last_change > STALL_S:
            log(f'⛔정지 감지: {STALL_S}s 동안 진행 없음 ({n}/{EXPECT})')
            last_change = time.time()
        if n >= EXPECT:
            log('완료 — 게이트 독립 재도출:')
            for gid, ok, msg in G.gates(rows):
                log(f'  [{"PASS" if ok else "FAIL"}] {gid}: {msg[:100]}')
            return
        if not runner_alive() and n < EXPECT:
            log(f'⛔러너가 죽었다 ({n}/{EXPECT})')
            return
        time.sleep(30)


if __name__ == '__main__':
    main()
