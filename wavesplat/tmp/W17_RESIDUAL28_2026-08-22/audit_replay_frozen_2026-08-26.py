"""W17 동결 감사 재현 (2026-08-26) — ★**읽기 전용. 저장소 파일을 쓰지 않는다.**

물음: `gates_w17.py` 가 동결 뒤에 움직였다(★자기동결 위반). **판정이 움직였는가?**
방법: 동결 판본(`git show e64eea1:…/gates_w17.py`, md5 `3dca297…`)을 **그대로** 꺼내
      저장된 `results.jsonl` + 차용 팔에 돌리고, 현행 `verdict_w17.json` 과 키 단위로 댄다.

⚠**주의**: 동결판 `ARMS` 는 세 팔뿐이라 `gates()` 의 `new` 필터가 넷째 팔 16행을
자동으로 배제한다 ⟹ `G-0` 는 `48/48` 로 **PASS** 한다(현판은 `64/64`). 통과 여부는 같다.

실행:  python3 audit_replay_frozen_2026-08-26.py
"""
import hashlib, importlib.util, json, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN_COMMIT, FROZEN_MD5 = 'e64eea1', '3dca297156532845e03220fe38c55bd6'
REL = 'wavesplat/tmp/W17_RESIDUAL28_2026-08-22/gates_w17.py'
REPO = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'],
                               cwd=HERE, text=True).strip()

blob = subprocess.check_output(['git', 'show', f'{FROZEN_COMMIT}:{REL}'], cwd=REPO)
assert hashlib.md5(blob).hexdigest() == FROZEN_MD5, 'FROZEN_md5.txt 와 안 맞는다'

tmpd = tempfile.mkdtemp()
open(os.path.join(tmpd, 'gates_w17.py'), 'wb').write(blob)
spec = importlib.util.spec_from_file_location('gates_w17_frozen',
                                              os.path.join(tmpd, 'gates_w17.py'))
G = importlib.util.module_from_spec(spec); spec.loader.exec_module(G)
assert list(G.ARMS) == ['SI-NOSM-STEPS10', 'SI-NOSM-LR10X', 'SI-NOSM-NOCLAMP'], list(G.ARMS)

rows = [json.loads(l) for l in open(os.path.join(HERE, 'results.jsonl'))]
for arm, d in G.BORROWED.items():          # run_w17.py:151-154 과 같은 병합
    f = os.path.abspath(os.path.join(HERE, d, 'results.jsonl'))
    rows += [r for r in (json.loads(l) for l in open(f)) if r.get('cell') == arm]

gates = G.gates(rows)
ok = [r for r in rows if 'error' not in r]
res = {a: G.ratio_vs_ref(ok, a) for a in G.ARMS}
base = G.ratio_vs_ref(ok, 'ORA-WPHI-SI-NOSM')['ratio']
lab = G.label(res, base)
cur = json.load(open(os.path.join(HERE, 'verdict_w17.json')))

bad = []
for n_, p_, d_ in gates:
    print(f'  [{"PASS" if p_ else "FAIL"}] {n_}: {d_}')
    if not p_:
        bad.append(n_)
print(f'\n동결판 라벨  : {lab[0]}\n현행 verdict : {cur["label"]}')
assert lab[0] == cur['label'], (lab[0], cur['label'])
assert abs(base - cur['base']) < 1e-15
for a in G.ARMS:                            # 사전등록 세 팔은 비트 동일이어야 한다
    for k in ('n', 'above', 'ratio', 'p_one_sided'):
        assert res[a][k] == cur['arms'][a][k], (a, k, res[a][k], cur['arms'][a][k])
    assert abs(G.share(res[a]['ratio'], base) - cur['share'][a]) < 1e-15
assert not bad, bad
print('\n✅사전등록 세 팔 · base · share · 게이트 통과여부 · 라벨 = **전부 동일**.')
print('⚠차이는 셋뿐: 사후 팔 항목 추가 · G-0/G-1 상세 문자열(48/48 → 64/64) · `_amended`.')
