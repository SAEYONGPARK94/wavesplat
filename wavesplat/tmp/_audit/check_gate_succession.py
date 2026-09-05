"""★형제 배치 사이에서 **차단 게이트가 조용히 사라지는 것**을 막는다.

## 왜 생겼나 (2026-09-05)
`T16` 은 `G-INNER` 를 **차단 게이트로 신설**했다(`gates_t16.py:553` 의 `GATES_BLOCKING`,
`PREREG.md` 에 5회). 그 등록문 자신이 *"**`G-EDGE` 도 `G-FLOOR` 도 이 방향을 보지 않는다**"*
라고 **미리** 적었다(`T16/PREREG.md:505`).
그 다음 배치 `T17` 은 `scale_rules_t16` 을 임포트해 **같은 측정면 규칙 위에서** 돌았는데
`G-INNER` 가 `gates_t17.py`.`analyze_t17.py`.`PREREG.md` 어디에도 **0회**다 —
논의된 흔적조차 없다. 동결 행에 다시 걸어 보면 `k̂₂ <= 3.5·k̂₁` 이 **2/72 셀**
(`BS13b|w2`.`BS13c|w5`; 팔 재라벨 좌표 `_prime` 로 재면 `BS13a|w5` 를 더해 **3/72**)에서
깨지고, 그 셀들은 **전부 `edge_hit = False`** 다 ⟹ 남은 `G-EDGE` 는 구조적으로 그 방향을
볼 수 없다. `T17` 은 라벨 없이(`label = null`) 닫혔다.

★부류: ***"형제 배치로 넘어갈 때 차단 게이트가 아무 기록 없이 증발한다."***
저장소가 이미 이름 붙인 인접 결함 셋의 **배치 간 판**이다 —
`E1`(게이트가 사라지면 원장에 적는다, 배치 **안**) · `check_gate_coverage`(사전등록↔구현) ·
`T5-ENS` 의 *"차단 게이트 14개가 전부 장식이 될 뻔했다"*.
셋 다 **한 배치 안**만 본다 ⟹ 배치 **사이**는 아무도 보지 않았다.

## 판정 규약 — ⛔무엇을 「형제」로 보는가 (이 검사기의 유일한 설계 선택)
휴리스틱(이름 접두어.날짜.제목 유사도)을 **쓰지 않는다**. 저장소가 배운 것은
*"조용히 잘못 묶는 분류기는 없느니만 못하다"* 이기 때문이다. 대신 **코드가 남긴 사실**을 쓴다:

    승계 간선 S → P  ⟺  (a) S 의 `.py` 중 하나가 문자열 상수로 **P 의 디렉터리 이름 전체**를
                            들고 있고(`os.path.join(HERE, '..', 'T16_BANDFIX_2026-09-05')`),
                        (b) 같은 파일 묶음이 **P 안에 실제로 있는 모듈**을 임포트한다.

둘 다 `ast` 로 읽는다. (a) 는 **문자열 상수 전체 일치**라 산문.독스트링.주석은 간선을 만들지
못하고, (b) 는 **모듈 파일이 P 안에 물리적으로 존재**해야 하므로 동명 모듈의 오해소도 막는다.
⟹ 간선은 **추정이 아니라 사실**이다: *S 는 P 의 판정.규칙 코드를 실제로 실행한다.*
그리고 간선은 전부 인쇄되므로, 잘못 묶였다면 **눈에 보인다**(조용하지 않다).

## 의무 — 무엇을 요구하는가
P 의 **차단 게이트** 이름 하나하나에 대해, S 의 산출물(`*.py`.`*.md`.`verdict*.json`)에
그 이름이 **한 번이라도 나타나야 한다**. 나타나면 통과다 — 승계했든, 이름을 바꿨든,
*"§6.2 `G-KTRAIN` 은 실측에 학습 코퍼스가 없어 버린다"* 라고 **버린 사유를 적었든** 상관없다.

⛔**이 검사기는 「부재」를 잡지 「약함」을 잡지 않는다**(`check_gate_coverage` 와 같은 자기한계).
지나가는 언급으로도 통과한다. 그것이 의도다 — 막으려는 것은 **약한 승계**가 아니라
***침묵***이고, 언급이 하나 있으면 사람이 따라갈 고리가 생긴다.

## 차단 게이트를 어디서 읽나 (두 관례, 둘 다 `ast` 리터럴)
  1. 모듈 수준 `GATES_BLOCKING = (...)`            — `T16`.`T17`.`T18`
  2. `def blocked_by(rows):` 안의 `block = (...)`  — `T11`.`T14`.`T15` 외 다수
⚠**둘 다 없는 배치는 판정 불가**다(선언이 없으므로 무엇이 차단이었는지 알 수 없다).
그 간선은 `⚠판정불가` 로 **인쇄하되 래칫하지 않는다** — 이 검사기의 이름 붙은 사각지대이고
숫자를 아래 출력이 매번 인쇄한다. ⛔"괜찮다" 로 읽지 마라.

## 사용
    python3 tmp/_audit/check_gate_succession.py [tmp 경로]   # 사람이 읽는 보고, rc=1 on 위반
    python3 tmp/_audit/check_gate_succession.py --keys        # 래칫용 `KEY\t…` (scripts/check_all.py)
    python3 tmp/_audit/check_gate_succession.py --selftest     # ★변이 시험 8종 (모의 tmp + 실자료 앵커)

런 0회 — 동결 파일만 읽는다.
"""
import ast
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TMP = os.path.abspath(os.path.join(HERE, '..'))

# ⛔토큰 경계 **양쪽** — `check_gate_coverage._NAME`.`check_prereg.GATE_RE` 가 배운 것과 같다.
#   오른쪽 가드가 없으면 `G-N` 이 `G-NOLEAK` 안에서 물려 **사라진 게이트를 "있다"** 고 한다
#   (⚠이 저장소는 「있는 것을 없다고」 하는 검사기와 「없는 것을 있다고」 하는 검사기를
#     **똑같이 나쁘다**고 친다 — 후자가 바로 이 방향이다).
#   왼쪽 가드가 없으면 라벨 안쪽(`RANGEANG-H30`)에서 물린다.
def _mention_re(name):
    return re.compile(r'(?<![0-9A-Za-z_\-])' + re.escape(name) + r'(?![0-9A-Za-z_\-])')


def batch_dirs(tmp):
    out = []
    for d in sorted(glob.glob(os.path.join(tmp, '*'))):
        if not os.path.isdir(d) or os.path.basename(d).startswith('_'):
            continue
        if judge_files(d):
            out.append(d)
    return out


def judge_files(d):
    """판정 코드 — `check_batch_artifacts._judge_files()` 와 같은 규약."""
    out = []
    for pat in ('gates_*.py', 'analyze_*.py', 'run_*.py'):
        out += sorted(glob.glob(os.path.join(d, pat)))
    return out


def blocking_gates(d):
    """(이름집합, 파일, 관례) 또는 (None, None, None).

    ⛔`GATES_BLOCKING` 을 먼저 본다 — 그것을 선언하는 배치는 `blocked_by` 안에
    파생 목록을 두므로, 뒤 관례를 먼저 보면 **같은 집합을 두 번** 읽거나 좁게 읽는다.
    """
    for p in sorted(glob.glob(os.path.join(d, 'gates_*.py'))):
        try:
            tree = ast.parse(open(p, encoding='utf-8').read())
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == 'GATES_BLOCKING'
                    for t in node.targets):
                got = _literal_names(node.value)
                if got:
                    return got, p, 'GATES_BLOCKING'
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'blocked_by':
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Assign) and any(
                            isinstance(t, ast.Name) and t.id in ('block', 'BLOCK')
                            for t in sub.targets):
                        got = _literal_names(sub.value)
                        if got:
                            return got, p, 'blocked_by.block'
    return None, None, None


def _literal_names(node):
    try:
        v = ast.literal_eval(node)
    except Exception:
        return None
    if not isinstance(v, (tuple, list, set)):
        return None
    names = {s for s in v if isinstance(s, str) and s.startswith('G-')}
    return names or None


def lineage(tmp, dirs):
    """{S: {P: [공유 모듈, …]}} — ⛔사실만. 이름.날짜 휴리스틱 0개."""
    names = {os.path.basename(d) for d in dirs}
    mod_home = {}                      # 모듈명 → {그 모듈 파일을 가진 배치}
    for d in dirs:
        for p in glob.glob(os.path.join(d, '*.py')):
            mod_home.setdefault(os.path.basename(p)[:-3], set()).add(os.path.basename(d))

    out = {}
    for d in dirs:
        s = os.path.basename(d)
        lits, imps = set(), set()
        for p in sorted(glob.glob(os.path.join(d, '*.py'))):
            try:
                tree = ast.parse(open(p, encoding='utf-8').read())
            except Exception:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                        and node.value in names:
                    lits.add(node.value)          # ★**전체 일치**만 — 산문은 못 만든다
                elif isinstance(node, ast.Import):
                    imps |= {a.name.split('.')[0] for a in node.names}
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imps.add(node.module.split('.')[0])
        for pb in sorted(lits):
            if pb == s:
                continue
            shared = sorted(m for m in imps if pb in mod_home.get(m, ()))
            if shared:
                out.setdefault(s, {})[pb] = shared
    return out


def artifacts_text(d):
    """S 의 산출물 전문 — `.py`.`.md`.`verdict*.json`.

    ⚠`results*.jsonl` 은 읽지 않는다(수 GB이고, 게이트 이름은 자료가 아니라 **문서**에 산다).
    """
    buf = []
    for pat in ('*.py', '*.md', 'verdict*.json'):
        for p in sorted(glob.glob(os.path.join(d, pat))):
            try:
                buf.append(open(p, encoding='utf-8', errors='replace').read())
            except OSError:
                pass
    return '\n'.join(buf)


def scan(tmp):
    """(keys, edges, unreadable, coverage) — 런 0회."""
    dirs = batch_dirs(tmp)
    edges = lineage(tmp, dirs)
    blk = {os.path.basename(d): blocking_gates(d) for d in dirs}
    keys, rows, unreadable = [], [], []
    for s in sorted(edges):
        text = artifacts_text(os.path.join(tmp, s))
        for pb in sorted(edges[s]):
            gates, gfile, how = blk.get(pb, (None, None, None))
            if not gates:
                unreadable.append((s, pb))
                continue
            silent = sorted(g for g in gates if not _mention_re(g).search(text))
            rows.append((s, pb, how, edges[s][pb], sorted(gates), silent))
            keys += [f'DROP:{s}|{pb}|{g}' for g in silent]
    cov = (len(rows), len(rows) + len(unreadable))
    return sorted(keys), rows, unreadable, cov


def main(argv):
    tmp = DEFAULT_TMP
    keys_only = '--keys' in argv
    args = [a for a in argv if not a.startswith('--')]
    if args:
        tmp = os.path.abspath(args[0])
    keys, rows, unreadable, cov = scan(tmp)
    if keys_only:
        for k in keys:
            print(f'KEY\t{k}')
        return 0
    print(f'── 형제 승계 (차단 게이트) — {tmp}')
    print(f'   승계 간선 {cov[1]}개 중 **대조 가능** {cov[0]}개 '
          f'(나머지 {len(unreadable)}개는 선행 배치가 차단 목록을 선언하지 않는다)')
    bad = 0
    for s, pb, how, mods, gates, silent in rows:
        head = f'   {s} ← {pb}  [{how}, 차단 {len(gates)}개, 공유 모듈 {mods}]'
        if not silent:
            print(head + '  ✅전부 언급됨')
            continue
        bad += len(silent)
        print(head)
        print(f'      ⛔조용히 사라진 차단 게이트 {len(silent)}: ' + ', '.join(silent))
    if unreadable:
        print(f'   ⚠판정 불가 {len(unreadable)}건 — 선행 배치가 `GATES_BLOCKING` 도 '
              f'`blocked_by.block` 도 선언하지 않는다(이 검사기의 **이름 붙인 사각지대**; '
              f'⛔"괜찮다" 가 아니다):')
        pre = sorted({p for _, p in unreadable})
        print('      ' + ', '.join(pre[:12]) + (f' … 외 {len(pre) - 12}배치' if len(pre) > 12 else ''))
    print(f'── 합계: 조용한 소실 {bad}건 / 키 {len(keys)}개')
    if bad:
        print('⛔수리 방법 = **후속 배치의 사전등록에 한 줄** 적는 것이다 '
              '("`G-…` 는 …이므로 승계하지 않는다"). ⛔동결 배치는 고치지 말고 '
              '`scripts/check_all.py` 의 `GATE_SUCCESSION_KNOWN` 에 사유와 함께 등록하라.')
    return 1 if bad else 0


# ───────────────────────── 자가진단 (★변이 시험) ─────────────────────────
_PRED_BLOCKING = '''\
GATES_BLOCKING = ('G-N', 'G-NOLEAK', 'G-KEEP', 'G-PROSE')
def blocked_by(rows):
    return []
'''
_PRED_BLOCKEDBY = '''\
def blocked_by(rows):
    block = ('G-N', 'G-NOLEAK', 'G-KEEP', 'G-PROSE')
    return list(block)
'''
_RULE = 'VALUE = 1\n'


def _mock(root, pred, succ, *, succ_import=True, succ_literal=True,
          mention_keep=True, mention_prose=True, mention_noleak=True,
          pred_body=_PRED_BLOCKING):
    """모의 `tmp/` — ⛔실제 배치는 한 글자도 건드리지 않는다."""
    pd, sd = os.path.join(root, pred), os.path.join(root, succ)
    os.makedirs(pd, exist_ok=True)
    os.makedirs(sd, exist_ok=True)
    open(os.path.join(pd, f'gates_{pred.lower()}.py'), 'w').write(pred_body)
    open(os.path.join(pd, 'rule_mod.py'), 'w').write(_RULE)
    body = ['import os, sys']
    if succ_literal:
        body.append(f"P = os.path.join(os.path.dirname(__file__), '..', '{pred}')")
    if succ_import:
        body.append('import rule_mod')
    if mention_keep:
        body.append("# G-KEEP 은 그대로 승계한다")
    if mention_noleak:
        body.append("GATES_BLOCKING = ('G-NOLEAK',)")
    open(os.path.join(sd, f'gates_{succ.lower()}.py'), 'w').write('\n'.join(body) + '\n')
    open(os.path.join(sd, 'PREREG.md'), 'w').write(
        '# PREREG\n' + ('`G-PROSE` 는 실측에 해당 없어 버린다.\n' if mention_prose else ''))
    return pd, sd


def selftest():
    import shutil
    import tempfile
    ok, n = True, 0

    def chk(label, cond, extra=''):
        nonlocal ok, n
        n += 1
        print(f'  [{"OK  " if cond else "FAIL"}] {label}' + (f'  {extra}' if extra else ''))
        ok = ok and bool(cond)

    root = tempfile.mkdtemp(prefix='gsucc_')
    try:
        # M1 기준 — 언급된 셋은 통과, 언급 없는 `G-N` 만 걸린다.
        _mock(root, 'PRED_2026-01-01', 'SUCC_2026-01-02')
        k, rows, unread, cov = scan(root)
        chk('M1 기준: 언급이 의무를 만족한다',
            k == ['DROP:SUCC_2026-01-02|PRED_2026-01-01|G-N'], f'{k}')
        # ★M1b 접두어 함정 — `G-N` 은 `G-NOLEAK` 안에서 물리면 안 된다(「없는 것을 있다고」).
        chk('M1b 접두어 함정: G-NOLEAK 이 G-N 을 가리지 않는다',
            'DROP:SUCC_2026-01-02|PRED_2026-01-01|G-NOLEAK' not in k
            and 'DROP:SUCC_2026-01-02|PRED_2026-01-01|G-N' in k)
        chk('M1c 산문(.md) 언급도 센다',
            'DROP:SUCC_2026-01-02|PRED_2026-01-01|G-PROSE' not in k)
        chk('M1d 주석 언급도 센다',
            'DROP:SUCC_2026-01-02|PRED_2026-01-01|G-KEEP' not in k)
        shutil.rmtree(root)

        # M2 변이 — 언급을 지우면 **새 키가 나타난다**(비공허성).
        root = tempfile.mkdtemp(prefix='gsucc_')
        _mock(root, 'PRED_2026-01-01', 'SUCC_2026-01-02',
              mention_keep=False, mention_prose=False)
        k2 = scan(root)[0]
        chk('M2 변이: 언급을 지우면 키가 3개로 늘어난다', len(k2) == 3, f'{k2}')
        shutil.rmtree(root)

        # M3 변이 — 임포트를 지우면 간선이 사라진다(디렉터리 이름만으로는 안 묶는다).
        root = tempfile.mkdtemp(prefix='gsucc_')
        _mock(root, 'PRED_2026-01-01', 'SUCC_2026-01-02', succ_import=False,
              mention_keep=False, mention_prose=False)
        k3, r3, u3, _ = scan(root)
        chk('M3 변이: 이름 상수만 있고 임포트가 없으면 간선 0', k3 == [] and r3 == [] and u3 == [])
        shutil.rmtree(root)

        # M4 변이 — 디렉터리 이름 상수를 지우면 간선이 사라진다(모듈명 우연 일치 방지).
        root = tempfile.mkdtemp(prefix='gsucc_')
        _mock(root, 'PRED_2026-01-01', 'SUCC_2026-01-02', succ_literal=False,
              mention_keep=False, mention_prose=False)
        k4, r4, u4, _ = scan(root)
        chk('M4 변이: 임포트만 있고 이름 상수가 없으면 간선 0', k4 == [] and r4 == [] and u4 == [])
        shutil.rmtree(root)

        # M5 — 옛 관례(`blocked_by` 안 `block` 튜플)도 읽는다.
        root = tempfile.mkdtemp(prefix='gsucc_')
        _mock(root, 'PRED_2026-01-01', 'SUCC_2026-01-02', mention_keep=False,
              mention_prose=False, pred_body=_PRED_BLOCKEDBY)
        k5, r5, _, _ = scan(root)
        chk('M5 옛 관례 blocked_by.block 추출', len(k5) == 3
            and r5 and r5[0][2] == 'blocked_by.block', f'{r5[0][2] if r5 else None}')
        shutil.rmtree(root)

        # M6 — 선행 배치가 차단 목록을 선언하지 않으면 **판정 불가**로 분리된다(래칫 오염 방지).
        root = tempfile.mkdtemp(prefix='gsucc_')
        _mock(root, 'PRED_2026-01-01', 'SUCC_2026-01-02', mention_keep=False,
              mention_prose=False, pred_body='X = 1\n')
        k6, r6, u6, _ = scan(root)
        chk('M6 선언 없는 선행 배치는 판정불가(키 0, 사각지대 1)',
            k6 == [] and r6 == [] and len(u6) == 1)
        shutil.rmtree(root)
        root = None

        # ★M7 — **실자료 앵커**. 이 검사기가 만들어진 사건 자체를 잡는가.
        real, rreal, _, _ = scan(DEFAULT_TMP)
        want = ('DROP:T17_REALSCALE_2026-09-05|T16_BANDFIX_2026-09-05|G-INNER')
        chk('M7 실자료 앵커: T16 → T17 의 G-INNER 소실을 잡는다', want in real)
        # ★M8 — **항진 아님**: 의무는 달성 가능하고 실제로 달성한 간선이 있다.
        #   ⛔전건 위반이면 그 의무는 자료가 아니라 **조문**을 재는 것이다(`B12` 부류).
        green = [(s, p) for s, p, _, _, _, sil in rreal if not sil]
        chk('M8 항진 아님: 실자료에서 위반 0인 승계 간선이 존재한다',
            len(green) >= 3, f'{len(green)}건 {green[:3]}')
        # ★M9 — **오분류 아님**: 이름.날짜가 붙어 있고 둘 다 `GATES_BLOCKING` 을 선언하는
        #   `T17`↔`T18` 은 **임포트 간선이 없다**(측정 질문이 다르다) ⟹ 키 0.
        #   휴리스틱 분류기였다면 여기서 조용히 잘못 묶였을 자리다.
        chk('M9 오분류 아님: 인접 이름·같은 날 T18 은 T17/T16 과 간선 0',
            not any(x.startswith('DROP:T18_FACTORIAL_2026-09-05|') for x in real))
    finally:
        if root and os.path.isdir(root):
            shutil.rmtree(root)
    print(f'  {"✅" if ok else "⛔"} 자가진단 {n}종')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(selftest() if '--selftest' in sys.argv else main(sys.argv[1:]))
