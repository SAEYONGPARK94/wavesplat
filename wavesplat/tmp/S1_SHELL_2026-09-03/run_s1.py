"""`S1-SHELL` 러너 — ⛔**신규 런 0회**. `S_ENS` 동결 행을 셸로 재집계한다."""
import glob, hashlib, json, os, re, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_s1 as G                                                 # noqa: E402
OUT = os.path.join(HERE, 'results_s1.jsonl'); META = os.path.join(HERE, 'meta_s1.json')
FROZEN = os.path.join(HERE, 'FROZEN_md5.txt')
SRC = sorted(glob.glob(os.path.join(G.SENS, 'results*.jsonl')))


def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def check_frozen():
    if not os.path.exists(FROZEN):
        return False, 'FROZEN_md5.txt 없음'
    want = {}
    for line in open(FROZEN, encoding='utf-8'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        h, _, rel = line.partition('  ')
        want[rel.strip()] = h.strip()
    bad = []
    for p in SRC:
        rel = os.path.relpath(p, os.path.join(HERE, '..', '..'))
        if rel not in want:
            bad.append(f'미동결:{rel}')
        elif want[rel] != md5(p):
            bad.append(f'불일치:{rel}')
    return (not bad), ('; '.join(bad) if bad else f'{len(SRC)}/{len(SRC)} 일치')


def rms_group(shell, names):
    """셸 결합 = **동일 가중 RMS 합성**(PREREG §1). ⛔단순 평균 아님."""
    v = [shell[s] for s in names if shell.get(s) is not None]
    return float(np.sqrt(np.mean(np.square(v)))) if v else float('nan')


def norun_check():
    pat = re.compile(r'^\s*(from|import)\s+wavesplat\b', re.M)
    hits = [fn for fn in ('run_s1.py', 'analyze_s1.py', 'gates_s1.py')
            if pat.search(open(os.path.join(HERE, fn), encoding='utf-8').read())]
    return (not hits), ('; '.join(hits) if hits else '추정기 임포트 0')


def main():
    rows = [json.loads(l) for p in SRC for l in open(p, encoding='utf-8') if l.strip()]
    by = {}
    for r in rows:
        by.setdefault((r['episode'], r['stratum']), {})[r['arm']] = r
    out = []
    for (ep, st), d in sorted(by.items()):
        b, e = d.get(G.ARM_BASE), d.get(G.ARM_ENS)
        if b is None or e is None:
            out.append(dict(episode=ep, stratum=st, error='짝 없음')); continue
        sb, se = b['avg_shell_rmse'], e['avg_shell_rmse']
        out.append(dict(
            episode=ep, stratum=st, seed=b.get('seed'),
            glob_base=b['avg_nRMSE'], glob_ens=e['avg_nRMSE'],
            nearfield_base=b['avg_RMSE_near'], nearfield_ens=e['avg_RMSE_near'],
            near_base=rms_group(sb, G.NEAR), near_ens=rms_group(se, G.NEAR),
            far_base=rms_group(sb, G.FAR), far_ens=rms_group(se, G.FAR),
            shell_base={s: sb.get(s) for s in G.SHELLS},
            shell_ens={s: se.get(s) for s in G.SHELLS},
            k_base=b.get('ensemble_k'), k_ens=e.get('ensemble_k')))
    with open(OUT, 'w', encoding='utf-8') as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    fok, fmsg = check_frozen(); nok, nmsg = norun_check()
    json.dump(dict(frozen_ok=fok, frozen_msg=fmsg, norun=nok, norun_msg=nmsg,
                   thr=G.min_w(G.N_EPISODES), n_pairs=len(out)),
              open(META, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'  짝 {len(out)} · FROZEN {fmsg} · {nmsg}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
