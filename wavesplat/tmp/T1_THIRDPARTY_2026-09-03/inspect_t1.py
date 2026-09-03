"""`T1` 1단계 — **자료 무결성과 구조만** 본다. ⛔통계 설계와 무관한 사실만 기록한다.

왜 분리했나: 통계 설계(귀무·문턱)는 지금 적대적 감사를 받고 있다. 그 결과가 나오기 전에
**설계와 무관한 사실**(크기 일치 · 아카이브 구조 · shape · 메타의 solver 명시 여부)은
먼저 확정해 둘 수 있다. ⛔**PRIMARY 통계는 여기서 계산하지 않는다.**
"""
import hashlib, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
ZIP = os.path.abspath(os.path.join(ROOT, '..', 'data',
                                   'thirdparty_synth_zenodo17359600', 'Data.zip'))
WANT = 12828148812          # ⛔Content-Length, 결과 전 고정(NOTE.md)
OUT = os.path.join(HERE, 'inspect_t1.json')


def main():
    rep = {'zip_path': ZIP}
    if not os.path.exists(ZIP):
        rep['error'] = 'zip 없음'
        json.dump(rep, open(OUT, 'w'), ensure_ascii=False, indent=1)
        print('⛔zip 없음'); return 1
    size = os.path.getsize(ZIP)
    rep['size'] = size
    rep['size_matches_content_length'] = (size == WANT)   # ★G-SIZE
    print(f"크기 {size} / {WANT}  일치={rep['size_matches_content_length']}")
    if size != WANT:
        rep['error'] = '크기 불일치 — ⛔G-SIZE 실패이므로 여기서 멈춘다'
        json.dump(rep, open(OUT, 'w'), ensure_ascii=False, indent=1)
        print('⛔크기 불일치 — 멈춘다'); return 1

    import zipfile
    with zipfile.ZipFile(ZIP) as z:
        names = z.namelist()
        rep['n_entries'] = len(names)
        rep['sample_entries'] = names[:20]
        mats = [n for n in names if n.lower().endswith('.mat')]
        docs = [n for n in names if os.path.splitext(n)[1].lower()
                in ('.txt', '.md', '.pdf', '.json', '.yml', '.yaml', '')]
        rep['n_mat'] = len(mats)
        rep['doc_entries'] = docs[:20]
        print(f'엔트리 {len(names)} · .mat {len(mats)} · 문서류 {len(docs)}')

        # ★G-SOLVER — 메타/README 가 생성 solver 를 명시하는가
        solver_hits = []
        for d in docs:
            if d.endswith('/'):
                continue
            try:
                txt = z.read(d)[:200000].decode('utf-8', 'replace')
            except Exception:                                       # noqa: BLE001
                continue
            for kw in ('HOS', 'higher-order spectral', 'boundary element',
                       'potential flow', 'solver', 'Stokes', 'JONSWAP', 'linear'):
                if kw.lower() in txt.lower():
                    solver_hits.append({'file': d, 'kw': kw})
        rep['solver_mentions'] = solver_hits[:40]
        rep['solver_named'] = bool(solver_hits)

        # 첫 .mat 하나만 구조 확인 (⛔통계 아님)
        if mats:
            m0 = sorted(mats)[0]
            rep['first_mat'] = m0
            raw = z.read(m0)
            rep['first_mat_bytes'] = len(raw)
            rep['first_mat_md5'] = hashlib.md5(raw).hexdigest()[:12]
            import io as _io
            from scipy.io import loadmat
            try:
                d = loadmat(_io.BytesIO(raw))
                rep['first_mat_keys'] = {k: list(np.asarray(v).shape)
                                         for k, v in d.items() if not k.startswith('__')}
            except Exception as e:                                  # noqa: BLE001
                try:
                    import h5py
                    with h5py.File(_io.BytesIO(raw), 'r') as f:
                        rep['first_mat_keys'] = {k: list(f[k].shape) for k in f}
                        rep['first_mat_fmt'] = 'v7.3(hdf5)'
                except Exception as e2:                             # noqa: BLE001
                    rep['first_mat_error'] = f'{type(e).__name__}/{type(e2).__name__}'
            print('첫 .mat', m0, rep.get('first_mat_keys'))
    json.dump(rep, open(OUT, 'w'), ensure_ascii=False, indent=1)
    print('✅ inspect_t1.json 기록')
    return 0


if __name__ == '__main__':
    sys.exit(main())
