"""B-17 — **개입이 실제로 무엇을 하는지**를 추정기 런 0회로 미리 낸다 (순수 기하).

⚠**결과 노출이 아니다** — 여기서 나오는 것은 *관측 잡음장* 뿐이고 추정기 성능은 없다.
그러나 사전등록이 *"팔이 무엇을 하는지"* 를 결과 전에 적어야 하므로 이 표를 PREREG 에 싣는다.
★특히 **논문 곡선을 실측 기하에 얹으면 층마다 잡음 수준이 반대로 움직인다**는 것을
결과 전에 드러낸다 — 그것이 `RANGEANG-MM` · `CONST-010` 통제 팔의 존재 이유다.

    python3 tmp/B17_REALNOISE_2026-08-25/probe_sigma.py
"""
import json
import os
import sys
import warnings

warnings.filterwarnings('ignore')
import numpy as np                                        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(os.path.abspath(os.path.join(HERE, '..', '..')))
import gates_b17 as G                                     # noqa: E402
from noise_models import scene_noise                      # noqa: E402

OUT = os.path.join(HERE, 'sigma_probe.json')


def main():
    res = {}
    print(f"{'record':7s} {'stratum':6s} {'arm':13s} "
          f"{'r_med':>8s} {'cosθ_med':>9s} {'clamp':>6s} "
          f"{'σ_min':>7s} {'σ_med':>7s} {'σ_max':>7s} {'σ_rms':>7s} {'σ_rms/σ_surf':>12s}")
    for rec in G.RECORDS:
        st = 'LARGE' if rec in G.LARGE_BOX else 'SMALL'
        for arm in G.ARM_ORDER:
            sc = scene_noise(rec, G.ARMS[arm], win=G.WIN, sigma0=G.SIGMA0,
                             sigma_r=G.SIGMA_R, d_ref=G.D_REF, cos_min=G.COS_MIN,
                             mm_target=G.MM_TARGET)
            s = sc['sigma_stats']
            res.setdefault(rec, {})[arm] = dict(
                stratum=st, sg=sc['sg'], n_obs=sc['n_obs_used'], **s)
            f = lambda k: ('    n/a' if s.get(k) is None else f'{s[k]:7.3f}')  # noqa: E731
            print(f"{rec:7s} {st:6s} {arm:13s} {f('r_med'):>8s} "
                  f"{('      n/a' if s.get('cos_med') is None else format(s['cos_med'], '9.3f'))} "
                  f"{s['frac_cos_clamped'] * 100:5.1f}% "
                  f"{s['min']:7.4f} {s['med']:7.4f} {s['max']:7.4f} {s['rms']:7.4f} "
                  f"{s['rms'] / sc['sg']:11.3f}", flush=True)
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1)

    print('\n## 층별 σ_rms 중앙 (배포 CONST=0.0200 대비 배수)')
    for arm in G.ARM_ORDER:
        row = []
        for st, sub in (('대형', G.LARGE_BOX), ('소형', G.SMALL_BOX)):
            v = [res[r][arm]['rms'] for r in sub]
            row.append(f'{st} {np.median(v):.4f} ({np.median(v) / G.SIGMA_DEPLOY:5.2f}×)')
        print(f'  {arm:13s} ' + '   ·   '.join(row))
    print(f'\n★ sigma_probe.json 기록 (추정기 런 0회)')


if __name__ == '__main__':
    main()
