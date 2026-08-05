import os, sys
sys.path.insert(0, os.path.abspath('.'))
from wavesplat.data_loader import EpisodeDataLoader
EPS = ['BF3_5comp_seed0_uniform','BF3_5comp_seed0_slope','BF4_5comp_seed0_uniform',
       'BF4_15comp_seed0_uniform','BF5_15comp_seed0_uniform','BF4_30comp_seed0_uniform',
       'BF5_30comp_seed0_uniform','BF2_1comp_seed0_uniform']
for e in EPS:
    l = EpisodeDataLoader(f'data_finemesh_2026-07-23/episode_{e}.h5')
    n = len(l.get_tracking_frames(tp_factor=2.0))
    print(f'{e:32s} tracking_frames={n:4d}  Tp={l.Tp_true:.2f}')
