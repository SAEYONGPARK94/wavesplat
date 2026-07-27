import sys, os
from PIL import Image
g = 'results/gifs/real_estimation_2026-07-27.gif'
im = Image.open(g)
n = im.n_frames
out = 'tmp/real_gif_2026-07-27'
for name, idx in [('frame_first', 0), ('frame_mid', n//2), ('frame_last', n-1)]:
    im.seek(idx)
    im.convert('RGB').save(f'{out}/{name}.png')
print('n_frames', n, 'size', im.size, 'saved to', out)
