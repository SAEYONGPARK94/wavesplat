set -e
cd /home/vexit/wavesplat/wavesplat
python3 tmp/real_gif_2026-07-27/make_real_estimation_gif.py \
  --record AA15b --win 3 \
  --out results/gifs/real_estimation_AA15b_w3_2026-07-27.gif \
  --info-key AA15b_w3
python3 tmp/real_gif_2026-07-27/make_real_estimation_gif.py \
  --record YS17 --win 0 \
  --out results/gifs/real_estimation_YS17_w0_2026-07-27.gif \
  --info-key YS17_w0 \
  --note "Honest contrast: field skill is ~0 at this site; the rho ~ 0.7 on record for YS17 is the ENCOUNTER-POINT forecast, a different metric."
echo ALLDONE
