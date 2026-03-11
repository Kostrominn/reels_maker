#!/usr/bin/env bash
set -euo pipefail
cd /Users/mac/Documents/reels_maker
while pgrep -f "run_vova_batches.py --ref yadisk:disk:/Интервью/Вова/IMG_0003.mov --other yadisk:disk:/Интервью/Вова/IMG_0019.mov --offsets storage/sync/offsets_vova_0019.json --start-seconds 3300 --end-seconds 3900 --chunk-seconds 600 --max-clips 6 --run-tag 20260221_cam19 --output-base storage/output/final_reels_vova" >/dev/null; do
  sleep 30
done
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
  ./.venv/bin/python -u scripts/run_vova_batches.py \
  --ref yadisk:disk:/Интервью/Вова/IMG_0003.mov \
  --other yadisk:disk:/Интервью/Вова/IMG_0019.mov \
  --offsets storage/sync/offsets_vova_0019.json \
  --start-seconds 3900 \
  --end-seconds 7520.15 \
  --chunk-seconds 600 \
  --max-clips 6 \
  --run-tag 20260221_cam19 \
  --output-base storage/output/final_reels_vova \
  >> storage/output/final_reels_vova/autobatch_20260221_cam19_from3900.log 2>&1
