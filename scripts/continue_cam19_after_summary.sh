#!/usr/bin/env bash
set -euo pipefail
cd /Users/mac/Documents/reels_maker
SUMMARY="/Users/mac/Documents/reels_maker/storage/output/final_reels_vova/run_20260221_cam19_batch_s3331.79_d568.21_syncfix2/summary.json"
while [ ! -f "$SUMMARY" ]; do
  sleep 30
done
REELS_MAKER_FFMPEG_TIMEOUT_MULTICAM=1800 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
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
