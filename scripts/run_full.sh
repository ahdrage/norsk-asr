#!/usr/bin/env bash
# Full open-data run on the pod: all Stortinget shards + all NST + NPSC train (Bokmål), ~1,700 h,
# packed onto local disk, then one run with the pilot's best setup (top encoder layer unfrozen).
# Transcripts are also written to data/lm_text for the language model.
#   scripts/run_full.sh            # prep, then train
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=/root/work/.uv-cache HF_HOME=/root/work/.hf
cd /root/work/norsk-asr
mkdir -p /root/work/logs
W=$(python3 -c "import os; q=open('/sys/fs/cgroup/cpu.max').read().split(); print(int(q[0])//int(q[1]) if q[0]!='max' else os.cpu_count())")
DATA=/root/work/data/train_full

if [ ! -f /root/work/logs/full_prep.done ]; then
  uv run python -m norsk_asr.prep train --out "$DATA" --stortinget-shards 10000 --nst-shards 10000 --npsc \
    --lm-text /root/work/data/lm_text --workers "$W" > /root/work/logs/full_prep.log 2>&1
  touch /root/work/logs/full_prep.done
fi
uv run python -m norsk_asr.train --train-dir "$DATA" --eval-dir data/eval --out runs/full_top1 \
  --epochs 5 --lr 3e-4 --warmup 500 --max-batch-seconds 600 --eval-every 4000 --eval-limit 500 --workers 8 \
  --unfreeze-top-k 1 --unfreeze-after 2000 --encoder-lr 1e-5 > /root/work/logs/full_top1.log 2>&1
echo RUN_FULL_DONE >> /root/work/logs/full_top1.log
