#!/usr/bin/env bash
# Pilot on the pod: ~485 h (100 Stortinget shards + all NST) packed into RAM (/dev/shm), then two
# runs side by side on one GPU: A keeps the encoder frozen, B unfreezes the top encoder layer
# after 2000 steps. Usage: scripts/run_pilot.sh prep | train
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=/root/work/.uv-cache HF_HOME=/root/work/.hf
cd /root/work/norsk-asr
mkdir -p /root/work/logs
W=$(python3 -c "import os; q=open('/sys/fs/cgroup/cpu.max').read().split(); print(int(q[0])//int(q[1]) if q[0]!='max' else os.cpu_count())")
DATA=/dev/shm/train_pilot

case "$1" in
  prep)
    uv run python -m norsk_asr.prep train --out "$DATA" --stortinget-shards 100 --nst-shards 10000 --workers "$W"
    ;;
  train)
    common=(--train-dir "$DATA" --eval-dir data/eval --epochs 6 --lr 3e-4 --warmup 500
            --max-batch-seconds 600 --eval-every 2000 --eval-limit 500 --workers 8)
    nohup uv run python -m norsk_asr.train "${common[@]}" --out runs/pilot_frozen --unfreeze-top-k 0 --unfreeze-after 0 \
      > /root/work/logs/pilot_frozen.log 2>&1 &
    nohup uv run python -m norsk_asr.train "${common[@]}" --out runs/pilot_top1 --unfreeze-top-k 1 --unfreeze-after 2000 \
      --encoder-lr 1e-5 > /root/work/logs/pilot_top1.log 2>&1 &
    wait
    ;;
esac
