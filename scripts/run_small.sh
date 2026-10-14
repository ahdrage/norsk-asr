#!/usr/bin/env bash
# Small end-to-end check on the pod before the pilot: ~20 h of training audio, a short
# training run that unfreezes the top encoder layers partway, and an NB-Whisper baseline on
# the same eval subsets. Run from /root/work/norsk-asr; logs go to /root/work/logs.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=/root/work/.uv-cache HF_HOME=/root/work/.hf
cd /root/work/norsk-asr
mkdir -p /root/work/logs
W=$(python3 -c "import os; q=open('/sys/fs/cgroup/cpu.max').read().split(); print(int(q[0])//int(q[1]) if q[0]!='max' else os.cpu_count())")

[ -f data/eval/fleurs_nb.jsonl ] || uv run python -m norsk_asr.prep eval --out data/eval
[ -n "$(ls data/train_small/*.jsonl 2>/dev/null)" ] || \
  uv run python -m norsk_asr.prep train --out data/train_small --stortinget-shards 5 --nst-shards 25 --workers "$W"

uv run python -m norsk_asr.train --train-dir data/train_small --eval-dir data/eval --out runs/small \
  --max-steps 1500 --warmup 200 --unfreeze-after 600 --max-batch-seconds 600 \
  --eval-every 250 --eval-limit 300 --workers 8

uv run python -m norsk_asr.evaluate --ckpt runs/small/best.pt --eval-dir data/eval --limit 300 --out results/small
uv run python -m norsk_asr.baseline --model NbAiLab/nb-whisper-large --eval-dir data/eval --limit 300 \
  --out results/nb-whisper-large-300
echo "RUN_SMALL_DONE"
