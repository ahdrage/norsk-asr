#!/usr/bin/env bash
# After run_full.sh: greedy eval on the test subsets (same as the baseline, --limit 1000) and on the
# dev sets, with log-prob dumps for LM decoding (done on CPU); then upload the run to the Hub repo.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=/root/work/.uv-cache HF_HOME=/root/work/.hf UV_NO_SYNC=1
cd /root/work/norsk-asr
run=${1:-full_top1}
uv run python -m norsk_asr.evaluate --ckpt runs/$run/best.pt --eval-dir data/eval --limit 1000 --out results/$run --dump-logprobs
uv run python -m norsk_asr.evaluate --ckpt runs/$run/best.pt --eval-dir data/dev --limit 1000 --out results/${run}_dev --dump-logprobs

RUN=$run uv run python - <<'PY'
import os
from huggingface_hub import HfApi
from norsk_asr.train import export_checkpoint

api = HfApi()
repo, run = "ahdrage/norsk-asr", os.environ["RUN"]
export_checkpoint(f"runs/{run}/best.pt", f"runs/{run}")  # publish safetensors, not pickles
api.upload_folder(repo_id=repo, folder_path=f"runs/{run}", path_in_repo=f"runs/{run}",
                  allow_patterns=["model.safetensors", "config.json", "units.json", "args.json", "metrics.jsonl"])
api.upload_folder(repo_id=repo, folder_path=f"results/{run}", path_in_repo=f"results/{run}",
                  ignore_patterns=["*.npz"])
print("uploaded", run, "to", repo)
PY
echo FINISH_DONE
