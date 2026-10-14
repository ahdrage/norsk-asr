#!/usr/bin/env bash
# After the pilot: evaluate both runs on the same subsets as the NB-Whisper baseline
# (--limit 1000, same sampling seed) and upload the runs to the private Hub repo.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=/root/work/.uv-cache HF_HOME=/root/work/.hf
cd /root/work/norsk-asr

[ -f results/nb-whisper-large/summary.json ] || uv run python -m norsk_asr.baseline --model NbAiLab/nb-whisper-large \
  --eval-dir data/eval --limit 1000 --batch-size 32 --out results/nb-whisper-large
for run in pilot_frozen pilot_top1; do
  uv run python -m norsk_asr.evaluate --ckpt runs/$run/best.pt --eval-dir data/eval --limit 1000 --out results/$run
done

uv run python - <<'EOF'
from huggingface_hub import HfApi

api = HfApi()
repo = "ahdrage/norsk-asr"
api.create_repo(repo, private=True, exist_ok=True)
for run in ("pilot_frozen", "pilot_top1"):
    api.upload_folder(repo_id=repo, folder_path=f"runs/{run}", path_in_repo=f"runs/{run}",
                      allow_patterns=["best.pt", "units.json", "args.json", "metrics.jsonl"])
api.upload_folder(repo_id=repo, folder_path="results", path_in_repo="results")
print("uploaded to", repo)
EOF
echo FINISH_DONE
