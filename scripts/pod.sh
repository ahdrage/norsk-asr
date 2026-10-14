#!/usr/bin/env bash
# Manage a RunPod GPU pod for training. Secrets live in ~/.config/norsk-asr/env:
#   RUNPOD_API_KEY=...   HUGGINGFACE_TOKEN=...
#
#   [DISK_GB=300] scripts/pod.sh create [gpu_type] [gpu_count]   # default: one H100 SXM, 50 GB disk
#   scripts/pod.sh setup                           # sync code, install deps, log in to the Hub
#   scripts/pod.sh status | ssh | sync | stop | delete | balance
#
# The pod id is kept in ~/.config/norsk-asr/pod_id. Always `delete` when done: a stopped pod
# still bills for its volume, and a running one bills per second.
set -euo pipefail

CONF="$HOME/.config/norsk-asr"
# shellcheck disable=SC1091
source "$CONF/env"
API="https://rest.runpod.io/v1"
KEY="$HOME/.ssh/runpod_norsk_asr"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

api() { curl -fsS -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" "$@"; }
pod_id() { cat "$CONF/pod_id"; }

ssh_target() {
  api "$API/pods/$(pod_id)" | python3 -c '
import json, sys
p = json.load(sys.stdin)
port = (p.get("portMappings") or {}).get("22")
ip = p.get("publicIp")
if not (ip and port):
    sys.exit("pod has no public SSH yet (status: %s)" % p.get("desiredStatus"))
print(ip, port)'
}

case "${1:-}" in
  create)
    gpu="${2:-NVIDIA H100 80GB HBM3}"
    count="${3:-1}"
    body=$(python3 - "$gpu" "$count" "${DISK_GB:-50}" <<'EOF'
import json, os, sys
print(json.dumps({
    "name": "norsk-asr",
    "imageName": "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04",
    "gpuTypeIds": [sys.argv[1]],
    "gpuCount": int(sys.argv[2]),
    "cloudType": "SECURE",
    "containerDiskInGb": int(sys.argv[3]),  # local disk under /root; /workspace is slow network storage
    "volumeInGb": 200,
    "volumeMountPath": "/workspace",
    "ports": ["22/tcp"],
    "minVCPUPerGPU": 16,
    "minRAMPerGPU": 64,
    "env": {"PUBLIC_KEY": open(os.path.expanduser("~/.ssh/runpod_norsk_asr.pub")).read().strip()},
}))
EOF
)
    api -X POST "$API/pods" -d "$body" | tee /dev/stderr | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])' > "$CONF/pod_id"
    echo "created pod $(pod_id)"
    ;;
  balance)
    curl -fsS -X POST "https://api.runpod.io/graphql" -H "Authorization: Bearer $RUNPOD_API_KEY" \
      -H "Content-Type: application/json" -d '{"query":"query { myself { clientBalance currentSpendPerHr } }"}' |
      python3 -c 'import json,sys; m=json.load(sys.stdin)["data"]["myself"]; print("balance $%.2f, spending $%.2f/h" % (m["clientBalance"], m["currentSpendPerHr"]))'
    ;;
  status)
    api "$API/pods/$(pod_id)" | python3 -m json.tool
    ;;
  ssh)
    read -r ip port < <(ssh_target)
    shift
    exec ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -p "$port" "root@$ip" "$@"
    ;;
  sync)
    # Code only; data is prepared on the pod from the Hub, and eval sets are small enough to copy.
    read -r ip port < <(ssh_target)
    rsync -rlptz --delete --no-owner --no-group -e "ssh -i $KEY -p $port -o StrictHostKeyChecking=accept-new" \
      --exclude .venv --exclude data --exclude runs --exclude results --exclude logs --exclude scratch --exclude __pycache__ \
      "$REPO/" "root@$ip:/root/work/norsk-asr/"
    ;;
  setup)
    "$0" ssh 'mkdir -p /root/work/norsk-asr; command -v rsync >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync >/dev/null)'
    "$0" sync
    # The token goes over SSH stdin into the Hub's token file, not into the pod's env config.
    printf '%s' "$HUGGINGFACE_TOKEN" | "$0" ssh 'mkdir -p ~/.cache/huggingface && cat > ~/.cache/huggingface/token && chmod 600 ~/.cache/huggingface/token'
    "$0" ssh 'set -e
      nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
      command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
      export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=/root/work/.uv-cache HF_HOME=/root/work/.hf
      mkdir -p $HF_HOME && cp ~/.cache/huggingface/token $HF_HOME/token
      cd /root/work/norsk-asr && uv sync -q
      # Pods with a CUDA 12.x driver cannot run the locked cu130 build; swap in the cu126 build of the
      # same torch version. Run scripts set UV_NO_SYNC=1 so `uv run` keeps it.
      if ! uv run python -c "import torch, sys; sys.exit(not torch.cuda.is_available())" 2>/dev/null; then
        v=$(uv run python -c "import torch; print(torch.__version__.split(\"+\")[0])")
        uv pip install -q --python .venv/bin/python "torch==$v+cu126" --index-url https://download.pytorch.org/whl/cu126
      fi
      UV_NO_SYNC=1 uv run python -c "import torch; print(\"torch\", torch.__version__, \"cuda ok:\", torch.cuda.is_available())"'
    ;;
  stop)
    api -X POST "$API/pods/$(pod_id)/stop" && echo "stopped $(pod_id)"
    ;;
  delete)
    api -X DELETE "$API/pods/$(pod_id)" && echo "deleted $(pod_id)" && rm "$CONF/pod_id"
    ;;
  *)
    sed -n '2,10p' "$0"; exit 1
    ;;
esac
