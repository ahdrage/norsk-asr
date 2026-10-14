#!/usr/bin/env bash
# Manage a RunPod GPU pod for training. Secrets live in ~/.config/norsk-asr/env:
#   RUNPOD_API_KEY=...   HF_TOKEN=...
#
#   scripts/pod.sh create [gpu_type] [gpu_count]   # default: one H100 SXM
#   scripts/pod.sh status | ssh | sync | stop | delete
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
    body=$(python3 - "$gpu" "$count" <<'EOF'
import json, os, sys
print(json.dumps({
    "name": "norsk-asr",
    "imageName": "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04",
    "gpuTypeIds": [sys.argv[1]],
    "gpuCount": int(sys.argv[2]),
    "cloudType": "SECURE",
    "containerDiskInGb": 50,
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
    rsync -az --delete -e "ssh -i $KEY -p $port -o StrictHostKeyChecking=accept-new" \
      --exclude .venv --exclude data --exclude runs --exclude scratch --exclude __pycache__ \
      "$REPO/" "root@$ip:/workspace/norsk-asr/"
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
