#!/usr/bin/env bash
# Download Qwen3.8-27B (BF16, ~55GB, 18 shards) into a local HF cache.
#
# Usage:
#   HF_HOME=./.hf-cache ./scripts/download_model.sh [REPO_ID]
#
# HF_HOME keeps the cache inside the workspace (default ./.hf-cache);
# omit it to use the system default ~/.cache/huggingface.
# The download resumes: re-run it and only missing/incomplete shards fetch.
set -euo pipefail

REPO="${1:-Qwen/Qwen3.8-27B}"

python3 - "$REPO" <<'EOF'
import sys
from huggingface_hub import snapshot_download

repo = sys.argv[1]
p = snapshot_download(
    repo,
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
    max_workers=4,   # lower this if the CDN keeps returning 408/IncompleteMessage
)
print("done:", p)
EOF
