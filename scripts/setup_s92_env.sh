#!/usr/bin/env bash
# Isolate self-play upgrades while reusing the existing CUDA PyTorch stack.
set -Eeuo pipefail

BASE=/mnt/source/chengqili/env/verl
TARGET=/mnt/source/chengqili/env/eav-selfplay-s92
if [[ ! -x "$BASE/bin/python" ]]; then
  echo "Missing s92 base environment: $BASE" >&2
  exit 2
fi
if [[ ! -x "$TARGET/bin/python" ]]; then
  python3 -m venv "$TARGET"
fi
BASE_PACKAGES="$BASE/lib/python3.12/site-packages"
TARGET_PACKAGES="$TARGET/lib/python3.12/site-packages"
printf '%s\n' "$BASE_PACKAGES" > "$TARGET_PACKAGES/s92_base.pth"
"$TARGET/bin/python" -m pip install deepspeed==0.16.9 transformers==4.57.1
"$TARGET/bin/python" -c 'import torch, transformers, deepspeed; from transformers import Qwen2_5_VLForConditionalGeneration; print(torch.__version__, transformers.__version__, deepspeed.__version__)'
