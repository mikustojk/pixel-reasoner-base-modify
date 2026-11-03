#!/usr/bin/env bash
# Eight independent local 27B vision judges and a resumable audit client.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JUDGE_PYTHON="${FILTER_JUDGE_PYTHON:-/mnt/source/chengqili/miniconda3/envs/swift/bin/python}"
CLIENT_PYTHON="${FILTER_CLIENT_PYTHON:-/mnt/source/chengqili/env/eav-selfplay-s92/bin/python}"
OUT="${FILTER_OUTPUT_DIR:-$ROOT/data/deepeyes47k/quality_27b_v1}"
LOG_ROOT="${FILTER_LOG_ROOT:-$ROOT/outputs/data_filter_27b}"
GPU_LIST="${FILTER_GPUS:-0,1,2,3,4,5,6,7}"
PORT_BASE="${FILTER_PORT_BASE:-46080}"
CONCURRENCY="${FILTER_CONCURRENCY_PER_SERVER:-8}"
MAX_SEQS="${FILTER_MAX_NUM_SEQS:-8}"
SAMPLE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sample) SAMPLE="$2"; shift 2 ;;
    --output-dir) OUT="$2"; shift 2 ;;
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --help|-h)
      echo "Usage: bash scripts/run_filter_s92.sh [--sample N] [--output-dir PATH] [--gpus 0,1,...]"
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
cd "$ROOT"
mkdir -p "$LOG_ROOT" "$OUT"
RUN_DIR="$LOG_ROOT/$(date -u +%Y%m%dT%H%M%SZ)-${BASHPID}"
mkdir "$RUN_DIR"
echo "Filter run log directory: $RUN_DIR"
exec > "$RUN_DIR/launcher.log" 2>&1
export PYTHONUTF8=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
PIDS=()
URLS=()
cleanup() {
  local pid
  for pid in "${PIDS[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for gpu in "${GPUS[@]}"; do
  [[ "$gpu" =~ ^[0-7]$ ]] || { echo "Invalid GPU: $gpu" >&2; exit 2; }
  port=$((PORT_BASE + gpu))
  if curl -fsS --max-time 1 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
    echo "Port $port already has a server; stop the prior filter run first." >&2
    exit 2
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$JUDGE_PYTHON" -u -m vllm.entrypoints.openai.api_server \
    --model /mnt/source/models/Qwen/Qwen3.8-27B --served-model-name eav_filter_27b \
    --host 127.0.0.1 --port "$port" --tensor-parallel-size 1 \
    --max-model-len 8192 --gpu-memory-utilization 0.88 --max-num-seqs "$MAX_SEQS" \
    --dtype bfloat16 --limit-mm-per-prompt '{"image": 1, "video": 0}' \
    --mm-processor-kwargs '{"max_pixels": 4014080}' \
    --default-chat-template-kwargs '{"enable_thinking": false}' --enforce-eager \
    > "$RUN_DIR/server_gpu${gpu}.log" 2>&1 &
  PIDS+=("$!")
  URLS+=("http://127.0.0.1:$port/v1")
  echo "Launched GPU $gpu judge pid=${PIDS[-1]} port=$port"
done

for index in "${!GPUS[@]}"; do
  port=$((PORT_BASE + GPUS[index]))
  deadline=$((SECONDS + 1800))
  until curl -fsS --max-time 3 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; do
    kill -0 "${PIDS[index]}" 2>/dev/null || { echo "GPU ${GPUS[index]} server exited" >&2; exit 1; }
    (( SECONDS < deadline )) || { echo "GPU ${GPUS[index]} server start timeout" >&2; exit 1; }
    sleep 5
  done
  echo "GPU ${GPUS[index]} ready"
done

COMMAND=("$CLIENT_PYTHON" -u scripts/filter_train_s92.py --output-dir "$OUT" \
         --concurrency-per-server "$CONCURRENCY" --sample "$SAMPLE")
for url in "${URLS[@]}"; do COMMAND+=(--server-url "$url"); done
"${COMMAND[@]}" > "$RUN_DIR/filter.log" 2>&1
echo "Filter completed: $OUT"
