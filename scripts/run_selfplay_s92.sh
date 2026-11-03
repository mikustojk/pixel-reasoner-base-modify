#!/usr/bin/env bash
# s92 H200 entry: managed judge child + foreground training.
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHASE=smoke
TRAIN_PYTHON="${SELFPLAY_TRAIN_PYTHON:-/mnt/source/chengqili/env/eav-selfplay-s92/bin/python}"
TRAIN_CONFIG="${SELFPLAY_TRAIN_CONFIG:-configs/selfplay_s92.json}"
OUTPUT_ROOT="$PROJECT_ROOT/outputs/cluster"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase) PHASE="$2"; shift 2 ;;
    --train-python) TRAIN_PYTHON="$2"; shift 2 ;;
    --train-config) TRAIN_CONFIG="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --help|-h)
      echo "Usage: bash scripts/run_selfplay_s92.sh [--phase smoke|train|all|check] [--train-python PATH] [--train-config PATH] [--output-root PATH]"
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$PHASE" in all|smoke|train|check) ;; *) echo "Invalid phase: $PHASE" >&2; exit 2 ;; esac
cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_ROOT"
RUN_DIR="$OUTPUT_ROOT/$(date -u +%Y%m%dT%H%M%SZ)-${BASHPID}"
mkdir "$RUN_DIR"
mkdir "$RUN_DIR/logs"
echo "Persistent run directory: $RUN_DIR"
exec > "$RUN_DIR/logs/launcher.log" 2>&1
trap 'rc=$?; echo "Launcher finished: exit_code=$rc run=$RUN_DIR"; exit "$rc"' EXIT
if [[ ! -x "$TRAIN_PYTHON" ]]; then
  echo "Missing training Python: $TRAIN_PYTHON. Set --train-python to the prepared DeepSpeed environment." >&2
  exit 2
fi
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
export PYTHONFAULTHANDLER=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Respect the scheduler's allocation. The parent cluster command may explicitly
# unset CUDA_VISIBLE_DEVICES; this script never expands an inherited allocation.
echo "phase=$PHASE train_python=$TRAIN_PYTHON train_config=$TRAIN_CONFIG CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset}"

PREFLIGHT_CONFIG="$TRAIN_CONFIG"
if [[ "$PHASE" == smoke ]]; then PREFLIGHT_CONFIG=configs/selfplay_s92_smoke.json; fi

"$TRAIN_PYTHON" -u scripts/preflight_selfplay.py \
  --config "$PREFLIGHT_CONFIG" \
  --output "$RUN_DIR/logs/environment.json" \
  > "$RUN_DIR/logs/preflight.log" 2>&1
if [[ "$PHASE" == check ]]; then exit 0; fi

run_stage() {
  local stage="$1" config="$2"
  echo "Starting $stage; log=$RUN_DIR/logs/$stage.log"
  # This command stays in the foreground. A nonzero exit stops the entry script
  # and is preserved for the scheduler; the Python owner cleans up its judge.
  "$TRAIN_PYTHON" -u train_selfplay.py \
    --config "$config" --output "$RUN_DIR/$stage" \
    > "$RUN_DIR/logs/$stage.log" 2>&1
  echo "Finished $stage"
}
if [[ "$PHASE" == all || "$PHASE" == smoke ]]; then
  run_stage smoke configs/selfplay_s92_smoke.json
fi
if [[ "$PHASE" == all || "$PHASE" == train ]]; then
  # A separate fresh run starts from the configured base, never smoke weights.
  run_stage train "$TRAIN_CONFIG"
fi
