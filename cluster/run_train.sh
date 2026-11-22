#!/bin/bash
# run_train.sh - HTCondor job script for CHTC. Runs on the GPU machine, inside
# the container named in the submit file.
#
# What it does:
#   1. unpacks the code tarball and the dataset tarball into the job's scratch dir
#   2. finds the newest checkpoint in /staging for this run and resumes from it
#   3. mirrors new checkpoints back to /staging every 3 minutes while training
#   4. at the end, tars the full results directory into the run's /staging folder
#
# So a job that is evicted, or killed at the GPU Lab time limit, can simply be
# resubmitted and it carries on from its last checkpoint.
#
# Usage (from the submit file):
#   arguments = $(Cluster) $(Process) <config.yaml> [extra args for the trainer...]
#
# Environment (set with `environment = "..."` in the submit file):
#   STAGING     required  your /staging directory
#   DATA_TAR    optional  dataset tarball  (default $STAGING/lodopab_subset.tar.gz)
#   TARBALL     optional  code tarball     (default code.tar.gz, shipped with the job)
#   CODE_DIR    optional  top-level folder inside the code tarball (default code)
#   ENTRY       optional  training script inside CODE_DIR (default train.py)
#   RUN_TAG     optional  suffix on the run name, keeps smoke runs separate
#
# The training script is expected to accept:
#   --config <yaml> --data-root <dir> --output-dir <dir> [--resume <ckpt>] [extra...]
# and to write checkpoints named epoch_NNNN.pt (zero-padded, so plain sort is
# chronological) somewhere under --output-dir.

set -euo pipefail

CLUSTER="$1"; PROCESS="$2"; CONFIG="$3"
shift 3
EXTRA=("$@")

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
: "${STAGING:?set STAGING in the submit file, e.g. environment = \"STAGING=/staging/<path>\"}"
DATA_TAR="${DATA_TAR:-$STAGING/lodopab_subset.tar.gz}"
TARBALL="${TARBALL:-code.tar.gz}"
CODE_DIR="${CODE_DIR:-code}"
ENTRY="${ENTRY:-train.py}"

# configs/foo.yaml -> foo. One /staging folder per config, so two experiments
# can never resume into each other's checkpoints.
RUN_NAME="$(basename "$CONFIG" .yaml)${RUN_TAG:-}"

RUN_DIR="$STAGING/runs/$RUN_NAME"
CKPT_DIR="$RUN_DIR/checkpoints"
mkdir -p "$CKPT_DIR"

WORK="$_CONDOR_SCRATCH_DIR"

# --------------------------------------------------------------------------
# Sanity
# --------------------------------------------------------------------------
echo "=== host: $(hostname) | run: $RUN_NAME | job: ${CLUSTER}.${PROCESS} ==="
nvidia-smi || echo "note: nvidia-smi not in container (expected)"
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"

# --------------------------------------------------------------------------
# Unpack code and data
# --------------------------------------------------------------------------
echo "=== code tarball: $TARBALL ==="
tar -xzf "$TARBALL" -C "$WORK"
mkdir -p "$WORK/data"
cp "$DATA_TAR" "$WORK/data.tar.gz"
tar -xzf "$WORK/data.tar.gz" -C "$WORK/data"
rm "$WORK/data.tar.gz"

RESULTS="$WORK/results"
mkdir -p "$RESULTS"

# --------------------------------------------------------------------------
# Look for a checkpoint to resume from.
#
# Resume only from full training checkpoints (model + optimizer + scheduler +
# epoch counter). A weights-only "best" copy would silently restart the
# optimizer, so it is mirrored to /staging but never resumed from.
#
# NOTE: this means a run meant to start fresh must have an EMPTY checkpoint
# directory. Check before submitting.
# --------------------------------------------------------------------------
LATEST="$(ls -1 "$CKPT_DIR"/epoch_*.pt 2>/dev/null | sort | tail -1 || true)"
if [ -n "$LATEST" ]; then
    echo "=== resuming from $LATEST ==="
else
    echo "=== no checkpoint found, starting fresh ==="
fi

# --------------------------------------------------------------------------
# Mirror checkpoints to staging every 3 minutes, in the background
# --------------------------------------------------------------------------
sync_checkpoints() {
    find "$RESULTS" -name 'epoch_*.pt' -exec cp -u {} "$CKPT_DIR/" \; 2>/dev/null || true
    find "$RESULTS" -name 'best.pt'    -exec cp -u {} "$CKPT_DIR/" \; 2>/dev/null || true
}

( while true; do sleep 180; sync_checkpoints; done ) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null || true' EXIT

# --------------------------------------------------------------------------
# Train
# --------------------------------------------------------------------------
cd "$WORK/$CODE_DIR"

ARGS=(
    --config "$CONFIG"
    --data-root "$WORK/data/lodopab"
    --output-dir "$RESULTS"
)
[ -n "$LATEST" ] && ARGS+=(--resume "$LATEST")
ARGS+=(${EXTRA[@]+"${EXTRA[@]}"})

echo "=== python $ENTRY ${ARGS[*]} ==="
python "$ENTRY" "${ARGS[@]}"

# --------------------------------------------------------------------------
# Final sync, then the full results tarball
# --------------------------------------------------------------------------
kill $SYNC_PID 2>/dev/null || true
sync_checkpoints

cd "$WORK"
tar -czf "results_${RUN_NAME}_${CLUSTER}_${PROCESS}.tar.gz" results
mv "results_${RUN_NAME}_${CLUSTER}_${PROCESS}.tar.gz" "$RUN_DIR/"

echo "=== done: checkpoints in $CKPT_DIR, results in $RUN_DIR ==="
