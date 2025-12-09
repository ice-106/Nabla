#!/bin/bash
# evaluate_mesh.sh - wrapper to run utils/Evaluation/mesh_evaluation.py
# Usage:
#   ./evaluate_mesh.sh data/test/gt data/test/pred data/outdir
#   Where first second and third arguments are ground truth folder, predicted folder, and output folder, respectively

set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EVAL_PY="$PROJECT_ROOT/utils/Evaluation/mesh_evaluation.py"

# Parse arguments
GT=$1
PRED=$2
OUTDIR=$3

mkdir -p "$OUTDIR"

echo "Running mesh evaluation..."

# Execute
exec "python" "$EVAL_PY" --gt "$GT" --pred "$PRED" --outdir "$OUTDIR" "${FORWARD_ARGS[@]}"