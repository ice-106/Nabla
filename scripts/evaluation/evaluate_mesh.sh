#!/bin/bash
# evaluate_mesh.sh - wrapper to run utils/Evaluation/mesh_evaluation.py
# Usage:
#   ./evaluate_mesh.sh data/test/gt data/test/pred data/outdir
#   Where first second and third arguments are ground truth folder, predicted folder, and output folder, respectively

set -x

EVAL_PY="./utils/evaluation/mesh_evaluation.py"

# Parse arguments
GT=$1
PRED=$2
OUTDIR=$3

mkdir -p "$OUTDIR"

echo "Running mesh evaluation..."

# Execute
exec "python" "$EVAL_PY" --gt "$GT" --pred "$PRED" --outdir "$OUTDIR"