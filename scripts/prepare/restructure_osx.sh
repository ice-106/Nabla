#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Restructure flat OSX demo output into 4 sub-folders (SMPLer-X style):
#
#   mesh/    ← *.obj
#   pkl/     ← *_body_data.pkl
#   render/  ← *_render.jpg
#   kpts/    ← *_kpts.jpg
#
# Usage:
#   bash restructure_osx.sh <results_dir>            # single video
#   bash restructure_osx.sh <results_dir> --batch     # all sub-dirs
# ---------------------------------------------------------------------------
set -euo pipefail

restructure_one() {
    local dir="$1"

    # Skip if already restructured (mesh/ sub-dir exists)
    if [[ -d "$dir/mesh" && -d "$dir/pkl" ]]; then
        echo "[skip] $dir  (already restructured)"
        return
    fi

    local obj_count=$(find "$dir" -maxdepth 1 -name '*.obj' | wc -l)
    local pkl_count=$(find "$dir" -maxdepth 1 -name '*_body_data.pkl' | wc -l)

    if [[ "$obj_count" -eq 0 && "$pkl_count" -eq 0 ]]; then
        echo "[skip] $dir  (no OSX output files found)"
        return
    fi

    mkdir -p "$dir"/{mesh,pkl,render,kpts}

    # Move .obj files → mesh/
    find "$dir" -maxdepth 1 -name '*.obj' -exec mv {} "$dir/mesh/" \;

    # Move _body_data.pkl files → pkl/
    find "$dir" -maxdepth 1 -name '*_body_data.pkl' -exec mv {} "$dir/pkl/" \;

    # Move _render.jpg files → render/
    find "$dir" -maxdepth 1 -name '*_render.jpg' -exec mv {} "$dir/render/" \;

    # Move _kpts.jpg files → kpts/
    find "$dir" -maxdepth 1 -name '*_kpts.jpg' -exec mv {} "$dir/kpts/" \;

    echo "[done] $dir  (obj=$obj_count  pkl=$pkl_count)"
}

# ── Main ──────────────────────────────────────────────────────────────────

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <results_dir> [--batch]"
    echo ""
    echo "  <results_dir>          Path to a single video result folder"
    echo "  <results_dir> --batch  Process every sub-directory in results_dir"
    exit 1
fi

RESULTS_DIR="$1"
BATCH="${2:-}"

if [[ "$BATCH" == "--batch" ]]; then
    echo "Batch mode: restructuring all sub-directories in $RESULTS_DIR"
    for sub in "$RESULTS_DIR"/*/; do
        restructure_one "$sub"
    done
else
    restructure_one "$RESULTS_DIR"
fi

echo "All done."
