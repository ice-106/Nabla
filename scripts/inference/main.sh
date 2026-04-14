#!/usr/bin/env bash
set -x

usage() {
    echo "Usage: $(basename "$0") <ALGORITHM> <VIDEO_DIR> <FORMAT> [--skip VIDEO,...] [--only VIDEO,...]"
    echo ""
    echo "Arguments:"
    echo "  ALGORITHM      Inference algorithm to use. One of: SMPLer-X, SMPLest-X, OSX, WiLoR"
    echo "  VIDEO_DIR      Path to directory containing per-video subdirectories"
    echo "  FORMAT         Video file format/extension (e.g. mp4)"
    echo "  --skip VIDEO,... Optional. Comma-separated list of video names to skip (without extension)"
    echo "  --only VIDEO,... Optional. Comma-separated list of video names to process exclusively (without extension)"
    echo ""
    echo "Examples:"
    echo "  bash $(basename "$0") SMPLer-X data/videos mp4"
    echo "  bash $(basename "$0") WiLoR data/videos mp4 --skip video_01,video_03"
    echo "  bash $(basename "$0") WiLoR data/videos mp4 --only video_02,video_04"
    exit 1
}

ALGORITHM=$1
VIDEO_DIR=$2
FORMAT=$3

SKIP_VIDEOS=""
ONLY_VIDEOS=""

# Validate required arguments
if [ $# -lt 3 ]; then
    usage
fi

# Parse optional flags
shift 3
while [ $# -gt 0 ]; do
    case "$1" in
        --skip)
            SKIP_VIDEOS=$2
            shift 2
            ;;
        --only)
            ONLY_VIDEOS=$2
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            ;;
    esac
done

# --skip and --only are mutually exclusive
if [ -n "$SKIP_VIDEOS" ] && [ -n "$ONLY_VIDEOS" ]; then
    echo "Error: --skip and --only cannot be used together"
    usage
fi

INPUT_VIDEO=$(echo "$VIDEO_DIR" | sed -E 's|.*data/||; s|\.[^.]+$||')

# Check if video directory exists
if [ ! -d "$VIDEO_DIR" ]; then
    echo "Error: Video directory $VIDEO_DIR not found"
    exit 1
fi

# Returns 0 (true) if the video is in a comma-separated list
in_list() {
    local video_name=$1
    local list=$2
    case ",$list," in
        *",${video_name},"*) return 0 ;;
        *) return 1 ;;
    esac
}

# Returns 0 (true) if the video should be skipped
should_skip() {
    local video_name=$1
    # Skip if video is in SKIP_VIDEOS blacklist
    if [ -n "$SKIP_VIDEOS" ] && in_list "$video_name" "$SKIP_VIDEOS"; then
        return 0
    fi
    # Skip if ONLY_VIDEOS whitelist is set and video is NOT in it
    if [ -n "$ONLY_VIDEOS" ] && ! in_list "$video_name" "$ONLY_VIDEOS"; then
        return 0
    fi
    return 1
}

case $ALGORITHM in
  "SMPLer-X")
    echo "Using SMPLer-X for inference"
    ;;
  "SMPLest-X")
    echo "Using SMPLest-X for inference"
    ;;
  "OSX")
    echo "Using OSX for inference"
    ;;
  "WiLoR")
    echo "Using WiLoR for inference"
    ;;
  *)
    echo "Unsupported algorithm: $ALGORITHM"
    exit 1
    ;;
esac

while IFS= read -r -d '' folder; do
    input_video=$(basename "$folder")

    # Check if this video should be skipped
    if should_skip "$input_video"; then
        echo "Skipping video: $input_video"
        continue
    fi

    echo "Processing video: $input_video"

    case $ALGORITHM in
      "SMPLer-X")
        bash utils/extraction/SMPLer-X/main/slurm_inference.sh "$input_video" "$FORMAT" 30 smpler_x_h32
        ;;
      "SMPLest-X")
        bash scripts/inference/SMPLest-X.sh "$folder"
        ;;
      "OSX")
        bash utils/extraction/OSX/demo/inference.sh "$input_video" "$FORMAT" 30
        ;;
      "WiLoR")
        bash utils/extraction/WiLoR/inference.sh "$input_video" "$FORMAT" 30
        ;;
      *)
        echo "Unsupported algorithm: $ALGORITHM"
        exit 1
        ;;
    esac
done < <(find "$VIDEO_DIR" -mindepth 1 -maxdepth 1 -type d -print0)

echo "All videos processed"
