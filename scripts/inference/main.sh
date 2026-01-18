#!/usr/bin/env bash
set -x

ALGORITHM=$1
VIDEO_DIR=$2
# The video names should be provided without the extension (e.g., trimmed_2025_03_31_batch_2_C0229 not trimmed_2025_03_31_batch_2_C0229.mp4).
FORMAT=$3
SKIP_VIDEOS=$4  
INPUT_VIDEO=$(echo "$VIDEO_DIR" | sed -E 's|.*data/||; s|\.[^.]+$||')

# Check if video directory exists
if [ ! -d "$VIDEO_DIR" ]; then
    echo "Error: Video directory $VIDEO_DIR not found"
    exit 1
fi

# Function to check if a video should be skipped
should_skip() {
    local video_name=$1
    if [ -z "$SKIP_VIDEOS" ]; then
        return 1  # Don't skip if no skip list provided
    fi
  # Robust membership check that tolerates spaces and avoids regex edge cases
  case ",$SKIP_VIDEOS," in
    *",${video_name},"*) return 0 ;;
    *) return 1 ;;
  esac
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
        echo "Skipping video (already processed): $input_video"
        continue
    fi
    
    
    # Check if this video should be skipped
    if should_skip "$input_video"; then
        echo "Skipping video (already processed): $input_video"
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
