#!/usr/bin/env bash
set -x

ALGORITHM=$1
VIDEO_DIR=$2
# The video names should be provided without the extension (e.g., trimmed_2025_03_31_batch_2_C0229 not trimmed_2025_03_31_batch_2_C0229.mp4).
SKIP_VIDEOS=$3  
INPUT_VIDEO=$(echo "$VIDEO_DIR" | sed -E 's|.*data/val_videos/||; s|\.[^.]+$||')

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
    # Check if video_name is in the comma-separated skip list
    echo ",$SKIP_VIDEOS," | grep -q ",$video_name,"
    return $?
}

case $ALGORITHM in
  "SMPLer-X")
    echo "Using SMPLer-X for inference"
    bash scripts/prepare/SMPLer-X.sh 
    ;;
  "SMPLest-X")
    echo "Using SMPLest-X for inference" 
    ;;
  "OSX")
    echo "Using OSX for inference" 
    bash scripts/prepare/OSX.sh 
    ;;
  *)
    echo "Unsupported algorithm: $ALGORITHM"
    exit 1
    ;;
esac

# Iterate through all videos in the data/val_videos folder
find "$VIDEO_DIR" -type f -print0 | while IFS= read -r -d '' video; do
    # Skip if not a file
    if [ ! -f "$video" ]; then
        continue
    fi
    input_video=$(echo "$video" | sed -E 's|.*data/val_videos/||; s|\.[^.]+$||')
    
    # Check if this video should be skipped
    if should_skip "$input_video"; then
        echo "Skipping video (already processed): $input_video"
        continue
    fi
    
    echo "Processing video: $input_video"
    
    case $ALGORITHM in
      "SMPLer-X")
        cp "$video" utils/extraction/SMPLer-X/demo/videos/$input_video.mp4
        bash utils/extraction/SMPLer-X/main/slurm_inference.sh "$input_video" mp4 30 smpler_x_h32
        ;;
      "SMPLest-X")
        bash scripts/inference/SMPLest-X.sh "$video"
        ;;
      "OSX")
        # bash scripts/inference/OSX.sh "$video"
        cp "$video" utils/extraction/OSX/demo/videos/$input_video.mp4
        bash utils/extraction/OSX/demo/inference.sh "$input_video" mp4 30
        ;;
      *)
        echo "Unsupported algorithm: $ALGORITHM"
        exit 1
        ;;
    esac
done

echo "All videos processed"
