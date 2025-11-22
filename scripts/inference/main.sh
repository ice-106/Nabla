#!/usr/bin/env bash
set -x

ALGORITHM=$1
VIDEO_DIR=$2

# Check if video directory exists
if [ ! -d "$VIDEO_DIR" ]; then
    echo "Error: Video directory $VIDEO_DIR not found"
    exit 1
fi

case $ALGORITHM in
  "SMPLer-X")
    echo "Using SMPLer-X for inference"
    sh scripts/inference/SMPLer-X.sh
    ;;
  "SMPLest-X")
    echo "Using SMPLest-X for inference"
    ;;
  *)
    echo "Unsupported algorithm: $ALGORITHM"
    exit 1
    ;;
esac

# Iterate through all videos in the data/val_videos folder
for video in "$VIDEO_DIR"/*; do
    # Skip if not a file
    if [ ! -f "$video" ]; then
        continue
    fi
    
    echo "Processing video: $video"
    
    case $ALGORITHM in
      "SMPLer-X")
        sh scripts/inference_SMPLer-X.sh "$video"
        ;;
      "SMPLest-X")
        sh scripts/inference_SMPLest-X.sh "$video"
        ;;
      *)
        echo "Unsupported algorithm: $ALGORITHM"
        exit 1
        ;;
    esac
done

echo "All videos processed"
