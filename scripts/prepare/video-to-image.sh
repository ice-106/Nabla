#!/bin/bash
##############################################################################
# Video to Image Extractor
#
# This script converts video files into sequences of image frames using FFmpeg.
# It processes all video files in a specified directory and extracts frames
# at a configurable frame rate, outputting them to organized subdirectories.
#
# Dependencies: ffmpeg
##############################################################################

set -euo pipefail  # Exit on error, undefined variables, and pipe failures

usage() {
  echo "Usage: $0 -d <directory> -t <video_type> -o <output_format> [-O <output_dir>] [-f <fps>]"
  echo ""
  echo "Required Arguments:"
  echo "  -d  Working directory containing the video files to process"
  echo "  -t  Video file extension to search for (mp4, mkv, gif, etc.)"
  echo "  -o  Output image format for extracted frames (jpg, png, bmp, etc.)"
  echo ""
  echo "Optional Arguments:"
  echo "  -O  Output base directory (default: same as input directory -d)"
  echo "  -f  Frames per second to extract (default: source video fps)"
  echo "  -h  Display this help message"
  exit 1
}

FPS=""

while getopts "d:t:o:O:f:h" opt; do
  case $opt in
    d) DIR="$OPTARG" ;;
    t) VIDEO_TYPE="$OPTARG" ;;
    o) OUTPUT_FMT="$OPTARG" ;;
    O) OUTPUT_DIR="$OPTARG" ;;
    f) FPS="$OPTARG" ;;
    h|*) usage ;;
  esac
done

[ -z "${DIR:-}" ] || [ -z "${VIDEO_TYPE:-}" ] || [ -z "${OUTPUT_FMT:-}" ] && usage
OUTPUT_DIR="${OUTPUT_DIR:-$DIR}"


if [ ! -d "$DIR" ]; then
  echo "Error: directory '$DIR' does not exist"
  exit 1
fi

##############################################################################
# Main Processing Loop: Iterate through all matching video files
#
# For each video file matching the extension pattern:
#   1. Extract the base filename (without extension)
#   2. Create a subdirectory in OUTPUT_DIR with the video's base name
#   3. Use ffmpeg to extract frames at the specified fps rate
#   4. Save frames as numbered image files (000001.ext, 000002.ext, etc.)
#
# ffmpeg flags:
#   -n           : Don't overwrite existing files
#   -i           : Input file
#   -f image2    : Output format as image sequence
#   -vf "fps=..."  : Frame rate filter
#   -q:v 0       : Maximum quality (0 = best)
#   < /dev/null  : Suppress interactive prompts
##############################################################################
for filepath in "$DIR"/*."$VIDEO_TYPE"; do
  [ -f "$filepath" ] || continue

  basename=$(basename "$filepath" ."$VIDEO_TYPE")
  outdir="$OUTPUT_DIR/$basename"
  mkdir -p "$outdir"

  if [ -z "$FPS" ]; then
    fps=$(ffprobe -v error -select_streams v:0 \
      -show_entries stream=r_frame_rate \
      -of default=noprint_wrappers=1:nokey=1 "$filepath")
  else
    fps="$FPS"
  fi

  echo "Processing: $filepath -> $outdir (fps=$fps)"
  ffmpeg -n -i "$filepath" \
    -f image2 -vf "fps=$fps" -q:v 0 \
    "$outdir/%06d.$OUTPUT_FMT" < /dev/null
done