#!/usr/bin/env bash
set -x

INPUT_VIDEO=$1
FORMAT=$2
FPS=$3

GPUS=1
JOB_NAME=inference_${INPUT_VIDEO}

GPUS_PER_NODE=$((${GPUS}<8?${GPUS}:8))
CPUS_PER_TASK=4 # ${CPUS_PER_TASK:-2}
SRUN_ARGS=${SRUN_ARGS:-""}

cd utils/extraction/WiLoR

IMG_PATH=images/${INPUT_VIDEO}
SAVE_DIR=results/${INPUT_VIDEO}

# video to images
mkdir -p $IMG_PATH
mkdir -p $SAVE_DIR

# Convert video to images if input is a video file
if [ "$FORMAT" == "mp4" ]; then
    ffmpeg -n -i videos/${INPUT_VIDEO}.${FORMAT} -f image2 -vf fps=${FPS}/1 -q:v 0 images/${INPUT_VIDEO}/%06d.jpg < /dev/null 
fi


# inference - process all images in one call (loads models once)
echo "Processing all images in: ${IMG_PATH}"
PYTHONPATH="$(pwd)":$PYTHONPATH \
conda run -n wilor python demo.py \
--img_folder ${IMG_PATH} \
--out_folder ${SAVE_DIR} \
--file_type "*.${FORMAT}" \
--save_mesh