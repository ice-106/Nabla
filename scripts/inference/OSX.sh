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

cd utils/extraction/OSX/demo

IMG_PATH=images/${INPUT_VIDEO}
SAVE_DIR=results/${INPUT_VIDEO}

# video to images
mkdir -p $IMG_PATH
mkdir -p $SAVE_DIR
ffmpeg -n -i videos/${INPUT_VIDEO}.${FORMAT} -f image2 -vf fps=${FPS}/1 -q:v 0 images/${INPUT_VIDEO}/%06d.jpg < /dev/null 

# inference
find ${IMG_PATH} -type f -name "*.jpg" -print0 | while IFS= read -r -d '' img_file; do
    echo "Processing image: $img_file"
    conda run -n osx python demo.py \
    --gpu 0 \
    --img_path ${img_file} \
    --output_folder ${SAVE_DIR}
done