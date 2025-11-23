#!/bin/bash
set -x

VIDEO_PATH=$1
INPUT_VIDEO=$(echo "$VIDEO_PATH" | sed -E 's|.*data/val_videos/||; s|\.[^.]+$||')

echo "Start infering mesh model with SMPLer-X"
conda create -n smplerx python=3.8 -y

conda run -n smplerx conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch -c nvidia -y
conda run -n smplerx pip install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
conda run -n smplerx pip install -r utils/extraction/SMPLer-X/requirements.txt

cd utils/extraction/SMPLer-X/main/transformer_utils
conda run -n smplerx pip install -v -e .
cd ../..
    
mkdir -p demo/videos

# Download necessary files for SMPLer-X inference
pip install huggingface_hub
# Download SMPL
hf download camenduru/SMPLer-X SMPL_NEUTRAL.pkl SMPL_FEMALE.pkl SMPL_MALE.pkl --local-dir common/utils/human_model_files/smpl
# Download SMPLX
hf download camenduru/SMPLer-X MANO_SMPLX_vertex_ids.pkl SMPL-X__FLAME_vertex_ids.npy SMPLX_FEMALE.npz SMPLX_MALE.npz SMPLX_NEUTRAL.npz SMPLX_NEUTRAL.pkl SMPLX_to_J14.pkl --local-dir common/utils/human_model_files/smplx
# Download mmdet
hf download camenduru/SMPLer-X faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth mmdet_faster_rcnn_r50_fpn_coco.py --local-dir pretrained_models/mmdet
# Download Pretrained Models 
hf download camenduru/SMPLer-X smpler_x_h32.pth.tar --local-dir pretrained_models

conda run -n smplerx pip install yapf==0.24.0 numpy==1.23
conda run -n smplerx conda install pytorch torchvision
conda run -n smplerx conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch -c nvidia -y

# Patch slurm_inference.sh from SMPLer-X
cat ../../../patches/SMPLer-X/slurm_inference.sh > main/slurm_inference.sh

# Patch mask conversion compatibility issue in PyTorch
cat ../../../patches/SMPLer-X/conversions.py > /usr/local/envs/smplerx/lib/python3.8/site-packages/torchgeometry/core/conversions.py

cd main