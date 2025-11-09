#!/bin/bash
set -x

INPUT_VIDEO=$1

echo "Start infering mesh model with SMPLer-X"
conda create -n smplerx python=3.8 -y && conda activate smplerx

conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch -c nvidia -y
pip install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
pip install -r utils/SMPLer-X/requirements.txt

cd utils/SMPLer-X/main/transformer_utils
pip install -v -e .
cd ../..

# Download necessary files for SMPLer-X inference
pip install huggingface_hub
# Download SMPL
huggingface-cli download camenduru/SMPLer-X SMPL_NEUTRAL.pkl SMPL_FEMALE.pkl SMPL_MALE.pkl --local-dir ../common/utils/human_model_files/smpl
# Download SMPLX
huggingface-cli download camenduru/SMPLer-X MANO_SMPLX_vertex_ids.pkl SMPL-X__FLAME_vertex_ids.npy SMPLX_FEMALE.npz SMPLX_MALE.npz SMPLX_NEUTRAL.npz SMPLX_NEUTRAL.pkl SMPLX_to_J14.pkl --local-dir ../common/utils/human_model_files/smplx
# Download mmdet
huggingface-cli download camenduru/SMPLer-X faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth mmdet_faster_rcnn_r50_fpn_coco.py --local-dir ../pretrained_models/mmdet
# Download Pretrained Models 
huggingface-cli download camenduru/SMPLer-X smpler_x_h32.pth.tar --local-dir ../pretrained_models

sh utils/SMPLer-X/main/slurm_inference.sh $INPUT_VIDEO mp4 30 smpler_x_h32