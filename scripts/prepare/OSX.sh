#!/bin/bash
set -x

echo "Start preparing OSX environment"
conda create -n osx python=3.9

conda run -n osx pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

cd utils/extraction/OSX
mkdir -p demo/videos

conda run -n osx bash install.sh
# Download pretrained models
mkdir -p pretrained_models
pip install gdown
gdown 1aytg6JHWGqkxTCzEiLQrZlhrTD87IRya --output pretrained_models/osx_l.pth.tar

# Download Human Model Files
gdown 1KT3Nd318QbiOpNX9Oii7HBid43a53zXB
unzip -nq human_model.zip -d common/utils/human_model_files
rm human_model.zip

conda run -n osx pip install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
conda run -n osx pip install ultralytics matplotlib-inline
conda run -n osx pip install --no-cache-dir "tqdm>=4.66.3" "setuptools>=70.0.0" "urllib3>=2.5.0" --break-system-packages 
conda run -n osx pip install numpy==1.23.5

# Patch demo.py to fix filename extraction issue
cat ../../../patches/OSX/demo.py > demo/demo.py

# Patch inference.sh to fix inference loop
cat ../../../patches/OSX/inference.sh > demo/inference.sh