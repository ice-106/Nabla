#!/bin/bash
set -x

echo "Start preparing OSX environment"
conda create -n osx python=3.9

conda run -n osx pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
conda run -n osx 

cd utils/extraction/OSX
conda run -n osx sh install.sh

conda run -n osx install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
conda run -n osx pip install ultralytics matplotlib-inline
conda run -n osx pip install --no-cache-dir "tqdm>=4.66.3" "setuptools>=70.0.0" "urllib3>=2.5.0" --break-system-packages 
conda run -n osx pip install numpy==1.23.5