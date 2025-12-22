#!/bin/bash
set -x

echo "Start preparing WiLoR environment"
conda create --name wilor python=3.10

conda run -n wilor pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
conda run -n wilor pip install --upgrade wheel setuptools pip==22

conda run -n wilor pip install -r requirements.txt


pip install huggingface_hub
huggingface-cli download "rolpotamias/WiLoR" "pretrained_models/detector.pt" --repo-type space --local-dir utils/extraction/WiLoR/pretrained_models/
huggingface-cli download "rolpotamias/WiLoR" "pretrained_models/wilor_final.ckpt" --repo-type space --local-dir utils/extraction/WiLoR/pretrained_models/

gdown 1y0ZZeT52Hlt5V5fEDoeJnLegVwM1ZRn0 --output utils/extraction/WiLoR/mano_data/MANO_RIGHT.pkl
gdown 1KN6OAyzd7JTbkBkhu09ue6ZPDLBEgwrQ --output utils/extraction/WiLoR/mano_data/MANO_LEFT.pkl

conda run -n wilor pip install matplotlib-inline
conda run -n wilor pip install ultralytics==8.3.217
conda run -n wilor pip install numpy==1.26.1
conda run -n wilor pip install pytorch-lightning==1.9.5

cat ../../../patches/WiLoR/demo.py > utils/extraction/WiLoR/demo.py
