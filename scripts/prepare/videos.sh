#!/usr/bin/env bash

VIDEOS_URL=$1

PYTHONPATH=../:$PYTHONPATH
python main/prepare.py $VIDEOS_URL