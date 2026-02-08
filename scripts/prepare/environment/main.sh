#!/usr/bin/env bash
set -x

ALGORITHM=$1
case $ALGORITHM in
  "smplerx")
    echo "Preparing SMPLer-X environment for inference"
    bash scripts/prepare/environment/SMPLer-X.sh 
    ;;
  "smplestx")
    echo "Preparing SMPLest-X environment for inference" 
    ;;
  "osx")
    echo "Preparing OSX environment for inference" 
    bash scripts/prepare/environment/OSX.sh 
    ;;
  "wilor")
    echo "Preparing WiLoR environment for inference" 
    bash scripts/prepare/environment/WiLoR.sh
    ;;
  *)
    echo "Unsupported algorithm: $ALGORITHM"
    exit 1
    ;;
esac