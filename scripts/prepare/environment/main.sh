#!/usr/bin/env bash
set -x

ALGORITHM=$1
case $ALGORITHM in
  "SMPLer-X")
    echo "Preparing SMPLer-X environment for inference"
    bash scripts/prepare/SMPLer-X.sh 
    ;;
  "SMPLest-X")
    echo "Preparing SMPLest-X environment for inference" 
    ;;
  "OSX")
    echo "Preparing OSX environment for inference" 
    bash scripts/prepare/OSX.sh 
    ;;
  "WiLoR")
    echo "Preparing WiLoR environment for inference" 
    bash scripts/prepare/WiLoR.sh
    ;;
  *)
    echo "Unsupported algorithm: $ALGORITHM"
    exit 1
    ;;
esac