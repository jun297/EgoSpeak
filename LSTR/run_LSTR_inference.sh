#!/bin/bash

PATH_TO_CONFIG_FILE=configs/EasyCom/LSTR/lstr_aa_easycom.yaml
PATH_TO_CHECKPOINT_FILE=checkpoints/lstr_easycom_AV.pth

echo "---"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Config file: $PATH_TO_CONFIG_FILE"
# echo "Checkpoint opt: $CHECKPOINT_OPT"
# echo "Epoch opt: $EPOCH_OPT"

echo python tools/test_net.py \
  --config_file $PATH_TO_CONFIG_FILE \
  --gpu $CUDA_VISIBLE_DEVICES \
  MODEL.CHECKPOINT $PATH_TO_CHECKPOINT_FILE \
  MODEL.LSTR.INFERENCE_MODE batch \
  $@

python tools/test_net.py \
  --config_file $PATH_TO_CONFIG_FILE \
  --gpu $CUDA_VISIBLE_DEVICES \
  MODEL.CHECKPOINT $PATH_TO_CHECKPOINT_FILE \
  MODEL.LSTR.INFERENCE_MODE batch \
  $@