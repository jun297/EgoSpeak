#!/bin/bash

PATH_TO_CONFIG_FILE=configs/EasyCom/LSTR/lstr_aa_easycom.yaml
shift  # Remove the first argument so $@ contains all other arguments

echo "Config file: $PATH_TO_CONFIG_FILE"
echo "Additional options: $@"
# EXAMPLE: 
# USE_WANDB True
# INPUT.MODALITY "visual+motion+object"

echo "---"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Config file: $PATH_TO_CONFIG_FILE"

echo python tools/train_net.py \
  --config_file $PATH_TO_CONFIG_FILE \
  --gpu $CUDA_VISIBLE_DEVICES \
  $@

python tools/train_net.py \
  --config_file $PATH_TO_CONFIG_FILE \
  --gpu $CUDA_VISIBLE_DEVICES \
  $@ 
  
