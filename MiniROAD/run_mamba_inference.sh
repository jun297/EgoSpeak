PATH_TO_CONFIG_FILE=configs/EasyCom/mamba_aa_easycom.yaml
PATH_TO_CHECKPOINT_FILE=checkpoints/mamba_easycom_AV.pth
shift  # Remove the first argument so $@ contains all other arguments

echo "Running MiniROAD inference"
echo "Config file: $PATH_TO_CONFIG_FILE"
echo "Additional options: $@"

# get the basename of the config file, without the extension
CONFIG_FILENAME=$(basename -- "$PATH_TO_CONFIG_FILE")
CONFIG_FILENAME="${CONFIG_FILENAME%.*}"

# --no_flow (for no audio) --no_rgb --wandb
# Pass all remaining arguments directly to the Python script
python -u main.py --config "$PATH_TO_CONFIG_FILE" \
      --save-as-cfg \
      --eval \
      --ckpt "$PATH_TO_CHECKPOINT_FILE" \
      $@

echo "Finished MiniROAD inference"
echo "python -u main.py --config '$PATH_TO_CONFIG_FILE' \
      --save-as-cfg \
      --ckpt '$PATH_TO_CHECKPOINT_FILE' \
      --eval \
      $@"