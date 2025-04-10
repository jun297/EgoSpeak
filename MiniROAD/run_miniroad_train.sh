#!/bin/bash
PATH_TO_CONFIG_FILE=configs/EasyCom/miniroad_aa_easycom.yaml

echo "Running MiniROAD training"
echo "Config file: $PATH_TO_CONFIG_FILE"
echo "Additional options: $@"

# Additional options: --no_flow (for no audio) --no_rgb --wandb
# Pass all remaining arguments directly to the Python script
python -u main.py --config "$PATH_TO_CONFIG_FILE" --save-as-cfg $@

echo "Finished MiniROAD training"
echo "Run python -u main.py --config $PATH_TO_CONFIG_FILE --save-as-cfg $@"