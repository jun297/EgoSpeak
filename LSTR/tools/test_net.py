# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os

from rekognition_online_action_detection.engines import do_inference
from rekognition_online_action_detection.models import build_model
from rekognition_online_action_detection.utils.checkpointer import \
    setup_checkpointer
from rekognition_online_action_detection.utils.env import setup_environment
from rekognition_online_action_detection.utils.logger import setup_logger
from rekognition_online_action_detection.utils.parser import load_cfg
from utils import validate_cfg


def main(cfg):
    validate_cfg(cfg)
    config_name = os.path.basename(cfg.OUTPUT_DIR)
    
    print(f"cfg.MODEL.CHECKPOINT: {cfg.MODEL.CHECKPOINT}")
    print(f"cfg.OUTPUT_DIR: {cfg.OUTPUT_DIR}")
    
    # Setup configurations
    device = setup_environment(cfg)
    checkpointer = setup_checkpointer(cfg, phase='test')
    logger = setup_logger(cfg, phase='test')
    
    print(cfg)
    print(f"DATA.IGNORE_INDEX: {cfg.DATA.IGNORE_INDEX}")
    # Build model
    model = build_model(cfg, device)

    # Load pretrained model
    checkpointer.load(model)
    
    do_inference(
        cfg,
        model,
        device,
        logger,
    )


if __name__ == '__main__':
    main(load_cfg())
