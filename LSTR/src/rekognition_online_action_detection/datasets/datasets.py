# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
                                     

__all__ = [
    'build_dataset',
    'build_data_loader',
]

import torch.utils.data as data
from rekognition_online_action_detection.utils.registry import Registry

DATA_LAYERS = Registry()


def build_dataset(cfg, phase, tag=''):
    data_names = ['YTConv', 'EGO4D']
    data_name = next((name for name in data_names if name in cfg.DATA.DATA_NAME), cfg.DATA.DATA_NAME)
    
    data_layer = DATA_LAYERS[cfg.MODEL.MODEL_NAME + tag + data_name]
    return data_layer(cfg, phase)


def build_data_loader(cfg, phase):
    data_loader = data.DataLoader(
        dataset=build_dataset(cfg, phase),
        batch_size=cfg.DATA_LOADER.BATCH_SIZE,
        shuffle=True if phase == 'train' else False,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
    )
    return data_loader
