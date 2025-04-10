import pickle
import os
import os.path as osp
import random
import numpy as np
import torch

def dump_pickle(lst, file_path, file_name):
    with open(osp.join(file_path, file_name + '.pkl'), 'wb') as f:
        pickle.dump(lst, f)

def create_dir(dir_path):
    if not osp.exists(dir_path):
        os.makedirs(dir_path)

def create_outdir(result_path, overwrite=False):
    i = 1
    new_result_path = result_path
    
    if overwrite == False:
        while osp.exists(new_result_path):
            new_result_path = f'{result_path}_{i}'
            i += 1
    
    create_dir(osp.join(new_result_path, 'ckpts'))
    create_dir(osp.join(new_result_path, 'runs'))
    return new_result_path

def set_seed(seed):
    # os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def parse_rgb_type(rgb_type):    
    if "rgb_kinetics_resnet50" in rgb_type:
        rgb_type = "rgb_kinetics_resnet50"
    elif "rgb_farl" in rgb_type:
        rgb_type = "rgb_farl"
    
    return rgb_type

def parse_flow_type(flow_type):
    if "audio_wav2vec2" in flow_type:
        flow_type = "audio_wav2vec2"
    
    return flow_type

def get_identifier(cfg):
    mode = "eval" if cfg["eval"] == True else "train"
    
    modality = ''
    if not cfg["no_flow"]:
        modality += 'A'
    if not cfg["no_rgb"]:
        modality += 'V'
    
    identifier = f'{cfg["model"]}_{cfg["data_name"]}'
    if 'A' in modality:
        identifier += f"_{cfg['flow_type']}"
    if 'V' in modality:
        identifier += f"_{cfg['rgb_type']}"
    
    identifier += f'_{modality}_epoch{cfg["num_epoch"]}_{mode}'
    if cfg["ckpt"] != None:
        identifier += f'_ckpt_{".".join(osp.basename(cfg["ckpt"]).split(".")[:-1])}'
    
    return identifier
    