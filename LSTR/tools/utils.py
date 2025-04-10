import os

import wandb


def parse_config(cfg):
    modality = get_modality(cfg)
    
    cfg_filename = os.path.basename(cfg.OUTPUT_DIR)
    
    os.environ["WANDB__SERVICE_WAIT"] = "3000"
    cfg_filename_abbr = cfg_filename.split('_1x_')[-1]
    wandb_postfix = generate_wandb_postfix(cfg, modality)
    scheduler_name = cfg.SOLVER.SCHEDULER.SCHEDULER_NAME
    learning_rate = cfg.SOLVER.BASE_LR
    weight_decay = cfg.SOLVER.WEIGHT_DECAY
    
    cfg_filename = cfg_filename.replace('long_512', f'long_{cfg.MODEL.LSTR.LONG_MEMORY_SECONDS}')
    cfg_filename = cfg_filename.replace('work_8', f'work_{cfg.MODEL.LSTR.WORK_MEMORY_SECONDS}')
    
    cfg.OUTPUT_DIR = os.path.join(os.path.dirname(cfg.OUTPUT_DIR), cfg_filename)
    
    if cfg.SAVE_HPARAM:
        # save different dir containing hparam
        hparam_info = '_'.join([f"lr_{learning_rate}", f"wd_{weight_decay}", f"schd_{scheduler_name}"])
        cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, hparam_info, modality)
        print(f"HPARAM: {hparam_info}")
        print(f"cfg.OUTPUT_DIR: {cfg.OUTPUT_DIR}")
            
    wandb_name = cfg_filename_abbr + '_' + wandb_postfix
    
    if cfg.SAVE_HPARAM:
        wandb_name += '_' + '_'.join([f"lr_{learning_rate}", f"wd_{weight_decay}", f"schd_{scheduler_name}"])

    if cfg.USE_WANDB:
        print("*** init wandb ***")
        print(f"wandb_name: {wandb_name}")

def get_modality(cfg):
    modality = ''
    if 'motion' in cfg.INPUT.MODALITY:
        modality += 'A'
    if 'visual' in cfg.INPUT.MODALITY:
        modality += 'V'
    if 'object' in cfg.INPUT.MODALITY:
        modality += 'T'
    elif cfg['INPUT']['MODALITY'] == 'twostream':
        modality = 'AV'
    
    if modality == '':
        raise ValueError(f"Unknown modality: {cfg['INPUT']['MODALITY']}")    
    
    return modality


def generate_wandb_postfix(cfg, modality):
    visual_feature_type = '_'.join(cfg['INPUT']['VISUAL_FEATURE'].split('_')[1:])
    audio_feature_type = '_'.join(cfg['INPUT']['MOTION_FEATURE'].split('_')[1:])
    text_feature_type = '_'.join(cfg['INPUT']['OBJECT_FEATURE'].split('_')[1:])
    
    if cfg.MODEL.LSTR.LONG_MEMORY_USE_PE == True:
        model_name = 'LSTR_TESTRA'
    else:
        model_name = 'TESTRA'

    if '_aa' in cfg.OUTPUT_DIR:
        model_name += '_aa'
    
    name_parts = [
        model_name,
        cfg['DATA']['DATA_NAME'],
        modality,        
        # visual_feature_type,
        # audio_feature_type,
        # text_feature_type,
        # f"epoch_{cfg['SOLVER']['NUM_EPOCHS']}"
    ]
    if cfg['MODEL']['CHECKPOINT']:
        name_parts.append(f"ckpt_{os.path.basename(cfg['MODEL']['CHECKPOINT']).split('.')[0]}")
    return '_'.join(name_parts)
  
def get_wandb_name(cfg, cfg_filename, modality):
    cfg_filename_abbr = cfg_filename.split('_1x_')[-1]
    wandb_postfix = generate_wandb_postfix(cfg, modality)
    scheduler_name = cfg.SOLVER.SCHEDULER.SCHEDULER_NAME
    learning_rate = cfg.SOLVER.BASE_LR
    weight_decay = cfg.SOLVER.WEIGHT_DECAY
    wandb_name = cfg_filename_abbr + '_' + wandb_postfix + '_' + '_'.join([f"lr_{learning_rate}", f"wd_{weight_decay}", f"schd_{scheduler_name}"])
    return wandb_name

def validate_cfg(cfg):
    if cfg.DATA.NUM_CLASSES != len(cfg.DATA.CLASS_NAMES):
        print(f"len(cfg.DATA.CLASS_NAMES): {len(cfg.DATA.CLASS_NAMES)} is not equal to cfg.DATA.NUM_CLASSES: {cfg.DATA.NUM_CLASSES}")
        cfg.DATA.CLASS_NAMES = cfg.DATA.CLASS_NAMES[:cfg.DATA.NUM_CLASSES]
        print(f"Adjusted cfg.DATA.CLASS_NAMES: {cfg.DATA.CLASS_NAMES}")
        
    