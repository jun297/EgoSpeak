import gc
import json
import os.path as osp
import sys

import numpy as np
import torch
import torch.utils.data as data
from configs.feature_sizes import FEATURE_SIZES
from datasets.dataset_builder import DATA_LAYERS
from utils.util import parse_flow_type, parse_rgb_type


@DATA_LAYERS.register("THUMOS")
@DATA_LAYERS.register("TVSERIES")
@DATA_LAYERS.register("EasyCom")
@DATA_LAYERS.register("EGO4D")
@DATA_LAYERS.register("IEMOCAP")
@DATA_LAYERS.register("IEMOCAP_DEBUG")
@DATA_LAYERS.register("YTConv")
class THUMOSDataset(data.Dataset):
    
    def __init__(self, cfg, mode='train'):
        self.root_path = cfg['root_path']
        self.mode = mode
        self.training = mode == 'train'
        self.window_size = cfg['window_size']
        self.enc_steps = cfg['window_size'] if 'window_size' in cfg else None
        self.dec_steps = cfg['query_num'] if 'query_num' in cfg else None
        
        self.stride = cfg['stride']
        data_name = cfg['data_name']
        self.model = cfg['model']
        self.vids = json.load(open(cfg['video_list_path']))[data_name][mode + '_session_set']                      
        self.num_classes = cfg['num_classes']
        self.inputs = []

        self._load_features(cfg)
        self._init_features()
        
    def _load_features(self, cfg):
        self.annotation_type = cfg['annotation_type']
        self.rgb_type = cfg['rgb_type']
        self.flow_type = cfg['flow_type']
        self.target_all = {}
        self.rgb_inputs = {}
        self.flow_inputs = {}
        dummy_target = np.zeros((self.window_size-1, self.num_classes))
        
        rgb_type = parse_rgb_type(cfg['rgb_type'])
        flow_type = parse_flow_type(cfg['flow_type'])
        
        dummy_rgb = np.zeros((self.window_size-1, FEATURE_SIZES[rgb_type]))
        dummy_flow = np.zeros((self.window_size-1, FEATURE_SIZES[flow_type]))
        for vid in self.vids:
            target = np.load(osp.join(self.root_path, self.annotation_type, vid + '.npy'))
            rgb = np.load(osp.join(self.root_path, self.rgb_type, vid + '.npy'))
            flow = np.load(osp.join(self.root_path, self.flow_type, vid + '.npy'))
                                                  
            if self.training:
                self.target_all[vid] = np.concatenate((dummy_target, target), axis=0)
                self.rgb_inputs[vid] = np.concatenate((dummy_rgb, rgb), axis=0)
                self.flow_inputs[vid] = np.concatenate((dummy_flow, flow), axis=0)
            else:
                self.target_all[vid] = target
                self.rgb_inputs[vid] = rgb
                self.flow_inputs[vid] = flow
    
    def _init_features(self):
        del self.inputs
        gc.collect()
        self.inputs = []
        if self.model == 'MiniROAD':
            for vid in self.vids:
                target = self.target_all[vid]
                                                         
                
                if self.training:
                    seed = np.random.randint(self.stride)
                    for start, end in zip(range(seed, target.shape[0], self.stride), 
                        range(seed + self.window_size, target.shape[0]+1, self.stride)):
                        self.inputs.append([
                            vid, start, end, target[start:end]
                        ])

                else:
                    start = 0
                    end = target.shape[0]
                    self.inputs.append([
                        vid, start, end, target[start:end]
                    ])
                    
        elif self.model == 'OADTR':
            for vid in self.vids:
                target = self.target_all[vid]
                                                   
                                                                       
                seed = np.random.randint(self.stride) if self.training else 0
                                                   
                for start, end in zip(
                    range(seed, target.shape[0], 1), 
                    range(seed + self.window_size, target.shape[0]+1, 1)
                    ):
                    enc_target = target[start:end]
                    dec_target = target[end:end + self.dec_steps]
                    distance_target, class_h_target = self.get_distance_target(target[start:end])
                    
                    self.inputs.append([
                        vid, start, end, enc_target, distance_target, class_h_target, dec_target
                    ])
            
        else:               
            for vid in self.vids:
                target = self.target_all[vid]
                seed = np.random.randint(self.stride) if self.training else 0
                for start, end in zip(
                    range(seed, target.shape[0], 1), 
                    range(seed + self.window_size, target.shape[0]+1, 1)
                    ):
                    self.inputs.append([
                        vid, start, end, target[start:end]
                    ])
    
    def get_distance_target(self, target_vector):
        target_matrix = np.zeros(self.enc_steps - 1)
        target_argmax = target_vector[self.enc_steps-1].argmax()
        for i in range(self.enc_steps - 1):
            if target_vector[i].argmax() == target_argmax:
                target_matrix[i] = 1.
        return target_matrix, target_vector[self.enc_steps-1]
    
    def __getitem__(self, index):
        if self.model == 'MiniROAD' or self.model == 'Transformer':
            vid, start, end, target = self.inputs[index]
            rgb_input = self.rgb_inputs[vid][start:end]
            flow_input = self.flow_inputs[vid][start:end]
            rgb_input = torch.tensor(rgb_input.astype(np.float32))
            flow_input = torch.tensor(flow_input.astype(np.float32))
            target = torch.tensor(target.astype(np.float32))
            return vid, rgb_input, flow_input, target
        
        elif self.model == 'OADTR':
            vid, start, end, enc_target, distance_target, class_h_target, dec_target = self.inputs[index]
            rgb_input = self.rgb_inputs[vid][start:end]
            flow_input = self.flow_inputs[vid][start:end]
            rgb_input = torch.tensor(rgb_input.astype(np.float32))
            flow_input = torch.tensor(flow_input.astype(np.float32))
            enc_target = torch.tensor(enc_target.astype(np.float32))
            distance_target = torch.tensor(distance_target.astype(np.float32))
            class_h_target = torch.tensor(class_h_target.astype(np.float32))
            dec_target = torch.tensor(dec_target.astype(np.float32))
            return vid, rgb_input, flow_input, enc_target, distance_target, class_h_target, dec_target

    def __len__(self):
        return len(self.inputs)
    

@DATA_LAYERS.register("THUMOS_ANTICIPATION")
@DATA_LAYERS.register("TVSERIES_ANTICIPATION")
@DATA_LAYERS.register("IEMOCAP_ANTICIPATION")
@DATA_LAYERS.register("EasyCom_ANTICIPATION")
@DATA_LAYERS.register("YTConv_ANTICIPATION")
@DATA_LAYERS.register("EGO4D_ANTICIPATION")
class THUMOSDataset(data.Dataset):
    
    def __init__(self, cfg, mode='train'):
        self.cfg = cfg
        self.root_path = cfg['root_path']
        self.mode = mode
        self.training = mode == 'train'
        self.window_size = cfg['window_size']
        self.stride = cfg['stride']
        self.anticipation_length = cfg['anticipation_length']
        data_name = cfg["data_name"].split('_')[0]
        if 'data_info_name' in cfg.keys() and cfg['data_info_name'] is not None:
            data_name = cfg['data_info_name']
        
        self.vids = json.load(open(cfg['video_list_path']))[data_name][mode + '_session_set']                      
        print(f'data_name: {data_name}')
        print(f'{mode} session set length: {len(self.vids)}')
        self.num_classes = cfg['num_classes']
        self.inputs = []
        
        self._load_features(cfg)
        self._init_features()
        
    def _load_features(self, cfg):
        self.annotation_type = cfg['annotation_type']
        self.rgb_type = cfg['rgb_type']
        self.flow_type = cfg['flow_type']
        
        self.target_all = {}
        self.rgb_inputs = {}
        self.flow_inputs = {}
        
        dummy_target = np.zeros((self.window_size-1, self.num_classes))
        dummy_rgb = np.zeros((self.window_size-1, FEATURE_SIZES[cfg['rgb_type']]))
        dummy_flow = np.zeros((self.window_size-1, FEATURE_SIZES[cfg['flow_type']]))
        
        for vid in self.vids:
            target = np.load(osp.join(self.root_path, self.annotation_type, vid + '.npy'))
            rgb = np.load(osp.join(self.root_path, self.rgb_type, vid + '.npy'))
            flow = np.load(osp.join(self.root_path, self.flow_type, vid + '.npy'))
            
            if self.training:
                self.target_all[vid] = np.concatenate((dummy_target, target), axis=0)
                self.rgb_inputs[vid] = np.concatenate((dummy_rgb, rgb), axis=0)
                self.flow_inputs[vid] = np.concatenate((dummy_flow, flow), axis=0)
            else:                      
                self.target_all[vid] = target
                self.rgb_inputs[vid] = rgb
                self.flow_inputs[vid] = flow
        
    def _init_features(self):
        del self.inputs
        gc.collect()
        self.inputs = []

        for vid in self.vids:
            target = self.target_all[vid]
            if self.training:
                seed = np.random.randint(self.stride)
                for start, end in zip(range(seed, target.shape[0], self.stride), 
                    range(seed + self.window_size, target.shape[0]-self.anticipation_length, self.stride)):
                    self.inputs.append([
                        vid, start, end, target[start:end], target[end:end+self.anticipation_length]
                    ])
            else:
                start = 0
                end = target.shape[0] - self.anticipation_length
                ant_target = []                                                                     
                                                                                                        
                for s in range(1, target.shape[0] - self.anticipation_length + 1):
                    ant_target.append(target[s:s + self.anticipation_length])
                
                self.inputs.append([
                    vid, start, end, target[start:end], np.array(ant_target)
                ])
                
                                                
    
    def __getitem__(self, index):
        vid, start, end, target, ant_target = self.inputs[index]
                                                  
        
        rgb_input = self.rgb_inputs[vid][start:end]
        flow_input = self.flow_inputs[vid][start:end]
        
        rgb_input = torch.tensor(rgb_input.astype(np.float32))
        flow_input = torch.tensor(flow_input.astype(np.float32))
        
        target = torch.tensor(target.astype(np.float32))
        ant_target = torch.tensor(ant_target.astype(np.float32))
        
        return vid, rgb_input, flow_input, target, ant_target

    def __len__(self):
        return len(self.inputs)                          
    
    
@DATA_LAYERS.register("FINEACTION")
class FINEACTIONDataset(data.Dataset):
    
    def __init__(self, cfg, mode='train'):
        self.root_path = cfg['root_path']
        self.mode = mode
        self.training = mode == 'train'
        self.window_size = cfg['window_size']
        self.stride = cfg['stride']
        data_name = cfg['data_name']
        self.vids = json.load(open(cfg['video_list_path']))[data_name][mode + '_session_set']                      
        self.num_classes = cfg['num_classes']
        self.inputs = []
        self._load_features(cfg)
        self._init_features()

    def _load_features(self, cfg):
        self.annotation_type = cfg['annotation_type']
        self.rgb_type = cfg['rgb_type']
        self.flow_type = cfg['flow_type']
        
    def _init_features(self, seed=0):
                          
        del self.inputs
        gc.collect()
        self.inputs = []
        for vid in self.vids:
            target = np.load(osp.join(self.root_path, self.annotation_type, vid + '.npy'))
            if self.training:
                seed = np.random.randint(self.stride)
                for start, end in zip(range(seed, target.shape[0], self.stride), 
                    range(seed + self.window_size, target.shape[0]+1, self.stride)):
                    self.inputs.append([
                        vid, start, end
                    ])
            else:
                start = 0
                end = target.shape[0]
                self.inputs.append([
                    vid, start, end
                ])

    def __getitem__(self, index):
        vid, start, end = self.inputs[index]
        rgb_input = np.load(osp.join(self.root_path, self.rgb_type, vid + '.npy'), mmap_mode='r')[start:end]
        flow_input = np.load(osp.join(self.root_path, self.flow_type, vid + '.npy'), mmap_mode='r')[start:end]
        target = np.load(osp.join(self.root_path, self.annotation_type, vid + '.npy'), mmap_mode='r')[start:end]
        rgb_input = torch.tensor(rgb_input.astype(np.float32))
        flow_input = torch.tensor(flow_input.astype(np.float32))
        target = torch.tensor(target.astype(np.float32))
        return rgb_input, flow_input, target

    def __len__(self):
        return len(self.inputs)    