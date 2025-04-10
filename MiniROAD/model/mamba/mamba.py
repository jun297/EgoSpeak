import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.feature_sizes import FEATURE_SIZES
from mamba_ssm import Mamba
# from mambapy.mamba import Mamba, MambaConfig
from model.model_builder import META_ARCHITECTURES
from utils.util import parse_flow_type, parse_rgb_type


@META_ARCHITECTURES.register("Mamba")
class OADMamba(nn.Module):
    
    def __init__(self, cfg):
        super(OADMamba, self).__init__()
        self.use_flow = not cfg['no_flow']
        self.use_rgb = not cfg['no_rgb']
        
        self.input_dim = 0
        
        rgb_type = parse_rgb_type(cfg['rgb_type'])
        flow_type = parse_flow_type(cfg['flow_type'])
        
        if self.use_rgb:
            self.input_dim += FEATURE_SIZES[rgb_type]
        if self.use_flow:
            self.input_dim += FEATURE_SIZES[flow_type]

        self.hidden_dim = cfg['hidden_dim']
        self.num_layers = cfg['num_layers']
        
        # raise NotImplementedError("The following code is not implemented yet")
        self.out_dim = cfg['num_classes']
        self.window_size = cfg['window_size']
        
        self.relu = nn.ReLU()
        
        self.embedding_dim = cfg['embedding_dim']
        self.d_state = cfg['d_state']
        self.d_conv = cfg['d_conv']
        self.expand = cfg['expand']

        # self.mamba_config = MambaConfig(d_model=self.embedding_dim, n_layers=self.num_layers)
        # self.mamba = Mamba(self.mamba_config)
        self.mamba = Mamba(
            d_model=self.embedding_dim, # Model dimension d_model
            d_state=self.d_state,  # SSM state expansion factor
            d_conv=self.d_conv,    # Local convolution width
            expand=self.expand,    # Block expansion factor
        )
        
        # self.gru = nn.GRU(self.embedding_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.layer1 = nn.Sequential(
            nn.Linear(self.input_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(p=cfg['dropout']),
        )
        self.f_classification = nn.Sequential(
            nn.Linear(self.hidden_dim, self.out_dim)
        )
        # self.h0 = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.h0 = torch.zeros(self.num_layers, 1, self.hidden_dim)

    def forward(self, rgb_input, flow_input):
        x = torch.empty(0, device='cuda:0')
        if self.use_rgb:
            x = torch.cat((x, rgb_input), dim=2)  # Concatenating along the channel dimension
        if self.use_flow:
            x = torch.cat((x, flow_input), dim=2)
        
        x = self.layer1(x)
        ht = self.mamba(x)
        
        logits = self.f_classification(ht)
        out_dict = {}
        if self.training:
            out_dict['logits'] = logits
        else:
            pred_scores = F.softmax(logits, dim=-1)
            out_dict['logits'] = pred_scores
        return out_dict

@META_ARCHITECTURES.register("MambaA")
class OADMambaA(nn.Module):
    
    def __init__(self, cfg):
        super(OADMambaA, self).__init__()
        self.use_flow = not cfg['no_flow']
        self.use_rgb = not cfg['no_rgb']
        self.input_dim = 0
        if self.use_rgb:
            self.input_dim += FEATURE_SIZES[cfg['rgb_type']]
        if self.use_flow:
            self.input_dim += FEATURE_SIZES[cfg['flow_type']]

        self.hidden_dim = cfg['hidden_dim']
        self.num_layers = cfg['num_layers']
        self.anticipation_length = cfg["anticipation_length"]
        self.out_dim = cfg['num_classes']
        
        self.embedding_dim = cfg['embedding_dim']
        self.d_state = cfg['d_state'] if 'd_state' in cfg else 16
        self.d_conv = cfg['d_conv'] if 'd_conv' in cfg else 4
        self.expand = cfg['expand'] if 'expand' in cfg else 2
        
        self.layer1 = nn.Sequential(
            nn.Linear(self.input_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(p=cfg['dropout'])
        )
        self.actionness = cfg['actionness']
        if self.actionness:
            self.f_actionness = nn.Sequential(
                nn.Linear(self.hidden_dim, 1),
            )
        self.relu = nn.ReLU()
        # self.mamba_config = MambaConfig(d_model=self.embedding_dim, n_layers=self.num_layers)
        # self.mamba = Mamba(self.mamba_config)
        self.mamba = Mamba(
            d_model=self.embedding_dim, # Model dimension d_model
            d_state=self.d_state,  # SSM state expansion factor
            d_conv=self.d_conv,    # Local convolution width
            expand=self.expand,    # Block expansion facto
        )
        
        # self.gru = nn.GRU(self.embedding_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.f_classification = nn.Sequential(
            nn.Linear(self.embedding_dim, self.out_dim)
        )
        self.anticipation_layer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.anticipation_length*self.embedding_dim),
        )
        self.cfg = cfg

    def forward(self, rgb_input, flow_input):
        x = torch.empty(0, device='cuda:0')
        if self.use_rgb:
            # [64, 128, 2048]
            x = torch.cat((x, rgb_input), dim=2)  # Concatenating along the channel dimension
        if self.use_flow:
            # [64, 128, 5120]
            x = torch.cat((x, flow_input), dim=2)

        # audio channel is too huge to directly concat...
        # to c

        # B, L, D: mamba
        B, S, feat_dim = x.shape # batch, length, dim
        # 
        x = self.layer1(x)
        ht = self.mamba(x)
        # h0 = torch.zeros(1, B, self.hidden_dim).to(x.device)
        # ht, _ = self.gru(x, h0)
        logits = self.f_classification(self.relu(ht))
        anticipation_ht = self.anticipation_layer(self.relu(ht)).view(B, S, self.anticipation_length, self.embedding_dim)
        anticipation_logits = self.f_classification(self.relu(anticipation_ht))
        
        out_dict = {}
        if self.training:
            out_dict['logits'] = logits
            out_dict['anticipation_logits'] = anticipation_logits
        else:
            pred_scores = F.softmax(logits, dim=-1)
            pred_anticipation_scores = F.softmax(anticipation_logits, dim=-1)
            out_dict['logits'] = pred_scores
            out_dict['anticipation_logits'] = pred_anticipation_scores

        return out_dict