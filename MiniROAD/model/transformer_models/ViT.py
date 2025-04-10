import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.feature_sizes import FEATURE_SIZES
from model.model_builder import META_ARCHITECTURES as registry

from .attn import AttentionLayer, FullAttention, ProbAttention
from .decoder import Decoder, DecoderLayer
from .PositionalEncoding import (FixedPositionalEncoding,
                                 LearnedPositionalEncoding)
from .Transformer import TransformerModel


# @registry.register('TRANSFORMER')
@registry.register('OADTR')
class OADTR(nn.Module):
    def __init__(self, cfg, use_representation=True,
        conv_patch_representation=False,
        positional_encoding_type="learned"
    ):
        super(OADTR, self).__init__()
        
        self.img_dim = cfg["window_size"] # img_dim
        self.out_dim = cfg["num_classes"]
        
        self.patch_dim = cfg["patch_dim"]
        self.query_num = cfg["query_num"]
        
        self.embedding_dim = cfg["embedding_dim"]
        self.num_heads = cfg["num_heads"]
        self.num_layers = cfg["num_layers"]
        self.hidden_dim = cfg["hidden_dim"]
        
        self.dropout_rate = cfg["dropout"]
        self.attn_dropout_rate = cfg["attn_dropout_rate"]
        
        self.conv_patch_representation = conv_patch_representation
        
        self.decoder_embedding_dim = cfg["decoder_embedding_dim"]
        self.decoder_num_heads = cfg["decoder_num_heads"]
        self.decoder_layers = cfg["decoder_layers"]
        self.decoder_embedding_dim_out = cfg["decoder_embedding_dim_out"]
        self.decoder_attn_dropout_rate = cfg["decoder_attn_dropout_rate"]
        
        self.use_flow = not cfg['no_flow']
        self.use_rgb = not cfg['no_rgb']
        
        self.num_channels = 0
        if self.use_rgb:
            self.num_channels += FEATURE_SIZES[cfg['rgb_type']]
        if self.use_flow:
            self.num_channels += FEATURE_SIZES[cfg['flow_type']]
        
        assert self.embedding_dim % self.num_heads == 0
        assert self.img_dim % self.patch_dim == 0
        
        self.num_patches = int(self.img_dim // self.patch_dim)
        self.seq_length = self.num_patches + 1
        # self.seq_length = self.num_patches + self.img_dim
        self.flatten_dim = self.patch_dim * self.patch_dim * self.num_channels
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))
        # self.cls_token = nn.Parameter(torch.zeros(1, self.img_dim, self.embedding_dim))

        self.linear_encoding = nn.Linear(self.flatten_dim, self.embedding_dim)
        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )
        elif positional_encoding_type == "fixed":
            self.position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )
        print('position encoding :', positional_encoding_type)

        self.pe_dropout = nn.Dropout(p=self.dropout_rate)

        self.encoder = TransformerModel(
            self.embedding_dim,
            self.num_layers,
            self.num_heads,
            self.hidden_dim,
            self.dropout_rate,
            self.attn_dropout_rate,
        )
        self.pre_head_ln = nn.LayerNorm(self.embedding_dim)

        d_model = self.decoder_embedding_dim
        use_representation = False  # False
        if use_representation:
            self.mlp_head = nn.Sequential(
                nn.Linear(self.embedding_dim , self.hidden_dim//2),
                # nn.Tanh(),
                nn.ReLU(),
                nn.Linear(self.hidden_dim//2, self.out_dim),
            )
        else:
            self.mlp_head = nn.Linear(self.embedding_dim + d_model, self.out_dim)

        if self.conv_patch_representation:
            # self.conv_x = nn.Conv2d(
            #     self.num_channels,
            #     self.embedding_dim,
            #     kernel_size=(self.patch_dim, self.patch_dim),
            #     stride=(self.patch_dim, self.patch_dim),
            #     padding=self._get_padding(
            #         'VALID', (self.patch_dim, self.patch_dim),
            #     ),
            # )
            self.conv_x = nn.Conv1d(
                self.num_channels,
                self.embedding_dim,
                kernel_size=self.patch_dim,
                stride=self.patch_dim,
                padding=self._get_padding(
                    'VALID',  (self.patch_dim),
                ),
            )
        else:
            self.conv_x = None

        self.to_cls_token = nn.Identity()
        
        # Decoder
        factor = 1  # 5
        dropout = self.decoder_attn_dropout_rate
        # d_model = self.decoder_embedding_dim
        n_heads = self.decoder_num_heads
        d_layers = self.decoder_layers
        d_ff = self.decoder_embedding_dim_out  # args.decoder_embedding_dim_out or 4*args.decoder_embedding_dim None
        activation = 'gelu'  # 'gelu'
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(FullAttention(True, factor, attention_dropout=dropout),  # True
                                   d_model, n_heads),  # ProbAttention  FullAttention
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout),  # False
                                   d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.decoder_cls_token = nn.Parameter(torch.zeros(1, self.query_num, d_model))
        if positional_encoding_type == "learned":
            self.decoder_position_encoding = LearnedPositionalEncoding(
                self.query_num, self.embedding_dim, self.query_num
            )
        elif positional_encoding_type == "fixed":
            self.decoder_position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )
        print('position decoding :', positional_encoding_type)
        self.classifier = nn.Linear(d_model, self.out_dim)
        self.after_dropout = nn.Dropout(p=self.dropout_rate)
        # self.merge_fc = nn.Linear(d_model, 1)
        # self.merge_sigmoid = nn.Sigmoid()


    def forward(self, sequence_input_rgb, sequence_input_flow):
        if self.use_rgb and self.use_flow:
            x = torch.cat((sequence_input_rgb, sequence_input_flow), 2)
        elif self.use_rgb:
            x = sequence_input_rgb
        elif self.use_flow:
            x = sequence_input_flow

        x = self.linear_encoding(x) # B, 128, 1204
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1) # B, 1, 1024
        # x = torch.cat((cls_tokens, x), dim=1)
        x = torch.cat((x, cls_tokens), dim=1) # B, seq+1, 1024
        try:
            x = self.position_encoding(x) # B, seq+1, 1024
        except:
            import pdb; pdb.set_trace()
        x = self.pe_dropout(x)   # not delete
        
        # apply transformer
        x = self.encoder(x)
        x = self.pre_head_ln(x)  # B, seq+1, 1024

        # decoder
        decoder_cls_token = self.decoder_cls_token.expand(x.shape[0], -1, -1) # [B, 8, 1024]
        # decoder_cls_token = self.after_dropout(decoder_cls_token)  # add
        # decoder_cls_token = self.decoder_position_encoding(decoder_cls_token)  # [128, 8, 1024]
        dec = self.decoder(decoder_cls_token, x)   # [128, 8, 1024]
        dec = self.after_dropout(dec)  # add
        
        # merge_atte = self.merge_sigmoid(self.merge_fc(dec))  # [128, 8, 1]
        # dec_for_token = (merge_atte*dec).sum(dim=1)  # [128, 1024]
        # dec_for_token = (merge_atte*dec).sum(dim=1)/(merge_atte.sum(dim=-2) + 0.0001)
        # *** RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x2048 and 1024x2)
        dec_for_token = dec.mean(dim=1) # [64, 1024]
        
        # dec_for_token = dec.max(dim=1)[0]
        dec_cls_out = self.classifier(dec) # [B, 8, NUM_CLASS?]
        # set_trace()
        # x = self.to_cls_token(x[:, 0])
        
        x = torch.cat((self.to_cls_token(x[:, -1]), dec_for_token), dim=1) # [B, 2048]
        x = self.mlp_head(x) # [B, NUM_CLASSES]
        
        # original OADTR
        # return x, dec_cls_out
        
        # x = F.log_softmax(x, dim=-1)
        out_dict = {}
        out_dict['logits'] = x.unsqueeze(1) # [B, 1, NUM_CLASSES] ex) [64, 1, 2]
        out_dict['dec_logits'] = dec_cls_out # [B, 8, NUM_CLASSES] ex) [64, 8, 2]
        
        # out_dict['logits'] = x #x.unsqueeze(1)
        return out_dict

    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)

@registry.register('Transformer')
class ViTEnc(nn.Module):
    def __init__(self, cfg, use_representation=True,
        conv_patch_representation=False,
        positional_encoding_type="learned"
    ):
        super(ViTEnc, self).__init__()
        self.img_dim = cfg["window_size"]
        self.out_dim = cfg["num_classes"]
        self.embedding_dim = cfg["embedding_dim"]
        self.patch_dim = cfg["patch_dim"]
        self.num_heads = cfg["num_heads"]
        self.num_layers = cfg["num_layers"]
        self.hidden_dim = cfg["hidden_dim"]
        self.dropout_rate = cfg["dropout"]
        self.use_flow = not cfg['no_flow']
        self.use_rgb = not cfg['no_rgb']
        self.num_channels= 0
        if self.use_rgb:
            self.num_channels += FEATURE_SIZES[cfg['rgb_type']]
        if self.use_flow:
            self.num_channels += FEATURE_SIZES[cfg['flow_type']]
        self.attn_dropout_rate = cfg["attn_dropout_rate"]
        assert self.embedding_dim % self.num_heads == 0
        assert self.img_dim % self.patch_dim == 0
        self.conv_patch_representation = conv_patch_representation
        self.num_patches = int(self.img_dim // self.patch_dim)
        self.seq_length = self.num_patches + 1
        # self.seq_length = self.num_patches + self.img_dim
        self.flatten_dim = self.patch_dim * self.patch_dim * self.num_channels
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))
        # self.cls_token = nn.Parameter(torch.zeros(1, self.img_dim, self.embedding_dim))

        self.linear_encoding = nn.Linear(self.flatten_dim, self.embedding_dim)
        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )
        elif positional_encoding_type == "fixed":
            self.position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )
        print('position encoding :', positional_encoding_type)

        self.pe_dropout = nn.Dropout(p=self.dropout_rate)

        self.encoder = TransformerModel(
            self.embedding_dim,
            self.num_layers,
            self.num_heads,
            self.hidden_dim,
            self.dropout_rate,
            self.attn_dropout_rate,
        )
        self.pre_head_ln = nn.LayerNorm(self.embedding_dim)

        use_representation = False  # False
        if use_representation:
            self.mlp_head = nn.Sequential(
                nn.Linear(self.embedding_dim , self.hidden_dim//2),
                # nn.Tanh(),
                nn.ReLU(),
                nn.Linear(self.hidden_dim//2, self.out_dim),
            )
        else:
            self.mlp_head = nn.Linear(self.embedding_dim, self.out_dim)

        if self.conv_patch_representation:
            # self.conv_x = nn.Conv2d(
            #     self.num_channels,
            #     self.embedding_dim,
            #     kernel_size=(self.patch_dim, self.patch_dim),
            #     stride=(self.patch_dim, self.patch_dim),
            #     padding=self._get_padding(
            #         'VALID', (self.patch_dim, self.patch_dim),
            #     ),
            # )
            self.conv_x = nn.Conv1d(
                self.num_channels,
                self.embedding_dim,
                kernel_size=self.patch_dim,
                stride=self.patch_dim,
                padding=self._get_padding(
                    'VALID',  (self.patch_dim),
                ),
            )
        else:
            self.conv_x = None

        self.to_cls_token = nn.Identity()


    def forward(self, sequence_input_rgb, sequence_input_flow):
        if self.use_rgb and self.use_flow:
            x = torch.cat((sequence_input_rgb, sequence_input_flow), 2)
        elif self.use_rgb:
            x = sequence_input_rgb
        elif self.use_flow:
            x = sequence_input_flow

        x = self.linear_encoding(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1) # B, 1, 1024
        # x = torch.cat((cls_tokens, x), dim=1)
        x = torch.cat((x, cls_tokens), dim=1) # B, seq+1, 1024
        
        x = self.position_encoding(x) # B, seq+1, 1024
        x = self.pe_dropout(x)   # not delete

        # apply transformer
        x = self.encoder(x)
        x = self.pre_head_ln(x)  # B, seq+1, 1024

        x = self.to_cls_token(x[:, 0]) # B, 1024
        # x = self.to_cls_token(x[:,0:self.img_dim]) # B, 1024
        x = self.mlp_head(x)
        # x = F.log_softmax(x, dim=-1)
        out_dict = {}
        out_dict['logits'] = x.unsqueeze(1)
        # out_dict['logits'] = x #x.unsqueeze(1)
        return out_dict

    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)
