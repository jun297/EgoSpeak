import torch
import torch.nn as nn
import torch.nn.functional as F
from criterions.loss_builder import CRITERIONS


@CRITERIONS.register('NONUNIFORM')
class OadLoss(nn.Module):
    
    def __init__(self, cfg, reduction='mean'):
        super(OadLoss, self).__init__()
        self.reduction = reduction
        self.num_classes = cfg['num_classes']
        self.loss_offset = cfg['loss_offset'] if 'loss_offset' in cfg else 0
        assert self.loss_offset==0, f'loss_offset is not 0: {self.loss_offset}'
        self.loss = self.end_loss

    def end_loss(self, out_dict, target):
        # logits: (B, seq, K) target: (B, seq, K)
        assert self.loss_offset < out_dict['logits'].shape[1], f'loss_offset is longer than the sequence loss_offset: {self.loss_offset}, logits.shape: {out_dict["logits"].shape}'
        logits = out_dict['logits']
        logits = logits[:,-1 - self.loss_offset,:].contiguous()
        target = target[:,-1 - self.loss_offset,:].contiguous()
        ce_loss = self.mlce_loss(logits, target)
        return ce_loss

    def mlce_loss(self, logits, target):
        '''
        multi label cross entropy loss. 
        logits: (B, K) target: (B, K) 
        '''
        logsoftmax = nn.LogSoftmax(dim=-1).to(logits.device)
        output = torch.sum(-F.normalize(target) * logsoftmax(logits), dim=1) # B
        if self.reduction == 'mean':
            loss = torch.mean(output)
        elif self.reduction == 'sum':
            loss = torch.sum(output)
        return loss

    def forward(self, out_dict, target): 
        return self.loss(out_dict, target)


@CRITERIONS.register('UNIFORM')
class OadUniformLoss(nn.Module):
    
    def __init__(self, cfg, reduction='mean'):
        super(OadUniformLoss, self).__init__()
        self.reduction = reduction
        self.num_classes = cfg['num_classes']
        # self.loss_offset = cfg['loss_offset'] if 'loss_offset' in cfg else 0
        self.loss = self.end_loss

    def end_loss(self, out_dict, target):
        # logits: (B, seq, K) target: (B, seq, K)
        logits = out_dict['logits']
        # logits = logits[:,-1 - self.loss_offset,:].contiguous()
        # target = target[:,-1 - self.loss_offset,:].contiguous()
        ce_loss = self.mlce_loss(logits, target)
        return ce_loss

    def mlce_loss(self, logits, target):
        '''
        multi label cross entropy loss. 
        logits: (B, K) target: (B, K) 
        '''
        logsoftmax = nn.LogSoftmax(dim=-1).to(logits.device)
        output = torch.sum(-target * logsoftmax(logits), dim=1) # B
        # output = torch.sum(-F.normalize(target) * logsoftmax(logits), dim=1) # B
        if self.reduction == 'mean':
            loss = torch.mean(output)
        elif self.reduction == 'sum':
            loss = torch.sum(output)
        return loss

    def forward(self, out_dict, target): 
        return self.loss(out_dict, target)

@CRITERIONS.register('OADTR')
class OadTRLoss(nn.Module):
    
    def __init__(self, cfg, reduction='mean'):
        super(OadTRLoss, self).__init__()
        self.reduction = reduction
        self.num_classes = cfg['num_classes']
        # self.loss_offset = cfg['loss_offset'] if 'loss_offset' in cfg else 0
        self.loss = self.end_loss

    def end_loss(self, out_dict, target):
        # logits: (B, seq, K) target: (B, seq, K)
        assert self.loss_offset < out_dict['logits'].shape[1], f'loss_offset is longer than the sequence loss_offset: {self.loss_offset}, logits.shape: {out_dict["logits"].shape}'
        logits = out_dict['logits']
        # logits = logits[:,-1 - self.loss_offset,:].contiguous()
        # target = target[:,-1 - self.loss_offset,:].contiguous()
        ce_loss = self.mlce_loss(logits, target)
        return ce_loss

    def mlce_loss(self, logits, target):
        '''
        multi label cross entropy loss. 
        logits: (B, K) target: (B, K) 
        '''
        logsoftmax = nn.LogSoftmax(dim=-1).to(logits.device)
        output = torch.sum(-F.normalize(target) * logsoftmax(logits), dim=1) # B
        if self.reduction == 'mean':
            loss = torch.mean(output)
        elif self.reduction == 'sum':
            loss = torch.sum(output)
        return loss

    def forward(self, out_dict, enc_target, dec_target): 
        return self.loss(out_dict, enc_target, dec_target)

@CRITERIONS.register('ANTICIPATION')
class OadAntLoss(nn.Module):
    
    # reduction was sum, not mean, what if we change to sum?
    def __init__(self, cfg, reduction='mean'):
        super(OadAntLoss, self).__init__()
        self.reduction = reduction
        self.loss = self.anticipation_loss
        self.num_classes = cfg['num_classes']
        
    def anticipation_loss(self, out_dict, target, ant_target):
        # anticipation_logits: (B, seq, ANTICIPATION_LENGTH, num_class)
        # ant_target: (B, ANTICIPATION_LENGTH, num_class)
        anticipation_logits = out_dict['anticipation_logits']
        # [B, window_size, anticipation_length, num_class]
        pred_anticipation_logits = anticipation_logits[:,-1,:,:].contiguous().view(-1, self.num_classes)
        
        anticipation_logit_targets = ant_target.view(-1, self.num_classes)
        ant_loss = self.mlce_loss(pred_anticipation_logits, anticipation_logit_targets)

        return ant_loss

    def ce_loss(self, out_dict, target):
        # logits: (B, seq, K) target: (B, seq, K)
        logits = out_dict['logits']
        logits = logits[:,-1,:].contiguous()
        target = target[:,-1,:].contiguous()
        ce_loss = self.mlce_loss(logits, target)
        return ce_loss

    def mlce_loss(self, logits, target):
        '''
        multi label cross entropy loss. 
        logits: (B, K) target: (B, K) 
        '''
        logsoftmax = nn.LogSoftmax(dim=-1).to(logits.device)
        output = torch.sum(-F.normalize(target) * logsoftmax(logits), dim=1) # B
        if self.reduction == 'mean':
            loss = torch.mean(output)
        elif self.reduction == 'sum':
            loss = torch.sum(output)

        return loss

    def forward(self, out_dict, target, ant_target): 
        return self.loss(out_dict, target, ant_target)



@CRITERIONS.register('UNIFORM_ANTICIPATION')
class OadUniformAntLoss(nn.Module):
    
    # reduction was sum, not mean, what if we change to sum?
    def __init__(self, cfg, reduction='mean'):
        super(OadUniformAntLoss, self).__init__()
        self.reduction = reduction
        self.loss = self.anticipation_loss
        self.num_classes = cfg['num_classes']

    def anticipation_loss(self, out_dict, target, ant_target):
        anticipation_logits = out_dict['anticipation_logits']
        pred_anticipation_logits = anticipation_logits[:,-1,:,:].contiguous().view(-1, self.num_classes)
        anticipation_logit_targets = ant_target.view(-1, self.num_classes)
        ant_loss = self.mlce_loss(pred_anticipation_logits, anticipation_logit_targets)
        return ant_loss

    def ce_loss(self, out_dict, target):
        # logits: (B, seq, K) target: (B, seq, K)
        logits = out_dict['logits']
        logits = logits[:,-1,:].contiguous()
        target = target[:,-1,:].contiguous()
        ce_loss = self.mlce_loss(logits, target)
        return ce_loss

    def mlce_loss(self, logits, target):
        '''
        multi label cross entropy loss. 
        logits: (B, K) target: (B, K) 
        '''
        logsoftmax = nn.LogSoftmax(dim=-1).to(logits.device)
        output = torch.sum(-target * logsoftmax(logits), dim=1) # B
        
        # output = torch.sum(-F.normalize(target) * logsoftmax(logits), dim=1) # B
        if self.reduction == 'mean':
            loss = torch.mean(output)
        elif self.reduction == 'sum':
            loss = torch.sum(output)

        return loss

    def forward(self, out_dict, target, ant_target): 
        return self.loss(out_dict, target, ant_target)
