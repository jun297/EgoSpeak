# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import os.path as osp
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from rekognition_online_action_detection.evaluation import compute_result
from tqdm import tqdm


def log_results(phase, cfg, compute_result, gt_targets, pred_scores, losses, data_loader, epoch=None, best_mAP=None, best_epoch=None):
    result = compute_result['perframe'](cfg, gt_targets, pred_scores)
    per_class_AP = result['mean_AP']
    
    per_class_AP_log = ' '.join(f'{class_name}: {class_AP:.5f}' for class_name, class_AP in per_class_AP.items())
    
    log = []
    if epoch is not None:
        log.append(f'\nEpoch {epoch:2}')
    
    log.append(f'{phase} det_loss: {losses["det"] / len(data_loader.dataset):.5f} det_mAP: {result["mean_AP"]:.5f} {per_class_AP_log}')
    
    if cfg.USE_WANDB:
        wandb.log({
            f"{phase.capitalize()} mAP": result['mean_AP'],
            f"{phase.capitalize()} per_class_AP": per_class_AP,
            f"{phase.capitalize()} Loss": losses['det'] / len(data_loader.dataset)
        })
    
    if cfg.MODEL.LSTR.V_N_CLASSIFIER:
        for loss_type in ['verb', 'noun']:
            if loss_type in losses:
                log.append(f'{phase} {loss_type}_loss: {losses[loss_type] / len(data_loader.dataset):.5f}')
    
    return log, result['mean_AP']

def do_perframe_det_train_rnn(cfg,
                          data_loaders,
                          model,
                          criterion,
                          optimizer,
                          scheduler,
                          device,
                          checkpointer,
                          logger):
    # Setup model on multiple GPUs
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    best_mAP, best_epoch = 0, 0
    best_per_class_AP_log = None

    for epoch in range(cfg.SOLVER.START_EPOCH, cfg.SOLVER.START_EPOCH + cfg.SOLVER.NUM_EPOCHS):
        # Reset
        det_losses = {phase: 0.0 for phase in cfg.SOLVER.PHASES}
        pred_losses = {phase: 0.0 for phase in cfg.SOLVER.PHASES}
        
        det_pred_scores = []
        det_gt_targets = []
        aa_pred_scores = []
        aa_gt_targets = []
        
        train_det_pred_scores = []
        train_det_gt_targets = []
        train_aa_pred_scores = []
        train_aa_gt_targets = []

        start = time.time()
        
        for phase in cfg.SOLVER.PHASES:
            training = phase == 'train'
            model.train(training)

            with torch.set_grad_enabled(training):
                pbar = tqdm(data_loaders[phase],
                            desc='{}ing epoch {}'.format(phase.capitalize(), epoch))
                
                # __getitem in LSTR:
                # return (fusion_visual_inputs, fusion_motion_inputs,
                #       fusion_object_inputs, memory_key_padding_mask, target)
                # __getitem__ in RNN:
                # vid, rgb_input, flow_input, text_input, target, ant_target
                for batch_idx, data in enumerate(pbar, start=1):
                    batch_size = data[1].shape[0]                                        
                    det_target = data[-2].to(device)
                    aa_target = data[-1].to(device)
                    
                    loss_names = list(zip(*cfg.MODEL.CRITERIONS))[0]
                    assert 'PRED_FUTURE' not in list(zip(*cfg.MODEL.CRITERIONS))[0]
                    # if 'PRED_FUTURE' in list(zip(*cfg.MODEL.CRITERIONS))[0]:
                    #     det_score, feat_out, feat_ori = model(*[x.to(device) for x in data[:-1]])
                    # else:
                    #     det_score = model(*[x.to(device) for x in data[1:4]])
                                        
                    out_dict = model(*[x.to(device) for x in data[1:4]])
                    # out_dict['logits']: [B, window_size, num_classes]
                    # out_dict['anticipation_logits']: [B, window_size, ant_len, num_classes]
                    
                    # [workmem_len + ant_len, batch_size, self.d_model]: [50, 16, 1024]
                    
                    det_score = out_dict['logits']
                    det_score = det_score
                    if cfg.MODEL.MINIROAD.ANTICIPATION_LENGTH > 0:
                        aa_score = out_dict['anticipation_logits'] # [B, window_size, anticipation_length, num_class]
                        aa_score = aa_score[:, -1, :, :].reshape(-1, cfg.DATA.NUM_CLASSES)
                        
                        aa_target = aa_target.reshape(-1, cfg.DATA.NUM_CLASSES)
                        det_loss = criterion[loss_names[0]](aa_score, aa_target)
                        anticipation_logits = out_dict['anticipation_logits']
        
                    else:
                        det_score = det_score.reshape(-1, cfg.DATA.NUM_CLASSES)
                        det_target = det_target.reshape(-1, cfg.DATA.NUM_CLASSES)
                        det_loss = criterion[loss_names[0]](det_score, det_target)

                        det_loss = criterion[loss_names[0]](det_score, det_target)
                        
                        
                    # if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                    #     aa_score = det_score[:, -cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES:, :]
                    #     # [16, 10, 3]
                    #     aa_target = det_target[:, -cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES:, :]
                        
                    # if cfg.MODEL.LSTR.LOSS_ANTICIPATE_ONLY:
                    #     det_score = det_score[:, -cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES:, :]
                    #     det_target = det_target[:, -cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES:, :]
                    
                    # det_score: [16, 50, 3], aa_score: [16, 10, 3]
                    # det_score can be different when aa_score 
                    
                    det_losses[phase] += det_loss.item() * batch_size
                    
                    # Output log for current batch
                    pbar.set_postfix({
                        'lr': '{:.7f}'.format(scheduler.get_last_lr()[0]),
                        'det_loss': '{:.5f}'.format(det_loss.item()),
                    })

                    if training:
                        optimizer.zero_grad()
                        loss = det_loss
                        if 'PRED_FUTURE' in list(zip(*cfg.MODEL.CRITERIONS))[0]:
                            loss += pred_loss
                        if cfg.MODEL.LSTR.V_N_CLASSIFIER:
                            loss += noun_loss + verb_loss
                        if loss.item() != 0:
                            loss.backward()
                            optimizer.step()
                            scheduler.step()
                        
                        if cfg.USE_WANDB:
                            wandb.log({"Learning Rate": scheduler.get_last_lr()[0]})
                            wandb.log({"Train Loss": det_loss.item()})
                        
                        det_score = det_score.softmax(dim=1).cpu().tolist()
                        det_target = det_target.cpu().tolist()
                        train_det_pred_scores.extend(det_score)
                        train_det_gt_targets.extend(det_target)
                        # len(train_det_pred_scores) + 16
                        
                        if cfg.MODEL.MINIROAD.ANTICIPATION_LENGTH > 0:
                            aa_score = aa_score.softmax(dim=1).cpu().tolist()
                            aa_target = aa_target.cpu().tolist()
                            train_aa_pred_scores.extend(aa_score)
                            train_aa_gt_targets.extend(aa_target)
                            
                    else:
                        # Prepare for evaluation
                        det_score = det_score.softmax(dim=1).cpu().tolist()
                        det_target = det_target.cpu().tolist()
                        det_pred_scores.extend(det_score)
                        det_gt_targets.extend(det_target)
                        
                        if cfg.MODEL.MINIROAD.ANTICIPATION_LENGTH > 0:
                            aa_score = aa_score.softmax(dim=1).cpu().tolist()
                            aa_target = aa_target.cpu().tolist()
                            aa_pred_scores.extend(aa_score)
                            aa_gt_targets.extend(aa_target)
                        
                        # print(f'len(det_score): {len(det_score)}, len(det_pred_scores): {len(det_pred_scores)}')  
                        # print(f'len(aa_score): {len(aa_score)}, len(aa_pred_scores): {len(aa_pred_scores)}')
            
        end = time.time()

        # After the phase loop, compute and log results
        for phase in cfg.SOLVER.PHASES:
            if phase == 'train':
                gt_targets = train_det_gt_targets
                pred_scores = train_det_pred_scores
            else:  # 'test'
                gt_targets = det_gt_targets
                pred_scores = det_pred_scores
            
            result = compute_result['perframe'](cfg, gt_targets, pred_scores)
            per_class_AP = result['per_class_AP']
            
            log = []
            log.append(f'\nEpoch {epoch:2}')
            log.append(f'{phase} det_loss: {det_losses[phase] / len(data_loaders[phase].dataset):.5f} det_mAP: {result["mean_AP"]:.5f}')
            
            per_class_AP_log = ' '.join(f'{class_name}: {class_AP:.5f}' for class_name, class_AP in per_class_AP.items())
            log.append(f'Per-class AP: {per_class_AP_log}')
            
            if cfg.USE_WANDB:
                wandb.log({
                    f"{phase.capitalize()} mAP": result['mean_AP'],
                    f"{phase.capitalize()} per_class_AP": per_class_AP,
                    f"{phase.capitalize()} Loss": det_losses[phase] / len(data_loaders[phase].dataset)
                })
            
            if cfg.MODEL.LSTR.V_N_CLASSIFIER:
                log.append(f'{phase} verb_loss: {verb_losses[phase] / len(data_loaders[phase].dataset):.5f}')
                log.append(f'{phase} noun_loss: {noun_losses[phase] / len(data_loaders[phase].dataset):.5f}')
            
            if phase == 'test':
                if result['mean_AP'] > best_mAP:
                    best_mAP = result['mean_AP']
                    best_epoch = epoch
                    best_per_class_AP_log = per_class_AP_log
                    checkpointer.save_best(epoch, model, optimizer)
                
                log.append(f'\nbest epoch: {best_epoch:2} best det_mAP: {best_mAP:.5f} best per-class AP: {best_per_class_AP_log}\n')
            
            logger.info(' | '.join(log))

        log.append(f'running time: {end - start:.2f} sec')
        logger.info(' | '.join(log))

        # Save checkpoint for model and optimizer
        if cfg.SOLVER.SAVE_EVERY_EPOCHS > 0 and epoch % cfg.SOLVER.SAVE_EVERY_EPOCHS == 0:
            checkpointer.save(epoch, model, optimizer)

        # Shuffle dataset for next epoch
        data_loaders['train'].dataset.shuffle()

# @TRAINER.register("OAD")
def train_one_epoch(trainloader, model, criterion, optimizer, scaler, epoch, use_wandb = False, writer=None, scheduler=None):
    epoch_loss = 0
    for it, (vid_name, rgb_input, flow_input, target) in enumerate(tqdm(trainloader, desc=f'Epoch:{epoch} Training', postfix=f'lr: {optimizer.param_groups[0]["lr"]:.7f}')):
        rgb_input, flow_input, target = rgb_input.cuda(), flow_input.cuda(), target.cuda()
        model.train()
        if scaler != None:
            with torch.cuda.amp.autocast():    
                out_dict = model(rgb_input, flow_input) 
                loss = criterion(out_dict, target)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out_dict = model(rgb_input, flow_input)
            loss = criterion(out_dict, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        
        epoch_loss += loss.item()
        if use_wandb == True:
            wandb.log({"Train Loss": loss.item()})
            
        if writer != None:
            writer.add_scalar("Train Loss", loss.item(), it+epoch*len(trainloader))
    return epoch_loss

# @TRAINER.register("OADTR")
def train_one_epoch_oadtr(trainloader, model, criterion, optimizer, scaler, epoch, use_wandb = False, writer=None, scheduler=None):
    epoch_loss = 0
    for it, (vid_name, rgb_input, flow_input, enc_target, distance_target, class_h_target, dec_target) in enumerate(tqdm(trainloader, desc=f'Epoch:{epoch} Training', postfix=f'lr: {optimizer.param_groups[0]["lr"]:.7f}')):
        rgb_input, flow_input, enc_target, distance_target, class_h_target = rgb_input.cuda(), flow_input.cuda(), enc_target.cuda(), distance_target.cuda(), class_h_target.cuda()
        model.train()
        if scaler != None:
            with torch.cuda.amp.autocast():    
                out_dict = model(rgb_input, flow_input) 
                loss = criterion(out_dict, enc_target, dec_target)   
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out_dict = model(rgb_input, flow_input) 
            loss = criterion(out_dict, enc_target, dec_target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()
        if use_wandb == True:
            wandb.log({"Train Loss": loss.item()})
            
        if writer != None:
            writer.add_scalar("Train Loss", loss.item(), it+epoch*len(trainloader))
            
    return epoch_loss


# @TRAINER.register("ANTICIPATION")
def ant_train_one_epoch(trainloader, model, criterion, optimizer, scaler, epoch, use_wandb = False, writer=None, scheduler=None):
    epoch_loss = 0
    for it, (vid_name, rgb_input, flow_input, text_input, target, ant_target) in enumerate(tqdm(trainloader, desc=f'Epoch:{epoch} Training', postfix=f'lr: {optimizer.param_groups[0]["lr"]:.7f}')):
        rgb_input, flow_input, text_input, target, ant_target = rgb_input.cuda(), flow_input.cuda(), text_input.cuda(), target.cuda(), ant_target.cuda()
        model.train()
        if scaler != None:
            with torch.cuda.amp.autocast():
                out_dict = model(rgb_input, flow_input, text_input)
                loss = criterion(out_dict, target, ant_target)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out_dict = model(rgb_input, flow_input, text_input) 
            loss = criterion(out_dict, target, ant_target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        epoch_loss += loss.item()
        
        if use_wandb == True:
            wandb.log({"Train Loss": loss.item()})
            
        if writer != None:
            writer.add_scalar("Train Loss", loss.item(), it+epoch*len(trainloader))
            
    return epoch_loss