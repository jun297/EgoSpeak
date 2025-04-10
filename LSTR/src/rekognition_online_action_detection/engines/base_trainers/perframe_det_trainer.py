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

def do_perframe_det_train(cfg,
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
        verb_losses = {phase: 0.0 for phase in cfg.SOLVER.PHASES}
        noun_losses = {phase: 0.0 for phase in cfg.SOLVER.PHASES}
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
                for batch_idx, data in enumerate(pbar, start=1):
                    batch_size = data[0].shape[0]
                    
                    # batch_size 16
                    # data[0]: visual_input, [16, 680, 2048]
                    # [BATCH_SIZE, 640 + 40 (LONG_MEMORY_NUM_SAMPLES + WORK_MEMORY_NUM_SAMPLES), FEATURE_SIZE]
                    # this if statement is for the case of EK100 dataset
                    if cfg.MODEL.LSTR.V_N_CLASSIFIER:
                        det_target, verb_target, noun_target = data[-1]
                        det_target = det_target.to(device)
                        verb_target = verb_target.to(device)
                        noun_target = noun_target.to(device)
                    else:
                        det_target = data[-1].to(device)

                    loss_names = list(zip(*cfg.MODEL.CRITERIONS))[0]
                    if 'PRED_FUTURE' in list(zip(*cfg.MODEL.CRITERIONS))[0]:
                        det_score, feat_out, feat_ori = model(*[x.to(device) for x in data[:-1]])
                    else:
                        det_score = model(*[x.to(device) for x in data[:-1]])

                    if cfg.MODEL.LSTR.V_N_CLASSIFIER:
                        det_score, verb_score, noun_score = det_score
                    else:
                        verb_score, noun_score = None, None
                        
                    if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                        aa_score = det_score[:, -cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES:, :]
                        # [16, 10, 3]
                        aa_target = det_target[:, -cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES:, :]
                        
                    if cfg.MODEL.LSTR.LOSS_ANTICIPATE_ONLY:
                        det_score = det_score[:, -cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES:, :]
                        det_target = det_target[:, -cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES:, :]
                    
                    # det_score: [16, 50, 3], aa_score: [16, 10, 3]
                    # det_score can be different when aa_score 
                                       
                    det_score = det_score.reshape(-1, cfg.DATA.NUM_CLASSES)
                    det_target = det_target.reshape(-1, cfg.DATA.NUM_CLASSES)
                    
                    assert len(det_score.shape) == 2                    
                    det_loss = criterion[loss_names[0]](det_score, det_target)
                    det_losses[phase] += det_loss.item() * batch_size

                    if 'PRED_FUTURE' in list(zip(*cfg.MODEL.CRITERIONS))[0]:
                        pred_loss = criterion['PRED_FUTURE'](feat_out, feat_ori)
                        pred_losses[phase] += pred_loss.item() * batch_size

                    if cfg.MODEL.LSTR.V_N_CLASSIFIER:
                        if cfg.MODEL.LSTR.LOSS_ANTICIPATE_ONLY:
                            verb_score = verb_score[:, -cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES:, :]
                            verb_target = verb_target[:, -cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES:, :]
                            noun_score = noun_score[:, -cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES:, :]
                            noun_target = noun_target[:, -cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES:, :]
                        verb_score = verb_score.reshape(-1, verb_score.shape[-1])
                        noun_score = noun_score.reshape(-1, noun_score.shape[-1])
                        verb_target = verb_target.reshape(-1, verb_target.shape[-1])
                        noun_target = noun_target.reshape(-1, noun_target.shape[-1])
                        verb_loss = criterion['MCE'](verb_score, verb_target)
                        noun_loss = criterion['MCE'](noun_score, noun_target)
                        verb_losses[phase] += verb_loss.item() * batch_size
                        noun_losses[phase] += noun_loss.item() * batch_size

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
                        
                        if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
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
                        
                        if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                            aa_score = aa_score.softmax(dim=1).cpu().tolist()
                            aa_target = aa_target.cpu().tolist()
                            aa_pred_scores.extend(aa_score)
                            aa_gt_targets.extend(aa_target)
                        
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
