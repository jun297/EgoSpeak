import json
import time

import numpy as np
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from trainer.eval_builder import EVAL
from utils import *
from utils import (event_accuracy, perframe_average_precision,
                   thumos_postprocessing)


@EVAL.register("OAD")
class Evaluate(nn.Module):
    
    def __init__(self, cfg):
        super(Evaluate, self).__init__()
        self.data_processing = thumos_postprocessing if 'THUMOS' in cfg['data_name'] else None
        self.metric = cfg['metric']
        self.eval_method = perframe_average_precision
        self.event_eval_method = event_accuracy
        self.all_class_names = json.load(open(cfg['video_list_path']))[cfg["data_name"].split('_')[0]]['class_index']
        self.cfg = cfg
        self.ignore_index = cfg['ignore_index'] if 'ignore_index' in cfg else -100
    
    def eval(self, model, dataloader, logger, criterion, use_wandb, epoch, writer, result_path):
        device = "cuda:0"
        model.eval()
        eval_preds = {}
        gt_targets_dict = {}
        
        with torch.no_grad():
            pred_scores, gt_targets = [], []
            start = time.time()
            eval_loss = 0.0
            for it, (vid_name, rgb_input, flow_input, target) in enumerate(tqdm(dataloader, desc='Evaluation:', leave=False)):
                rgb_input, flow_input, target = rgb_input.to(device), flow_input.to(device), target.to(device)
                out_dict = model(rgb_input, flow_input)
                
                if criterion != None:
                    loss = criterion(out_dict, target)
                    eval_loss += loss.item()
                    
                pred_logit = out_dict['logits']
                prob_val = pred_logit.squeeze().cpu().numpy()
                target_batch = target.squeeze().cpu().numpy()
                pred_scores += list(prob_val) 
                gt_targets += list(target_batch)
                
                eval_preds[vid_name] = prob_val
    
            end = time.time()
            num_frames = len(gt_targets)
            
            # save prediction scores
            # if result_path != None:
            time_taken = end - start
            logger.info(f'Processed {num_frames} frames in {time_taken:.1f} seconds ({num_frames / time_taken :.1f} FPS)')
                
            result_AP = self.eval_method(pred_scores, gt_targets, self.all_class_names, self.ignore_index, self.data_processing, 'AP')
            result_cAP = self.eval_method(pred_scores, gt_targets, self.all_class_names, self.ignore_index, self.data_processing, 'cAP')
            logger.info(f'per_class_ap: {result_AP["per_class_AP"]}')
            logger.info(f'all_class_names: {self.all_class_names}')
            logger.info(f'len(result_AP["per_class_AP"]: {len(result_AP["per_class_AP"])}')
            
            per_class_AP = result_AP["per_class_AP"]
            if use_wandb == True:
                wandb.log({"Validation Loss": eval_loss / len(dataloader)})
                wandb.log({"Validation per_class_AP": per_class_AP})            
                
            if writer != None:
                writer.add_scalar("Validation Loss", eval_loss / len(dataloader), epoch)
            
            sum_class_AP = 0.0
            
            for class_name in self.all_class_names:
                # TVSeries
                if class_name == 'background':
                    continue
                # THUMOS
                if class_name == 'Background' or class_name == 'Ambiguous':
                    continue
                
                # logger.info(f'{class_name}')
                class_AP = result_AP["per_class_AP"][class_name]
                sum_class_AP += class_AP
                
                class_cAP = result_cAP["per_class_AP"][class_name]
                logger.info(f'AP @ {class_name}: {class_AP:.4f}') 
                logger.info(f'cAP @ {class_name}: {class_cAP:.4f}')

            logger.info(f'sum_AP: {sum_class_AP}, mean_AP: {sum_class_AP / (len(self.all_class_names) - 1):.4f}')
            
            logger.info(f'pred_scores: {pred_scores[:2]}')
            logger.info(f'len(pred_scores): {len(pred_scores)}')
            logger.info(f'len(pred_scores[0]): {len(pred_scores[0])}')
            
        return result_AP['mean_AP'], result_cAP['mean_AP'], eval_preds
    
    def forward(self, model, dataloader, logger, criterion=None, use_wandb=False, epoch=None, writer=None, result_path = None):
        return self.eval(model, dataloader, logger, criterion, use_wandb, epoch, writer, result_path)

@EVAL.register("OADTR")
class EvaluateOADTR(nn.Module):
    
    def __init__(self, cfg):
        super(EvaluateOADTR, self).__init__()
        self.data_processing = thumos_postprocessing if 'THUMOS' in cfg['data_name'] else None
        self.metric = cfg['metric']
        self.eval_method = perframe_average_precision
        self.all_class_names = json.load(open(cfg['video_list_path']))[cfg["data_name"].split('_')[0]]['class_index']
        self.cfg = cfg
    
    def eval(self, model, dataloader, logger, criterion, use_wandb, epoch, writer):
        device = "cuda:0"
        model.eval()
        eval_preds = {}
        gt_targets_dict = {}
        
        with torch.no_grad():
            pred_scores, gt_targets = [], []
            start = time.time()
            eval_loss = 0.0
            for it, (vid_name, rgb_input, flow_input, enc_target, distance_target, class_h_target, dec_target) in enumerate(tqdm(dataloader, desc='Evaluation:', leave=False)):
                rgb_input, flow_input, target = rgb_input.to(device), flow_input.to(device), target.to(device)
                out_dict = model(rgb_input, flow_input)
                
                if criterion != None:
                    loss = criterion(out_dict, target)
                    eval_loss += loss.item()
                    
                pred_logit = out_dict['logits']
                prob_val = pred_logit.squeeze().cpu().numpy()
                target_batch = target.squeeze().cpu().numpy()
                pred_scores += list(prob_val) 
                gt_targets += list(target_batch)
                
                # eval_preds[vid_name] = pred_scores
                # gt_targets_dict[vid_name] = pred_scores
            
            if use_wandb == True:
                wandb.log({"Validation Loss": eval_loss / len(dataloader)})
                
            if writer != None:
                writer.add_scalar("Validation Loss", eval_loss / len(dataloader), epoch)
            
            end = time.time()
            num_frames = len(gt_targets)
            result_AP = self.eval_method(pred_scores, gt_targets, self.all_class_names, self.data_processing, 'AP')
            result_cAP = self.eval_method(pred_scores, gt_targets, self.all_class_names, self.data_processing, 'cAP')
            time_taken = end - start
            logger.info(f'Processed {num_frames} frames in {time_taken:.1f} seconds ({num_frames / time_taken :.1f} FPS)')
            # logger.info(f'per_class_ap: {result["per_class_AP"]}')
            logger.info(f'all_class_names: {self.all_class_names}')
            logger.info(f'len(result_AP["per_class_AP"]: {len(result_AP["per_class_AP"])}')
            
            sum_class_AP = 0.0
            
            for class_name in self.all_class_names:
                # TVSeries
                if class_name == 'background':
                    continue
                # THUMOS
                if class_name == 'Background' or class_name == 'Ambiguous':
                    continue
                
                # logger.info(f'{class_name}')
                class_AP = result_AP["per_class_AP"][class_name]
                sum_class_AP += class_AP
                
                class_cAP = result_cAP["per_class_AP"][class_name]
                logger.info(f'AP @ {class_name}: {class_AP:.4f}') 
                logger.info(f'cAP @ {class_name}: {class_cAP:.4f}')

            logger.info(f'sum_AP: {sum_class_AP}, mean_AP: {sum_class_AP / (len(self.all_class_names) - 1):.4f}')
            
            logger.info(f'pred_scores: {pred_scores[:2]}')
            logger.info(f'len(pred_scores): {len(pred_scores)}')
            logger.info(f'len(pred_scores[0]): {len(pred_scores[0])}')
            
        return result_AP['mean_AP'], result_cAP['mean_AP'], eval_preds
    
    def forward(self, model, dataloader, logger, criterion=None, use_wandb=False, epoch=None, writer=None):
        return self.eval(model, dataloader, logger, criterion, use_wandb, epoch, writer)


@EVAL.register("ANTICIPATION")
class ANT_Evaluate(nn.Module):
    
    def __init__(self, cfg):
        super(ANT_Evaluate, self).__init__()
        data_name = cfg["data_name"].split('_')[0]
        self.data_processing = thumos_postprocessing if data_name == 'THUMOS' else None
        self.metric = cfg['metric']
        self.eval_method = perframe_average_precision
        self.event_eval_method = event_accuracy
        self.all_class_names = json.load(open(cfg['video_list_path']))[data_name]['class_index']
        self.cfg = cfg
        
        if 'ignore_index' in cfg:
            self.ignore_index = cfg['ignore_index']
        else:
            raise NotImplementedError("Please set ignore_index in config")
            
    def eval(self, model, dataloader, logger, criterion, use_wandb, epoch, writer, result_path):
        device = "cuda:0"
        model.eval()
        ant_eval_pred_scores_dict = {}
        ant_gt_targets_dict = {}
        
        with torch.no_grad():
            pred_scores, gt_targets, ant_pred_scores, ant_gt_targets = [], [], [], []
            start = time.time()
            anticipation_mAPs = []
            for vid_name, rgb_input, flow_input, target, ant_target in tqdm(dataloader, desc='Evaluation:', leave=False):
                rgb_input, flow_input, target, ant_target = rgb_input.to(device), flow_input.to(device), target.to(device), ant_target.to(device)
                
                out_dict = model(rgb_input, flow_input)                
                pred_logit = out_dict['logits']
                ant_pred_logit = out_dict['anticipation_logits']
                
                prob_val = pred_logit.squeeze().cpu().numpy() # [8390, 3]
                target_batch = target.squeeze().cpu().numpy()
                ant_prob_val = ant_pred_logit.squeeze().cpu().numpy() # [8390, ANTICIPATION_LENGTH, 3]
                ant_target_batch = ant_target.squeeze().cpu().numpy()
                
                pred_scores += list(prob_val)
                gt_targets += list(target_batch)
                ant_pred_scores += list(ant_prob_val)
                ant_gt_targets += list(ant_target_batch)
                if isinstance(vid_name, tuple):
                    vid_name = vid_name[0]
                    
                ant_eval_pred_scores_dict[vid_name] = ant_prob_val
                ant_gt_targets_dict[vid_name] = ant_target_batch

            end = time.time()
            num_frames = len(gt_targets)
            
            result = self.eval_method(pred_scores, gt_targets, self.all_class_names, self.ignore_index, self.data_processing, self.metric)
            per_class_AP = result['per_class_AP']
            
            ant_pred_scores = np.array(ant_pred_scores)
            ant_gt_targets = np.array(ant_gt_targets)
            logger.info(f'OAD mAP: {result["mean_AP"]*100:.2f}')
            # Prepare the formatted string
            formatted_output = "per_class_AP:\n" + "\n".join(f"  {class_name:<25} : {ap_value:.4f}" for class_name, ap_value in per_class_AP.items())

            # Log the formatted string
            logger.info(formatted_output)
            
            if use_wandb == True:
                wandb.log({"OAD mAP": result["mean_AP"]})
            
            anticipation_mAPs_per_class = {key: [] for key in per_class_AP.keys()}
            
            for step in range(ant_gt_targets.shape[1]):                
                result[f'anticipation_{step+1}'] = self.eval_method(ant_pred_scores[:,step,:], ant_gt_targets[:,step,:], self.all_class_names, self.ignore_index, self.data_processing, self.metric)
                anticipation_mAPs.append(result[f'anticipation_{step+1}']['mean_AP'])
                formatted_output = "per_class_AP:\n" + "\n".join(f"  {class_name:<25} : {ap_value:.4f}" for class_name, ap_value in result[f'anticipation_{step+1}']['per_class_AP'].items())
                logger.info(f"Anticipation at step {step+1}: {result[f'anticipation_{step+1}']['mean_AP']*100:.2f}")
                logger.info(formatted_output)
                
                for class_name in self.all_class_names:
                    anticipation_mAPs_per_class[class_name].append(result[f'anticipation_{step+1}']['per_class_AP'][class_name])
                
                if use_wandb == True:
                    wandb.log({f"Anticipation at step {step+1}": result[f'anticipation_{step+1}']['mean_AP']})
            
            logger.info(f'Mean Anticipation mAP: {np.mean(anticipation_mAPs)*100:.2f}')
            formatted_output = "Mean Anticipation per_class_AP:\n" + "\n".join(f"  {class_name:<25} : {np.mean(anticipation_mAPs_per_class[class_name])*100:.4f}" for class_name in self.all_class_names)
            # logger.info(f'Mean Anticipation mAP per class: {np.mean(list(anticipation_mAPs_per_class.values()), axis=1)}')
            logger.info(formatted_output)

            if use_wandb == True:
                wandb.log({"Mean Anticipation mAP": np.mean(anticipation_mAPs)})
            
            time_taken = end - start
            logger.info(f'Processed {num_frames} frames in {time_taken:.1f} seconds ({num_frames / time_taken :.1f} FPS)')
        
        return np.mean(anticipation_mAPs), np.mean(anticipation_mAPs), ant_eval_pred_scores_dict
    
    # def forward(self, model, dataloader, logger):
    #     return self.eval(model, dataloader, logger)
    def forward(self, model, dataloader, logger, criterion=None, use_wandb=False, epoch=None, writer=None, result_path=None):
        return self.eval(model, dataloader, logger, criterion, use_wandb, epoch, writer, result_path)
