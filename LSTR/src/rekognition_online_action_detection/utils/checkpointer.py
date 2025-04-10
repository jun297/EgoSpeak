# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

__all__ = ['setup_checkpointer']

import os.path as osp

import torch


class Checkpointer(object):

    def __init__(self, cfg, phase):

        # Load pretrained checkpoint
        
        self.is_pretrained = False
        self.cfg = cfg
        self.checkpoint = self._load_checkpoint(cfg.MODEL.CHECKPOINT)
        self.except_classifier = cfg.MODEL.EXCEPT_CLASSIFIER if phase == 'train' else False
        
        if self.checkpoint is not None and phase == 'train':
            # further training with the same dataset: resume
            model_ckpt_str = cfg['MODEL']['CHECKPOINT'].replace('_', '')
            if cfg['DATA']['DATA_NAME'].lower() in model_ckpt_str.lower():
                cfg.SOLVER.START_EPOCH += self.checkpoint.get('epoch', 0)
            # further training with the different dataset: finetuning on pretrained weight
            else:
                self.is_pretrained = True
                print(f'WARNING) Finetuning on pretrained weight: {cfg.MODEL.CHECKPOINT}')
        
        elif self.checkpoint is None and phase != 'train':
            # no checkpoint in infernece mode: error
            raise RuntimeError('Cannot find checkpoint {}'.format(cfg.MODEL.CHECKPOINT))
        
        self.output_dir = cfg.OUTPUT_DIR

    def load(self, model, optimizer=None):
        if self.checkpoint is not None:
            # try load the ckpt first, then if there is a size mismatch, do other operation
            if not self.is_pretrained: # just resume
                # Attempt to load the state dictionary from the checkpoint
                model.load_state_dict(self.checkpoint['model_state_dict'])
                # Load the optimizer state if provided
                if optimizer is not None:
                    if 'optimizer_state_dict' in self.checkpoint:
                        optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
                    else:
                        print('Optimizer state dictionary not found in checkpoint.')

            else:
                # different model structure (pretrained)
                print(f'WARNING) Load checkpoint separately due to possiblity of different model structure, src/rekognition_online_action_detection/utils/checkpointer.py L51')
                model_dict = model.state_dict()
                unnecessary_keys = []
                unmatched_keys = []
                matched_ckpt = {}

                # loaded_state_dict = self.checkpoint['model_state_dict']

                # # Initialize a flag to track if all values match
                # all_values_match = True

                # # Iterate through all keys and compare values
                # for key in model_dict:
                #     import pdb; pdb.set_trace()
                #     if not torch.allclose(model_dict[key], loaded_state_dict[key]):
                #         print(f"Mismatch found in {key}")
                #         print(f"Model value: {model_dict[key]}")
                #         print(f"Loaded value: {loaded_state_dict[key]}")
                #         print(f"Difference: {model_dict[key] - loaded_state_dict[key]}")
                #         print("---")
                #         all_values_match = False
                        
                # Filter out unnecessary keys
                for k, v in self.checkpoint['model_state_dict'].items():
                    if k not in model_dict:
                        print(f'Unnecessary key: {k} not found in model')
                        unnecessary_keys.append(k)
                    elif model_dict[k].shape != v.shape:
                        print(f'Unnecessary key: {k} shape {v.shape} does not match {model_dict[k].shape}')
                        unmatched_keys.append(k)
                    elif self.except_classifier is True and 'classifier' in k:
                        print(f'Unnecessary key: {k} is classifier')
                        unnecessary_keys.append(k)
                    else:
                        matched_ckpt[k] = v

                print(f'Unloaded keys: {unnecessary_keys}')
                print(f'Unmathced keys: {unmatched_keys}')
                # Update the existing state dict with the matched keys
                # import pdb; pdb.set_trace()
                model_dict.update(matched_ckpt)
                model.load_state_dict(model_dict)
                # for unmatched_key in unmatched_keys:
                #     loaded_dict_shape = self.checkpoint['model_state_dict'][unmatched_key].shape
                #     model_dict_shape = model_dict[unmatched_key].shape
                #     if len(loaded_dict_shape) == 2:
                #         if loaded_dict_shape[0] != model_dict_shape[0]:
                #             model_dict[unmatched_key][:loaded_dict_shape[0], :] = self.checkpoint['model_state_dict'][unmatched_key]
                #             print(f'Updated key: {unmatched_key}, model_dict[{unmatched_key}][:{loaded_dict_shape[0]}, :] = checkpoint["model_state_dict"][{unmatched_key}]')
                #         else:
                #             model_dict[unmatched_key][:, :loaded_dict_shape[1]] = self.checkpoint['model_state_dict'][unmatched_key]
                #             print(f'Updated key: {unmatched_key}, model_dict[{unmatched_key}][:, :{loaded_dict_shape[1]}] = checkpoint["model_state_dict"][{unmatched_key}]')
                #     elif len(loaded_dict_shape) == 1:
                #         model_dict[unmatched_key][:loaded_dict_shape[0]] = self.checkpoint['model_state_dict'][unmatched_key]
                #         print(f'Updated key: {unmatched_key}, model_dict[{unmatched_key}][:{loaded_dict_shape[0]}] = checkpoint["model_state_dict"][{unmatched_key}]')
                #     elif len(loaded_dict_shape) >= 3:
                #         raise ValueError(f'Unsupported shape: {loaded_dict_shape}')
                # model.load_state_dict(model_dict)
                    
                    

    def save(self, epoch, model, optimizer):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, osp.join(self.output_dir, f'epoch-{epoch}.pth'))
        print(f'\nSave checkpoint to {osp.join(self.output_dir, f"epoch-{epoch}.pth")}')
        # osp.join(self.output_dir, f'{self.cfg.INPUT.MODALITY}_epoch-{epoch}.pth'))

    def save_best(self, epoch, model, optimizer):
        # save_path = osp.join(self.output_dir, f'{self.cfg.INPUT.MODALITY}_best.pth')
        save_path = osp.join(self.output_dir, f'best.pth')
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_path)
        print(f'\nSave checkpoint to {save_path}')

    def _load_checkpoint(self, checkpoint):
        if checkpoint is not None and osp.isfile(checkpoint):
            return torch.load(checkpoint, map_location=torch.device('cpu'))
        elif osp.isdir(checkpoint):
            import pdb; pdb.set_trace()
            
        return None


def setup_checkpointer(cfg, phase):
    return Checkpointer(cfg, phase)