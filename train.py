# Monkey patch for Python 3.10+ compatibility
import collections
if not hasattr(collections, 'MutableMapping'):
    import collections.abc
    collections.MutableMapping = collections.abc.MutableMapping

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tools.cfg import py2cfg
import os
import torch
from torch import nn
import cv2
import numpy as np
import argparse
from pathlib import Path
from tools.metric import Evaluator
from pytorch_lightning.loggers import CSVLogger
import random
import math

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    return parser.parse_args()

def get_edge_gt(mask, edge_width=3):
    """Generate edge ground truth from segmentation mask using Canny edge detector or Laplacian"""
    # mask: [B, H, W]
    edge_gts = []
    device = mask.device
    mask_np = mask.cpu().numpy().astype(np.uint8)
    
    for i in range(mask.shape[0]):
        curr_mask = mask_np[i]
        
        # Canny-like gradient approach on label map (finds transitions)
        dy = cv2.Sobel(curr_mask, cv2.CV_64F, 0, 1, ksize=3)
        dx = cv2.Sobel(curr_mask, cv2.CV_64F, 1, 0, ksize=3)
        edge = np.sqrt(dx**2 + dy**2)
        edge[edge > 0] = 1
        edge = (edge > 0.5).astype(np.float32)
        
        edge_gts.append(torch.from_numpy(edge).unsqueeze(0)) # [1, H, W]
        
    return torch.stack(edge_gts).to(device)

class Supervision_Train(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = config.net

        self.loss = config.loss
        
        # Additional Edge Loss
        self.edge_loss_fn = nn.BCEWithLogitsLoss()
        self.edge_loss_weight = getattr(config, 'edge_loss_weight', 0.2)

        # 对比损失权重的「最大值」
        self.lambda_contrast_max = float(getattr(config, 'lambda_contrast', os.getenv('LAMBDA_CONTRAST', '0.6')))
        self.contrast_warmup_epochs = int(getattr(config, 'contrast_warmup_epochs', os.getenv('CONTRAST_WARMUP_EPOCHS', '50')))
        self.contrast_schedule = str(getattr(config, 'contrast_schedule', os.getenv('CONTRAST_SCHEDULE', 'cosine'))).lower()

        self.metrics_train = Evaluator(num_class=config.num_classes)
        self.metrics_val = Evaluator(num_class=config.num_classes)
        
        self.train_losses = []
        self.val_losses = []

    def forward(self, x):
        # only net is used in the prediction/inference
        output = self.net(x)
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

    def training_step(self, batch, batch_idx):
        img, mask = batch['img'], batch['gt_semantic_seg']

        # Model forward
        output = self.net(img)
        
        # Handle tuple output (seg, aux, edge) or (seg, aux)
        edge_logits = None
        contrast_loss = None
        
        if isinstance(output, (tuple, list)):
            # Assume order: seg, aux_loss_val, [edge_logits]
            prediction = output[0]
            aux_loss_term = output[1] if len(output) > 1 else 0
            if len(output) > 2:
                edge_logits = output[2]
        else:
            prediction = output
            aux_loss_term = 0

        # Main Segmentation Loss
        seg_loss = self.loss(prediction, mask)
        
        # Add pre-calculated Aux Loss from model (if any)
        # FIX: Ensure aux_loss_term is a proper Tensor (loss value)
        # If it's a prediction map (tensor with shape [B, C, H, W]), calculate loss against mask
        if isinstance(aux_loss_term, torch.Tensor):
            if aux_loss_term.ndim == 4: # Prediction map [B, C, H, W]
                aux_loss_val = self.loss(aux_loss_term, mask)
                seg_loss += 0.4 * aux_loss_val
            elif aux_loss_term.ndim == 0: # Scalar loss
                seg_loss += 0.4 * aux_loss_term

        # Contrastive Loss (Optional)
        if hasattr(self.net, 'last_contrastive_loss') and (self.net.last_contrastive_loss is not None):
            contrast_loss = self.net.last_contrastive_loss

        def get_contrast_lambda():
            if self.lambda_contrast_max <= 0: return 0.0
            if self.contrast_warmup_epochs <= 0: return self.lambda_contrast_max
            progress = min(1.0, max(0.0, (self.current_epoch + 1) / float(self.contrast_warmup_epochs)))
            if self.contrast_schedule == 'cosine':
                return self.lambda_contrast_max * 0.5 * (1.0 - math.cos(math.pi * progress))
            return self.lambda_contrast_max * progress

        lambda_contrast_now = get_contrast_lambda()

        total_loss = seg_loss
        if contrast_loss is not None and lambda_contrast_now > 0:
            total_loss = total_loss + lambda_contrast_now * contrast_loss
            
        # Edge Loss
        if edge_logits is not None:
            edge_gt = get_edge_gt(mask) # Generate GT on the fly
            # Resize logits if needed
            if edge_logits.shape[-2:] != edge_gt.shape[-2:]:
                edge_logits = F.interpolate(edge_logits, size=edge_gt.shape[-2:], mode='bilinear', align_corners=False)
            
            edge_loss = self.edge_loss_fn(edge_logits, edge_gt)
            total_loss += self.edge_loss_weight * edge_loss
            self.log('train_edge_loss', edge_loss, prog_bar=False, on_step=True, on_epoch=True)

        if self.config.use_aux_loss and isinstance(prediction, (tuple, list)):
             pre_mask = nn.Softmax(dim=1)(prediction[0])
        else:
             pre_mask = nn.Softmax(dim=1)(prediction)

        pre_mask = pre_mask.argmax(dim=1)
        for i in range(mask.shape[0]):
            self.metrics_train.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())

        self.train_losses.append(total_loss.item())
        
        accuracy = (pre_mask == mask).float().mean()
        
        self.log('train_loss', total_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_seg_loss', seg_loss, prog_bar=False, on_step=True, on_epoch=True)
        if contrast_loss is not None:
            self.log('train_contrast_loss', contrast_loss, prog_bar=False, on_step=True, on_epoch=True)
        self.log('train_contrast_lambda', torch.tensor(lambda_contrast_now, device=self.device), prog_bar=False, on_step=True, on_epoch=True)
        self.log('train_accuracy', accuracy, prog_bar=True, on_step=True, on_epoch=True)
        self.log('learning_rate', self.optimizers().param_groups[0]['lr'], prog_bar=True, on_step=True)
        
        return {"loss": total_loss, "accuracy": accuracy}

    def on_train_epoch_end(self):
        if 'vaihingen' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'potsdam' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        else:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union())
            F1 = np.nanmean(self.metrics_train.F1())

        OA = np.nanmean(self.metrics_train.OA())
        iou_per_class = self.metrics_train.Intersection_over_Union()
        eval_value = {'mIoU': mIoU*100.0,
                      'F1': F1*100.0,
                      'OA': OA*100.0}
        print('训练集指标:', eval_value)

        iou_value = {}
        for class_name, iou in zip(self.config.classes, iou_per_class):
            iou_value[class_name] = iou*100.0
        print('各类别IoU:', iou_value)
        
        log_dict = {
            'train_mIoU': mIoU*100.0, 
            'train_F1': F1*100.0, 
            'train_OA': OA*100.0,
            'train_loss_epoch': np.mean(self.train_losses)
        }
        
        for class_name, iou in zip(self.config.classes, iou_per_class):
            log_dict[f'train_IoU_{class_name}'] = iou*100.0
            
        self.log_dict(log_dict, prog_bar=True)
        
        self.metrics_train.reset()
        self.train_losses = []

    def validation_step(self, batch, batch_idx):
        img, mask = batch['img'], batch['gt_semantic_seg']
        prediction = self.forward(img)
        
        pre_mask = nn.Softmax(dim=1)(prediction)
        pre_mask = pre_mask.argmax(dim=1)
        for i in range(mask.shape[0]):
            self.metrics_val.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())

        loss_val = self.loss(prediction, mask)
        
        self.val_losses.append(loss_val.item())
        
        accuracy_val = (pre_mask == mask).float().mean()
        
        self.log('val_loss', loss_val, prog_bar=True, on_step=True, on_epoch=True)
        self.log('val_accuracy', accuracy_val, prog_bar=True, on_step=True, on_epoch=True)
        
        return {"loss_val": loss_val, "accuracy_val": accuracy_val}

    def on_validation_epoch_end(self):
        if 'vaihingen' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'potsdam' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        else:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union())
            F1 = np.nanmean(self.metrics_val.F1())

        OA = np.nanmean(self.metrics_val.OA())
        iou_per_class = self.metrics_val.Intersection_over_Union()

        eval_value = {'mIoU': mIoU*100.0,
                      'F1': F1*100.0,
                      'OA': OA*100.0}
        print('验证集指标:', eval_value)
        iou_value = {}
        for class_name, iou in zip(self.config.classes, iou_per_class):
            iou_value[class_name] = iou*100.0
        print('各类别IoU:', iou_value)

        log_dict = {
            'val_mIoU': mIoU*100.0, 
            'val_F1': F1*100.0, 
            'val_OA': OA*100.0,
            'val_loss_epoch': np.mean(self.val_losses)
        }
        
        for class_name, iou in zip(self.config.classes, iou_per_class):
            log_dict[f'val_IoU_{class_name}'] = iou*100.0
            
        self.log_dict(log_dict, prog_bar=True)

        self.metrics_val.reset()
        self.val_losses = []

    def configure_optimizers(self):
        optimizer = self.config.optimizer
        lr_scheduler = self.config.lr_scheduler

        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        return self.config.train_loader

    def val_dataloader(self):
        return self.config.val_loader


def main():
    args = get_args()
    config = py2cfg(args.config_path)
    seed_everything(42)

    checkpoint_callback = ModelCheckpoint(save_top_k=config.save_top_k, monitor=config.monitor,
                                          save_last=config.save_last, mode=config.monitor_mode,
                                          dirpath=config.weights_path,
                                          filename=config.weights_name)
    logger = CSVLogger('lightning_logs', name=config.log_name)

    model = Supervision_Train(config)

    if hasattr(config, 'resume_ckpt_path') and config.resume_ckpt_path:
        model = Supervision_Train.load_from_checkpoint(config.resume_ckpt_path, config=config)

    trainer = pl.Trainer(devices=config.gpus, max_epochs=config.max_epoch, accelerator='auto',
                         check_val_every_n_epoch=config.check_val_every_n_epoch,
                         callbacks=[checkpoint_callback], strategy='auto',
                         logger=logger,
                         accumulate_grad_batches=getattr(config, 'accumulate_grad_batches', 1),
                         precision=getattr(config, 'precision', 32))
    trainer.fit(model=model, ckpt_path=config.resume_ckpt_path)


if __name__ == "__main__":
   main()
