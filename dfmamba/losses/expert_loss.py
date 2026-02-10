import torch
import torch.nn as nn
import torch.nn.functional as F
from .useful_loss import OHEM_CELoss, EdgeLoss
from .dice import DiceLoss

class RSExpertLoss(nn.Module):
    """
    Expert Loss for Remote Sensing Semantic Segmentation.
    Combines:
    1. OHEM (Online Hard Example Mining): Focuses on hard-to-classify pixels (e.g., small cars, clutter).
    2. Dice Loss: Directly optimizes the IoU metric.
    3. Edge Loss: Enforces boundary consistency, crucial for buildings and roads.
    4. Multi-Scale Deep Supervision: Provides gradient guidance to intermediate layers.
    """
    def __init__(self, ignore_index=6, aux_weights=[0.4, 0.4, 0.4], class_weights=None):
        super().__init__()
        self.aux_weights = aux_weights
        self.ignore_index = ignore_index
        
        # Handle class weights
        if class_weights is not None:
             if not isinstance(class_weights, torch.Tensor):
                 class_weights = torch.tensor(class_weights).float()
             if torch.cuda.is_available():
                 class_weights = class_weights.cuda()
        
        # Components
        self.ohem = OHEM_CELoss(thresh=0.7, ignore_index=ignore_index, weight=class_weights)
        self.dice = DiceLoss(smooth=0.05, ignore_index=ignore_index)
        # We use the compute_edge_loss method from EdgeLoss helper, or a lightweight version
        self.edge_helper = EdgeLoss(ignore_index=ignore_index) 
        
        # Aux Loss (Standard CE is sufficient for guidance)
        self.aux_criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)

    def forward(self, inputs, targets):
        # Inputs can be (main_logits, [aux_list]) or just main_logits
        if isinstance(inputs, tuple) and len(inputs) == 2:
            main_logits, aux_logits_list = inputs
        else:
            main_logits = inputs
            aux_logits_list = []

        # --- 1. Main Branch Loss ---
        # OHEM Cross Entropy
        loss_ohem = self.ohem(main_logits, targets)
        
        # Dice Loss
        loss_dice = self.dice(main_logits, targets)
        
        # Edge Loss
        loss_edge = self.edge_helper.compute_edge_loss(main_logits, targets)
        
        # Weighted Sum: OHEM is dominant, Dice helps metric, Edge refines details
        # Weights: OHEM=1.0, Dice=1.0, Edge=1.0 (Adjust based on convergence)
        total_loss = loss_ohem + loss_dice + loss_edge

        # --- 2. Deep Supervision (Auxiliary) Loss ---
        if self.training and isinstance(aux_logits_list, list) and len(aux_logits_list) > 0:
            aux_loss_sum = 0.0
            for i, aux_logit in enumerate(aux_logits_list):
                # Apply corresponding weight (default to 0.4 if index exceeds)
                w = self.aux_weights[i] if i < len(self.aux_weights) else 0.4
                
                # Use standard CE for aux to keep it stable
                aux_loss = self.aux_criterion(aux_logit, targets)
                aux_loss_sum += w * aux_loss
            
            total_loss += aux_loss_sum

        return total_loss

