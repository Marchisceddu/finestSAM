import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .config import cfg

ALPHA = 0.7
GAMMA = 2


class DiceLoss(nn.Module):
    
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets,num_masks, smooth=1):
        """
        Compute the DICE loss, similar to generalized IOU for masks
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            num_masks: Number of masks in the batch
        """

        inputs = inputs.squeeze()
        targets = targets.squeeze()

        inputs = inputs.sigmoid()
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)

        numerator = 2 * (inputs * targets).sum(1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + smooth) / (denominator + smooth)
        
        return loss.sum() / num_masks



class FocalLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, num_masks, alpha=ALPHA, gamma=GAMMA):
        
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            num_masks: Number of masks in the batch
            
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = -1 (no weighting).
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
            Returns:
                Loss tensor
        """
        inputs = inputs.squeeze()
        targets = targets.squeeze()

        prob = inputs.sigmoid()
        inputs = inputs.flatten(1)
        prob = prob.flatten(1)
        targets = targets.flatten(1)

        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean(1).sum() / num_masks
    

class Calc_iou(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred_mask: torch.Tensor, gt_mask: torch.Tensor):
         
        pred_mask = (pred_mask >= 0.5).float()
        pred_mask = pred_mask.squeeze()
        gt_mask = gt_mask.squeeze()
        
        pred_mask = pred_mask.flatten(1)
        gt_mask = gt_mask.flatten(1)
            
        intersection = (pred_mask * gt_mask).sum(1)
        union = pred_mask.sum(1) + gt_mask.sum(1) - intersection
        epsilon = 1e-7
        batch_iou = intersection / (union + epsilon)

        return batch_iou