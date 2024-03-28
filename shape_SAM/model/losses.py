import torch
import torch.nn as nn
import torch.nn.functional as F

ALPHA = 0.8
GAMMA = 2


class FocalLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        inputs = F.sigmoid(inputs)
        inputs = torch.clamp(inputs, min=0, max=1)
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        BCE = F.binary_cross_entropy(inputs, targets, reduction='none')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP)**gamma * BCE
        focal_loss = focal_loss.mean()

        return focal_loss


class DiceLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)
        inputs = torch.clamp(inputs, min=0, max=1)
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice
    
    

class Calc_iou(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred_mask: torch.Tensor, gt_mask: torch.Tensor):
      pred_mask = torch.round(pred_mask - 0.5)
      pred_mask = torch.clamp(pred_mask, min=0.0, max=1.0).float()
      intersection = torch.sum(torch.mul(pred_mask, gt_mask), dim=(2, 3))
      union = torch.sum(pred_mask, dim=(2,3)) + torch.sum(gt_mask, dim=(2, 3)) - intersection
      epsilon = 1e-7
      batch_iou = intersection / (union + epsilon)

      return batch_iou
