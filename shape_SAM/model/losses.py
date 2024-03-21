import torch
import torch.nn as nn
import torch.nn.functional as F

ALPHA = 0.8
GAMMA = 2


class FocalLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super().__init__()

    # PROBLEMA: viene applicata una riduzione media nel calcolo dell'entropia incrociata binaria (BCE), che risulta in BCE per l'intera maschera. Ciò porta al calcolo dei coefficienti di focus a livello di istanza.
    # def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
    #     inputs = F.sigmoid(inputs)
    #     inputs = torch.clamp(inputs, min=0, max=1)
    #     #flatten label and prediction tensors
    #     inputs = inputs.view(-1)
    #     targets = targets.view(-1)

    #     BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
    #     BCE_EXP = torch.exp(-BCE)
    #     focal_loss = alpha * (1 - BCE_EXP)**gamma * BCE

    #     return focal_loss
    
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
