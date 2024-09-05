import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    
    def __init__(self, smooth: int = 1e-7):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, num_masks: int) -> torch.Tensor:
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
        inputs = inputs.sigmoid()    

        inputs = inputs.flatten(1)
        targets = targets.flatten(1)

        numerator = 2 * (inputs * targets).sum(1)
        denominator = inputs.sum(1) + targets.sum(1)
        loss = 1 - (numerator + self.smooth) / (denominator + self.smooth)
        
        return loss.sum() / num_masks


class FocalLoss(nn.Module):

    def __init__(self, gamma: int, alpha: float = -1):
        """
        Args:
            alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, num_masks: int) -> torch.Tensor:
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            num_masks: Number of masks in the batch
            
            Returns:
                Loss tensor
        """
        prob = inputs.sigmoid()

        inputs = inputs.flatten(1)
        prob = prob.flatten(1)
        targets = targets.flatten(1)

        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean(1).sum() / num_masks
    

class CalcIoU(nn.Module):

    def __init__(self, smooth: int = 1e-7):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the Intersection over Union (IoU) loss.
        Args:
            pred_mask: A float tensor of arbitrary shape.
                    The predictions for each example.
            gt_mask: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        """
        inputs = (inputs > 0).float()

        inputs = inputs.flatten(1)
        targets = targets.flatten(1)

        intersection = (inputs * targets).sum(1)
        union = inputs.sum(1) + targets.sum(1) - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)

        return iou