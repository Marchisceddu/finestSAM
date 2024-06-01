import time
import torch
import lightning as L
import torch.nn.functional as F
from .utils import (
    AverageMeter,
    validate
)
from .losses import (
    IoULoss,
    DiceLoss,
    FocalLoss
)
from ..model import shape_SAM
from box import Box
from lightning.fabric.fabric import _FabricOptimizer
from torch.utils.data import DataLoader


def train_custom(
    cfg: Box,
    fabric: L.Fabric,
    model: shape_SAM,
    optimizer: _FabricOptimizer,
    scheduler: _FabricOptimizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
):
    """The SAM training loop."""

    focal_loss = FocalLoss()
    dice_loss = DiceLoss()
    iou_loss = IoULoss()

    for epoch in range(1, cfg.num_epochs + 1):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        focal_losses = AverageMeter()
        dice_losses = AverageMeter()
        iou_losses = AverageMeter()
        total_losses = AverageMeter()
        end = time.time()

        for iter, batched_data in enumerate(train_dataloader):
            data_time.update(time.time() - end)

            outputs = model(batched_input=batched_data, multimask_output=True)

            batched_pred_masks = []
            iou_predictions = []
            logits = []
            for item in outputs:
                # Take mask, iou_prediction and low_res_logits from the output
                batched_pred_masks.append(item["masks"])
                iou_predictions.append(item["iou_predictions"])
                logits.append(item["low_res_logits"])

            num_masks = sum(len(pred_mask) for pred_mask in batched_pred_masks)

            loss_focal = torch.tensor(0., device=fabric.device)
            loss_dice = torch.tensor(0., device=fabric.device)
            loss_iou = torch.tensor(0., device=fabric.device)

            for pred_masks, data, iou_prediction in zip(batched_pred_masks, batched_data, iou_predictions):

                separated_masks = torch.unbind(pred_masks, dim=1) # 3 output masks
                separated_scores = torch.unbind(iou_prediction, dim=1) # scores for each mask

                iou_prediction_means = [torch.mean(score) for score in separated_scores]
                best_index = torch.argmax(torch.tensor(iou_prediction_means))

                pred_masks = separated_masks[best_index]
                iou_prediction = separated_scores[best_index]

                batch_iou = iou_loss(pred_masks, data["gt_masks"])
                loss_focal += focal_loss(pred_masks, data["gt_masks"], num_masks)
                loss_dice += dice_loss(pred_masks, data["gt_masks"], num_masks)
                loss_iou += F.mse_loss(iou_prediction, batch_iou, reduction='mean')

            focal_alpha = 20.
            loss_total = focal_alpha * loss_focal + loss_dice + loss_iou

            optimizer.zero_grad()
            fabric.backward(loss_total)
            optimizer.step()
            scheduler.step()

            batch_time.update(time.time() - end)
            end = time.time()

            focal_losses.update(loss_focal.item(), cfg.batch_size)
            dice_losses.update(loss_dice.item(), cfg.batch_size)
            iou_losses.update(loss_iou.item(), cfg.batch_size)
            total_losses.update(loss_total.item(), cfg.batch_size)
            best_score = torch.mean(iou_prediction)

            fabric.print(f'Epoch: [{epoch}][{iter+1}/{len(train_dataloader)}]'
                         f' | Time [{batch_time.val:.3f}s ({batch_time.avg:.3f}s)]'
                         f' | Data [{data_time.val:.3f}s ({data_time.avg:.3f}s)]'
                         f' | a Focal Loss [{focal_alpha * focal_losses.val:.4f} ({focal_alpha * focal_losses.avg:.4f})]'
                         f' | Dice Loss [{dice_losses.val:.4f} ({dice_losses.avg:.4f})]'
                         f' | IoU Loss [{iou_losses.val:.4f} ({iou_losses.avg:.4f})]'
                         f' | Total Loss [{total_losses.val:.4f} ({total_losses.avg:.4f})]'
                         f' | Mask quality [({best_score:.4f})]')
            steps = epoch * len(train_dataloader) + iter
            log_info = {
                'Loss': total_losses.val,
                'alpha focal loss': focal_alpha * focal_losses.val,
                'dice loss': dice_losses.val,
            }
            fabric.log_dict(log_info, step=steps)

        if (epoch > 1 and cfg.eval_interval > 0 and epoch % cfg.eval_interval == 0) or (epoch == cfg.num_epochs):
            validate(fabric, cfg, model, val_dataloader, epoch)
