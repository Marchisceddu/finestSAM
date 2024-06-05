import time
import torch
import lightning as L
import torch.nn.functional as F
from .utils import (
    AverageMeter,
    validate,
    print_and_log_metrics,
    save
)
from .losses import (
    IoULoss,
    DiceLoss,
    FocalLoss
)
from ..model import FinestSAM
from box import Box
from lightning.fabric.fabric import _FabricOptimizer
from torch.utils.data import DataLoader


def train_custom(
    cfg: Box,
    fabric: L.Fabric,
    model: FinestSAM,
    optimizer: _FabricOptimizer,
    scheduler: _FabricOptimizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
):
    """The SAM training loop."""

    # Initialize the losses
    focal_loss = FocalLoss(alpha=cfg.losses.focal_alpha, gamma=cfg.losses.focal_gamma)
    dice_loss = DiceLoss()
    iou_loss = IoULoss()
    last_score = 0.

    for epoch in range(1, cfg.num_epochs + 1):
        # Initialize the meters
        batch_time = AverageMeter()
        data_time = AverageMeter()
        focal_losses = AverageMeter()
        dice_losses = AverageMeter()
        iou_losses = AverageMeter()
        total_losses = AverageMeter()
        end = time.time()

        for iter, batched_data in enumerate(train_dataloader):
            data_time.update(time.time() - end)

            # Forward pass
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

            # Compute the losses
            for pred_masks, data, iou_prediction in zip(batched_pred_masks, batched_data, iou_predictions):

                separated_masks = torch.unbind(pred_masks, dim=1)
                separated_scores = torch.unbind(iou_prediction, dim=1)

                iou_prediction_means = [torch.mean(score) for score in separated_scores]
                best_index = torch.argmax(torch.tensor(iou_prediction_means))
                pred_masks = separated_masks[best_index]
                iou_prediction = separated_scores[best_index]

                batch_iou = iou_loss(pred_masks, data["gt_masks"])
                loss_focal += focal_loss(pred_masks, data["gt_masks"], num_masks)
                loss_dice += dice_loss(pred_masks, data["gt_masks"], num_masks)

                print(batch_iou)
                loss_iou += F.mse_loss(iou_prediction, batch_iou, reduction='mean')

            loss_total = cfg.losses.focal_ratio * loss_focal + cfg.losses.dice_ratio * loss_dice + loss_iou

            # Backward pass
            optimizer.zero_grad()
            fabric.backward(loss_total)
            optimizer.step()
            scheduler.step()

            batch_time.update(time.time() - end)
            end = time.time()

            # Update the meters
            focal_losses.update(loss_focal.item(), cfg.batch_size)
            dice_losses.update(loss_dice.item(), cfg.batch_size)
            iou_losses.update(loss_iou.item(), cfg.batch_size)
            total_losses.update(loss_total.item(), cfg.batch_size)
            best_score = torch.mean(iou_prediction)

            print_and_log_metrics(fabric, cfg, epoch, iter, batch_time, data_time, focal_losses, dice_losses, iou_losses, total_losses, best_score, train_dataloader)

        # Validate the model
        if (epoch > 1 and cfg.eval_interval > 0 and epoch % cfg.eval_interval == 0) or (epoch == cfg.num_epochs):
            last_score = validate(fabric, cfg, model, val_dataloader, epoch, last_score)
            #save(fabric, model, cfg.sav_dir, "c")