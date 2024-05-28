import os
import time
import torch
import numpy as np
import lightning as L
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from .segment_anything.utils.transforms import ResizeLongestSide
from .model import shape_SAM
from .utils import AverageMeter
from .losses import (
    Calc_iou,
    DiceLoss,
    FocalLoss
)
from typing import Tuple
from box import Box
from lightning.fabric.fabric import _FabricOptimizer
from torch.utils.data import DataLoader


def configure_opt(cfg: Box, model: shape_SAM) -> Tuple[_FabricOptimizer, _FabricOptimizer]:

    def lr_lambda(step):
        if step < cfg.opt.warmup_steps:
            return step / cfg.opt.warmup_steps
        elif step < cfg.opt.steps[0]:
            return 1.0
        elif step < cfg.opt.steps[1]:
            return 1 / cfg.opt.decay_factor
        else:
            return 1 / (cfg.opt.decay_factor**2)

    optimizer = torch.optim.Adam(model.model.parameters(), lr=cfg.opt.learning_rate, weight_decay=cfg.opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


def validate(
        fabric: L.Fabric, 
        cfg: Box,
        model: shape_SAM, 
        val_dataloader: DataLoader, 
        epoch: int,
    ): 
    model.eval()
    ious = AverageMeter()
    
    for iter, batched_data in enumerate(val_dataloader):

        predictor = model.get_predictor()
        pred_masks = []
        gt_masks = [data["gt_masks"] for data in batched_data]  
        num_images = len(batched_data)
        for data in batched_data:
            predictor.set_image(data["image"])
            masks, _, _ = predictor.predict_torch(
                point_coords=data["point_coords"],
                point_labels=data["point_labels"],
                boxes=None,
                multimask_output=False,
            )
            pred_masks.append(masks)

        for pred_mask, gt_mask in zip(pred_masks, gt_masks):
            batch_stats = smp.metrics.get_stats(
                pred_mask,
                gt_mask.int(),
                mode='binary',
                threshold=0.5,
            )
            batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
            ious.update(batch_iou, num_images)
        fabric.print(
            f'Val: [{epoch}] - [{iter}/{len(val_dataloader)}]: Mean IoU: [{ious.avg:.4f}]'
        )

    fabric.print(f'Validation [{epoch}]: Mean IoU: [{ious.avg:.4f}]')
    save(fabric, model, cfg, epoch)
    model.train()


def save(
    fabric: L.Fabric, 
    model: shape_SAM, 
    out_dir: str,
    epoch: int,
):
    fabric.print(f"Saving checkpoint to {out_dir}")
    state_dict = model.model.state_dict()
    if fabric.global_rank == 0:
        torch.save(state_dict, os.path.join(out_dir, f"epoch-{epoch:06d}-ckpt.pth"))


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
    calc_iou = Calc_iou()

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

                batch_iou = calc_iou(pred_masks, data["gt_masks"])
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


def train_11_iterations(
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
    calc_iou = Calc_iou()

    new_logits = []
    new_point_coords = []
    new_point_labels = []

    for epoch in range(1, cfg.num_epochs + 1):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        focal_losses = AverageMeter()
        dice_losses = AverageMeter()
        iou_losses = AverageMeter()
        total_losses = AverageMeter()
        end = time.time()

        random_number = np.random.randint(2, 11)

        for iteration in range(1, 12): # Fa 11 iterazioni per ogni epoca come scritto nel paper # opt

            are_logits = True if iteration > 1 else False
            only_logits = True if iteration == random_number or iteration == 11 else False

            for iter, batched_data in enumerate(train_dataloader):
                data_time.update(time.time() - end)

                # After the first iteration, the model will receive the low_res_logits as input
                if iteration > 1:
                    for sample, logits, point_coords, point_labels in zip(batched_data, new_logits, new_point_coords, new_point_labels):
                        sample["mask_inputs"] = logits.clone().detach().unsqueeze(1)

                        if only_logits:
                            sample["point_coords"] = None
                            sample["point_labels"] = None
                        else:
                            if sample["point_coords"] is None:
                                save(fabric, model, cfg, epoch)

                            sample["point_coords"] = point_coords
                            sample["point_labels"] = point_labels

                    # Reset the lists
                    new_point_coords = []
                    new_point_labels = []

                outputs = model(batched_input=batched_data, multimask_output=True, are_logits=are_logits)

                batched_pred_masks = []
                iou_predictions = []
                raw_logits = []
                for item in outputs:
                    # Take mask, iou_prediction and low_res_logits from the output
                    batched_pred_masks.append(item["masks"])
                    iou_predictions.append(item["iou_predictions"])
                    raw_logits.append(item["low_res_logits"])

                num_masks = sum(len(pred_mask) for pred_mask in batched_pred_masks)

                loss_focal = torch.tensor(0., device=fabric.device)
                loss_dice = torch.tensor(0., device=fabric.device)
                loss_iou = torch.tensor(0., device=fabric.device)

                for pred_masks, data, iou_prediction, logits in zip(batched_pred_masks, batched_data, iou_predictions, raw_logits):

                    separated_masks = torch.unbind(pred_masks, dim=1) # 3 output masks
                    separated_scores = torch.unbind(iou_prediction, dim=1) # scores for each mask
                    separated_logits = torch.unbind(logits, dim=1)

                    iou_prediction_means = [torch.mean(score) for score in separated_scores]
                    best_index = torch.argmax(torch.tensor(iou_prediction_means))

                    pred_masks = separated_masks[best_index]
                    iou_prediction = separated_scores[best_index]
                    new_logits.append(separated_logits[best_index])

                    p, l = calc_points_train(pred_masks,  data["gt_masks"], model.model.image_encoder.img_size, data["original_size"], fabric.device, cfg) # opt
                    new_point_coords.append(p)
                    new_point_labels.append(l)

                    batch_iou = calc_iou(pred_masks, data["gt_masks"])
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
                            f'Iteration: [{iteration}]'
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


def calc_points_train(
        pred_mask: torch.Tensor, 
        gt_mask: torch.Tensor, 
        img_size: int,
        original_size: tuple, 
        device: torch.device, 
        cfg: Box
    ):
    transform = ResizeLongestSide(img_size)

    pred_mask = (pred_mask >= 0.5).float()
    pred_mask = pred_mask.squeeze()
    gt_mask = gt_mask.squeeze()

    point_coords = []
    point_labels = []

    positive_points = cfg.dataset.positive_points
    negative_points = cfg.dataset.negative_points
    foreground_coordinates_list = []
    background_coordinates_list= []

    for i, (mask, gtm) in enumerate(zip(pred_mask, gt_mask)):
         # Estrazione dei nuovi punti di foreground e background
        new_foreground_points = (mask == 0.) & (gtm == 1.)
        new_background_points = (mask == 1.) & (gtm == 0.)

        # Individua le coordinate dei nuovi punti di foreground e background
        foreground_coordinates_list.append(torch.nonzero(new_foreground_points, as_tuple=True))
        background_coordinates_list.append(torch.nonzero(new_background_points, as_tuple=True))

        if len(foreground_coordinates_list[i][0]) < positive_points:
            positive_points = len(foreground_coordinates_list[i][0]) 
        if len(background_coordinates_list[i][0]) < negative_points:
            negative_points = len(background_coordinates_list[i][0])
    for mask, gtm, foreground_coordinates, background_coordinates in zip(pred_mask, gt_mask, foreground_coordinates_list, background_coordinates_list):

        if not (positive_points == 0):            
            temp_list_point = []
            for i in range(0, positive_points):
                idx = np.random.randint(0, len(foreground_coordinates[0]))
                temp_list_point.append([foreground_coordinates[1][idx].item(), foreground_coordinates[0][idx].item()])
            new_foreground_points = temp_list_point.copy()
            list_label_1 = [1] * len(new_foreground_points)
        else:
            new_foreground_points = []
            list_label_1 = []

        if not (negative_points == 0):
            temp_list_point = []
            for i in range(0, negative_points):
                idx = np.random.randint(0, len(background_coordinates[0]))
                temp_list_point.append([background_coordinates[1][idx].item(), background_coordinates[0][idx].item()])
            new_background_points = temp_list_point.copy()
            list_label_0 = [0] * len(new_background_points)
        else:
            new_background_points = []
            list_label_0 = []

        actual_point_coords = new_foreground_points + new_background_points
        actual_point_labels = list_label_1 + list_label_0

        point_coords.append(actual_point_coords)
        point_labels.append(actual_point_labels)

    # Normalizzo i punti
    if positive_points + negative_points > 0:
      point_coords = transform.apply_coords(np.array(point_coords), original_size)
      point_coords = torch.tensor(np.stack(point_coords, axis=0)).to(device)

      point_labels = torch.as_tensor(point_labels, dtype=torch.int).to(device)
    else:
      point_coords = None
      point_labels = None

    return  point_coords, point_labels