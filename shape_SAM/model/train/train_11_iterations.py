import time
import torch
import numpy as np
import lightning as L
import torch.nn.functional as F
from .utils import (
    AverageMeter,
    validate,
    save,
    print_and_log_metrics
)
from .losses import (
    IoULoss,
    DiceLoss,
    FocalLoss
)
from ..model import shape_SAM
from ..segment_anything.utils.transforms import ResizeLongestSide
from box import Box
from lightning.fabric.fabric import _FabricOptimizer
from torch.utils.data import DataLoader


def train_11_iterations(
    cfg: Box,
    fabric: L.Fabric,
    model: shape_SAM,
    optimizer: _FabricOptimizer,
    scheduler: _FabricOptimizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
):
    """
    The SAM training loop.
    This function is used to train the model with 11 iterations as described in the paper.
    """

    focal_loss = FocalLoss(alpha=cfg.opt.focal_alpha, gamma=cfg.opt.focal_gamma)
    dice_loss = DiceLoss()
    iou_loss = IoULoss()

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

        for iteration in range(1, 12): # Fa 11 iterazioni per ogni epoca come scritto nel paper

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

                    p, l = calc_points_train(pred_masks,  data["gt_masks"], model.model.image_encoder.img_size, data["original_size"], fabric.device, cfg)
                    new_point_coords.append(p)
                    new_point_labels.append(l)

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

                print_and_log_metrics(fabric, cfg, epoch, iter, batch_time, data_time, focal_losses, dice_losses, iou_losses, total_losses, best_score, train_dataloader)

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