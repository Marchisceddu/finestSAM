import time
import torch
import lightning as L
import matplotlib.pyplot as plt
import torch.nn.functional as F
from .utils import (
    AverageMeter,
    Metrics,
    validate,
    print_and_log_metrics,
    save
)
from .losses import (
    CalcIoU,
    DiceLoss,
    FocalLoss
)
from ..model import FinestSAM
from box import Box
from lightning.fabric.fabric import _FabricOptimizer
from torch.utils.data import DataLoader

#ELIMINARE
from ..predictions.utils import (
    show_mask,
)
import os

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
    calc_iou = CalcIoU()

    if cfg.prompts.use_logits: cfg.prompts.use_masks = False
    epoch_logits = []

    val_score = 0.

    out_plots = os.path.join(cfg.out_dir, "plots")
    os.makedirs(out_plots, exist_ok=True)
    metrics = {
        "focal": [],
        "dice": [],
        "space_iou": [],
        "total": [],
        "iou": [],
        "iou_pred": [],
    }
    
    for epoch in range(1, cfg.num_epochs+1):
        # Initialize the meters
        epoch_metrics = Metrics()
        end = time.time()

        for iter, batched_data in enumerate(train_dataloader):
            epoch_metrics.data_time.update(time.time()-end)

            # Se presenti e selezionati dalle impostazioni passa i logits dell'epoca precedente
            if epoch > 1 and cfg.prompts.use_logits: [data.update({"mask_inputs": logits.clone().detach().unsqueeze(1)}) for data, logits in zip(batched_data, epoch_logits)]

            # Forward pass
            outputs = model(batched_input=batched_data, multimask_output=cfg.multimask_output, are_logits=cfg.prompts.use_logits)

            batched_pred_masks = []
            batched_iou_predictions = []
            batched_logits = []
            for item in outputs:
                # Take mask, iou_prediction and low_res_logits from the output
                batched_pred_masks.append(item["masks"])
                batched_iou_predictions.append(item["iou_predictions"])
                batched_logits.append(item["low_res_logits"])

            num_masks = sum(len(pred_mask) for pred_mask in batched_pred_masks) # NON HA SENSO CALCOLARE LE MASCHERE QUA SE POI LA LOSS VIENE ESEGUITA PER TUTTE LE IMMAGINI

            iter_metrics = {
                "loss_focal": torch.tensor(0., device=fabric.device),
                "loss_dice": torch.tensor(0., device=fabric.device),
                "loss_iou": torch.tensor(0., device=fabric.device),
                "iou": torch.tensor(0., device=fabric.device),
                "iou_pred": torch.tensor(0., device=fabric.device),
            }

            # Compute the losses
            for data, pred_masks, iou_predictions, logits in zip(batched_data, batched_pred_masks, batched_iou_predictions, batched_logits):

                if cfg.multimask_output:
                    # Separa la tripla di maschere predette
                    separated_masks = torch.unbind(pred_masks, dim=1)
                    separated_scores = torch.unbind(iou_predictions, dim=1)
                    separated_logits = torch.unbind(logits, dim=1)

                    # Seleziona solo quella con il punteggio migliore
                    best_index = torch.argmax(torch.tensor([torch.mean(score) for score in separated_scores]))
                    pred_masks = separated_masks[best_index]
                    iou_predictions = separated_scores[best_index]
                    logits = separated_logits[best_index]

                if cfg.prompts.use_logits: epoch_logits.append(logits)

                # Aggiorna metriche
                batch_iou = calc_iou(pred_masks, data["gt_masks"])
                iter_metrics["iou"] += torch.mean(batch_iou)
                iter_metrics["iou_pred"] += torch.mean(iou_predictions)

                mask1 = torch.tensor([[[1, 0, 1, 0, 1, 0],
                                    [0, 1, 0, 1, 0, 1],
                                    [1, 0, 1, 0, 1, 0],
                                    [0, 1, 0, 1, 0, 1],
                                    [1, 0, 1, 0, 1, 0],
                                    [0, 1, 0, 1, 0, 1]],
                                    [[0, 1, 0, 1, 0, 1],
                                    [1, 0, 1, 0, 1, 0],
                                    [0, 1, 0, 1, 0, 1],
                                    [1, 0, 1, 0, 1, 0],
                                    [0, 1, 0, 1, 0, 1],
                                    [1, 0, 1, 0, 1, 0]]], dtype=torch.float32)

                mask2 = torch.tensor([[[1, 0, 1, 0, 1, 0],
                                    [0, 1, 0, 1, 0, 1],
                                    [1, 0, 1, 0, 1, 0],
                                    [0, 1, 0, 1, 0, 1],
                                    [1, 0, 1, 0, 1, 0],
                                    [0, 1, 0, 1, 0, 1]],
                                    [[0, 1, 0, 1, 0, 1],
                                    [1, 0, 1, 0, 1, 0],
                                    [0, 1, 0, 1, 0, 1],
                                    [1, 0, 1, 0, 1, 0],
                                    [0, 1, 0, 1, 0, 1],
                                    [1, 0, 1, 0, 1, 0]]], dtype=torch.float32)
                
                print(calc_iou(mask1, mask2))
                print(focal_loss(mask1, mask2, 2))
                print(dice_loss(mask1, mask2, 2))

                # Calcola loss
                iter_metrics["loss_focal"] += focal_loss(pred_masks, data["gt_masks"], len(pred_masks))
                iter_metrics["loss_dice"] += dice_loss(pred_masks, data["gt_masks"], len(pred_masks))
                iter_metrics["loss_iou"] += F.mse_loss(iou_predictions, batch_iou, reduction='mean')

                plt.imshow(data["imo"])
                for i, mask in enumerate(pred_masks > 0):
                    show_mask(mask.clone().detach(), plt.gca(), seed=i)
                plt.axis('off')
                plt.savefig(os.path.join(cfg.out_dir, "p.png"))
                plt.clf()

            loss_total = cfg.losses.focal_ratio * iter_metrics["loss_focal"] + cfg.losses.dice_ratio * iter_metrics["loss_dice"] + cfg.losses.iou_ratio * iter_metrics["loss_iou"]

            # Backward pass
            optimizer.zero_grad()
            fabric.backward(loss_total)
            optimizer.step()
            scheduler.step()

            epoch_metrics.batch_time.update(time.time() - end)
            end = time.time()

            # Update the meters
            epoch_metrics.focal_losses.update(iter_metrics["loss_focal"].item(), cfg.batch_size)
            epoch_metrics.dice_losses.update(iter_metrics["loss_dice"].item(), cfg.batch_size)
            epoch_metrics.space_iou_losses.update(iter_metrics["loss_iou"].item(), cfg.batch_size)
            epoch_metrics.total_losses.update(loss_total.item(), cfg.batch_size)
            epoch_metrics.ious.update(iter_metrics["iou"].item()/cfg.batch_size, cfg.batch_size)
            epoch_metrics.ious_pred.update(iter_metrics["iou_pred"].item()/cfg.batch_size, cfg.batch_size)

            print_and_log_metrics(fabric, cfg, epoch, iter, epoch_metrics, train_dataloader)

        # Validate the model
        if (cfg.eval_interval > 0 and epoch % cfg.eval_interval == 0) or (epoch == cfg.num_epochs):
            val_score = validate(fabric, cfg, model, val_dataloader, epoch, val_score)
            #save(fabric, model, cfg.sav_dir, "c")
        
        if epoch % 50 == 0:
            save(fabric, model, cfg.sav_dir, f"{epoch}")

        # Aggiorna le metriche per i plot
        # metrics["dice"].append(epoch_metrics.dice_losses.avg)
        # metrics["focal"].append(epoch_metrics.focal_losses.avg)
        # metrics["space_iou"].append(epoch_metrics.space_iou_losses.avg)
        # metrics["total"].append(epoch_metrics.total_losses.avg)
        # metrics["iou"].append(epoch_metrics.ious.avg)
        # metrics["iou_pred"].append(epoch_metrics.ious_pred.avg)

        # # Plot per ciascun parametro
        # plt.figure(figsize=(10, 6))
        # plt.plot(metrics["dice"], label="Dice Loss")
        # plt.title("Dice Loss")
        # plt.legend()
        # plt.savefig(os.path.join(out_plots, "dice_loss.png"))
        # plt.clf()
        # plt.close()

        # plt.figure(figsize=(10, 6))
        # plt.plot(metrics["focal"], label="Focal Loss")
        # plt.title("Focal Loss")
        # plt.legend()
        # plt.savefig(os.path.join(out_plots, "focal_loss.png"))
        # plt.clf()
        # plt.close()

        # plt.figure(figsize=(10, 6))
        # plt.plot(metrics["space_iou"], label="Space IoU Loss")
        # plt.title("Space IoU Loss")
        # plt.legend()
        # plt.savefig(os.path.join(out_plots, "space_iou_loss.png"))
        # plt.clf()
        # plt.close()

        # plt.figure(figsize=(10, 6))
        # plt.plot(metrics["total"], label="Total Loss")
        # plt.title("Total Loss")
        # plt.legend()
        # plt.savefig(os.path.join(out_plots, "total_loss.png"))
        # plt.clf()
        # plt.close()

        # plt.figure(figsize=(10, 6))
        # plt.plot(metrics["iou"], label="IoU")
        # plt.title("IoU")
        # plt.legend()
        # plt.savefig(os.path.join(out_plots, "iou.png"))
        # plt.clf()
        # plt.close()

        # plt.figure(figsize=(10, 6))
        # plt.plot(metrics["iou_pred"], label="IoU Pred")
        # plt.title("IoU Pred")
        # plt.xlabel('Epoch')
        # plt.ylabel('Value')
        # plt.legend()
        # plt.savefig(os.path.join(out_plots, "iou_pred.png"))
        # plt.clf()
        # plt.close()

        # # Plotting all metrics except 'total' with the default y-axis
        # plt.plot(metrics["dice"], label="Dice Loss")
        # plt.plot(metrics["focal"], label="Focal Loss")
        # plt.plot(metrics["space_iou"], label="Space IoU Loss")
        # plt.plot(metrics["iou"], label="IoU")
        # plt.plot(metrics["iou_pred"], label="IoU Pred")
        # plt.title("All Metrics")
        # plt.xlabel('Epoch')
        # plt.ylabel('Value')
        # plt.legend()

        # # Creating a second y-axis for the 'total' metric
        # plt2 = plt.gca().twinx()
        # plt2.plot(metrics["total"], color='black', linestyle='--', label="Total Loss")
        # plt2.set_ylabel('Total Loss')

        # plt.legend(loc='upper left')
        # plt.savefig(os.path.join(out_plots, "all_metrics.png"))
        # plt.clf()
        # plt.close()

