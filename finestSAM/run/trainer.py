import os
import time
import torch
import lightning as L
import torch.nn.functional as F
import torch.nn as nn
from box import Box
from torch.utils.data import DataLoader
from lightning.fabric.fabric import _FabricOptimizer
from lightning.fabric.loggers import TensorBoardLogger
from monai.losses import DiceLoss, FocalLoss
from monai.metrics import compute_iou, compute_dice
from finestSAM.utils import (
    Metrics,
    WarmupReduceLROnPlateau,
    configure_opt,
    validate,
    compute_dataset_stats,
    print_and_log_metrics,
    plot_history,
    save_train_metrics,
    save_val_metrics,
    log_event
)
from finestSAM.model.model import FinestSAM
from finestSAM.data.dataset import load_dataset


def call_train(cfg: Box, dataset_path: str):
    """
    Entry point for training the model.
    
    Args:
        cfg (Box): The configuration file.
        dataset_path (str): The path to the dataset.
    """

    losses_enabled = any([cfg.losses.focal.enabled, cfg.losses.dice.enabled, cfg.losses.iou.enabled, cfg.losses.cross_entropy.enabled])
    metrics_enabled = any([cfg.metrics.iou.enabled, cfg.metrics.dice.enabled])
    
    if not losses_enabled and not metrics_enabled:
        raise ValueError("At least one loss or one metric must be enabled in the configuration to start training.")

    loggers = [TensorBoardLogger(cfg.sav_dir, name="loggers_finestSAM")]

    fabric = L.Fabric(accelerator=cfg.device,
                      devices=cfg.num_devices,
                      strategy="auto",
                      num_nodes=cfg.num_nodes,
                      precision=cfg.precision, 
                      loggers=loggers)

    fabric.launch(train, cfg, dataset_path)


def train(fabric: L.Fabric, *args, **kwargs):
    """
    Main training function.
    
    Args:
        fabric (L.Fabric): The lightning fabric.
        *args: The positional arguments:
            [0] - cfg (Box): The configuration file.
            [1] - dataset_path (str): The path to the dataset.
        **kwargs: The keyword arguments:
            not used.
    """
    # Get the arguments
    cfg = args[0]
    dataset_path = args[1]

    fabric.seed_everything(cfg.seed_device)

    if fabric.global_rank == 0: 
        os.makedirs(os.path.join(cfg.sav_dir, "loggers_finestSAM"), exist_ok=True)

    # Set matmul precision for Tensor Cores
    torch.set_float32_matmul_precision(cfg.matmul_precision)

    # Load the dataset
    img_size = cfg.model.get("img_size", 1024)
    train_data, val_data = load_dataset(cfg, img_size, dataset_path, fabric=fabric)
    train_data = fabric._setup_dataloader(train_data)
    val_data = fabric._setup_dataloader(val_data)

    # Auto-compute stats if requested
    if cfg.model.get("compute_stats", False):
        if cfg.model.pixel_mean is None or cfg.model.pixel_std is None:
            mean, std = compute_dataset_stats(train_data, fabric)
            cfg.model.pixel_mean = mean
            cfg.model.pixel_std = std
            fabric.print(f"Computed dataset stats: Mean={mean}, Std={std}")
            log_event(cfg.out_dir, f"Computed dataset {dataset_path} stats: Mean={mean}, Std={std}")

    with fabric.device:
        model = FinestSAM(cfg)
        model.setup()
        model.train()
        model.to(fabric.device)

    # Configure the optimizer and scheduler
    optimizer, scheduler = configure_opt(cfg, model, fabric)
    model, optimizer = fabric.setup(model, optimizer)

    train_loop(cfg, fabric, model, optimizer, scheduler, train_data, val_data) 

def train_loop(
    cfg: Box,
    fabric: L.Fabric,
    model: FinestSAM,
    optimizer: _FabricOptimizer,
    scheduler: _FabricOptimizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
):
    """
    The SAM training loop.
    """

    # Initialize the losses
    focal_loss = FocalLoss(gamma=cfg.losses.focal.gamma, reduction="mean")
    dice_loss = DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    last_lr = scheduler.get_last_lr()
    best_val_iou = 0.
    best_val_dsc = 0.
    best_iou_ckpt_path = ""
    best_dsc_ckpt_path = ""

    os.makedirs(cfg.out_dir, exist_ok=True)
    metrics_history = {
        "total_loss": [],
        "focal_loss": [],
        "dice_loss": [],
        "ce_loss": [],
        "iou_loss": [],
        "train_iou": [],
        "train_dsc": [],
        "val_iou": [],
        "val_dsc": [],
        "epochs": [],
    }

    # Initial validation
    val_results = {}
    if cfg.eval_interval > 0:
        val_results = validate(fabric, cfg, model, val_dataloader, 0)
        if fabric.global_rank == 0:
            save_val_metrics(0, val_results, cfg.out_dir)

    for epoch in range(1, cfg.num_epochs+1):
        # Initialize the meters
        epoch_metrics = Metrics()
        end = time.time()

        for iter, batched_data in enumerate(train_dataloader):
            torch.cuda.empty_cache()
            epoch_metrics.data_time.update(time.time()-end)

            outputs = model(batched_input=batched_data, multimask_output=cfg.multimask_output)

            batched_pred_masks = []
            batched_iou_predictions = []
            for item in outputs:
                batched_pred_masks.append(item["masks"])
                batched_iou_predictions.append(item["iou_predictions"])

            batch_size = len(batched_data)

            iter_metrics = {
                "loss_focal": torch.tensor(0., device=fabric.device),
                "loss_dice": torch.tensor(0., device=fabric.device),
                "loss_iou": torch.tensor(0., device=fabric.device),
                "loss_ce": torch.tensor(0., device=fabric.device),
                "iou": torch.tensor(0., device=fabric.device),
                "iou_pred": torch.tensor(0., device=fabric.device),
                "dsc": torch.tensor(0., device=fabric.device),
            }

            # Compute the losses
            for data, pred_masks, iou_predictions in zip(batched_data, batched_pred_masks, batched_iou_predictions):

                if cfg.multimask_output:
                    separated_masks = torch.unbind(pred_masks, dim=1)
                    separated_scores = torch.unbind(iou_predictions, dim=1)

                    best_index = torch.argmax(torch.tensor([torch.mean(score) for score in separated_scores]))
                    pred_masks = separated_masks[best_index]
                    iou_predictions = separated_scores[best_index]
                else:
                    pred_masks = pred_masks.squeeze(1)
                    iou_predictions = iou_predictions.squeeze(1)

                # Update the metrics
                mask_pred_binary = (pred_masks > 0).float()

                if cfg.metrics.iou.enabled:
                    batch_iou = compute_iou(y_pred=mask_pred_binary.unsqueeze(1), y=data["gt_masks"].unsqueeze(1), ignore_empty=False)
                    iter_metrics["iou"] += torch.mean(batch_iou)

                if cfg.metrics.dice.enabled:
                    batch_dsc = compute_dice(y_pred=mask_pred_binary.unsqueeze(1), y=data["gt_masks"].unsqueeze(1), ignore_empty=False)
                    iter_metrics["dsc"] += torch.mean(batch_dsc)

                iter_metrics["iou_pred"] += torch.mean(iou_predictions)

                # Calculate the losses
                gt_masks_unsqueezed = data["gt_masks"].float().unsqueeze(1)
                pred_masks_unsqueezed = pred_masks.unsqueeze(1)
                
                if cfg.losses.focal.enabled:
                    iter_metrics["loss_focal"] += focal_loss(pred_masks_unsqueezed, gt_masks_unsqueezed)
                
                if cfg.losses.dice.enabled:
                    iter_metrics["loss_dice"] += dice_loss(pred_masks_unsqueezed, gt_masks_unsqueezed)

                if cfg.losses.cross_entropy.enabled:
                    iter_metrics["loss_ce"] += ce_loss(pred_masks_unsqueezed, gt_masks_unsqueezed)
                
                if cfg.losses.iou.enabled:
                    if not cfg.metrics.iou.enabled:
                        batch_iou = compute_iou(y_pred=mask_pred_binary.unsqueeze(1), y=data["gt_masks"].unsqueeze(1), ignore_empty=False)
                    
                    iter_metrics["loss_iou"] += F.mse_loss(iou_predictions, batch_iou.flatten(), reduction='mean')

            loss_total = 0.
            if cfg.losses.focal.enabled:
                loss_total += cfg.losses.focal.weight * iter_metrics["loss_focal"]
            if cfg.losses.dice.enabled:
                loss_total += cfg.losses.dice.weight * iter_metrics["loss_dice"]
            if cfg.losses.cross_entropy.enabled:
                loss_total += cfg.losses.cross_entropy.weight * iter_metrics["loss_ce"]
            if cfg.losses.iou.enabled:
                loss_total += cfg.losses.iou.weight * iter_metrics["loss_iou"]

            # Backward pass
            optimizer.zero_grad()
            fabric.backward(loss_total)
            optimizer.step()

            # Step the scheduler if it's LambdaLR or WarmupReduceLROnPlateau
            if isinstance(scheduler, (torch.optim.lr_scheduler.LambdaLR, WarmupReduceLROnPlateau)):
                scheduler.step()
                if scheduler.get_last_lr() != last_lr:
                    last_lr = scheduler.get_last_lr()
                    fabric.print(f"learning rate changed to: {last_lr}")
                    log_event(cfg.out_dir, f"Epoch {epoch} | Iter {iter}: Learning rate changed to {last_lr}")

            epoch_metrics.batch_time.update(time.time() - end)
            end = time.time()

            # Update the meters
            epoch_metrics.total_losses.update(loss_total.item(), batch_size)
            if cfg.losses.focal.enabled:
                epoch_metrics.focal_losses.update(iter_metrics["loss_focal"].item(), batch_size)
            if cfg.losses.dice.enabled:
                epoch_metrics.dice_losses.update(iter_metrics["loss_dice"].item(), batch_size)
            if cfg.losses.cross_entropy.enabled:
                epoch_metrics.ce_losses.update(iter_metrics["loss_ce"].item(), batch_size)
            if cfg.losses.iou.enabled:
                epoch_metrics.space_iou_losses.update(iter_metrics["loss_iou"].item(), batch_size)
            
            epoch_metrics.ious_pred.update(iter_metrics["iou_pred"].item(), batch_size)
            if cfg.metrics.iou.enabled:
                epoch_metrics.ious.update(iter_metrics["iou"].item(), batch_size)
            if cfg.metrics.dice.enabled:
                epoch_metrics.dsc.update(iter_metrics["dsc"].item(), batch_size)

            print_and_log_metrics(fabric, cfg, epoch, iter, epoch_metrics, train_dataloader)

        # Step the scheduler if it is ReduceLROnPlateau or WarmupReduceLROnPlateau
        if isinstance(scheduler, (torch.optim.lr_scheduler.ReduceLROnPlateau, WarmupReduceLROnPlateau)):
            scheduler.step(epoch_metrics.total_losses.avg)
            if scheduler.get_last_lr() != last_lr:
                last_lr = scheduler.get_last_lr()
                fabric.print(f"learning rate changed to: {last_lr}")
                log_event(cfg.out_dir, f"Epoch {epoch}: Learning rate changed to {last_lr}")

        if (cfg.eval_interval > 0 and epoch % cfg.eval_interval == 0) or (epoch == cfg.num_epochs):
            
            val_results = validate(fabric, cfg, model, val_dataloader, epoch)

            if "iou" in val_results:
                val_iou = val_results["iou"]
                if val_iou > best_val_iou:
                    best_val_iou = val_iou
                    if os.path.exists(best_iou_ckpt_path):
                        try:
                            os.remove(best_iou_ckpt_path)
                        except OSError as e:
                            fabric.print(f"Error deleting old best_iou checkpoint: {e}")
                    
                    ckpt_name = f"best_iou_epoch_{epoch}_val_{val_iou:.4f}"
                    best_iou_ckpt_path = os.path.join(cfg.sav_dir, ckpt_name + ".pth")
                    model.save(fabric, cfg.sav_dir, ckpt_name)
                    fabric.print(f"New best IoU model saved: {ckpt_name}.pth")
                    log_event(cfg.out_dir, f"Epoch {epoch}: New best IoU model saved: {ckpt_name}.pth (IoU: {val_iou:.4f})")

            if "dsc" in val_results:
                val_dsc = val_results["dsc"]
                if val_dsc > best_val_dsc:
                    best_val_dsc = val_dsc
                    if os.path.exists(best_dsc_ckpt_path):
                        try:
                            os.remove(best_dsc_ckpt_path)
                        except OSError as e:
                            fabric.print(f"Error deleting old best_dsc checkpoint: {e}")
                    
                    ckpt_name = f"best_dsc_epoch_{epoch}_val_{val_dsc:.4f}"
                    best_dsc_ckpt_path = os.path.join(cfg.sav_dir, ckpt_name + ".pth")
                    model.save(fabric, cfg.sav_dir, ckpt_name)
                    fabric.print(f"New best DSC model saved: {ckpt_name}.pth")
                    log_event(cfg.out_dir, f"Epoch {epoch}: New best DSC model saved: {ckpt_name}.pth (DSC: {val_dsc:.4f})")
            
            if fabric.global_rank == 0:
                save_val_metrics(epoch, val_results, cfg.out_dir)

        metrics_history["epochs"].append(epoch)
        metrics_history["total_loss"].append(epoch_metrics.total_losses.avg)
        if cfg.losses.focal.enabled:
            metrics_history["focal_loss"].append(cfg.losses.focal.weight * epoch_metrics.focal_losses.avg)
        if cfg.losses.dice.enabled:
            metrics_history["dice_loss"].append(cfg.losses.dice.weight * epoch_metrics.dice_losses.avg)
        if cfg.losses.cross_entropy.enabled:
            metrics_history["ce_loss"].append(cfg.losses.cross_entropy.weight * epoch_metrics.ce_losses.avg)
        if cfg.losses.iou.enabled:
            metrics_history["iou_loss"].append(cfg.losses.iou.weight * epoch_metrics.space_iou_losses.avg)
        
        if cfg.metrics.iou.enabled:
            metrics_history["train_iou"].append(epoch_metrics.ious.avg)
        if cfg.metrics.dice.enabled:
            metrics_history["train_dsc"].append(epoch_metrics.dsc.avg)
        
        if "iou" in val_results:
            metrics_history["val_iou"].append(val_results["iou"])

        if "dsc" in val_results:
            metrics_history["val_dsc"].append(val_results["dsc"])
            
        if "total_loss" in val_results:
            if "val_total_loss" not in metrics_history:
                metrics_history["val_total_loss"] = []
            metrics_history["val_total_loss"].append(val_results["total_loss"])

        if "focal_loss" in val_results:
            if "val_focal_loss" not in metrics_history:
                metrics_history["val_focal_loss"] = []
            metrics_history["val_focal_loss"].append(val_results["focal_loss"])
            
        if "dice_loss" in val_results:
            if "val_dice_loss" not in metrics_history:
                metrics_history["val_dice_loss"] = []
            metrics_history["val_dice_loss"].append(val_results["dice_loss"])
            
        if "ce_loss" in val_results:
            if "val_ce_loss" not in metrics_history:
                metrics_history["val_ce_loss"] = []
            metrics_history["val_ce_loss"].append(val_results["ce_loss"])
            
        if "iou_loss" in val_results:
            if "val_iou_loss" not in metrics_history:
                metrics_history["val_iou_loss"] = []
            metrics_history["val_iou_loss"].append(val_results["iou_loss"])

        if fabric.global_rank == 0:
            plot_history(metrics_history, cfg.out_dir)
            save_train_metrics(metrics_history, cfg.out_dir)