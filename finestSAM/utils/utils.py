import os
import sys
import math
import time
import torch
import numpy as np
import lightning as L
import matplotlib.pyplot as plt
from monai.metrics import compute_iou, compute_dice
from box import Box
from typing import Tuple, Dict, Union, Optional, Any, List
from torch.utils.data import DataLoader
from lightning.fabric.fabric import _FabricOptimizer
from finestSAM.model.model import FinestSAM
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss, FocalLoss


class AverageMeter:
    """
    Computes and stores the average and current value.
    
    Attributes:
        - val: Current value.
        - avg: Average value.
        - sum: Sum of values.
        - count: Number of values.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val / n
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

class Metrics:
    """
    Metrics class for training and validation.
    
    Attributes:
        - batch_time: Average processing time per batch.
        - data_time: Average data loading time per batch.
        - focal_losses: Average focal loss.
        - dice_losses: Average dice loss.
        - ce_losses: Average Cross Entropy loss.
        - space_iou_losses: Average space IoU loss (distance between the predicted IoU and the real IoU).
        - total_losses: Average total loss.
        - ious: Average real IoU.
        - ious_pred: Average predicted IoU (from IoU Head).
        - dsc: Average Dice Score.
    """
        
    def __init__(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()

        self.focal_losses = AverageMeter()
        self.dice_losses = AverageMeter()
        self.ce_losses = AverageMeter()
        self.space_iou_losses = AverageMeter()
        self.total_losses = AverageMeter()

        self.ious = AverageMeter()
        self.ious_pred = AverageMeter()
        self.dsc = AverageMeter()


class WarmupReduceLROnPlateau:
    """
    Warmup and ReduceLROnPlateau scheduler.
    
    Attributes:
        - optimizer: Optimizer to be scheduled.
        - warmup_steps: Number of warmup steps.
        - patience: Number of epochs with no improvement after which learning rate will be reduced.
        - factor: Factor by which the learning rate will be reduced.
        - threshold: Threshold for measuring the new optimum, to only focus on significant changes.
        - cooldown: Number of epochs to wait before resuming normal operation after lr has been reduced.
        - min_lr: Minimum learning rate.
    """
    
    def __init__(self, optimizer, warmup_steps, patience, factor, threshold, cooldown, min_lr):
        self.warmup_steps = warmup_steps
        self.optimizer = optimizer
        self.step_num = 0
        
        self.warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, 
            lr_lambda=lambda step: float(step) / float(max(1, warmup_steps))
        )
        
        self.plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=factor, 
            patience=patience, 
            threshold=threshold, 
            cooldown=cooldown, 
            min_lr=min_lr
        )

    def step(self, metrics=None):
        if metrics is None:
            if self.step_num < self.warmup_steps:
                self.warmup_scheduler.step()
            self.step_num += 1
        else:
            if self.step_num >= self.warmup_steps:
                self.plateau_scheduler.step(metrics)

    def state_dict(self):
        return {
            'step_num': self.step_num,
            'warmup_scheduler': self.warmup_scheduler.state_dict(),
            'plateau_scheduler': self.plateau_scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.step_num = state_dict['step_num']
        self.warmup_scheduler.load_state_dict(state_dict['warmup_scheduler'])
        self.plateau_scheduler.load_state_dict(state_dict['plateau_scheduler'])
        
    def get_last_lr(self):
        if self.step_num < self.warmup_steps:
            return self.warmup_scheduler.get_last_lr()
        # ReduceLROnPlateau doesn't support get_last_lr in all versions, fallback
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


def configure_opt(cfg: Box, model: FinestSAM, fabric: L.Fabric) -> Tuple[_FabricOptimizer, _FabricOptimizer]:

    def lr_lambda(step):
        step_list = cfg.sched.LambdaLR.steps

        if step < cfg.sched.LambdaLR.warmup_steps:
            return float(step) / float(max(1, cfg.sched.LambdaLR.warmup_steps))
            
        decay_factor = 1.0
        if isinstance(step_list, list) and len(step_list) > 0:
            for cutoff_step in step_list:
                if step >= cutoff_step:
                     decay_factor *= (1.0 / cfg.sched.LambdaLR.decay_factor)
                
        return decay_factor
    
    # Configure optimizer (only trainable parameters)
    trainable = [p for p in model.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.opt.learning_rate, weight_decay=cfg.opt.weight_decay)

    # Print number of parameters
    all_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in trainable)
    fabric.print(f"Trainable: {trainable_params:,} ({100 * trainable_params / all_params:.4f}%) | Total: {all_params:,}")
    log_event(cfg.out_dir, f"Trainable: {trainable_params:,} ({100 * trainable_params / all_params:.4f}%) | Total: {all_params:,}")

    # Configure scheduler
    if cfg.sched.type == "ReduceLROnPlateau":
        if cfg.sched.ReduceLROnPlateau.get("warmup_steps", 0) > 0:
            scheduler = WarmupReduceLROnPlateau(
                optimizer=optimizer,
                warmup_steps=cfg.sched.ReduceLROnPlateau.warmup_steps,
                patience=cfg.sched.ReduceLROnPlateau.epoch_patience,
                factor=cfg.sched.ReduceLROnPlateau.decay_factor,
                threshold=cfg.sched.ReduceLROnPlateau.threshold,
                cooldown=cfg.sched.ReduceLROnPlateau.cooldown,
                min_lr=cfg.sched.ReduceLROnPlateau.min_lr
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer, 
                factor=cfg.sched.ReduceLROnPlateau.decay_factor, 
                patience=cfg.sched.ReduceLROnPlateau.epoch_patience, 
                threshold=cfg.sched.ReduceLROnPlateau.threshold, 
                cooldown=cfg.sched.ReduceLROnPlateau.cooldown,
                min_lr=cfg.sched.ReduceLROnPlateau.min_lr
            )
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    return optimizer, scheduler


def validate(
        fabric: L.Fabric, 
        cfg: Box,
        model: FinestSAM, 
        val_dataloader: DataLoader, 
        epoch: int
    ) -> Dict[str, float]: 
    """
    Validation function
    Computes IoU, Dice Score (F1 Score), and losses for the validation dataset.

    Args:
        fabric (L.Fabric): The lightning fabric.
        cfg (Box): The configuration file.
        model (FinestSAM): The model.
        val_dataloader (DataLoader): The validation dataloader.
        epoch (int): The current epoch.
        
    Returns:
        Dict[str, float]: Dictionary containing computed metrics.
    """
    model.eval()
    
    # Initialize losses
    focal_loss = FocalLoss(gamma=cfg.losses.focal.gamma, reduction="mean")
    dice_loss = DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    metrics_to_compute = {}
    if cfg.metrics.iou.enabled:
        metrics_to_compute['iou'] = AverageMeter()
    if cfg.metrics.dice.enabled:
        metrics_to_compute['dsc'] = AverageMeter()
    
    if cfg.losses.focal.enabled:
        metrics_to_compute['loss_focal'] = AverageMeter()
    if cfg.losses.dice.enabled:
        metrics_to_compute['loss_dice'] = AverageMeter()
    if cfg.losses.cross_entropy.enabled:
        metrics_to_compute['loss_ce'] = AverageMeter()
    if cfg.losses.iou.enabled:
        metrics_to_compute['loss_iou'] = AverageMeter()
        
    metrics_to_compute['total_loss'] = AverageMeter()
    
    with torch.no_grad():
        for iter, batched_data in enumerate(val_dataloader):
            batch_size = len(batched_data)
            
            # Forward pass
            outputs = model(batched_input=batched_data, multimask_output=cfg.multimask_output)

            batched_pred_masks = []
            batched_iou_predictions = []
            for item in outputs:
                batched_pred_masks.append(item["masks"])
                batched_iou_predictions.append(item["iou_predictions"])

            iter_metrics = {
                "loss_focal": torch.tensor(0., device=fabric.device),
                "loss_dice": torch.tensor(0., device=fabric.device),
                "loss_iou": torch.tensor(0., device=fabric.device),
                "loss_ce": torch.tensor(0., device=fabric.device),
                "iou": torch.tensor(0., device=fabric.device),
                "dsc": torch.tensor(0., device=fabric.device),
            }

            # Compute the losses and metrics
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

                # Metrics
                mask_pred_binary = (pred_masks > 0).float()

                if cfg.metrics.iou.enabled:
                    batch_iou = compute_iou(y_pred=mask_pred_binary.unsqueeze(1), y=data["gt_masks"].unsqueeze(1), ignore_empty=False)
                    iter_metrics["iou"] += torch.mean(batch_iou)

                if cfg.metrics.dice.enabled:
                    batch_dsc = compute_dice(y_pred=mask_pred_binary.unsqueeze(1), y=data["gt_masks"].unsqueeze(1), ignore_empty=False)
                    iter_metrics["dsc"] += torch.mean(batch_dsc)

                # Losses
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

            # Calculate total loss
            loss_total = 0.
            if cfg.losses.focal.enabled:
                loss_total += cfg.losses.focal.weight * iter_metrics["loss_focal"]
            if cfg.losses.dice.enabled:
                loss_total += cfg.losses.dice.weight * iter_metrics["loss_dice"]
            if cfg.losses.cross_entropy.enabled:
                loss_total += cfg.losses.cross_entropy.weight * iter_metrics["loss_ce"]
            if cfg.losses.iou.enabled:
                loss_total += cfg.losses.iou.weight * iter_metrics["loss_iou"]

            metrics_to_compute['total_loss'].update(loss_total.item(), batch_size)
            
            if cfg.losses.focal.enabled:
                metrics_to_compute['loss_focal'].update(iter_metrics['loss_focal'].item(), batch_size)
            if cfg.losses.dice.enabled:
                metrics_to_compute['loss_dice'].update(iter_metrics['loss_dice'].item(), batch_size)
            if cfg.losses.cross_entropy.enabled:
                metrics_to_compute['loss_ce'].update(iter_metrics['loss_ce'].item(), batch_size)
            if cfg.losses.iou.enabled:
                metrics_to_compute['loss_iou'].update(iter_metrics['loss_iou'].item(), batch_size)

            if cfg.metrics.iou.enabled:
                metrics_to_compute['iou'].update(iter_metrics['iou'].item(), batch_size)
            if cfg.metrics.dice.enabled:
                metrics_to_compute['dsc'].update(iter_metrics['dsc'].item(), batch_size)
            
            # Simple logging for progress
            display_str = f'Val: [{epoch}] - [{iter+1}/{len(val_dataloader)}]:'
            display_str += f" Loss: [{metrics_to_compute['total_loss'].avg:.4f}] |"
            if 'iou' in metrics_to_compute:
                display_str += f" IoU: [{metrics_to_compute['iou'].avg:.4f}] |"
            if 'dsc' in metrics_to_compute:
                display_str += f" DSC: [{metrics_to_compute['dsc'].avg:.4f}]"
            
            fabric.print(display_str)

        display_str = f'Validation [{epoch}]:'
        results = {}
        
        display_str += f" Loss: [{metrics_to_compute['total_loss'].avg:.4f}] |"
        results['total_loss'] = metrics_to_compute['total_loss'].avg
        
        if 'iou' in metrics_to_compute:
            display_str += f" Mean IoU: [{metrics_to_compute['iou'].avg:.4f}] |"
            results['iou'] = metrics_to_compute['iou'].avg
        if 'dsc' in metrics_to_compute:
            display_str += f" Mean DSC: [{metrics_to_compute['dsc'].avg:.4f}]"
            results['dsc'] = metrics_to_compute['dsc'].avg
            
        if cfg.losses.focal.enabled:
            results['focal_loss'] = cfg.losses.focal.weight * metrics_to_compute['loss_focal'].avg
        if cfg.losses.dice.enabled:
            results['dice_loss'] = cfg.losses.dice.weight * metrics_to_compute['loss_dice'].avg
        if cfg.losses.cross_entropy.enabled:
            results['ce_loss'] = cfg.losses.cross_entropy.weight * metrics_to_compute['loss_ce'].avg
        if cfg.losses.iou.enabled:
            results['iou_loss'] = cfg.losses.iou.weight * metrics_to_compute['loss_iou'].avg
            
        fabric.print(display_str)
        
        model.train()
        return results


def compute_dataset_stats(dataloader: DataLoader, fabric: L.Fabric = None) -> Tuple[List[float], List[float]]:
    """
    Computes the mean and standard deviation of the dataset for normalization.
    
    Args:
        dataloader (DataLoader): The dataloader containing the images.
        fabric (L.Fabric): Optional fabric instance for logging.
        
    Returns:
        Tuple[List[float], List[float]]: The mean and standard deviation of the dataset.
    """
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images = 0
    
    if fabric:
        fabric.print("Computiong dataset stats...")
    else:
        print("Computiong dataset stats...")
        
    for batch in dataloader:
        for item in batch:
            img = item["image"] # [C, H, W] tensor, uint8
            img = img.float()
            
            # Ensure accumulators are on the correct device
            if mean.device != img.device:
                mean = mean.to(img.device)
                std = std.to(img.device)

            # Mean over H, W
            mean += img.mean(dim=(1, 2))
            std += img.std(dim=(1, 2))
            total_images += 1
            
    mean /= total_images
    std /= total_images
    
    # Return as lists
    return mean.tolist(), std.tolist()


def print_and_log_metrics(
    fabric: L.Fabric,
    cfg: Box,
    epoch: int,
    iter: int,
    metrics: Metrics,
    train_dataloader: DataLoader,
):
    """
    Print and log the metrics for the training.
    """
    display_str = f'Epoch: [{epoch}][{iter+1}/{len(train_dataloader)}]' \
                  f' | Time [{metrics.batch_time.val:.3f}s ({metrics.batch_time.avg:.3f}s)]' \
                  f' | Data [{metrics.data_time.val:.3f}s ({metrics.data_time.avg:.3f}s)]'

    if cfg.losses.focal.enabled:
        display_str += f' | Focal [{cfg.losses.focal.weight * metrics.focal_losses.val:.4f} ({cfg.losses.focal.weight * metrics.focal_losses.avg:.4f})]'
    if cfg.losses.dice.enabled:
        display_str += f' | Dice [{cfg.losses.dice.weight * metrics.dice_losses.val:.4f} ({cfg.losses.dice.weight * metrics.dice_losses.avg:.4f})]'
    if cfg.losses.iou.enabled:
        display_str += f' | IoU L. [{cfg.losses.iou.weight * metrics.space_iou_losses.val:.4f} ({cfg.losses.iou.weight * metrics.space_iou_losses.avg:.4f})]'
    if cfg.losses.cross_entropy.enabled:
        display_str += f' | CE [{cfg.losses.cross_entropy.weight * metrics.ce_losses.val:.4f} ({cfg.losses.cross_entropy.weight * metrics.ce_losses.avg:.4f})]'
    
    display_str += f' | Total [{metrics.total_losses.val:.4f} ({metrics.total_losses.avg:.4f})]'

    if cfg.metrics.iou.enabled:
        display_str += f' | IoU [{metrics.ious.val:.4f} ({metrics.ious.avg:.4f})]'
    
    display_str += f' | Pred IoU [{metrics.ious_pred.val:.4f} ({metrics.ious_pred.avg:.4f})]'

    if cfg.metrics.dice.enabled:
        display_str += f' | DSC [{metrics.dsc.val:.4f} ({metrics.dsc.avg:.4f})]'

    fabric.print(display_str)
    
    steps = epoch * len(train_dataloader) + iter    
    log_info = {
        'total loss': metrics.total_losses.val,
    }
    if cfg.losses.focal.enabled:
        log_info['focal loss'] = cfg.losses.focal.weight * metrics.focal_losses.val
    if cfg.losses.dice.enabled:
        log_info['dice loss'] = cfg.losses.dice.weight * metrics.dice_losses.val
    if cfg.losses.iou.enabled:
        log_info['iou loss'] = cfg.losses.iou.weight * metrics.space_iou_losses.val
    if cfg.losses.cross_entropy.enabled:
        log_info['ce loss'] = cfg.losses.cross_entropy.weight * metrics.ce_losses.val
    
    if cfg.metrics.iou.enabled:
        log_info['train_iou'] = metrics.ious.val
    if cfg.metrics.dice.enabled:
        log_info['train_dsc'] = metrics.dsc.val
    fabric.log_dict(log_info, step=steps)


def plot_history(
        metrics_history: Dict[str, list],
        out_plots: str,
        name: str = "log"
    ):
    """Plots and saves training history graphs for losses and metrics.

    Generates a figure with three side-by-side subplots:
    1. Losses (total, focal, dice, IoU) with an automatic Y-axis.
    2. IoU (Train/Validation) with a shared Y-axis.
    3. Dice Score (Train/Validation) with a shared Y-axis.

    The IoU and Dice plots share the same Y-axis limits (from a calculated
    minimum across both metrics up to 1.0) for direct comparison.

    Args:
        metrics_history (Dict[str, list]): A dictionary containing the history 
            of metrics. Expected keys include 'epochs', 'total_loss', 
            'focal_loss', 'dice_loss', 'iou_loss', 'ce_loss', 'train_iou', 'val_iou', 
            'train_dsc', 'val_dsc'.
        out_plots (str): The path to the output directory where the
            '{name}.png' file will be saved.
        name (str): The base name for the saved plot file (without extension).

    Side Effects:
        - Saves a PNG image ('{name}.png') to the `out_plots`
          directory.
        - Prints error messages to stderr if 'epochs' data is missing or empty.
        - Prints a warning to stderr if the 'serif' font cannot be set.
    """
    
    # --- Data Validation ---
    # Ensure at least one data point exists
    if not metrics_history.get("epochs"):
        print("Error: No 'epochs' data found in metrics_history.", file=sys.stderr)
        return
    
    epochs = metrics_history["epochs"]
    if not epochs:
        print("Error: The 'epochs' list is empty.", file=sys.stderr)
        return
    
    max_epoch = epochs[-1]

    # --- Global Style Settings ---
    try:
        plt.rc('font', family='serif')
    except Exception as e:
        print(f"Warning: Could not set 'serif' font. Using default. Details: {e}", file=sys.stderr)
        
    plt.rcParams.update({
        'font.size': 20,
        'axes.titlesize': 24,
        'axes.labelsize': 22,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 18, 
        'axes.titleweight': 'bold'
    })

    # --- Create the Figure with 4 side-by-side Subplots ---
    fig, (ax_train_loss, ax_val_loss, ax_iou, ax_dsc) = plt.subplots(1, 4, figsize=(40, 9)) 
    
    colors = {
        'total_loss': '#d62728',  # Red
        'dice_loss': '#17becf',   # Cyan
        'focal_loss': '#ff7f0e',  # Orange
        'iou_loss': '#2ca02c',    # Green
        'ce_loss': '#9467bd',     # Purple
        'train_set': '#ff7f0e',   # Orange (for Train)
        'val_set': '#1f77b4',     # Blue (for Val)
    }
    
    line_width = 2.5
    # Common X-axis ticks
    if max_epoch <= 25:
        ticks = list(range(1, max_epoch + 1))
    else:
        ticks = [1] + list(range(25, max_epoch + 1, 25)) 

    # --- Plot 1: Training Losses (Left) ---

    ax_train_loss.plot(epochs, metrics_history["total_loss"], label="Train Total Loss", 
                 color=colors['total_loss'], linestyle='-', linewidth=line_width)
    
    if "focal_loss" in metrics_history and any(metrics_history["focal_loss"]):
        ax_train_loss.plot(epochs, metrics_history["focal_loss"], label="Focal Loss", 
                    color=colors['focal_loss'], linestyle='-', linewidth=line_width)
    
    if "dice_loss" in metrics_history and any(metrics_history["dice_loss"]):
        ax_train_loss.plot(epochs, metrics_history["dice_loss"], label="Dice Loss", 
                    color=colors['dice_loss'], linestyle='-', linewidth=line_width)
    
    if "iou_loss" in metrics_history and any(metrics_history["iou_loss"]):
        ax_train_loss.plot(epochs, metrics_history["iou_loss"], label="IoU Loss", 
                    color=colors['iou_loss'], linestyle='-', linewidth=line_width)
    if "ce_loss" in metrics_history and any(metrics_history["ce_loss"]):
        ax_train_loss.plot(epochs, metrics_history["ce_loss"], label="CE Loss", 
                     color=colors['ce_loss'], linestyle='-', linewidth=line_width)
    
    ax_train_loss.set_title("Train Loss", loc='left')
    ax_train_loss.legend(loc='upper right', frameon=True, fancybox=True)
    ax_train_loss.set_xlabel("Epoch")
    ax_train_loss.set_ylabel("Value")
    ax_train_loss.grid(False)
    ax_train_loss.set_xticks(ticks)
    if max_epoch > 1:
        ax_train_loss.set_xlim(left=1, right=max_epoch)


    # --- Plot 2: Validation Losses (Center Left) ---

    if "val_total_loss" in metrics_history:
        ax_val_loss.plot(epochs, metrics_history["val_total_loss"], label="Val Total Loss", 
                     color=colors['total_loss'], linestyle='-', linewidth=line_width)

    if "val_focal_loss" in metrics_history and any(metrics_history["val_focal_loss"]):
        ax_val_loss.plot(epochs, metrics_history["val_focal_loss"], label="Focal Loss", 
                    color=colors['focal_loss'], linestyle='-', linewidth=line_width)
    
    if "val_dice_loss" in metrics_history and any(metrics_history["val_dice_loss"]):
        ax_val_loss.plot(epochs, metrics_history["val_dice_loss"], label="Dice Loss", 
                    color=colors['dice_loss'], linestyle='-', linewidth=line_width)
    
    if "val_iou_loss" in metrics_history and any(metrics_history["val_iou_loss"]):
        ax_val_loss.plot(epochs, metrics_history["val_iou_loss"], label="IoU Loss", 
                    color=colors['iou_loss'], linestyle='-', linewidth=line_width)
    if "val_ce_loss" in metrics_history and any(metrics_history["val_ce_loss"]):
        ax_val_loss.plot(epochs, metrics_history["val_ce_loss"], label="CE Loss", 
                     color=colors['ce_loss'], linestyle='-', linewidth=line_width)
    
    ax_val_loss.set_title("Val Loss", loc='left')
    ax_val_loss.legend(loc='upper right', frameon=True, fancybox=True)
    ax_val_loss.set_xlabel("Epoch")
    ax_val_loss.set_ylabel("Value")
    ax_val_loss.grid(False)
    ax_val_loss.set_xticks(ticks)
    if max_epoch > 1:
        ax_val_loss.set_xlim(left=1, right=max_epoch)
    
    # Automatic Y-scale is fine for losses, or we could share it with train loss if desired.
    # For now, separate Y-scale is often better if ranges differ significantly.


    # --- Metrics Data ---
    train_iou_data = metrics_history["train_iou"]
    val_iou_data = metrics_history["val_iou"]
    train_dsc_data = metrics_history["train_dsc"]
    val_dsc_data = metrics_history["val_dsc"]
    
    # --- Calculate shared Y-axis limits for metrics ---
    # Calculate the absolute minimum across ALL 4 metric lists
    min_metric_val = min(
        min(train_iou_data), 
        min(val_iou_data), 
        min(train_dsc_data), 
        min(val_dsc_data)
    )
    # Round down to the nearest 0.1 and add 0.05 padding
    shared_lower_lim = max(0, math.floor(min_metric_val * 10) / 10 - 0.05)
    shared_upper_lim = 1.0 # Fixed upper limit at 1.0


    # --- Plot 3: IoU (Center Right) ---
    
    ax_iou.plot(epochs, train_iou_data, label="Train IoU", 
                color=colors['train_set'], linestyle='-', linewidth=line_width)
    ax_iou.plot(epochs, val_iou_data, label="Val IoU", 
                color=colors['val_set'], linestyle='-', linewidth=line_width)

    ax_iou.set_title("IoU", loc='left')
    ax_iou.legend(loc='lower right', frameon=True, fancybox=True)
    ax_iou.set_xlabel("Epoch")
    ax_iou.set_ylabel("Value")
    ax_iou.grid(False)
    ax_iou.set_xticks(ticks)
    if max_epoch > 1:
        ax_iou.set_xlim(left=1, right=max_epoch)
    
    # Apply shared Y-scale
    ax_iou.set_ylim(shared_lower_lim, shared_upper_lim) 


    # --- Plot 4: Dice Score (Right) ---
    
    ax_dsc.plot(epochs, train_dsc_data, label="Train DSC", 
                color=colors['train_set'], linestyle='-', linewidth=line_width)
    ax_dsc.plot(epochs, val_dsc_data, label="Val DSC", 
                color=colors['val_set'], linestyle='-', linewidth=line_width)

    ax_dsc.set_title("DSC", loc='left')
    ax_dsc.legend(loc='lower right', frameon=True, fancybox=True)
    ax_dsc.set_xlabel("Epoch")
    ax_dsc.set_ylabel("Value")
    ax_dsc.grid(False)
    ax_dsc.set_xticks(ticks)
    if max_epoch > 1:
        ax_dsc.set_xlim(left=1, right=max_epoch)
    
    # Apply shared Y-scale
    ax_dsc.set_ylim(shared_lower_lim, shared_upper_lim)

    # --- Save Figure ---
    
    fig.tight_layout() 
    
    if not os.path.exists(out_plots):
        os.makedirs(out_plots, exist_ok=True)

    output_filename = os.path.join(out_plots, f"{name}.png")
    
    try:
        fig.savefig(output_filename, bbox_inches='tight', dpi=300)
        print(f"Combined plots saved to: {output_filename}")
    except Exception as e:
        print(f"Error saving plot file: {e}", file=sys.stderr)
    
    plt.close(fig)
    plt.rcdefaults()


def save_train_metrics(
    metrics_history: Dict[str, list],
    out_dir: str,
    name: str = "metrics"
):
    """
    Saves the training metrics to a text file.
    
    Args:
        metrics_history (Dict[str, list]): Dictionary containing the metrics history.
        out_dir (str): Directory where the file will be saved.
        name (str): Base name of the output file (default: "metrics").
    """
    if not metrics_history.get("epochs"):
        return

    latest_idx = len(metrics_history["epochs"]) - 1
    epoch = metrics_history["epochs"][latest_idx]

    train_headers = ["Epoch", "Total Loss"]
    train_values = [epoch, metrics_history["total_loss"][latest_idx]]

    if "focal_loss" in metrics_history and any(metrics_history["focal_loss"]):
        train_headers.append("Focal Loss")
        train_values.append(metrics_history["focal_loss"][latest_idx])

    if "dice_loss" in metrics_history and any(metrics_history["dice_loss"]):
        train_headers.append("Dice Loss")
        train_values.append(metrics_history["dice_loss"][latest_idx])

    if "iou_loss" in metrics_history and any(metrics_history["iou_loss"]):
        train_headers.append("IoU Loss")
        train_values.append(metrics_history["iou_loss"][latest_idx])

    if "ce_loss" in metrics_history and any(metrics_history["ce_loss"]):
        train_headers.append("CE Loss")
        train_values.append(metrics_history["ce_loss"][latest_idx])

    if "train_iou" in metrics_history and any(metrics_history["train_iou"]):
        train_headers.append("Train IoU")
        train_values.append(metrics_history["train_iou"][latest_idx])

    if "train_dsc" in metrics_history and any(metrics_history["train_dsc"]):
        train_headers.append("Train DSC")
        train_values.append(metrics_history["train_dsc"][latest_idx])
    
    train_filename = f"train_{name}.txt"
    _write_metrics_file(out_dir, train_filename, train_headers, train_values)
    print(f"Metrix train saved to: {os.path.join(out_dir, train_filename)}")


def save_val_metrics(
    epoch: int,
    results: Dict[str, float],
    out_dir: str,
    name: str = "metrics"
):
    """
    Saves the validation metrics to a text file.
    
    Args:
        epoch (int): Current epoch.
        results (Dict[str, float]): Dictionary of validation results.
        out_dir (str): Directory where the file will be saved.
        name (str): Base name of the output file (default: "metrics").
    """
    val_headers = ["Epoch"]
    val_values = [epoch]

    if "total_loss" in results:
        val_headers.append("Val Loss")
        val_values.append(results["total_loss"])

    if "iou" in results:
        val_headers.append("Val IoU")
        val_values.append(results["iou"])
    
    if "dsc" in results:
        val_headers.append("Val DSC")
        val_values.append(results["dsc"])
    
    val_filename = f"val_{name}.txt"
    _write_metrics_file(out_dir, val_filename, val_headers, val_values)
    print(f"Metrix val saved to: {os.path.join(out_dir, val_filename)}")


def _write_metrics_file(out_dir, filename, headers, values):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    output_path = os.path.join(out_dir, filename)
    
    # Format values (4 decimal places for floats)
    formatted_values = []
    for v in values:
        if isinstance(v, float):
            formatted_values.append(f"{v:.4f}")
        else:
            formatted_values.append(str(v))
            
    # Write to file
    mode = 'a' if os.path.exists(output_path) else 'w'
    with open(output_path, mode) as f:
        if mode == 'w':
            f.write("\t".join(headers) + "\n")
        f.write("\t".join(formatted_values) + "\n")


def log_event(out_dir: str, message: str, filename: str = "training_events.txt"):
    """
    Logs an event message with a timestamp to a text file.

    Args:
        out_dir (str): Directory where the log file is saved.
        message (str): The message to log.
        filename (str): The name of the log file (default: "training_events.txt").
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_path = os.path.join(out_dir, filename)
    
    with open(log_path, "a") as f:
        f.write(f"[{timestamp}] {message}\n")   


def show_anns(
        anns: list, 
        opacity: float = 0.35
    ):
    '''
    Show annotations on the image.

    Args:
        anns (list): The list of annotations, which is the output list of the automatic predictor.
        opacity (float): The opacity of the masks.
    '''

    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [opacity]])
        img[m] = color_mask
    ax.imshow(img)


def show_mask(mask, ax, random_color=True, seed=None):
    '''
    Show a single mask on the image.
    
    Args:
        mask (torch.Tensor): The mask to be shown.
        ax (matplotlib.axes.Axes): The axes to show the mask on.
        random_color (bool): Whether to use a random color for the mask.
        seed (int): The seed for the random color.
    '''
    np.random.seed(seed)
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.cpu().numpy().reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1].cpu().numpy()
    neg_points = coords[labels==0].cpu().numpy()
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))