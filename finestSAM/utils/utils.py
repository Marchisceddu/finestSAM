import os
import sys
import math
import time
import random
import torch
import torch.distributed as dist
import numpy as np
import lightning as L
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm.auto import tqdm
from monai.metrics import compute_iou, compute_dice, compute_hausdorff_distance
from box import Box
from typing import Tuple, Dict, Union, Optional, Any, List, Set
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

    def sync(self, fabric: Optional[L.Fabric]):
        """Synchronize `sum` and `count` across ranks (DDP-safe)."""
        if fabric is None:
            return
        if not _is_distributed(fabric):
            return
        device = fabric.device
        t = torch.tensor([float(self.sum), float(self.count)], device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        self.sum = float(t[0].item())
        self.count = float(t[1].item())
        self.avg = self.sum / self.count if self.count else 0.0

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
        - hd95: Average 95th Percentile Hausdorff Distance.
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
        self.hd95 = AverageMeter()

    def sync(self, fabric: Optional[L.Fabric]):
        """Synchronize all meters across ranks (DDP-safe)."""
        for meter in (
            self.batch_time,
            self.data_time,
            self.focal_losses,
            self.dice_losses,
            self.ce_losses,
            self.space_iou_losses,
            self.total_losses,
            self.ious,
            self.ious_pred,
            self.dsc,
            self.hd95,
        ):
            meter.sync(fabric)


def _is_distributed(fabric: L.Fabric) -> bool:
    """True if we're in a multi-process distributed context."""
    try:
        world_size = int(getattr(fabric, "world_size", 1) or 1)
    except Exception:
        world_size = 1
    return world_size > 1 and dist.is_available() and dist.is_initialized()


def seed_everything(fabric: Optional[L.Fabric], seed: int, deterministic: bool = True):
    """
    Sets seeds for Python, NumPy, and PyTorch to ensure reproducibility.
    
    Args:
        fabric: Optional Lightning Fabric instance.
        seed: Base seed integer.
        deterministic: If True, enforces deterministic algorithms (slower).
    """
    # Resolve Rank and Calculate Effective Seed
    rank = fabric.global_rank if fabric is not None else 0
    seed = int(seed) + rank

    # Set Environment Variables (Must be done before CUDA initialization)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    if deterministic:
        # Required for deterministic algorithms in cuBLAS >= 10.2
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Apply Seeds
    # If Fabric is available, let it handle the heavy lifting first
    if fabric is not None:
        # workers=True attempts to seed dataloaders if possible
        fabric.seed_everything(seed, workers=True)
    else:
        # Manual fallback
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Configure PyTorch Backend for Determinism
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        
        # Disable cuDNN benchmarking (which selects fastest algo dynamically)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        # Disable TensorFloat-32 (TF32) on Ampere+ GPUs for bit-wise precision
        # Note: This significantly impacts performance on A100/H100/RTX30xx+
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        # Restore defaults for performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


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
    if getattr(fabric, "global_rank", 0) == 0:
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

    # Initialize totals
    totals: Dict[str, torch.Tensor] = {}
    if cfg.metrics.iou.enabled:
        totals["iou"] = torch.tensor(0.0, device=fabric.device)
    if cfg.metrics.dice.enabled:
        totals["dsc"] = torch.tensor(0.0, device=fabric.device)
    if cfg.metrics.hd95.enabled:
        totals["hd95"] = torch.tensor(0.0, device=fabric.device)
    totals["iou_pred"] = torch.tensor(0.0, device=fabric.device)

    if cfg.losses.focal.enabled:
        totals["loss_focal"] = torch.tensor(0.0, device=fabric.device)
    if cfg.losses.dice.enabled:
        totals["loss_dice"] = torch.tensor(0.0, device=fabric.device)
    if cfg.losses.cross_entropy.enabled:
        totals["loss_ce"] = torch.tensor(0.0, device=fabric.device)
    if cfg.losses.iou.enabled:
        totals["loss_iou"] = torch.tensor(0.0, device=fabric.device)

    totals["total_loss"] = torch.tensor(0.0, device=fabric.device)
    total_count = torch.tensor(0.0, device=fabric.device)

    output_images = cfg.get("print_images", 0)
    saved_images = 0
    images_out_dir = os.path.join(cfg.out_dir, "images")
    
    with torch.no_grad():
        is_rank0 = getattr(fabric, "global_rank", 0) == 0
        total_batches = len(val_dataloader)
        
        if str(output_images).lower() == "all":
            max_images = total_batches
        else:
            max_images = int(output_images)

        save_batch_indices: Set[int] = set()
        if is_rank0 and total_batches > 0 and max_images > 0:
            if max_images >= total_batches:
                save_batch_indices = set(range(total_batches))
            else:
                lin_positions = np.linspace(0, total_batches - 1, num=max_images, dtype=int)
                save_batch_indices = set(int(pos) for pos in lin_positions.tolist())

        val_iter = enumerate(val_dataloader)
        pbar = None
        if is_rank0:
            pbar = tqdm(
                val_iter,
                total=total_batches,
                desc=f"Val {epoch}",
                leave=False,
                dynamic_ncols=True,
            )
            val_iter = pbar

        for iter, batched_data in val_iter:
            batch_size = len(batched_data)
            total_count += float(batch_size)
            
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
                "iou_pred": torch.tensor(0., device=fabric.device),
                "dsc": torch.tensor(0., device=fabric.device),
                "hd95": torch.tensor(0., device=fabric.device),
            }

            samples_for_selection: List[Tuple[torch.Tensor, Dict[str, Any], torch.Tensor]] = []

            # Compute the losses and metrics
            for data, pred_masks, iou_predictions in zip(batched_data, batched_pred_masks, batched_iou_predictions):

                # Default DSC placeholder in case the metric is disabled
                sample_dsc = torch.tensor(0.0, device=fabric.device)

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
                    sample_dsc = torch.mean(batch_dsc).detach()
                """
                else:
                    # Still compute DSC for qualitative selection
                    sample_dsc = torch.mean(
                        compute_dice(
                            y_pred=mask_pred_binary.unsqueeze(1),
                            y=data["gt_masks"].unsqueeze(1),
                            ignore_empty=False,
                        )
                    ).detach()
                """
                if cfg.metrics.hd95.enabled:
                    batch_hd95 = compute_hausdorff_distance(y_pred=mask_pred_binary.unsqueeze(1).cpu(), y=data["gt_masks"].unsqueeze(1).cpu(), include_background=False, percentile=95).to(fabric.device)

                    # If the prediction is empty, the Hausdorff distance is set to the maximum possible distance
                    img_size = cfg.model.get("img_size", 1024)
                    max_dist = math.sqrt(img_size**2 + img_size**2)
                    batch_hd95 = torch.where(torch.isnan(batch_hd95), torch.full_like(batch_hd95, max_dist), batch_hd95)
                    
                    iter_metrics["hd95"] += torch.mean(batch_hd95)

                iter_metrics["iou_pred"] += torch.mean(iou_predictions)
                
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

                samples_for_selection.append((sample_dsc, data, mask_pred_binary))

            
            # Save qualitative results on the selected batches, exporting every sample in the batch
            if (is_rank0 and iter in save_batch_indices and samples_for_selection):
                for sample_idx, (sample_dsc, sel_data, sel_pred_mask) in enumerate(samples_for_selection):
                    if sel_data.get("original_image") is None or sel_data.get("original_size") is None:
                        continue

                    gt_mask = sel_data.get("gt_masks")
                    if gt_mask is not None:
                        gt_mask = gt_mask.float()
                        if gt_mask.ndim == 3:
                            gt_mask = gt_mask.any(dim=0).float()
                        gt_mask = gt_mask.squeeze()
                        gt_mask = gt_mask.unsqueeze(0).unsqueeze(0)
                        gt_mask = F.interpolate(gt_mask, size=sel_data["original_size"], mode="nearest").squeeze()

                    pred_to_save = sel_pred_mask
                    if pred_to_save.ndim == 3:
                        pred_to_save = pred_to_save.any(dim=0).float()
                    pred_to_save = pred_to_save.squeeze()
                    pred_to_save = F.interpolate(pred_to_save.unsqueeze(0).unsqueeze(0), size=sel_data["original_size"], mode="nearest").squeeze()

                    save_prediction_visual(
                        out_dir=images_out_dir,
                        base_name=f"val_b{iter:04d}_s{sample_idx:03d}",
                        image=np.array(sel_data["original_image"]),
                        gt_mask=gt_mask,
                        pred_mask=pred_to_save,
                    )
                    saved_images += 1

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

            # Accumulate sums (each iter_metrics entry is already a sum across samples in the batch)
            totals["total_loss"] += loss_total.detach()
            if cfg.losses.focal.enabled:
                totals["loss_focal"] += iter_metrics["loss_focal"].detach()
            if cfg.losses.dice.enabled:
                totals["loss_dice"] += iter_metrics["loss_dice"].detach()
            if cfg.losses.cross_entropy.enabled:
                totals["loss_ce"] += iter_metrics["loss_ce"].detach()
            if cfg.losses.iou.enabled:
                totals["loss_iou"] += iter_metrics["loss_iou"].detach()

            if cfg.metrics.iou.enabled:
                totals["iou"] += iter_metrics["iou"].detach()
            if cfg.metrics.dice.enabled:
                totals["dsc"] += iter_metrics["dsc"].detach()
            if cfg.metrics.hd95.enabled:
                totals["hd95"] += iter_metrics["hd95"].detach()

            totals["iou_pred"] += iter_metrics["iou_pred"].detach()
            
            # Update progress bar
            if pbar is not None and total_count.item() > 0:
                denom_local = float(total_count.item())
                postfix = {
                    "loss": f"{(totals['total_loss'] / denom_local).item():.4f}",
                }
                postfix["p_iou"] = f"{(totals['iou_pred'] / denom_local).item():.4f}"
                if "iou" in totals:
                    postfix["iou"] = f"{(totals['iou'] / denom_local).item():.4f}"
                if "dsc" in totals:
                    postfix["dsc"] = f"{(totals['dsc'] / denom_local).item():.4f}"
                if "hd95" in totals:
                    postfix["hd95"] = f"{(totals['hd95'] / denom_local).item():.2f}"
                pbar.set_postfix(postfix)

        if pbar is not None:
            pbar.close()

        # All-reduce totals and count to get global averages
        if _is_distributed(fabric):
            for t in totals.values():
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_count, op=dist.ReduceOp.SUM)

        denom = float(total_count.item()) if float(total_count.item()) > 0 else 1.0
        results: Dict[str, float] = {}
        results["total_loss"] = float((totals["total_loss"] / denom).item())
        results["iou_pred"] = float((totals["iou_pred"] / denom).item())
        if "iou" in totals:
            results["iou"] = float((totals["iou"] / denom).item())
        if "dsc" in totals:
            results["dsc"] = float((totals["dsc"] / denom).item())
        if "hd95" in totals:
            results["hd95"] = float((totals["hd95"] / denom).item())

        if cfg.losses.focal.enabled:
            results["focal_loss"] = float((cfg.losses.focal.weight * (totals["loss_focal"] / denom)).item())
        if cfg.losses.dice.enabled:
            results["dice_loss"] = float((cfg.losses.dice.weight * (totals["loss_dice"] / denom)).item())
        if cfg.losses.cross_entropy.enabled:
            results["ce_loss"] = float((cfg.losses.cross_entropy.weight * (totals["loss_ce"] / denom)).item())
        if cfg.losses.iou.enabled:
            results["iou_loss"] = float((cfg.losses.iou.weight * (totals["loss_iou"] / denom)).item())

        display_str = f"Validation [{epoch}]: Loss: [{results['total_loss']:.4f}]"
        if "iou" in results:
            display_str += f" | Mean IoU: [{results['iou']:.4f}]"
        display_str += f" | Pred IoU: [{results['iou_pred']:.4f}]"
        if "dsc" in results:
            display_str += f" | Mean DSC: [{results['dsc']:.4f}]"
        if "hd95" in results:
            display_str += f" | Mean HD95: [{results['hd95']:.4f}]"
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

    if fabric is not None and _is_distributed(fabric):
        # Reduce sums and counts across ranks
        t_mean = mean.to(fabric.device, dtype=torch.float64)
        t_std = std.to(fabric.device, dtype=torch.float64)
        t_cnt = torch.tensor([float(total_images)], device=fabric.device, dtype=torch.float64)
        dist.all_reduce(t_mean, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_std, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_cnt, op=dist.ReduceOp.SUM)
        mean = t_mean.to(dtype=torch.float32)
        std = t_std.to(dtype=torch.float32)
        total_images = int(t_cnt.item())
            
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
    print_metrics: bool = True,
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
    if cfg.metrics.hd95.enabled:
        display_str += f' | HD95 [{metrics.hd95.val:.4f} ({metrics.hd95.avg:.4f})]'

    if print_metrics:
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
    if cfg.metrics.hd95.enabled:
        log_info['train_hd95'] = metrics.hd95.val
    if getattr(fabric, "global_rank", 0) == 0:
        fabric.log_dict(log_info, step=steps)


def plot_history(
        metrics_history: Dict[str, list],
        out_plots: str,
        eval_interval: int,
        num_epochs: int,
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
            'train_dsc', 'val_dsc', 'train_hd95', 'val_hd95'.
        out_plots (str): The path to the output directory where the
            '{name}.png' file will be saved.
        name (str): The base name for the saved plot file (without extension).

    Side Effects:
        - Saves a PNG image ('{name}.png') to the `out_plots`
          directory.
        - Prints error messages to stderr if 'epochs' data is missing or empty.
        - Prints a warning to stderr if the 'serif' font cannot be set.
    """
    
    # Ensure at least one data point exists
    if not metrics_history.get("epochs"):
        print("Error: No 'epochs' data found in metrics_history.", file=sys.stderr)
        return
    
    epochs = metrics_history["epochs"]
    
    # Calculate validation epochs based on config
    val_epochs = []
    for e in epochs:
        is_val = False
        if eval_interval > 0 and e % eval_interval == 0:
            is_val = True
        if e == num_epochs:
            is_val = True
        
        if is_val:
            val_epochs.append(e)
    
    # Create a set of unique sorted val epochs
    val_epochs = sorted(list(set(val_epochs)))

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

    fig, (ax_train_loss, ax_val_loss, ax_iou, ax_dsc, ax_hd95) = plt.subplots(1, 5, figsize=(50, 9)) 
    
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
    max_step = 20
    if max_epoch <= max_step:
        ticks = list(range(1, max_epoch + 1))
    else:
        ticks = [1] + list(range(max_step, max_epoch + 1, max_step)) 

    # --- Plot 1: Training Losses ---
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


    # --- Plot 2: Validation Losses ---
    if "val_total_loss" in metrics_history:
        ax_val_loss.plot(val_epochs, metrics_history["val_total_loss"], label="Val Total Loss", 
                     color=colors['total_loss'], linestyle='-', linewidth=line_width, marker='o')

    if "val_focal_loss" in metrics_history and any(metrics_history["val_focal_loss"]):
        ax_val_loss.plot(val_epochs, metrics_history["val_focal_loss"], label="Focal Loss", 
                    color=colors['focal_loss'], linestyle='-', linewidth=line_width, marker='o')
    
    if "val_dice_loss" in metrics_history and any(metrics_history["val_dice_loss"]):
        ax_val_loss.plot(val_epochs, metrics_history["val_dice_loss"], label="Dice Loss", 
                    color=colors['dice_loss'], linestyle='-', linewidth=line_width, marker='o')
    
    if "val_iou_loss" in metrics_history and any(metrics_history["val_iou_loss"]):
        ax_val_loss.plot(val_epochs, metrics_history["val_iou_loss"], label="IoU Loss", 
                    color=colors['iou_loss'], linestyle='-', linewidth=line_width, marker='o')
    if "val_ce_loss" in metrics_history and any(metrics_history["val_ce_loss"]):
        ax_val_loss.plot(val_epochs, metrics_history["val_ce_loss"], label="CE Loss", 
                     color=colors['ce_loss'], linestyle='-', linewidth=line_width, marker='o')
    
    ax_val_loss.set_title("Val Loss", loc='left')
    ax_val_loss.legend(loc='upper right', frameon=True, fancybox=True)
    ax_val_loss.set_xlabel("Epoch")
    ax_val_loss.set_ylabel("Value")
    ax_val_loss.grid(False)
    ax_val_loss.set_xticks(ticks)
    if max_epoch > 1:
        ax_val_loss.set_xlim(left=1, right=max_epoch)

    # --- Metrics Data ---
    train_iou_data = metrics_history["train_iou"]
    val_iou_data = metrics_history["val_iou"]
    train_dsc_data = metrics_history["train_dsc"]
    val_dsc_data = metrics_history["val_dsc"]
    
    # --- Calculate shared Y-axis limits for metrics ---
    min_metric_val = min(
        min(train_iou_data), 
        min(val_iou_data), 
        min(train_dsc_data), 
        min(val_dsc_data)
    )
    # Round down to the nearest 0.1 and add 0.05 padding
    shared_lower_lim = max(0, math.floor(min_metric_val * 10) / 10 - 0.05)
    shared_upper_lim = 1.0 # Fixed upper limit at 1.0


    # --- Plot 3: IoU ---
    ax_iou.plot(epochs, train_iou_data, label="Train IoU", 
                color=colors['train_set'], linestyle='-', linewidth=line_width)
    ax_iou.plot(val_epochs, val_iou_data, label="Val IoU", 
                color=colors['val_set'], linestyle='-', linewidth=line_width, marker='o')

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


    # --- Plot 4: Dice Score ---
    ax_dsc.plot(epochs, train_dsc_data, label="Train DSC", 
                color=colors['train_set'], linestyle='-', linewidth=line_width)
    ax_dsc.plot(val_epochs, val_dsc_data, label="Val DSC", 
                color=colors['val_set'], linestyle='-', linewidth=line_width, marker='o')

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

    # --- Plot 5: HD95 ---
    if "train_hd95" in metrics_history:
        train_hd95_data = metrics_history["train_hd95"]
        ax_hd95.plot(epochs, train_hd95_data, label="Train HD95", 
                    color=colors['train_set'], linestyle='-', linewidth=line_width)
    
    if "val_hd95" in metrics_history:
        val_hd95_data = metrics_history["val_hd95"]
        ax_hd95.plot(val_epochs, val_hd95_data, label="Val HD95", 
                    color=colors['val_set'], linestyle='-', linewidth=line_width, marker='o')

    ax_hd95.set_title("HD95", loc='left')
    ax_hd95.legend(loc='upper right', frameon=True, fancybox=True)
    ax_hd95.set_xlabel("Epoch")
    ax_hd95.set_ylabel("Value")
    ax_hd95.grid(False)
    ax_hd95.set_xticks(ticks)
    if max_epoch > 1:
        ax_hd95.set_xlim(left=1, right=max_epoch)

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


def save_metrics(
    *,
    split: str,
    out_dir: str,
    name: str = "metrics",
    epoch: Optional[int] = None,
    results: Optional[Dict[str, float]] = None,
    metrics_history: Optional[Dict[str, list]] = None,
):
    """
    Saves metrics to a text file based on the specified split.

    - split='train': expects `metrics_history` and writes the latest epoch row.
    - split='val': expects `epoch` and `results` and appends a row.
    - split='test': expects `results` and writes a single row (no epoch unless provided).

    Args:
        split (str): One of 'train', 'val', or 'test'.
        out_dir (str): Directory where the metrics file will be saved.
        name (str): Base name for the metrics file (default: "metrics").
        epoch (Optional[int]): Current epoch number (required for 'val' split).
        results (Optional[Dict[str, float]]): Dictionary of metric results for 'val' or 'test' splits.
        metrics_history (Optional[Dict[str, list]]): Dictionary of metric histories for 'train' split.
    """
    split = str(split).lower().strip()
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of: 'train', 'val', 'test'")

    if split == "train":
        if not metrics_history or not metrics_history.get("epochs"):
            return
        latest_idx = len(metrics_history["epochs"]) - 1
        epoch_val = metrics_history["epochs"][latest_idx]

        headers = ["Epoch", "Total Loss"]
        values = [epoch_val, metrics_history["total_loss"][latest_idx]]

        for key, label in (
            ("focal_loss", "Focal Loss"),
            ("dice_loss", "Dice Loss"),
            ("iou_loss", "IoU Loss"),
            ("ce_loss", "CE Loss"),
            ("train_iou", "Train IoU"),
            ("train_iou_pred", "Train Pred IoU"),
            ("train_dsc", "Train DSC"),
            ("train_hd95", "Train HD95"),
        ):
            if key in metrics_history and any(metrics_history[key]):
                headers.append(label)
                values.append(metrics_history[key][latest_idx])

        filename = f"train_{name}.txt"
        _write_metrics_file(out_dir, filename, headers, values)
        print(f"Metrix train saved to: {os.path.join(out_dir, filename)}")
        return

    # val/test
    if not results:
        return

    prefix = "Val" if split == "val" else "Test"
    headers: List[str] = []
    values: List[Union[int, float, str]] = []

    if epoch is not None:
        headers.append("Epoch")
        values.append(int(epoch))
    elif split == "val":
        raise ValueError("epoch is required when split='val'")

    def add(metric_key: str, label: str):
        if metric_key in results:
            headers.append(label)
            values.append(results[metric_key])

    add("total_loss", f"{prefix} Loss")
    add("focal_loss", f"{prefix} Focal Loss")
    add("dice_loss", f"{prefix} Dice Loss")
    add("ce_loss", f"{prefix} CE Loss")
    add("iou_loss", f"{prefix} IoU Loss")
    add("iou_pred", f"{prefix} Pred IoU")
    add("iou", f"{prefix} IoU")
    add("dsc", f"{prefix} DSC")
    add("hd95", f"{prefix} HD95")

    if not headers:
        return

    filename = f"{split}_{name}.txt"
    _write_metrics_file(out_dir, filename, headers, values)
    print(f"Metrix {split} saved to: {os.path.join(out_dir, filename)}")


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


def save_prediction_visual(
        *,
        out_dir: str,
        base_name: str,
        image: np.ndarray,
        gt_mask: Optional[torch.Tensor],
        pred_mask: torch.Tensor,
    ) -> str:
    """
    Save a prediction visualization.
    Save an image with the original image, the ground truth mask (green line), and the predicted mask (red line).
    
    Args:
        out_dir (str): Directory where the image will be saved.
        base_name (str): Base name for the image file.
        image (np.ndarray): The image to save.
        gt_mask (Optional[torch.Tensor]): The ground truth mask (optional).
        pred_mask (torch.Tensor): The predicted mask.
    """
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.axis("off")
    ax.imshow(image)

    def _prepare_binary_mask(mask: torch.Tensor) -> np.ndarray:
        mask_np = mask.detach().cpu().numpy()
        if mask_np.ndim > 2:
            mask_np = np.any(mask_np, axis=0)
        mask_np = np.squeeze(mask_np)
        return (mask_np > 0.5).astype(np.uint8)

    legend_handles = []

    if gt_mask is not None:
        gt_np = _prepare_binary_mask(gt_mask)
        if gt_np.any():
            ax.contour(gt_np, levels=[0.5], colors="lime", linewidths=2.0, linestyles="-")
            legend_handles.append(Line2D([0], [0], color="lime", lw=2, label="GT"))

    pred_np = _prepare_binary_mask(pred_mask)
    if pred_np.any():
        ax.contour(pred_np, levels=[0.5], colors="red", linewidths=2.0, linestyles="-")
        legend_handles.append(Line2D([0], [0], color="red", lw=2, label="Prediction"))

    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right")
    ax.set_title("GT (Green) vs Prediction (Red)")

    fig.tight_layout()
    output_path = os.path.join(out_dir, f"{base_name}.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return output_path


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