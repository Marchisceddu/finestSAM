import os
import sys
import math
import torch
import numpy as np
import lightning as L
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from box import Box
from typing import Tuple, Dict, Union, Optional, Any
from torch.utils.data import DataLoader
from lightning.fabric.fabric import _FabricOptimizer
from finestSAM.core.model import FinestSAM


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Metrics:
    """Metrics class for training and validation.
    
    Attributes:
        - batch_time: Average processing time per batch.
        - data_time: Average data loading time per batch.
        - focal_losses: Average focal loss.
        - dice_losses: Average dice loss.
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
        self.space_iou_losses = AverageMeter()
        self.total_losses = AverageMeter()

        self.ious = AverageMeter()
        self.ious_pred = AverageMeter()
        self.dsc = AverageMeter()


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

    # Configure scheduler
    if cfg.sched.type == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, 
                                                               factor=cfg.sched.ReduceLROnPlateau.decay_factor, 
                                                               patience=cfg.sched.ReduceLROnPlateau.epoch_patience, 
                                                               threshold=cfg.sched.ReduceLROnPlateau.threshold, 
                                                               cooldown=cfg.sched.ReduceLROnPlateau.cooldown,
                                                               min_lr=cfg.sched.ReduceLROnPlateau.min_lr)
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    return optimizer, scheduler


def validate(
        fabric: L.Fabric, 
        cfg: Box,
        model: FinestSAM, 
        val_dataloader: DataLoader, 
        epoch: int
    ) -> Tuple[float, float]: 
    """
    Validation function
    Computes IoU and Dice Score (F1 Score) for the validation dataset.

    Args:
        fabric (L.Fabric): The lightning fabric.
        cfg (Box): The configuration file.
        model (FinestSAM): The model.
        val_dataloader (DataLoader): The validation dataloader.
        epoch (int): The current epoch.
        
    Returns:
        Tuple[float, float]: (mean_iou, mean_dice)
    """
    model.eval()
    ious = AverageMeter()
    dsc = AverageMeter()
    
    with torch.no_grad():
        for iter, batched_data in enumerate(val_dataloader):

            predictor = model.get_predictor()
            
            # Generate predictions for each image in the batch
            pred_masks = []
            for data in batched_data:
                predictor.set_image(data["original_image"])
                masks, stability_scores, _  = predictor.predict_torch(
                    point_coords=data.get("point_coords", None),
                    point_labels=data.get("point_labels", None),
                    boxes=data.get("boxes", None),
                    multimask_output=cfg.multimask_output,
                )

                if cfg.multimask_output:
                    # For each mask, get the mask with the highest stability score
                    separated_masks = torch.unbind(masks, dim=1)
                    separated_scores = torch.unbind(stability_scores, dim=1)

                    stability_score = [torch.mean(score) for score in separated_scores]
                    pred_masks.append(separated_masks[torch.argmax(torch.tensor(stability_score))])
                else:
                    pred_masks.append(masks.squeeze(1))

            gt_masks = [data["gt_masks"] for data in batched_data]  
            num_images = len(batched_data)
            
            # Compute IoU and Dice for each image in the batch
            for pred_mask, gt_mask in zip(pred_masks, gt_masks):
                batch_stats = smp.metrics.get_stats(
                    pred_mask,
                    gt_mask.int(),
                    mode='binary',
                    threshold=0.5,
                )

                batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
                ious.update(batch_iou.item(), num_images)
                
                batch_dice = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")
                dsc.update(batch_dice.item(), num_images)
            
            fabric.print(f'Val: [{epoch}] - [{iter+1}/{len(val_dataloader)}]: Mean IoU: [{ious.avg:.4f}] | Mean DSC: [{dsc.avg:.4f}]')

        fabric.print(f'Validation [{epoch}]: Mean IoU: [{ious.avg:.4f}] | Mean DSC: [{dsc.avg:.4f}]')

    model.train()

    return ious.avg, dsc.avg


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
    fabric.print(f'Epoch: [{epoch}][{iter+1}/{len(train_dataloader)}]'
                 f' | Time [{metrics.batch_time.val:.3f}s ({metrics.batch_time.avg:.3f}s)]'
                 f' | Data [{metrics.data_time.val:.3f}s ({metrics.data_time.avg:.3f}s)]'
                 f' | Focal Loss [{cfg.losses.focal_ratio * metrics.focal_losses.val:.4f} ({cfg.losses.focal_ratio * metrics.focal_losses.avg:.4f})]'
                 f' | Dice Loss [{cfg.losses.dice_ratio * metrics.dice_losses.val:.4f} ({cfg.losses.dice_ratio * metrics.dice_losses.avg:.4f})]'
                 f' | Space IoU Loss [{cfg.losses.iou_ratio * metrics.space_iou_losses.val:.4f} ({cfg.losses.iou_ratio * metrics.space_iou_losses.avg:.4f})]'
                 f' | Total Loss [{metrics.total_losses.val:.4f} ({metrics.total_losses.avg:.4f})]'
                 f' | IoU [{metrics.ious.val:.4f} ({metrics.ious.avg:.4f})]'
                 f' | Pred IoU [{metrics.ious_pred.val:.4f} ({metrics.ious_pred.avg:.4f})]'
                 f' | DSC [{metrics.dsc.val:.4f} ({metrics.dsc.avg:.4f})]')
    
    steps = epoch * len(train_dataloader) + iter    
    log_info = {
        'total loss': metrics.total_losses.val,
        'focal loss': cfg.losses.focal_ratio * metrics.focal_losses.val,
        'dice loss':  cfg.losses.dice_ratio * metrics.dice_losses.val,
        'iou loss':   cfg.losses.iou_ratio * metrics.space_iou_losses.val,
        'train_iou':  metrics.ious.val,
        'train_dsc': metrics.dsc.val,
    }
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
            'focal_loss', 'dice_loss', 'iou_loss', 'train_iou', 'val_iou', 
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

    # --- Create the Figure with 3 side-by-side Subplots ---
    fig, (ax_loss, ax_iou, ax_dsc) = plt.subplots(1, 3, figsize=(33, 9)) 
    
    colors = {
        'total_loss': '#d62728',  # Red
        'dice_loss': '#17becf',   # Cyan
        'focal_loss': '#ff7f0e',  # Orange
        'iou_loss': '#2ca02c',    # Green
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

    ax_loss.plot(epochs, metrics_history["total_loss"], label="Total Loss", 
                 color=colors['total_loss'], linestyle='-', linewidth=line_width)
    ax_loss.plot(epochs, metrics_history["focal_loss"], label="Focal Loss", 
                 color=colors['focal_loss'], linestyle='-', linewidth=line_width)
    ax_loss.plot(epochs, metrics_history["dice_loss"], label="Dice Loss", 
                 color=colors['dice_loss'], linestyle='-', linewidth=line_width)
    ax_loss.plot(epochs, metrics_history["iou_loss"], label="IoU Loss", 
                 color=colors['iou_loss'], linestyle='-', linewidth=line_width)
    
    ax_loss.set_title("Loss", loc='left')
    ax_loss.legend(loc='upper right', frameon=True, fancybox=True)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Value")
    ax_loss.grid(False)
    ax_loss.set_xticks(ticks)
    if max_epoch > 1:
        ax_loss.set_xlim(left=1, right=max_epoch)
    # Automatic Y-scale

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


    # --- Plot 2: IoU (Center) ---
    
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


    # --- Plot 3: Dice Score (Right) ---
    
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

    train_headers = [
        "Epoch", "Total Loss", "Focal Loss", "Dice Loss", "IoU Loss", 
        "Train IoU", "Train DSC"
    ]
    train_values = [
        epoch,
        metrics_history["total_loss"][latest_idx],
        metrics_history["focal_loss"][latest_idx],
        metrics_history["dice_loss"][latest_idx],
        metrics_history["iou_loss"][latest_idx],
        metrics_history["train_iou"][latest_idx],
        metrics_history["train_dsc"][latest_idx]
    ]
    
    train_filename = f"train_{name}.txt"
    _write_metrics_file(out_dir, train_filename, train_headers, train_values)
    print(f"Metrix train saved to: {os.path.join(out_dir, train_filename)}")


def save_val_metrics(
    epoch: int,
    val_iou: float,
    val_dsc: float,
    out_dir: str,
    name: str = "metrics"
):
    """
    Saves the validation metrics to a text file.
    
    Args:
        epoch (int): Current epoch.
        val_iou (float): Validation IoU.
        val_dsc (float): Validation DSC.
        out_dir (str): Directory where the file will be saved.
        name (str): Base name of the output file (default: "metrics").
    """
    val_headers = [
        "Epoch", "Val IoU", "Val DSC"
    ]
    val_values = [
        epoch,
        val_iou,
        val_dsc
    ]
    
    val_filename = f"val_{name}.txt"
    _write_metrics_file(out_dir, val_filename, val_headers, val_values)
    print(f"Metrix val saved to: {os.path.join(out_dir, val_filename)}")


def _write_metrics_file(out_dir, filename, headers, values):
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