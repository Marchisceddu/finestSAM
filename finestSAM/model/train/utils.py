import os
import torch
import lightning as L
import segmentation_models_pytorch as smp
from ..model import FinestSAM
from typing import Tuple
from box import Box
from lightning.fabric.fabric import _FabricOptimizer
from torch.utils.data import DataLoader

from ..predictions.utils import (
    show_mask,
)
import matplotlib.pyplot as plt


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
    def __init__(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()

        self.focal_losses = AverageMeter()
        self.dice_losses = AverageMeter()
        self.space_iou_losses = AverageMeter()
        self.total_losses = AverageMeter()

        self.ious = AverageMeter()
        self.ious_pred = AverageMeter()


def configure_opt(cfg: Box, model: FinestSAM) -> Tuple[_FabricOptimizer, _FabricOptimizer]:

    def lr_lambda(step):
        if step < cfg.opt.warmup_steps:
            return step / cfg.opt.warmup_steps
        elif cfg.opt.steps == None or step < cfg.opt.steps[0]:
            return 1.0
        elif step < cfg.opt.steps[1]:
            return 1 / cfg.opt.decay_factor
        else:
            return 1 / (cfg.opt.decay_factor**2)

    optimizer = torch.optim.Adam(model.model.parameters(), lr=cfg.opt.learning_rate, weight_decay=cfg.opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


def save(
    fabric: L.Fabric, 
    model: FinestSAM, 
    out_dir: str,
    name: str = "ckpt"
):
    """
    Save the model checkpoint.
    
    Args:
        fabric (L.Fabric): The lightning fabric.
        model (FinestSAM): The model.
        out_dir (str): The output directory.
        name (str): The name of the checkpoint without .pth.
    """

    fabric.print(f"Saving checkpoint to {out_dir}")
    name = name + ".pth"
    state_dict = model.model.state_dict()
    if fabric.global_rank == 0:
        torch.save(state_dict, os.path.join(out_dir, name))


def validate(
        fabric: L.Fabric, 
        cfg: Box,
        model: FinestSAM, 
        val_dataloader: DataLoader, 
        epoch: int,
        last_score: float = 0.
    ) -> float: 
    """
    Validation function for the SAM model.

    Args:
        fabric (L.Fabric): The lightning fabric.
        cfg (Box): The configuration file.
        model (FinestSAM): The model.
        val_dataloader (DataLoader): The validation dataloader.
        epoch (int): The current epoch.
        last_score (float): The last score.
    """
    model.eval()
    ious = AverageMeter()
    
    with torch.no_grad():
        for iter, batched_data in enumerate(val_dataloader):

            predictor = model.get_predictor()
            
            pred_masks = []
            for data in batched_data:
                predictor.set_image(data["original_image"])
                masks, stability_scores, _  = predictor.predict_torch(
                    point_coords=data["point_coords"],
                    point_labels=data["point_labels"],
                    boxes=data["boxes"],
                    multimask_output=cfg.multimask_output,
                )

                if cfg.multimask_output:
                    # For each mask, get the mask with the highest stability score
                    separated_masks = torch.unbind(masks, dim=1)
                    separated_scores = torch.unbind(stability_scores, dim=1)

                    stability_score = [torch.mean(score) for score in separated_scores]
                    pred_masks.append(separated_masks[torch.argmax(torch.tensor(stability_score))])

                    # STAMPA DI DEBUG ELIMINARE
                    plt.imshow(data["original_image"])
                    for i, mask in enumerate(separated_masks[torch.argmax(torch.tensor(stability_score))]):
                        show_mask(mask, plt.gca(), seed=i)
                    plt.axis('off')
                    plt.savefig(os.path.join(cfg.out_dir, "m.png"))
                    plt.show()
                    plt.clf()
                    plt.close('all')
                else:
                    pred_masks.append(masks)

            gt_masks = [data["gt_masks"] for data in batched_data]  
            num_images = len(batched_data)
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
                f'Val: [{epoch}] - [{iter+1}/{len(val_dataloader)}]: Mean IoU: [{ious.avg:.4f}]'
            )

        fabric.print(f'Validation [{epoch}]: Mean IoU: [{ious.avg:.4f}]')

        if ious.avg > last_score + (cfg.eval_improvement * last_score):
            last_score = ious.avg
            save(fabric, model, cfg.sav_dir, "best")

    model.train()

    return last_score


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
    
    Args:
        fabric (L.Fabric): The fabric object.
        cfg (Box): The configuration file.
        epoch (int): The current epoch.
        iter (int): The current iteration.
        metrics (Metrics): The metrics object.
        train_dataloader (DataLoader): The training dataloader.
    """

    fabric.print(f'Epoch: [{epoch}][{iter+1}/{len(train_dataloader)}]'
                 f' | Time [{metrics.batch_time.val:.3f}s ({metrics.batch_time.avg:.3f}s)]'
                 f' | Data [{metrics.data_time.val:.3f}s ({metrics.data_time.avg:.3f}s)]'
                 f' | Focal Loss [{metrics.focal_losses.val:.4f} ({metrics.focal_losses.avg:.4f})]'
                 f' | Dice Loss [{metrics.dice_losses.val:.4f} ({metrics.dice_losses.avg:.4f})]'
                 f' | Space IoU Loss [{metrics.space_iou_losses.val:.4f} ({metrics.space_iou_losses.avg:.4f})]'
                 f' | Total Loss [{metrics.total_losses.val:.4f} ({metrics.total_losses.avg:.4f})]'
                 f' | IoU [{metrics.ious.val:.4f} ({metrics.ious.avg:.4f})]'
                 f' | Pred IoU [{metrics.ious_pred.val:.4f} ({metrics.ious_pred.avg:.4f})]')
    steps = epoch * len(train_dataloader) + iter
    log_info = {
        'total loss': metrics.total_losses.val,
        'focal loss': metrics.focal_losses.val,
        'dice loss':  metrics.dice_losses.val,
    }
    fabric.log_dict(log_info, step=steps)

def print_graphs(metrics: dict[list], out_plots: str):
    """
    Print the graphs for the metrics.
    
    Args:
        metrics (dict[list]): The metrics to plot.
        out_plots (str): The output directory for the plots.
    """
    metric_names = ["dice_loss", "focal_loss", "space_iou_loss", "total_loss", "iou", "iou_pred"]

    for metric_name in metric_names:
        plt.plot(metrics[metric_name], label=metric_name.capitalize())
        plt.title(metric_name.capitalize())
        plt.legend()
        plt.savefig(os.path.join(out_plots, f"{metric_name}.png"))
        plt.show()
        plt.clf()
        
    for metric_name in metric_names:
        plt.plot(metrics[metric_name], label=metric_name.capitalize())
    plt.title("All Metrics")
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()

    plt2 = plt.gca().twinx()
    plt2.plot(metrics["total_loss"], color='black', linestyle='--', label="Total Loss")
    plt2.set_ylabel('Total Loss')

    plt.legend(loc='upper left')
    plt.savefig(os.path.join(out_plots, "all_metrics.png"))
    plt.show()
    plt.clf()

    plt.close('all')