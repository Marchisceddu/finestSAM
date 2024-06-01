import os
import torch
import lightning as L
import segmentation_models_pytorch as smp
from ..model import shape_SAM
from typing import Tuple
from box import Box
from lightning.fabric.fabric import _FabricOptimizer
from torch.utils.data import DataLoader


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


def save(
    fabric: L.Fabric, 
    model: shape_SAM, 
    out_dir: str,
    epoch: int,
):
    """Save the model checkpoint."""

    fabric.print(f"Saving checkpoint to {out_dir}")
    state_dict = model.model.state_dict()
    if fabric.global_rank == 0:
        torch.save(state_dict, os.path.join(out_dir, f"epoch-{epoch:06d}-ckpt.pth"))


def validate(
        fabric: L.Fabric, 
        cfg: Box,
        model: shape_SAM, 
        val_dataloader: DataLoader, 
        epoch: int,
    ): 
    """Validation function for the SAM model.""" # Aggiungere la possibilità di scegliere su cosa validare punti o box o entrambi

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
    save(fabric, model, cfg.out_dir, epoch)
    model.train()