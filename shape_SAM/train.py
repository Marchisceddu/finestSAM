import os
import torch
import lightning as L
import segmentation_models_pytorch as smp
from model.config import cfg
from model.dataset import load_datasets
from model.model import shape_SAM
from model.utils import AverageMeter
from model.train import (
    Train_custom, 
    Train_11_iterations,
    configure_opt
)
from box import Box
from lightning.fabric.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

torch.autograd.set_detect_anomaly(True)
torch.set_float32_matmul_precision('high')


def validate(fabric: L.Fabric, model: shape_SAM, val_dataloader: DataLoader, epoch: int = 0): # CAMBIARE PER FUNZIONARE CON MULTIMASK TRUE E CON TUTTI I CAMBIAMENTI CHE SONO STATI FATTI
    model.eval()
    ious = AverageMeter()
    f1_scores = AverageMeter()

    with torch.no_grad():
        for iter, data in enumerate(val_dataloader):
            images, bboxes, gt_masks, centers = data
            num_images = images.size(0)
            pred_masks, _ = model(images, bboxes=bboxes, centers=centers)
            for pred_mask, gt_mask in zip(pred_masks, gt_masks):
                batch_stats = smp.metrics.get_stats(
                    pred_mask,
                    gt_mask.int(),
                    mode='binary',
                    threshold=0.5,
                )
                batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
                batch_f1 = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")
                ious.update(batch_iou, num_images)
                f1_scores.update(batch_f1, num_images)
            fabric.print(
                f'Val: [{epoch}] - [{iter}/{len(val_dataloader)}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]'
            )

    fabric.print(f'Validation [{epoch}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]')

    fabric.print(f"Saving checkpoint to {cfg.out_dir}")
    state_dict = model.model.state_dict()
    if fabric.global_rank == 0:
        torch.save(state_dict, os.path.join(cfg.out_dir, f"epoch-{epoch:06d}-f1{f1_scores.avg:.2f}-ckpt.pth"))
    model.train()


def main(cfg: Box) -> None:
    main_directory = os.path.dirname(os.path.abspath(__file__))
    cfg.out_dir = os.path.join(main_directory, cfg.out_dir)

    fabric = L.Fabric(accelerator="auto",
                      devices=cfg.num_devices,
                      strategy="auto",
                      loggers=[TensorBoardLogger(cfg.out_dir, name="loggers_shape_SAM")])
    fabric.launch()

    fabric.seed_everything(cfg.seed_device)

    if fabric.global_rank == 0:
        os.makedirs(cfg.out_dir, exist_ok=True)

    with fabric.device:
        model = shape_SAM(cfg)
        model.setup()
        model.to(fabric.device)

    train_data, val_data = load_datasets(cfg, model.model.image_encoder.img_size)
    train_data = fabric._setup_dataloader(train_data)
    val_data = fabric._setup_dataloader(val_data)

    optimizer, scheduler = configure_opt(cfg, model)
    model, optimizer = fabric.setup(model, optimizer)

    train = None
    if cfg.train_type == "custom":
        train = Train_custom(cfg, fabric, model, optimizer, scheduler, train_data, val_data)
    elif cfg.train_type == "11-iteration":
        train = Train_11_iterations(cfg, fabric, model, optimizer, scheduler, train_data, val_data)
    else:
        raise ValueError(f"Unknown training type: {cfg.train_type}")
    
    train()
    #validate(fabric, model, train_data, epoch=0)


if __name__ == "__main__":
    main(cfg)
