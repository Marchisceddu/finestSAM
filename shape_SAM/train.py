
import os
import cv2
import time
import numpy as np
import lightning as L
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from box import Box
from model.config import cfg
from model.dataset import load_datasets
from lightning.fabric.fabric import _FabricOptimizer
from lightning.fabric.loggers import TensorBoardLogger
from model.model import shape_SAM
from torch.utils.data import DataLoader
from model.utils import (
    AverageMeter
)
from model.losses import (
    Calc_iou,
    DiceLoss,
    FocalLoss
)

torch.autograd.set_detect_anomaly(True)
torch.set_float32_matmul_precision('high')

def save(
    fabric: L.Fabric, 
    model: shape_SAM, 
    cfg: Box,
    epoch: int = 0,
    f1_scores: AverageMeter = AverageMeter(),
):
    fabric.print(f"Saving checkpoint to {cfg.out_dir}")
    state_dict = model.model.state_dict()
    if fabric.global_rank == 0:
        torch.save(state_dict, os.path.join(cfg.out_dir, f"epoch-{epoch:06d}-f1{f1_scores.avg:.2f}-ckpt.pth"))
    model.train()


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


def train_sam(
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
                #take mask, iou_prediction and low_res_logits from the output
                batched_pred_masks.append(item["masks"])
                iou_predictions.append(item["iou_predictions"])
                
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

                ### STAMPA (ELIMINARE)
                stamp = pred_masks[2].clone() > 0.0 # elimina il gradiente dalla maschera predetta e trasforma in bool per essere stampata
                single_frame = data["imo"]
                annotation_rgb = np.zeros_like(single_frame)
                annotation_rgb[stamp.squeeze().cpu().numpy()] = [1, 255, 255] 
                
                '''
                if(epoch > 1):
                  for i,mask in enumerate(pred_masks[2]):
                    img = data["imo"]
                    test = test.squeeze()
                    for p in test[2]:
                      p = p.cpu().numpy()
                      x, y = p
                      x=round(x)
                      y=round(y)
                      cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
                '''
                
                # annotation = gt_mask[2].squeeze().cpu().numpy() * 255
                # annotation_rgb = np.repeat(annotation[..., np.newaxis], 3, axis=2).astype(np.uint8)
                
                image_with_annotation = cv2.addWeighted(data["imo"], 1, annotation_rgb, 0.5, 0)
                plt.imshow(image_with_annotation)
                plt.axis('off')
                plt.show()
                
                ### FINE STAMPA

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
            #validate(fabric, model, val_dataloader, epoch)
            save(fabric, model, cfg, epoch)


def configure_opt(cfg: Box, model: shape_SAM):

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


def main(cfg: Box) -> None:
    current_directory = os.path.dirname(os.path.abspath(__file__))
    cfg.out_dir = os.path.join(current_directory, cfg.out_dir)

    fabric = L.Fabric(accelerator="cpu",
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
    #val_data = fabric._setup_dataloader(val_data)

    #optimizer, scheduler = configure_opt(cfg, model)
    #model, optimizer = fabric.setup(model, optimizer)

    for iter, batched_data in enumerate(train_data):
        continue

    #train_sam(cfg, fabric, model, optimizer, scheduler, train_data, val_data)
    #validate(fabric, model, train_data, epoch=0)


if __name__ == "__main__":
    main(cfg)
