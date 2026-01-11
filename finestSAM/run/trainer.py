import os
import time
import torch
import lightning as L
import torch.nn.functional as F
from box import Box
from torch.utils.data import DataLoader
from lightning.fabric.fabric import _FabricOptimizer
from lightning.fabric.loggers import TensorBoardLogger
from finestSAM.model.losses import (
    DiceLoss,
    FocalLoss,
    CalcIoU,
    CalcDSC
)
from finestSAM.utils import (
    Metrics,
    WarmupReduceLROnPlateau,
    configure_opt,
    validate,
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

    with fabric.device:
        model = FinestSAM(cfg)
        model.setup()
        model.train()
        model.to(fabric.device)

    # Load the dataset
    train_data, val_data = load_dataset(cfg, model.model.image_encoder.img_size, dataset_path, fabric=fabric)
    train_data = fabric._setup_dataloader(train_data)
    val_data = fabric._setup_dataloader(val_data)

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
    focal_loss = FocalLoss(gamma=cfg.losses.focal_gamma, alpha=cfg.losses.focal_alpha)
    dice_loss = DiceLoss()
    calc_iou = CalcIoU()
    calc_dsc = CalcDSC()

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
        "iou_loss": [],
        "train_iou": [],
        "train_dsc": [],
        "val_iou": [],
        "val_dsc": [],
        "epochs": [],
    }

    # Initial validation
    val_iou, val_dsc = 0., 0.
    if cfg.eval_interval > 0:
        val_iou, val_dsc = validate(fabric, cfg, model, val_dataloader, 0)
        if fabric.global_rank == 0:
            save_val_metrics(0, val_iou, val_dsc, cfg.out_dir)

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
                batch_iou = calc_iou(pred_masks, data["gt_masks"])
                batch_dsc = calc_dsc(pred_masks, data["gt_masks"])
                
                iter_metrics["iou"] += torch.mean(batch_iou)
                iter_metrics["dsc"] += torch.mean(batch_dsc)
                iter_metrics["iou_pred"] += torch.mean(iou_predictions)

                # Calculate the losses
                iter_metrics["loss_focal"] += focal_loss(pred_masks, data["gt_masks"].float(), len(pred_masks)) 
                iter_metrics["loss_dice"] += dice_loss(pred_masks, data["gt_masks"].float(), len(pred_masks))
                iter_metrics["loss_iou"] += F.mse_loss(iou_predictions, batch_iou, reduction='mean')

            loss_total = cfg.losses.focal_ratio * iter_metrics["loss_focal"] + cfg.losses.dice_ratio * iter_metrics["loss_dice"] + cfg.losses.iou_ratio * iter_metrics["loss_iou"]

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
            epoch_metrics.focal_losses.update(iter_metrics["loss_focal"].item(), batch_size)
            epoch_metrics.dice_losses.update(iter_metrics["loss_dice"].item(), batch_size)
            epoch_metrics.space_iou_losses.update(iter_metrics["loss_iou"].item(), batch_size)
            epoch_metrics.total_losses.update(loss_total.item(), batch_size)
            epoch_metrics.ious.update(iter_metrics["iou"].item()/batch_size, batch_size)
            epoch_metrics.ious_pred.update(iter_metrics["iou_pred"].item()/batch_size, batch_size)
            epoch_metrics.dsc.update(iter_metrics["dsc"].item()/batch_size, batch_size)

            print_and_log_metrics(fabric, cfg, epoch, iter, epoch_metrics, train_dataloader)

        # Step the scheduler if it is ReduceLROnPlateau or WarmupReduceLROnPlateau
        if isinstance(scheduler, (torch.optim.lr_scheduler.ReduceLROnPlateau, WarmupReduceLROnPlateau)):
            scheduler.step(epoch_metrics.total_losses.avg)
            if scheduler.get_last_lr() != last_lr:
                last_lr = scheduler.get_last_lr()
                fabric.print(f"learning rate changed to: {last_lr}")
                log_event(cfg.out_dir, f"Epoch {epoch}: Learning rate changed to {last_lr}")

        if (cfg.eval_interval > 0 and epoch % cfg.eval_interval == 0) or (epoch == cfg.num_epochs):
            
            val_iou, val_dsc = validate(fabric, cfg, model, val_dataloader, epoch)

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
                save_val_metrics(epoch, val_iou, val_dsc, cfg.out_dir)

        metrics_history["epochs"].append(epoch)
        metrics_history["total_loss"].append(epoch_metrics.total_losses.avg)
        metrics_history["focal_loss"].append(cfg.losses.focal_ratio * epoch_metrics.focal_losses.avg)
        metrics_history["dice_loss"].append(cfg.losses.dice_ratio * epoch_metrics.dice_losses.avg)
        metrics_history["iou_loss"].append(cfg.losses.iou_ratio * epoch_metrics.space_iou_losses.avg)
        metrics_history["train_iou"].append(epoch_metrics.ious.avg)
        metrics_history["train_dsc"].append(epoch_metrics.dsc.avg)
        metrics_history["val_iou"].append(val_iou)
        metrics_history["val_dsc"].append(val_dsc)

        if fabric.global_rank == 0:
            plot_history(metrics_history, cfg.out_dir)
            save_train_metrics(metrics_history, cfg.out_dir)