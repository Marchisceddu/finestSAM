import os
import time
import torch
import lightning as L
import torch.nn.functional as F
from box import Box
from torch.utils.data import DataLoader
from lightning.fabric.fabric import _FabricOptimizer
from lightning.fabric.loggers import TensorBoardLogger
from .utils import (
    AverageMeter,
    Metrics,
    validate,
    print_and_log_metrics,
    print_graphs,
    save
)
from .losses import (
    CalcIoU,
    DiceLoss,
    FocalLoss
)
from ..model import FinestSAM
from .utils import configure_opt
from ..dataset import load_dataset


def call_train(cfg: Box):
    # Set up the output directory
    main_directory = os.path.dirname(os.path.abspath(__file__)).rsplit('/', 2)[0]
    cfg.sav_dir = os.path.join(main_directory, cfg.sav_dir)
    cfg.out_dir = os.path.join(main_directory, cfg.out_dir)

    loggers = [TensorBoardLogger(cfg.sav_dir, name="loggers_finestSAM")]

    fabric = L.Fabric(accelerator=cfg.device,
                      devices=cfg.num_devices,
                      strategy="auto",
                      num_nodes=cfg.num_nodes, 
                      loggers=loggers)
    
    fabric.launch(train, cfg)


def train(fabric, *args, **kwargs):
    """
    Main training function.
    
    Args:
        fabric (L.Fabric): The lightning fabric.
        *args: The positional arguments:
            [0] - cfg (Box): The configuration file.
        **kwargs: The keyword arguments:
            not used.
    """
    # Get the arguments
    cfg = args[0]

    fabric.seed_everything(cfg.seed_device)

    if fabric.global_rank == 0: 
        os.makedirs(os.path.join(cfg.sav_dir, "loggers_finestSAM"), exist_ok=True)

    with fabric.device:
        model = FinestSAM(cfg)
        model.setup()
        model.train()
        model.to(fabric.device)

    # Load the dataset
    train_data, val_data = load_dataset(cfg, model.model.image_encoder.img_size)
    train_data = fabric._setup_dataloader(train_data)
    val_data = fabric._setup_dataloader(val_data)

    # Configure the optimizer and scheduler
    optimizer, scheduler = configure_opt(cfg, model)
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
    
    Args:
        cfg (Box): The configuration file.
        fabric (L.Fabric): The lightning fabric.
        model (FinestSAM): The model.
        optimizer (_FabricOptimizer): The optimizer.
        scheduler (_FabricOptimizer): The scheduler.
        train_dataloader (DataLoader): The training dataloader.
        val_dataloader (DataLoader): The validation dataloader.
    """

    # Initialize the losses
    focal_loss = FocalLoss(gamma=cfg.losses.focal_gamma, alpha=cfg.losses.focal_alpha)
    dice_loss = DiceLoss()
    calc_iou = CalcIoU()

    if cfg.prompts.use_logits: cfg.prompts.use_masks = False
    epoch_logits = []

    val_score = 0.
    last_lr = scheduler.get_last_lr()

    out_plots = os.path.join(cfg.out_dir, "plots")
    os.makedirs(out_plots, exist_ok=True)
    metrics = {
        "focal_loss": [],
        "dice_loss": [],
        "space_iou_loss": [],
        "total_loss": [],
        "iou": [],
        "iou_pred": [],
    }

    for epoch in range(1, cfg.num_epochs+1):
        # Initialize the meters
        epoch_metrics = Metrics()
        end = time.time()

        for iter, batched_data in enumerate(train_dataloader):
            torch.cuda.empty_cache()

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

            batch_size = len(batched_data)

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
                else:
                    pred_masks = pred_masks.squeeze(1)
                    iou_predictions = iou_predictions.squeeze(1)
                    logits = logits.squeeze(1)

                if cfg.prompts.use_logits: epoch_logits.append(logits)

                # Aggiorna metriche
                batch_iou = calc_iou(pred_masks, data["gt_masks"])
                iter_metrics["iou"] += torch.mean(batch_iou)
                iter_metrics["iou_pred"] += torch.mean(iou_predictions)

                # Calcola loss
                iter_metrics["loss_focal"] += focal_loss(pred_masks, data["gt_masks"], len(pred_masks)) 
                iter_metrics["loss_dice"] += dice_loss(pred_masks, data["gt_masks"], len(pred_masks))
                iter_metrics["loss_iou"] += F.mse_loss(iou_predictions, batch_iou, reduction='mean')

            loss_total = cfg.losses.focal_ratio * iter_metrics["loss_focal"] + cfg.losses.dice_ratio * iter_metrics["loss_dice"] + cfg.losses.iou_ratio * iter_metrics["loss_iou"]

            # Backward pass
            optimizer.zero_grad()
            fabric.backward(loss_total)
            optimizer.step()

            epoch_metrics.batch_time.update(time.time() - end)
            end = time.time()

            # Update the meters
            epoch_metrics.focal_losses.update(iter_metrics["loss_focal"].item(), batch_size)
            epoch_metrics.dice_losses.update(iter_metrics["loss_dice"].item(), batch_size)
            epoch_metrics.space_iou_losses.update(iter_metrics["loss_iou"].item(), batch_size)
            epoch_metrics.total_losses.update(loss_total.item(), batch_size)
            epoch_metrics.ious.update(iter_metrics["iou"].item()/batch_size, batch_size)
            epoch_metrics.ious_pred.update(iter_metrics["iou_pred"].item()/batch_size, batch_size)

            print_and_log_metrics(fabric, cfg, epoch, iter, epoch_metrics, train_dataloader)

        scheduler.step(epoch_metrics.total_losses.avg)
        if scheduler.get_last_lr() != last_lr:
            last_lr = scheduler.get_last_lr()
            print(f"learning rate changed to: {last_lr}")

        # Validate the model
        if (cfg.eval_interval > 0 and epoch % cfg.eval_interval == 0) or (epoch == cfg.num_epochs):
            val_score = validate(fabric, cfg, model, val_dataloader, epoch, val_score)
            
        # Aggiorna le metriche per i plot
        metrics["dice_loss"].append(cfg.losses.dice_ratio * epoch_metrics.dice_losses.avg)
        metrics["focal_loss"].append(cfg.losses.focal_ratio * epoch_metrics.focal_losses.avg)
        metrics["space_iou_loss"].append(cfg.losses.iou_ratio * epoch_metrics.space_iou_losses.avg)
        metrics["total_loss"].append(epoch_metrics.total_losses.avg)
        metrics["iou"].append(epoch_metrics.ious.avg)
        metrics["iou_pred"].append(epoch_metrics.ious_pred.avg)

        print_graphs(metrics, out_plots)