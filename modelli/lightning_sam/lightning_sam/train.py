import os
import time

import lightning as L
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from box import Box
from config import cfg
from dataset import load_datasets
from lightning.fabric.fabric import _FabricOptimizer
from lightning.fabric.loggers import TensorBoardLogger
from losses import DiceLoss
from losses import FocalLoss
from model import Model
from torch.utils.data import DataLoader
from utils import AverageMeter
from utils import calc_iou
from tqdm import tqdm
from sklearn.model_selection import KFold

torch.set_float32_matmul_precision('high')


def validate(
    fabric: L.Fabric, 
    model: Model, 
    cfg: Box,
    val_dataloader: DataLoader, 
    epoch: int = 0
):
    model.eval()
    ious = AverageMeter()
    f1_scores = AverageMeter()

    with torch.no_grad():
        for iter, data in enumerate(val_dataloader):
            images, bboxes, gt_masks = data
            num_images = images.size(0)
            pred_masks, _ = model(images, bboxes)
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
    model: Model,
    optimizer: _FabricOptimizer,
    scheduler: _FabricOptimizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
):
    """
        The SAM training loop.

        Args:
            cfg: The configuration dictionary.
            fabric: The Fabric instance.
            model: The model to train.
            optimizer: The optimizer to use.
            scheduler: The learning rate scheduler to use.
            train_dataloader: The training DataLoader.
            val_dataloader: The validation DataLoader.
    """

    # Definizione delle funzioni di perdita 
    focal_loss = FocalLoss()
    dice_loss = DiceLoss()

    for epoch in range(1, cfg.num_epochs): #aggiungere tqdm 
        # Definizione delle variabili per calcolare il tempo
        batch_time = AverageMeter()
        data_time = AverageMeter()
        focal_losses = AverageMeter()
        dice_losses = AverageMeter()
        iou_losses = AverageMeter()
        total_losses = AverageMeter()

        validated = False

        for iter, data in tqdm(enumerate(train_dataloader)): 
            if epoch > 1 and epoch % cfg.eval_interval == 0 and not validated:
                validate(fabric, model, cfg, val_dataloader, epoch)
                validated = True

            data_time.update(time.time() - end) # Aggiorna il tempo impiegato per caricare i dati
            images, bboxes, gt_masks = data # Estrae le immagini, le bounding boxes e le maschere ground truth
            batch_size = images.size(0) # Dimensione del batch
            pred_masks, iou_predictions = model(images, bboxes) # Ottiene le previsioni del modello

            print("caricato modello e predizioni") # DEBUG

            # Calcolo delle perdite per ogni maschera predetta
            num_masks = sum(len(pred_mask) for pred_mask in pred_masks)
            loss_focal = torch.tensor(0., device=fabric.device)
            loss_dice = torch.tensor(0., device=fabric.device)
            loss_iou = torch.tensor(0., device=fabric.device)
            for pred_mask, gt_mask, iou_prediction in tqdm(zip(pred_masks, gt_masks, iou_predictions)):
                batch_iou = calc_iou(pred_mask, gt_mask)
                loss_focal += focal_loss(pred_mask, gt_mask, num_masks)
                loss_dice += dice_loss(pred_mask, gt_mask, num_masks)
                loss_iou += F.mse_loss(iou_prediction, batch_iou, reduction='sum') / num_masks

            # Calcolo della perdita totale
            loss_total = 20. * loss_focal + loss_dice + loss_iou

            optimizer.zero_grad() # Azzeramento dei gradienti
            fabric.backward(loss_total) # Retropropagazione del gradiente attraverso la libreria Fabric
            optimizer.step() # Aggiornamento dei pesi del modello
            scheduler.step() # Aggiornamento del tasso di apprendimento
            batch_time.update(time.time() - end) # Aggiornamento del tempo di batch
            end = time.time()  # Aggiornamento del tempo di fine

            # Aggiornamento delle medie delle perdite
            focal_losses.update(loss_focal.item(), batch_size)
            dice_losses.update(loss_dice.item(), batch_size)
            iou_losses.update(loss_iou.item(), batch_size)
            total_losses.update(loss_total.item(), batch_size)

            # Stampa delle statistiche di addestramento
            fabric.print(f'Epoch: [{epoch}][{iter+1}/{len(train_dataloader)}]'
                         f' | Time [{batch_time.val:.3f}s ({batch_time.avg:.3f}s)]'
                         f' | Data [{data_time.val:.3f}s ({data_time.avg:.3f}s)]'
                         f' | Focal Loss [{focal_losses.val:.4f} ({focal_losses.avg:.4f})]'
                         f' | Dice Loss [{dice_losses.val:.4f} ({dice_losses.avg:.4f})]'
                         f' | IoU Loss [{iou_losses.val:.4f} ({iou_losses.avg:.4f})]'
                         f' | Total Loss [{total_losses.val:.4f} ({total_losses.avg:.4f})]')


def train_sam_CV(
    cfg: Box,
    fabric: L.Fabric,
    model: Model,
    optimizer: _FabricOptimizer,
    scheduler: _FabricOptimizer,
    dataloader: DataLoader,
):
    # Recupera i dati originali dal DataLoader
    original_data = dataloader.dataset

    # Dividi gli indici dei dati in k fold
    kf = KFold(n_splits=cfg.k_fold, shuffle=True)

    for k, train_index, val_index in tqdm(enumerate(kf.split(original_data))):
        # Estrai i dati di addestramento e validazione per questa iterazione
        train_subset = torch.utils.data.Subset(original_data, train_index)
        val_subset = torch.utils.data.Subset(original_data, val_index)

        # Crea nuovi DataLoader per addestramento e validazione
        train_loader = fabric._setup_dataloader(torch.utils.data.DataLoader(train_subset, batch_size=cfg.batch_size, shuffle=True))
        val_loader = fabric._setup_dataloader(torch.utils.data.DataLoader(val_subset, batch_size=cfg.batch_size, shuffle=False))

        # Addestra il modello utilizzando train_loader e valuta sul val_loader
        train_sam(cfg, fabric, model, optimizer, scheduler, train_loader, val_loader)


def configure_opt(cfg: Box, model: Model):

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
    # Ottiene il percorso della cartella di output
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    cfg.out_dir = os.path.join(current_directory, cfg.out_dir)

    # fabric = L.Fabric(accelerator="auto",
    #                   devices=1,
    #                   strategy="auto",
    #                   loggers=[TensorBoardLogger(cfg.out_dir, name="lightning-sam")])
    fabric = L.Fabric(accelerator="cpu",
                  strategy="auto",
                  loggers=[TensorBoardLogger(cfg.out_dir, name="lightning-sam")])
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    print(f"Using {fabric.device} devices")

    if fabric.global_rank == 0:
        os.makedirs(cfg.out_dir, exist_ok=True)

    with fabric.device:
        model = Model(cfg)
        model.setup()
        model.to(fabric.device)

    train_data, val_data = load_datasets(cfg, model.model.image_encoder.img_size)
    train_data = fabric._setup_dataloader(train_data)
    val_data = fabric._setup_dataloader(val_data)

    optimizer, scheduler = configure_opt(cfg, model)
    model, optimizer = fabric.setup(model, optimizer)

    # Train a epoche
    # train_sam(cfg, fabric, model, optimizer, scheduler, train_data, val_data)
    # validate(fabric, model, cfg, val_data, epoch=0)

    # Train con cross validation
    # train_sam_CV(cfg, fabric, model, optimizer, scheduler, train_data)
    # validate(fabric, model, cfg, val_data, epoch=0)

if __name__ == "__main__":
    main(cfg)
