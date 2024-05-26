import os
import time
import torch
import numpy as np
import lightning as L
import torch.nn.functional as F
from .segment_anything.utils.transforms import ResizeLongestSide
from .model import shape_SAM
from .utils import AverageMeter
from .losses import (
    Calc_iou,
    DiceLoss,
    FocalLoss
)
from typing import Dict, List, Tuple
from box import Box
from lightning.fabric.fabric import _FabricOptimizer
from torch.utils.data import DataLoader


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
    cfg: Box,
    epoch: int = 0, # andrà eliminato l'assegnamento a 0
    f1_scores: AverageMeter = AverageMeter(), # andrà eliminato l'assegnamento ad AverageMeter() (vedere se lasciare f1_score comme metrica)
):
    fabric.print(f"Saving checkpoint to {cfg.out_dir}")
    state_dict = model.model.state_dict()
    if fabric.global_rank == 0:
        torch.save(state_dict, os.path.join(cfg.out_dir, f"epoch-{epoch:06d}-f1{f1_scores.avg:.2f}-ckpt.pth"))
    model.train()


class Train_custom():

    def __init__(self,
                cfg: Box,
                fabric: L.Fabric,
                model: shape_SAM,
                optimizer: _FabricOptimizer,
                scheduler: _FabricOptimizer,
                train_dataloader: DataLoader,
                val_dataloader: DataLoader,
        ):
        super().__init__()
        self.cfg = cfg
        self.fabric = fabric
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()
        self.calc_iou = Calc_iou()

    def forward(self):
        """The SAM training loop."""
        
        for epoch in range(1, self.cfg.num_epochs + 1):
            batch_time = AverageMeter()
            data_time = AverageMeter()
            focal_losses = AverageMeter()
            dice_losses = AverageMeter()
            iou_losses = AverageMeter()
            total_losses = AverageMeter()
            end = time.time()

            for iter, batched_data in enumerate(self.train_dataloader):
                data_time.update(time.time() - end)

                outputs = self.model(batched_input=batched_data, multimask_output=True)

                loss_total, loss_focal, loss_dice, loss_iou, best_score = self.calc_losses(outputs, batched_data, loss_focal, loss_dice, loss_iou)

                self.backward(loss_total)

                # Measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # Update the average loss
                focal_losses.update(loss_focal.item(), self.cfg.batch_size)
                dice_losses.update(loss_dice.item(), self.cfg.batch_size)
                iou_losses.update(loss_iou.item(), self.cfg.batch_size)
                total_losses.update(loss_total.item(), self.cfg.batch_size)

                # Log the training information
                self.iteration_results(epoch, iter, batch_time, data_time, focal_losses, dice_losses, iou_losses, total_losses, best_score)

            if (epoch > 1 and self.cfg.eval_interval > 0 and epoch % self.cfg.eval_interval == 0) or (epoch == self.cfg.num_epochs):
                #validate(fabric, model, val_dataloader, epoch)
                save(self.fabric, self.model, self.cfg, epoch)

    def calc_losses(self, 
                    outputs: List[Dict[str, torch.Tensor]], 
                    batched_data: Dict[str, torch.Tensor],
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:

        batched_pred_masks = []
        batched_iou_predictions = []
        batched_logits = []
        iou_means = int

        for item in outputs:
            # Take mask, iou_prediction and low_res_logits from the output
            batched_pred_masks.append(item["masks"])
            batched_iou_predictions.append(item["iou_predictions"])
            batched_logits.append(item["low_res_logits"])
            
        num_masks = sum(len(pred_mask) for pred_mask in batched_pred_masks)

        loss_focal = torch.tensor(0., device=self.fabric.device)
        loss_dice = torch.tensor(0., device=self.fabric.device)
        loss_iou = torch.tensor(0., device=self.fabric.device)
        
        for data, pred_masks, iou_predictions, logits in zip(batched_data, batched_pred_masks, batched_iou_predictions, batched_logits):

            separated_masks = torch.unbind(pred_masks, dim=1) # 3 output masks
            separated_scores = torch.unbind(iou_predictions, dim=1) # scores for each mask

            iou_prediction_means = [torch.mean(score) for score in separated_scores]
            best_index = torch.argmax(torch.tensor(iou_prediction_means))

            pred_masks = separated_masks[best_index]
            iou_predictions = separated_scores[best_index]

            iou_means = torch.mean(torch.stack([torch.mean(iou_predictions), iou_means])) # mean of the iou predictions

            batch_iou = self.calc_iou(pred_masks, data["gt_masks"])
            loss_focal += self.focal_loss(pred_masks, data["gt_masks"], num_masks)
            loss_dice += self.dice_loss(pred_masks, data["gt_masks"], num_masks)
            loss_iou += F.mse_loss(iou_predictions, batch_iou, reduction='mean')

        loss_total = self.cfg.opt.focal_alpgha * loss_focal + loss_dice + loss_iou

        return loss_total, loss_focal, loss_dice, loss_iou, iou_means
    
    def backward(self, loss: torch.Tensor):
        self.optimizer.zero_grad()
        self.fabric.backward(loss)
        self.optimizer.step()
        self.scheduler.step()

    def iteration_results(self,
                        epoch: int,
                        iter: int,
                        batch_time: AverageMeter,
                        data_time: AverageMeter,
                        focal_losses: AverageMeter,
                        dice_losses: AverageMeter,
                        iou_losses: AverageMeter,
                        total_losses: AverageMeter,
                        best_score: torch.Tensor,
        ):
        self.fabric.print(f'Epoch: [{epoch}][{iter+1}/{len(self.train_dataloader)}]'
                                    f' | Time [{batch_time.val:.3f}s ({batch_time.avg:.3f}s)]'
                                    f' | Data [{data_time.val:.3f}s ({data_time.avg:.3f}s)]'
                                    f' | a Focal Loss [{self.cfg.opt.focal_alpgha * focal_losses.val:.4f} ({self.cfg.opt.focal_alpgha * focal_losses.avg:.4f})]'
                                    f' | Dice Loss [{dice_losses.val:.4f} ({dice_losses.avg:.4f})]'
                                    f' | IoU Loss [{iou_losses.val:.4f} ({iou_losses.avg:.4f})]'
                                    f' | Total Loss [{total_losses.val:.4f} ({total_losses.avg:.4f})]'
                                    f' | Mask quality [({best_score:.4f})]')
        steps = epoch * len(self.train_dataloader) + iter
        log_info = {
            'Loss': total_losses.val,
            'alpha focal loss': self.cfg.opt.focal_alpgha * focal_losses.val,
            'dice loss': dice_losses.val,
        }
        self.fabric.log_dict(log_info, step=steps)


class Train_11_iterations(Train_custom):
    
    def __init__(self,
                cfg: Box,
                fabric: L.Fabric,
                model: shape_SAM,
                optimizer: _FabricOptimizer,
                scheduler: _FabricOptimizer,
                train_dataloader: DataLoader,
                val_dataloader: DataLoader,
        ):
        super().__init__(cfg, fabric, model, optimizer, scheduler, train_dataloader, val_dataloader)
        self.new_logits = []
        self.new_point_coords = []
        self.new_point_labels = []

    def forward(self):
        """
        The SAM training loop.
        
        This method is used to train the model for 11 iterations (paper p.17 point:Training algorithm).
        """

        for epoch in range(1, self.cfg.num_epochs + 1):
            batch_time = AverageMeter()
            data_time = AverageMeter()
            focal_losses = AverageMeter()
            dice_losses = AverageMeter()
            iou_losses = AverageMeter()
            total_losses = AverageMeter()
            end = time.time()

            random_iteration = np.random.randint(2, 11)

            for iteration in range(1, 12): # Fa 11 iterazioni per ogni epoca come scritto nel paper

                are_logits = True if iteration > 1 else False
                only_logits = True if iteration == random_iteration or iteration == 11 else False

                for iter, batched_data in enumerate(self.train_dataloader):
                    data_time.update(time.time() - end)

                    # After the first iteration, the model will receive the low_res_logits as input
                    if iteration > 1:
                        for sample, logits, point_coords, point_labels in zip(batched_data, self.new_logits, self.new_point_coords, self.new_point_labels):
                            sample["mask_inputs"] = logits.clone().detach().unsqueeze(1)

                            if only_logits:
                                sample["point_coords"] = None
                                sample["point_labels"] = None
                            else:
                                if sample["point_coords"] is None:
                                    save(self.fabric, self.model, self.cfg, epoch)

                                sample["point_coords"] = point_coords
                                sample["point_labels"] = point_labels
                        
                        # Reset the lists
                        self.new_point_coords = []
                        self.new_point_labels = []

                    outputs = self.model(batched_input=batched_data, multimask_output=True, are_logits=are_logits)

                    loss_total, loss_focal, loss_dice, loss_iou, best_score = self.calc_losses(outputs, batched_data)

                    self.backward(loss_total)

                    # Measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()

                    # Update the average loss
                    focal_losses.update(loss_focal.item(), self.cfg.batch_size)
                    dice_losses.update(loss_dice.item(), self.cfg.batch_size)
                    iou_losses.update(loss_iou.item(), self.cfg.batch_size)
                    total_losses.update(loss_total.item(), self.cfg.batch_size)

                    # Log the training information
                    self.iteration_results(epoch, iter, batch_time, data_time, focal_losses, dice_losses, iou_losses, total_losses, best_score)

    def calc_losses(self, 
                    outputs: List[Dict[str, torch.Tensor]], 
                    batched_data: Dict[str, torch.Tensor],
        ) -> Tuple[torch.Tensor | int]:

        batched_pred_masks = []
        batched_iou_predictions = []
        batched_logits = []
        iou_means = int

        for item in outputs:
            # Take mask, iou_prediction and low_res_logits from the output
            batched_pred_masks.append(item["masks"])
            batched_iou_predictions.append(item["iou_predictions"])
            batched_logits.append(item["low_res_logits"])
            
        num_masks = sum(len(pred_mask) for pred_mask in batched_pred_masks)

        loss_focal = torch.tensor(0., device=self.fabric.device)
        loss_dice = torch.tensor(0., device=self.fabric.device)
        loss_iou = torch.tensor(0., device=self.fabric.device)
        
        for data, pred_masks, iou_predictions, logits in zip(batched_data, batched_pred_masks, batched_iou_predictions, batched_logits):

            separated_masks = torch.unbind(pred_masks, dim=1) # 3 output masks
            separated_scores = torch.unbind(iou_prediction, dim=1) # scores for each mask
            separated_logits = torch.unbind(logits, dim=1)

            iou_prediction_means = [torch.mean(score) for score in separated_scores]
            best_index = torch.argmax(torch.tensor(iou_prediction_means))

            pred_masks = separated_masks[best_index]
            iou_prediction = separated_scores[best_index]
            self.new_logits.append(separated_logits[best_index])

            iou_means = torch.mean(torch.stack([torch.mean(iou_predictions), iou_means])) # mean of the iou predictions

            p, l = self.calc_points_train(pred_masks,  data["gt_masks"], self.model.model.image_encoder.img_size, data["original_size"], self.fabric.device) # opt
            self.new_point_coords.append(p)
            self.new_point_labels.append(l)

            batch_iou = self.calc_iou(pred_masks, data["gt_masks"])
            loss_focal += self.focal_loss(pred_masks, data["gt_masks"], num_masks)
            loss_dice += self.dice_loss(pred_masks, data["gt_masks"], num_masks)
            loss_iou += F.mse_loss(iou_prediction, batch_iou, reduction='mean')

        loss_total = self.cfg.opt.focal_alpgha * loss_focal + loss_dice + loss_iou

        return loss_total, loss_focal, loss_dice, loss_iou, iou_means
    
    def calc_points_train(self, 
                          pred_mask: torch.Tensor, 
                          gt_mask: torch.Tensor, 
                          img_size: int,
                          original_size: tuple, 
                          device: torch.device
        ) -> Tuple[torch.Tensor, torch.Tensor]: # boh questa funzione mi fa schifo vorrei migliorarla
        transform = ResizeLongestSide(img_size)

        pred_mask = (pred_mask >= 0.5).float()
        pred_mask = pred_mask.squeeze()
        gt_mask = gt_mask.squeeze()
        
        point_coords = []
        point_labels = []

        positive_points = self.cfg.dataset.positive_points
        negative_points = self.cfg.dataset.negative_points
        foreground_coordinates_list = []
        background_coordinates_list= []

        for i, (mask, gtm) in enumerate(zip(pred_mask, gt_mask)):
            # Estrazione dei nuovi punti di foreground e background
            new_foreground_points = (mask == 0.) & (gtm == 1.)
            new_background_points = (mask == 1.) & (gtm == 0.)
            
            # Individua le coordinate dei nuovi punti di foreground e background
            foreground_coordinates_list.append(torch.nonzero(new_foreground_points, as_tuple=True))
            background_coordinates_list.append(torch.nonzero(new_background_points, as_tuple=True))

            if len(foreground_coordinates_list[i][0]) < positive_points:
                positive_points = len(foreground_coordinates_list[i][0]) 
            if len(background_coordinates_list[i][0]) < negative_points:
                negative_points = len(background_coordinates_list[i][0])
        for mask, gtm, foreground_coordinates, background_coordinates in zip(pred_mask, gt_mask, foreground_coordinates_list, background_coordinates_list):

            if not (positive_points == 0):            
                temp_list_point = []
                for i in range(0, positive_points):
                    idx = np.random.randint(0, len(foreground_coordinates[0]))
                    temp_list_point.append([foreground_coordinates[1][idx].item(), foreground_coordinates[0][idx].item()])
                new_foreground_points = temp_list_point.copy()
                list_label_1 = [1] * len(new_foreground_points)
            else:
                new_foreground_points = []
                list_label_1 = []
            
            if not (negative_points == 0):
                temp_list_point = []
                for i in range(0, negative_points):
                    idx = np.random.randint(0, len(background_coordinates[0]))
                    temp_list_point.append([background_coordinates[1][idx].item(), background_coordinates[0][idx].item()])
                new_background_points = temp_list_point.copy()
                list_label_0 = [0] * len(new_background_points)
            else:
                new_background_points = []
                list_label_0 = []
            
            actual_point_coords = new_foreground_points + new_background_points
            actual_point_labels = list_label_1 + list_label_0
            
            point_coords.append(actual_point_coords)
            point_labels.append(actual_point_labels)

        # Normalizzo i punti
        if positive_points + negative_points > 0:
            point_coords = transform.apply_coords(np.array(point_coords), original_size)
            point_coords = torch.tensor(np.stack(point_coords, axis=0)).to(device)

            point_labels = torch.as_tensor(point_labels, dtype=torch.int).to(device)
        else:
            point_coords = None
            point_labels = None

        return  point_coords, point_labels
