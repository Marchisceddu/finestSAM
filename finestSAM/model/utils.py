import os
import torch
import lightning as L
from .model import FinestSAM
from box import Box
from typing import Tuple
from lightning.fabric.loggers import TensorBoardLogger


def set_model(cfg: Box) -> Tuple[FinestSAM, L.Fabric]:
    """Set the device and the model."""
    loggers = [TensorBoardLogger(cfg.sav_dir, name="loggers_finestSAM")] if torch.is_grad_enabled() else None

    fabric = L.Fabric(accelerator=cfg.device,
                      devices=cfg.num_devices,
                      strategy="auto",
                      loggers=loggers)
    fabric.launch()

    fabric.seed_everything(cfg.seed_device)

    if torch.is_grad_enabled() and fabric.global_rank == 0: 
        os.makedirs(os.path.join(cfg.sav_dir, "loggers_finestSAM"), exist_ok=True)

    with fabric.device:
        model = FinestSAM(cfg)
        model.setup()
        model.train() if torch.is_grad_enabled() else model.eval()
        model.to(fabric.device)
    
    return model, fabric