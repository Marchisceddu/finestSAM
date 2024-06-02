import os
import lightning as L
from .model import FinestSAM
from box import Box
from typing import Tuple
from lightning.fabric.loggers import TensorBoardLogger


def set_model(cfg: Box, save_loggers: bool = True) -> Tuple[FinestSAM, L.Fabric]:
    """Set the device and the model."""
    loggers = [TensorBoardLogger(cfg.sav_dir, name="loggers_finestSAM")] if save_loggers else None

    fabric = L.Fabric(accelerator=cfg.device,
                      devices=cfg.num_devices,
                      strategy="auto",
                      loggers=loggers)
    fabric.launch()

    fabric.seed_everything(cfg.seed_device)

    if save_loggers and fabric.global_rank == 0: 
        os.makedirs(os.path.join(cfg.sav_dir, "loggers_finestSAM"), exist_ok=True)

    with fabric.device:
        model = FinestSAM(cfg)
        model.setup()
        model.to(fabric.device)
    
    return model, fabric