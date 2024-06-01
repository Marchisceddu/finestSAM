import os
import lightning as L
from .model import shape_SAM
from box import Box
from typing import Tuple
from lightning.fabric.loggers import TensorBoardLogger


def set_model(cfg: Box, save_loggers: True) -> Tuple[shape_SAM, L.Fabric]:
    """Set the device and the model."""
    loggers = [TensorBoardLogger(cfg.out_dir, name="loggers_shape_SAM")] if save_loggers else None

    fabric = L.Fabric(accelerator=cfg.device,
                      devices=cfg.num_devices,
                      strategy="auto",
                      loggers=loggers)
    fabric.launch()

    fabric.seed_everything(cfg.seed_device)

    if save_loggers and fabric.global_rank == 0:
        os.makedirs(cfg.out_dir, exist_ok=True)

    with fabric.device:
        model = shape_SAM(cfg)
        model.setup()
        model.to(fabric.device)
    
    return model, fabric