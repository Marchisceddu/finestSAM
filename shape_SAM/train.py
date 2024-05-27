import os
import torch
import lightning as L
from model.config import cfg
from model.dataset import load_datasets
from model.model import shape_SAM
from model.train import (
    train_custom, 
    train_11_iterations,
    configure_opt
)
from box import Box
from lightning.fabric.loggers import TensorBoardLogger

torch.autograd.set_detect_anomaly(True)
torch.set_float32_matmul_precision('high')


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

    if cfg.train_type == "custom":
        train_custom(cfg, fabric, model, optimizer, scheduler, train_data, val_data)
    elif cfg.train_type == "11-iteration":
        train_11_iterations(cfg, fabric, model, optimizer, scheduler, train_data, val_data)
    else:
        raise ValueError(f"Unknown training type: {cfg.train_type}")
    
    #validate(fabric, cfg, model, train_data, epoch=0)


if __name__ == "__main__":
    main(cfg)
