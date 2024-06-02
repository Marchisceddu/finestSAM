import os
import lightning as L
from .train_custom import train_custom
from .train_11_iterations import train_11_iterations
from .utils import configure_opt
from ..dataset import load_dataset
from ..utils import set_model
from box import Box


def train(cfg: Box):
    """Main training function."""

    # Set up the output directory
    main_directory = os.path.dirname(os.path.abspath(__file__)).rsplit('/', 2)[0]
    cfg.out_dir = os.path.join(main_directory, cfg.out_dir)

    # Set up the model and device
    model, fabric = set_model(cfg)

    # Load the dataset
    train_data, val_data = load_dataset(cfg, model.model.image_encoder.img_size)
    train_data = fabric._setup_dataloader(train_data)
    val_data = fabric._setup_dataloader(val_data)

    # Configure the optimizer and scheduler
    optimizer, scheduler = configure_opt(cfg, model)
    model, optimizer = fabric.setup(model, optimizer)

    # Train the model based on the training type
    if cfg.train_type == "custom":
        train_custom(cfg, fabric, model, optimizer, scheduler, train_data, val_data)
    elif cfg.train_type == "11-iteration":
        train_11_iterations(cfg, fabric, model, optimizer, scheduler, train_data, val_data)
    else:
        raise ValueError(f"Unknown training type: {cfg.train_type}")