import os
import torch
import lightning as L
from box import Box
from finestSAM.core.model import FinestSAM
from finestSAM.data.dataset import load_test_dataset
from finestSAM.utils import validate


def call_test(cfg: Box, dataset_path: str, checkpoint_path: str = None, model_type: str = None):
    """
    Evaluate the model on a test dataset.
    
    Args:
        cfg (Box): The configuration file.
        dataset_path (str): The path to the test dataset.
        checkpoint_path (str, optional): Path to the checkpoint file. 
                                         If None, uses the one in cfg.
        model_type (str, optional): The type of the model (vit_b, vit_l, vit_h).
    """
    
    if checkpoint_path:
        cfg.model.checkpoint = checkpoint_path
    
    if model_type:
        cfg.model.type = model_type
        
    print(f"Loading checkpoint from: {cfg.model.checkpoint}")
    print(f"Model type: {cfg.model.type}")

    fabric = L.Fabric(accelerator=cfg.device,
                      devices=cfg.num_devices,
                      strategy="auto")
    
    fabric.launch(test, cfg, dataset_path)


def test(fabric: L.Fabric, *args, **kwargs):
    """
    Evaluate the model on a test dataset.
    
    Args:
        fabric (L.Fabric): The lightning fabric.
        *args: The positional arguments:
            [0] - cfg (Box): The configuration file.
            [1] - dataset_path (str): The path to the test dataset.
        **kwargs: The keyword arguments:
            not used.
    """
    cfg = args[0]
    dataset_path = args[1]
    
    fabric.seed_everything(cfg.seed_device)

    with fabric.device:
        model = FinestSAM(cfg)
        model.setup()
        model.eval()
        model.to(fabric.device)

    img_size = model.model.image_encoder.img_size
    test_dataloader = load_test_dataset(cfg, img_size, dataset_path, fabric=fabric)
    test_dataloader = fabric._setup_dataloader(test_dataloader)

    fabric.print(f"Starting testing on dataset: {dataset_path}")
    
    mean_iou, mean_dsc = validate(fabric, cfg, model, test_dataloader, epoch=0)

    fabric.print("\nTest Results:")
    fabric.print(f"Mean IoU: {mean_iou:.4f}")
    fabric.print(f"Mean DSC: {mean_dsc:.4f}")
