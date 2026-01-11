import os
import cv2
import torch
import lightning as L
import matplotlib.pyplot as plt
from box import Box
from finestSAM.utils import (
    show_anns,
)
from finestSAM.model.model import FinestSAM


def call_predict(cfg: Box, input_path: str, opacity: float = 0.9, checkpoint_path: str = None, model_type: str = None):
    """
    Perform automatic predictions on an input image.
    
    Args:
        cfg (Box): The configuration object.
        input_path (str): The path to the input image.
        opacity (float): The opacity of the mask.
        checkpoint_path (str, optional): Path to the checkpoint file.
        model_type (str, optional): The type of the model (vit_b, vit_l, vit_h).
    """

    if checkpoint_path:
        cfg.model.checkpoint = checkpoint_path
    
    if model_type:
        cfg.model.type = model_type

    print(f"Loading checkpoint from: {cfg.model.checkpoint}")
    print(f"Model type: {cfg.model.type}")

    if opacity is not None:
        cfg.opacity = opacity

    # Get the image path
    image_path = input_path

    # Get the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load the model 
    with torch.no_grad():
        fabric = L.Fabric(accelerator=cfg.device, #  "tpu" is not supported in this version
                      devices=1,
                      strategy="auto")
        
        fabric.seed_everything(cfg.seed_device)

        with fabric.device:
            model = FinestSAM(cfg)
            model.setup()
            model.eval()
            model.to(fabric.device)

        # Predict the masks
        predictor = model.get_automatic_predictor()
        masks = predictor.generate(image)
           
    # Create the output directory if it does not exist
    os.makedirs(cfg.out_dir, exist_ok=True)

    # Save the predictions as a .png file
    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=100)  # 6.4 inches * 100 dpi = 640 pixels
    ax.set_position([0, 0, 1, 1])  # [left, bottom, width, height]
    plt.imshow(image)
    show_anns(masks, opacity=opacity)
    plt.axis('off')
    plt.savefig(os.path.join(cfg.out_dir, "output.png"), dpi=100, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.clf()

    fabric.print("Predictions saved in:", os.path.join(cfg.out_dir, "output.png"))