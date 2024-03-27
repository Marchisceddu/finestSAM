import os
import cv2
import numpy as np
import lightning as L
import matplotlib.pyplot as plt
from model.utils import show_anns
from lightning.fabric.loggers import TensorBoardLogger
from model.model import shape_SAM
from model.config import cfg


def pred(path):
    # Get the image
    current_directory = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_directory, path)

    image = cv2.imread(image_path)

    # Load the model 
    fabric = L.Fabric(accelerator="auto",
                    devices=cfg.num_devices,
                    strategy="auto")
    fabric.launch()
    fabric.seed_everything(cfg.seed + fabric.global_rank)

    with fabric.device:
        model = shape_SAM(cfg)
        model.setup()
        model.to(fabric.device)

    # Get the masks
    predictor = model.get_automatic_predictor(min_mask_region_area = 300)
    masks = predictor.generate(image)

    # Show the image with the masks
    plt.figure(figsize=(8,8))
    plt.imshow(image)
    show_anns(masks, opacity=1)
    plt.axis('off')
    plt.show()

    #  Save the masks

if __name__ == '__main__':
    pred('../dataset/images/0.png')