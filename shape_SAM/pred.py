import os
import cv2
import numpy as np
import lightning as L
import matplotlib.pyplot as plt
from lightning.fabric.loggers import TensorBoardLogger
from model.model import shape_SAM
from model.config import cfg


def show_anns(anns, opacity=0.35):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [opacity]])
        img[m] = color_mask
    ax.imshow(img)


def pred(path):
    # Ottiene il percorso dell'immagine
    current_directory = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_directory, path)

    image = cv2.imread(image_path)

    # Carica il modello con il salvataggio presente in cfg
    fabric = L.Fabric(accelerator="auto",
                    devices=cfg.num_devices,
                    strategy="auto")
    fabric.launch()
    fabric.seed_everything(cfg.seed + fabric.global_rank)

    with fabric.device:
        model = shape_SAM(cfg)
        model.setup()
        model.to(fabric.device)

    # Esegue la predizione di tutte le maschere
    predictor = model.get_automatic_predictor(min_mask_region_area = 300)
    masks = predictor.generate(image)

    # Visualizza l'immagine con le maschere
    plt.figure(figsize=(8,8))
    plt.imshow(image)
    show_anns(masks, opacity=1)
    plt.axis('off')
    plt.show()

    # Trasforma le maschere in shape

if __name__ == '__main__':
    pred('../dataset/images/0.png')