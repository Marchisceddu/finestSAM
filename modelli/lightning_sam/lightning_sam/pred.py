from model import Model
from config import cfg
import lightning as L
from lightning.fabric.loggers import TensorBoardLogger
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

'''
    esempio di come potrebbe essere, documentazione utile 
    tutte le maschere -> https://colab.research.google.com/github/facebookresearch/segment-anything/blob/main/notebooks/automatic_mask_generator_example.ipynb#scrollTo=68364513
    singola maschera per punto -> https://colab.research.google.com/github/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb#scrollTo=b4a4b25c
'''

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def pred(path):
    # Ottiene il percorso dell'immagine
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    image_path = os.path.join(current_directory, path)

    image = cv2.imread(image_path)

    # Carica il modello con il salvataggio presente in cfg
    fabric = L.Fabric(accelerator="auto",
                    devices=cfg.num_devices,
                    strategy="auto",
                    loggers=[TensorBoardLogger(cfg.out_dir, name="lightning-sam")])
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    with fabric.device:
        model = Model(cfg)
        model.setup()
        model.to(fabric.device)

    # Esegue la predizione di tutte le maschere
    predictor = model.get_all_predictor()
    masks = predictor.generate(image)

    # Visualizza l'immagine con le maschere
    plt.figure(figsize=(8,8))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    pred('../../../dataset/coco/images/0.png')