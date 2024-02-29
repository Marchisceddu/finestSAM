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
    https://colab.research.google.com/github/facebookresearch/segment-anything/blob/main/notebooks/automatic_mask_generator_example.ipynb#scrollTo=68364513
    https://colab.research.google.com/github/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb#scrollTo=b4a4b25c
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

def prova_pred(path):
    # Ottieni il percorso assoluto del file che si sta eseguendo
    current_file_path = os.path.abspath(__file__)

    # Ottieni il percorso della directory in cui si trova il file che si sta eseguendo
    current_directory = os.path.dirname(current_file_path)

    # Costruisci il percorso completo del file che desideri cercare
    image_path = os.path.join(current_directory, path)

    image = cv2.imread(image_path)

    # Carica il modello con il salvataggio presente in cfg
    fabric = L.Fabric(accelerator="auto",
                    devices="auto",
                    strategy="auto",
                    loggers=[TensorBoardLogger(cfg.out_dir, name="lightning-sam")])
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    with fabric.device:
        model = Model(cfg)
        model.setup()

    # Esegue la predizione di tutte le maschere
    predictor = model.get_all_predictor()
    masks = predictor.generate(image.to(fabric.device))

    plt.figure(figsize=(8,8))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.show()

prova_pred('../../../dataset/coco/images/0.png')