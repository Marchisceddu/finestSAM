import os
import matplotlib.pyplot as plt
from torchvision.datasets import CocoDetection
from torchvision import transforms
from PIL import ImageDraw


def display_COCO(images_path, annotation_path):
    # Definisci la trasformazione per le immagini
    transform = transforms.ToTensor()

    # Carica i dataset di addestramento e di test COCO
    coco_dataset = CocoDetection(root = images_path, annFile = annotation_path, transform = transform)

    num_images_to_display = len(os.listdir(images_path))
    for i in range(num_images_to_display):
        # Carica l'immagine e le sue etichette dal set di addestramento
        image, targets = coco_dataset[i]

        # Converti l'immagine da tensore a formato immagine PIL
        image_pil = transforms.ToPILImage()(image)

        # Prepara l'oggetto per disegnare sopra l'immagine
        draw = ImageDraw.Draw(image_pil)

        # Aggiungi le etichette al grafico
        for target in targets:
            # Le annotazioni nel formato COCO contengono le coordinate dei bordi del segmento
            segmentation = target['segmentation']
            for seg in segmentation:
                # Converti le coordinate in una lista di tuple (x, y)
                points = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]
                # Disegna i bordi del segmento sull'immagine
                draw.line(points, fill='red', width=2)

        # Visualizza l'immagine con le etichette
        plt.figure()
        plt.imshow(image_pil)
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    # esempio di utilizzo
    display_COCO("../../dataset/images/", "../../dataset/annotations.json") # il percorso deve partire dalla cartella dove si trova il file .py