import cv2
import numpy as np

# Carica l'immagine PNG
image = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

# Applica una sogliatura per creare una maschera binaria
_, binary_mask = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

# Salvare la maschera binaria
cv2.imwrite('binary_mask.png', binary_mask)
