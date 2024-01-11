import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('cartella_di_output/Immagine_originale.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
edges = cv.Canny(img,100,200)

cv.imwrite("./cartella_di_output/Canny.jpg", edges)