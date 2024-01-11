import cv2 as cv
import sys
import numpy as np
import matplotlib as plt
import math

# prende l'immagine e la salva in img
img = cv.imread(cv.samples.findFile("./cartella_di_output/Immagine_originale.jpg"), cv.IMREAD_GRAYSCALE)

# se non riesce a leggere l'immagine, esce
if img is None:
    sys.exit("Could not read the image.")

#blurriamo l'immagine
blur = cv.GaussianBlur(img,(5,5),0)

#creiamo il treshold con Otsu
ret3,otsu_test = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

#prendiamo i contorni con canny
edges = cv.Canny(otsu_test,100,200)

# stampo l'immagine
cv.imwrite("./cartella_di_output/Otsu+Canny1.jpg", opening)
    



