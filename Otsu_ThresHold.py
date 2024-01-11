import cv2 as cv
import sys
import matplotlib as plt
import numpy as np

# prende l'immagine e la salva in img
img = cv.imread(cv.samples.findFile("./cartella_di_output/Immagine_originale.jpg"), cv.IMREAD_GRAYSCALE)

# se non riesce a leggere l'immagine, esce
if img is None:
    sys.exit("Could not read the image.")

blur = cv.GaussianBlur(img,(5,5),0)
ret3,otsu_test = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

# stampo l'immagine

# noise removal
kernel = np.ones((2,2),np.uint8)
opening = cv.morphologyEx(otsu_test,cv.MORPH_OPEN,kernel, iterations = 2)
                          


cv.imwrite("./cartella_di_output/Otsu.jpg", opening)
    



