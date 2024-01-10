import cv2 as cv
import sys

# prende l'immagine e la salva in img
img = cv.imread(cv.samples.findFile("./cartella_di_output/page_1.jpg"), cv.IMREAD_GRAYSCALE)

# se non riesce a leggere l'immagine, esce
if img is None:
    sys.exit("Could not read the image.")

# trasforma l'immgaine in scala di grigi
#imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# disegna i bordi dell'immmagine
# ret, thresh = cv.threshold(imgray, 127, 255, 0)
# contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
# cv.drawContours(img, contours, -1, (0, 255, 0), 3)

# trasforma l'immagine in binario
# threshold_value = 150
# max_value = 255
# _, binary_image = cv.threshold(imgray, threshold_value, max_value, cv.THRESH_BINARY)

gaussian_test=cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)

# stampo l'immagine
cv.imshow("Image", gaussian_test)
cv.waitKey(0)