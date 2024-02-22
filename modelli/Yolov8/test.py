from ultralytics import YOLO
import cv2
import random
import numpy as np

img = cv2.imread('datasets/val/images/image01.png')

# Load a model
# model = YOLO('runs/segment/train3/weights/best.pt')  # load a custom model
model = YOLO('runs/segment/train3/weights/best.pt')  # load a custom model

# Predict with the model
results = model(img)  # predict on an image

colors = [random.choices(range(256), k=3) for _ in range(80)]
print(results)
i = 0
for result in results:
    for mask, box in zip(result.masks.xy, result.boxes):
        points = np.int32([mask])
        # cv2.polylines(img, points, True, (255, 0, 0), 1)
        cv2.fillPoly(img, points, colors[i])
        i += 1

cv2.imshow("Image", img)
cv2.waitKey(0)

# cv2.imwrite("", img)