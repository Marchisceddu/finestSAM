import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw

# Carica un'immagine di esempio
image_path = "./cartella_di_output/page_1.jpg"
image = Image.open(image_path)

# Carica il modello Faster R-CNN preaddestrato
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Trasforma l'immagine in tensor
image_tensor = F.to_tensor(image).unsqueeze(0)

# Esegui l'infereza
with torch.no_grad():
    predictions = model(image_tensor)

# Estrai le bounding box, le etichette e i punteggi di confidenza
boxes =  predictions[0]['boxes']    
labels = predictions[0]['labels']
scores = predictions[0]['scores']

# Visualizza le bounding box sull'immagine
draw = ImageDraw.Draw(image)
for box, label, score in zip(boxes, labels, scores):
    box = [coord for coord in box.tolist()]
    draw.rectangle(box, outline="red", width=2)
    draw.text((box[0], box[1]), f"Class: {label.item()}, Score: {round(score.item(), 3)}", fill="red")

# Salva o visualizza l'immagine risultante
image.show()
