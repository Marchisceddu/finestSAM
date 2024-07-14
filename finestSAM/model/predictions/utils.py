import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torchvision.transforms as transforms

# Per predittore automatico
def show_anns(anns, opacity=0.35):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [opacity]])
        img[m] = color_mask
    ax.imshow(img)

# def show_anns_pil(anns, img, opacity=0.35):
#     if len(anns) == 0:
#         return None
    
#     transform = transforms.ToTensor()

#     img = transforms.ToPILImage()(img)
    
#     # Ordina le annotazioni in base all'area
#     sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    
#     # Crea un'immagine trasparente
#     height, width = sorted_anns[0]['segmentation'].shape
#     #img = Image.new("RGBA", (width, height), (255, 255, 255, 0))
#     draw = ImageDraw.Draw(img, "RGBA")
    
#     for ann in sorted_anns:
#         m = ann['segmentation']
#         color_mask = tuple((np.random.random(3) * 255).astype(int)) + (int(255 * opacity),)
        
#         # Disegna il poligono con PIL
#         for i in range(height):
#             for j in range(width):
#                 if m[i, j]:
#                     draw.point((j, i), fill=color_mask)
        
#         # Disegna il bordo colorato
#         border_color = tuple((np.random.random(3) * 255).astype(int)) + (255,)
#         for i in range(height):
#             for j in range(width):
#                 if m[i, j]:
#                     # Controlla i vicini per determinare i bordi
#                     if i > 0 and not m[i-1, j]:
#                         draw.point((j, i-1), fill=border_color)
#                     if i < height - 1 and not m[i+1, j]:
#                         draw.point((j, i+1), fill=border_color)
#                     if j > 0 and not m[i, j-1]:
#                         draw.point((j-1, i), fill=border_color)
#                     if j < width - 1 and not m[i, j+1]:
#                         draw.point((j+1, i), fill=border_color)
    
#     img.save("output.png")


# Per predittori manuali
# (AGGIUSTARE PER FARLO FUNZIONARE CON GPU)
def show_mask(mask, ax, random_color=True, seed=None):
    np.random.seed(seed)
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    #mask_image = mask.cpu().numpy().reshape(h, w, 1) * color.reshape(1, 1, -1) for predictor
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1].cpu().numpy()
    neg_points = coords[labels==0].cpu().numpy()
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   
