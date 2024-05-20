import numpy as np
import matplotlib.pyplot as plt
import torch
from .config import cfg
from .segment_anything.utils.transforms import ResizeLongestSide


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calc_points_train(pred_mask: torch.Tensor, gt_mask: torch.Tensor, img_size: int, original_size: tuple, device: torch.device):
    transform = ResizeLongestSide(img_size)

    pred_mask = (pred_mask >= 0.5).float()
    pred_mask = pred_mask.squeeze()
    gt_mask = gt_mask.squeeze()
    
    point_coords = []
    point_labels = []

    positive_points = cfg.dataset.positive_points
    negative_points = cfg.dataset.negative_points
    foreground_coordinates_list = []
    background_coordinates_list= []

    for i, (mask, gtm) in enumerate(zip(pred_mask, gt_mask)):
         # Estrazione dei nuovi punti di foreground e background
        new_foreground_points = (mask == 0.) & (gtm == 1.)
        new_background_points = (mask == 1.) & (gtm == 0.)
        
        # Individua le coordinate dei nuovi punti di foreground e background
        foreground_coordinates_list.append(torch.nonzero(new_foreground_points, as_tuple=True))
        background_coordinates_list.append(torch.nonzero(new_background_points, as_tuple=True))

        if len(foreground_coordinates_list[i][0]) < positive_points:
            positive_points = len(foreground_coordinates_list[i][0]) 
        if len(background_coordinates_list[i][0]) < negative_points:
            negative_points = len(background_coordinates_list[i][0])
    for mask, gtm, foreground_coordinates, background_coordinates in zip(pred_mask, gt_mask, foreground_coordinates_list, background_coordinates_list):

        if not (positive_points == 0):            
            temp_list_point = []
            for i in range(0, positive_points):
                idx = np.random.randint(0, len(foreground_coordinates[0]))
                temp_list_point.append([foreground_coordinates[1][idx].item(), foreground_coordinates[0][idx].item()])
            new_foreground_points = temp_list_point.copy()
            list_label_1 = [1] * len(new_foreground_points)
        else:
            new_foreground_points = []
            list_label_1 = []
        
        if not (negative_points == 0):
            temp_list_point = []
            for i in range(0, negative_points):
                idx = np.random.randint(0, len(background_coordinates[0]))
                temp_list_point.append([background_coordinates[1][idx].item(), background_coordinates[0][idx].item()])
            new_background_points = temp_list_point.copy()
            list_label_0 = [0] * len(new_background_points)
        else:
            new_background_points = []
            list_label_0 = []
        
        actual_point_coords = new_foreground_points + new_background_points
        actual_point_labels = list_label_1 + list_label_0
        
        point_coords.append(actual_point_coords)
        point_labels.append(actual_point_labels)

    # Normalizzo i punti
    if positive_points + negative_points > 0:
      point_coords = transform.apply_coords(np.array(point_coords), original_size)
      point_coords = torch.tensor(np.stack(point_coords, axis=0)).to(device)

      point_labels = torch.as_tensor(point_labels, dtype=torch.int).to(device)
    else:
      point_coords = None
      point_labels = None

    return  point_coords, point_labels

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


# Per predittori manuali+
# (AGGIUSTARE PER FARLO FUNZIONARE CON GPU)
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.cpu().numpy().reshape(h, w, 1) * color.reshape(1, 1, -1)
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
