import os
import cv2
import torch
import numpy as np
import lightning as L
import matplotlib.pyplot as plt
from model.utils import show_anns
from model.model import shape_SAM
from model.config import cfg
from model.dataset import COCODataset

# SISTEMARE IN UTILS
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


def pred_auto(path):
    # Get the image
    current_directory = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_directory, path)

    image = cv2.imread(image_path)

    # Load the model 
    fabric = L.Fabric(accelerator="auto",
                    devices=cfg.num_devices,
                    strategy="auto")
    fabric.launch()
    fabric.seed_everything(cfg.seed_device)

    with fabric.device:
        model = shape_SAM(cfg)
        model.setup()
        model.to(fabric.device)

    # Get the masks
    predictor = model.get_automatic_predictor(min_mask_region_area = 300)
    masks = predictor.generate(image)

    # Show the image with the masks
    plt.figure(figsize=(8,8))
    plt.imshow(image)
    show_anns(masks, opacity=1)
    plt.axis('off')
    plt.show()

    #  Save the masks


def pred_boxes():
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Load the model 
    fabric = L.Fabric(accelerator="auto",
                    devices=cfg.num_devices,
                    strategy="auto")
    fabric.launch()
    fabric.seed_everything(cfg.seed_device)

    with fabric.device:
        model = shape_SAM(cfg)
        model.setup()
        model.to(fabric.device)

    # Get the dataset for boxes
    dataset_path = os.path.join(current_directory, cfg.dataset.root_dir)
    annotations_path = os.path.join(current_directory, cfg.dataset.annotation_file)

    dataset = COCODataset(root_dir=dataset_path,
                          annotation_file=annotations_path,
                          transform=None)

    # Get the predictor
    predictor = model.get_predictor()

    for image_id in dataset.image_ids:
        image_info = dataset.coco.loadImgs(image_id)[0]
        image_path = os.path.join(dataset.root_dir, image_info['file_name'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_ids = dataset.coco.getAnnIds(imgIds=image_id)
        anns = dataset.coco.loadAnns(ann_ids)
        bboxes = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            bboxes.append([x, y, x + w, y + h])
        bboxes = torch.as_tensor(bboxes, device=model.model.device)
        transformed_boxes = predictor.transform.apply_boxes_torch(bboxes, image.shape[:2])

        predictor.set_image(image)
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        ) 

        # Show the image with the masks
        for i, mask in enumerate(masks):
            plt.figure(figsize=(10,10))
            plt.imshow(image)
            show_mask(mask, plt.gca())
            #show_points(point_coords, point_labels, plt.gca())
            #plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.show() 

        #  Save the masks


def pred_points():
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Load the model 
    fabric = L.Fabric(accelerator="auto",
                    devices=cfg.num_devices,
                    strategy="auto")
    fabric.launch()
    fabric.seed_everything(cfg.seed_device)

    with fabric.device:
        model = shape_SAM(cfg)
        model.setup()
        model.to(fabric.device)

    # Get the dataset for boxes
    dataset_path = os.path.join(current_directory, cfg.dataset.root_dir)
    annotations_path = os.path.join(current_directory, cfg.dataset.annotation_file)

    dataset = COCODataset(root_dir=dataset_path,
                          annotation_file=annotations_path,
                          transform=None)

    # Get the predictor
    predictor = model.get_predictor()

    for image_id in dataset.image_ids:
        image_info = dataset.coco.loadImgs(image_id)[0]
        image_path = os.path.join(dataset.root_dir, image_info['file_name'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_ids = dataset.coco.getAnnIds(imgIds=image_id)
        anns = dataset.coco.loadAnns(ann_ids)
        point_coords = []
        point_labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            mask = dataset.coco.annToMask(ann)
            
            # Get the points
            list_point_0 = []
            list_point_1 = []
            for j in range(y, y + h):
                for i in range(x, x + w):
                    if i >= 0 and i < len(mask[0]) and j >= 0 and j < len(mask):
                        if mask[j][i]:
                            list_point_1.append([i, j])
                        else:
                            list_point_0.append([i, j])

            temp_list_point = []
            for i in range(0, cfg.dataset.positive_points):
                idx = np.random.randint(0, len(list_point_1))
                temp_list_point.append(list_point_1[idx])
            list_point_1 = temp_list_point.copy()

            temp_list_point = []
            for i in range(0, cfg.dataset.negative_points):
                idx = np.random.randint(0, len(list_point_0))
                temp_list_point.append(list_point_0[idx])
            list_point_0 = temp_list_point.copy()

            list_label_0 = [0] * len(list_point_0)
            list_label_1 = [1] * len(list_point_1)

            point_coords.append(list_point_1 + list_point_0)
            point_labels.append(list_label_1 + list_label_0)

        point_coords = torch.as_tensor(point_coords, device=model.model.device)
        point_coords = predictor.transform.apply_coords_torch(point_coords, image.shape[:2])
        point_labels = torch.as_tensor(point_labels, dtype=torch.int, device=model.model.device)
        

        predictor.set_image(image)
        masks, scores, _ = predictor.predict_torch(
            point_coords=point_coords,
            point_labels=point_labels,
            boxes=None,
            multimask_output=False,
        ) 

        # Show the image with the masks
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10,10))
            plt.imshow(image)
            show_mask(mask, plt.gca())
            #show_points(point_coords, point_labels, plt.gca())
            #plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.show()  

        #  Save the masks


if __name__ == '__main__':
    #pred_auto('../dataset/images/0.png')
    pred_points()