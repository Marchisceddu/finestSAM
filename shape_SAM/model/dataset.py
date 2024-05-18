import os
import cv2
import torch
import random
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
from pycocotools.coco import COCO
from .segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from .config import cfg

import matplotlib.pyplot as plt

class COCODataset(Dataset):

    def __init__(self, root_dir, annotation_file, transform=None, seed=None):
        self.seed = seed
        self.root_dir = root_dir
        self.transform = transform
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())

        # Filter out image_ids without any annotations
        self.image_ids = [image_id for image_id in self.image_ids if len(self.coco.getAnnIds(imgIds=image_id)) > 0]

    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        # Set the seed for reproducibility
        np.random.seed(self.seed)

        # Restor the image from the folder
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.root_dir, image_info['file_name'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imo = image.copy()

        # Get original size of the image
        H, W, _ = image.shape
        original_size = (H, W)

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        boxes = []
        point_coords = []
        point_labels = []
        masks = []

        # Get box, point and mask for any annotations
        for ann in anns:
            # Get the bounding box
            x, y, w, h = ann['bbox']
            # Add random noise to each coordinate with standard deviation equal to 10% of the box sidelength, to a maximum of 20 pixels
            x = max(0, int(x + np.random.normal(0, 0.1 * w)))
            y = max(0, int(y + np.random.normal(0, 0.1 * h)))
            w = min(W - x, int(w + np.random.normal(0, 0.1 * w)))
            h = min(H - y, int(h + np.random.normal(0, 0.1 * h)))

            boxes.append([x, y, x + w, y + h])

            # Get the mask
            mask = self.coco.annToMask(ann)
            masks.append(mask)
            
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
    
        if self.transform:
            image, resized_masks, boxes, point_coords = self.transform(image, masks, np.array(boxes), np.array(point_coords))

        # Convert the data to tensor
        boxes = torch.tensor(np.stack(boxes, axis=0))
        masks = torch.tensor(np.stack(masks, axis=0)).float()
        resized_masks = torch.tensor(np.stack(resized_masks, axis=0)).float()
        point_coords = torch.tensor(np.stack(point_coords, axis=0))
        point_labels = torch.as_tensor(point_labels, dtype=torch.int)

        resized_masks = resized_masks.unsqueeze(1)
        
        return image, original_size, point_coords, point_labels, boxes, masks, resized_masks, imo
    

def collate_fn(batch):
    batched_data = []

    for data in batch:
        image, original_size, point_coord, point_label, boxes, masks, resized_masks, imo = data

        if not cfg.use_boxes:
            boxes = None
        if not cfg.use_points:
            point_coord = None
            point_label = None
        if not cfg.use_masks:
            resized_masks = None

        batched_data.append({
            "image": image,
            "original_size": original_size,
            "point_coords": point_coord,
            "point_labels": point_label,
            "boxes": boxes,
            "mask_inputs": resized_masks,
            "gt_masks": masks,
            "imo": imo
        })

    return batched_data


class ResizeAndPad:

    def __init__(self, target_size):
        self.target_size = target_size
        self.transform = ResizeLongestSide(target_size)
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image, masks, boxes, point_coords):
        # Resize image and masks
        og_h, og_w, _ = image.shape
        image = self.transform.apply_image(image)
        masks = [torch.tensor(self.transform.apply_image(mask)) for mask in masks]        
        image = self.to_tensor(image)

        # Resize masks to 1/4th resolution of the image 
        resized_masks = []
        for mask in masks:
            # CAPITRE SE METTENDO FLOAT POI SI DEVE RICONVERTIRE IN BYTE?
            mask = F.max_pool2d(mask.unsqueeze(0).unsqueeze(0).float(), kernel_size=4, stride=4).squeeze()
            mask = self.preprocess_masks(mask)
            resized_masks.append(mask)

        # Adjust bounding boxes and point coordinates
        boxes = self.transform.apply_boxes(boxes, (og_h, og_w))
        point_coords = self.transform.apply_coords(point_coords, (og_h, og_w))

        return image, resized_masks, boxes, point_coords
    
    def preprocess_masks(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Pad
        h, w = x.shape[-2:]
        padh = 1024//4 - h
        padw = 1024//4 - w
        x = F.pad(x, (0, padw, 0, padh))
        return x


def load_datasets(cfg, img_size):
    transform = ResizeAndPad(img_size)

    # Ottiene il percorso del dataset
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    dataset_path = os.path.join(current_directory, cfg.dataset.root_dir)
    annotations_path = os.path.join(current_directory, cfg.dataset.annotation_file)

    train = COCODataset(root_dir=dataset_path,
                        annotation_file=annotations_path,
                        transform=transform,
                        seed=cfg.seed_dataloader)
    
    train_dataloader = DataLoader(train,
                                  batch_size=cfg.batch_size,
                                  shuffle=True,
                                  num_workers=cfg.num_workers,
                                  collate_fn=collate_fn)
    
    # CREARE FUNZIONE PER PRENDERE IL 10/20% DEL DATASET PER LA VALIDAZIONE 
    # val = COCODataset(root_dir=cfg.dataset.root_dir,
    #                   annotation_file=cfg.dataset.annotation_file,
    #                   transform=transform)
    # val_dataloader = DataLoader(val,
    #                             batch_size=cfg.batch_size,
    #                             shuffle=True,
    #                             num_workers=cfg.num_workers,
    #                             collate_fn=collate_fn)
    val_dataloader = DataLoader # per ora restituisce una roba vuota

    return train_dataloader, val_dataloader