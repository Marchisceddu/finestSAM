import os
import cv2
import tqdm
import torch
import random
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
from scipy.spatial import Voronoi
from typing import Tuple, List
from box import Box
from pycocotools.coco import COCO
from .segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split
)

import matplotlib.pyplot as plt # ELIMNINRE
import matplotlib.patches as patches # ELIMINARE


class COCODataset(Dataset):

    def __init__(
            self, 
            root_dir: str, 
            annotation_file: str, 
            cfg: Box,
            transform: transforms.Compose = None, 
            seed: int = None
        ):
        """
        Args:
            root_dir (str): The root directory of the images.
            annotation_file (str): The path to the annotation file.
            cfg (Box): The configuration file.
            transform (transforms.Compose): The transformation to apply to the data.
            seed (int): The seed for the random number generator.
        """
        self.cfg = cfg
        self.seed = seed
        self.root_dir = root_dir
        self.transform = transform
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())

        # Filter out image_ids without any annotations
        self.image_ids = [image_id for image_id in self.image_ids if len(self.coco.getAnnIds(imgIds=image_id)) > 0]

        # Data for __getitem__
        self.points_1 = []
        self.points_0 = []
        self.masks = []
        self.boxes = []
        self.ann_valid = []
        if self.cfg.dataset.use_center:
            self.centroids = []
        
        # Calcola i dati principali per ogni immagine
        bar = tqdm.tqdm(total = len(self.image_ids), desc = "Uploading dataset...", leave=False)
        for image_id in self.image_ids:
            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            anns = self.coco.loadAnns(ann_ids)

            centroids = []
            masks = []
            ann_valid = []
            points_0 = []
            points_1 = []

            for ann in anns:
                # Get the bounding box
                x, y, w, h = ann['bbox']

                # Get the mask
                mask = self.coco.annToMask(ann)
                masks.append(mask)
                
                # Get the points for the mask
                roi = mask[y:y + h, x:x + w] # Remove if you don't want the negative points only within the box.
                list_points_1, list_points_0 = ([(px + x, py + y) for py, px in zip(*np.where(roi == v))] for v in [1, 0])
                # list_points_1 = [(px + x, py + y) for py, px in zip(*np.where(roi == 1))]
                # list_points_0 = [(px + x, py + y) for py, px in zip(*np.where(roi == 0))]
                
                points_1.append(list_points_1)
                points_0.append(list_points_0)

                is_valid = False
                if self.cfg.dataset.use_center: center_of_mass = None
                n_pos, n_neg = (self.cfg.dataset.positive_points, self.cfg.dataset.negative_points)

                """
                During the conversion of the resolution of the mask, some details can be lost, 
                and the annootation becomes less accurate and too small to be used.
                So, we need to filter out those annotations and keep only the ones that at least
                have the points that are needed for the training.
                """
                if len(list_points_1) >= n_pos and len(list_points_0) >= n_neg: 
                    is_valid = True
                    if n_pos > 0 and self.cfg.dataset.use_center:
                        # implemente Centroidal Voronoi Tessellation (CVT)
                        # Questo approccio sfrutta i concetti di centroidi Voronoi per determinare il punto più centrale all'interno della maschera binaria.
                        try:
                            vor = Voronoi(list_points_1) # Calcola i diagrammi di Voronoi utilizzando i punti forniti. I punti di Voronoi sono i centroidi dei poligoni di Voronoi, che corrispondono ai punti più centrali rispetto ai punti di campionamento.
                            center_of_mass = vor.points[np.argsort(np.linalg.norm(vor.points - vor.points.mean(axis=0), axis=1))][0] # Trova il punto più centrale, che corrisponde al generatore del diagramma Voronoi
                        except Exception as e:
                            print(e)
                            is_valid = False
                
                ann_valid.append(is_valid)
                if self.cfg.dataset.use_center: centroids.append(center_of_mass)   
        
            # Append the data for the image
            self.points_1.append(points_1)
            self.points_0.append(points_0)
            self.ann_valid.append(ann_valid)
            self.masks.append(masks)
            if self.cfg.dataset.use_center: self.centroids.append(centroids)

            bar.update(1)

    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        Args:
            idx (int): The index of the image to get.
        Returns:
            Tuple: 
                The image, 
                the original image,
                the original size of the image, 
                the point coordinates, 
                the point labels, 
                the boxes, 
                the masks, 
                the resized masks, 
        """
        # Set the seed for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Restor the image from the folder
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.root_dir, image_info['file_name'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image = image.copy()

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
        for i, ann in enumerate(anns):
            # Get the bounding box
            x, y, w, h = ann['bbox']

            # Add random noise to each coordinate with standard deviation equal to 10% of the box sidelength, to a maximum of 20 pixels
            x = max(0, int(x + np.random.normal(0, 0.1 * w)))
            y = max(0, int(y + np.random.normal(0, 0.1 * h)))
            w = min(W - x, int(w + np.random.normal(0, 0.1 * w)))
            h = min(H - y, int(h + np.random.normal(0, 0.1 * h)))

            # Check if the new box is contained in the image 
            x = max(0, x)
            y = max(0, y)
            w = min(w, W - x)
            h = min(h, H - y)
            # if x + w > W:
            #     w = W - x
            # if y + h > H:
            #     h = H - y
            # if x < 0:
            #     x = 0
            # if y < 0:
            #     y = 0

            fig, ax = plt.subplots()
            ax.imshow(original_image)
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.axis('off')
            plt.show()

            # Get the masks
            mask = self.masks[idx][i].copy()

            points_1 = self.points_1[idx][i].copy()
            points_0 = self.points_0[idx][i].copy()

            n_pos, n_neg = (self.cfg.dataset.positive_points, self.cfg.dataset.negative_points)
            
            if self.ann_valid[idx][i]: 
                masks.append(mask)
                boxes.append([x, y, x + w, y + h])
                
                if n_pos > 0 and self.cfg.dataset.use_center:
                    center_of_mass = self.centroids[idx][i].copy()
                    n_pos = n_pos-1 if n_pos > 1 else 0
                
                points_1, points_0 = (random.sample(points, n_points) for points, n_points in zip([points_1, points_0], [n_pos, n_neg]))
                if 'center_of_mass' in locals(): points_1.append(center_of_mass)

                label_1, label_0 = ([v] * len(points) for points, v in zip([points_1, points_0], [1, 0]))
                #label_0 = [0] * len(points_0)
                print(label_1, label_0)

                point_coords.append(points_1 + points_0)
                point_labels.append(label_1 + label_0)

                # fig, ax = plt.subplots()
                # ax.imshow(original_image)
                # for p in list_point_1:
                #     circle = patches.Circle(p, radius=10, color='g')
                #     ax.add_patch(circle)
                # plt.axis('off')
                # plt.show()
    
        if self.transform:
            image, resized_masks, boxes, point_coords = self.transform(image, masks, np.array(boxes), np.array(point_coords))

        # Convert the data to tensor
        boxes = torch.tensor(np.stack(boxes, axis=0))
        masks = torch.tensor(np.stack(masks, axis=0)).float()
        resized_masks = torch.tensor(np.stack(resized_masks, axis=0)).float()
        point_coords = torch.tensor(np.stack(point_coords, axis=0))
        point_labels = torch.as_tensor(point_labels, dtype=torch.int)

        # Add channel dimension to the masks for compatibility with the model
        resized_masks = resized_masks.unsqueeze(1)
        
        return image, original_image, original_size, point_coords, point_labels, boxes, masks, resized_masks
    

class ResizeAndPad:

    def __init__(self, target_size: int):
        self.target_size = target_size
        self.transform = ResizeLongestSide(target_size)
        self.to_tensor = transforms.ToTensor()

    def __call__(
            self, 
            image: np.ndarray, 
            masks: List[np.ndarray], 
            boxes: np.ndarray, 
            point_coords: np.ndarray
        ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor]:
        # Resize image and masks
        og_h, og_w, _ = image.shape
        image = self.transform.apply_image(image)
        masks = [torch.tensor(self.transform.apply_image(mask)) for mask in masks]        
        image = self.to_tensor(image)

        # Resize masks to 1/4th resolution of the image
        resized_masks = []
        for mask in masks:
            mask = F.max_pool2d(mask.unsqueeze(0).unsqueeze(0).float(), kernel_size=4, stride=4).squeeze()
            resized_masks.append(mask)

        # Adjust bounding boxes and point coordinates
        boxes = self.transform.apply_boxes(boxes, (og_h, og_w))
        point_coords = self.transform.apply_coords(point_coords, (og_h, og_w))

        return image, resized_masks, boxes, point_coords


def get_collate_fn(cfg: Box, type):
    def collate_fn(batch: List[Tuple]):
        batched_data = []

        for data in batch:
            image, original_image, original_size, point_coord, point_label, boxes, masks, resized_masks = data

            data = {
                "image": image,
                "original_size": original_size,
                "gt_masks": masks,
            }

            if cfg.prompts.use_boxes:
                data["boxes"] = boxes
            if cfg.prompts.use_points:
                data["point_coords"] = point_coord
                data["point_labels"] = point_label
            if cfg.prompts.use_masks:
                data["mask_inputs"] = resized_masks

            if type == "val":
                data["original_image"] = original_image

            batched_data.append(data)

        return batched_data
    
    return collate_fn


def load_dataset(cfg: Box, img_size: int) -> Tuple[DataLoader, DataLoader]:
    """
    Load the dataset and return the dataloaders for training and validation.
    Args:
        cfg (Box): The configuration file.
        img_size (int): The size of the image to resize to.
    Returns:
        Tuple[DataLoader, DataLoader]: The training and validation dataloaders.
    """
    # Set up the transformation for the dataset
    transform = ResizeAndPad(img_size)

    # Load the dataset
    main_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if cfg.dataset.auto_split:
        data_path = os.path.join(main_directory, cfg.dataset.path.root_dir)
        annotations_path = os.path.join(main_directory, cfg.dataset.path.annotation_file)

        data = COCODataset(root_dir=data_path,
                        annotation_file=annotations_path,
                        cfg=cfg,
                        transform=transform,
                        seed=cfg.seed_dataloader)
        
         # Calc the size of the validation set
        total_size = len(data)
        val_size = int(total_size * cfg.dataset.val_size)

        # Set the seed 
        generator = torch.Generator()
        if cfg.dataset.seed_split != None:
            generator.manual_seed(cfg.dataset.seed_split)

        # Split the dataset into training and validation
        train_data, val_data = random_split(data, [total_size - val_size, val_size], generator=generator)
    else:
        train_path = os.path.join(main_directory, cfg.dataset.train.root_dir)
        val_path =  os.path.join(main_directory, cfg.dataset.val.root_dir)
        train_annotations_path = os.path.join(main_directory, cfg.dataset.train.annotation_file)
        val_annotations_path = os.path.join(main_directory, cfg.dataset.train.annotation_file)

        train_data = COCODataset(root_dir=train_path,
                        annotation_file=train_annotations_path,
                        cfg=cfg,
                        transform=transform,
                        seed=cfg.seed_dataloader)
    
        val_data = COCODataset(root_dir=val_path,
                        annotation_file=val_annotations_path,
                        cfg=cfg,
                        transform=transform,
                        seed=cfg.seed_dataloader)

    train_dataloader = DataLoader(train_data,
                                  batch_size=cfg.batch_size,
                                  shuffle=True,
                                  num_workers=cfg.num_workers,
                                  collate_fn=get_collate_fn(cfg, "train"))

    val_dataloader = DataLoader(val_data,
                                batch_size=cfg.batch_size,
                                shuffle=False,
                                num_workers=cfg.num_workers,
                                collate_fn=get_collate_fn(cfg, "val"))

    return train_dataloader, val_dataloader