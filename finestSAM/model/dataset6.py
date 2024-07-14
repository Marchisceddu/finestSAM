import os
import cv2
import tqdm
import torch
import random
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from box import Box
from typing import Tuple, List
from pycocotools.coco import COCO
from scipy.spatial import Voronoi
from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split
)
from .segment_anything.utils.transforms import ResizeLongestSide
from .segment_anything.utils.amg import build_point_grid

import matplotlib.pyplot as plt # ELIMNINRE
import matplotlib.patches as patches # ELIMINARE


class COCODataset(Dataset):

    def __init__(
            self, 
            images_dir: str, 
            annotation_file: str, 
            cfg: Box,
            transform: transforms.Compose = None, 
            seed: int = None,
            sav: str = None,
        ):
        """
        Args:
            images_dir (str): The root directory of the images.
            annotation_file (str): The path to the annotation file.
            cfg (Box): The configuration file.
            transform (transforms.Compose): The transformation to apply to the data.
            seed (int): The seed for the random number generator.
        """
        self.cfg = cfg
        self.seed = seed
        self.images_dir = images_dir
        self.transform = transform
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())

        # Filter out image_ids without any annotations
        self.image_ids = [image_id for image_id in self.image_ids if len(self.coco.getAnnIds(imgIds=image_id)) > 0]

        # Data for __getitem__
        self.points_1 = []
        self.points_0 = []
        self.masks = []
        if sav:
            dati = torch.load(sav)
            self.ann_valid = dati['ann_valid']
            if self.cfg.dataset.use_center: self.centroids = dati['centroids']
        else:         
            self.ann_valid = []
            if self.cfg.dataset.use_center: self.centroids = []

        # Calcola i dati principali per ogni immagine
        bar = tqdm.tqdm(total = len(self.image_ids), desc = "Uploading dataset...", leave=False)
        for image_id in self.image_ids:
            image_info = self.coco.loadImgs(image_id)[0]
            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            anns = self.coco.loadAnns(ann_ids)

            H, W = (image_info['height'], image_info['width'])
            automatic_grid = build_point_grid(32) * np.array((H, W))[None, ::-1]

            masks = []
            points_0 = []
            points_1 = []
            if not sav:
                centroids = []
                ann_valid = []

            for ann in anns:
                # Get the bounding box
                x, y, w, h = ann['bbox']

                # Get the mask
                mask = self.coco.annToMask(ann)
                masks.append(mask)
                
                # Get the points for the mask
                roi = mask[y:y + h, x:x + w] # Remove if you don't want the negative points only within the box.
                list_points_1, list_points_0 = ([(px + x, py + y) for py, px in zip(*np.where(roi == v))] for v in [1, 0])
                
                points_1.append(list_points_1)
                points_0.append(list_points_0)

                if not sav:
                    center_point = None
                    n_pos, n_neg = (self.cfg.dataset.positive_points, self.cfg.dataset.negative_points)
                    is_valid = len(list_points_1) >= n_pos and len(list_points_0) >= n_neg

                    """
                    During the conversion of the resolution of the mask, some details can be lost, 
                    and the annootation becomes less accurate and too small to be used.
                    So, we need to filter out those annotations and keep only the ones that at least
                    have the points that are needed for the training.
                    """
                    if is_valid and n_pos > 0 and self.cfg.dataset.use_center: 
                            points = np.array(list_points_1)
                            center_index = np.argsort(np.linalg.norm(np.array(list_points_1) - points.mean(axis=0), axis=1))[0]
                            center_point = points[center_index]

                            # AGGIUNGERE UN CONFIG PER SCEGLIERE SE ALLINEARE O MENO IL CENTRO DI MASSA ALLA GRIGLIA
                            distances = np.linalg.norm(automatic_grid - center_point, axis=1)
                            nearest_point_index = np.argmin(distances)
                            center_point = automatic_grid[nearest_point_index]
                    
                    ann_valid.append(is_valid)
                    if self.cfg.dataset.use_center: centroids.append(center_point)
        
            # Append the data for the image
            self.points_1.append(points_1)
            self.points_0.append(points_0)
            self.masks.append(masks)
            if not sav:
                self.ann_valid.append(ann_valid)
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
        image_path = os.path.join(self.images_dir, image_info['file_name'])
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
            '''
            #calcola il rumore massimo apllicabile, in modo tale che non superi i 20 px
            max_w_noise = int(min(0.1 * w, 20))
            max_h_noise = int(min(0.1 * h, 20))

            # decide il rumore che dev'essere applicato a x e y
            noise_x = random.randint(-max_w_noise, max_w_noise)
            noise_y = random.randint(-max_w_noise, max_h_noise)

            # applica a x e y il rumore e controlla che restino interni all'immagine
            new_x = max(0, x + noise_x)
            if new_x >= W: new_x = x
            new_y = max(0, y + noise_y)
            if new_y >= H: new_y = y

            # decide il rumore che dev'essere applicato a w e h (PER APPLICARLO BISOGNA METTERE UN ULTERIORE CONTROLLO PER FARE IN MODO CHE w E h NON POSSANO ESSRE MINORI DI 0)
            # noise_x = random.randint(-max_w_noise, max_w_noise)
            # noise_y = random.randint(-max_h_noise, max_h_noise)

            # applica il rumore a w e h e controlla che la box finale resti interna all'immagine
            new_w = min(W - new_x, w + noise_x)
            new_h = min(H - new_y, h + noise_y)

            if new_w > W:
              new_w = W - new_x
            if new_h > H:
              new_h = H - new_y

            x = new_x
            y = new_y
            h = new_h - 1 
            w = new_w - 1 

            # CODICE VECCHIO NON FUNZIONA
            x = max(0, int(x + np.random.normal(0, 0.1 * w)))
            y = max(0, int(y + np.random.normal(0, 0.1 * h)))
            w = min(W - x, int(w + np.random.normal(0, 0.1 * w)))
            h = min(H - y, int(h + np.random.normal(0, 0.1 * h)))

            #Check if the new box is contained in the image 
            if x + w > W:
                w = W - x
            if y + h > H:
                h = H - y
            if x < 0:
                x = 0
            if y < 0:
                y = 0

            fig, ax = plt.subplots()
            ax.imshow(original_image)
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.axis('off')
            plt.show()
            '''
        
            # Get the masks
            mask = self.masks[idx][i].copy()

            points_1 = self.points_1[idx][i].copy()
            points_0 = self.points_0[idx][i].copy()

            n_pos, n_neg = (self.cfg.dataset.positive_points, self.cfg.dataset.negative_points)
            
            if self.ann_valid[idx][i]: 
                masks.append(mask)
                boxes.append([x, y, x + w, y + h])
                
                if n_pos > 0 and self.cfg.dataset.use_center:
                    center_point = self.centroids[idx][i].copy()
                    n_pos = n_pos-1 if n_pos > 1 else 0
                
                points_1, points_0 = (random.sample(points, n_points) for points, n_points in zip([points_1, points_0], [n_pos, n_neg]))
                if 'center_point' in locals(): 
                    points_1.append(center_point)

                label_1, label_0 = ([v] * len(points) for points, v in zip([points_1, points_0], [1, 0]))

                point_coords.append(points_1 + points_0)
                point_labels.append(label_1 + label_0)

                # fig, ax = plt.subplots()
                # ax.imshow(original_image)
                # for p in points_1:
                #     circle = patches.Circle(p, radius=5, color='g')
                #     ax.add_patch(circle)
                # for p in automatic_grid:
                #     circle = patches.Circle(p, radius=1, color='r')
                #     ax.add_patch(circle)
                # plt.axis('off')
                # plt.show()

                # import time
                # time.sleep(10)
    
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
    

class ResizeData:

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
        #image = self.to_tensor(image)
        image = torch.as_tensor(image)
        image = image.permute(2, 0, 1).contiguous()

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
    # Set the seed 
    generator = torch.Generator()
    if cfg.dataset.seed != None:
        generator.manual_seed(cfg.dataset.seed)

    # Set up the transformation for the dataset
    transform = ResizeData(img_size)

    # Load the dataset
    main_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if cfg.dataset.auto_split:
        data_root_path = os.path.join(main_directory, cfg.dataset.split_path.root_dir)
        data_path = os.path.join(data_root_path, cfg.dataset.split_path.images_dir)
        annotations_path = os.path.join(data_root_path, cfg.dataset.split_path.annotation_file)
        sav_path = os.path.join(data_root_path, cfg.dataset.split_path.sav)

        data = COCODataset(images_dir=data_path,
                        annotation_file=annotations_path,
                        cfg=cfg,
                        transform=transform,
                        seed=cfg.seed_dataloader,
                        sav=(sav_path if os.path.exists(sav_path) else None))
        
        if not os.path.exists(sav_path):
            save_data = {
                'ann_valid': data.ann_valid,
                'centroids': data.centroids if hasattr(data, 'centroids') else None
            }
            torch.save(save_data, sav_path)
            print("Dataset saved")
        
        # Calc the size of the validation set
        total_size = len(data)
        val_size = int(total_size * cfg.dataset.split_path.val_size)

        # Split the dataset into training and validation
        train_data, val_data = random_split(data, [total_size - val_size, val_size], generator=generator)
    else:
        train_root_path = os.path.join(main_directory, cfg.dataset.no_split_path.train.root_dir)
        train_path = os.path.join(train_root_path, cfg.dataset.no_split_path.train.images_dir)
        train_annotations_path = os.path.join(train_root_path, cfg.dataset.no_split_path.train.annotation_file)
        train_sav_path = os.path.join(train_root_path, cfg.dataset.no_split_path.train.sav)

        val_root_path = os.path.join(main_directory, cfg.dataset.no_split_path.val.root_dir)    
        val_path =  os.path.join(val_root_path, cfg.dataset.no_split_path.val.images_dir)
        val_annotations_path = os.path.join(val_root_path, cfg.dataset.no_split_path.val.annotation_file)
        val_sav_path = os.path.join(val_root_path, cfg.dataset.no_split_path.val.sav)


        train_data = COCODataset(images_dir=train_path,
                        annotation_file=train_annotations_path,
                        cfg=cfg,
                        transform=transform,
                        seed=cfg.seed_dataloader,
                        sav=(train_sav_path if os.path.exists(train_sav_path) else None))
    
        val_data = COCODataset(images_dir=val_path,
                        annotation_file=val_annotations_path,
                        cfg=cfg,
                        transform=transform,
                        seed=cfg.seed_dataloader,
                        sav=(val_sav_path if os.path.exists(val_sav_path) else None))
        
        if not os.path.exists(train_sav_path) or not os.path.exists(val_sav_path):
            train_save_data = {
                'ann_valid': train_data.ann_valid,
                'centroids': train_data.centroids if hasattr(train_data, 'centroids') else None
            }
            val_save_data = {
                'ann_valid': val_data.ann_valid,
                'centroids': val_data.centroids if hasattr(val_data, 'centroids') else None
            }
            torch.save(train_save_data, train_sav_path)
            torch.save(val_save_data, val_sav_path)
            print("Dataset saved")

    train_dataloader = DataLoader(train_data,
                                  batch_size=cfg.batch_size,
                                  shuffle=True,
                                  generator=generator,
                                  num_workers=cfg.num_workers,
                                  collate_fn=get_collate_fn(cfg, "val"))

    val_dataloader = DataLoader(val_data,
                                batch_size=cfg.batch_size,
                                shuffle=False,
                                num_workers=cfg.num_workers,
                                collate_fn=get_collate_fn(cfg, "val"))

    return train_dataloader, val_dataloader