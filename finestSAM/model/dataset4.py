import os
import cv2
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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tqdm
from .predictions.utils import (
    show_mask,
)

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

        self.lists_point_1 = []
        self.lists_point_0 = []
        self.masks = []
        self.isValid = []
        if self.cfg.dataset.use_center:
            self.centroids = []
        
        bar = tqdm.tqdm(total = len(self.image_ids), desc = "Uploading dataset...", leave=False)
        for image_id in self.image_ids:
            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            anns = self.coco.loadAnns(ann_ids)

            centroids = []
            masks = []
            isValid = []
            lists_point_0 = []
            lists_point_1 = []

            # Get box, point and mask for any annotations
            for l, ann in enumerate(anns):
                # Get the bounding box
                x, y, w, h = ann['bbox']

                # Get the mask
                mask = self.coco.annToMask(ann)
                masks.append(mask)
                
                # Get the points for the mask
                roi = mask[y:y + h, x:x + w] # Remove if you don't want the negative points only within the box.
                list_point_1 = [(px + x, py + y) for py, px in zip(*np.where(roi == 1))]
                list_point_0 = [(px + x, py + y) for py, px in zip(*np.where(roi == 0))]

                lists_point_1.append(list_point_1)
                lists_point_0.append(list_point_0)

                n_pos = self.cfg.dataset.positive_points
                n_neg = self.cfg.dataset.negative_points

                """
                During the conversion of the resolution of the mask, some details can be lost, 
                and the annootation becomes less accurate and too small to be used.
                So, we need to filter out those annotations and keep only the ones that at least
                have the points that are needed for the training.
                """
                if len(list_point_1) >= n_pos and len(list_point_0) >= n_neg: 

                    if n_pos > 0 and self.cfg.dataset.use_center:
                        # implemente Centroidal Voronoi Tessellation (CVT)
                        # Questo approccio sfrutta i concetti di centroidi Voronoi per determinare il punto più centrale all'interno della maschera binaria.
                        try:
                            vor = Voronoi(list_point_1) # Calcola i diagrammi di Voronoi utilizzando i punti forniti. I punti di Voronoi sono i centroidi dei poligoni di Voronoi, che corrispondono ai punti più centrali rispetto ai punti di campionamento.
                            center_of_mass = vor.points[np.argsort(np.linalg.norm(vor.points - vor.points.mean(axis=0), axis=1))][0] # Trova il punto più centrale, che corrisponde al generatore del diagramma Voronoi
                            centroids.append(center_of_mass)

                            isValid.append(True)
                        except Exception as e:
                            centroids.append(np.array([0., 0.])) # NON LI USA, AUMENTA SOLO L'INDICE
                            isValid.append(False)
                    else:
                        isValid.append(True)
                else:
                    if self.cfg.dataset.use_center: centroids.append(np.array([0., 0.]))
                    isValid.append(False)
            
            self.lists_point_1.append(lists_point_1)
            self.lists_point_0.append(lists_point_0)
            self.isValid.append(isValid)
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
            if x + w > W:
                w = W - x
            if y + h > H:
                h = H - y
            if x < 0:
                x = 0
            if y < 0:
                y = 0

            # Get the masks
            mask = self.masks[idx][i].copy()

            list_point_1 = self.lists_point_1[idx][i].copy()
            list_point_0 = self.lists_point_0[idx][i].copy()

            n_pos = self.cfg.dataset.positive_points
            n_neg = self.cfg.dataset.negative_points

            """
            During the conversion of the resolution of the mask, some details can be lost, 
            and the annootation becomes less accurate and too small to be used.
            So, we need to filter out those annotations and keep only the ones that at least
            have the points that are needed for the training.  
            """
            if self.isValid[idx][i]: 
                masks.append(mask)
                boxes.append([x, y, x + w, y + h])
                
                if n_pos > 0 and self.cfg.dataset.use_center:
                    try:
                        center_of_mass = self.centroids[idx][i].copy()
                        n_pos = n_pos-1 if n_pos > 1 else 0
                    except Exception as e:
                        print(e)

                list_point_1 = random.sample(list_point_1, n_pos)
                if 'center_of_mass' in locals(): list_point_1.append(center_of_mass)
                list_point_0 = random.sample(list_point_0, n_neg)

                # fig, ax = plt.subplots()
                # ax.imshow(original_image)
                # for p in list_point_1:
                #    circle = patches.Circle(p, radius=10, color='g')
                #    ax.add_patch(circle)
                # plt.axis('off')
                # plt.show()

                list_label_0 = [0] * len(list_point_0)
                list_label_1 = [1] * len(list_point_1)

                point_coords.append(list_point_1 + list_point_0)
                point_labels.append(list_label_1 + list_label_0)

        # plt.imshow(original_image)
        # for i, mask in enumerate(masks):
        #     show_mask(mask, plt.gca(), seed=i)
        # plt.axis('off')
        # plt.savefig(f"ooooo/{idx}.png")
    
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
    # Set the seed 
    generator = torch.Generator()
    if cfg.dataset.seed_split != None:
        generator.manual_seed(cfg.dataset.seed_split)

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
                                  generator=generator,
                                  num_workers=cfg.num_workers,
                                  collate_fn=get_collate_fn(cfg, "train"))

    val_dataloader = DataLoader(val_data,
                                batch_size=cfg.batch_size,
                                shuffle=False,
                                generator=generator,
                                num_workers=cfg.num_workers,
                                collate_fn=get_collate_fn(cfg, "val"))

    return train_dataloader, val_dataloader