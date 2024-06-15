import os
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
from typing import Tuple, List
from box import Box
from pycocotools.coco import COCO
from .segment_anything.utils.transforms import ResizeLongestSide
from .segment_anything.utils.amg import build_point_grid
from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split
)
from config import cfg_train as cfg


class COCODataset(Dataset):

    def __init__(
            self, 
            root_dir: str, 
            annotation_file: str, 
            transform: transforms.Compose = None, 
            seed: int = None
        ):
        self.seed = seed
        self.root_dir = root_dir
        self.transform = transform
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())

        # Filter out image_ids without any annotations
        self.image_ids = [image_id for image_id in self.image_ids if len(self.coco.getAnnIds(imgIds=image_id)) > 0]

    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Tuple[int, int], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        '''
        Si può predirre con la griglia mi sono studiato l'automatic, bisogna creare una griglia generica di 32x32 con valori delle coordinate che vadano da 0 a 1 (build_point_grid amg file)
        e poi sclare questa griglia alla dimensione dell'immagine (riga 238 automatic), 
        trovare i punti positivi di una maschera,
        applcare l'algoritmo della distanza per assegnare questo punto alla distanza più vicino alla griglia auto,
        normalizzare i punti come si faceva di già
        dovrebbe essere fatto anche se le immagini non sono quasdrate perchè la griglia non si forma sul quadrato ma sulla dimensione originale dell'immagine e poi viene scalata,
        lo fa anche il predittore in automatico
        '''
        points = build_point_grid(32)
        points_scale = np.array((H, W))[None, ::-1]
        points_for_image = points * points_scale

        # Get box, point and mask for any annotations
        for ixx, ann in enumerate(anns):
            # Get the bounding box
            x, y, w, h = ann['bbox']

            # Add random noise to each coordinate with standard deviation equal to 10% of the box sidelength, to a maximum of 20 pixels
            x = max(0, int(x + np.random.normal(0, 0.1 * w)))
            y = max(0, int(y + np.random.normal(0, 0.1 * h)))
            w = min(W - x, int(w + np.random.normal(0, 0.1 * w)))
            h = min(H - y, int(h + np.random.normal(0, 0.1 * h)))

            # check if the new box is contained in the image 
            if x + w > W:
                w = W - x
            if y + h > H:
                h = H - y
            if x < 0:
                x = 0
            if y < 0:
                y = 0

            # Get the mask
            mask = self.coco.annToMask(ann)
            
            # Get the points for the mask
            list_point_0 = []
            list_point_1 = []
            for j in range(y, y + h):
                for i in range(x, x + w):
                    if i >= 0 and i < len(mask[0]) and j >= 0 and j < len(mask):
                        if mask[j][i]:
                            list_point_1.append([i, j])
                        else:
                            list_point_0.append([i, j])

            """
            During the conversion of the resolution of the mask, some details can be lost, 
            and the annootation becomes less accurate and too small to be used.
            So, we need to filter out those annotations and keep only the ones that at least
            have the points that are needed for the training.  
            """
            if len(list_point_1) >= cfg.dataset.positive_points and len(list_point_0) >= cfg.dataset.negative_points: 
                masks.append(mask)
                boxes.append([x, y, x + w, y + h])

                if False: # Mettere il controllo in config, se entra qui scegli un punto random altrimenti usa la griglia di punti
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
                else:
                    # Assegna un punto random di list_point_1 al più vicino presente a points_for_image
                    idx = np.random.randint(0, len(list_point_1))
                    distances = np.linalg.norm(points_for_image - list_point_1[idx], axis=1)
                    nearest_point_index = np.argmin(distances)
                    nearest_point = points_for_image[nearest_point_index]
                    
                    list_point_1 = [nearest_point]
                    list_point_0 = []

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

        # Add channel dimension to the masks for compatibility with the model
        resized_masks = resized_masks.unsqueeze(1)
        
        return image, original_size, point_coords, point_labels, boxes, masks, resized_masks, imo
    

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


def collate_fn(batch: List[Tuple[torch.Tensor, Tuple[int, int], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    batched_data = []

    for data in batch:
        image, original_size, point_coord, point_label, boxes, masks, resized_masks, imo = data

        if cfg.train_type == "custom":
            if not cfg.custom_cfg.use_boxes:
                boxes = None
            if not cfg.custom_cfg.use_points:
                point_coord = None
                point_label = None
            if not cfg.custom_cfg.use_masks:
                resized_masks = None

        batched_data.append({
            "image": image,
            "original_size": original_size,
            "point_coords": point_coord,
            "point_labels": point_label,
            "boxes": boxes,
            "mask_inputs": resized_masks,
            "gt_masks": masks,
            "imo": imo,
        })

    return batched_data


def load_dataset(cfg: Box, img_size: int) -> Tuple[DataLoader, DataLoader]:
    # Set up the transformation for the dataset
    transform = ResizeAndPad(img_size)

    # Load the dataset
    main_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(main_directory, cfg.dataset.root_dir)
    annotations_path = os.path.join(main_directory, cfg.dataset.annotation_file)

    data = COCODataset(root_dir=dataset_path,
                        annotation_file=annotations_path,
                        transform=transform,
                        seed=cfg.seed_dataloader)
    
    # Calc the size of the validation set
    total_size = len(data)
    val_size = int(total_size * cfg.dataset.val_size)

    # Split the dataset into training and validation
    train_data, val_data = random_split(data, [total_size - val_size, val_size])

    train_dataloader = DataLoader(train_data,
                                  batch_size=cfg.batch_size,
                                  shuffle=True,
                                  num_workers=cfg.num_workers,
                                  collate_fn=collate_fn)

    val_dataloader = DataLoader(val_data,
                                batch_size=cfg.batch_size,
                                shuffle=False,
                                num_workers=cfg.num_workers,
                                collate_fn=collate_fn)

    return train_dataloader, val_dataloader