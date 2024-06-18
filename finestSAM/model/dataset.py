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
from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split
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
        for ann in anns:
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
            if len(list_point_1) > self.cfg.dataset.positive_points and len(list_point_0) > self.cfg.dataset.negative_points: 
                masks.append(mask)
                boxes.append([x, y, x + w, y + h])

                temp_list_point = []
                for i in range(0, self.cfg.dataset.positive_points):
                    idx = np.random.randint(0, len(list_point_1))
                    temp_list_point.append(list_point_1[idx])
                list_point_1 = temp_list_point.copy()

                temp_list_point = []
                for i in range(0, self.cfg.dataset.negative_points):
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

def get_collate_fn(cfg: Box):
    def collate_fn(batch: List[Tuple]):
        batched_data = []

        for data in batch:
            image, original_image, original_size, point_coord, point_label, boxes, masks, resized_masks = data

            if cfg.train_type == "custom":
                if not cfg.prompts.use_boxes:
                    boxes = None
                if not cfg.prompts.use_points:
                    point_coord = None
                    point_label = None
                if not cfg.prompts.use_masks:
                    resized_masks = None

            import matplotlib.pyplot as plt
            plt.imshow(original_image)
            plt.show()

            batched_data.append({
                "image": image,
                "original_size": original_size,
                "point_coords": point_coord,
                "point_labels": point_label,
                "boxes": boxes,
                "mask_inputs": resized_masks,
            })

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
        if cfg.seed_dataloader != None:
            generator.manual_seed(cfg.seed_dataloader)

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
                                  collate_fn=get_collate_fn(cfg))

    val_dataloader = DataLoader(val_data,
                                batch_size=cfg.batch_size,
                                shuffle=False,
                                num_workers=cfg.num_workers,
                                collate_fn=get_collate_fn(cfg))

    return train_dataloader, val_dataloader