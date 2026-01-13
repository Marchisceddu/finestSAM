import os
import cv2
import tqdm
import torch
import random
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from box import Box
from typing import Tuple, List, Optional, TypedDict
from pycocotools.coco import COCO
from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split
)
from finestSAM.model.segment_anything.utils.transforms import ResizeLongestSide
from finestSAM.model.segment_anything.utils.amg import build_point_grid


class ValidAnn(TypedDict):
    ann_id: int
    bbox: List[int]  # [x, y, w, h]
    center_point: Optional[np.ndarray]


class Sample(TypedDict):
    image_id: int
    file_name: str
    valid_anns: List[ValidAnn]


class COCODataset(Dataset):
    """
    Given a COCO dataset, this class loads the images and annotations, 
    and builds the dataset for training.

    Args:
        images_dir (str): The root directory of the images.
        annotation_file (str): The path to the annotation file.
        cfg (Box): The configuration file.
        transform (transforms.Compose): The transformation to apply to the data.
        seed (int): The seed for the random number generator.
        sav_path (str): The path to the file where the data is saved/loaded from.
        use_cache (bool): Whether to use the saved data if it exists.
    """

    def __init__(
            self, 
            images_dir: str, 
            annotation_file: str, 
            cfg: Box,
            transform: transforms.Compose = None, 
            seed: int = None,
            sav_path: str = None,
            use_cache: bool = True,
            fabric = None
        ):
        self.fabric = fabric
        self.cfg = cfg
        self.seed = seed
        self.images_dir = images_dir
        self.transform = transform
        self.coco = COCO(annotation_file)
        self.samples: List[Sample] = [] # Internal storage for valid samples

        self._load_or_build_dataset(sav_path, use_cache)

    def _load_or_build_dataset(self, sav_path: str, use_cache: bool):
        """
        Loads dataset metadata from cache or builds it from scratch.
        """
        needs_build = True
        
        if self.fabric is not None:
             # Distributed barrier: wait for rank 0 to potentially handle cache
            if self.fabric.global_rank != 0:
                self.fabric.barrier()

        needs_build = True
        
        if sav_path and use_cache and os.path.exists(sav_path):
            try:
                self._print(f"Attempting to load cached dataset info from {sav_path}...")
                data = torch.load(sav_path, weights_only=False)
                self.samples = data['samples']
                needs_build = False
                self._print(f"Cached data loaded successfully. {len(self.samples)} valid images found.")
            except Exception as e:
                self._print(f"Warning: Failed to load cache from {sav_path}. Rebuilding... Error: {e}")
                needs_build = True
            
        if needs_build:
            self._print("Building dataset info...")
            self._build_dataset_index()
            
            if sav_path:
                # only rank 0 saves
                if self.fabric is None or self.fabric.global_rank == 0:
                    try:
                        self._print(f"Saving dataset info to {sav_path}...")
                        save_data = {
                            'samples': self.samples
                        }
                        torch.save(save_data, sav_path)
                        self._print("Dataset info saved successfully.")
                    except Exception as e:
                         self._print(f"Warning: Failed to save cache to {sav_path}. Error: {e}")
                
                if self.fabric is not None and self.fabric.global_rank == 0:
                    # Release the barrier after saving
                    self.fabric.barrier()
        else:
             if self.fabric is not None and self.fabric.global_rank == 0:
                 # Rank 0 didn't need to build (loaded from cache), so release the barrier for others
                 self.fabric.barrier()

    def _print(self, msg: str):
        """
        Print a message, using fabric if available.
        """
        if self.fabric is not None:
            self.fabric.print(msg)
        else:
            print(msg)

    def _build_dataset_index(self):
        """
        Iterates over all COCO images, filters out those with 0 annotations 
        or 0 valid annotations (based on points config), and calculates centroids.
        """
        image_ids = list(self.coco.imgs.keys())
        
        # Filter out image_ids without any annotations
        image_ids = [img_id for img_id in image_ids if len(self.coco.getAnnIds(imgIds=img_id)) > 0]

        # Precompute base automatic grid 
        base_grid = None
        if self.cfg.dataset.snap_to_grid and self.cfg.dataset.use_center:
             base_grid = build_point_grid(32)

        bar = tqdm.tqdm(total=len(image_ids), desc="Indexing dataset...", leave=False)
        
        for image_id in image_ids:
            image_info = self.coco.loadImgs(image_id)[0]
            H, W = (image_info['height'], image_info['width'])
            
            # Compute automatic grid
            automatic_grid = None 
            if base_grid is not None:
                automatic_grid = base_grid * np.array((H, W))[None, ::-1]

            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            anns = self.coco.loadAnns(ann_ids)
            
            valid_anns = []

            for ann in anns:
                if ann.get('iscrowd', 0) == 1:
                    continue

                mask = self.coco.annToMask(ann)
                x, y, w, h = ann['bbox']
                
                roi = mask[y:y + h, x:x + w]
                roi_indices_1 = np.where(roi == 1)
                roi_indices_0 = np.where(roi == 0)
                
                n_points_1 = roi_indices_1[0].size
                n_points_0 = roi_indices_0[0].size
                
                n_pos_req, n_neg_req = (self.cfg.dataset.positive_points, self.cfg.dataset.negative_points)
                is_valid = n_points_1 >= n_pos_req and n_points_0 >= n_neg_req

                if is_valid:
                    center_point = None
                    if n_pos_req > 0 and self.cfg.dataset.use_center:
                        points = np.stack((roi_indices_1[1] + x, roi_indices_1[0] + y), axis=1)

                        # Calculate centroid
                        mean_point = points.mean(axis=0)
                        dists = np.linalg.norm(points - mean_point, axis=1)
                        center_index = np.argmin(dists)
                        center_point = points[center_index]

                        if self.cfg.dataset.snap_to_grid and automatic_grid is not None:
                            distances = np.linalg.norm(automatic_grid - center_point, axis=1)
                            nearest_point_index = np.argmin(distances)
                            center_point = automatic_grid[nearest_point_index]

                    valid_anns.append({
                        'ann_id': ann['id'],
                        'bbox': [x, y, w, h],
                        'center_point': center_point
                    })

            if len(valid_anns) > 0:
                self.samples.append({
                    'image_id': image_id,
                    'file_name': image_info['file_name'],
                    'valid_anns': valid_anns
                })

            bar.update(1)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        Args:
            idx (int): The index of the image to get.
        Returns:
            Tuple: 
                The image (torch.uint8), 
                the original image (np.ndarray),
                the original size of the image (tuple), 
                the point coordinates (torch.float32), 
                the point labels (torch.int), 
                the boxes (torch.float32), 
                the masks (torch.uint8),
                the resized masks (torch.uint8), 
        """
        # Set the seed for reproducibility
        if self.seed is not None:
            random.seed(self.seed + idx)
            np.random.seed(self.seed + idx)

        # Retrieve sample metadata
        sample_info = self.samples[idx]
        image_id = sample_info['image_id']
        valid_anns_metadata = sample_info['valid_anns']

        # Restore the image from the folder
        image_path = os.path.join(self.images_dir, sample_info['file_name'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image = image.copy()

        # Get original size of the image
        H, W, _ = image.shape
        original_size = (H, W)

        # Prepare containers
        boxes = []
        point_coords = []
        point_labels = []
        masks = []

        # Load annotation objects from COCO using stored IDs
        ann_ids = [meta['ann_id'] for meta in valid_anns_metadata]
        anns = self.coco.loadAnns(ann_ids)

        for i, ann in enumerate(anns):
            meta = valid_anns_metadata[i]
            x, y, w, h = meta['bbox']

            # Add random noise to each coordinate with standard deviation equal to 10% of the box sidelength, to a maximum of 20 pixels
            x1_prompt, y1_prompt = float(x), float(y)
            x2_prompt, y2_prompt = float(x + w), float(y + h)

            side_len = float(max(w, h))
            sigma = min(0.1 * side_len, 20.0)
            if sigma > 0:
                dx1, dy1, dx2, dy2 = np.random.normal(loc=0.0, scale=sigma, size=(4,))
                x1_noisy = x1_prompt + float(dx1)
                y1_noisy = y1_prompt + float(dy1)
                x2_noisy = x2_prompt + float(dx2)
                y2_noisy = y2_prompt + float(dy2)

                x1_noisy = float(np.clip(x1_noisy, 0.0, float(W - 1)))
                y1_noisy = float(np.clip(y1_noisy, 0.0, float(H - 1)))
                x2_noisy = float(np.clip(x2_noisy, 0.0, float(W - 1)))
                y2_noisy = float(np.clip(y2_noisy, 0.0, float(H - 1)))

                # Ensure box is valid; otherwise fall back to the original box.
                if (x2_noisy - x1_noisy) >= 1.0 and (y2_noisy - y1_noisy) >= 1.0:
                    x1_prompt, y1_prompt, x2_prompt, y2_prompt = x1_noisy, y1_noisy, x2_noisy, y2_noisy
        
            # Generate the mask
            mask = self.coco.annToMask(ann)
            
            # Extract points
            roi = mask[y:y + h, x:x + w]
            py_1, px_1 = np.where(roi == 1)
            py_0, px_0 = np.where(roi == 0)
            
            list_points_1 = list(zip(px_1 + x, py_1 + y))
            list_points_0 = list(zip(px_0 + x, py_0 + y))

            n_pos, n_neg = (self.cfg.dataset.positive_points, self.cfg.dataset.negative_points)
            
            # Process Center Point logic
            center_point = meta.get('center_point')
            if n_pos > 0 and self.cfg.dataset.use_center and center_point is not None:
                n_pos = n_pos - 1 if n_pos > 0 else 0
            
            # Sampling points
            sample_points_1 = random.sample(list_points_1, n_pos) if len(list_points_1) >= n_pos else list_points_1
            sample_points_0 = random.sample(list_points_0, n_neg) if len(list_points_0) >= n_neg else list_points_0

            # Add center point if required
            if self.cfg.dataset.use_center and center_point is not None:
                sample_points_1.append(tuple(center_point))

            # Prepare labels
            labels_1 = [1] * len(sample_points_1)
            labels_0 = [0] * len(sample_points_0)

            # Append to batch lists
            masks.append(mask)
            boxes.append([x1_prompt, y1_prompt, x2_prompt, y2_prompt])
            point_coords.append(sample_points_1 + sample_points_0)
            point_labels.append(labels_1 + labels_0)
    
        if self.transform:
            image, resized_masks, boxes, point_coords = self.transform(
                image, 
                masks, 
                np.array(boxes, dtype=np.float32), 
                np.array(point_coords, dtype=np.float32)
            )

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        masks = torch.as_tensor(np.stack(masks, axis=0))
        resized_masks = torch.as_tensor(np.stack(resized_masks, axis=0)) if isinstance(resized_masks, list) else resized_masks
        point_coords = torch.as_tensor(point_coords, dtype=torch.float32)
        point_labels = torch.as_tensor(point_labels, dtype=torch.int)

        # Add channel dimension to the masks for compatibility with the model
        if resized_masks.ndim == 3:
            resized_masks = resized_masks.unsqueeze(1)
        
        return image, original_image, original_size, point_coords, point_labels, boxes, masks, resized_masks
    

class ResizeData:
    """
    This class handles data resizing and preprocessing (images, masks, boxes, points) 
    to prepare them for finestSAM model input.

    The internal transformations applied are:
    1. Resizing (ResizeLongestSide): Images and masks are resized while maintaining the 
       aspect ratio, such that the longest side equals `target_size`.
    2. Permutation and Conversion: The image is converted to a tensor, and channels are reordered 
       from (H, W, C) to (C, H, W) format.
    3. Mask Downsampling: Masks are reduced in resolution (to 1/4 of the resized dimension) 
       using 4x4 kernel max pooling. This is necessary because the finestSAM model expects 
       low-resolution masks during training.
    4. Coordinate Adjustment: Bounding box and prompt point coordinates are transformed 
       to correspond to the new dimensions of the resized image.
    """

    def __init__(self, target_size: int):
        self.target_size = target_size
        self.transform = ResizeLongestSide(target_size)

    def __call__(
            self, 
            image: np.ndarray, 
            masks: List[np.ndarray], 
            boxes: np.ndarray, 
            point_coords: np.ndarray
        ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Args:
            image (np.ndarray): The image to resize.
            masks (List[np.ndarray]): The masks to resize.
            boxes (np.ndarray): The bounding boxes to resize.
            point_coords (np.ndarray): The point coordinates to resize.
        Returns:
            Tuple:
                The resized image (torch.uint8),
                the resized masks (torch.uint8),
                the resized bounding boxes (torch.float32),
                the resized point coordinates (torch.float32).
        """
        # Resize image and masks
        og_h, og_w, _ = image.shape
        image = self.transform.apply_image(image)
        masks = [torch.tensor(self.transform.apply_image(mask)) for mask in masks]
        image = torch.as_tensor(image)
        image = image.permute(2, 0, 1).contiguous()

        # Resize masks to 1/4th resolution of the image
        resized_masks = []
        for mask in masks:
            mask = F.max_pool2d(mask.unsqueeze(0).unsqueeze(0).float(), kernel_size=4, stride=4).squeeze().to(torch.uint8)
            resized_masks.append(mask)

        # Adjust bounding boxes and point coordinates
        boxes = self.transform.apply_boxes(boxes, (og_h, og_w))
        point_coords = self.transform.apply_coords(point_coords, (og_h, og_w))

        return image, resized_masks, boxes, point_coords


def get_collate_fn(cfg: Box, type: str = None):
    """
    Get the collate function for the dataset.
    
    Args:
        cfg (Box): The configuration file.
        type (str, optional): The type of the dataset (None or "eval").  
        if "eval" is specified, the original image is included in the batch.
        Defaults to None.
    Returns:
        Callable: The collate function.
    """
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

            if type and type == "eval":
                data["original_image"] = original_image

            batched_data.append(data)

        return batched_data
    
    return collate_fn


def load_dataset(
        cfg: Box, 
        img_size: int,
        dataset_path: str,
        fabric = None
    ) -> Tuple[DataLoader, DataLoader]:
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
    if os.path.exists(os.path.join(dataset_path, "train")) and os.path.exists(os.path.join(dataset_path, "val")):
        auto_split = False
        print("Dataset already split found.")
    elif os.path.exists(os.path.join(dataset_path, "data")):
        auto_split = True
        print("Unsplit dataset found. Will auto-split.")
    else:
        raise ValueError(f"Dataset structure not recognized in {dataset_path}. Expected 'train'/'val' or 'data' subdirectories.")

    if auto_split:
        data_root_path = os.path.join(dataset_path, "data")
        data_path = os.path.join(data_root_path, "images")
        annotations_path = os.path.join(data_root_path, "annotations.json")
        sav_path = os.path.join(data_root_path, cfg.dataset.sav)

        data = COCODataset(images_dir=data_path,
                        annotation_file=annotations_path,
                        cfg=cfg,
                        transform=transform,
                        seed=cfg.seed_dataloader,
                        sav_path=sav_path,
                        use_cache=cfg.dataset.use_cache,
                        fabric=fabric)
        
        # Calc the size of the validation set
        total_size = len(data)
        val_size = int(total_size * cfg.dataset.val_size)

        # Split the dataset into training and validation
        train_data, val_data = random_split(data, [total_size - val_size, val_size], generator=generator)
    else:
        train_root_path = os.path.join(dataset_path, "train")
        train_path = os.path.join(train_root_path, "images")
        train_annotations_path = os.path.join(train_root_path, "annotations.json")
        train_sav_path = os.path.join(train_root_path, cfg.dataset.sav)

        val_root_path = os.path.join(dataset_path, "val")    
        val_path =  os.path.join(val_root_path, "images")
        val_annotations_path = os.path.join(val_root_path, "annotations.json")
        val_sav_path = os.path.join(val_root_path, cfg.dataset.sav)


        train_data = COCODataset(images_dir=train_path,
                        annotation_file=train_annotations_path,
                        cfg=cfg,
                        transform=transform,
                        seed=cfg.seed_dataloader,
                        sav_path=train_sav_path,
                        use_cache=cfg.dataset.use_cache,
                        fabric=fabric)

        val_data = COCODataset(images_dir=val_path,
                        annotation_file=val_annotations_path,
                        cfg=cfg,
                        transform=transform,
                        seed=cfg.seed_dataloader,
                        sav_path=val_sav_path,
                        use_cache=cfg.dataset.use_cache,
                        fabric=fabric)
    
    train_dataloader = DataLoader(train_data,
                                  batch_size=cfg.batch_size,
                                  shuffle=True,
                                  generator=generator,
                                  num_workers=cfg.num_workers,
                                  collate_fn=get_collate_fn(cfg, "train"))

    val_dataloader = DataLoader(val_data,
                                batch_size=cfg.batch_size,
                                shuffle=False,
                                num_workers=cfg.num_workers,
                                collate_fn=get_collate_fn(cfg, "eval"))

    return train_dataloader, val_dataloader


def load_test_dataset(
        cfg: Box,
        img_size: int,
        dataset_path: str,
        fabric = None
    ) -> DataLoader:
    """
    Load the test dataset and return the dataloader.

    Args:
        cfg (Box): The configuration file.
        img_size (int): The size of the image to resize to.
        dataset_path (str): The path to the test dataset.
    Returns:
        DataLoader: The test dataloader.
    """
    # Set the seed 
    generator = torch.Generator()
    if cfg.dataset.seed != None:
        generator.manual_seed(cfg.dataset.seed)

    # Set up the transformation for the dataset
    transform = ResizeData(img_size)

    # Load the dataset
    images_path = os.path.join(dataset_path, "images")
    annotations_path = os.path.join(dataset_path, "annotations.json")
    
    if not os.path.exists(images_path) or not os.path.exists(annotations_path):
        raise ValueError(f"Dataset structure not recognized in {dataset_path}. Expected 'images' directory and 'annotations.json' file.")

    sav_path = os.path.join(dataset_path, cfg.dataset.sav)

    test_data = COCODataset(images_dir=images_path,
                    annotation_file=annotations_path,
                    cfg=cfg,
                    transform=transform,
                    seed=cfg.seed_dataloader,
                    sav_path=sav_path,
                    use_cache=cfg.dataset.use_cache,
                    fabric=fabric)
    
    test_dataloader = DataLoader(test_data,
                                  batch_size=cfg.batch_size,
                                  shuffle=False,
                                  num_workers=cfg.num_workers,
                                  collate_fn=get_collate_fn(cfg, "eval"))

    return test_dataloader