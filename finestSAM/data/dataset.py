import os
import cv2
import tqdm
import torch
import random
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from box import Box
from typing import Tuple, List, Optional, TypedDict, Dict, Any
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
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
        images_dir (str): Directory containing the images.
        annotation_file (str): Path to the COCO annotation file.
        cfg (Box): Configuration object.
        transform (transforms.Compose, optional): Transformations to apply.
        seed (int, optional): Seed for random number generation.
        vary_seed_by_epoch (bool): Whether to vary the seed by epoch.
        sav_path (str, optional): Path to save/load the cached dataset.
        use_cache (bool): Whether to use caching.
        fabric: (optional): Lightning Fabric object for distributed training.
    """

    def __init__(
        self, 
        images_dir: str, 
        annotation_file: str, 
        cfg: Box,
        transform: transforms.Compose = None, 
        seed: int = None,
        vary_seed_by_epoch: bool = False,
        sav_path: str = None,
        use_cache: bool = True,
        fabric = None
    ):
        self.images_dir = images_dir
        self.annotation_file = annotation_file
        self.cfg = cfg
        self.transform = transform
        self.seed = seed
        self.vary_seed_by_epoch = vary_seed_by_epoch
        self.fabric = fabric
        
        self.epoch = 0
        self.coco = COCO(annotation_file)
        self.samples: List[Sample] = []
        
        # Load data
        self._load_or_build_dataset(sav_path, use_cache)

    def set_epoch(self, epoch: int) -> None:
        """
        Set current epoch for deterministic noise variation.
        
        Args:
            epoch (int): Current epoch number.
        """
        self.epoch = int(epoch)


    def __len__(self):
        return len(self.samples)

    def _get_rng(self, idx: int) -> Tuple[random.Random, np.random.Generator]:
        """
        Get deterministic random generators for the given index.

        Args:
            idx (int): Index of the sample.
        """
        if self.seed is None:
            return random, np.random

        # Calculate a unique offset. 
        # By adding (epoch * total_samples
        # we essentially 'flatten' the (epoch, index) grid into a linear sequence of unique seeds.
        epoch_offset = (self.epoch * len(self)) if self.vary_seed_by_epoch else 0
        unique_seed = int(self.seed) + epoch_offset + int(idx)

        py_rng = random.Random(unique_seed)
        np_rng = np.random.default_rng(unique_seed)
        
        return py_rng, np_rng

    def _print(self, msg: str):
        if self.fabric is not None:
            self.fabric.print(msg)
        else:
            print(msg)

    def _load_or_build_dataset(self, sav_path: str, use_cache: bool):
        """
        Load dataset from cache or build it from COCO annotations.

        Args:
            sav_path (str): Path to save/load the cached dataset.
            use_cache (bool): Whether to use caching.
        """
        # Wait for rank 0
        if self.fabric is not None and self.fabric.global_rank != 0:
            self.fabric.barrier()

        loaded = False
        if sav_path and use_cache and os.path.exists(sav_path):
            try:
                self._print(f"Loading cached dataset from {sav_path}...")
                data = torch.load(sav_path, weights_only=False)
                self.samples = data['samples']
                loaded = True
                self._print(f"Cache loaded. {len(self.samples)} samples found.")
            except Exception as e:
                self._print(f"Warning: Cache load failed ({e}). Rebuilding...")

        if not loaded:
            self._print("Building dataset index...")
            self._build_dataset_index()
            
            # Save cache (Rank 0 only)
            if sav_path and (self.fabric is None or self.fabric.global_rank == 0):
                try:
                    self._print(f"Saving dataset to {sav_path}...")
                    torch.save({'samples': self.samples}, sav_path)
                    self._print("Dataset saved.")
                except Exception as e:
                    self._print(f"Warning: Failed to save cache ({e}).")

        # Release non-zero ranks
        if self.fabric is not None and self.fabric.global_rank == 0:
            self.fabric.barrier()

    def _build_dataset_index(self):
        """
        Build the dataset index from COCO annotations.
        Filters annotations based on point sufficiency and computes center points.
        """
       # Get all image IDs with annotations 
        image_ids = sorted([img_id for img_id in self.coco.imgs.keys() 
                            if len(self.coco.getAnnIds(imgIds=img_id)) > 0])

        # Precompute base automatic grid
        base_grid = None
        if self.cfg.dataset.snap_to_grid and self.cfg.dataset.use_center:
            base_grid = build_point_grid(32)

        desc = "Indexing..."
        for image_id in tqdm.tqdm(image_ids, desc=desc, leave=False):
            image_info = self.coco.loadImgs(image_id)[0]
            H, W = image_info['height'], image_info['width']
            
            # Prepare automatic grid for this image size
            automatic_grid = None
            if base_grid is not None:
                automatic_grid = base_grid * np.array((H, W))[None, ::-1]

            ann_ids = sorted(self.coco.getAnnIds(imgIds=image_id))
            anns = self.coco.loadAnns(ann_ids)
            valid_anns = []

            for ann in anns:
                if ann.get('iscrowd', 0) == 1:
                    continue

                mask = self.coco.annToMask(ann)
                x, y, w, h = ann['bbox']
                
                # Check point sufficiency
                roi = mask[int(y):int(y+h), int(x):int(x+w)]
                if not self._check_validity(roi):
                    continue
                
                # Calculate center point
                center_point = self._calculate_center(roi, x, y, automatic_grid)

                valid_anns.append({
                    'ann_id': ann['id'],
                    'bbox': [x, y, w, h],
                    'center_point': center_point
                })

            if valid_anns:
                self.samples.append({
                    'image_id': image_id,
                    'file_name': image_info['file_name'],
                    'valid_anns': sorted(valid_anns, key=lambda a: a['ann_id'])
                })

    def _check_validity(self, roi: np.ndarray) -> bool:
        """
        Check if ROI has enough positive/negative points.
        
        Args:
            roi (np.ndarray): Region of interest mask.
        Returns:
            bool: True if valid, False otherwise.
        """
        n_pos = np.count_nonzero(roi)
        n_neg = roi.size - n_pos
        return (n_pos >= self.cfg.dataset.positive_points and 
                n_neg >= self.cfg.dataset.negative_points)

    def _calculate_center(self, roi: np.ndarray, x: int, y: int, grid: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        Calculate the center point of the ROI.
        If configured, snaps the center to the nearest grid point.
        
        Args:
            roi (np.ndarray): Region of interest mask.
            x (int): X offset of the ROI in the image.
            y (int): Y offset of the ROI in the image.
            grid (Optional[np.ndarray]): Precomputed grid for snapping.
        Returns:
            Optional[np.ndarray]: Center point coordinates or None.
        """
        if not (self.cfg.dataset.positive_points > 0 and self.cfg.dataset.use_center):
            return None

        ys, xs = np.where(roi == 1)
        points = np.stack((xs + x, ys + y), axis=1)
        
        # Geometric centroid
        mean_point = points.mean(axis=0)
        dists = np.linalg.norm(points - mean_point, axis=1)
        center_point = points[np.argmin(dists)]

        # Snap to grid
        if self.cfg.dataset.snap_to_grid and grid is not None:
            grid_dists = np.linalg.norm(grid - center_point, axis=1)
            center_point = grid[np.argmin(grid_dists)]
            
        return center_point

    def _add_box_noise(self, bbox: List[float], H: int, W: int, rng: np.random.Generator) -> List[float]:
        """
        Adds random noise to the bounding box.
        The noise is Gaussian with stddev = min(0.1 * side_length, 20.0).

        Args:
            bbox (List[float]): Original bounding box [x, y, w, h].
            H (int): Image height.
            W (int): Image width.
            rng (np.random.Generator): Numpy random generator.
        Returns:
            List[float]: Noisy bounding box [x1, y1, x2, y2].
        """
        x, y, w, h = bbox
        x1, y1 = float(x), float(y)
        x2, y2 = float(x + w), float(y + h)

        side_len = float(max(w, h))
        sigma = min(0.1 * side_len, 20.0)

        if sigma <= 0:
            return [x1, y1, x2, y2]

        dx1, dy1, dx2, dy2 = rng.normal(loc=0.0, scale=sigma, size=4)
        
        nx1 = np.clip(x1 + dx1, 0, W - 1)
        ny1 = np.clip(y1 + dy1, 0, H - 1)
        nx2 = np.clip(x2 + dx2, 0, W - 1)
        ny2 = np.clip(y2 + dy2, 0, H - 1)

        # Validate box integrity
        if (nx2 - nx1) >= 1.0 and (ny2 - ny1) >= 1.0:
            return [float(nx1), float(ny1), float(nx2), float(ny2)]
        
        return [x1, y1, x2, y2]

    def _sample_mask_points(self, mask: np.ndarray, bbox: List[int], meta: ValidAnn, rng: random.Random) -> Tuple[List, List]:
        """
        Samples positive and negative points from the mask.
        
        Args:
            mask (np.ndarray): The binary mask of the object.
            bbox (List[int]): Bounding box [x, y, w, h].
            meta (ValidAnn): Metadata for the annotation.
            rng (random.Random): Python random generator.
        Returns:
            Tuple[List, List]: Sampled positive and negative points.
        """
        x, y, w, h = [int(v) for v in bbox]
        roi = mask[y:y+h, x:x+w]
        
        # Get coordinates relative to image
        py_1, px_1 = np.where(roi == 1)
        py_0, px_0 = np.where(roi == 0)
        
        pts_1 = list(zip(px_1 + x, py_1 + y))
        pts_0 = list(zip(px_0 + x, py_0 + y))

        n_pos = self.cfg.dataset.positive_points
        n_neg = self.cfg.dataset.negative_points
        
        # Handle center point exclusion from count
        center_pt = meta.get('center_point')
        if n_pos > 0 and self.cfg.dataset.use_center and center_pt is not None:
            n_pos = max(0, n_pos - 1)

        # Sampling
        sampled_1 = rng.sample(pts_1, n_pos) if len(pts_1) >= n_pos else pts_1
        sampled_0 = rng.sample(pts_0, n_neg) if len(pts_0) >= n_neg else pts_0

        if self.cfg.dataset.use_center and center_pt is not None:
            sampled_1.append(tuple(center_pt))
            
        return sampled_1, sampled_0

    def __getitem__(self, idx: int) -> Tuple:
        """
        Get a data sample by index.

        Args:
            idx (int): Index of the sample.
        Returns:
            Tuple: (image, original_image, (H, W), point_coords, point_labels, boxes, masks, resized_masks)
        """
        py_rng, np_rng = self._get_rng(idx)
        
        sample = self.samples[idx]
        valid_anns = sorted(sample['valid_anns'], key=lambda a: a['ann_id'])

        # Load Image
        image_path = os.path.join(self.images_dir, sample['file_name'])
        image_bgr = cv2.imread(image_path)
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        original_image = image.copy()
        H, W, _ = image.shape

        # Containers
        boxes, point_coords, point_labels, masks = [], [], [], []

        # Load all annotations for this image at once
        ann_ids = [m['ann_id'] for m in valid_anns]
        coco_anns = self.coco.loadAnns(ann_ids)
        # Create a map for quick access to metadata
        meta_map = {m['ann_id']: m for m in valid_anns}

        for ann in coco_anns:
            meta = meta_map[ann['id']]
            
            # Process Box (with Noise)
            noisy_box = self._add_box_noise(meta['bbox'], H, W, np_rng)
            boxes.append(noisy_box)

            # Process Mask & Points
            mask = self.coco.annToMask(ann)
            pts_1, pts_0 = self._sample_mask_points(mask, meta['bbox'], meta, py_rng)
            
            masks.append(mask)
            point_coords.append(pts_1 + pts_0)
            point_labels.append([1] * len(pts_1) + [0] * len(pts_0))

        # Transform & Convert
        if self.transform:
            image, resized_masks, boxes, point_coords = self.transform(
                image, 
                masks, 
                np.array(boxes, dtype=np.float32), 
                np.array(point_coords, dtype=np.float32)
            )

        # Final Tensor Conversion
        out_boxes = torch.as_tensor(boxes, dtype=torch.float32)
        out_masks = torch.as_tensor(np.stack(masks, axis=0))
        out_resized_masks = (torch.as_tensor(np.stack(resized_masks, axis=0)) 
                             if isinstance(resized_masks, list) else resized_masks)
        
        if out_resized_masks.ndim == 3:
            out_resized_masks = out_resized_masks.unsqueeze(1)

        out_points = torch.as_tensor(point_coords, dtype=torch.float32)
        out_labels = torch.as_tensor(point_labels, dtype=torch.int)

        return (image, original_image, (H, W), out_points, out_labels, 
                out_boxes, out_masks, out_resized_masks)


class ResizeData:
    """
    Preprocesses images, masks, boxes, and points for FinestSAM.
    
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

    Args:
        target_size (int): The target size for the longest side of the image.
    """

    def __init__(self, target_size: int):
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
            image (np.ndarray): The input image.
            masks (List[np.ndarray]): List of mask arrays.
            boxes (np.ndarray): Array of bounding boxes.
            point_coords (np.ndarray): Array of point coordinates.

        Returns:
            Tuple containing the transformed image tensor, list of resized mask tensors,
            transformed bounding boxes, and transformed point coordinates.
        """
        
        og_h, og_w = image.shape[:2]

        # Resize Image
        img_resized = self.transform.apply_image(image)
        img_tensor = torch.as_tensor(img_resized).permute(2, 0, 1).contiguous()

        # Resize & Downsample Masks
        resized_masks = []
        for m in masks:
            m_res = self.transform.apply_image(m)
            m_tensor = torch.tensor(m_res).unsqueeze(0).unsqueeze(0).float()
            # Downsample by 4
            m_small = F.max_pool2d(m_tensor, kernel_size=4, stride=4).squeeze().to(torch.uint8)
            resized_masks.append(m_small)

        # Transform Coordinates
        boxes_res = self.transform.apply_boxes(boxes, (og_h, og_w))
        points_res = self.transform.apply_coords(point_coords, (og_h, og_w))

        return img_tensor, resized_masks, boxes_res, points_res


def get_collate_fn(cfg: Box, mode: str = None):
    """
    Get the collate function for the DataLoader.
    
    Args:
        cfg (Box): Configuration object.
        mode (str): Mode of operation, either "train" or "eval".
                    if "eval", original images are included.
    Returns:
        A collate function to be used in DataLoader.
    """
    
    def collate_fn(batch: List[Tuple]):
        batched_data = []
        for item in batch:
            (image, orig_img, orig_size, pts, labels, boxes, masks, res_masks) = item

            data = {
                "image": image,
                "original_size": orig_size,
                "gt_masks": masks,
            }

            if cfg.prompts.use_boxes:
                data["boxes"] = boxes
            if cfg.prompts.use_points:
                data["point_coords"] = pts
                data["point_labels"] = labels
            if cfg.prompts.use_masks:
                data["mask_inputs"] = res_masks
            
            if mode == "eval":
                data["original_image"] = orig_img

            batched_data.append(data)
        return batched_data
    
    return collate_fn

def _seed_worker():
    """Deterministically seed workers based on PyTorch initial seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_dataset_epoch(dataset: Dataset, epoch: int) -> None:
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)

def set_dataloader_epoch(dataloader: DataLoader, epoch: int) -> None:
    sampler = getattr(dataloader, "sampler", None)
    if sampler is None:
        inner = getattr(dataloader, "_dataloader", None)
        if inner:
            sampler = getattr(inner, "sampler", None)
            
    if sampler and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(int(epoch))


def load_dataset(cfg: Box, img_size: int, dataset_path: str, fabric=None) -> Tuple[DataLoader, DataLoader]:
    generator = torch.Generator()
    if cfg.dataset.seed is not None:
        generator.manual_seed(cfg.dataset.seed)

    transform = ResizeData(img_size)
    
    # Check structure
    has_split = os.path.exists(os.path.join(dataset_path, "train"))
    has_data = os.path.exists(os.path.join(dataset_path, "data"))
    
    if not (has_split or has_data):
        raise ValueError(f"Invalid dataset structure in {dataset_path}")

    # Helper to init dataset
    def create_ds(path, ann_file, sav_name, vary_seed):
        return COCODataset(
            images_dir=path,
            annotation_file=ann_file,
            cfg=cfg,
            transform=transform,
            seed=cfg.dataset.seed,
            vary_seed_by_epoch=vary_seed,
            sav_path=os.path.join(os.path.dirname(path), sav_name),
            use_cache=cfg.dataset.use_cache,
            fabric=fabric
        )

    if has_data and not has_split:
        print("Unsplit dataset found. Auto-splitting...")
        base_path = os.path.join(dataset_path, "data")
        img_path = os.path.join(base_path, "images")
        ann_path = os.path.join(base_path, "annotations.json")
        
        # Build index once
        full_ds = create_ds(img_path, ann_path, cfg.dataset.sav, False)
        
        # Split indices
        total = len(full_ds)
        val_len = int(total * cfg.dataset.val_size)
        perm = torch.randperm(total, generator=generator).tolist()
        val_idx = sorted(perm[:val_len])
        train_idx = sorted(perm[val_len:])

        # Re-instantiate to separate train/val behavior (vary_seed)
        train_data = create_ds(img_path, ann_path, cfg.dataset.sav, True)
        val_data = create_ds(img_path, ann_path, cfg.dataset.sav, False)
        
        # Assign samples
        train_data.samples = [full_ds.samples[i] for i in train_idx]
        val_data.samples = [full_ds.samples[i] for i in val_idx]
        
    else:
        print("Split dataset found.")
        train_root = os.path.join(dataset_path, "train")
        val_root = os.path.join(dataset_path, "val")
        
        train_data = create_ds(
            os.path.join(train_root, "images"),
            os.path.join(train_root, "annotations.json"),
            cfg.dataset.sav, True
        )
        val_data = create_ds(
            os.path.join(val_root, "images"),
            os.path.join(val_root, "annotations.json"),
            cfg.dataset.sav, False
        )

    # Dataloaders
    train_loader = DataLoader(
        train_data, batch_size=cfg.batch_size, shuffle=True,
        generator=generator, worker_init_fn=_seed_worker,
        num_workers=cfg.num_workers, collate_fn=get_collate_fn(cfg, "train")
    )
    val_loader = DataLoader(
        val_data, batch_size=cfg.batch_size, shuffle=False,
        worker_init_fn=_seed_worker, num_workers=cfg.num_workers, 
        collate_fn=get_collate_fn(cfg, "eval")
    )

    return train_loader, val_loader


def load_test_dataset(cfg: Box, img_size: int, dataset_path: str, fabric=None) -> DataLoader:
    transform = ResizeData(img_size)
    img_path = os.path.join(dataset_path, "images")
    ann_path = os.path.join(dataset_path, "annotations.json")
    sav_path = os.path.join(dataset_path, cfg.dataset.sav)

    if not os.path.exists(img_path):
        raise ValueError(f"Images not found in {img_path}")

    test_data = COCODataset(
        images_dir=img_path,
        annotation_file=ann_path,
        cfg=cfg,
        transform=transform,
        seed=cfg.dataset.seed,
        vary_seed_by_epoch=False,
        sav_path=sav_path,
        use_cache=cfg.dataset.use_cache,
        fabric=fabric
    )

    return DataLoader(
        test_data, batch_size=cfg.batch_size, shuffle=False,
        worker_init_fn=_seed_worker, num_workers=cfg.num_workers,
        collate_fn=get_collate_fn(cfg, "eval")
    )