import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class COCODataset(Dataset):

    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())

        # Filter out image_ids without any annotations
        self.image_ids = [image_id for image_id in self.image_ids if len(self.coco.getAnnIds(imgIds=image_id)) > 0]

    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        # Restor the image from the folder
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.root_dir, image_info['file_name'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get original size of the image
        H, W, _ = image.shape
        original_size = (H, W)

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        boxes = []
        point_coords = []
        masks = []

        # Get box, point and mask for any annotations
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            point_coords.append([[x + w / 2, y + h / 2]])
            mask = self.coco.annToMask(ann)
            masks.append(mask)

        if self.transform:
            image, masks, boxes, point_coords = self.transform(image, masks, np.array(boxes), np.array(point_coords))

        # Create the labels for the points
        point_labels = np.ones((len(point_coords), 1))

        # Convert the data to tensor
        boxes = torch.tensor(np.stack(boxes, axis=0))
        masks = torch.tensor(np.stack(masks, axis=0)).float().to(torch.float64)
        point_coords = torch.tensor(np.stack(point_coords, axis=0))
        point_labels = torch.as_tensor(point_labels, dtype=torch.int)

        masks = masks.unsqueeze(1)
        
        return image, original_size, point_coords, point_labels, boxes, masks


def collate_fn(batch):
    image, original_size, point_coords, point_labels, boxes, masks = zip(*batch)

    batched_input = {
        "image": image[0],
        "original_size": original_size[0],
        "point_coords": point_coords[0],
        "point_labels": point_labels[0],
        "boxes": boxes[0],
        "mask_inputs": masks[0]
    }

    return [batched_input]


class ResizeAndPad:

    def __init__(self, target_size):
        self.target_size = target_size
        self.transform = ResizeLongestSide(target_size)
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image, masks, bboxes, point_coords):
        # Resize image and masks
        og_h, og_w, _ = image.shape
        image = self.transform.apply_image(image)
        masks = [torch.tensor(self.transform.apply_image(mask)) for mask in masks]
        image = self.to_tensor(image)

        # Resize masks to 1/4th resolution of the image 
        # SAM RICHIEDE DELLE MASCHERE 4X RISOLUZIONE INFERIORE DELLE IMMAGINI, PERO LE NOSTRE LOSS FUNCTION SONO FATTE PER DELLE MASCHERE DI RISOLUZIONE UGUALE, CAPIRE SE SONO DA CAMBIARE E SE SI COME CAMBIARLE
        resized_masks = []
        for mask in masks:
            resized_mask = transforms.Resize((mask.shape[0] // 4, mask.shape[1] // 4))(mask)
            resized_masks.append(resized_mask)

        # Pad image and masks to form a square
        _, h, w = image.shape
        max_dim = max(w, h)
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2

        padding = (pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h)
        image = transforms.Pad(padding)(image)
        resized_masks = [transforms.Pad(padding)(mask) for mask in resized_masks]

        # Adjust bounding boxes
        bboxes = self.transform.apply_boxes(bboxes, (og_h, og_w))
        bboxes = [[bbox[0] + pad_w, bbox[1] + pad_h, bbox[2] + pad_w, bbox[3] + pad_h] for bbox in bboxes]

        point_coords = self.transform.apply_coords(point_coords, (og_h, og_w))
        point_coords[..., 0] += pad_w
        point_coords[..., 1] += pad_h

        return image, resized_masks, bboxes, point_coords


def load_datasets(cfg, img_size):
    transform = ResizeAndPad(img_size)

   # Ottiene il percorso del dataset
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    dataset_path = os.path.join(current_directory, cfg.dataset.root_dir)
    annotations_path = os.path.join(current_directory, cfg.dataset.annotation_file)

    train = COCODataset(root_dir=dataset_path,
                        annotation_file=annotations_path,
                        transform=transform)
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
    
    # Visualize the dataset CAMBIARE 
    # image, bboxes, masks = train[0]
    # image = np.transpose(image, (1, 2, 0))
    # for i in range(len(masks)):
    #     plt.figure(figsize=(8, 8))
    #     plt.imshow(image)
    #     plt.imshow(masks[i], alpha=0.5, cmap='jet')  # Overlay masks on the image
    #     plt.axis('off')
    #     plt.show()

    return train_dataloader, val_dataloader