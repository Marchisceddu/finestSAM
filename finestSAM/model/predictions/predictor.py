import os
import cv2
import torch
import numpy as np
import lightning as L
import geopandas as gpd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from box import Box
from shapely.geometry import Polygon
from .utils import (
    show_anns,
    show_mask,
    show_points,
    show_box
)
from ..utils import set_model
from ..model import FinestSAM
from ..dataset import COCODataset

def show_anns_on_image(image, anns, opacity=0.35):
    if len(anns) == 0:
        return image
    
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    # Converti l'immagine in un array RGBA (con canale alpha)
    img_rgba = np.array(image.convert("RGBA"))

    # Creiamo un array per l'immagine con le maschere sovrapposte
    mask_img = np.zeros_like(img_rgba, dtype=np.uint8)

    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.randint(0, 255, 3), [int(opacity * 255)]])
        mask_img[m] = color_mask

    # Combina l'immagine originale con le maschere
    combined_img = Image.alpha_composite(Image.fromarray(img_rgba), Image.fromarray(mask_img))

    return combined_img


def automatic_predictions(
        cfg: Box, 
        path: str,
        approx_accuracy: float = 0.01
    ):
    """
    Predict the masks of the image and save them in a shapefile.
    
    Args:
        cfg (Box): The configuration file.
        path (str): The path of the image.
        approx_accuracy (float): The approximation accuracy of the polygons.
    """
    # Get the paths
    main_directory = os.path.dirname(os.path.abspath(__file__)).rsplit('/', 2)[0]
    cfg.sav_dir = os.path.join(main_directory, cfg.sav_dir)
    cfg.out_dir = os.path.join(main_directory, cfg.out_dir)
    image_path = os.path.join(main_directory, path)

    # Get the image
    image = cv2.imread(image_path)

    # Load the model 
    model, fabric = set_model(cfg, save_loggers=False)

    # Predict the masks
    predictor = model.get_automatic_predictor(min_mask_region_area = 300)
    masks = predictor.generate(image)

    # Show the image with the masks RISCRIVERLO BENE
    polygons = []
    for i, mask in enumerate(masks):
        mask = mask["segmentation"].astype(np.uint8)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        
        for contour in contours:
          epsilon = approx_accuracy * cv2.arcLength(contour, True)
          approx = cv2.approxPolyDP(contour, epsilon, True)
          points = [(int(point[0][0]), -int(point[0][1])) for point in approx]
          polygons.append(Polygon(points))
                  
    
    # Crea un GeoDataFrame utilizzando i poligoni Shapely
    gdf = gpd.GeoDataFrame(geometry=polygons)

    # Salva il GeoDataFrame in un file Shapefile
    gdf.to_file(os.path.join(cfg.out_dir, "output.shp"))

    image = Image.fromarray(image)  # Converti l'array numpy a un oggetto PIL.Image
    combined_image = show_anns_on_image(image, masks, opacity=0.35)

    # Salva l'immagine combinata come file PNG
    combined_image.save('./output.png')
    
    # fig, ax = plt.subplots()
    # gdf.boundary.plot(ax=ax)
    # ax.set_aspect('equal')
    # ax.axis('off')
    # ax.set_xticks([])
    # plt.savefig("./output.svg")
    # plt.show()


# Predittori manuali, da cambiare
def pred_boxes(cfg: Box):
    main_directory = os.path.dirname(os.path.abspath(__file__))

    # Load the model 
    fabric = L.Fabric(accelerator=cfg.device,
                    devices=cfg.num_devices,
                    strategy="auto")
    fabric.launch()
    fabric.seed_everything(cfg.seed_device)

    with fabric.device:
        model = FinestSAM(cfg)
        model.setup()
        model.to(fabric.device)

    # Get the dataset for boxes
    dataset_path = os.path.join(main_directory, cfg.dataset.root_dir)
    annotations_path = os.path.join(main_directory, cfg.dataset.annotation_file)

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

        print(transformed_boxes)

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


def pred_points(cfg: Box):
    main_directory = os.path.dirname(os.path.abspath(__file__))

    # Load the model 
    fabric = L.Fabric(accelerator=cfg.device,
                    devices=cfg.num_devices,
                    strategy="auto")
    fabric.launch()
    fabric.seed_everything(cfg.seed_device)

    with fabric.device:
        model = FinestSAM(cfg)
        model.setup()
        model.to(fabric.device)

    # Get the dataset for boxes
    dataset_path = os.path.join(main_directory, cfg.dataset.root_dir)
    annotations_path = os.path.join(main_directory, cfg.dataset.annotation_file)

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