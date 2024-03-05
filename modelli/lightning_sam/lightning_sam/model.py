import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry
from segment_anything import SamPredictor, SamAutomaticMaskGenerator
import numpy as np


class Model(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self):
        self.model = sam_model_registry[self.cfg.model.type](checkpoint=self.cfg.model.checkpoint)
        self.model.train()
        if self.cfg.model.freeze.image_encoder:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
        if self.cfg.model.freeze.prompt_encoder:
            for param in self.model.prompt_encoder.parameters():
                param.requires_grad = False
        if self.cfg.model.freeze.mask_decoder:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = False

    def forward(self, images, bboxes=None, centers=None):
        if not bboxes and not centers:
            raise ValueError("Either bboxes or centers must be provided")
        _, _, H, W = images.shape
        image_embeddings = self.model.image_encoder(images)
        pred_masks = []
        ious = []
        if not centers:
            centers = [None] * len(bboxes)
        if not bboxes:
            bboxes = [None] * len(centers)
        for embedding, bbox, center in zip(image_embeddings, bboxes, centers):
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=center,
                boxes=bbox,
                masks=None,
            )

            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=embedding.unsqueeze(0),
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            masks = F.interpolate(
                low_res_masks,
                (H, W),
                mode="bilinear",
                align_corners=False,
            )
            pred_masks.append(masks.squeeze(1))
            ious.append(iou_predictions)

        return pred_masks, ious

    def get_predictor(self):
        return (self.model)