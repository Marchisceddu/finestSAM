import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from box import Box
from typing import Any, Dict, List
from .segment_anything import sam_model_registry
from .segment_anything import SamPredictor, SamAutomaticMaskGenerator


class FinestSAM(nn.Module):

    def __init__(self, cfg: Box):
        super().__init__()
        self.cfg = cfg

    def setup(self):
        """Set up the model."""
        checkpoint = os.path.join(self.cfg.sav_dir, self.cfg.model.checkpoint)

        self.model = sam_model_registry[self.cfg.model.type](checkpoint=checkpoint)

        if torch.is_grad_enabled():
            if self.cfg.model_layer.freeze.image_encoder:
                for param in self.model.image_encoder.parameters():
                    param.requires_grad = False
            if self.cfg.model_layer.freeze.prompt_encoder:
                for param in self.model.prompt_encoder.parameters():
                    param.requires_grad = False
            if self.cfg.model_layer.freeze.mask_decoder:
                for param in self.model.mask_decoder.parameters():
                    param.requires_grad = False

    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
        are_logits: bool = False
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
                (H or W must have the minimum size of self.model.image_encoder.img_size)
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
                (must be 1/4 the size of the image post-transformation, so self.model.image_encoder.img_size//4)
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.model.image_encoder(input_images)

        input_masks = [x["mask_inputs"] if "mask_inputs" in x and x["mask_inputs"] is not None else None for x in batched_input]
        if not are_logits:
            input_masks = [self.preprocess(mask) if mask is not None else None for mask in input_masks]

        outputs = []
        for image_record, curr_embedding, masks in zip(batched_input, image_embeddings, input_masks):
            
            if "point_coords" in image_record and image_record["point_coords"] is not None:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None

            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=masks,
            )

            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )

            masks = self.model.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )

            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )

        return outputs
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Pad to a square input."""
        h, w = x.shape[-2:]
        img_size = max(h, w) 
        padh = img_size - h
        padw = img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    
    def get_predictor(self):
        return SamPredictor(self.model)

    def get_automatic_predictor(
            self, 
            min_mask_region_area: int = 0
        ): # aggiungere altre paramentri per modificare
        return SamAutomaticMaskGenerator(model=self.model, 
                                         min_mask_region_area=min_mask_region_area)