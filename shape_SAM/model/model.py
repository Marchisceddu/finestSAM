import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any, Dict, List

from .segment_anything import sam_model_registry
from .segment_anything import SamPredictor, SamAutomaticMaskGenerator


class shape_SAM(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self):
        current_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        checkpoint = os.path.join(current_directory, self.cfg.out_dir, self.cfg.model.checkpoint)

        self.model = sam_model_registry[self.cfg.model.type](checkpoint=checkpoint)
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


    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
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
        input_images = torch.stack([self.model.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.model.image_encoder(input_images)

        input_masks = [self.preprocess_masks(x["mask_inputs"]) for x in batched_input]

        outputs = []
        for image_record, curr_embedding, mask in zip(batched_input, image_embeddings, input_masks):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None

            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=mask,
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
  
    def preprocess_masks(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Pad
        h, w = x.shape[-2:]
        padh = self.model.image_encoder.img_size//4 - h
        padw = self.model.image_encoder.img_size//4 - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    
    def get_predictor(self):
        return SamPredictor(model = self.model)

    def get_automatic_predictor(self, min_mask_region_area = 0):
        return SamAutomaticMaskGenerator(model = self.model, min_mask_region_area = min_mask_region_area)