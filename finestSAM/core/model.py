import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from box import Box
from typing import Any, Dict, List
from .segment_anything import sam_model_registry
from .segment_anything import SamPredictor, SamAutomaticMaskGenerator
from .lora import inject_lora_sam


class FinestSAM(nn.Module):

    def __init__(self, cfg: Box):
        super().__init__()
        self.cfg = cfg

    def setup(self):
        """Set up the model."""
        checkpoint = os.path.join(self.cfg.sav_dir, self.cfg.model.checkpoint)

        try:
            self.model = sam_model_registry[self.cfg.model.type](checkpoint=checkpoint)
        except (RuntimeError, FileNotFoundError, KeyError) as e:
            if isinstance(e, RuntimeError) and ("Error(s) in loading state_dict" in str(e) or "size mismatch" in str(e)):
                 raise RuntimeError(
                    f"\n\nERROR: Failed to load checkpoint '{self.cfg.model.checkpoint}' for model type '{self.cfg.model.type}'.\n"
                    "Please ensure that the checkpoint corresponds to the selected model type.\n"
                    "You can specify the correct model type in the configuration file or via arguments."
                ) from e
            elif isinstance(e, FileNotFoundError):
                 raise FileNotFoundError(
                    f"\n\nERROR: Checkpoint file not found at '{checkpoint}'.\n"
                    "Please ensure the file path is correct and exists.\n"
                ) from e
            elif isinstance(e, KeyError):
                 raise KeyError(
                    f"\n\nERROR: Invalid model type '{self.cfg.model.type}'.\n"
                    f"Available types are: {list(sam_model_registry.keys())}.\n"
                ) from e
            else:
                raise e
          
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

            lora_cfg = getattr(self.cfg.model_layer, "LORA", None)
            if lora_cfg:
              self.model = inject_lora_sam(self.model, lora_cfg=lora_cfg)

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
                already transformed for input to the model (dtype: torch.float32).
                (H or W must have the minimum size of self.model.image_encoder.img_size)
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2 (dtype: torch.float32). Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN (dtype: torch.int).
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4 (dtype: torch.float32).
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW (dtype: torch.uint8).
                (must be 1/4 the size of the image post-transformation, so self.model.image_encoder.img_size//4)
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW (dtype: torch.float32), where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC (dtype: torch.float32).
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW (dtype: torch.float32), where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        input_images = torch.stack([self.model.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.model.image_encoder(input_images)

        input_masks = [x["mask_inputs"] if "mask_inputs" in x and x["mask_inputs"] is not None else None for x in batched_input]
        input_masks = [self._pad(mask.float()) if mask is not None else None for mask in input_masks]

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
    
    def _pad(self, x: torch.Tensor) -> torch.Tensor:
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
            pred_iou_thresh: float = 0.88,
            stability_score_thresh: float = 0.95,
            stability_score_offset: float = 1.0,
            box_nms_thresh: float = 0.7,
            min_mask_region_area: int = 300
        ):
        return SamAutomaticMaskGenerator(model=self.model, 
                                          pred_iou_thresh=pred_iou_thresh,
                                          stability_score_thresh=stability_score_thresh,
                                          stability_score_offset=stability_score_offset,
                                          box_nms_thresh=box_nms_thresh,
                                          min_mask_region_area=min_mask_region_area)
        
    def save(self, fabric: L.Fabric, out_dir: str, name: str = "ckpt"):
        """
        Save the model checkpoint.
        
        Args:
            fabric (L.Fabric): The lightning fabric.
            out_dir (str): The output directory.
            name (str): The name of the checkpoint without .pth.
        """
        fabric.print(f"Saving checkpoint to {out_dir}")
        name = name + ".pth"
        state_dict = self.model.state_dict()
        if fabric.global_rank == 0:
            torch.save(state_dict, os.path.join(out_dir, name))