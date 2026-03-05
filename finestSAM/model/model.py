import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from box import Box
from typing import Any, Dict, List, Optional
from .segment_anything import sam_model_registry
from .segment_anything import SamPredictor, SamAutomaticMaskGenerator
from .lora import inject_lora_sam


class FinestSAM(nn.Module):
    def __init__(self, cfg: Box):
        super().__init__()
        self.cfg = cfg
        self.model = None

    def setup(self):
        """Set up the model."""
        ckpt_path = os.path.join(self.cfg.sav_dir, self.cfg.model.checkpoint)
        
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"\n\nERROR: Checkpoint file not found at '{ckpt_path}'.\n"
                "Please ensure the file path is correct.\n"
            )

        try:
            state_dict = torch.load(ckpt_path, map_location="cpu")
        except Exception as e:
            raise RuntimeError(f"ERROR: Failed to load state dict from {ckpt_path}. Reason: {e}") from e

        if self._is_lora_checkpoint(state_dict):
            self._init_from_lora_weights(state_dict)
        else:
            self._init_from_base_weights(ckpt_path)

    def _init_from_base_weights(self, checkpoint_path: str):
        """Load official SAM weights from checkpoint and apply LoRA if specified."""
        # Build Base SAM with weights
        self.model = self._build_base_model(checkpoint=checkpoint_path)
        
        # Freeze parameters
        self._apply_freezing()
        
        # Inject LoRA
        lora_cfg = getattr(self.cfg.model_layer, "LORA", None)
        if lora_cfg:
            self.model = inject_lora_sam(self.model, lora_cfg=lora_cfg)

    def _init_from_lora_weights(self, state_dict: Dict[str, Any]):
        """Load custom LoRA checkpoint."""
        # Build Empty Base SAM
        self.model = self._build_base_model(checkpoint=None)
        
        # Freeze parameters
        self._apply_freezing()
        
        # Inject LoRA structure (Required for keys to match)
        lora_cfg = getattr(self.cfg.model_layer, "LORA", None)
        if lora_cfg:
            self.model = inject_lora_sam(self.model, lora_cfg=lora_cfg)

        # Load weights into the final structure
        self._load(state_dict)

    def _is_lora_checkpoint(self, state_dict: Dict[str, Any]) -> bool:
        """Checks if the state dict contains LoRA-specific keys."""
        lora_markers = [".adapter.", ".adapters.", "lora_A", "lora_B", ".layer.", ".qkv.qkv."]
        
        for key in state_dict.keys():
            if any(marker in key for marker in lora_markers):
                return True
        return False

    def _load(self, state_dict: Dict[str, Any]):
        """Loads the state dict into the model."""
        try:
            self.model.load_state_dict(state_dict, strict=False)
        except RuntimeError as e:
            if "size mismatch" in str(e):
                raise RuntimeError(
                    f"\n\nERROR: Size mismatch loading LoRA checkpoint '{self.cfg.model.checkpoint}'.\n"
                    "Ensure 'img_size' and LoRA targets match the training config.\n"
                ) from e
            raise e
        except KeyError as e:
            raise KeyError(f"\n\nERROR: Key mismatch. Check model configuration.\n") from e

    def _build_base_model(self, checkpoint: Optional[str] = None):
        """
        Build the base SAM model without LoRA.

        Args:
            checkpoint (str, optional): Path to the checkpoint file.
                                       If None, builds model without loading weights.
        Returns:
            nn.Module: The SAM model.
        """
        img_size = self.cfg.model.get("img_size", 1024)
        pixel_mean = self.cfg.model.get("pixel_mean", None)
        pixel_std = self.cfg.model.get("pixel_std", None)
        
        try:
            return sam_model_registry[self.cfg.model.type](
                checkpoint=checkpoint,
                image_size=img_size,
                pixel_mean=pixel_mean,
                pixel_std=pixel_std,
            )
        except KeyError as e:
            raise KeyError(
                f"Invalid model type '{self.cfg.model.type}'. Available: {list(sam_model_registry.keys())}"
            ) from e
    
        
    def _apply_freezing(self):
        """Apply freezing to model layers based on configuration."""
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
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Args:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model (dtype: torch.float32).
                (H and W must have the maximum size of self.model.image_encoder.img_size)
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
                (The largest dimension must be at most 1/4 of the largest dimension of the input image)
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

        padh = self.model.image_encoder.img_size // 4 - h
        padw = self.model.image_encoder.img_size // 4 - w
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
        min_mask_region_area: int = 0
    ):
        """
        Get the automatic mask generator.
        Generates a grid of point prompts over the image, then filters
        low quality and duplicate masks. 

        Args:
          pred_iou_thresh (float): A filtering threshold in [0,1], using the
            model's predicted mask quality.
          stability_score_thresh (float): A filtering threshold in [0,1], using
            the stability of the mask under changes to the cutoff used to binarize
            the model's mask predictions.
          stability_score_offset (float): The amount to shift the cutoff when
            calculated the stability score.
          box_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks.
          min_mask_region_area (int): If >0, postprocessing will be applied
            to remove disconnected regions and holes in masks with area smaller
            than min_mask_region_area. Requires opencv.
        Returns:
            SamAutomaticMaskGenerator: The automatic mask generator with the specified parameters.
        """

        return SamAutomaticMaskGenerator(
            model=self.model, 
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            stability_score_offset=stability_score_offset,
            box_nms_thresh=box_nms_thresh,
            min_mask_region_area=min_mask_region_area
        )
        
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