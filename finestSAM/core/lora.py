import math
import torch
from torch import nn


class LoRALinear(nn.Module):
    """
    Implements Low-Rank Adaptation (LoRA) for a frozen Linear layer.
    
    This module freezes the weights of the provided layer and adds a parallel 
    low-rank branch optimized during training. The forward pass is computed as:
    output = layer(x) + (alpha / r) * B(A(x)).

    Args:
        layer (nn.Linear): The original linear layer to wrap and freeze.
        r (int): The rank of the low-rank approximation (dimension of the latent space). 
                 If 0, LoRA is disabled and the layer acts as a standard frozen linear layer.
        lora_alpha (float): Scaling factor for the LoRA weights. This parameter acts 
                            similarly to a specific learning rate for the adapter. 
                            When changing 'r', scale 'lora_alpha' proportionally to maintain training stability.
        lora_dropout (float): Dropout probability applied to the input of the LoRA branch.
        lora_bias (bool): If True, enables a learnable bias in the projection matrix B.
    """

    def __init__(
        self,
        layer: nn.Linear,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        lora_bias: bool = False,
    ):
        super().__init__()
        if r < 0:
            raise ValueError("LoRA rank must be non-negative")

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 0.0
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout and lora_dropout > 0 else nn.Identity()

        # Keep the original Linear layer but freeze its parameters.
        self.layer = layer
        for p in self.layer.parameters():
            p.requires_grad = False

        if r > 0:
            # Extract device and dtype from the layer to ensure consistency.
            device = self.layer.weight.device
            dtype = self.layer.weight.dtype
            
            # Create LoRA layers A and B.
            # Matrix A projects down to rank 'r', Matrix B projects up to output dimension.
            self.lora_A = nn.Linear(self.layer.in_features, r, bias=False)
            self.lora_B = nn.Linear(r, self.layer.out_features, bias=lora_bias)
            
            # Align Device/Dtype.
            self.lora_A.to(device=device, dtype=dtype)
            self.lora_B.to(device=device, dtype=dtype)

            # Initialize LoRA weights.
            self.reset_parameters()
        else:
            # No-op mode: no adapter if r=0.
            self.lora_A = None
            self.lora_B = None

    def reset_parameters(self) -> None:
        """
        Initializes the LoRA weights to ensure the identity function at the start of training.
        """

        if self.r == 0:
            return
        
        """
        - Matrix A is initialized with Kaiming Uniform. This is required to preserve 
          the variance of the input features through the projection.
        - Matrix B is initialized to zeros. This ensures that initially, the output 
          of the LoRA branch is zero.
        
        Consequently, the total output is exactly the same as the pre-trained layer 
        at initialization (W_layer + 0). If we initialized A to zero as well, gradients 
        with respect to B would be null, creating a saddle point that hinders learning.
        """
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        if self.lora_B.bias is not None:
            nn.init.zeros_(self.lora_B.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward pass combining the frozen layer and the learnable LoRA branch.
        """
        # Frozen layer output
        layer_out = self.layer(x)
        if self.r == 0:
            return layer_out

        # LoRA branch: Input -> Dropout -> A -> B -> Scaling
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        
        # Combine layer output with residual LoRA output
        return layer_out + lora_out


def _wrap_linear(module: nn.Module, name: str, *, r: int, alpha: float, dropout: float, bias: bool):
    """
    Replaces a specific child Linear layer within a module with a LoRALinear wrapper.

    Args:
        module (nn.Module): The parent module containing the layer to replace.
        name (str): The name of the attribute (child layer) to wrap.
        r (int): LoRA rank.
        alpha (float): LoRA scaling factor.
        dropout (float): LoRA dropout probability.
        bias (bool): Whether to enable bias in the LoRA branch.
    
    Returns:
        None
    """
    child = getattr(module, name, None)
    # Check if the child exists, is already wrapped, or is not a Linear layer.
    if child is None or isinstance(child, LoRALinear) or not isinstance(child, nn.Linear):
        return
    setattr(module, name, LoRALinear(child, r=r, lora_alpha=alpha, lora_dropout=dropout, lora_bias=bias))


def _wrap_mlp_layers(mlp: nn.Module, *, r: int, alpha: float, dropout: float, bias: bool):
    """
    Iterates through a ModuleList named 'layers' within an MLP and wraps valid Linear layers.

    Args:
        mlp (nn.Module): The MLP module containing a '.layers' ModuleList.
        r (int): LoRA rank.
        alpha (float): LoRA scaling factor.
        dropout (float): LoRA dropout probability.
        bias (bool): Whether to enable bias in the LoRA branch.
    """
    # Check if the MLP has a 'layers' attribute.
    if not hasattr(mlp, "layers"):
        return
    for idx, layer in enumerate(mlp.layers):
        # Skip if the layer is already wrapped or not a Linear layer.
        if isinstance(layer, LoRALinear) or not isinstance(layer, nn.Linear):
            continue
        mlp.layers[idx] = LoRALinear(layer, r=r, lora_alpha=alpha, lora_dropout=dropout, lora_bias=bias)


def inject_lora_sam(sam_model: nn.Module, lora_cfg) -> nn.Module:
    """
    Injects LoRA adapters into a Segment Anything Model (SAM) architecture based on the configuration.

    This function traverses the SAM Image Encoder (ViT) and Mask Decoder, replacing specific 
    linear projections with LoRALinear layers as specified in the `lora_cfg`.

    Args:
        sam_model (nn.Module): The SAM model instance to modify.
        lora_cfg: A configuration object containing settings for 'encoder' and 'decoder'.

    Returns:
        nn.Module: The modified SAM model with injected LoRA adapters.
    """

    # --- Image Encoder (ViT) ---
    if lora_cfg.encoder.enabled:
        cfg = lora_cfg.encoder
        r = cfg.lora_r
        alpha = cfg.lora_alpha
        dropout = cfg.lora_dropout
        bias = cfg.lora_bias
        targets = cfg.lora_targets

        if r > 0:
            for module in sam_model.image_encoder.modules():
                # ViT Attention: qkv (fused) and proj (output)
                if hasattr(module, "qkv") and targets.get("qkv", False):
                    _wrap_linear(module, "qkv", r=r, alpha=alpha, dropout=dropout, bias=bias)
                
                if hasattr(module, "proj") and targets.get("proj", False):
                    _wrap_linear(module, "proj", r=r, alpha=alpha, dropout=dropout, bias=bias)

                # ViT MLP: lin1, lin2
                for name in ("lin1", "lin2"):
                    if targets.get(f"mlp_{name}", False):
                         _wrap_linear(module, name, r=r, alpha=alpha, dropout=dropout, bias=bias)


    # --- Mask Decoder ---
    if lora_cfg.decoder.enabled:
        cfg = lora_cfg.decoder
        r = cfg.lora_r
        alpha = cfg.lora_alpha
        dropout = cfg.lora_dropout
        bias = cfg.lora_bias
        targets = cfg.lora_targets

        if r > 0:
            for module in sam_model.mask_decoder.modules():
                # TwoWayTransformer Attention: q_proj, k_proj, v_proj, out_proj
                for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                    if targets.get(name, False):
                        _wrap_linear(module, name, r=r, alpha=alpha, dropout=dropout, bias=bias)
                
                # MLP
                for name in ("lin1", "lin2"):
                     if targets.get(f"mlp_{name}", False):
                        _wrap_linear(module, name, r=r, alpha=alpha, dropout=dropout, bias=bias)

            # Mask Decoder Specialized Layers (Hypernetworks & IoU Head)
            if targets.get("hypernet_mlp", False) and hasattr(sam_model.mask_decoder, "output_hypernetworks_mlps"):
                for mlp in sam_model.mask_decoder.output_hypernetworks_mlps:
                    _wrap_mlp_layers(mlp, r=r, alpha=alpha, dropout=dropout, bias=bias)
            
            if targets.get("iou_head_mlp", False) and hasattr(sam_model.mask_decoder, "iou_prediction_head"):
                _wrap_mlp_layers(sam_model.mask_decoder.iou_prediction_head, r=r, alpha=alpha, dropout=dropout, bias=bias)

    return sam_model