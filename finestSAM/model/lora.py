import math
import torch
from torch import nn
from typing import Dict, Optional, Tuple


class _LoRABlock(nn.Module):
    """
    Single LoRA adapter block (A + B matrices).
    Input -> Dropout -> A -> B -> Scaling

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        r (int): LoRA rank.
        alpha (float): LoRA alpha value.
        dropout (float): Dropout rate.
        bias (bool): Whether to include bias in the linear layers.
        device (torch.device, optional): Device to store the module on.
        dtype (torch.dtype, optional): Data type of the module.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        alpha: float,
        dropout: float,
        bias: bool,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()
        self.r = r
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=bias)
        
        if device or dtype:
            self.lora_A.to(device=device, dtype=dtype)
            self.lora_B.to(device=device, dtype=dtype)
            
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        if self.lora_B.bias is not None:
            nn.init.zeros_(self.lora_B.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


class LoRALinear(nn.Module):
    """
    Implements Low-Rank Adaptation (LoRA) for a frozen Linear layer.

    Args:
        layer (nn.Linear): The original Linear layer to be adapted.
        r (int): LoRA rank.
        lora_alpha (float): LoRA alpha value.
        lora_dropout (float): Dropout rate.
        lora_bias (bool): Whether to include bias in the linear layers.
    """
    def __init__(
        self,
        layer: nn.Linear,
        r: int,
        alpha: float,
        dropout: float,
        bias: bool,
    ):
        super().__init__()
        if r < 0:
            raise ValueError("LoRA rank must be non-negative")
        
        self.layer = layer
        self.r = r
        
        # Freeze original layer
        for p in self.layer.parameters():
            p.requires_grad = False

        if r > 0:
            self.adapter = _LoRABlock(
                in_features=layer.in_features,
                out_features=layer.out_features,
                r=r,
                alpha=alpha,
                dropout=dropout,
                bias=bias,
                device=layer.weight.device,
                dtype=layer.weight.dtype
            )
        else:
            self.adapter = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layer(x)
        if self.adapter:
            out = out + self.adapter(x)
        return out


class LoRA_QKV(nn.Module):
    """
    Implements Low-Rank Adaptation (LoRA) specifically for the SAM Image Encoder qkv layer.
    Splits the adaptation into separate Q, K, and V branches.

    Args:
        qkv_layer (nn.Linear): The original QKV layer to be adapted.
        r (int): LoRA rank.
        alpha (float): LoRA alpha value.
        dropout (float): Dropout rate.
        bias (bool): Whether to include bias in the linear layers.
        enable_q (bool): Whether to enable Q adaptation.
        enable_k (bool): Whether to enable K adaptation.
        enable_v (bool): Whether to enable V adaptation.
    """
    def __init__(
        self,
        qkv_layer: nn.Linear,
        r: int,
        alpha: float,
        dropout: float,
        bias: bool,
        enable_q: bool,
        enable_k: bool,
        enable_v: bool,
    ):
        super().__init__()
        self.qkv = qkv_layer
        self.dim = qkv_layer.in_features
        self.r = r
        
        assert qkv_layer.out_features == 3 * self.dim, "QKV layer output features must be 3 * Input features"

        # Freeze original layer
        for p in self.qkv.parameters():
            p.requires_grad = False

        self.adapters = nn.ModuleDict()
        
        if r > 0:
            common_kwargs = {
                "in_features": self.dim,
                "out_features": self.dim,
                "r": r,
                "alpha": alpha,
                "dropout": dropout,
                "bias": bias,
                "device": qkv_layer.weight.device,
                "dtype": qkv_layer.weight.dtype,
            }
            
            if enable_q:
                self.adapters['q'] = _LoRABlock(**common_kwargs)
            if enable_k:
                self.adapters['k'] = _LoRABlock(**common_kwargs)
            if enable_v:
                self.adapters['v'] = _LoRABlock(**common_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original forward pass (B, H, W, 3*C)
        qkv = self.qkv(x)
        
        if self.r > 0 and self.adapters:
            # Add residuals to Q, K, V slices if adapters exist
            if 'q' in self.adapters:
                qkv[..., :self.dim] += self.adapters['q'](x)
            if 'k' in self.adapters:
                qkv[..., self.dim:2*self.dim] += self.adapters['k'](x)
            if 'v' in self.adapters:
                qkv[..., -self.dim:] += self.adapters['v'](x)
            
        return qkv


def _wrap_linear(module: nn.Module, name: str, **kwargs):
    """
    Replaces a specific child Linear layer within a module with a LoRALinear or LoRA_QKV wrapper.
    """
    # Prevent creating nested LoRA layers if 'module' is already a LoRA wrapper
    if isinstance(module, (LoRALinear, LoRA_QKV)):
        return

    child = getattr(module, name, None)
    if child is None:
        return
    
    # Check if already wrapped or not a Linear layer
    if isinstance(child, (LoRALinear, LoRA_QKV)) or not isinstance(child, nn.Linear):
        return

    qkv_config = kwargs.pop('qkv_config', None)
    
    if qkv_config is not None:
        new_module = LoRA_QKV(
            child, 
            enable_q=qkv_config.get("q_proj", False),
            enable_k=qkv_config.get("k_proj", False),
            enable_v=qkv_config.get("v_proj", False),
            **kwargs
        )
    else:
        new_module = LoRALinear(child, **kwargs)
        
    setattr(module, name, new_module)


def _wrap_mlp_layers(mlp: nn.Module, **kwargs):
    """Wraps linear layers in an MLP's ModuleList."""
    if not hasattr(mlp, "layers"):
        return
    for idx, layer in enumerate(mlp.layers):
        if isinstance(layer, LoRALinear) or not isinstance(layer, nn.Linear):
            continue
        mlp.layers[idx] = LoRALinear(layer, **kwargs)


def _get_lora_params(cfg) -> Dict:
    """Helper to extract common LoRA parameters from config."""
    return {
        "r": cfg.lora_r,
        "alpha": cfg.lora_alpha,
        "dropout": cfg.lora_dropout,
        "bias": cfg.lora_bias,
    }


def inject_lora_sam(sam_model: nn.Module, lora_cfg) -> nn.Module:
    """
    Injects LoRA adapters into a Segment Anything Model (SAM) architecture.
    """

    # --- Image Encoder (ViT) ---
    if lora_cfg.encoder.enabled:
        # FREEZE encoder
        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False
            
        params = _get_lora_params(lora_cfg.encoder)
        targets = lora_cfg.encoder.lora_targets

        if params["r"] > 0:
            for module in sam_model.image_encoder.modules():
                # ViT Attention (Fused QKV)
                if hasattr(module, "qkv") and (targets.get("q_proj", False) or targets.get("k_proj", False) or targets.get("v_proj", False)):
                    _wrap_linear(module, "qkv", qkv_config=targets, **params)
                
                if hasattr(module, "out_proj") and targets.get("out_proj", False):
                    _wrap_linear(module, "out_proj", **params)

                # ViT MLP
                for name in ("lin1", "lin2"):
                    if targets.get(f"mlp_{name}", False):
                         _wrap_linear(module, name, **params)

    # --- Mask Decoder ---
    if lora_cfg.decoder.enabled:
        # FREEZE decoder
        for param in sam_model.mask_decoder.parameters():
            param.requires_grad = False

        params = _get_lora_params(lora_cfg.decoder)
        targets = lora_cfg.decoder.lora_targets

        if params["r"] > 0:
            for module in sam_model.mask_decoder.modules():
                # TwoWayTransformer Attention
                for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                    if targets.get(name, False):
                        _wrap_linear(module, name, **params)
                
                # MLP
                for name in ("lin1", "lin2"):
                     if targets.get(f"mlp_{name}", False):
                        _wrap_linear(module, name, **params)

            # Mask Decoder Specialized Layers (Hypernetworks & IoU Head)
            if targets.get("hypernet_mlp", False) and hasattr(sam_model.mask_decoder, "output_hypernetworks_mlps"):
                for mlp in sam_model.mask_decoder.output_hypernetworks_mlps:
                    _wrap_mlp_layers(mlp, **params)
            
            if targets.get("iou_head_mlp", False) and hasattr(sam_model.mask_decoder, "iou_prediction_head"):
                _wrap_mlp_layers(sam_model.mask_decoder.iou_prediction_head, **params)

    return sam_model