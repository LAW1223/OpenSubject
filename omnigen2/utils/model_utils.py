"""
Model utilities for handling dimension mismatches and model adaptations.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import logging
# Avoid circular import - import only when needed
# from ..models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel

if hasattr(torch.nn, 'RMSNorm'):
    from torch.nn import RMSNorm
else:
    try:
        from ...ops.triton.layer_norm import RMSNorm
    except ImportError:
        from torch.nn import RMSNorm

logger = logging.getLogger(__name__)


def check_and_adapt_text_feat_dim(
    model: "OmniGen2Transformer2DModel",
    yaml_config: Dict[str, Any],
    force_reinit: bool = False
) -> "OmniGen2Transformer2DModel":
    """
    Check if text_feat_dim in YAML config matches the loaded model config.
    If not, reinitialize the caption embedder layers with new dimensions.
    
    Args:
        model: The loaded OmniGen2Transformer2DModel
        yaml_config: The YAML configuration dictionary
        force_reinit: If True, force reinitialize even if dimensions match
    
    Returns:
        The model with potentially updated caption embedder layers
    """
    # Get text_feat_dim from YAML config
    yaml_text_feat_dim = yaml_config.get('model', {}).get('arch_opt', {}).get('text_feat_dim')
    
    if yaml_text_feat_dim is None:
        logger.warning("text_feat_dim not found in YAML config, using model default")
        return model
    
    # Get text_feat_dim from loaded model config
    model_text_feat_dim = model.config.text_feat_dim
    
    logger.info(f"YAML config text_feat_dim: {yaml_text_feat_dim}")
    logger.info(f"Model config text_feat_dim: {model_text_feat_dim}")
    
    # Check if dimensions match
    if yaml_text_feat_dim == model_text_feat_dim and not force_reinit:
        logger.info("text_feat_dim dimensions match, no adaptation needed")
        return model
    
    # Dimensions don't match, need to reinitialize caption embedder
    logger.warning(
        f"text_feat_dim mismatch detected: YAML={yaml_text_feat_dim}, "
        f"Model={model_text_feat_dim}. Reinitializing caption embedder layers."
    )
    
    # Update model config
    model.config.text_feat_dim = yaml_text_feat_dim
    
    # Reinitialize caption embedder with new dimensions
    _reinitialize_caption_embedder(model, yaml_text_feat_dim)
    
    logger.info("Caption embedder successfully reinitialized with new dimensions")
    return model


def _reinitialize_caption_embedder(
    model: "OmniGen2Transformer2DModel", 
    new_text_feat_dim: int
) -> None:
    """
    Reinitialize the caption embedder layers with new text feature dimensions.
    
    Args:
        model: The OmniGen2Transformer2DModel to modify
        new_text_feat_dim: The new text feature dimension
    """
    # Access the time_caption_embed module
    time_caption_embed = model.time_caption_embed
    
    # Get current configuration
    hidden_size = model.config.hidden_size
    norm_eps = model.config.norm_eps
    
    # Create new caption embedder with correct dimensions
    new_caption_embedder = nn.Sequential(
        RMSNorm(new_text_feat_dim, eps=norm_eps),
        nn.Linear(new_text_feat_dim, hidden_size, bias=True),
    )
    
    # Initialize weights properly
    _initialize_caption_embedder_weights(new_caption_embedder)
    
    # Replace the old caption embedder
    time_caption_embed.caption_embedder = new_caption_embedder
    
    logger.info(
        f"Reinitialized caption_embedder: "
        f"RMSNorm({new_text_feat_dim}) -> Linear({new_text_feat_dim}, {hidden_size})"
    )


def _initialize_caption_embedder_weights(caption_embedder: nn.Sequential) -> None:
    """
    Initialize the weights of the caption embedder following the original initialization scheme.
    
    Args:
        caption_embedder: The caption embedder sequential module
    """
    # Initialize Linear layer weights and bias
    linear_layer = caption_embedder[1]  # Second element is the Linear layer
    nn.init.trunc_normal_(linear_layer.weight, std=0.02)
    nn.init.zeros_(linear_layer.bias)
    
    logger.debug("Caption embedder weights initialized")


def create_dimension_adapter(
    input_dim: int, 
    output_dim: int, 
    hidden_dim: Optional[int] = None,
    dropout: float = 0.0
) -> nn.Module:
    """
    Create a dimension adapter to transform features from input_dim to output_dim.
    This can be used as an alternative to reinitializing the caption embedder.
    
    Args:
        input_dim: Input feature dimension
        output_dim: Output feature dimension  
        hidden_dim: Hidden dimension for the adapter (if None, uses simple linear projection)
        dropout: Dropout probability
    
    Returns:
        A neural network module that transforms features
    """
    if hidden_dim is None:
        # Simple linear projection
        adapter = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=True),
            nn.LayerNorm(output_dim) if output_dim > input_dim else nn.Identity()
        )
    else:
        # Two-layer MLP adapter
        adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, output_dim, bias=True),
            nn.LayerNorm(output_dim)
        )
    
    # Initialize weights
    for module in adapter.modules():
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    return adapter


def get_model_text_feat_dim(model: "OmniGen2Transformer2DModel") -> int:
    """
    Get the text feature dimension from a loaded model.
    
    Args:
        model: The OmniGen2Transformer2DModel
        
    Returns:
        The text feature dimension
    """
    return model.config.text_feat_dim


def validate_model_config_consistency(
    model: "OmniGen2Transformer2DModel",
    yaml_config: Dict[str, Any]
) -> Dict[str, bool]:
    """
    Validate consistency between model config and YAML config for key parameters.
    
    Args:
        model: The loaded model
        yaml_config: The YAML configuration
        
    Returns:
        Dictionary with validation results for each parameter
    """
    results = {}
    
    # Check text_feat_dim
    yaml_text_feat_dim = yaml_config.get('model', {}).get('arch_opt', {}).get('text_feat_dim')
    model_text_feat_dim = model.config.text_feat_dim
    results['text_feat_dim'] = (yaml_text_feat_dim == model_text_feat_dim)
    
    # Check other important dimensions
    arch_opt = yaml_config.get('model', {}).get('arch_opt', {})
    
    dimension_checks = [
        'hidden_size', 'num_layers', 'num_attention_heads', 
        'num_kv_heads', 'patch_size', 'in_channels'
    ]
    
    for param in dimension_checks:
        yaml_val = arch_opt.get(param)
        model_val = getattr(model.config, param, None)
        if yaml_val is not None and model_val is not None:
            results[param] = (yaml_val == model_val)
    
    return results