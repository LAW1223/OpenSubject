"""
OmniGen2 utilities module.
"""

from .model_utils import (
    check_and_adapt_text_feat_dim,
    validate_model_config_consistency,
    get_model_text_feat_dim,
    create_dimension_adapter
)

__all__ = [
    'check_and_adapt_text_feat_dim',
    'validate_model_config_consistency', 
    'get_model_text_feat_dim',
    'create_dimension_adapter'
]
