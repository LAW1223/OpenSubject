import warnings
import itertools
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

import torch
import torch.nn as nn

from einops import rearrange

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.attention_processor import Attention
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin

from ..attention_processor import OmniGen2AttnProcessorFlash2Varlen, OmniGen2AttnProcessor
from .repo import OmniGen2RotaryPosEmbed
from .block_lumina2 import LuminaLayerNormContinuous, LuminaRMSNormZero, LuminaFeedForward, Lumina2CombinedTimestepCaptionEmbedding

from ...utils.import_utils import is_triton_available, is_flash_attn_available
from ...utils.teacache_util import TeaCacheParams

if is_triton_available():
    from ...ops.triton.layer_norm import RMSNorm
else:
    from torch.nn import RMSNorm

from ...taylorseer_utils import derivative_approximation, taylor_formula, taylor_cache_init
from ...cache_functions import cache_init, cal_type

logger = logging.get_logger(__name__)

class OmniGen2TransformerBlock(nn.Module):
    """
    Transformer block for OmniGen2 model.
    
    This block implements a transformer layer with:
    - Multi-head attention with flash attention
    - Feed-forward network with SwiGLU activation
    - RMS normalization
    - Optional modulation for conditional generation
    
    Args:
        dim: Dimension of the input and output tensors
        num_attention_heads: Number of attention heads
        num_kv_heads: Number of key-value heads
        multiple_of: Multiple of which the hidden dimension should be
        ffn_dim_multiplier: Multiplier for the feed-forward network dimension
        norm_eps: Epsilon value for normalization layers
        modulation: Whether to use modulation for conditional generation
        use_fused_rms_norm: Whether to use fused RMS normalization
        use_fused_swiglu: Whether to use fused SwiGLU activation
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        num_kv_heads: int,
        multiple_of: int,
        ffn_dim_multiplier: float,
        norm_eps: float,
        modulation: bool = True,
    ) -> None:
        """Initialize the transformer block."""
        super().__init__()
        self.head_dim = dim // num_attention_heads
        self.modulation = modulation

        try:
            processor = OmniGen2AttnProcessorFlash2Varlen()
        except ImportError:
            processor = OmniGen2AttnProcessor()

        # Initialize attention layer
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=dim // num_attention_heads,
            qk_norm="rms_norm",
            heads=num_attention_heads,
            kv_heads=num_kv_heads,
            eps=1e-5,
            bias=False,
            out_bias=False,
            processor=processor,
        )

        # Initialize feed-forward network
        self.feed_forward = LuminaFeedForward(
            dim=dim,
            inner_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier
        )

        # Initialize normalization layers
        if modulation:
            self.norm1 = LuminaRMSNormZero(
                embedding_dim=dim,
                norm_eps=norm_eps,
                norm_elementwise_affine=True
            )
        else:
            self.norm1 = RMSNorm(dim, eps=norm_eps)

        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        """
        Initialize the weights of the transformer block.
        
        Uses Xavier uniform initialization for linear layers and zero initialization for biases.
        """
        nn.init.xavier_uniform_(self.attn.to_q.weight)
        nn.init.xavier_uniform_(self.attn.to_k.weight)
        nn.init.xavier_uniform_(self.attn.to_v.weight)
        nn.init.xavier_uniform_(self.attn.to_out[0].weight)

        nn.init.xavier_uniform_(self.feed_forward.linear_1.weight)
        nn.init.xavier_uniform_(self.feed_forward.linear_2.weight)
        nn.init.xavier_uniform_(self.feed_forward.linear_3.weight)
        
        if self.modulation:
            nn.init.zeros_(self.norm1.linear.weight)
            nn.init.zeros_(self.norm1.linear.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the transformer block.

        Args:
            hidden_states: Input hidden states tensor
            attention_mask: Attention mask tensor
            image_rotary_emb: Rotary embeddings for image tokens
            temb: Optional timestep embedding tensor

        Returns:
            torch.Tensor: Output hidden states after transformer block processing
        """
        enable_taylorseer = getattr(self, 'enable_taylorseer', False)
        if enable_taylorseer:
            if self.modulation:
                if temb is None:
                    raise ValueError("temb must be provided when modulation is enabled")
                    
                if self.current['type'] == 'full':
                    self.current['module'] = 'total'
                    taylor_cache_init(cache_dic=self.cache_dic, current=self.current)

                    norm_hidden_states, gate_msa, scale_mlp, gate_mlp = self.norm1(hidden_states, temb)
                    attn_output = self.attn(
                        hidden_states=norm_hidden_states,
                        encoder_hidden_states=norm_hidden_states,
                        attention_mask=attention_mask,
                        image_rotary_emb=image_rotary_emb,
                    )
                    hidden_states = hidden_states + gate_msa.unsqueeze(1).tanh() * self.norm2(attn_output)
                    mlp_output = self.feed_forward(self.ffn_norm1(hidden_states) * (1 + scale_mlp.unsqueeze(1)))
                    hidden_states = hidden_states + gate_mlp.unsqueeze(1).tanh() * self.ffn_norm2(mlp_output)

                    derivative_approximation(cache_dic=self.cache_dic, current=self.current, feature=hidden_states)

                elif self.current['type'] == 'Taylor': 
                    self.current['module'] = 'total'
                    hidden_states = taylor_formula(cache_dic=self.cache_dic, current=self.current)
            else:
                norm_hidden_states = self.norm1(hidden_states)
                attn_output = self.attn(
                    hidden_states=norm_hidden_states,
                    encoder_hidden_states=norm_hidden_states,
                    attention_mask=attention_mask,
                    image_rotary_emb=image_rotary_emb,
                )
                hidden_states = hidden_states + self.norm2(attn_output)
                mlp_output = self.feed_forward(self.ffn_norm1(hidden_states))
                hidden_states = hidden_states + self.ffn_norm2(mlp_output)
        else:
            if self.modulation:
                if temb is None:
                    raise ValueError("temb must be provided when modulation is enabled")
                    
                norm_hidden_states, gate_msa, scale_mlp, gate_mlp = self.norm1(hidden_states, temb)
                attn_output = self.attn(
                    hidden_states=norm_hidden_states,
                    encoder_hidden_states=norm_hidden_states,
                    attention_mask=attention_mask,
                    image_rotary_emb=image_rotary_emb,
                )
                hidden_states = hidden_states + gate_msa.unsqueeze(1).tanh() * self.norm2(attn_output)
                mlp_output = self.feed_forward(self.ffn_norm1(hidden_states) * (1 + scale_mlp.unsqueeze(1)))
                hidden_states = hidden_states + gate_mlp.unsqueeze(1).tanh() * self.ffn_norm2(mlp_output)
            else:
                norm_hidden_states = self.norm1(hidden_states)
                attn_output = self.attn(
                    hidden_states=norm_hidden_states,
                    encoder_hidden_states=norm_hidden_states,
                    attention_mask=attention_mask,
                    image_rotary_emb=image_rotary_emb,
                )
                hidden_states = hidden_states + self.norm2(attn_output)
                mlp_output = self.feed_forward(self.ffn_norm1(hidden_states))
                hidden_states = hidden_states + self.ffn_norm2(mlp_output)

        return hidden_states


class OmniGen2Transformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    """
    OmniGen2 Transformer 2D Model.
    
    A transformer-based diffusion model for image generation with:
    - Patch-based image processing
    - Rotary position embeddings
    - Multi-head attention
    - Conditional generation support
    
    Args:
        patch_size: Size of image patches
        in_channels: Number of input channels
        out_channels: Number of output channels (defaults to in_channels)
        hidden_size: Size of hidden layers
        num_layers: Number of transformer layers
        num_refiner_layers: Number of refiner layers
        num_attention_heads: Number of attention heads
        num_kv_heads: Number of key-value heads
        multiple_of: Multiple of which the hidden dimension should be
        ffn_dim_multiplier: Multiplier for feed-forward network dimension
        norm_eps: Epsilon value for normalization layers
        axes_dim_rope: Dimensions for rotary position embeddings
        axes_lens: Lengths for rotary position embeddings
        text_feat_dim: Dimension of text features
        timestep_scale: Scale factor for timestep embeddings
        use_fused_rms_norm: Whether to use fused RMS normalization
        use_fused_swiglu: Whether to use fused SwiGLU activation
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["Omnigen2TransformerBlock"]
    _skip_layerwise_casting_patterns = ["x_embedder", "norm"]

    @register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 16,
        out_channels: Optional[int] = None,
        hidden_size: int = 2304,
        num_layers: int = 26,
        num_refiner_layers: int = 2,
        num_attention_heads: int = 24,
        num_kv_heads: int = 8,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        norm_eps: float = 1e-5,
        axes_dim_rope: Tuple[int, int, int] = (32, 32, 32),
        axes_lens: Tuple[int, int, int] = (300, 512, 512),
        text_feat_dim: int = 1024,
        timestep_scale: float = 1.0
    ) -> None:
        """Initialize the OmniGen2 transformer model."""
        super().__init__()

        # Validate configuration
        if (hidden_size // num_attention_heads) != sum(axes_dim_rope):
            raise ValueError(
                f"hidden_size // num_attention_heads ({hidden_size // num_attention_heads}) "
                f"must equal sum(axes_dim_rope) ({sum(axes_dim_rope)})"
            )
        
        self.out_channels = out_channels or in_channels

        # Initialize embeddings
        self.rope_embedder = OmniGen2RotaryPosEmbed(
            theta=10000,
            axes_dim=axes_dim_rope,
            axes_lens=axes_lens,
            patch_size=patch_size,
        )

        self.x_embedder = nn.Linear(
            in_features=patch_size * patch_size * in_channels,
            out_features=hidden_size,
        )

        self.ref_image_patch_embedder = nn.Linear(
            in_features=patch_size * patch_size * in_channels,
            out_features=hidden_size,
        )

        self.time_caption_embed = Lumina2CombinedTimestepCaptionEmbedding(
            hidden_size=hidden_size,
            text_feat_dim=text_feat_dim,
            norm_eps=norm_eps,
            timestep_scale=timestep_scale
        )

        # Initialize transformer blocks
        self.noise_refiner = nn.ModuleList([
            OmniGen2TransformerBlock(
                hidden_size,
                num_attention_heads,
                num_kv_heads,
                multiple_of,
                ffn_dim_multiplier,
                norm_eps,
                modulation=True
            )
            for _ in range(num_refiner_layers)
        ])

        self.ref_image_refiner = nn.ModuleList([
            OmniGen2TransformerBlock(
                hidden_size,
                num_attention_heads,
                num_kv_heads,
                multiple_of,
                ffn_dim_multiplier,
                norm_eps,
                modulation=True
            )
            for _ in range(num_refiner_layers)
        ])

        self.context_refiner = nn.ModuleList(
            [
                OmniGen2TransformerBlock(
                    hidden_size,
                    num_attention_heads,
                    num_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    modulation=False
                )
                for _ in range(num_refiner_layers)
            ]
        )

        # 3. Transformer blocks
        self.layers = nn.ModuleList(
            [
                OmniGen2TransformerBlock(
                    hidden_size,
                    num_attention_heads,
                    num_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    modulation=True
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = LuminaLayerNormContinuous(
            embedding_dim=hidden_size,
            conditioning_embedding_dim=min(hidden_size, 1024),
            elementwise_affine=False,
            eps=1e-6,
            bias=True,
            out_dim=patch_size * patch_size * self.out_channels
        )
        
        # Add learnable embeddings to distinguish different images
        self.image_index_embedding = nn.Parameter(torch.randn(5, hidden_size)) # support max 5 ref images

        self.gradient_checkpointing = False

        self.initialize_weights()

        # TeaCache settings
        self.enable_teacache = False
        self.teacache_rel_l1_thresh = 0.05
        self.teacache_params = TeaCacheParams()

        coefficients = [-5.48259225, 11.48772289, -4.47407401, 2.47730926, -0.03316487]
        self.rescale_func = np.poly1d(coefficients)

    def initialize_weights(self) -> None:
        """
        Initialize the weights of the model.
        
        Uses Xavier uniform initialization for linear layers.
        """
        nn.init.xavier_uniform_(self.x_embedder.weight)
        nn.init.constant_(self.x_embedder.bias, 0.0)

        nn.init.xavier_uniform_(self.ref_image_patch_embedder.weight)
        nn.init.constant_(self.ref_image_patch_embedder.bias, 0.0)

        nn.init.zeros_(self.norm_out.linear_1.weight)
        nn.init.zeros_(self.norm_out.linear_1.bias)
        nn.init.zeros_(self.norm_out.linear_2.weight)
        nn.init.zeros_(self.norm_out.linear_2.bias)
        
        nn.init.normal_(self.image_index_embedding, std=0.02)

    def img_patch_embed_and_refine(
        self,
        hidden_states,                # 主图像的隐藏状态，形状: [batch_size, patch_num, channels]，表示待生成图像的补丁序列
        ref_image_hidden_states,      # 参考图像的隐藏状态，形状: [batch_size, total_ref_patch_num, channels]，表示用作条件的参考图像补丁序列
        padded_img_mask,             # 主图像的填充掩码，形状: [batch_size, max_patch_num]，True表示有效补丁，False表示填充补丁
        padded_ref_img_mask,         # 参考图像的填充掩码，形状: [batch_size, max_ref_patch_num]，True表示有效补丁，False表示填充补丁
        noise_rotary_emb,            # 主图像的旋转位置编码，形状: [batch_size, max_patch_num, dim]，用于空间位置编码
        ref_img_rotary_emb,          # 参考图像的旋转位置编码，形状: [batch_size, max_ref_patch_num, dim]，用于空间位置编码
        l_effective_ref_img_len,     # 每个样本中每个参考图像的有效长度列表，格式: [[ref1_len, ref2_len, ...], ...]
        l_effective_img_len,         # 每个样本中主图像的有效长度列表，格式: [img_len1, img_len2, ...]
        temb                         # 时间步嵌入，形状: [batch_size, hidden_size]，用于扩散过程的时间条件
    ):
        """
        图像补丁嵌入和精化函数
        
        该函数是OmniGen2模型的核心组件之一，负责处理主图像和参考图像的补丁嵌入、
        位置编码添加以及特征精化。主要步骤包括：
        1. 将主图像和参考图像转换为补丁嵌入
        2. 为参考图像添加索引嵌入以区分不同的参考图像
        3. 使用noise_refiner精化主图像特征
        4. 使用ref_image_refiner精化参考图像特征
        5. 将精化后的参考图像和主图像合并为统一序列
        
        Returns:
            combined_img_hidden_states: 合并后的图像隐藏状态，形状: [batch_size, max_combined_len, hidden_size]
                                       序列顺序: [参考图像1, 参考图像2, ..., 主图像]
        """
        batch_size = len(hidden_states)
        # 计算每个样本中参考图像和主图像的总长度，用于确定合并后序列的最大长度
        max_combined_img_len = max([img_len + sum(ref_img_len) for img_len, ref_img_len in zip(l_effective_img_len, l_effective_ref_img_len)])
    
        # 步骤1: 将图像补丁转换为嵌入向量
        hidden_states = self.x_embedder(hidden_states)                           # 主图像补丁嵌入
        ref_image_hidden_states = self.ref_image_patch_embedder(ref_image_hidden_states)  # 参考图像补丁嵌入
        
        # 步骤2: 为参考图像添加索引嵌入，用于区分不同的参考图像
        # 这样模型可以学习到第1个参考图像、第2个参考图像等的概念
        for i in range(batch_size):
            shift = 0  # 当前样本中参考图像序列的偏移量
            for j, ref_img_len in enumerate(l_effective_ref_img_len[i]):
                # 为第j个参考图像的所有补丁添加对应的索引嵌入
                ref_image_hidden_states[i, shift:shift + ref_img_len, :] = ref_image_hidden_states[i, shift:shift + ref_img_len, :] + self.image_index_embedding[j]
                shift += ref_img_len

        # 步骤3: 使用noise_refiner精化主图像特征
        # 这些层专门用于处理带噪声的主图像，提升特征质量
        for layer in self.noise_refiner:
            hidden_states = layer(hidden_states, padded_img_mask, noise_rotary_emb, temb)

        # 步骤4: 准备批处理格式来精化参考图像
        # 将序列格式的参考图像转换为批处理格式，以便并行处理每个参考图像
        flat_l_effective_ref_img_len = list(itertools.chain(*l_effective_ref_img_len))  # 展平参考图像长度列表
        num_ref_images = len(flat_l_effective_ref_img_len)                              # 总参考图像数量
        max_ref_img_len = max(flat_l_effective_ref_img_len)                             # 最长参考图像的长度

        # 创建批处理张量：将每个参考图像作为批次中的一个样本
        batch_ref_img_mask = ref_image_hidden_states.new_zeros(num_ref_images, max_ref_img_len, dtype=torch.bool)
        batch_ref_image_hidden_states = ref_image_hidden_states.new_zeros(num_ref_images, max_ref_img_len, self.config.hidden_size)
        batch_ref_img_rotary_emb = hidden_states.new_zeros(num_ref_images, max_ref_img_len, ref_img_rotary_emb.shape[-1], dtype=ref_img_rotary_emb.dtype)
        batch_temb = temb.new_zeros(num_ref_images, *temb.shape[1:], dtype=temb.dtype)

        # 将序列格式的参考图像重组为批处理格式
        idx = 0  # 当前处理的参考图像在批次中的索引
        for i in range(batch_size):
            shift = 0  # 当前样本中参考图像序列的偏移量
            for ref_img_len in l_effective_ref_img_len[i]:
                # 设置掩码：标记有效的补丁位置
                batch_ref_img_mask[idx, :ref_img_len] = True
                # 复制隐藏状态
                batch_ref_image_hidden_states[idx, :ref_img_len] = ref_image_hidden_states[i, shift:shift + ref_img_len]
                # 复制旋转位置编码
                batch_ref_img_rotary_emb[idx, :ref_img_len] = ref_img_rotary_emb[i, shift:shift + ref_img_len]
                # 复制时间嵌入（每个参考图像使用对应样本的时间嵌入）
                batch_temb[idx] = temb[i]
                shift += ref_img_len
                idx += 1

        # 步骤5: 使用ref_image_refiner精化参考图像特征
        # 这些层专门用于处理参考图像，提升其作为条件信息的质量
        for layer in self.ref_image_refiner:
            batch_ref_image_hidden_states = layer(batch_ref_image_hidden_states, batch_ref_img_mask, batch_ref_img_rotary_emb, batch_temb)

        # 步骤6: 将批处理格式的参考图像转换回序列格式
        idx = 0
        for i in range(batch_size):
            shift = 0
            for ref_img_len in l_effective_ref_img_len[i]:
                # 将精化后的参考图像特征复制回原始序列格式
                ref_image_hidden_states[i, shift:shift + ref_img_len] = batch_ref_image_hidden_states[idx, :ref_img_len]
                shift += ref_img_len
                idx += 1
            
        # 步骤7: 合并参考图像和主图像为统一序列
        # 创建合并后的张量，序列顺序为：[参考图像1, 参考图像2, ..., 主图像]
        combined_img_hidden_states = hidden_states.new_zeros(batch_size, max_combined_img_len, self.config.hidden_size)
        for i, (ref_img_len, img_len) in enumerate(zip(l_effective_ref_img_len, l_effective_img_len)):
            # 先放置所有参考图像
            combined_img_hidden_states[i, :sum(ref_img_len)] = ref_image_hidden_states[i, :sum(ref_img_len)]
            # 再放置主图像
            combined_img_hidden_states[i, sum(ref_img_len):sum(ref_img_len) + img_len] = hidden_states[i, :img_len]

        return combined_img_hidden_states

    def flat_and_pad_to_seq(self, hidden_states, ref_image_hidden_states):
        batch_size = len(hidden_states)
        p = self.config.patch_size
        device = hidden_states[0].device

        img_sizes = [(img.size(1), img.size(2)) for img in hidden_states]
        l_effective_img_len = [(H // p) * (W // p) for (H, W) in img_sizes]

        if ref_image_hidden_states is not None:
            ref_img_sizes = [[(img.size(1), img.size(2)) for img in imgs] if imgs is not None else None for imgs in ref_image_hidden_states]
            l_effective_ref_img_len = [[(ref_img_size[0] // p) * (ref_img_size[1] // p) for ref_img_size in _ref_img_sizes] if _ref_img_sizes is not None else [0] for _ref_img_sizes in ref_img_sizes]
        else:
            ref_img_sizes = [None for _ in range(batch_size)]
            l_effective_ref_img_len = [[0] for _ in range(batch_size)]

        max_ref_img_len = max([sum(ref_img_len) for ref_img_len in l_effective_ref_img_len])
        max_img_len = max(l_effective_img_len)

        # ref image patch embeddings
        flat_ref_img_hidden_states = []
        for i in range(batch_size):
            if ref_img_sizes[i] is not None:
                imgs = []
                for ref_img in ref_image_hidden_states[i]:
                    C, H, W = ref_img.size()
                    ref_img = rearrange(ref_img, 'c (h p1) (w p2) -> (h w) (p1 p2 c)', p1=p, p2=p)
                    imgs.append(ref_img)

                img = torch.cat(imgs, dim=0)
                flat_ref_img_hidden_states.append(img)
            else:
                flat_ref_img_hidden_states.append(None)

        # image patch embeddings
        flat_hidden_states = []
        for i in range(batch_size):
            img = hidden_states[i]
            C, H, W = img.size()
            
            img = rearrange(img, 'c (h p1) (w p2) -> (h w) (p1 p2 c)', p1=p, p2=p)
            flat_hidden_states.append(img)
        
        padded_ref_img_hidden_states = torch.zeros(batch_size, max_ref_img_len, flat_hidden_states[0].shape[-1], device=device, dtype=flat_hidden_states[0].dtype)
        padded_ref_img_mask = torch.zeros(batch_size, max_ref_img_len, dtype=torch.bool, device=device)
        for i in range(batch_size):
            if ref_img_sizes[i] is not None:
                padded_ref_img_hidden_states[i, :sum(l_effective_ref_img_len[i])] = flat_ref_img_hidden_states[i]
                padded_ref_img_mask[i, :sum(l_effective_ref_img_len[i])] = True

        padded_hidden_states = torch.zeros(batch_size, max_img_len, flat_hidden_states[0].shape[-1], device=device, dtype=flat_hidden_states[0].dtype)
        padded_img_mask = torch.zeros(batch_size, max_img_len, dtype=torch.bool, device=device)
        for i in range(batch_size):
            padded_hidden_states[i, :l_effective_img_len[i]] = flat_hidden_states[i]
            padded_img_mask[i, :l_effective_img_len[i]] = True

        return (
            padded_hidden_states,
            padded_ref_img_hidden_states,
            padded_img_mask,
            padded_ref_img_mask,
            l_effective_ref_img_len,
            l_effective_img_len,
            ref_img_sizes,
            img_sizes,
        )
    
    def forward(
        self,
        hidden_states: Union[torch.Tensor, List[torch.Tensor]],
        timestep: torch.Tensor,
        text_hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        text_attention_mask: torch.Tensor,
        ref_image_hidden_states: Optional[List[List[torch.Tensor]]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        enable_taylorseer = getattr(self, 'enable_taylorseer', False)
        if enable_taylorseer:
            cal_type(self.cache_dic, self.current)
        
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        # ================================================================================================
        # 第一阶段：输入预处理和条件嵌入 (Condition, positional & patch embedding)
        # ================================================================================================
        
        # 获取batch大小，hidden_states可能是tensor或tensor列表
        batch_size = len(hidden_states)
        is_hidden_states_tensor = isinstance(hidden_states, torch.Tensor)

        # 如果输入是单个tensor，转换为列表格式以支持变长序列处理
        # hidden_states的形状: [batch, channels, height, width] -> List[[channels, height, width], ...]
        if is_hidden_states_tensor:
            assert hidden_states.ndim == 4  # 确保是4维tensor: [B, C, H, W]
            hidden_states = [_hidden_states for _hidden_states in hidden_states]

        device = hidden_states[0].device

        # 时间步和文本条件嵌入融合
        # temb: 时间步嵌入，用于告诉模型当前去噪的阶段
        # text_hidden_states: 经过时间调制的文本特征，融合了时间信息
        temb, text_hidden_states = self.time_caption_embed(timestep, text_hidden_states, hidden_states[0].dtype)

        # ================================================================================================
        # 第二阶段：序列化和填充处理 (Flattening and padding to sequence)
        # ================================================================================================
        
        # 将2D图像patch展平为1D序列，并进行填充对齐
        # 这是DiT的关键步骤：将图像从空间表示转换为序列表示
        # 
        # 📌 重要：这里的hidden_states有两个状态：
        # • 输入的hidden_states: 来自diffusion的噪声latents，形状[B, 16, H/16, W/16] 
        # • 输出的hidden_states: 转换后的patch序列，形状[B, seq_len, hidden_dim]
        (
            hidden_states,           # 输出：目标图像的patch序列 (从噪声latents转换而来)
            ref_image_hidden_states, # 输出：参考图像的patch序列 (编辑时的原图)
            img_mask,               # 目标图像的有效patch掩码
            ref_img_mask,           # 参考图像的有效patch掩码  
            l_effective_ref_img_len, # 参考图像有效长度列表
            l_effective_img_len,     # 目标图像有效长度列表
            ref_img_sizes,          # 参考图像尺寸信息
            img_sizes,              # 目标图像尺寸信息
        ) = self.flat_and_pad_to_seq(hidden_states, ref_image_hidden_states)
        
        # ================================================================================================
        # 第三阶段：3D旋转位置编码生成 (3D Rotary Position Embedding)
        # ================================================================================================
        
        # 为不同类型的序列生成专门的旋转位置编码
        # OmniGen2使用3D RoPE支持可变分辨率和多模态序列
        (
            context_rotary_emb,    # 文本上下文的位置编码
            ref_img_rotary_emb,    # 参考图像的2D位置编码 
            noise_rotary_emb,      # 噪声图像的2D位置编码
            rotary_emb,           # 联合序列的位置编码
            encoder_seq_lengths,   # 编码器序列长度(文本部分)
            seq_lengths,          # 总序列长度(文本+图像)
        ) = self.rope_embedder(
            freqs_cis,            # 预计算的旋转频率
            text_attention_mask,   # 文本注意力掩码
            l_effective_ref_img_len,
            l_effective_img_len,
            ref_img_sizes,
            img_sizes,
            device,
        )

        # ================================================================================================
        # 第四阶段：文本上下文细化 (Context refinement)
        # ================================================================================================
        
        # 通过专门的transformer层对文本特征进行细化
        # 这确保文本语义在与图像融合前得到充分处理
        for layer in self.context_refiner:
            text_hidden_states = layer(
                text_hidden_states,    # 文本特征
                text_attention_mask,   # 文本掩码，忽略padding部分
                context_rotary_emb     # 文本专用的位置编码
            )
        
        # ================================================================================================
        # 第五阶段：图像patch嵌入和细化 (Image patch embedding and refinement)
        # ================================================================================================
        
        # 对目标图像和参考图像进行patch嵌入，并通过noise refiner进行细化
        # 这是图像特征的预处理，为后续的联合attention做准备
        combined_img_hidden_states = self.img_patch_embed_and_refine(
            hidden_states,             # 目标图像patch序列
            ref_image_hidden_states,   # 参考图像patch序列
            img_mask,                 # 目标图像掩码
            ref_img_mask,             # 参考图像掩码
            noise_rotary_emb,         # 噪声图像位置编码
            ref_img_rotary_emb,       # 参考图像位置编码
            l_effective_ref_img_len,
            l_effective_img_len,
            temb,                     # 时间嵌入，用于条件化
        )

        # ================================================================================================
        # 第六阶段：联合序列构建 (Joint sequence construction)
        # ================================================================================================
        
        # 计算最大序列长度，用于创建统一的张量
        max_seq_len = max(seq_lengths)

        # 创建联合注意力掩码和特征张量
        # 形状: [batch_size, max_seq_len]
        attention_mask = hidden_states.new_zeros(batch_size, max_seq_len, dtype=torch.bool)
        # 形状: [batch_size, max_seq_len, hidden_size]  
        joint_hidden_states = hidden_states.new_zeros(batch_size, max_seq_len, self.config.hidden_size)
        
        # 为每个样本构建联合序列: [文本特征 | 图像特征]
        for i, (encoder_seq_len, seq_len) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
            # 设置有效序列的注意力掩码
            attention_mask[i, :seq_len] = True
            
            # 前半部分放置文本特征 (encoder部分)
            joint_hidden_states[i, :encoder_seq_len] = text_hidden_states[i, :encoder_seq_len]
            
            # 后半部分放置图像特征 (decoder部分)  
            joint_hidden_states[i, encoder_seq_len:seq_len] = combined_img_hidden_states[i, :seq_len - encoder_seq_len]

        hidden_states = joint_hidden_states

        if self.enable_teacache:
            teacache_hidden_states = hidden_states.clone()
            teacache_temb = temb.clone()
            modulated_inp, _, _, _ = self.layers[0].norm1(teacache_hidden_states, teacache_temb)
            if self.teacache_params.is_first_or_last_step:
                should_calc = True
                self.teacache_params.accumulated_rel_l1_distance = 0
            else:
                self.teacache_params.accumulated_rel_l1_distance += self.rescale_func(
                    ((modulated_inp - self.teacache_params.previous_modulated_inp).abs().mean() \
                        / self.teacache_params.previous_modulated_inp.abs().mean()).cpu().item()
                )
                if self.teacache_params.accumulated_rel_l1_distance < self.teacache_rel_l1_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    self.teacache_params.accumulated_rel_l1_distance = 0
            self.teacache_params.previous_modulated_inp = modulated_inp

        if self.enable_teacache:
            if not should_calc:
                hidden_states += self.teacache_params.previous_residual
            else:
                ori_hidden_states = hidden_states.clone()
                for layer_idx, layer in enumerate(self.layers):
                    if torch.is_grad_enabled() and self.gradient_checkpointing:
                        hidden_states = self._gradient_checkpointing_func(
                            layer, hidden_states, attention_mask, rotary_emb, temb
                        )
                    else:
                        hidden_states = layer(hidden_states, attention_mask, rotary_emb, temb)
                self.teacache_params.previous_residual = hidden_states - ori_hidden_states
        else:
            if enable_taylorseer:
                self.current['stream'] = 'layers_stream'

            for layer_idx, layer in enumerate(self.layers):
                if enable_taylorseer:
                    layer.current = self.current
                    layer.cache_dic = self.cache_dic
                    layer.enable_taylorseer = True
                    self.current['layer'] = layer_idx

                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    hidden_states = self._gradient_checkpointing_func(
                        layer, hidden_states, attention_mask, rotary_emb, temb
                    )
                else:
                    hidden_states = layer(hidden_states, attention_mask, rotary_emb, temb)

        # 4. Output norm & projection
        hidden_states = self.norm_out(hidden_states, temb)

        p = self.config.patch_size
        output = []
        for i, (img_size, img_len, seq_len) in enumerate(zip(img_sizes, l_effective_img_len, seq_lengths)):
            height, width = img_size
            output.append(rearrange(hidden_states[i][seq_len - img_len:seq_len], '(h w) (p1 p2 c) -> c (h p1) (w p2)', h=height // p, w=width // p, p1=p, p2=p))
        if is_hidden_states_tensor:
            output = torch.stack(output, dim=0)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)
            
        if enable_taylorseer:
            self.current['step'] += 1

        if not return_dict:
            return output
        return Transformer2DModelOutput(sample=output)