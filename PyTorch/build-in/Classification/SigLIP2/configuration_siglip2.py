import os
from typing import Union
from transformers import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

class Siglip2VisionConfig(PretrainedConfig):
    r"""
    Siglip2VisionConfig 是用于存储 Siglip2VisionModel 配置的类。
    它用于实例化一个 Siglip2 视觉编码器，根据指定的参数定义模型架构。
    
    默认配置对应于 SigLIP-2 ViT-Base 模型。
    """

    model_type = "siglip2_vision"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_patches=None, # 如果不传，会自动计算 (image_size // patch_size) ** 2
        vision_use_head=True,
        # 用于 Flash Attention 或 SDPA 的实现选择
        _attn_implementation="eager", 
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.vision_use_head = vision_use_head
        self._attn_implementation = _attn_implementation

        # 自动计算 num_patches，这对于位置编码的初始化至关重要
        if num_patches is None:
            self.num_patches = (image_size // patch_size) ** 2
        else:
            self.num_patches = num_patches

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果从 checkpoint 加载，确保 vision_config 字典被正确使用
        if "vision_config" in config_dict:
            config_dict = config_dict["vision_config"]

        return cls.from_dict(config_dict, **kwargs)


class Siglip2TextConfig(PretrainedConfig):
    r"""
    Siglip2TextConfig 是用于存储 Siglip2TextModel 配置的类。
    """

    model_type = "siglip2_text"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=2048,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        # 投影层大小，即文本嵌入最后映射到的维度
        projection_size=768, 
        _attn_implementation="eager",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.projection_size = projection_size
        self._attn_implementation = _attn_implementation

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果从 checkpoint 加载，确保 text_config 字典被正确使用
        if "text_config" in config_dict:
            config_dict = config_dict["text_config"]

        return cls.from_dict(config_dict, **kwargs)


class Siglip2Config(PretrainedConfig):
    r"""
    Siglip2Config 是用于存储 Siglip2Model 配置的类。
    它包含实例化 Siglip2VisionConfig 和 Siglip2TextConfig 所需的所有参数。

    Args:
        vision_config (`dict`, *optional*):
            用于初始化 Siglip2VisionConfig 的字典。
        text_config (`dict`, *optional*):
            用于初始化 Siglip2TextConfig 的字典。
        kwargs (*optional*):
            传递给父类 PretrainedConfig 的关键字参数。
    """

    model_type = "siglip2"
    is_composition = True

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        projection_dim=768,
        logit_scale_init_value=2.6592,
        logit_bias_init_value=-10.0,
        **kwargs,
    ):
        # 1. 初始化 Vision Config
        if vision_config is None:
            self.vision_config = Siglip2VisionConfig()
            logger.info("vision_config is None. Initializing the Siglip2VisionConfig with default values.")
        elif isinstance(vision_config, dict):
            self.vision_config = Siglip2VisionConfig(**vision_config)
        elif isinstance(vision_config, Siglip2VisionConfig):
            self.vision_config = vision_config
        else:
            raise TypeError(f"vision_config should be a dict or Siglip2VisionConfig, but got {type(vision_config)}")

        # 2. 初始化 Text Config
        if text_config is None:
            self.text_config = Siglip2TextConfig()
            logger.info("text_config is None. Initializing the Siglip2TextConfig with default values.")
        elif isinstance(text_config, dict):
            self.text_config = Siglip2TextConfig(**text_config)
        elif isinstance(text_config, Siglip2TextConfig):
            self.text_config = text_config
        else:
            raise TypeError(f"text_config should be a dict or Siglip2TextConfig, but got {type(text_config)}")

        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.logit_bias_init_value = logit_bias_init_value
        
        # 确保初始化父类
        super().__init__(**kwargs)
    
    @classmethod
    def from_vision_text_configs(cls, vision_config: Siglip2VisionConfig, text_config: Siglip2TextConfig, **kwargs):
        r"""
        从 vision_config 和 text_config 实例化 Siglip2Config。
        """
        return cls(vision_config=vision_config, text_config=text_config, **kwargs)

    def to_dict(self):
        """
        将配置序列化为字典。
        """
        output = super().to_dict()
        output["vision_config"] = self.vision_config.to_dict()
        output["text_config"] = self.text_config.to_dict()
        return output