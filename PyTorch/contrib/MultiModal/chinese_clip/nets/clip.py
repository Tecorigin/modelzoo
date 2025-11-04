import numpy as np
import torch
import torch_sdaa
from torch import nn
from transformers import BertTokenizer, BertModel
from .bert import Transformer
from .simple_tokenizer import SimpleTokenizer, tokenize
from .vit import VisionTransformer

class CLIP(nn.Module):
    def __init__(
        self,
        bert_type           = "huggingface",
        embed_dim          = 512,
        # vision
        input_resolution   = 224,
        vision_layers      = 12,
        vision_width       = 768,
        vision_patch_size  = 16,
        # text
        context_length      = 77,
        **kwargs
    ):
        super().__init__()
        self.context_length = context_length

        vision_heads    = vision_width // 64
        self.visual     = VisionTransformer(
            input_resolution    = input_resolution,
            patch_size          = vision_patch_size,
            width               = vision_width,
            layers              = vision_layers,
            heads               = vision_heads,
            output_dim          = embed_dim
        )

        self.bert_type = bert_type

        model_path = kwargs['huggingface_model_name']
        # 加载 tokenizer
        tokenizer_path = model_path
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        # 加载 transformer 模型
        self.transformer = BertModel.from_pretrained(model_path)

        transformer_width = self.transformer.config.hidden_size
        self.text_projection        = nn.Parameter(torch.empty(transformer_width, embed_dim))
        nn.init.normal_(self.text_projection, std=transformer_width ** -0.5)
        self.ln_final               = nn.LayerNorm(transformer_width)
        self.logit_scale            = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def encode_image(self, image):
        return self.visual(image.type(self.visual.conv1.weight.dtype))

    def encode_text(self, text):
        x = self.tokenizer(text, return_tensors="pt", padding=True)
        input_ids = x.input_ids.to(self.visual.conv1.weight.device)
        attention_mask = x.attention_mask.to(self.visual.conv1.weight.device)
        token_type_ids = x.token_type_ids.to(self.visual.conv1.weight.device)
        x = self.transformer(input_ids=input_ids, attention_mask=attention_mask,
                             token_type_ids=token_type_ids).pooler_output
        x = self.ln_final(x).type(self.dtype)
        x = x @ self.text_projection
        return x

    def forward(self, image, text):
        image_features  = self.encode_image(image)
        if torch_sdaa.amp.is_autocast_enabled():
            with torch_sdaa.amp.autocast(enabled=False):
                text_features   = self.encode_text(text)
            with torch_sdaa.amp.autocast(enabled=True):
                image_features  = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features   = text_features / text_features.norm(dim=-1, keepdim=True)

                logit_scale         = self.logit_scale.exp()
                logits_per_image    = logit_scale * image_features @ text_features.t()
                logits_per_text     = logits_per_image.t()
        else:
            text_features = self.encode_text(text)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            logit_scale = self.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text
