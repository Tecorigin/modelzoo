import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from dataclasses import dataclass

# ==========================================
# 1. 极简配置类 (Configuration)
# ==========================================
@dataclass
class Siglip2VisionConfig:
    """
    SigLIP 2 视觉模型配置 (针对 ImageNet 分类精简版)
    默认参数对应 ViT-Base (86M params)
    """
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_channels: int = 3
    image_size: int = 224
    patch_size: int = 16
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    num_labels: int = 1000  # ImageNet 类别数
    
    def __post_init__(self):
        # 自动计算 patch 数量，用于位置编码初始化
        self.num_patches = (self.image_size // self.patch_size) ** 2

# ==========================================
# 2. 核心模块 (Building Blocks)
# ==========================================

class Siglip2VisionEmbeddings(nn.Module):
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size
        self.image_size = config.image_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid", # SigLIP 不做 padding
        )

        self.num_patches = config.num_patches
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)
        
        # 注册位置 ID
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_patches).expand((1, -1)),
            persistent=False,
        )

    def resize_positional_embeddings(self, positional_embeddings, spatial_shapes, max_length):
        """
        SigLIP 2 核心特性：动态插值位置编码
        """
        batch_size = spatial_shapes.shape[0]
        embed_dim = positional_embeddings.shape[-1]
        
        # 将 flat 的位置编码还原回 2D 网格: (H_grid, W_grid, Dim)
        grid_size = int(self.num_patches**0.5)
        pos_embed_grid = positional_embeddings.view(grid_size, grid_size, embed_dim)
        
        # Permute to (1, Dim, H, W) for interpolate
        pos_embed_grid = pos_embed_grid.permute(2, 0, 1).unsqueeze(0)
        
        result_embeddings = torch.zeros(
            (batch_size, max_length, embed_dim),
            device=positional_embeddings.device,
            dtype=positional_embeddings.dtype
        )

        for i in range(batch_size):
            h_pixels, w_pixels = spatial_shapes[i]
            # 计算目标 grid 大小
            h_grid = h_pixels // self.patch_size
            w_grid = w_pixels // self.patch_size
            
            # 双线性插值
            resized = F.interpolate(
                pos_embed_grid,
                size=(h_grid, w_grid),
                mode="bilinear",
                align_corners=False,
                antialias=True 
            )
            
            # Flatten back: (Dim, H*W) -> (H*W, Dim)
            resized = resized.squeeze(0).flatten(1).transpose(0, 1)
            
            # 填入结果
            seq_len = resized.shape[0]
            if seq_len > max_length:
                seq_len = max_length # 截断保护
            result_embeddings[i, :seq_len, :] = resized[:seq_len, :]
            
        return result_embeddings

    def forward(self, pixel_values: torch.Tensor, spatial_shapes: torch.Tensor) -> torch.Tensor:
        # pixel_values: [B, 3, H, W]
        patch_embeds = self.patch_embedding(pixel_values) 
        # [B, C, H_grid, W_grid] -> [B, C, Seq_Len] -> [B, Seq_Len, C]
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        
        # 获取位置编码权重
        pos_weights = self.position_embedding.weight # [Num_Patches, Dim]
        
        # 动态插值位置编码
        resized_pos_embeds = self.resize_positional_embeddings(
            pos_weights, spatial_shapes, max_length=embeddings.shape[1]
        )
        
        return embeddings + resized_pos_embeds

class Siglip2MLP(nn.Module):
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act = nn.GELU() # 默认 GELU

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class Siglip2Attention(nn.Module):
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # --- 修改部分开始 ---
        # 原理：SDPA 内部公式是 Softmax( (Q @ K.T) / sqrt(d_k) ) @ V
        # 我们的目标是：Softmax( (Q @ K.T) * self.scale ) @ V
        # 因此需要：q_scaled = q * self.scale * sqrt(d_k)
        # 这样进入 SDPA 后，内部除以 sqrt(d_k) 正好抵消，剩下 self.scale
        
        d_k = q.size(-1)
        # 使用 math.sqrt(d_k) 也可以，或者 (d_k ** 0.5)
        q_scaled = q * (self.scale * (d_k ** 0.5)) 
        
        # 兼容旧版本 PyTorch，不再传 scale 参数
        attn_output = F.scaled_dot_product_attention(q_scaled, k, v)
        # --- 修改部分结束 ---
        
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, seq_len, self.embed_dim)
        return self.out_proj(attn_output)

class Siglip2EncoderLayer(nn.Module):
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = Siglip2Attention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = Siglip2MLP(config)

    def forward(self, hidden_states):
        # Pre-Norm 结构
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class Siglip2Encoder(nn.Module):
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.layers = nn.ModuleList([Siglip2EncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states

class Siglip2VisionTransformer(nn.Module):
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.embeddings = Siglip2VisionEmbeddings(config)
        self.encoder = Siglip2Encoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values, spatial_shapes):
        hidden_states = self.embeddings(pixel_values, spatial_shapes)
        hidden_states = self.encoder(hidden_states)
        hidden_states = self.post_layernorm(hidden_states)
        return hidden_states

# ==========================================
# 3. 最终分类模型 (Model Wrapper)
# ==========================================

class Siglip2ForImageClassification(nn.Module):
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels
        
        # 骨干网络
        self.vision_model = Siglip2VisionTransformer(config)
        
        # 分类头
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # 初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)

    def forward(self, pixel_values, labels=None):
        """
        标准的 ImageNet 训练接口
        Args:
            pixel_values: [Batch, 3, Height, Width]
            labels: [Batch] (0 ~ num_classes-1)
        """
        b, c, h, w = pixel_values.shape
        device = pixel_values.device
        
        # 1. 自动生成 spatial_shapes (SigLIP 2 必需)
        # 假设 Batch 内所有图片尺寸一致 (ImageNet 默认行为)
        spatial_shapes = torch.tensor([[h, w]], device=device).expand(b, 2)
        
        # 2. 骨干网络前向传播
        # Output: [Batch, Seq_Len, Hidden_Size]
        sequence_output = self.vision_model(pixel_values, spatial_shapes)
        
        # 3. 全局平均池化 (Global Average Pooling) - 替代 CLS token
        # [Batch, Seq_Len, Hidden_Size] -> [Batch, Hidden_Size]
        pooled_output = sequence_output.mean(dim=1)
        
        # 4. 分类映射
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
            
        return logits

# ==========================================
# 4. 工厂函数 & 测试代码
# ==========================================

def Model(num_classes=1000, model_size='base'):
    if model_size == 'base':
        config = Siglip2VisionConfig(
            hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, num_labels=num_classes
        )
    elif model_size == 'large':
        config = Siglip2VisionConfig(
            hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, intermediate_size=4096, num_labels=num_classes
        )
    elif model_size == 'so400m':
        config = Siglip2VisionConfig(
            hidden_size=1152, num_hidden_layers=27, num_attention_heads=16, intermediate_size=4304, num_labels=num_classes
        )
    else:
        raise ValueError("Unsupported model size")
        
    return Siglip2ForImageClassification(config)

if __name__ == "__main__":
    print("正在初始化 SigLIP 2 (ImageNet Classification)...")
    
    # 1. 创建模型
    model = Model(model_size='base', num_classes=1000)
    
    # 2. 模拟输入数据 (Batch=2, RGB, 224x224)
    pixel_values = torch.randn(2, 3, 224, 224)
    labels = torch.tensor([0, 999], dtype=torch.long) # 假设两个标签
    
    # 3. 运行前向传播
    print("开始前向传播测试...")
    loss, logits = model(pixel_values, labels=labels)
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Logits Shape: {logits.shape}") # 应该是 (2, 1000)
    
    # 4. 运行反向传播 (验证梯度)
    print("开始反向传播测试...")
    loss.backward()
    
    print("测试完成！所有梯度计算正常。")
    print(f"FC层梯度范数: {model.classifier.weight.grad.norm().item():.4f}")

