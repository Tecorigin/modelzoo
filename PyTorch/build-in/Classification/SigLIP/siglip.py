import torch
import torch.nn as nn
from typing import Tuple, Optional

class SiglipVisionConfig:
    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_hidden_attention_heads=12, # 注意：代码里要统一用这个名字
        num_channels=3,
        image_size=224,
        patch_size=16,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens: int = None,
        **kwargs      
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_attention_heads = num_hidden_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens

class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__() # 修复：加括号
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        # 修复：Convo2d -> Conv2d
        self.patch_embeddings = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size)**2
        self.num_positions = self.num_patches
        self.position_embeddings = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    # 修复：FloatTesor -> FloatTensor
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape
        # 修复：变量名一致性 patch_embedding -> patch_embeddings
        patch_embeds = self.patch_embeddings(pixel_values)
        embeddings = patch_embeds.flatten(2)
        embeddings = embeddings.transpose(1, 2)
        # 修复：变量名一致性 position_embedding -> position_embeddings
        embeddings = embeddings + self.position_embeddings(self.position_ids)
        return embeddings

class SiglipAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        # 修复：使用 Config 中定义的正确字段名
        self.num_heads = config.num_hidden_attention_heads 
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        
        # 修复：value_states.query_states 是错误的写法，直接 view 即可
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        
        attn_weights = (torch.matmul(query_states, key_states.transpose(2,3)) * self.scale)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights

class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Self Attention Block
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states
        
        # MLP Block
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # 修复：加上残差连接，否则深层网络无法训练
        hidden_states = residual + hidden_states 
        
        return hidden_states

class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
    
    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)
        return hidden_states

class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values)
        # 修复：参数名对应 inputs_embeds
        last_hidden_state = self.encoder(inputs_embeds=hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state        

class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)
    
    def forward(self, pixel_values) -> torch.Tensor:
        return self.vision_model(pixel_values=pixel_values)

import torch
import torch.nn as nn

# 假设之前的类定义 (SiglipVisionConfig, SiglipVisionModel 等) 已经在上面定义好了
# 如果是在同一个文件中，直接接在后面即可。

class SiglipForImageClassification(nn.Module):

    def __init__(self, config: SiglipVisionConfig, num_classes: int = 1000):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        
        self.vision_model = SiglipVisionModel(config)
        
        self.classifier = nn.Linear(config.hidden_size, num_classes)
        
        self._init_weights(self.classifier)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, pixel_values: torch.Tensor, labels: torch.Tensor = None):

        backbone_output = self.vision_model(pixel_values)

        pooled_output = backbone_output.mean(dim=1)

        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            
        if loss is not None:
            return logits, loss
        return logits

def Model(num_classes=1000, model_size='base'):

    if model_size == 'base':
        config = SiglipVisionConfig(
            hidden_size=768,
            intermediate_size=3072,
            num_hidden_layers=12,
            num_hidden_attention_heads=12,
            image_size=224,
            patch_size=16
        )
    elif model_size == 'large':
        config = SiglipVisionConfig(
            hidden_size=1024,
            intermediate_size=4096,
            num_hidden_layers=24,
            num_hidden_attention_heads=16,
            image_size=224,
            patch_size=16
        )
    else:
        raise ValueError(f"Unknown model_size: {model_size}")

    # 2. 实例化分类模型
    model = SiglipForImageClassification(config, num_classes=num_classes)
    
    return model

if __name__ == "__main__":
    import torch
    
    # 1. 设置测试参数
    MODEL_SIZE = 'base'
    NUM_CLASSES = 1000
    IMAGE_SIZE = 224
    BATCH_SIZE = 2
    
    print(f"--- 开始测试 SigLIP (Generation 1) Image Classification ---")
    print(f"配置: Size={MODEL_SIZE}, Classes={NUM_CLASSES}, Input={IMAGE_SIZE}x{IMAGE_SIZE}")

    # 2. 初始化模型
    try:
        model = Model(num_classes=NUM_CLASSES, model_size=MODEL_SIZE)
        print("✅ 模型初始化成功")
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        exit(1)

    # 3. 构造伪造的输入数据
    # 输入形状: [Batch_Size, Channels, Height, Width]
    pixel_values = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
    # 标签形状: [Batch_Size] (范围 0 ~ NUM_CLASSES-1)
    labels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,), dtype=torch.long)
    
    print(f"输入数据形状: {pixel_values.shape}")
    print(f"标签数据形状: {labels.shape}")

    # 4. 前向传播测试 (Forward Pass)
    print("\n--- 执行前向传播 ---")
    # 开启训练模式 (启用 Dropout 等)
    model.train() 
    
    try:
        # 注意：你定义的 SiglipForImageClassification 返回顺序是 (logits, loss)
        logits, loss = model(pixel_values, labels=labels)
        
        print(f"✅ 前向传播成功")
        print(f"Loss 值: {loss.item():.6f}")
        print(f"Logits 形状: {logits.shape} (预期: [{BATCH_SIZE}, {NUM_CLASSES}])")
        
        assert logits.shape == (BATCH_SIZE, NUM_CLASSES), "输出 Logits 形状不匹配！"
        assert not torch.isnan(loss), "Loss 出现了 NaN！请检查初始化或输入数据。"

    except Exception as e:
        print(f"❌ 前向传播出错: {e}")
        exit(1)

    # 5. 反向传播测试 (Backward Pass)
    print("\n--- 执行反向传播 ---")
    try:
        # 清空梯度
        model.zero_grad()
        
        # 反向传播
        loss.backward()
        
        # 检查分类头的梯度是否存在，且不为 0
        # 如果模型中间有断层（例如忘了加 residual），梯度可能传不回来
        grad_norm = model.classifier.weight.grad.norm().item()
        
        print(f"✅ 反向传播成功")
        print(f"分类层梯度范数 (Grad Norm): {grad_norm:.6f}")
        
        if grad_norm == 0.0:
            print("⚠️ 警告: 梯度为 0，可能模型结构有问题或被冻结。")
        else:
            print("🎉 测试通过：梯度正常回传。")

    except Exception as e:
        print(f"❌ 反向传播出错: {e}")
        exit(1)

    # 6. (可选) 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型总参数量: {total_params / 1e6:.2f} M")