import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange

# =============================================================================
# 工具函数 (Helper Functions)
# =============================================================================

def pair(t):
    """
    将单个值转换为元组，如果已经是元组则直接返回
    用于处理image_size和patch_size参数，支持正方形和矩形
    
    Args:
        t: 整数或元组
    Returns:
        tuple: (t, t) 如果t是整数，否则返回t本身
    """
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    """
    生成2D正弦余弦位置编码
    这是一种固定的位置编码，不需要学习，基于Transformer原论文的位置编码
    
    Args:
        h: 高度方向的位置数量（patch数量）
        w: 宽度方向的位置数量（patch数量）  
        dim: 编码维度，必须是4的倍数
        temperature: 温度参数，控制编码的频率范围
        dtype: 数据类型
    
    Returns:
        torch.Tensor: 形状为(h*w, dim)的位置编码张量
    """
    # 生成网格坐标，indexing="ij"表示使用矩阵索引方式
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    # 确保维度是4的倍数，因为我们需要对x和y各自进行sin和cos编码
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    # 计算频率，omega形状为(dim//4,)
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)  # 计算实际频率值

    # 将坐标展平并与频率相乘
    y = y.flatten()[:, None] * omega[None, :]  # 形状: (h*w, dim//4)
    x = x.flatten()[:, None] * omega[None, :]  # 形状: (h*w, dim//4)
    # 拼接sin和cos编码：x_sin, x_cos, y_sin, y_cos
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

# =============================================================================
# 模型组件 (Model Components)  
# =============================================================================

class FeedForward(nn.Module):
    """
    前馈网络 (Feed Forward Network)
    在Transformer中，每个注意力层后面都跟着一个前馈网络
    结构：LayerNorm -> Linear -> GELU -> Linear
    """
    def __init__(self, dim, hidden_dim):
        """
        初始化前馈网络
        
        Args:
            dim: 输入维度
            hidden_dim: 隐藏层维度，通常是输入维度的2-4倍
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),           # 层归一化，稳定训练
            nn.Linear(dim, hidden_dim),   # 第一个线性层，升维
            nn.GELU(),                   # GELU激活函数，比ReLU更平滑
            nn.Linear(hidden_dim, dim),   # 第二个线性层，降维回原始维度
        )
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为(batch_size, seq_len, dim)
        
        Returns:
            torch.Tensor: 输出张量，形状与输入相同
        """
        return self.net(x)

class Attention(nn.Module):
    """
    多头自注意力机制 (Multi-Head Self-Attention)
    这是Transformer的核心组件，允许每个位置关注输入序列的所有位置
    """
    def __init__(self, dim, heads = 8, dim_head = 64):
        """
        初始化注意力模块
        
        Args:
            dim: 输入维度
            heads: 注意力头的数量
            dim_head: 每个注意力头的维度
        """
        super().__init__()
        inner_dim = dim_head *  heads  # 计算所有头的总维度
        self.heads = heads  # 注意力头数
        self.scale = dim_head ** -0.5  # 缩放因子，等于1/sqrt(dim_head)
        self.norm = nn.LayerNorm(dim)  # 输入的层归一化

        self.attend = nn.Softmax(dim = -1)  # Softmax用于计算注意力权重

        # 将输入投影到查询(Q)、键(K)、值(V)，一次性生成所有头的Q、K、V
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)  # 输出投影层

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为(batch_size, seq_len, dim)
        
        Returns:
            torch.Tensor: 输出张量，形状与输入相同
        """
        x = self.norm(x)  # 首先进行层归一化

        # 生成Q、K、V，然后分成3个张量
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # 将Q、K、V重新排列为多头格式：(batch, seq_len, heads * dim_head) -> (batch, heads, seq_len, dim_head)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        # 计算注意力分数：Q * K^T，并应用缩放因子
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # 应用softmax得到注意力权重
        attn = self.attend(dots)

        # 将注意力权重应用到值V上
        out = torch.matmul(attn, v)
        # 重新排列回原始格式：(batch, heads, seq_len, dim_head) -> (batch, seq_len, heads * dim_head)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)  # 通过输出投影层

class Transformer(nn.Module):
    """
    Transformer编码器
    由多个Transformer层堆叠而成，每层包含注意力机制和前馈网络
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        """
        初始化Transformer编码器
        
        Args:
            dim: 输入维度
            depth: 层数
            heads: 注意力头数
            dim_head: 每个注意力头的维度
            mlp_dim: 前馈网络的隐藏维度
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)  # 最终的层归一化
        self.layers = nn.ModuleList([])  # 创建多个Transformer层
        for _ in range(depth):
            # 每层包含注意力和前馈网络
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为(batch_size, seq_len, dim)
        
        Returns:
            torch.Tensor: 输出张量，形状与输入相同
        """
        # 逐层处理
        for attn, ff in self.layers:
            x = attn(x) + x  # 注意力机制 + 残差连接
            x = ff(x) + x    # 前馈网络 + 残差连接
        return self.norm(x)  # 最终层归一化

class SimpleViT(nn.Module):
    """
    简化的Vision Transformer
    相比标准ViT，去掉了CLS token，使用平均池化
    使用2D正弦余弦位置编码而非可学习位置编码
    """
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
        """
        初始化SimpleViT
        
        Args:
            image_size: 输入图像尺寸，可以是整数或(height, width)元组
            patch_size: patch尺寸，可以是整数或(height, width)元组
            num_classes: 分类数量
            dim: 模型维度
            depth: Transformer层数
            heads: 注意力头数
            mlp_dim: 前馈网络隐藏维度
            channels: 输入图像通道数，默认3(RGB)
            dim_head: 每个注意力头的维度
        """
        super().__init__()
        # 处理图像和patch尺寸，支持矩形
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        # 确保图像尺寸能被patch尺寸整除
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        # 计算每个patch的维度
        patch_dim = channels * patch_height * patch_width

        # 图像到patch嵌入的转换
        self.to_patch_embedding = nn.Sequential(
            # 使用einops重新排列，将图像分割成patches
            # 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)'
            # 将形状(batch, channels, height, width)转换为(batch, num_patches, patch_dim)
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),      # 对patch进行归一化
            nn.Linear(patch_dim, dim),     # 线性投影到模型维度
            nn.LayerNorm(dim),            # 再次归一化
        )

        # 生成2D正弦余弦位置编码
        self.pos_embedding = posemb_sincos_2d(
            h = image_height // patch_height,    # patch的高度数量
            w = image_width // patch_width,      # patch的宽度数量
            dim = dim,                          # 位置编码维度
        ) 

        # Transformer编码器
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        # 池化方式：使用平均池化而非CLS token
        self.pool = "mean"
        # 恒等映射，为了保持接口一致性
        self.to_latent = nn.Identity()

        # 分类头：线性层
        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        """
        前向传播
        
        Args:
            img: 输入图像张量，形状为(batch_size, channels, height, width)
        
        Returns:
            torch.Tensor: 分类logits，形状为(batch_size, num_classes)
        """
        # 获取设备信息，确保位置编码在正确设备上
        device = img.device

        # 将图像转换为patch embeddings
        x = self.to_patch_embedding(img)  # (batch_size, num_patches, dim)
        # 添加位置编码
        x += self.pos_embedding.to(device, dtype=x.dtype)

        # 通过Transformer编码器
        x = self.transformer(x)  # (batch_size, num_patches, dim)
        # 平均池化：对所有patch的特征求平均
        x = x.mean(dim = 1)  # (batch_size, dim)

        # 通过恒等映射（占位符）
        x = self.to_latent(x)
        # 分类头：输出最终的分类logits
        return self.linear_head(x)  # (batch_size, num_classes)
