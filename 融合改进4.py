import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import torch
from config import get_args

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_d = nn.AdaptiveAvgPool3d((None, None, 1))
        self.pool_h = nn.AdaptiveAvgPool3d((None, 1, None))
        self.pool_w = nn.AdaptiveAvgPool3d((1, None, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv3d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm3d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, d, h, w = x.size()
        x_d = self.pool_d(x).squeeze(2)
        x_h = self.pool_h(x).squeeze(3)
        x_w = self.pool_w(x).permute(0, 1, 4, 3, 2).squeeze(2)

        # 确保所有张量在第4个维度上具有相同的大小
        assert x_d.size(3) == x_h.size(3) == x_w.size(3)

        y = torch.cat([x_d, x_h, x_w], dim=1)

        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_d, x_h, x_w = torch.split(y, [d, h, w], dim=2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


'''
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.query_conv = nn.Conv3d(in_channels, in_channels//2, 1)
        self.key_conv = nn.Conv3d(in_channels, in_channels//2, 1)
        self.value_conv = nn.Conv3d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, depth, height, width = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, depth * height * width).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, depth * height * width)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch_size, -1, depth * height * width)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, depth, height, width)
        out = self.gamma * out + x

        return out
'''


class SelfAttention(nn.Module):
    def __init__(self, in_channels, heads=8):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.head_dim = in_channels // heads
        self.heads = heads
        assert self.head_dim * heads == in_channels, "Incompatible number of heads and in_channels"

        self.values = nn.Conv3d(in_channels, self.heads * self.head_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.keys = nn.Conv3d(in_channels, self.heads * self.head_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.queries = nn.Conv3d(in_channels, self.heads * self.head_dim, kernel_size=1, stride=1, padding=0,
                                 bias=False)
        self.fc_out = nn.Conv3d(self.heads * self.head_dim, in_channels, kernel_size=1, stride=1, padding=0)

        self.norm = nn.BatchNorm3d(in_channels)  # 归一化层
        self.activation = nn.ReLU()  # 激活函数

    def forward(self, x):
        N, C, D, H, W = x.size()
        residual = x  # 保存输入以用于残差连接
        # 添加空间位置向量嵌入的部分

        # Apply convolutions to values, keys, and queries
        values = self.values(x).view(N, self.heads, self.head_dim, D, H, W)
        keys = self.keys(x).view(N, self.heads, self.head_dim, D, H, W)
        queries = self.queries(x).view(N, self.heads, self.head_dim, D, H, W)

        # Permute dimensions for matrix multiplication
        values = values.permute(0, 1, 3, 4, 5, 2).contiguous()
        keys = keys.permute(0, 1, 3, 4, 5, 2).contiguous()
        queries = queries.permute(0, 1, 3, 4, 5, 2).contiguous()

        # Calculate attention scores
        energy = torch.einsum("nhdxyz,nhexyz->nhedxy", [queries, keys])
        attention = F.softmax(energy, dim=-1)

        # Apply attention to values
        out = torch.einsum("nhedxy,nhdxyz->nhexyz", [attention, values]).reshape(N, self.heads * self.head_dim, D, H, W)

        # Reshape and apply final convolution
        out = self.fc_out(out)
        out = self.norm(out)
        out = self.activation(out)
        out = out + residual

        return out


class DecoderSelfAttention(nn.Module):
    def __init__(self, in_channels, heads=4):
        super(DecoderSelfAttention, self).__init__()
        self.in_channels = in_channels
        self.head_dim = in_channels // heads
        self.heads = heads
        assert self.head_dim * heads == in_channels, "Incompatible number of heads and in_channels"

        self.values = nn.Conv3d(in_channels, self.heads * self.head_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.keys = nn.Conv3d(in_channels, self.heads * self.head_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.queries = nn.Conv3d(in_channels, self.heads * self.head_dim, kernel_size=1, stride=1, padding=0,
                                 bias=False)
        self.fc_out = nn.Conv3d(self.heads * self.head_dim, in_channels, kernel_size=1, stride=1, padding=0)

        self.norm = nn.BatchNorm3d(in_channels)  # 归一化层
        self.activation = nn.ReLU()  # 激活函数

    def forward(self, x, encoder_out):
        N, C, D, H, W = x.size()
        residual = x  # 保存输入以用于残差连接
        # 添加空间位置向量嵌入的部分

        # Apply convolutions to values, keys, and queries
        values = self.values(encoder_out).view(N, self.heads, self.head_dim, D, H, W)
        keys = self.keys(encoder_out).view(N, self.heads, self.head_dim, D, H, W)
        queries = self.queries(x).view(N, self.heads, self.head_dim, D, H, W)

        # Permute dimensions for matrix multiplication
        values = values.permute(0, 1, 3, 4, 5, 2).contiguous()
        keys = keys.permute(0, 1, 3, 4, 5, 2).contiguous()
        queries = queries.permute(0, 1, 3, 4, 5, 2).contiguous()

        # Calculate attention scores
        energy = torch.einsum("nhdxyz,nhexyz->nhedxy", [queries, keys])
        attention = F.softmax(energy, dim=-1)

        # Apply attention to values
        out = torch.einsum("nhedxy,nhdxyz->nhexyz", [attention, values]).reshape(N, self.heads * self.head_dim, D, H, W)

        # Reshape and apply final convolution
        out = self.fc_out(out)
        out = self.norm(out)
        out = self.activation(out)
        out = out + residual

        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=1):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm3d(out_planes, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 3
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        self.attention = scale
        return x * self.attention


class PositionalEncoding3D(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding3D, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

    def forward(self, x):
        if x.dim() == 4:
            # 2D positional encoding
            pe = torch.zeros(x.size(0), x.size(2), x.size(3), self.d_model)
            pe.requires_grad = False
            pos = torch.arange(0, x.size(2), dtype=torch.float).unsqueeze(0).unsqueeze(0)
            div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
            pe[:, :, :, 0::2] = torch.sin(pos * div_term)
            pe[:, :, :, 1::2] = torch.cos(pos * div_term)
        elif x.dim() == 5:
            # 3D positional encoding
            pe = torch.zeros(x.size(0), x.size(2), x.size(3), x.size(4), self.d_model)
            pe.requires_grad = False
            # print(pe.shape)
            pos = torch.arange(0, self.d_model / 2, dtype=torch.float).unsqueeze(0).unsqueeze(0)
            # print(pos.shape)
            div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
            pe[:, :, :, :, 0::2] = torch.sin(pos * div_term)
            pe[:, :, :, :, 1::2] = torch.cos(pos * div_term)
        else:
            raise ValueError("Positional encoding input must have 4 or 5 dimensions")

        # 将 pe 移动到与 x 相同的设备上
        pe = pe.permute(0, 4, 1, 2, 3)
        pe = pe.to(x.device)
        return x + pe


class OrdinalEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(OrdinalEmbedding, self).__init__()
        self.fc = nn.Linear(input_dim, embedding_dim)

    def forward(self, x):
        return self.fc(x)

class OrdinalEmbeddingMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super(OrdinalEmbeddingMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embedding_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        return self.fc2(x)


class ScoreToEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(ScoreToEmbedding, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)  # 隐藏层大小可调整
        self.fc2 = nn.Linear(64, embedding_dim)
        self.activation = nn.ReLU()  # 确保非负
        self.sigmoid = nn.Sigmoid()  # 将值限制在 [0, 1]

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)  # 限制输出范围
        return x


class OrdinalEmbeddingConv(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(OrdinalEmbeddingConv, self).__init__()
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=embedding_dim, kernel_size=1)

    def forward(self, x):
        # 将输入 [batch_size, 1] 视为 [batch_size, 1, 1]
        x = x.unsqueeze(-1)  # [4, 1] -> [4, 1, 1]
        x = self.conv(x)  # 卷积变换
        return x.squeeze(-1)  # [4, 128, 1] -> [4, 128]


class MultiHeadAttention2D(nn.Module):
    def __init__(self, input_dim, num_heads=8):
        super(MultiHeadAttention2D, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        # Linear transformations for Q, K, and V
        self.W_q = nn.Linear(input_dim, input_dim)
        self.W_k = nn.Linear(input_dim, input_dim)
        self.W_v = nn.Linear(input_dim, input_dim)

        # Linear transformation for output
        self.W_out = nn.Linear(input_dim, input_dim)

    def forward(self, q, k, v):
        batch_size = q.size(0)

        # Linear transformation
        Q = self.W_q(q)
        K = self.W_k(k)
        V = self.W_v(v)

        # Splitting into multiple heads
        Q = Q.view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        K = K.view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        V = V.view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        # Concatenate heads and apply final linear transformation
        attention_output = attention_output.transpose(0, 1).contiguous().view(batch_size, -1)
        output = self.W_out(attention_output)
        return output


class MultiHeadAttention1D(nn.Module):
    def __init__(self, input_dim, num_heads=8):
        """
        多头自注意力机制，适配 1D 数据。

        参数:
        - input_dim: 输入特征维度。
        - num_heads: 注意力头的数量。
        """
        super(MultiHeadAttention1D, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        assert self.head_dim * num_heads == input_dim, "Input dimension must be divisible by num_heads."

        # 使用 1x1 卷积替代线性层 (对 1D 特征映射)
        self.W_q = nn.Conv1d(input_dim, input_dim, kernel_size=1, bias=False)
        self.W_k = nn.Conv1d(input_dim, input_dim, kernel_size=1, bias=False)
        self.W_v = nn.Conv1d(input_dim, input_dim, kernel_size=1, bias=False)

        # 输出卷积
        self.W_out = nn.Conv1d(input_dim, input_dim, kernel_size=1, bias=False)

        # 归一化层和激活函数
        self.norm = nn.LayerNorm(input_dim)
        self.activation = nn.ReLU()

    def forward(self, q, k, v):
        """
        前向传播。

        参数:
        - q: 查询张量，形状为 (batch_size, input_dim)。
        - k: 键张量，形状为 (batch_size, input_dim)。
        - v: 值张量，形状为 (batch_size, input_dim)。

        返回:
        - 输出张量，形状与输入相同。
        """
        batch_size, input_dim = q.size()

        # 调整形状以适配 Conv1d (需要增加一个维度作为序列长度)
        q = q.unsqueeze(-1)  # (batch_size, input_dim, 1)
        k = k.unsqueeze(-1)  # (batch_size, input_dim, 1)
        v = v.unsqueeze(-1)  # (batch_size, input_dim, 1)

        # 生成 Q, K, V
        Q = self.W_q(q).view(batch_size, self.num_heads, self.head_dim, -1)  # (batch_size, num_heads, head_dim, 1)
        K = self.W_k(k).view(batch_size, self.num_heads, self.head_dim, -1)  # (batch_size, num_heads, head_dim, 1)
        V = self.W_v(v).view(batch_size, self.num_heads, self.head_dim, -1)  # (batch_size, num_heads, head_dim, 1)

        # 转置维度以计算注意力
        Q = Q.permute(0, 1, 3, 2)  # (batch_size, num_heads, 1, head_dim)
        K = K.permute(0, 1, 3, 2)  # (batch_size, num_heads, 1, head_dim)
        V = V.permute(0, 1, 3, 2)  # (batch_size, num_heads, 1, head_dim)

        # 计算注意力分数 (scaled dot-product attention)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch_size, num_heads, 1, 1)
        attention_weights = F.softmax(scores, dim=-1)

        # 应用注意力到 V
        attention_output = torch.matmul(attention_weights, V)  # (batch_size, num_heads, 1, head_dim)

        # 恢复形状
        attention_output = attention_output.permute(0, 1, 3, 2).contiguous().view(batch_size, input_dim,
                                                                                  -1)  # (batch_size, input_dim, 1)

        # 输出变换
        out = self.W_out(attention_output).squeeze(-1)  # 恢复为 (batch_size, input_dim)
        # out = out + q.squeeze(-1)
        # 应用归一化和激活函数
        out = self.norm(out)
        out = self.activation(out)

        return out


class CognitiveAttentionModule(nn.Module):
    def __init__(self, input_dim, num_heads=8):
        super(CognitiveAttentionModule, self).__init__()
        self.multihead_self_attention = MultiHeadAttention1D(input_dim, num_heads)
        self.multihead_cross_attention = MultiHeadAttention1D(input_dim, num_heads)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, input_q, input_k, input_v):
        # Self-attention
        input = self.layer_norm(input_q)
        self_attention_output = self.multihead_self_attention(input, input, input)
        self_attention_output = self_attention_output + input
        
        # Cross-attention
        cross_attention_output = self.multihead_cross_attention(self_attention_output, input_k, input_v)
        
        #output = self.layer_norm(cross_attention_output + input_v)
        return cross_attention_output


class CognitiveAttentionModule1D(nn.Module):
    def __init__(self, input_dim, num_heads=8):
        super(CognitiveAttentionModule1D, self).__init__()
        self.multihead_self_attention = MultiHeadAttention1D(input_dim, num_heads)
        self.multihead_cross_attention = MultiHeadAttention1D(input_dim, num_heads)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, input_q, input_k, input_v):
        # Self-attention
        input = self.layer_norm(input_q)
        self_attention_output = self.multihead_self_attention(input, input, input)
        self_attention_output = self_attention_output + input
        # Cross-attention
        cross_attention_output = self.multihead_cross_attention(self_attention_output, input_k, input_v)
        output = self.layer_norm(cross_attention_output + self_attention_output)
        return output


'''class ContrastiveLossModel(nn.Module):  # 继承 nn.Module
    def __init__(self, tau=0.07):
        super(ContrastiveLossModel, self).__init__()  # 调用父类构造函数
        self.tau = tau  # 学习率预热的参数

    def cosine_similarity(self, x1, x2):
        """计算余弦相似度"""
        return F.cosine_similarity(x1, x2, dim=-1)

    def forward(self, text_features, image_features):
        """计算对比损失"""
        batch_size = image_features.shape[0]

        # 对特征进行L2标准化
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

        # 计算余弦相似度
        similarity_matrix = self.cosine_similarity(image_features.unsqueeze(1), text_features.unsqueeze(0)) / self.tau  # [batch_size, batch_size]

        # 取出正样本相似度
        pos_sim = torch.diag(similarity_matrix)  # 正样本相似度 [batch_size]

        # 公式 1: Limage
        L_image = -torch.log(torch.exp(pos_sim) / (torch.sum(torch.exp(similarity_matrix), dim=1) + 1e-10))

        # 公式 2: Ltext
        L_text = -torch.log(torch.exp(pos_sim) / (torch.sum(torch.exp(similarity_matrix), dim=0) + 1e-10))

        # 返回平均损失
        loss = (L_image.mean() + L_text.mean()) / 2
        return loss'''

class LabelGuidedLoss(torch.nn.Module):
    def __init__(self):
        """
        初始化损失类。继承自 torch.nn.Module。
        """
        super(LabelGuidedLoss, self).__init__()

    def forward(self, image_features, score_features, labels):
        """
        前向计算标签引导的损失，使得 AD 和 NC 在分数特征和影像特征上表现出不同的效果。

        参数:
        - image_features: MRI 图像特征, shape: [batch_size, feature_dim]
        - score_features: 分数特征, shape: [batch_size, feature_dim]
        - labels: 标签, shape: [batch_size], 1 表示 AD，0 表示 NC

        返回:
        - 平均损失值。
        """
        # 将 labels 从 [batch_size, 1] 变为 [batch_size]
        labels = labels.view(-1)

        # 计算影像特征和分数特征之间的余弦相似度
        cosine_similarity = F.cosine_similarity(image_features, score_features)

        # 初始化损失
        loss = torch.zeros_like(cosine_similarity)

        # AD 标签为 1，最小化余弦距离（让影像和分数特征更相似）
        loss[labels == 1] = 1 - cosine_similarity[labels == 1]  # 使相似度更大

        # NC 标签为 0，最大化余弦距离（让影像和分数特征不相似）
        loss[labels == 0] = cosine_similarity[labels == 0]  # 使相似度更小

        # 返回平均损失
        return loss.mean()


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        """
        初始化对比损失函数。

        参数：
        - margin: 用于控制不同类之间的最小距离。
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, image_features, score_features, labels):
        """
        前向计算对比损失。

        参数：
        - image_features: MRI 图像特征, shape: [batch_size, feature_dim]
        - score_features: 分数特征, shape: [batch_size, feature_dim]
        - labels: 样本标签, shape: [batch_size], 1 表示 AD，0 表示 NC

        返回：
        - 平均对比损失值。
        """
        # L2 归一化
        image_features = F.normalize(image_features, dim=-1)
        score_features = F.normalize(score_features, dim=-1)

        # 计算每个样本之间的欧几里得距离
        distances = torch.norm(image_features - score_features, dim=1)

        # 初始化损失
        loss = torch.zeros_like(distances)

        # 对 AD 样本（label == 1）最小化距离
        loss[labels == 1] = distances[labels == 1] ** 2

        # 对 NC 样本（label == 0）最大化距离（使用 margin 控制）
        loss[labels == 0] = torch.clamp(self.margin - distances[labels == 0], min=0) ** 2

        # 返回平均损失
        return loss.mean()


class CombinedLoss(torch.nn.Module):
    def __init__(self, margin=1.0, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha  # 权重因子

    def forward(self, image_features, score_features, labels):
        """
        结合分类损失和对比损失。
        """
        # L2 归一化
        image_features = F.normalize(image_features, dim=-1)
        score_features = F.normalize(score_features, dim=-1)

        # 计算欧几里得距离
        distances = torch.norm(image_features - score_features, dim=1)

        # 分类损失（用于约束 AD 和 NC 样本特征差异）
        contrastive_loss = torch.zeros_like(distances)
        contrastive_loss[labels == 1] = distances[labels == 1] ** 2
        contrastive_loss[labels == 0] = torch.clamp(self.margin - distances[labels == 0], min=0) ** 2

        # 一致性损失（最小化所有样本的特征距离）
        consistent_loss = distances.mean()

        # 加权组合
        return self.alpha * consistent_loss + (1 - self.alpha) * contrastive_loss.mean()


class SimilarityLoss(torch.nn.Module):
    def __init__(self, margin=0.5):
        """
        初始化无标签引导的损失类。
        参数：
        - margin: 控制特征之间的距离范围。
        """
        super(SimilarityLoss, self).__init__()
        self.margin = margin

    def forward(self, image_features, score_features):
        """
        前向计算损失函数。
        参数：
        - image_features: MRI 图像特征, shape: [batch_size, feature_dim]
        - score_features: 分数特征, shape: [batch_size, feature_dim]

        返回：
        - 平均损失值。
        """
        # 归一化特征，确保计算的距离是公平的
        image_features = F.normalize(image_features, dim=-1)
        score_features = F.normalize(score_features, dim=-1)

        # 计算余弦相似性
        cosine_similarity = F.cosine_similarity(image_features, score_features, dim=-1)

        # 损失公式：希望特征既保持一定的相似性，又避免完全一致
        loss = (1 - cosine_similarity) ** 2 + F.relu(cosine_similarity - self.margin) ** 2

        return loss.mean()


class CosineSimilarityLoss(torch.nn.Module):
    def __init__(self):
        """
        初始化无标签的余弦相似性损失类。
        """
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, image_features, score_features):
        """
        前向计算损失函数。
        参数：
        - image_features: MRI 图像特征, shape: [batch_size, feature_dim]
        - score_features: 分数特征, shape: [batch_size, feature_dim]

        返回：
        - 平均余弦相似损失值。
        """
        # 归一化特征，确保余弦相似度计算公平
        image_features = F.normalize(image_features, dim=-1)
        score_features = F.normalize(score_features, dim=-1)

        # 计算余弦相似度
        cosine_similarity = F.cosine_similarity(image_features, score_features, dim=-1)

        # 损失为1减去余弦相似度，鼓励特征更相似
        loss = 1 - cosine_similarity

        return loss.mean()


'''class MultiHeadAttention2D(nn.Module):
    def __init__(self, input_dim, num_heads=8, dropout=0.1):
        super(MultiHeadAttention2D, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        assert input_dim % num_heads == 0, "输入维度必须是头数的整数倍。"

        # Q, K, V 的线性变换
        self.W_q = nn.Linear(input_dim, input_dim)
        self.W_k = nn.Linear(input_dim, input_dim)
        self.W_v = nn.Linear(input_dim, input_dim)

        # 输出的线性变换
        self.W_out = nn.Linear(input_dim, input_dim)

        # Dropout for attention weights
        self.dropout = nn.Dropout(dropout)

        # Initialize weights using Xavier initialization for better convergence
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_out.weight)

    def forward(self, q, k, v, mask=None):
        batch_size, input_dim = q.size()

        # 线性变换并分头
        Q = self.W_q(q).view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        K = self.W_k(k).view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        V = self.W_v(v).view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)

        # 缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Optional mask for attention
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        attention_output = torch.matmul(attention_weights, V)

        # 拼接各个头并应用线性变换
        attention_output = attention_output.transpose(0, 1).contiguous().view(batch_size, input_dim)
        output = self.W_out(attention_output)
        return output

class CognitiveAttentionModule(nn.Module):
    def __init__(self, input_dim, num_heads=8, dropout=0.1):
        super(CognitiveAttentionModule, self).__init__()
        self.multihead_self_attention = MultiHeadAttention2D(input_dim, num_heads, dropout)
        self.multihead_cross_attention = MultiHeadAttention2D(input_dim, num_heads, dropout)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_q, input_k, input_v, self_attention_mask=None, cross_attention_mask=None):
        # 自注意力
        self_attention_output = self.multihead_self_attention(input_q, input_q, input_q, mask=self_attention_mask)
        self_attention_output = self.dropout(self_attention_output)
        self_attention_output = self.layer_norm1(self_attention_output + input_q)  # 残差连接

        # 交叉注意力
        cross_attention_output = self.multihead_cross_attention(self_attention_output, input_k, input_v, mask=cross_attention_mask)
        cross_attention_output = self.dropout(cross_attention_output)
        output = self.layer_norm2(cross_attention_output + self_attention_output)

        return output'''

class feature_Net4s_tp4(nn.Module):
    def __init__(self, dropout=0.0):
        nn.Module.__init__(self)

        self.feature_extractor1 = nn.Sequential(
            nn.Conv3d(1, 16, 3),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, 3),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2),
            nn.Conv3d(16, 32, 3),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2),
            nn.Conv3d(32, 64, 3),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, 3),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2),
            nn.Conv3d(64, 128, 3),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 128, 1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
        )
        self.classifier1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.classifier2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.classifier3 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.locat = PositionalEncoding3D(128)
        self.ca = SelfAttention(128)
        self.de = DecoderSelfAttention(128)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        features_x = self.feature_extractor1(x)
        features_x = self.locat(features_x)
        map = self.ca(features_x)
        features_x = self.de(features_x, map) * features_x
        features_x = self.pool(features_x)
        features_x = features_x.view(features_x.shape[0], -1)
        logits1 = self.classifier1(features_x)
        logits2 = self.classifier2(features_x)
        logits3 = self.classifier3(features_x)
        return logits1, logits2, logits3

class feature_Net4s_tp4_ferturex(nn.Module):
    def __init__(self, dropout=0.0):
        nn.Module.__init__(self)

        self.feature_extractor1 = nn.Sequential(
            nn.Conv3d(1, 16, 3),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, 3),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2),
            nn.Conv3d(16, 32, 3),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2),
            nn.Conv3d(32, 64, 3),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, 3),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2),
            nn.Conv3d(64, 128, 3),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 128, 1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
        )
        self.classifier1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.classifier2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.classifier3 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.locat = PositionalEncoding3D(128)
        self.ca = SelfAttention(128)
        self.de = DecoderSelfAttention(128)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        features_x = self.feature_extractor1(x)
        features_x = self.locat(features_x)
        map = self.ca(features_x)
        features_x = self.de(features_x, map) * features_x
        features_x = self.pool(features_x)
        features_x = features_x.view(features_x.shape[0], -1)
        return features_x


class feature_Net4s_tp4_feature(nn.Module):
    def __init__(self, dropout=0.3, contrastive_loss=False, learnable_alpha=False):
        self.contrastive_loss_flag = contrastive_loss
        self.learnable_alpha_flag = learnable_alpha
        nn.Module.__init__(self)

        self.feature_extractor1 = nn.Sequential(
            nn.Conv3d(1, 16, 3),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, 3),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2),
            nn.Conv3d(16, 32, 3),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2),
            nn.Conv3d(32, 64, 3),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, 3),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2),
            nn.Conv3d(64, 128, 3),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 128, 1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
        )
        
        self.classifier0 = nn.Sequential(
			nn.Dropout(dropout),
            #nn.Linear(256, 128),
            #nn.ReLU(),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 32),
			nn.ReLU(),
			nn.Linear(32, 2),
		)
        '''self.classifier00 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )'''
        '''self.classifier0 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )'''
        self.classifier1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.classifier2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.classifier3 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.locat = PositionalEncoding3D(128)
        self.ca = SelfAttention(128)
        self.de = DecoderSelfAttention(128)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.cam = CognitiveAttentionModule(128, 8)
        self.dp = ScoreToEmbedding(input_dim=1, embedding_dim=128)  # 生成值范围还是有问题
        #self.feature_x = feature_Net4s_tp4_ferturex().cuda()
        self.args = get_args()
        #self.QL = nn.Parameter(nn.init.xavier_normal_(torch.empty(1, 128)))
        #self.QL = nn.Parameter(torch.empty(1, 128).uniform_(0, 1))  # 生成值的范围问题
        #self.QL.requires_grad = True
        #self.QL = nn.Parameter(torch.randn(1, 128) * nn.init.xavier_normal_(torch.empty(1, 128)))
        self.cross_ronghe =SimilarityLoss()

        self.learnable_alpha_flag = learnable_alpha
        if self.learnable_alpha_flag:
            self.alpha1 = nn.Parameter(
                torch.tensor(1.0, requires_grad=True)
            )
            self.alpha2 = nn.Parameter(
                torch.tensor(1.0, requires_grad=True)
            )
            self.alpha3 = nn.Parameter(
                torch.tensor(1.0, requires_grad=True)
            )

    def compute_constrastive_loss(self, score1_mat, score2_mat, score3_mat, img_features):
        # compute similarity matrix
        score1_mat = F.normalize(score1_mat, p=2, dim=1)
        score2_mat = F.normalize(score2_mat, p=2, dim=1)
        score3_mat = F.normalize(score3_mat, p=2, dim=1)

        img_features = F.normalize(img_features, p=2, dim=1)

        # sim_mat1[i, j] : the similarity between img_features[i] and score1_mat[j]
        sim_mat1 = torch.matmul(img_features, score1_mat.t())
        sim_mat2 = torch.matmul(img_features, score2_mat.t())
        sim_mat3 = torch.matmul(img_features, score3_mat.t())

        # apply softmax, get probability img-to-score
        prob_img2score1 = F.softmax(sim_mat1, dim=1)
        prob_img2score2 = F.softmax(sim_mat2, dim=1)
        prob_img2score3 = F.softmax(sim_mat3, dim=1)

        # apply softmax  get probability score-to-img
        prob_score2img1 = F.softmax(sim_mat1, dim=0)
        prob_score2img2 = F.softmax(sim_mat2, dim=0)
        prob_score2img3 = F.softmax(sim_mat3, dim=0)

        # compute cross entropy loss for img socre alignment Limg2score = -1/N sum(log(prob_img_to_score(i,i)))
        loss_img2score1 = -torch.mean(torch.log(torch.diag(prob_img2score1) + 1e-10))
        loss_img2score2 = -torch.mean(torch.log(torch.diag(prob_img2score2) + 1e-10))
        loss_img2score3 = -torch.mean(torch.log(torch.diag(prob_img2score3) + 1e-10))

        # compute cross entropy loss for score img alignment Lscore2img = -1/N sum(log(prob_score_to_img(i,i)))
        loss_score2img1 = -torch.mean(torch.log(torch.diag(prob_score2img1) + 1e-10))
        loss_score2img2 = -torch.mean(torch.log(torch.diag(prob_score2img2) + 1e-10))
        loss_score2img3 = -torch.mean(torch.log(torch.diag(prob_score2img3) + 1e-10))
        
        
        # compute constrastive loss
        if self.learnable_alpha_flag:   
            # normalize coefficients
            alpha_sum = self.alpha1 + self.alpha2 + self.alpha3
            normalized_alpha1 = self.alpha1 / alpha_sum
            normalized_alpha2 = self.alpha2 / alpha_sum
            normalized_alpha3 = self.alpha3 / alpha_sum

            constrastive_loss = normalized_alpha1 * (loss_img2score1 + loss_score2img1) + normalized_alpha2 * (
                    loss_img2score2 + loss_score2img2) + normalized_alpha3 * (loss_img2score3 + loss_score2img3) / 6
        else:
            constrastive_loss = (loss_img2score1 + loss_score2img1 + loss_img2score2 +\
                  loss_score2img2 + loss_img2score3 + loss_score2img3) / 6
            
        return constrastive_loss

    def forward(self, logits1, logits2, logits3,labels, x,feature_x0):
        #feature_x0 = self.feature_x(x)
        logits1_ca = self.dp(logits1)
        logits2_ca = self.dp(logits2)
        logits3_ca = self.dp(logits3)
        sizeq = logits1_ca.shape[0]

        QL = torch.zeros(sizeq, 128, device=x.device)
        #QL = self.QL.repeat(sizeq, 1)
        #cross1= self.cross_ronghe(feature_x0, QL)
        Q1 = self.cam(QL, logits1_ca, logits1_ca)
        #cross2= self.cross_ronghe(feature_x0, Q1,labels)
        Q2 = self.cam(Q1, logits2_ca, logits2_ca)
        #cross3 = self.cross_ronghe(feature_x0, Q2, labels)
        Q3 = self.cam(Q2, logits3_ca, logits3_ca)
        #cross4 = self.cross_ronghe(feature_x0, Q3,labels)
        #Q4 = torch.cat((feature_x0, Q3), dim=1)#拼接操作，需修改分类层从256
        Q4 = self.cam(Q3, feature_x0,feature_x0) #融合操作

        #Q4 = feature_x0+Q3
        features_xx = Q4
        out_predit = self.classifier0(features_xx)
        if not self.contrastive_loss_flag:
            

            cross2 = self.cross_ronghe(feature_x0,logits1_ca)
            cross3 = self.cross_ronghe(feature_x0, logits2_ca)
            cross4 = self.cross_ronghe(feature_x0, logits3_ca)

            total_cross =cross4+cross3+cross2
        else:

            constrastive_loss = self.compute_constrastive_loss(logits1_ca, logits2_ca, logits3_ca, feature_x0)
            total_cross = constrastive_loss
            
        return out_predit, features_xx, total_cross

class feature_Net4s_tp4_feature_jiaohuan(nn.Module):
    def __init__(self, dropout=0.0):
        nn.Module.__init__(self)
        self.feature_extractor1 = nn.Sequential(
            nn.Conv3d(1, 16, 3),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, 3),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2),
            nn.Conv3d(16, 32, 3),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2),
            nn.Conv3d(32, 64, 3),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, 3),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2),
            nn.Conv3d(64, 128, 3),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 128, 1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
        )
        self.classifier0 = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 32),
			nn.ReLU(),
			nn.Linear(32, 2),
		)
        '''self.classifier0 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )'''
        self.classifier1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.classifier2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.classifier3 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.locat = PositionalEncoding3D(128)
        self.ca = SelfAttention(128)
        self.de = DecoderSelfAttention(128)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.cam = CognitiveAttentionModule(128, 8)
        self.dp = OrdinalEmbedding(1, 128)
        self.feature_x = feature_Net4s_tp4_ferturex().cuda()
        self.args = get_args()
        #self.QL = nn.Parameter(torch.randn(1, 128))
        self.QL = nn.Parameter(torch.randn(1, 128) * nn.init.xavier_normal_(torch.empty(1, 128)))
    def forward(self, logits1, logits2, logits3, x):
        feature_x0 = self.feature_x(x)
        logits1_ca = self.dp(logits1)
        logits2_ca = self.dp(logits2)
        logits3_ca = self.dp(logits3)
        sizeq = logits1_ca.shape[0]
        QL = self.QL.expand(sizeq, -1)
        Q1 = self.cam(QL, feature_x0, feature_x0)
        Q2 = self.cam(Q1, logits1_ca, logits1_ca)
        Q3 = self.cam(Q2, logits2_ca, logits2_ca)
        Q4 = self.cam(Q3, logits3_ca, logits3_ca)
        features_xx = Q4
        out_predit = self.classifier0(features_xx)
        return out_predit, features_xx

class feature_Net4s_tp4_remove_fenshu_adnc(nn.Module):
    def __init__(self, dropout=0.0):
        nn.Module.__init__(self)

        self.feature_extractor1 = nn.Sequential(
            nn.Conv3d(1, 16, 3),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, 3),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2),
            nn.Conv3d(16, 32, 3),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2),
            nn.Conv3d(32, 64, 3),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, 3),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2),
            nn.Conv3d(64, 128, 3),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 128, 1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
        )
        self.classifier0 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

        self.locat = PositionalEncoding3D(128)
        self.ca = SelfAttention(128)
        self.de = DecoderSelfAttention(128)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        features_x = self.feature_extractor1(x)
        features_x = self.locat(features_x)
        map = self.ca(features_x)
        features_x = self.de(features_x, map) * features_x
        features_x = self.pool(features_x)
        features_x = features_x.view(features_x.shape[0], -1)
        output = self.classifier0(features_x)
        return output, features_x


class feature_Net4s_tp1(nn.Module):
    def __init__(self, dropout=0.3):
        nn.Module.__init__(self)

        self.feature_extractor1 = nn.Sequential(
            nn.Conv3d(1, 16, 3),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, 3),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2),
        )

        self.feature_extractor2 = nn.Sequential(
            nn.Conv3d(16, 32, 3),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2),
            nn.Conv3d(32, 64, 3),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, 3),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2),
            nn.Conv3d(64, 128, 3),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 128, 1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
        )

        self.classifier1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.classifier2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.classifier3 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.locat = PositionalEncoding3D(16)
        self.ca = SelfAttention(16)
        self.de = DecoderSelfAttention(16)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        features_x = self.feature_extractor1(x)
        features_x = self.locat(features_x)
        map = self.ca(features_x)
        features_x = self.de(features_x, map) * features_x
        features_x = self.feature_extractor2(features_x)
        features_x = self.pool(features_x)

        features_x = features_x.view(features_x.shape[0], -1)
        logits1 = self.classifier1(features_x)
        logits2 = self.classifier2(features_x)
        logits3 = self.classifier3(features_x)
        return logits1, logits2, logits3, features_x


class feature_Net16(nn.Module):
    def __init__(self, dropout=0.5):
        nn.Module.__init__(self)
        # self.SpatialGate = SpatialGate()

        self.feature_extractor1 = nn.Sequential(
            nn.Conv3d(1, 16, 3),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, 3),
            nn.BatchNorm3d(16),
            nn.MaxPool3d(2, stride=2),
            nn.ReLU(),
            nn.Conv3d(16, 32, 3),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3),
            nn.BatchNorm3d(32),
            nn.MaxPool3d(2, stride=2),
            nn.ReLU(),
            nn.Conv3d(32, 64, 3),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, 3),
            nn.BatchNorm3d(64),
            nn.MaxPool3d(2, stride=2),
            nn.ReLU(),
            nn.Conv3d(64, 128, 3),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 128, 3),
            nn.BatchNorm3d(128),
            nn.MaxPool3d(2, stride=2),
            nn.ReLU(),
            nn.Conv3d(128, 256, 3),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            # nn.Conv3d(256, 256, 3),
            # nn.BatchNorm3d(256),
        )

        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.classifier2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.classifier3 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        features_x = self.feature_extractor1(x)
        features_x = self.pool(features_x)
        # print(features_x.shape)
        features_x = features_x.view(features_x.shape[0], -1)
        # print(features_x.shape)
        logits1 = self.classifier1(features_x)
        logits2 = self.classifier2(features_x)
        logits3 = self.classifier3(features_x)
        return logits1, logits2, logits3


class Net_raw(nn.Module):
    def __init__(self, dim=256, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, dim // 8, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(dim // 8),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(dim // 8, dim // 4, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(dim // 4),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(dim // 4, dim // 2, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(dim // 2),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(dim // 2, dim, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(dim),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        self.ca = SelfAttention(256)
        self.classifier1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
        )
        self.classifier2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
        )
        self.classifier3 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)
        conv5_out = self.ca(conv5_out) * conv5_out
        out = conv5_out.view(conv5_out.size(0), -1)
        logits1 = self.classifier1(out)
        logits2 = self.classifier2(out)
        logits3 = self.classifier3(out)
        return logits1, logits2, logits3


'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x1 = torch.randn(2, 1, 121, 145, 121).to(device) 
x2 = torch.randn(2, 1, 121, 145, 121).to(device) 
model1 = feature_Net().to(device) 
output = model1(x1, x2)
yyy = output.shape
print(yyy)
'''
