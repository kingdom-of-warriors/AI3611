import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# +++ 定义 Transformer 层块 +++
class TransformerLayerBlock(nn.Module):
    """将 Transformer 层封装成一个 Module"""
    def __init__(self, ninp, nhead, nhid, dropout, focal_window):
        super().__init__()
        # 局部窗口自注意力
        self.local_attn = LocalMultiheadAttention(ninp, nhead, focal_window, dropout)
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(ninp, nhid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(nhid, ninp),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(ninp)
        self.norm2 = nn.LayerNorm(ninp)
        self.dropout = nn.Dropout(dropout) # 添加 dropout 层（如果需要应用在注意力或 FFN 输出之后）

    def forward(self, x, mask):
        # 局部多头注意力 + 残差 + 层归一化
        attn_output = self.local_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output)) # 应用 dropout
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output)) # 应用 dropout
        return x


class SpeechAwareTransformer(nn.Module):
    """专为语音转录类数据优化的Transformer模型"""
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.2, 
                 fixed_len=128, use_lstm_hybrid=True, focal_window=32):
        super(SpeechAwareTransformer, self).__init__()
        
        self.model_type = 'SpeechAwareTransformer'
        self.ninp = ninp
        self.focal_window = focal_window
        
        # 嵌入层
        self.encoder = nn.Embedding(ntoken, ninp)
        self.pos_encoder = nn.Embedding(fixed_len, ninp)
        
        # 混合LSTM-Transformer结构
        self.use_lstm_hybrid = use_lstm_hybrid
        if use_lstm_hybrid:
            self.lstm_layer = nn.LSTM(ninp, ninp, 1, dropout=dropout, batch_first=False)
        
        # 局部注意力Transformer层 - 修改部分开始
        self.transformer_layers = nn.ModuleList() # 直接初始化 ModuleList
        for _ in range(nlayers):
            # 创建 TransformerLayerBlock 实例并添加到 ModuleList
            layer_block = TransformerLayerBlock(ninp, nhead, nhid, dropout, focal_window)
            self.transformer_layers.append(layer_block)
        
        # 解码器
        self.decoder = nn.Linear(ninp, ntoken)
        
        # 初始化
        self.init_weights()
        
    def init_weights(self):
        # LSTM风格初始化
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.uniform_(self.pos_encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)
        
    def _generate_local_mask(self, seq_len):
        """生成局部注意力掩码，限制每个位置只关注周围focal_window个词"""
        mask = torch.ones(seq_len, seq_len)
        for i in range(seq_len):
            start = max(0, i - self.focal_window)
            end = min(seq_len, i + self.focal_window + 1)
            mask[i, start:end] = 0
        
        return mask.masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))
        
    def forward(self, src, hidden=None):
        device = src.device
        seq_len, batch_size = src.size()
        
        # 局部注意力掩码
        local_mask = self._generate_local_mask(seq_len).to(device)
        # 嵌入
        src = self.encoder(src) * math.sqrt(self.ninp)
        
        # 位置编码
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(1).expand(seq_len, batch_size)
        pos_emb = self.pos_encoder(pos)
        x = src + pos_emb
        
        # 可选的LSTM层，模拟LSTM的局部序列建模能力
        if self.use_lstm_hybrid:
            if hidden is None:
                x, hidden = self.lstm_layer(x)
            else:
                x, hidden = self.lstm_layer(x, hidden)
        
        # Transformer层 - 修改部分开始
        for layer in self.transformer_layers: # 迭代 ModuleList 中的 Block 实例
            x = layer(x, local_mask) # 直接调用 Block 的 forward 方法
        
        # 解码
        output = self.decoder(x)
        
        return F.log_softmax(output, dim=-1)
        
# 辅助类：局部注意力机制
class LocalMultiheadAttention(nn.Module):
    """实现局部窗口的多头注意力"""
    def __init__(self, embed_dim, num_heads, window_size, dropout=0.0):
        super(LocalMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # 确保 window_size 是整数，否则可能在掩码生成时出错
        self.window_size = int(window_size)
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None):
        seq_len, batch_size, embed_dim = query.size()

        # 线性变换并重塑: (Seq, Batch, Embed) -> (Batch, Head, Seq, Head_Dim)
        q = self.q_proj(query).view(seq_len, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        k = self.k_proj(key).view(seq_len, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        v = self.v_proj(value).view(seq_len, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

        # 注意力计算: (Batch, Head, Seq, Head_Dim) x (Batch, Head, Head_Dim, Seq) -> (Batch, Head, Seq, Seq)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)


        if attn_mask is not None:
            scores = scores + attn_mask.unsqueeze(0).unsqueeze(0) # 扩展以匹配 (Batch, Head, Seq, Seq)


        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # 应用注意力权重: (Batch, Head, Seq, Seq) x (Batch, Head, Seq, Head_Dim) -> (Batch, Head, Seq, Head_Dim)
        attn_output = torch.matmul(attn_weights, v)

        # 重塑回原始形状: (Batch, Head, Seq, Head_Dim) -> (Seq, Batch, Embed)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(seq_len, batch_size, embed_dim)

        return self.out_proj(attn_output)