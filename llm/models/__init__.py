# 从各个模块导入模型类
from .rnn import RNNModel
from .tfs import TransformerModel, PositionalEncoding
from .speech_tfs import SpeechAwareTransformer, LocalMultiheadAttention, TransformerLayerBlock

# 声明公开的模型类名称，用于 from models import * 时导入这些类
__all__ = [
    'RNNModel',               # RNN/LSTM/GRU 模型
    'TransformerModel',       # 标准 Transformer 模型
    'PositionalEncoding',     # 标准位置编码
    'SpeechAwareTransformer', # 语音感知 Transformer 模型
    'LocalMultiheadAttention', # 局部多头注意力
    'TransformerLayerBlock',   # Transformer 层封装
]