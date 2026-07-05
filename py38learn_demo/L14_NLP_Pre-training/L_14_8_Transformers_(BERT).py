# 来自Transformers的双向编码器表示（BERT）

import torch
from torch import nn
# 🔔【D2L 关键点 1】：导入 D2L 库的 PyTorch 版本。
# 后续所有以 d2l. 开头的组件（如底层 Transformer 块）都由此提供。
from d2l import torch as d2l

# =====================================================================
# 1. 辅助功能：数据预处理
# =====================================================================
def get_tokens_and_segments(tokens_a, tokens_b=None):
    """获取输入序列的词元及其片段索引"""
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0和1分别标记片段A和B（即第一句和第二句）
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments

# =====================================================================
# 2. 核心组件一：BERT 编码器
# =====================================================================
class BERTEncoder(nn.Module):
    """BERT编码器：利用D2L的EncoderBlock堆叠而成，负责抽取文本深层特征"""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        # 三种嵌入层
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, num_hiddens))
        
        self.blks = nn.Sequential()
        for i in range(num_layers):
            # 🔔【D2L 关键点 2】：调用 d2l.EncoderBlock。
            # 这是 D2L 自定义的 Transformer 编码器块。它内部优雅地封装了：
            # 多头自注意力（Multi-head Attention） + 位置感知前馈网络（FFN） + 残差连接（Residual） + 层规范化（LayerNorm）。
            # 传参中的 norm_shape, ffn_num_input, ffn_num_hiddens 等都是 D2L 块所要求的特定超参数。
            self.blks.add_module(f"{i}", d2l.EncoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads, dropout, use_bias=True))

    def forward(self, tokens, segments, valid_lens):
        # 将三种 Embedding 相加融合
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        
        # 逐层通过 D2L 的 EncoderBlock 块
        for blk in self.blks:
            # 🔔【D2L 关键点 3】：传递 valid_lens（有效长度）给 D2L 的模块。
            # 这是 D2L 的一大特色设计：在计算注意力时，blk 内部会自动根据 valid_lens 
            # 屏蔽掉（Mask）短句子后面无意义的 Padding 占位符，防止它们干扰模型的注意力计算。
            X = blk(X, valid_lens)
        return X

# =====================================================================
# 3. 核心组件二：掩蔽语言模型任务 (MLM)
# =====================================================================
class MaskLM(nn.Module):
    """BERT预训练任务 1：掩蔽语言模型（完形填空分支）"""
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        # 用纯 PyTorch 原生线性层构建的小型多层感知机（MLP）
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, vocab_size))

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        
        # 核心技巧：使用高级索引，只把被 mask 位置的向量“抠”出来
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        
        # 通过 MLP 映射到全词表空间，输出预测概率
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat

# =====================================================================
# 4. 核心组件三：下一句预测任务 (NSP)
# =====================================================================
class NextSentencePred(nn.Module):
    """BERT预训练任务 2：下一句预测（二分类分支）"""
    def __init__(self, num_inputs, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Linear(num_inputs, 2)

    def forward(self, X):
        return self.output(X)

# =====================================================================
# 5. 终极整合：完整的 BERT 模型
# =====================================================================
class BERTModel(nn.Module):
    """整合整个 BERT 模型架构"""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 hid_in_features=768, mlm_in_features=768, nsp_in_features=768):
        super(BERTModel, self).__init__()
        # 组装前面定义好的、包含 D2L 块的编码器
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape,
                    ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                    dropout, max_len=max_len, key_size=key_size,
                    query_size=query_size, value_size=value_size)
        
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens),
                                    nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        # 🔔【D2L 关键点 4】：在此处显式将 valid_lens 喂给编码器。
        # 确保数据在流经底层的 D2L EncoderBlock 时能正确应用 Padding 遮蔽。
        encoded_X = self.encoder(tokens, segments, valid_lens)
        
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
            
        # 提取句首第一个标记 <cls>（索引为0）的向量，运行 NSP 分支
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        
        return encoded_X, mlm_Y_hat, nsp_Y_hat

# =====================================================================
# 6. 测试运行与验证
# =====================================================================
if __name__ == "__main__":
    # 🔔【D2L 关键点 5】：D2L 独特的超参数配置风格。
    # D2L 书中实验为了在单张显卡或 CPU 上快速跑通，常采用较小的模型：
    # 比如这里 num_layers=2（原版 BERT-Base 是 12 层），num_heads=4（原版是 12 个）。
    # 此外，norm_shape 和 ffn_num_input 输入维度与层规范化形状也是 D2L 特有的显式参数配置。
    vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
    norm_shape, ffn_num_input, num_layers, dropout = [768], 768, 2, 0.2
    
    # 初始化模型
    bert = BERTModel(vocab_size, num_hiddens, norm_shape, ffn_num_input,
                     ffn_num_hiddens, num_heads, num_layers, dropout)
    
    # 创造模拟数据
    tokens = torch.randint(0, vocab_size, (2, 8))
    segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], 
                             [0, 0, 0, 1, 1, 1, 1, 1]])
    
    # 🔔【D2L 关键点 6】：构造符合 D2L 自带机制的有效长度张量。
    # 告诉模型：第一条序列 8 个词全有效；第二条序列只有前 7 个词有效，第 8 个是用来对齐的补丁（Padding）。
    valid_lens = torch.tensor([8, 7]) 
    mlm_positions = torch.tensor([[1, 5, 2], [6, 1, 5]]) 
    
    # 前向传播
    encoded_X, mlm_Y_hat, nsp_Y_hat = bert(tokens, segments, valid_lens, mlm_positions)
    
    print("✓ 模型整合成功并成功运行！")
    print(f"-> 语言特征表示 encoded_X 形状:  {encoded_X.shape}")
    print(f"-> 完形填空任务 mlm_Y_hat  形状: {mlm_Y_hat.shape}")
    print(f"-> 下一句预测任务 nsp_Y_hat  形状: {nsp_Y_hat.shape}")

