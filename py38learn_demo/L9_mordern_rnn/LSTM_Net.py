import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 超级简易的“宇宙数据源”
# ==========================================
token_to_idx = {'a': 0, 'b': 1, 'c': 2}
idx_to_token = ['a', 'b', 'c']
vocab_size = 3

inputs = torch.tensor([[0, 1]])  # 输入 ['a', 'b'] -> [0, 1]
labels = torch.tensor([[1, 2]])  # 标签 ['b', 'c'] -> [1, 2]

# ==========================================
# 2. 极简 LSTM 模型定义
# ==========================================
class EasyLSTM(nn.Module):
    def __init__(self, vocab_size, num_hiddens):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        
        # 🚀 核心改动：把 nn.RNN 零件无缝抠掉，换成官方的 nn.LSTM 零件！
        self.lstm = nn.LSTM(input_size=vocab_size, hidden_size=num_hiddens, batch_first=True)
        # 分类收割全连接层保持不变
        self.fc = nn.Linear(num_hiddens, vocab_size)

    def forward(self, x, state):
        # 1. 独热编码把数字变成高维特征向量 [1, 2] -> [1, 2, 3]
        x_one_hot = F.one_hot(x, self.vocab_size).float()
        
        # 2. 趟过 LSTM 发动机
        # 🚀 注意：这里的 state 不再是一个单一的张量，而是一个包含了 (H, C) 的元组！
        out, state = self.lstm(x_one_hot, state)
        
        # 3. 压扁展平三维特征，送入全连接分类器
        out_flattened = out.reshape(-1, out.shape[-1])
        output = self.fc(out_flattened)
        
        return output, state

# ==========================================
# 3. 实例化与训练循环
# ==========================================
num_hiddens = 5
net = EasyLSTM(vocab_size, num_hiddens)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)

print("[*] 极简 LSTM 开始特训...")
for epoch in range(100):
    # 🚀 状态初始化关键：LSTM 的初始状态必须同时包含 H 和 C 两个全零张量！
    # 形状都是：[层数=1, batch_size=1, num_hiddens=5]
    h_0 = torch.zeros((1, 1, num_hiddens))
    c_0 = torch.zeros((1, 1, num_hiddens))
    state = (h_0, c_0)  # 打包成记忆元组
    
    # 1. 前向传播
    y_hat, state = net(inputs, state)
    
    # 2. 计算差距与反向传播
    loss = loss_fn(y_hat, labels.reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:3d} | 交叉熵损失 (Loss): {loss.item():.4f}")

# ==========================================
# 4. 闭卷考试
# ==========================================
print("\n[+] 训练结束，开始闭卷考试！")
with torch.no_grad():
    test_input = torch.tensor([[0]])  # 喂一个 'a'
    h_0 = torch.zeros((1, 1, num_hiddens))
    c_0 = torch.zeros((1, 1, num_hiddens))
    
    pred_scores, _ = net(test_input, (h_0, c_0))
    pred_idx = pred_scores.argmax(dim=1).item()
    
    print(f" -> 当我们输入字符: '{idx_to_token[0]}'")
    print(f" -> LSTM 预测的下一个字符是: '{idx_to_token[pred_idx]}'")