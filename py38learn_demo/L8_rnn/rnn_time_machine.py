"""import os
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from d2l import torch as d2l

# =====================================================================
# 1. 核心模型大类（保持你的原汁原味，增加了设备自动对齐）
# =====================================================================
class RNNModel(nn.Module):
    #循环神经网络骨架大类
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        
        # 自动兼容双向网络配置
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        # 转换输入形状至 Time-Step First: [num_steps, batch_size, vocab_size]
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # 压扁展平前向传播特征，送入线性分类层
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.RNN 和 nn.GRU 使用张量存储状态
            return torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens), device=device)
        else:
            # nn.LSTM 使用元组 (H, C) 存储状态
            return (torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens), device=device),
                    torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens), device=device))


# =====================================================================
# 2. 训练中枢总调度大类（核心重构部分）
# =====================================================================
class RNNTrainer:
    #训练总调度流水线大类，负责所有训练战术和策略的统筹
    def __init__(self, net, vocab, lr=1.0, theta=1, use_random_iter=False):
        self.net = net
        self.vocab = vocab
        self.lr = lr
        self.theta = theta  # 梯度裁剪阈值
        self.use_random_iter = use_random_iter
        
        # 定义内部微观算子：损失函数与优化器
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr)
        
        # 获取当前模型所在的硬件设备，实现动态对齐
        self.device = next(net.parameters()).device

    def grad_clipping(self):
        #微观防空洞：梯度裁剪
        params = [p for p in self.net.parameters() if p.requires_grad]
        norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in params))
        if norm > self.theta:
            for param in params:
                param.grad[:] *= self.theta / norm

    def predict(self, prefix, num_preds):
        #文本生成与预测（预热机制）
        state = self.net.begin_state(batch_size=1, device=self.device)
        outputs = [self.vocab[prefix[0]]]
        
        get_input = lambda: torch.tensor([outputs[-1]], device=self.device).reshape((1, 1))
        
        # 1. 预热期：只吃前缀更新隐藏状态，不记录输出
        for y in prefix[1:]:
            _, state = self.net(get_input(), state)
            outputs.append(self.vocab[y])
            
        # 2. 预测期：正式向后续写预测
        for _ in range(num_preds):
            y, state = self.net(get_input(), state)
            outputs.append(int(y.argmax(dim=1).reshape(1)))
            
        return ''.join([self.vocab.idx_to_token[i] for i in outputs])

    def _train_epoch(self, train_iter):
        #执行单轮 Epoch 里的数据接力
        state = None
        metric = d2l.Accumulator(2)  # 累加器：[损失之和, 词元数量]
        
        for X, Y in train_iter:
            # 根据迭代器类型，决定是初始化还是分离隐藏状态的记忆小本本
            if state is None or self.use_random_iter:
                state = self.net.begin_state(batch_size=X.shape[0], device=self.device)
            else:
                if isinstance(state, tuple):
                    state = (s.detach() for s in state)
                else:
                    state.detach_()
            
            # 时间步转置，对齐张量序列
            y = Y.T.reshape(-1)
            X, y = X.to(self.device), y.to(self.device)
            
            # 前向传播调用
            y_hat, state = self.net(X, state)
            loss = self.loss_fn(y_hat, y.long()).mean()
            
            # 反向传播与优化
            self.optimizer.zero_grad()
            loss.backward()
            self.grad_clipping()  # 调用内部的裁剪函数
            self.optimizer.step()
            
            metric.add(loss * y.numel(), y.numel())
            
        # 返回当前轮次的困惑度
        return math.exp(metric[0] / metric[1])

    def fit(self, train_iter, num_epochs):
        #启动宏观循环，总领整个训练生命周期
        print(f"[*] 启动面向对象化大类训练，设备: {self.device}")
        
        animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                                legend=['train'], xlim=[1, num_epochs])

        for epoch in range(num_epochs):
            ppl = self._train_epoch(train_iter)
            
            # 每 50 次日志汇报与测试续写
            if (epoch + 1) % 50 == 0:
                pred_text = self.predict('time traveller', 10)
                print(f'Epoch {epoch + 1:3d} | 困惑度: {ppl:.2f} | 续写测试: {pred_text}')
                animator.add(epoch + 1, [ppl])
                
        print(f'\n[+] 训练完成。最终困惑度: {ppl:.1f}')
        print(f"-> 最终长续写 1: {self.predict('time traveller', 50)}")
        print(f"-> 最终长续写 2: {self.predict('traveller', 50)}")


# =====================================================================
# 3. 宏观调度入口（数据采购与大类拼装）
# =====================================================================
if __name__ == '__main__':
    # 1. 统一获取硬件环境
    device = d2l.try_gpu()
    print(f'使用设备: {device}')

    # 2. 采购数据流（对齐落入当前文件的统一目录）
    batch_size, num_steps = 32, 35
    data = d2l.TimeMachine(batch_size, num_steps)
    train_loader = data.get_dataloader(True)
    dataset = train_loader.dataset
    train_iter = DataLoader(dataset, batch_size, shuffle=False, drop_last=True)
    vocab = data.vocab

    print(f'词表大小: {len(vocab)} | 训练批次数量: {len(train_iter)}')

    # 3. 采购并组装微观零件层
    num_hiddens = 256
    rnn_layer = nn.RNN(len(vocab), num_hiddens)
    net = RNNModel(rnn_layer, vocab_size=len(vocab)).to(device)

    # 4. 实例化训练调度中枢大类
    trainer = RNNTrainer(net=net, vocab=vocab, lr=1.0, theta=1)
    
    # 5. 训练前随机预测试
    print('训练前随机续写:', trainer.predict('time traveller', 10))
    print("-" * 50)

    # 6. 一键点火，开始训练
    trainer.fit(train_iter, num_epochs=500)
"""



import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 超级简易的“宇宙数据源”
# ==========================================
# 假设我们想让网络学会一件事：看到字母 'a'，就预测下一个是 'b'；看到 'b'，就预测下一个是 'c'
# 我们的词表只有 3 个字符：['a', 'b', 'c']，大小 (vocab_size) 为 3
token_to_idx = {'a': 0, 'b': 1, 'c': 2}
idx_to_token = ['a', 'b', 'c']
vocab_size = 3

# 模拟一个 Batch 的训练数据
# 输入 inputs 代表 ['a', 'b']，换成数字就是 [0, 1]
# 标签 labels 代表正确的预测应该是 ['b', 'c']，数字就是 [1, 2]
inputs = torch.tensor([[0, 1]])  # 形状: [batch_size=1, num_steps=2]
labels = torch.tensor([[1, 2]])  # 形状: [batch_size=1, num_steps=2]

# ==========================================
# 2. 简易 RNN 网络模型定义（麻雀虽小，五脏俱全）
# ==========================================
class EasyRNN(nn.Module):
    def __init__(self, vocab_size, num_hiddens):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        
        # 零件 A：官方核心 RNN 单元（输入通道 3，隐藏特征 5）
        self.rnn = nn.RNN(input_size=vocab_size, hidden_size=num_hiddens, batch_first=True)
        # 零件 B：全连接分类收割层（把 5 维记忆特征还原回 3 维的字符概率）
        self.fc = nn.Linear(num_hiddens, vocab_size)

    def forward(self, x, state):
        # 1. 独热编码：数字变高维特征向量。[1, 2] -> [1, 2, 3] (float32 必须强转)
        x_one_hot = F.one_hot(x, self.vocab_size).float()
        
        # 2. 趟过循环发动机：拿到所有时间步的特征 out 和最新的记忆小本本 state
        out, state = self.rnn(x_one_hot, state)
        
        # 3. 分类收割：把三维特征压扁成二维矩阵，送入全连接层
        # out 形状: [1, 2, 5] -> reshape 后变为 [2, 5]
        out_flattened = out.reshape(-1, out.shape[-1])
        output = self.fc(out_flattened)
        
        return output, state

# ==========================================
# 3. 实例化与五连鞭训练循环
# ==========================================
num_hiddens = 5  # 隐藏层设置 5 维特征记忆
net = EasyRNN(vocab_size, num_hiddens)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

print("[*] 极简 RNN 开始特训...")
for epoch in range(100):
    # 每一轮开始，初始化一张干净的全零记忆小本本 [层的数量=1, batch_size=1, num_hiddens=5]
    state = torch.zeros((1, 1, num_hiddens))
    
    # 1. 前向传播
    y_hat, state = net(inputs, state)
    
    # 2. 计算损失：标签也要压扁成一维长向量 [1, 2] -> [2]
    loss = loss_fn(y_hat, labels.reshape(-1))
    
    # 3. 反向传播更新三连
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:3d} | 交叉熵损失 (Loss): {loss.item():.4f}")

# ==========================================
# 4. 见证奇迹的盲猜预测期
# ==========================================
print("\n[+] 训练结束，开始闭卷考试！")
with torch.no_grad():
    test_input = torch.tensor([[0]])  # 给网络喂一个字母 'a' (数字 0)
    test_state = torch.zeros((1, 1, num_hiddens))
    
    # 让网络预测
    pred_scores, _ = net(test_input, test_state)
    pred_idx = pred_scores.argmax(dim=1).item()
    
    print(f" -> 当我们输入字符: '{idx_to_token[0]}'")
    print(f" -> 循环神经网络预测的下一个字符是: '{idx_to_token[pred_idx]}'")


