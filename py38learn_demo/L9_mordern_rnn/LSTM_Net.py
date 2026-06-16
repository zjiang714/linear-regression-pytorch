import torch
import torch.nn as nn
import torch.nn.functional as F


class CompleteLSTMModel(nn.Module):
    """一个高度内聚、自己管理隐状态的高级 LSTM 网络大类"""
    def __init__(self, vocab_size, num_hiddens):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        
        # 核心零件 A：官方 LSTM 单元 (batch_first=True 让输入形状更符合人类直觉)
        self.lstm = nn.LSTM(input_size=vocab_size, hidden_size=num_hiddens, batch_first=True)
        # 核心零件 B：全连接分类收割层
        self.fc = nn.Linear(num_hiddens, vocab_size)

    def begin_state(self, batch_size, device):
        """
        🚀 完美的自收纳！把初始化 (H, C) 的逻辑彻底锁死在大类内部。
        并且通过外部传入的 device 强行让状态和网络老老实实呆在同一张显卡上。
        """
        h_0 = torch.zeros((1, batch_size, self.num_hiddens), device=device)
        c_0 = torch.zeros((1, batch_size, self.num_hiddens), device=device)
        return (h_0, c_0)  # 打包成记忆元组返回

    def forward(self, x, state):
        """前向传播指挥部"""
        # 1. 独热编码把数字变成高维特征向量：形状从 [batch_size, num_steps] 蜕变为 [batch_size, num_steps, vocab_size]
        x_one_hot = F.one_hot(x, self.vocab_size).float()
        
        # 2. 趟过循环发动机：拿到所有时间步的特征 out 和最新的记忆小本本 state 元组
        out, state = self.lstm(x_one_hot, state)
        
        # 3. 分类收割：把三维特征矩阵压扁成二维，送入全连接层
        # out 形状: [batch_size, num_steps, num_hiddens] -> [batch_size * num_steps, num_hiddens]
        out_flattened = out.reshape(-1, out.shape[-1])
        output = self.fc(out_flattened)
        
        return output, state


# =====================================================================
# 2. 宏观控制台：原材料准备与一键点火训练
# =====================================================================
if __name__ == '__main__':
    # 1. 自动嗅探当前的硬件环境（有显卡用 CUDA，没显卡用 CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] 当前代码运行硬件环境: {device}")

    # 2. 准备宇宙简易数据集（看到 a 预测 b，看到 b 预测 c）
    token_to_idx = {'a': 0, 'b': 1, 'c': 2}
    idx_to_token = ['a', 'b', 'c']
    vocab_size = 3

    # 模拟打包好的 Batch 数据（形状: [batch_size=1, num_steps=2]）
    inputs = torch.tensor([[0, 1]], device=device)  # 输入 ['a', 'b']
    labels = torch.tensor([[1, 2]], device=device)  # 标签 ['b', 'c']

    # 3. 实例化我们封装好的高级大类，并一键推入硬件中
    num_hiddens = 5
    net = CompleteLSTMModel(vocab_size, num_hiddens)
    net = net.to(device)

    # 4. 聘请执行“梯度下降”的教练大管家和损失函数
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.2)

    # 5. 轰轰烈烈的特训主循环
    print("\n[*] 启动面向对象化大类特训...")
    print("-" * 40)
    
    for epoch in range(100):
        # 🌟 外部调用变得极其丝滑！直接跟大类索要初始化的 (H, C) 记忆
        state = net.begin_state(batch_size=inputs.shape[0], device=device)
        
        # 训练五连鞭第一鞭：前向传播
        y_hat, state = net(inputs, state)
        
        # 第二鞭：计算 Loss (把标签也压扁成一维长向量以一一对齐)
        loss = loss_fn(y_hat, labels.reshape(-1))
        
        # 第三、四、五鞭：擦除老梯度 -> 反向传播算账 -> 优化器迈步动手改权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 每隔 20 轮汇报一次战果
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d} | 交叉熵损失 (Loss): {loss.item():.4f}")

    print("-" * 40)
    print("[+] 训练完成！")

    # =====================================================================
    # 6. 盲猜预测期（闭卷考试模式）
    # =====================================================================
    # 🚀 开启全局“只看答案、不要梯度、暴省显存”的安全防误触模式
    with torch.no_grad():
        test_char = 'a'
        test_input = torch.tensor([[token_to_idx[test_char]]], device=device)  # 包装成张量送入
        
        # 再次利用大类自收纳的方法获取干净的全零初始状态
        test_state = net.begin_state(batch_size=1, device=device)
        
        # 让网络进行盲猜
        pred_scores, _ = net(test_input, test_state)
        
        # 用 argmax 捞出概率最高的那道单选题答案的索引
        pred_idx = pred_scores.argmax(dim=1).item()
        
        print(f"\n[+] 闭卷测试结果：")
        print(f" -> 当我们给网络喂入字符: '{test_char}'")
        print(f" -> 经过特训后的 LSTM 预测的下一个字符是: '{idx_to_token[pred_idx]}'")