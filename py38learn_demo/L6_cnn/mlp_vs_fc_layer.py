import torch
import torch.nn as nn

# =====================================================================
# 1. 什么是【全连接层】（Fully Connected Layer）？
# 答：它只是一个“单层组件/砖块”。在 PyTorch 中用 nn.Linear 表示。
# =====================================================================
class SingleFCLayerDemo(nn.Module):
    def __init__(self, input_dim=4, output_dim=2):
        super().__init__()
        # 这就是一个单孤零零的【全连接层】
        self.fc_layer = nn.Linear(in_features=input_dim, out_features=output_dim)

    def forward(self, x):
        return self.fc_layer(x)


# =====================================================================
# 2. 什么是【多层感知机】（MLP, Multilayer Perceptron）？
# 答：它是一座“完整的房子”。是由【多个全连接层】和【非线性激活函数】串联组合而成的完整网络。
# =====================================================================
class MLPNetworkDemo(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=8, output_dim=2):
        super().__init__()
        # 这就是一个标准的【多层感知机（MLP）】架构
        self.mlp_network = nn.Sequential(
            # 第一层全连接：把输入特征扩展到隐藏层
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.ReLU(),  # 激活函数：MLP的灵魂，不加激活函数的堆叠毫无意义
            
            # 第二层全连接：在隐藏层内部进一步提炼特征
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),  # 激活函数
            
            # 第三层全连接：输出层，映射到最终的分类/回归结果
            nn.Linear(in_features=hidden_dim, out_features=output_dim)
        )

    def forward(self, x):
        return self.mlp_network(x)


# =====================================================================
# 3. 运行测试：从打印的结构和参数量，一眼看清两者的区别
# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print(" 核心对比：全连接层（组件） VS 多层感知机（完整网络）")
    print("=" * 60)

    # 模拟输入数据：假设有 3 个样本，每个样本有 4 个特征（例如：[身高, 体重, 年龄, 血压]）
    dummy_input = torch.randn(3, 4)

    # ---- 测试单个全连接层 ----
    fc_model = SingleFCLayerDemo(input_dim=4, output_dim=2)
    fc_output = fc_model(dummy_input)
    
    print("\n【📌 1. 单个全连接层组件展示】")
    print(fc_model)
    print(f"输入形状: {dummy_input.shape} -> 输出形状: {fc_output.shape}")

    # ---- 测试多层感知机网络 ----
    mlp_model = MLPNetworkDemo(input_dim=4, hidden_dim=8, output_dim=2)
    mlp_output = mlp_model(dummy_input)
    
    print("\n【📌 2. 完整的多层感知机（MLP）网络展示】")
    print(mlp_model)
    print(f"输入形状: {dummy_input.shape} -> 输出形状: {mlp_output.shape}")

    print("\n" + "=" * 60)
    print("💡 结论总结：")
    print("1. 全连接层（nn.Linear）是代码里的一行，代表一个局部的『线性计算组件』。")
    print("2. 多层感知机（MLP）是代码里的一块，是把『多个全连接层+激活函数』打包组装起来的『网络架构』。")
    print("=" * 60)