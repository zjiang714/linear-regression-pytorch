import torch
import random


def create_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


def data_loader(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


def linreg(X, w, b):
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


# 这是一个封装好的“一键入口”函数
def run_linear_regression_experiment(true_w, true_b, lr=0.03, epochs=3, batch_size=10):
    """
    只需调用此函数，即可完成数据生成、训练及结果对比。
    """
    # 1. 生成数据
    X, y = create_data(true_w, true_b, 1000)

    # 2. 初始化参数
    w = torch.normal(0, 0.01, size=(len(true_w), 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    # 3. 训练循环
    for epoch in range(epochs):
        for batch_X, batch_y in data_loader(batch_size, X, y):
            l = squared_loss(linreg(batch_X, w, b), batch_y)
            l.sum().backward()
            sgd([w, b], lr, batch_size)

        with torch.no_grad():
            train_l = squared_loss(linreg(X, w, b), y)
            print(f'轮次 {epoch + 1}, 损失: {float(train_l.mean()):.6f}')

    return w, b

def read_data_set():
    def main():
        # 设置你想要的“真实规律”
        true_w = torch.tensor([2, -3.4])
        true_b = 4.2

        # 只调用这一个入口函数
        w_hat, b_hat = run_linear_regression_experiment(
            true_w=true_w,
            true_b=true_b,
            lr=0.03,
            epochs=5
        )

        print("\n--- 实验报告 ---")
        print(f"训练出的权重: {w_hat.reshape(-1).detach().numpy()}")
        print(f"训练出的偏置: {b_hat.item():.4f}")

