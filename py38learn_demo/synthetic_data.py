import random
from pyexpat import features
import matplotlib.pyplot as plt
import torch

def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))   # 必须返回

true_w = torch.tensor([2.0, -3.4])  # 改成 float
true_b = 4.2

trian_features, train_labels = synthetic_data(true_w, true_b, 1000)

print(trian_features.shape)
print(train_labels.shape)