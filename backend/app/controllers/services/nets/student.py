import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # SEAGenerator 通常生成 3 个特征
        input_size = 3
        
        # 极简学生模型 - 几乎就是个线性分类器
        self.fc = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()
        
        # 使用较小的初始权重，进一步限制模型能力
        nn.init.uniform_(self.fc.weight, -0.01, 0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        # 确保输入是二维张量
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        # 极简前向传播 - 没有非线性变换层
        x = self.fc(x)
        return self.sigmoid(x)