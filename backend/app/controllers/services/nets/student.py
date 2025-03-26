import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # SEAGenerator 通常生成 3 个特征
        input_size = 3
        
        # 简单的学生模型架构
        self.fc1 = nn.Linear(input_size, 8)
        self.fc2 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 确保输入是二维张量
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.sigmoid(x)