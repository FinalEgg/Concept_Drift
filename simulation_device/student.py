import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 修改输入大小以匹配SEAGenerator生成的特征数量
        input_size = 3
        
        # 匹配后端学生模型架构
        self.fc1 = nn.Linear(input_size, 8)
        self.fc2 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.sigmoid(x)