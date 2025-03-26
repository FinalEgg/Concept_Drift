import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # SEAGenerator 通常生成 3 个特征
        input_size = 3
        
        # 简化的教师模型架构，去除规范化层
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, 1)
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 确保输入是二维张量
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return self.sigmoid(x)