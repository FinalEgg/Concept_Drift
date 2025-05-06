import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # SEAGenerator 通常生成 3 个特征
        input_size = 3
        
        # 大幅缩减的教师模型架构
        self.fc1 = nn.Linear(input_size, 4)
        self.fc2 = nn.Linear(4, 2)
        self.fc3 = nn.Linear(2, 1)
        self.dropout = nn.Dropout(0.1)  # 轻微的dropout
        self.sigmoid = nn.Sigmoid()

        # 保留权重初始化以提高模型质量
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        # 确保输入是二维张量
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.sigmoid(x)