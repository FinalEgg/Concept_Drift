import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class ModelService:
    def __init__(self):
        # 可根据需要初始化其他参数
        pass

    def load_model_from_file(self, file_path: str, model_cls: type):
        """
        从文件读取模型参数，并返回加载完毕的模型
        参数:
            file_path: 模型参数文件路径
            model_cls: 定义神经网络结构的 nn.Module 子类
        返回:
            加载好参数的模型对象，如果加载失败返回 None
        """
        model = model_cls()
        try:
            # 若保存时使用的是 torch.save(model.state_dict(), file_path)
            checkpoint = torch.load(file_path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint)
            model.eval()
            return model
        except Exception as e:
            print(f"加载模型时出错: {e}")
            return None

    def train_model(self, model: nn.Module, X: torch.Tensor, y: torch.Tensor, 
                    epochs: int = 10, batch_size: int = 32, lr: float = 0.01):
        """
        使用指定数据集训练模型
        参数:
            model: 需要训练的神经网络模型
            X: 特征张量
            y: 标签张量
            epochs: 训练的轮数
            batch_size: 每个批次的样本数
            lr: 学习率
        返回:
            训练后的模型
        """
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss = loss.item()
            print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
        return model