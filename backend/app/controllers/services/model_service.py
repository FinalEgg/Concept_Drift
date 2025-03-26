import torch
import torch.nn as nn
import torch.optim as optim
import os
import pickle
from skmultiflow.data import SEAGenerator
import time
from backend.app.controllers.services.nets.teacher import Net as TeacherNet
from backend.app.controllers.services.nets.student import Net as StudentNet


class ModelService:
    def __init__(self):
        # 检查是否有可用的GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"模型服务使用设备: {self.device}")
        
        # 修改模型存储路径为 services/weights
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = os.path.join(current_dir, 'weights')
        # 确保模型存储目录存在
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
    
    def train_teacher_model(self, device_id: str, epochs: int = 100, sample_size: int = 1000):
        """
        初始化并训练老师模型
        
        参数:
        device_id: 设备ID，用于保存模型
        epochs: 训练轮数
        sample_size: 生成的样本量
        
        返回:
        训练好的老师模型
        """
        print(f"开始为设备 {device_id} 训练老师模型...")
        start_time = time.time()
        
        # 生成训练数据
        stream = SEAGenerator(random_state=42)
        X, y = stream.next_sample(sample_size)
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)  # 改为FloatTensor
        
        # 创建老师模型
        teacher_model = TeacherNet().to(self.device)
        
        # 训练老师模型
        criterion = nn.BCELoss()  # 改用BCELoss
        optimizer = torch.optim.Adam(teacher_model.parameters())
        
        for epoch in range(epochs):
            outputs = teacher_model(X_tensor).squeeze()  # 确保维度匹配
            loss = criterion(outputs, y_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (epoch+1) % 20 == 0:
                print(f"老师模型训练 - Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        
        # 评估老师模型准确率
        with torch.no_grad():
            predictions = teacher_model(X_tensor).squeeze()
            predicted_labels = (predictions > 0.5).float()
            teacher_acc = (predicted_labels == y_tensor).float().mean().item()
        
        print(f"老师模型训练完成，用时 {time.time() - start_time:.2f} 秒")
        print(f"老师模型准确率: {teacher_acc:.4f}")
        
        # 保存模型权重
        model_path = os.path.join(self.model_dir, f'teacher_{device_id}.pt')
        torch.save(teacher_model.state_dict(), model_path)
        print(f"老师模型已保存到 {model_path}")
        
        return teacher_model
    
    def train_student_model(self, teacher_model, device_id: str, epochs: int = 100, sample_size: int = 1000):
        """
        训练学生模型(老师模型的蒸馏模型)
        
        参数:
        teacher_model: 训练好的老师模型
        device_id: 设备ID，用于保存模型
        epochs: 训练轮数
        sample_size: 生成的样本量
        
        返回:
        学生模型的状态字典
        """
        print(f"开始为设备 {device_id} 训练学生模型...")
        start_time = time.time()
        
        # 生成训练数据
        stream = SEAGenerator(random_state=42)
        X, y = stream.next_sample(sample_size)
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # 创建学生模型
        student_model = StudentNet().to(self.device)
        
    
        # 获取老师模型的输出
        with torch.no_grad():
            teacher_outputs = teacher_model(X_tensor).squeeze()
        
        # 训练学生模型(知识蒸馏)
        optimizer = torch.optim.Adam(student_model.parameters())
        
        for epoch in range(epochs):
            student_outputs = student_model(X_tensor).squeeze()
            # 修改：使用合适的知识蒸馏损失
            # 直接用MSE损失代替KLDivLoss可能更简单
            loss = nn.MSELoss()(student_outputs, teacher_outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (epoch+1) % 20 == 0:
                print(f"学生模型训练 - Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        
        # 评估学生模型准确率（修改评估部分）
        with torch.no_grad():
            y_tensor = torch.FloatTensor(y).to(self.device)
            student_predictions = student_model(X_tensor).squeeze()
            predicted_labels = (student_predictions > 0.5).float()
            student_acc = (predicted_labels == y_tensor).float().mean().item() 

        print(f"学生模型训练完成，用时 {time.time() - start_time:.2f} 秒")
        print(f"学生模型准确率: {student_acc:.4f}")
        
        # 保存模型权重
        model_path = os.path.join(self.model_dir, f'student_{device_id}.pt')
        torch.save(student_model.state_dict(), model_path)
        print(f"学生模型已保存到 {model_path}")
        
        # 返回模型参数
        cpu_state_dict = {k: v.cpu() for k, v in student_model.state_dict().items()}
        return cpu_state_dict
    
    def load_model(self, device_id: str, model_type: str = 'teacher'):
        """
        读取模型
        
        参数:
        device_id: 设备ID
        model_type: 模型类型，'teacher'或'student'
        
        返回:
        加载了参数的模型
        """
        model_path = os.path.join(self.model_dir, f'{model_type}_{device_id}.pt')
        
        if not os.path.exists(model_path):
            print(f"找不到设备 {device_id} 的 {model_type} 模型")
            return None
        
        try:
            if model_type == 'teacher':
                model = TeacherNet()
                model.load_state_dict(torch.load(model_path, map_location=self.device))
            else:
                model = StudentNet()
                model.load_state_dict(torch.load(model_path, map_location=self.device))
            
            model.to(self.device)
            model.eval()  # 设置为评估模式
            
            print(f"成功加载设备 {device_id} 的 {model_type} 模型")
            return model
        
        except Exception as e:
            print(f"加载模型时出错: {e}")
            return None
