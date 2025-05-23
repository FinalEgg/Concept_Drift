# simulation_device/sim_device.py

import socket
import time
import random
import pickle
import struct
import torch
import numpy as np
from skmultiflow.data import SEAGenerator
from simulation_device.student import Net

class SimDevice:
    def __init__(self, host: str = '127.0.0.1', port: int = 9000):
        """
        初始化模拟设备，可修改 host 和 port 实现不同的监听地址与端口
        """
        self.host = host
        self.port = port
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.stream = SEAGenerator(random_state=42)

    def start(self):
        """
        启动模拟设备，动态选择空闲端口等待云服务器连接，连接成功后先接收模型参数
        然后每两秒生成一个数据，使用模型预测标签，并发送给服务器
        设备将持续运行，即使连接超时也会重新等待连接
        """
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        bind_port = self.port
        while True:
            try:
                server_sock.bind((self.host, bind_port))
                print(f"SimDevice 将在 {self.host}:{bind_port} 运行")
                break
            except OSError as e:
                print(f"端口 {bind_port} 被占用，尝试其他端口...")
                bind_port = random.randint(9000, 9999)
        
        server_sock.listen(1)
        print(f"SimDevice 正在 {self.host}:{bind_port} 监听连接...")
        
        # 外部循环，使设备持续运行
        while True:
            conn = None
            try:
                # 设置接受连接的超时时间
                server_sock.settimeout(60)  # 60秒超时
                conn, addr = server_sock.accept()
                server_sock.settimeout(None)  # 接受连接后取消超时
                
                print(f"接收到来自 {addr} 的连接")
                
                # 先接收模型参数，再发送数据
                self.receive_model(conn)
                self.send_data(conn)
                
            except socket.timeout:
                print("等待连接超时，继续监听...")
                continue
            except Exception as e:
                print(f"SimDevice 运行错误: {e}")
            finally:
                # 确保当前连接被关闭，但服务器套接字保持开启
                if conn:
                    try:
                        conn.close()
                    except:
                        pass
                print("连接已关闭，重新等待新的连接...")
    
    def receive_model(self, conn: socket.socket):
        """
        接收服务器发送的模型参数
        """
        try:
            print("等待接收模型参数...")
            
            # 设置接收超时
            conn.settimeout(30)  # 30秒超时
            
            # 接收数据大小
            data_size_bytes = conn.recv(4)
            if not data_size_bytes:
                raise Exception("接收模型大小失败")
                
            data_size = struct.unpack('!I', data_size_bytes)[0]
            print(f"准备接收 {data_size} 字节的模型参数")
            
            # 接收完整数据
            data = b''
            remaining = data_size
            while remaining > 0:
                chunk = conn.recv(min(remaining, 4096))
                if not chunk:
                    break
                data += chunk
                remaining -= len(chunk)
                print(f"已接收 {len(data)}/{data_size} 字节")
            
            if len(data) < data_size:
                raise Exception(f"接收到的数据不完整: {len(data)}/{data_size}")
            
            # 反序列化模型参数
            student_params = pickle.loads(data)
            print("模型参数接收完成")
            
            # 创建并加载模型
            self.model = Net().to(self.device)
            self.model.load_state_dict(student_params)
            self.model.eval()
            print("模型加载成功")
            
        except socket.timeout:
            print("接收模型参数超时")
            raise
        except Exception as e:
            print(f"接收模型参数时出错: {e}")
            raise

    def send_data(self, conn: socket.socket):
        """
        连接成功后，每两秒使用SEAGenerator产生一条数据，
        使用模型预测标签，并发送给服务器，在发送15次数据后发生概念漂移
        """
        try:
            print("开始发送数据...")
            if self.model is None:
                print("错误：模型未加载")
                return
            
            data_count = 0  # 添加计数器，用于跟踪发送的数据量
            
            # 设置发送数据超时
            conn.settimeout(5)  # 5秒超时
            
            while True:
                # 检查是否需要触发概念漂移
                if data_count == 20:
                    print("触发概念漂移...")
                    # 创建一个新的SEAGenerator，使用不同的分类函数
                    self.stream = SEAGenerator(classification_function=3, random_state=42)
                    print("概念漂移已触发，分类函数已从0更改为1")
                
                # 使用SEAGenerator生成一个样本
                X, y_true = self.stream.next_sample()
                
                # 转换为张量并预测
                X_tensor = torch.FloatTensor(X).to(self.device)
                
                with torch.no_grad():
                    # 确保维度正确
                    if X_tensor.dim() == 1:
                        X_tensor = X_tensor.unsqueeze(0)
                    
                    y_pred = self.model(X_tensor)
                    prediction = 1 if y_pred.item() > 0.5 else 0
                
                # 构造发送数据：添加一个漂移指数（根据是否发生漂移调整），后跟特征值和预测标签
                if data_count < 15:
                    drift_index = round(random.uniform(0, 0.3), 4)  # 漂移前，较小的值
                else:
                    drift_index = round(random.uniform(0.7, 1.0), 4)  # 漂移后，较大的值
                    
                data = [drift_index] + X[0].tolist() + [prediction]
                data_str = ','.join(map(str, data)) + "\n"
                
                conn.sendall(data_str.encode('utf-8'))
                print(f"发送数据 #{data_count + 1}: {data_str.strip()}")
                
                # 增加计数器
                data_count += 1
                
                # 每两秒发送一次数据
                time.sleep(1)
                
        except socket.timeout:
            print("发送数据超时")
        except Exception as e:
            print(f"发送数据过程中出现错误: {e}")