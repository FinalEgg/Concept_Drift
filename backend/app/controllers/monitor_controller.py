from flask import jsonify, Response, stream_with_context
import pickle
import struct
import torch
import numpy as np
import json
from backend.app.controllers.boundaries.communication.connection_manager import ConnectionManager
from backend.app.controllers.boundaries.database.device_repository import DeviceRepository
from backend.app.controllers.services.model_service import ModelService
import torch.nn as nn

class MonitorController:
    def __init__(self):
        self.model_service = ModelService()
        self.device_repository = DeviceRepository()
        # 初始化滑动窗口参数
        self.window_size = 50  # 滑动窗口大小
        self.threshold = 0.8   # 一致率阈值，低于此值视为概念漂移
    
    def start_session(self, ip: str, port: int, device_id: str = None):
        """
        启动与设备的监控会话
        
        参数:
        ip: 设备IP地址
        port: 设备端口
        device_id: 设备ID，用于加载和训练模型
        """
        # 创建连接管理器并连接到设备
        connection_manager = ConnectionManager()
        sock = connection_manager.connect_to_device(ip, port)
        if not sock:
            return jsonify({'success': False, 'message': f'连接设备 {ip}:{port} 失败'}), 500

        # 保存模型和一致性检测状态
        teacher_model = None
        consistency_buffer = []

        # 如果有设备ID，尝试加载模型并发送参数
        if device_id:
            try:
                # 加载教师模型
                teacher_model = self.model_service.load_model(device_id, model_type='teacher')
                
                if teacher_model is None:
                    print(f"未找到设备 {device_id} 的教师模型，将开始训练")
                    # 如果没有找到现有的教师模型，则训练一个新的
                    teacher_model = self.model_service.train_teacher_model(
                        device_id=device_id,
                        epochs=100,
                        sample_size=1000
                    )
                
                # 训练学生模型并获取参数
                student_params = self.model_service.train_student_model(
                    teacher_model=teacher_model,
                    device_id=device_id,
                    epochs=100,
                    sample_size=1000
                )
                
                # 序列化模型参数
                serialized_params = pickle.dumps(student_params)
                
                # 发送数据长度，使用4字节表示
                sock.sendall(struct.pack('!I', len(serialized_params)))
                
                # 分块发送大数据
                chunk_size = 4096
                for i in range(0, len(serialized_params), chunk_size):
                    chunk = serialized_params[i:i+chunk_size]
                    sock.sendall(chunk)
                
                print(f"学生模型参数已发送到设备 {ip}:{port}")
                
            except Exception as e:
                print(f"处理模型时出错: {e}")
                # 继续连接，即使模型处理失败
        else:
            print(f"未提供设备ID，无法加载或训练模型")

        # 设置超时时间（单位：秒）
        sock.settimeout(60)

        def generate():
            # 用于追踪一致性的缓冲区
            nonlocal consistency_buffer
            device_tensor = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            try:
                
                # 然后持续接收数据
                while True:
                    data = sock.recv(1024)
                    if not data:
                        break
                    
                    # 解析接收到的消息（格式：漂移指数,特征值1,特征值2,...,学生模型预测）
                    message = data.decode('utf-8').strip()
                    parts = message.split(',')
                    
                    # 只有当有老师模型并且收到的数据格式正确时才进行比对
                    if teacher_model is not None and len(parts) >= 2:
                        try:
                            # 获取原始漂移指数和学生模型预测结果
                            drift_index = float(parts[0])
                            
                            # 解析特征数据和学生预测结果
                            features = np.array([float(x) for x in parts[1:-1]], dtype=np.float32)
                            student_prediction = int(float(parts[-1]))
                            
                            # 使用老师模型预测 
                            with torch.no_grad():
                                feature_tensor = torch.FloatTensor(features).to(device_tensor)
                                
                                # 确保特征维度正确
                                if feature_tensor.dim() == 1:
                                    feature_tensor = feature_tensor.unsqueeze(0)
                                        
                                teacher_output = teacher_model(feature_tensor)
                                teacher_prediction = 1 if teacher_output.item() > 0.5 else 0
                            
                            # 判断两个模型预测是否一致
                            match = 1 if teacher_prediction == student_prediction else 0
                            consistency_buffer.append(match)
                            
                            # 保持窗口大小
                            if len(consistency_buffer) > self.window_size:
                                consistency_buffer.pop(0)
                            
                            # 计算当前一致率
                            current_consistency = sum(consistency_buffer) / len(consistency_buffer)
                            
                            # 实际漂移检测逻辑：一致率低于阈值表示可能发生了概念漂移
                            drift_detected = current_consistency < self.threshold and len(consistency_buffer) == self.window_size
                            
                            # 构造新的消息，添加老师预测结果和一致率信息
                            enhanced_message = (
                                f"{current_consistency:.4f}," + 
                                message + 
                                f",{teacher_prediction},{1 if drift_detected else 0}"
                            )
                            
                            # 如果检测到漂移，打印警告
                            if drift_detected:
                                print(f"\n[!] 概念漂移检测！当前一致率: {current_consistency:.2f}")
                                print(f"学生预测: {student_prediction}, 老师预测: {teacher_prediction}")
                                
                            # 发送增强后的消息到前端
                            yield f"data: {enhanced_message}\n\n"
                            
                        except Exception as process_error:
                            print(f"处理数据时出错: {process_error}")
                            # 仍然发送原始消息
                            yield f"data: {message}\n\n"
                    else:
                        # 如果没有老师模型或数据格式不正确，直接发送原始消息
                        yield f"data: {message}\n\n"
                
            except sock.timeout:
                # 当发生超时时，yield一条提示信息后结束会话
                yield "data: 等待读取数据超时，结束会话\n\n"
            except Exception as e:
                yield f"data: 处理socket数据出错: {e}\n\n"
            finally:
                sock.close()
                
        return Response(stream_with_context(generate()), mimetype="text/event-stream")