# simulation_device/sim_device.py

import socket
import time
import random

class SimDevice:
    def __init__(self, host: str = '127.0.0.1', port: int = 9000):
        """
        初始化模拟设备，可修改 host 和 port 实现不同的监听地址与端口
        """
        self.host = host
        self.port = port

    def start(self):
        """
        启动模拟设备，动态选择空闲端口等待云服务器连接，连接成功后每秒发送随机数据
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
        try:
            print(f"SimDevice 正在 {self.host}:{bind_port} 监听连接...")
            conn, addr = server_sock.accept()
            print(f"接收到来自 {addr} 的连接")
            self.send_data(conn)
        except Exception as e:
            print(f"SimDevice 运行错误: {e}")
        finally:
            server_sock.close()

    def send_data(self, conn: socket.socket):
        """
        连接成功后，每秒随机产生20个维度+1个标签数据（共21个float），并发送给云服务器
        """
        try:
            while True:
                # 生成共21个随机浮点数，前20个表示特征，最后一个作为标签
                data = [round(random.uniform(0, 1), 4) for _ in range(21)]
                data_str = ','.join(map(str, data)) + "\n"
                conn.sendall(data_str.encode('utf-8'))
                print(f"发送数据: {data_str.strip()}")
                time.sleep(1)
        except Exception as e:
            print(f"发送数据过程中出现错误: {e}")
        finally:
            conn.close()
    
    