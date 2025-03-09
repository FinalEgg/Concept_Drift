# backend/app/boundaries/communication/connection_manager.py

import socket

class ConnectionManager:
    def connect_to_device(self, ip: str, port: int):
        """
        向指定设备的ip地址及端口建立socket连接请求
        """
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((ip, port))
            print(f"成功连接到设备：{ip}:{port}")
            return s
        except Exception as e:
            print(f"连接到设备 {ip}:{port} 失败: {e}")
            return None