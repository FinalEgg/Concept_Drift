# run.py
from __init__ import create_app
from simulation_device.sim_device import SimDevice
import threading
import mysql.connector
import os

def check_db_connection():
    from backend.app.controllers.boundaries.database.db_config import DB_CONFIG
    try:
        # 尝试连接到数据库
        conn = mysql.connector.connect(**DB_CONFIG)
        conn.close()
        print("数据库连接正常！")
        return True
    except mysql.connector.Error as err:
        print(f"数据库连接错误: {err}")
        return False

def init_database():
    try:
        # 连接到MySQL而不指定数据库
        conn = mysql.connector.connect(
            host="localhost",
            port=3306,
            user="root",
            password="12345"
        )
        
        cursor = conn.cursor()
        
        # 创建数据库
        cursor.execute("CREATE DATABASE IF NOT EXISTS studd")
        cursor.execute("USE studd")
        
        # 读取并执行初始化SQL脚本
        script_path = "backend/app/controllers/boundaries/database/init_db.sql"
        with open(script_path, 'r') as file:
            # 按语句拆分SQL脚本
            sql_commands = file.read().split(';')
            for command in sql_commands:
                if command.strip():
                    cursor.execute(command)
        
        conn.commit()
        cursor.close()
        conn.close()
        print("数据库初始化完成！")
        return True
    except Exception as e:
        print(f"数据库初始化失败: {e}")
        return False

def start_sim_device():
    sim_device = SimDevice(host='127.0.0.1', port=9000)
    sim_device.start()

app = create_app()

if __name__ == '__main__':
    # 检查数据库连接
    if not check_db_connection():
        print("尝试初始化数据库...")
        if not init_database():
            print("无法初始化数据库，请确保MySQL服务已启动并具有正确的访问权限")
            exit(1)
    
    # 启动模拟设备
    sim_thread = threading.Thread(target=start_sim_device, daemon=True)
    sim_thread.start()
    
    # 启动Flask应用
    app.run(debug=True)