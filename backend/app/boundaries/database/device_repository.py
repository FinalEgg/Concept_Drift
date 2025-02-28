from typing import List
import mysql.connector
from backend.app.entities.edgeDevice import EdgeDevice
from .db_config import DB_CONFIG

class DeviceRepository:
    def __init__(self):
        self.config = DB_CONFIG

    def get_connection(self):
        return mysql.connector.connect(**self.config)

    def find_devices_by_user_id(self, user_id: str) -> List[EdgeDevice]:
        connection = None
        cursor = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            query = """
                SELECT device_id, device_name, ip_address, port, model_id
                FROM Device 
                WHERE user_id = %s
            """
            cursor.execute(query, (user_id,))
            devices = []
            
            for device_data in cursor.fetchall():
                device = EdgeDevice(
                    device_name=device_data['device_name'],
                    ip_address=device_data['ip_address'],
                    port=device_data['port']
                )
                device._device_id = device_data['device_id']
                devices.append(device)
                
            return devices
            
        except mysql.connector.Error as err:
            print(f"数据库错误: {err}")
            return []
        finally:
            if cursor:
                cursor.close()
            if connection and connection.is_connected():
                connection.close()

    def add_device(self, device: EdgeDevice, user_id: str, model_id: str) -> bool:
        connection = None
        cursor = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            query = """
                INSERT INTO Device (device_id, user_id, model_id, device_name, ip_address, port)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(query, (
                device.device_id,
                user_id,
                model_id,
                device.device_name,
                device.ip_address,
                device.port
            ))
            
            connection.commit()
            return True
            
        except mysql.connector.Error as err:
            print(f"数据库错误: {err}")
            if connection:
                connection.rollback()
            return False
        finally:
            if cursor:
                cursor.close()
            if connection and connection.is_connected():
                connection.close()