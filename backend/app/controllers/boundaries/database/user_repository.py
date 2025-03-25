from typing import Optional
import mysql.connector
from backend.app.entities.user import User
from .db_config import DB_CONFIG

class UserRepository:
    def __init__(self):
        self.config = DB_CONFIG
    
    def get_connection(self):
        return mysql.connector.connect(**self.config)

    def find_user_by_credentials(self, username: str, password: str) -> Optional[User]:
        connection = None
        cursor = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            query = """
                SELECT user_id, user_name, user_password, permission 
                FROM User 
                WHERE user_name = %s AND user_password = %s
            """
            cursor.execute(query, (username, password))
            user_data = cursor.fetchone()
            
            # 确保消费完所有结果
            cursor.fetchall()
            
            if user_data:
                devices_query = """
                    SELECT device_id 
                    FROM Device 
                    WHERE user_id = %s
                """
                cursor.execute(devices_query, (user_data['user_id'],))
                devices = [row['device_id'] for row in cursor.fetchall()]
                
                return User(
                    user_data['user_name'],
                    user_data['user_password'],
                    user_data['permission'],
                    devices,
                    user_data['user_id']   # 使用数据库中的user_id
                )
            
            return None
            
        except mysql.connector.Error as err:
            print(f"数据库错误: {err}")
            return None
        finally:
            if cursor:
                cursor.close()
            if connection and connection.is_connected():
                connection.close()

    def create_user(self, user: User) -> bool:
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            # 插入用户数据
            insert_query = """
                INSERT INTO User (user_id, user_name, user_password, permission)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(insert_query, (
                user.user_id,
                user.username,
                user.password,
                user.permission_level
            ))
            
            # 如果有设备关联，在Device表中创建关联
            if user.devices:
                for device_id in user.devices:
                    device_query = """
                        UPDATE Device 
                        SET user_id = %s 
                        WHERE device_id = %s
                    """
                    cursor.execute(device_query, (user.user_id, device_id))
            
            connection.commit()
            return True
            
        except mysql.connector.Error as err:
            print(f"数据库错误: {err}")
            connection.rollback()
            return False
        finally:
            if 'connection' in locals() and connection.is_connected():
                cursor.close()
                connection.close()

    def update_user(self, user: User) -> bool:
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            # 更新用户信息
            update_query = """
                UPDATE User 
                SET user_name = %s, user_password = %s, permission = %s
                WHERE user_id = %s
            """
            cursor.execute(update_query, (
                user.username,
                user.password,
                user.permission_level,
                user.user_id
            ))
            
            # 更新设备关联
            # 首先移除所有现有关联
            cursor.execute("""
                UPDATE Device 
                SET user_id = NULL 
                WHERE user_id = %s
            """, (user.user_id,))
            
            # 添加新的关联
            if user.devices:
                for device_id in user.devices:
                    cursor.execute("""
                        UPDATE Device 
                        SET user_id = %s 
                        WHERE device_id = %s
                    """, (user.user_id, device_id))
            
            connection.commit()
            return True
            
        except mysql.connector.Error as err:
            print(f"数据库错误: {err}")
            connection.rollback()
            return False
        finally:
            if 'connection' in locals() and connection.is_connected():
                cursor.close()
                connection.close()