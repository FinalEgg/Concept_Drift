from typing import List
import mysql.connector
from .db_config import DB_CONFIG
from backend.app.entities.dataModel import DataModel

class ModelRepository:
    def __init__(self):
        self.config = DB_CONFIG

    def get_connection(self):
        return mysql.connector.connect(**self.config)

    def get_all_models(self) -> List[dict]:
        connection = None
        cursor = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            query = "SELECT model_id, model_name, data_dim, dataType FROM Model"
            cursor.execute(query)
            return cursor.fetchall()
        except mysql.connector.Error as err:
            print(f"数据库错误: {err}")
            return []
        finally:
            if cursor:
                cursor.close()
            if connection and connection.is_connected():
                connection.close()

    def create_model(self, model_data: dict, user_id: str) -> str:
        connection = None
        cursor = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            # 创建一个新的DataModel实例
            model = DataModel(
                name=model_data['modelName'],
                description=model_data.get('description', ''),
                dimension=model_data['dimension']
            )
            
            # 将数据类型列表转换为字符串
            data_types = ','.join(model_data['dataTypes'])

            # 插入模型数据
            query = """
                INSERT INTO Model (model_id, user_id, model_name, data_dim, dataType)
                VALUES (%s, %s, %s, %s, %s)
            """
            cursor.execute(query, (
                model.model_id,
                user_id,
                model.name,
                model.dimension,
                data_types
            ))
            
            connection.commit()
            return model.model_id
            
        except mysql.connector.Error as err:
            print(f"数据库错误: {err}")
            if connection:
                connection.rollback()
            raise Exception('创建模型失败')
        finally:
            if cursor:
                cursor.close()
            if connection and connection.is_connected():
                connection.close()