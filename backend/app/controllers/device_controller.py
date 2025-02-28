from flask import jsonify
from backend.app.boundaries.database.device_repository import DeviceRepository
from backend.app.entities.edgeDevice import EdgeDevice

class DeviceController:
    def __init__(self):
        self.device_repository = DeviceRepository()
    
    def get_user_devices(self, user_id):
        try:
            devices = self.device_repository.find_devices_by_user_id(user_id)
            return jsonify({
                'success': True,
                'devices': [{
                    'id': device.device_id,
                    'name': device.device_name,
                    'ip': device.ip_address,
                    'port': device.port,
                    'model_id': device.model_id
                } for device in devices]
            })
        except Exception as e:
            print(f"获取设备列表失败: {str(e)}")
            return jsonify({
                'success': False,
                'message': '获取设备列表失败'
            }), 500
        
    def add_device(self, device_data: dict):
        try:
            if not device_data:
                return jsonify({
                    'success': False,
                    'message': '无效的请求数据'
                }), 400

            device = EdgeDevice(
                device_name=device_data['deviceName'],
                ip_address=device_data['ipAddress'],
                port=device_data['port']
            )

            success = self.device_repository.add_device(
                device, 
                device_data['userId'],
                device_data['modelId']  # 添加模型ID
            )
            
            if success:
                return jsonify({
                    'success': True,
                    'message': '设备添加成功'
                })
            else:
                return jsonify({
                    'success': False,
                    'message': '设备添加失败'
                }), 500
        except Exception as e:
            print(f"添加设备失败: {str(e)}")
            return jsonify({
                'success': False,
                'message': '服务器错误'
            }), 500
    