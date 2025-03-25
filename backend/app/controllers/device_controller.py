# Python
from flask import jsonify
from backend.app.controllers.boundaries.database.device_repository import DeviceRepository
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
                    'dim': getattr(device, '_dim', None)
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

            # 将 dimensions 转换为整数
            device_dim = int(device_data['dimensions'])
            success = self.device_repository.add_device(
                device, 
                device_data['userId'],
                device_dim
            )
            
            if success:
                return jsonify({
                    'success': True,
                    'message': '设备添加成功',
                    'device': {
                        'id': device.device_id,
                        'name': device.device_name,
                        'ip': device.ip_address,
                        'port': device.port,
                        'dim': device_dim
                    }
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
    
    def update_device(self, device_data: dict):
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
            # 设置设备ID
            device._device_id = device_data['deviceId']
            device_dim = int(device_data['dimensions'])
            success = self.device_repository.update_device(device, device_dim)
            if success:
                return jsonify({
                    'success': True,
                    'message': '设备更新成功',
                    'device': {
                        'id': device.device_id,
                        'name': device.device_name,
                        'ip': device.ip_address,
                        'port': device.port,
                        'dim': device_dim
                    }
                })
            else:
                return jsonify({
                    'success': False,
                    'message': '设备更新失败'
                }), 500
        except Exception as e:
            print(f"更新设备失败: {str(e)}")
            return jsonify({
                'success': False,
                'message': '服务器错误'
            }), 500

    def get_device_detail(self, device_id):
        try:
            device = self.device_repository.find_device_by_id(device_id)
            if device:
                return jsonify({
                    'success': True,
                    'device': {
                        'id': device.device_id,
                        'name': device.device_name,
                        'ip': device.ip_address,
                        'port': device.port,
                        'dim': device._dim
                    }
                })
            else:
                return jsonify({
                    'success': False,
                    'message': '设备未找到'
                }), 404
        except Exception as e:
            print(f"获取设备详情失败: {e}")
            return jsonify({
                'success': False,
                'message': '服务器错误'
            }), 500
    
    def delete_device(self, device_data: dict):
        try:
            device_id = device_data.get('deviceId')
            if not device_id:
                return jsonify({
                    'success': False,
                    'message': '缺少设备ID'
                }), 400
            success = self.device_repository.delete_device(device_id)
            if success:
                return jsonify({
                    'success': True,
                    'message': '设备已删除'
                })
            else:
                return jsonify({
                    'success': False,
                    'message': '设备删除失败'
                }), 500
        except Exception as e:
            print(f"删除设备失败: {e}")
            return jsonify({
                'success': False,
                'message': '服务器错误'
            }), 500