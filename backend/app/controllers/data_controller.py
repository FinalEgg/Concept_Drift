# backend/app/controllers/device_controller.py
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