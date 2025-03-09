# routes.py

from flask import Blueprint, send_from_directory, request, jsonify
from backend.app.controllers.device_controller import DeviceController
from backend.app.controllers.auth_controller import AuthController
from backend.app.boundaries.communication.connection_manager import ConnectionManager
import json

main = Blueprint('main', __name__)
auth_controller = AuthController()

@main.route('/')
def index():
    return send_from_directory('frontend', 'index.html')

@main.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('frontend/static', path)

@main.route('/login.html')
def login_page():
    return send_from_directory('frontend', 'login.html')

@main.route('/register.html')
def register_page():
    return send_from_directory('frontend', 'register.html')

@main.route('/dashboard.html')
def dashboard_page():
    return send_from_directory('frontend', 'dashboard.html')

# API 路由
@main.route('/api/login', methods=['POST'])
def login():
    return auth_controller.login()

@main.route('/api/register', methods=['POST'])
def register():
    return auth_controller.register()

@main.route('/api/devices/<user_id>', methods=['GET'])
def get_user_devices(user_id):
    device_controller = DeviceController()
    return device_controller.get_user_devices(user_id)

@main.route('/device/<device_id>')
def device_detail(device_id):
    return send_from_directory('frontend', 'monitor.html')

@main.route('/add-device')
def add_device_page():
    return send_from_directory('frontend', 'device_add.html')

@main.route('/api/devices/add', methods=['POST'])
def add_device():
    device_controller = DeviceController()
    data = request.get_json()
    return device_controller.add_device(data)

# 新增：连接设备的 API 路由
@main.route('/api/connect', methods=['GET'])
def connect_device_api():
    ip = request.args.get('ip')
    port = request.args.get('port')
    if not ip or not port:
        return jsonify({'success': False, 'message': '缺少ip或port参数'}), 400

    connection_manager = ConnectionManager()
    sock = connection_manager.connect_to_device(ip, int(port))
    if sock:
        return jsonify({'success': True, 'message': f'成功连接到设备：{ip}:{port}'})
    else:
        return jsonify({'success': False, 'message': f'连接设备 {ip}:{port} 失败'}), 500
    
@main.route('/api/devices/update', methods=['POST'])
def update_device():
    device_controller = DeviceController()
    data = request.get_json()
    return device_controller.update_device(data)

@main.route('/api/devices/detail', methods=['GET'])
def device_detail_api():
    device_id = request.args.get('deviceId')
    device_controller = DeviceController()
    return device_controller.get_device_detail(device_id)

@main.route('/api/devices/delete', methods=['POST'])
def delete_device():
    device_controller = DeviceController()
    data = request.get_json()
    return device_controller.delete_device(data)

@main.route('/device_edit/<device_id>')
def device_edit_page(device_id):
    # 可以通过模板渲染传递 device_id，如使用 render_template
    # 此处直接返回静态文件供前端通过 JS 根据 URL 获取 device_id 参数
    return send_from_directory('frontend', 'device_edit.html')