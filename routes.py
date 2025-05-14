# routes.py
from flask import Blueprint, send_from_directory, request, jsonify
from backend.app.controllers.device_controller import DeviceController
from backend.app.controllers.auth_controller import AuthController
from backend.app.controllers.monitor_controller import MonitorController
from flask import send_from_directory, make_response
import os
import json

main = Blueprint('main', __name__)
auth_controller = AuthController()

@main.route('/')
def index():
    return send_from_directory('frontend', 'index.html')


# 将 @app.route 改为 @main.route，并删除或替换先前的 serve_static 函数
@main.route('/scripts/<path:filename>')
def serve_script(filename):
    response = make_response(send_from_directory(os.path.join('frontend', 'scripts'), filename))
    # 脚本文件缓存1天
    response.headers['Cache-Control'] = 'public, max-age=86400'
    return response

# 修改静态文件处理函数添加缓存控制，替换现有的 serve_static 函数
@main.route('/static/<path:path>')
def serve_static(path):
    response = make_response(send_from_directory('frontend/static', path))
    
    # 根据文件类型设置不同的缓存策略
    if path.endswith('.css') or path.endswith('.js'):
        # CSS和JS文件缓存1天
        response.headers['Cache-Control'] = 'public, max-age=86400'
    elif path.endswith(('.png', '.jpg', '.jpeg', '.gif', '.ico')):
        # 图片缓存7天
        response.headers['Cache-Control'] = 'public, max-age=604800'
    else:
        # 其他文件缓存4小时
        response.headers['Cache-Control'] = 'public, max-age=14400'
    
    return response

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

@main.route('/api/monitor/connect', methods=['GET'])
def monitor_connect():
    ip = request.args.get('ip')
    port = request.args.get('port')
    device_id = request.args.get('deviceId')
    
    if not ip or not port:
        return jsonify({'success': False, 'message': '缺少ip或port参数'}), 400
    
    monitor_controller = MonitorController()
    print(f"开始连接设备 {ip}:{port}，设备ID: {device_id}")
    return monitor_controller.start_session(ip, int(port), device_id)
    
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
