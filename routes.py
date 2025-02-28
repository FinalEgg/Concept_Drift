from flask import Blueprint, send_from_directory, request
from backend.app.controllers.device_controller import DeviceController
from backend.app.controllers.auth_controller import AuthController
from backend.app.boundaries.database.model_repository import ModelRepository
from flask import jsonify
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
    return send_from_directory('frontend', 'device_detail.html')

@main.route('/add-device')
def add_device_page():
    return send_from_directory('frontend', 'device_add.html')

@main.route('/api/devices/add', methods=['POST'])
def add_device():
    device_controller = DeviceController()
    data = request.get_json()
    return device_controller.add_device(data)

@main.route('/api/models', methods=['GET'])
def get_models():
    model_repository = ModelRepository()
    models = model_repository.get_all_models()
    return jsonify({
        'success': True,
        'models': models
    })

@main.route('/create-model.html')
def create_model_page():
    return send_from_directory('frontend', 'create_model.html')

@main.route('/api/models/add', methods=['POST'])
def add_model():
    try:
        data = request.get_json()
        user_cookie = request.cookies.get('user')
        if not user_cookie:
            return jsonify({
                'success': False,
                'message': '用户未登录'
            }), 401

        user_data = json.loads(user_cookie)
        # 在服务器控制台输出当前用户的ID
        print(f"当前用户ID: {user_data['user_id']}")

        model_repository = ModelRepository()
        model_id = model_repository.create_model(data, user_data['user_id'])
        
        return jsonify({
            'success': True,
            'message': '模型创建成功',
            'modelId': model_id
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500
    
