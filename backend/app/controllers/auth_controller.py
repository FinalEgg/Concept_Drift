from flask import jsonify, request, make_response
import json
from backend.app.boundaries.database.user_repository import UserRepository
from backend.app.entities.user import User

class AuthController:
    def __init__(self):
        self.user_repository = UserRepository()
    
    def login(self):
        try:
            data = request.get_json()
            if not data:
                return jsonify({
                    'success': False,
                    'message': '无效的请求数据'
                }), 400

            username = data.get('username')
            password = data.get('password')

            if not username or not password:
                return jsonify({
                    'success': False,
                    'message': '用户名和密码不能为空'
                }), 400

            # 验证用户凭据
            user = self.user_repository.find_user_by_credentials(username, password)
            
            if user:
                user_data = {
                    'user_id': user.user_id,
                    'username': user.username,
                    'permission': user.permission_level
                }
                response = make_response(jsonify({
                    'success': True,
                    'message': '登录成功',
                    'data': user_data
                }))
                # 设置HTTP-only cookie，确保后续接口能正确获取用户信息
                response.set_cookie('user', json.dumps(user_data), httponly=True)
                return response
            else:
                return jsonify({
                    'success': False,
                    'message': '用户名或密码错误'
                }), 401

        except Exception as e:
            print(f"登录出错: {str(e)}")
            return jsonify({
                'success': False,
                'message': '服务器错误，请稍后重试'
            }), 500

    def register(self):
        try:
            data = request.get_json()
            if not data:
                return jsonify({
                    'success': False,
                    'message': '无效的请求数据'
                }), 400

            username = data.get('username')
            password = data.get('password')

            if not username or not password:
                return jsonify({
                    'success': False,
                    'message': '用户名和密码不能为空'
                }), 400
            
            # 创建新用户对象(默认权限为1,设备列表为空)
            new_user = User(username, password, 1, [])
            
            # 尝试创建用户
            if self.user_repository.create_user(new_user):
                user_data = {
                    'user_id': new_user.user_id,
                    'username': new_user.username,
                    'permission': new_user.permission_level
                }
                response = make_response(jsonify({
                    'success': True,
                    'message': '注册成功',
                    'data': user_data
                }))
                # 注册成功也设置cookie，以便后续直接使用
                response.set_cookie('user', json.dumps(user_data), httponly=True)
                return response
            else:
                return jsonify({
                    'success': False,
                    'message': '注册失败，用户名可能已存在'
                }), 400

        except Exception as e:
            print(f"注册出错: {str(e)}")
            return jsonify({
                'success': False,
                'message': '服务器错误，请稍后重试'
            }), 500