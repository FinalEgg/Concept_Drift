from flask import Flask
from flask_cors import CORS
import os

def create_app():
    app = Flask(__name__,
                template_folder='frontend',  # 设置模板文件夹路径
                static_folder='frontend/static')  # 设置静态文件夹路径
    CORS(app)

    from routes import main
    app.register_blueprint(main)

    return app