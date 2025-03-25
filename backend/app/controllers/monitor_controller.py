from flask import jsonify, Response, stream_with_context
from backend.app.controllers.boundaries.communication.connection_manager import ConnectionManager

class MonitorController:
    def start_session(self, ip: str, port: int):
        connection_manager = ConnectionManager()
        sock = connection_manager.connect_to_device(ip, port)
        if not sock:
            return jsonify({'success': False, 'message': f'连接设备 {ip}:{port} 失败'}), 500

        # 设置超时时间（单位：秒）
        sock.settimeout(60)

        def generate():
            try:
                while True:
                    data = sock.recv(1024)
                    if not data:
                        break
                    message = data.decode('utf-8')
                    #在这里加入其他数据处理逻辑。
                    yield f"data: {message}\n\n"
            except sock.timeout:
                # 当发生超时时，yield一条提示信息后结束会话
                yield "data: 等待读取数据超时，结束会话\n\n"
            except Exception as e:
                yield f"data: 处理socket数据出错: {e}\n\n"
            finally:
                sock.close()
        return Response(stream_with_context(generate()), mimetype="text/event-stream")