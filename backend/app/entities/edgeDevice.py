from uuid import uuid4
from enum import Enum
from typing import Dict, Any, Optional
from datetime import datetime

class DeviceStatus(Enum):
    OFFLINE = "offline"
    ONLINE = "online"
    ERROR = "error"
    BUSY = "busy"

class EdgeDevice:
    def __init__(self, device_name: str, ip_address: str, port: int, model_id: str = None):
        self._device_id = str(uuid4())
        self._device_name = device_name
        self._ip_address = ip_address
        self._port = port
        self._model_id = model_id
        self._status = DeviceStatus.OFFLINE
        self._dim = None       # 新增：初始化 _dim 属性

    @property
    def device_id(self) -> str:
        return self._device_id

    @property
    def device_name(self) -> str:
        return self._device_name

    @property
    def ip_address(self) -> str:
        return self._ip_address

    @property
    def port(self) -> int:
        return self._port

    @property
    def model_id(self) -> str:
        return self._model_id

    @model_id.setter
    def model_id(self, value: str):
        self._model_id = value

    @property
    def dim(self) -> int:
        """公开设备数据维度属性"""
        return self._dim

    @dim.setter
    def dim(self, value: int):
        self._dim = value

    def validate_data(self, data: Dict[str, Any]) -> bool:
        """由于已删除数据模型，此处默认返回True"""
        return True