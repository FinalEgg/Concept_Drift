from uuid import uuid4
from enum import Enum
from typing import Dict, Any, Optional
from datetime import datetime
from .dataModel import DataModel

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
        self._data_model = None  # 初始化 data_model 为 None

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
    def data_model(self) -> Optional[DataModel]:
        return self._data_model

    @data_model.setter
    def data_model(self, model: DataModel):
        self._data_model = model

    def validate_data(self, data: Dict[str, Any]) -> bool:
        """验证数据是否符合数据模型的格式要求"""
        if self._data_model is None:
            return False
        return self._data_model.validate(data)

    @property
    def model_id(self) -> str:
        return self._model_id

    @model_id.setter
    def model_id(self, value: str):
        self._model_id = value