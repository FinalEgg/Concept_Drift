from uuid import uuid4
import numpy as np
from typing import List, Literal

DataType = Literal['int', 'float', 'bool']

class DataModel:
    def __init__(self, name: str, description: str, dimension: int):
        self._model_id = str(uuid4())
        self._name = name
        self._dimension = dimension
        self._data = []     # 存储数据，每行是一个包含 dimension 个元素的列表
        self._labels = []   # 存储标签，每行是一个 float 数值
        self._column_types = ['float'] * dimension  # 默认所有列类型均为 float

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def data(self) -> List[List[float]]:
        return self._data

    @property
    def labels(self) -> List[float]:
        return self._labels

    def remove_data_point(self, index: int) -> bool:
        if 0 <= index < len(self._data):
            self._data.pop(index)
            self._labels.pop(index)
            return True
        return False

    def clear_data(self):
        self._data = []
        self._labels = []

    def get_data_size(self) -> int:
        return len(self._data)

    @property
    def column_types(self) -> List[str]:
        return self._column_types

    def set_column_type(self, column_index: int, data_type: DataType) -> bool:
        if 0 <= column_index < self._dimension and data_type in ['int', 'float', 'bool']:
            self._column_types[column_index] = data_type
            return True
        return False

    def add_data_point(self, data_point: List[float], label: float) -> bool:
        if len(data_point) != self._dimension:
            return False
        try:
            converted_point = []
            for i, value in enumerate(data_point):
                if self._column_types[i] == 'int':
                    converted_point.append(int(value))
                elif self._column_types[i] == 'bool':
                    converted_point.append(bool(value))
                else:  # float
                    converted_point.append(float(value))
            self._data.append(converted_point)
            self._labels.append(float(label))
            return True
        except (ValueError, TypeError):
            return False

    def get_column_data(self, column_index: int) -> List:
        if 0 <= column_index < self._dimension:
            return [row[column_index] for row in self._data]
        return []

    def to_numpy(self) -> tuple:
        converted_data = []
        for row in self._data:
            converted_row = []
            for i, value in enumerate(row):
                if self._column_types[i] == 'bool':
                    converted_row.append(float(value))  # 将 bool 转换为 float
                else:
                    converted_row.append(value)
            converted_data.append(converted_row)
        return np.array(converted_data), np.array(self._labels)