from uuid import uuid4

class User:
    def __init__(self, username: str, password: str, permission_level: int, devices: list, user_id: str = None):
        self._user_id = user_id if user_id is not None else str(uuid4())
        self._username = username
        self._password = password
        self._permission_level = permission_level
        self._devices = devices

    @property
    def user_id(self):
        return self._user_id

    @property
    def username(self):
        return self._username

    @username.setter
    def username(self, value: str):
        self._username = value

    @property
    def password(self):
        return self._password

    @password.setter
    def password(self, value: str):
        self._password = value

    @property
    def permission_level(self):
        return self._permission_level

    @permission_level.setter
    def permission_level(self, value: int):
        self._permission_level = value

    @property
    def devices(self):
        return self._devices

    @devices.setter
    def devices(self, value: list):
        self._devices = value

    def add_device(self, device: str):
        self._devices.append(device)

    def remove_device(self, device: str):
        if device in self._devices:
            self._devices.remove(device)