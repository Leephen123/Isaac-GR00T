import struct

import struct

class KeyMap:
    A = 0      # 按钮A
    B = 1      # 按钮B
    X = 2      # 按钮X
    Y = 3      # 按钮Y
    U = 4      # 上键 (Up)
    D = 5      # 下键 (Down)
    L = 6      # 左键 (Left)
    R = 7      # 右键 (Right)
    LB = 8     # 左上肩键
    RB = 9     # 右上肩键
    LT = 10    # 左下肩键 (Trigger)
    RT = 11    # 右下肩键 (Trigger)
    BACK = 12  # 选择键
    START = 13 # 开始键
    L3 = 14    # 左摇杆下压
    R3 = 15    # 右摇杆下压

class RemoteController:
    def __init__(self):
        # 摇杆数据 [-1.0, 1.0]
        self.lx = 0.0  # 左摇杆X
        self.ly = 0.0  # 左摇杆Y
        self.rx = 0.0  # 右摇杆X
        self.ry = 0.0  # 右摇杆Y
        
        # 按钮状态
        self.button = [0] * 16
        
        # 原始数据包
        self.raw_data = bytearray(40)

    def set(self, data):
        """更新手柄状态 (40字节数据包)"""
        if len(data) != 40:
            raise ValueError("需要40字节的输入数据")
            
        # 解析按钮状态
        buttons = struct.unpack("<H", data[2:4])[0]  # 小端序
        
        # 更新按钮状态
        for i in range(16):
            self.button[i] = (buttons >> i) & 0x1
            
        # 摇杆数据
        self.lx = struct.unpack("<f", data[4:8])[0]   # 左摇杆X
        self.ly = struct.unpack("<f", data[8:12])[0]  # 左摇杆Y
        self.rx = struct.unpack("<f", data[12:16])[0]  # 右摇杆X
        self.ry = struct.unpack("<f", data[16:20])[0]  # 右摇杆Y


