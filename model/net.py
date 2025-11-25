# model/net.py
import sys
import os

# -----------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../DotsAndBoxesAZ/model
root_dir = os.path.dirname(current_dir)                  # .../DotsAndBoxesAZ
sys.path.append(root_dir)
# -----------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config  # 现在这行代码可以正常工作了

class ResidualBlock(nn.Module):
    """
    残差块：保持输入输出尺寸不变，增强特征提取能力
    结构: Conv -> BN -> ReLU -> Conv -> BN -> Add -> ReLU
    """
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class DotsAndBoxesNet(nn.Module):
    def __init__(self):
        super(DotsAndBoxesNet, self).__init__()
        
        # 1. 基础参数
        self.board_rows = Config.BOARD_ROWS
        self.board_cols = Config.BOARD_COLS
        self.action_size = Config.ACTION_SIZE
        
        # 输入维度: (3, H, W)
        self.input_h = 2 * self.board_rows + 1
        self.input_w = 2 * self.board_cols + 1
        
        num_channels = 64
        
        # 2. 卷积骨干 (Backbone)
        self.conv_input = nn.Sequential(
            nn.Conv2d(3, num_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )
        
        # 残差塔
        self.res_tower = nn.Sequential(
            ResidualBlock(num_channels),
            ResidualBlock(num_channels),
            ResidualBlock(num_channels),
            ResidualBlock(num_channels)
        )
        
        # 3. 策略头 (Policy Head)
        self.policy_conv = nn.Sequential(
            nn.Conv2d(num_channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU()
        )
        self.policy_fc = nn.Linear(2 * self.input_h * self.input_w, self.action_size)
        
        # 4. 价值头 (Value Head)
        self.value_conv = nn.Sequential(
            nn.Conv2d(num_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        self.value_fc = nn.Sequential(
            nn.Linear(1 * self.input_h * self.input_w, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # 骨干
        x = self.conv_input(x)
        x = self.res_tower(x)
        
        # 策略
        pi = self.policy_conv(x)
        pi = pi.view(pi.size(0), -1)
        pi = self.policy_fc(pi)
        pi = F.log_softmax(pi, dim=1)
        
        # 价值
        v = self.value_conv(x)
        v = v.view(v.size(0), -1)
        v = self.value_fc(v)
        
        return pi, v

# -----------------------------------------------------
# 测试代码
# -----------------------------------------------------
if __name__ == "__main__":
    # 1. 实例化模型
    try:
        model = DotsAndBoxesNet()
        print("模型构建成功！")
        
        # 2. 创建模拟输入 (Batch=5)
        dummy_input = torch.randn(5, 3, 7, 7)
        
        # 3. 前向传播
        pi, v = model(dummy_input)
        
        # 4. 检查输出
        print("-" * 30)
        print(f"输入形状: {dummy_input.shape}")
        print(f"策略输出 (pi): {pi.shape}")
        print(f"价值输出 (v) : {v.shape}")
        print("-" * 30)
        print("测试通过！")
    except Exception as e:
        print(f"测试失败: {e}")