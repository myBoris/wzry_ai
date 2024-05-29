import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
class WzryNet(nn.Module):
    def __init__(self):
        super(WzryNet, self).__init__()
        # 增加卷积层的数量和通道数
        self.conv1 = nn.Conv2d(3, 64, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        # self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1)
        # self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1)
        # self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, stride=1)

        self.output_size = 747  # 输出维度

        # 计算卷积层输出的特征图尺寸
        conv_output_size = self._get_conv_output_size(640, 640)

        # 定义全连接层
        self.fc1 = nn.Linear(conv_output_size, 256)  # 增加全连接层的神经元数量
        self.fc_output = nn.Linear(256, self.output_size)  # 初始化输出层

        # 初始化卷积层的权重
        self._initialize_weights()

        # 移动所有层到与模型相同的设备
        self.device = next(self.parameters()).device

    def _get_conv_output_size(self, height, width):
        # 定义一个虚拟输入，计算经过所有卷积层后的输出尺寸
        dummy_input = torch.zeros(1, 3, height, width)
        with torch.no_grad():
            x = F.relu(self.conv1(dummy_input))
            x = F.relu(self.conv2(x))
            # 根据需要添加更多卷积层
            # x = F.relu(self.conv3(x))
            # x = F.relu(self.conv4(x))
            # x = F.relu(self.conv5(x))
        return x.view(x.size(0), -1).size(1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 调整维度顺序以匹配期望的输入形状 [batch_size, channels, height, width]
        x = x.to(next(self.parameters()).device)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        # x = F.relu(self.conv5(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc_output(x)

        return x