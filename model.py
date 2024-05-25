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

        # 全连接层将在稍后初始化
        self.fc1 = None
        self.fc_output = None
        self.output_size = 747  # 输出维度

        # 初始化卷积层的权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        # 确保输入形状正确，如果是单张图片（3D 张量），则增加一个批次维度
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        # 调整维度顺序以匹配期望的输入形状 [batch_size, channels, height, width]
        x = x.permute(0, 3, 1, 2)

        # 移动输入到与模型相同的设备
        device = next(self.parameters()).device
        x = x.to(device)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        # x = F.relu(self.conv5(x))

        # 动态计算展平后的特征大小
        n_size = x.view(x.size(0), -1).size(1)
        if self.fc1 is None:
            self.fc1 = nn.Linear(n_size, 256).to(x.device)  # 增加全连接层的神经元数量
            self.fc_output = nn.Linear(256, self.output_size).to(x.device)  # 初始化输出层

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc_output(x)

        return x
