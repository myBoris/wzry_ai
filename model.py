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

    def forward(self, x):
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
