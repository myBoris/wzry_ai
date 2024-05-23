import torch.nn as nn
import torch


class BasicBlock(nn.Module):  # 针对18层和34层的残差结构
    expansion = 1  # 对应着残差结构中主分支采用的卷积核个数有没有发生变化，18层和34层残差结构中，第1层和第2层卷积核个数相同，此处设置expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        # 在初始化函数中传入以下参数：输入特征矩阵深度、输出特征矩阵深度(即主分支上卷积核个数)，步距默认取1，下采样参数默认设置为None(其对应虚线残差结构)
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        # 第1层卷积层，实线结构步距为1，虚线结构步距为2，通过传入参数stride=stride控制
        self.bn1 = nn.BatchNorm2d(out_channel)
        # 使用BN时，卷积中的参数bias设置为False；且BN层放在conv层和relu层中间。BN层的输入为卷积层输出特征矩阵深度。
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        # 不管实线还是虚线残差结构，第2层卷积层的步距都为1，故传入参数stride=1
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample  # 定义下采样方法

    def forward(self, x):
        identity = x  # 将输入特征矩阵x赋给short cut分支上作为输出值(这是下采样函数等于None，即实线结构的情况)
        if self.downsample is not None:  # 如果下采样函数不等于None的话，即是虚线结构的情况
            identity = self.downsample(x)  # 将输入特征矩阵x赋给下采样函数，得到的结果作为short cut分支的结果

        # 主支线
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)  # 注意这里不经过relu函数，需要将这里的输出和short cut支线的输出相加再经过relu函数

        out += identity  # 主分支输出与short cut分支输出相加
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4  # 在50层、101层和152层的残差结构中，第1层和第2层卷积核个数相同，第3层的卷积核个数是第1层、第2层的4倍，这里设置expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        # 无论实线结构还是虚线结构，第1层卷积层都是kernel_size=1, stride=1
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        # 实线残差结构第2层3×3卷积stride=1，而虚线残差结构第2层3×3卷积stride=2,因此出入参数stride=stride
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        # 第3层卷积层步距都为1，但是第3层卷积核个数为第1层和第2层卷积核个数的4倍，则卷积核个数out_channels=out_channel*self.expansion
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)  # BN层输入卷积层深度等于卷积层3输出特征矩阵的深度
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x  # 将输入特征矩阵x赋给short cut分支上作为输出值(这是下采样函数等于None，即实线结构的情况)
        if self.downsample is not None:  # 如果下采样函数不等于None的话，即是虚线结构的情况
            identity = self.downsample(x)  # 将输入特征矩阵x赋给下采样函数，得到的结果作为short cut分支的结果

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)  # 注意这里同样不经过relu函数，需要将这里的输出和short cut支线的输出相加再经过relu函数

        out += identity
        out = self.relu(out)

        return out


# 定义ResNet整个网络的框架部分
class ResNet(nn.Module):

    def __init__(self, block, blocks_num, num_classes=1000, include_top=True, groups=1, width_per_group=64):
        # block对应的是残差结构，根据不同的层结构传入不同的block，如定义18或34层网络结构，这里的block即为BasicBlock，若50,101,152,则block为Bottleneck
        # blocks_num为所使用残差结构的数目，这是一个列表参数，如对应34层而言，blocks_num即为[3,4,6,3]
        # num_classes指训练集的分类个数，include_top是为了方便在ResNet基础上搭建更复杂的网络
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64  # 输入特征矩阵深度，对应表格中maxpool后的特征矩阵深度，都是64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)  # 对应表格中的7×7卷积层，输入特征矩阵(rgb图像)深度为3，stride=2，bias=False
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 对应表格第2层，最大池化下采样
        self.layer1 = self._make_layer(block, 64, blocks_num[0])  # layer1对应表格中conv2所包含的一系列残差结构
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)  # layer2对应表格中conv3所包含的一系列残差结构
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)  # layer3对应表格中conv4所包含的一系列残差结构
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)  # layer4对应表格中conv5所包含的一系列残差结构
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)，自适应平均池化下采样
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():  # 初始化操作
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        # block即残差结构BasicBlock或Bottleneck；channel是残差结构中卷积层所使用卷积核的个数(对应第1层卷积核个数)
        # block_num指该层一共包含了多少个残差结构
        downsample = None
        # 对于第1层而言，没有输入stride，默认等于1；对于18层或34层网络而言，由于expansion=1，则in_channel=channel*expansion，不执行下列if语句
        # 而对于50,101,152层网络而言，expansion=4，in_channel！=channel*expansion，会执行下面的if语句
        # 但从第2层开始，stride=2,不论多少层的网络，都会生成虚线残差结构
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(  # 定义下采样函数
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                # 而对于50,101,152层网络而言，在conv2所对应的一系列残差结构的第1层中，虽然是虚线残差结构，但是只需要调整特征矩阵深度，因此第1层默认stride=1
                # 而对于cmv3,cnv4,conv5，不仅调整深度，还要将高和宽缩小为一半，因此在layer2,layer3,layer4中需要传入参数stride=2
                # 输出特征矩阵深度为channel * block.expansion
                nn.BatchNorm2d(channel * block.expansion))  # 对应的BN层传入的特征矩阵深度为channel * block.expansion

        layers = []  # 定义1个空列表
        # 因为不同深度的网络残差结构中的第1层卷积层操作不同，故需要分而治之
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride, groups=self.groups,
                            width_per_group=self.width_per_group))
        # 首先将第1层残差结构添加进去，block即BasicBlock或Bottleneck，传入参数有输入特征矩阵深度self.in_channel(64)，
        # 残差结构所对应主分支上第1层卷积层的卷积核个数channel,定义的下采样函数和stride参数
        # 对于18/34layers网络，第一层残差结构为实线结构，downsample=None；
        # 对50/101/152layers的网络，第一层残差结构为虚线结构，将特征矩阵的深度翻4倍，高和宽不变。且对于layer1而言，stride=1
        self.in_channel = channel * block.expansion
        # 对于18/34layers网络，expansion=1，输出深度不变；对于50/101/152layers的网络，expansion=4,输出深度翻4倍。

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel, groups=self.groups, width_per_group=self.width_per_group))
        # 通过循环，将剩下一系列的实线残差结构压入layers[]，不管是18/34/50/101/152layers，从它们的第2层开始，全都是实线残差结构。
        # 注意循环从1开始，因为第1层已经搭接好。传入输入特征矩阵深度和残差结构第1层卷积核个数
        return nn.Sequential(*layers)  # 构建好layers[]列表后，通过非关键字参数的形式传入到nn.Sequential，将定义的一系列层结构组合在一起并返回得到layer1

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)  # BN层位于卷积层和relu函数中间
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)  # conv2对应的一系列残差结构
        x = self.layer2(x)  # conv3对应的一系列残差结构
        x = self.layer3(x)  # conv4对应的一系列残差结构
        x = self.layer4(x)  # conv5对应的一系列残差结构

        if self.include_top:
            x = self.avgpool(x)  # 平均池化下采样
            x = torch.flatten(x, 1)  # 展平处理
            x = self.fc(x)  # 全连接

        return x


def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)
    # 对于resnet34，block选用BasicBlock，残差层个数分别是[3,4,6,3]。如果是resnet18,则为[2,2,2,2]


def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)
    # 对于resnet50，block选用Bottleneck，残差层个数分别是[3,4,6,3]。

def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)