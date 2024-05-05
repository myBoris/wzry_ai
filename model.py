import torch.nn as nn
import torch.nn.functional as F

class WzryNet(nn.Module):
    def __init__(self):
        super(WzryNet, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # The fully connected layers will be initialized later
        self.fc1 = None

        # 左手的操作，有4个，(无操作,移动，购买装备1，购买装备2)
        self.fc1_action = nn.Linear(512, 4)
        self.fc1_angle = nn.Linear(512, 360) #360度

        # 右手的操作，有18个，
        # (无操作,回城，恢复，召唤师技能，攻击，   5个
        # 攻击小兵，攻击塔，发起进攻，开始撤退，请求集合， 5个
        # 1技能，2技能，3技能，4技能，升级1技能， 5个
        # 升级2技能，升级3技能， 升级4技能, 装备技能) 4个
        self.fc2_action = nn.Linear(512, 19)
        self.fc2_type = nn.Linear(512, 3)  # 技能有(点击，滑动，长按)三种释放方式
        self.fc2_angle = nn.Linear(512, 360) #360度
        self.fc2_duration = nn.Linear(512, 1) #长按时间

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Dynamically calculate the size of the flattened features
        n_size = x.view(x.size(0), -1).size(1)
        if self.fc1 is None:
            self.fc1 = nn.Linear(n_size, 512).to(x.device)  # Initialize fc1 once the input size is known

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))

        # Outputs for the first set of actions
        action1 = self.fc1_action(x)
        angle1 = self.fc1_angle(x)

        # Outputs for the second set of actions
        action2 = self.fc2_action(x)
        type2 = self.fc2_type(x)
        angle2 = self.fc2_angle(x)
        duration2 = F.relu(self.fc2_duration(x))

        return (action1, angle1), (action2, type2, angle2, duration2)

