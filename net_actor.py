import torch
import torch.nn as nn
import torch.nn.functional as F


# Actor 网络
class NetDQN(nn.Module):
    def __init__(self):
        super(NetDQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2)

        conv_output_size = self._get_conv_output_size(640, 640)
        self.fc = nn.Linear(conv_output_size, 256)

        self.fc1 = nn.Linear(256, 256)
        self.fc_move = nn.Linear(256, 2)  # move_action_list Q-values
        self.fc_angle = nn.Linear(256, 360)  # angle_list Q-values
        self.fc_info = nn.Linear(256, 9)  # info_action_list Q-values
        self.fc_attack = nn.Linear(256, 11)  # attack_action_list Q-values
        self.fc_action_type = nn.Linear(256, 3)  # action_type_list Q-values
        self.fc_arg1 = nn.Linear(256, 360)  # arg1_list Q-values
        self.fc_arg2 = nn.Linear(256, 100)  # arg2_list Q-values
        self.fc_arg3 = nn.Linear(256, 5)  # arg3_list Q-values
        self._initialize_weights()

    def _get_conv_output_size(self, height, width):
        dummy_input = torch.zeros(1, 3, height, width)
        with torch.no_grad():
            x = F.relu(self.conv1(dummy_input))
            x = F.relu(self.conv2(x))
        return x.view(x.size(0), -1).size(1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.to(next(self.parameters()).device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        x = F.relu(self.fc1(x))

        move_action_q = self.fc_move(x)
        angle_q = self.fc_angle(x)
        info_action_q = self.fc_info(x)
        attack_action_q = self.fc_attack(x)
        action_type_q = self.fc_action_type(x)
        arg1_q = self.fc_arg1(x)
        arg2_q = self.fc_arg2(x)
        arg3_q = self.fc_arg3(x)

        return move_action_q, angle_q, info_action_q, attack_action_q, action_type_q, arg1_q, arg2_q, arg3_q
