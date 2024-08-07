import torch
import torch.nn as nn
import torch.nn.functional as F

from model.local_feature_extraction import BasicBlock, conv3x3


class FeatureFuser(nn.Module):

    def __init__(self, in_channels=1024):
        super(FeatureFuser, self).__init__()

        self.conv1 = conv3x3(in_channels, in_channels // 2)
        self.conv2 = conv3x3(in_channels // 2, in_channels // 4)

        self.layer1 = BasicBlock(in_channels, in_channels)
        self.layer2 = BasicBlock(in_channels // 2, in_channels // 2)
        self.layer3 = BasicBlock(in_channels // 2, in_channels // 2)
        self.layer4 = BasicBlock(in_channels // 4, in_channels // 4)

        # 对网络参数进行初始化
        # nn.init.kaiming_uniform_(self.conv1.weight)
        # nn.init.kaiming_uniform_(self.conv2.weight)
        # nn.init.kaiming_uniform_(self.conv3.weight)
        # nn.init.kaiming_uniform_(self.conv4.weight)

    def forward(self, x):
        x = self.layer1(x)
        x = self.conv1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)
        x = self.layer4(x)
        return x
