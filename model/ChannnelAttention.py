import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch

from model.local_feature_extraction import BasicBlock


class ChannelAttention(nn.Module):

    def __init__(self, in_planes=256, planes=256):
        super(ChannelAttention, self).__init__()
        self.in_planes = in_planes
        self.planes = planes

        self.layer1 = BasicBlock(self.in_planes, self.planes)
        self.layer2 = BasicBlock(self.planes, self.planes)
        self.layer3 = BasicBlock(self.planes, self.planes)
        self.layer4 = BasicBlock(self.planes, self.planes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        feature = x.clone()
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).expand(-1, -1, feature.shape[2], feature.shape[3])
        feature = torch.mul(feature, x)
        return feature
