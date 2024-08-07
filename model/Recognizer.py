import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch

from model.local_feature_extraction import BasicBlock


class Recognizer(nn.Module):
    def __init__(self, in_planes=256, planes=256, num_classes=37):
        super(Recognizer, self).__init__()
        self.in_planes = in_planes
        self.planes = planes

        self.layer1 = BasicBlock(self.in_planes, self.planes)
        self.layer2 = BasicBlock(self.planes, self.planes)
        self.layer3 = BasicBlock(self.planes, self.planes)
        self.layer4 = BasicBlock(self.planes, self.planes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.planes, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
