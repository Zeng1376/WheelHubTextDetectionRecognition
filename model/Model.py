import torch.nn as nn
import torch

from model.ChannnelAttention import ChannelAttention
from model.FeatureFuser import FeatureFuser
from model.local_feature_extraction import ResNetFeatureExtractor
from model.Resnet import resnet18
from model.Recognizer import Recognizer
from model.Detector import Detector
from model.processor import get_roi_feature


class Model(nn.Module):
    def __init__(self, use_attention=True):
        super(Model, self).__init__()
        self.local_feature_extractor = ResNetFeatureExtractor(3, 256)
        self.use_attention = use_attention
        self.feature_fuser = FeatureFuser(512)
        self.channel_attention = ChannelAttention(128, 128)
        self.recognizer = Recognizer(128,128)
        # self.recognizer = models.resnet18()
        # self.recognizer.fc = nn.Linear(self.recognizer.fc.in_features, 37)
        # self.recognizer.conv1 = nn.Conv2d(256,64,kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        img = x[:, :3, :, :]  # 这里是图片
        global_feature = x[:, 3:, :, :]
        local_feature = self.local_feature_extractor(img)
        feature = torch.cat((global_feature, local_feature), dim=1)
        feature = self.feature_fuser(feature)
        if self.use_attention:
            feature = self.channel_attention(feature)
        # x = F.interpolate(x, (228, 228), mode='bilinear', align_corners=False)
        feature = self.recognizer(feature)
        return feature
