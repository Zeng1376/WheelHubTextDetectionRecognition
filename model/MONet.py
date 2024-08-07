import numpy as np
import torchvision
import torch
from datetime import datetime
import torch.nn as nn
from torchvision import datasets, transforms
import cv2 as cv
from PIL import Image

from tools.during_train import train_one_epoch, test_one_epoch
from dataset.Dataset import WheelDataset
from model.Model import Model
from model.Detector import Detector
from model.processor import get_roi_feature


class MON(nn.Module):

    def __init__(self, cfg):
        super(MON, self).__init__()
        self.cfg = cfg
        self.detector = Detector(det=cfg['character_detector']['config_path'],
                                 det_weights=cfg['character_detector']['weight_path'])
        self.detector_word = Detector(det=cfg['word_detector']['config_path'],
                                 det_weights=cfg['word_detector']['weight_path'])
        self.model = Model(cfg["recognizer"]["use_attention"])
        self.model.load_state_dict(torch.load(cfg['recognizer']['weight_path']))
        self.model.cuda()
        self.model.eval()
        self.char_classes = "_0123456789abcdefghijklmnopqrstuvwxyz"

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.Normalize(mean=[0.458, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.224])
        ])

    def demo(self, img_path: str, show=True):
        image = cv.imread(img_path)
        image_show = np.full(image.shape, 255, dtype=np.uint8)

        img = Image.open(img_path)
        img = self.transform(img).cuda()
        img = torch.unsqueeze(img, 0)

        feature_map, polygons = self.detector.forward(img_path)
        polygons = polygons[0]['det_polygons']

        _, p_w = self.detector_word.forward(img_path)
        p_w = p_w[0]['det_polygons']

        # 判断每个字符的种类
        for p in polygons:
            # bb = cal_roi(p)
            im = get_roi_feature(img[0:1, :, :, :], p, scale=2,
                                 use_mask=self.cfg['recognizer']['use_mask'],
                                 use_dilate=self.cfg['recognizer']['use_dilate'])
            feature = get_roi_feature(feature_map, p, 2, use_mask=self.cfg['recognizer']['use_mask'],
                                      use_dilate=self.cfg['recognizer']['use_dilate'])
            feature = torch.cat((im, feature), dim=1)
            pred = self.model(feature)

            poly = []
            center = np.array([0,0])
            for i in range(len(p) // 2):
                center = center+np.array([int(p[2 * i]), int(p[2 * i + 1])])
                poly.append([int(p[2 * i]), int(p[2 * i + 1])])
            p_start = np.array(poly[0])  # 用于计算朝向
            p_end = np.array(poly[1])
            poly = np.array(poly, np.int32)
            center = center / (len(p) // 2)-np.array([10,-10])
            # cv.polylines(image, [np.array(poly, np.int32).reshape((-1, 1, 2))], True, (0, 0, 255), 1)
            # print(bb)

            # 计算朝向，便于后续写字
            direction_vector = p_start - p_end

            cv.putText(image_show, self.char_classes[torch.argmax(pred, dim=1)],
                       np.asarray(center,
                                  dtype=np.int32),
                       cv.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255), 2)

            # print(self.char_classes[torch.argmax(pred, dim=1)])
            # print(torch.argmax(pred, dim=1))
            # print("")

        for p in p_w:
            poly = []
            for i in range(len(p) // 2):
                poly.append([int(p[2 * i]), int(p[2 * i + 1])])
            poly = np.array(poly, np.int32)
            cv.polylines(image, [np.array(poly, np.int32).reshape((-1, 1, 2))], True, (0, 0, 255), 2)
            cv.polylines(image_show, [np.array(poly, np.int32).reshape((-1, 1, 2))], True, (0, 0, 255), 2)

        image = np.concatenate((image, image_show), axis=1)

        if show:
            cv.namedWindow(img_path, 0)
            cv.imshow(img_path, image)
            cv.waitKey(0)
            cv.imwrite("/home/h666/zengyue/gd/WHTextNet/demo_results/"+img_path[-10:], image)
            print("/home/h666/zengyue/gd/WHTextNet/demo_results/"+img_path[-10:])
