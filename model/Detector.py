from mmocr.apis import MMOCRInferencer
import os.path as osp
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import mmcv
import mmengine
import numpy as np
from rich.progress import track

from mmocr.registry import VISUALIZERS
from mmocr.structures import TextSpottingDataSample
from mmocr.utils import ConfigType, bbox2poly, crop_img, poly2bbox
from mmocr.apis.inferencers.base_mmocr_inferencer import (BaseMMOCRInferencer, InputsType, PredType,
                                                          ResType)
from mmocr.apis.inferencers.kie_inferencer import KIEInferencer
from mmocr.apis.inferencers.textdet_inferencer import TextDetInferencer
from mmocr.apis.inferencers.textrec_inferencer import TextRecInferencer


class Detector():
    """基于MMOCRInferencer修改网络，得到网络的模型以及中间参数"""

    def __init__(self,
                 det: Optional[Union[ConfigType, str]] = None,
                 det_weights: Optional[str] = None,
                 rec: Optional[Union[ConfigType, str]] = None,
                 rec_weights: Optional[str] = None,
                 kie: Optional[Union[ConfigType, str]] = None,
                 kie_weights: Optional[str] = None,
                 device: Optional[str] = None) -> None:
        self.DBnet = MMOCRInferencer(det, det_weights, rec, rec_weights, kie, kie_weights, device)  # DBNet作为检测头
        self.feature_map = None  # detector所生成的feature_map
        self.polygons = None  # detector生成的检测框

    def forward(self, inputs):
        """
        通过DB产生生成的feature-map以及bounding box?
        :param inputs:
        :return:
        """
        self.DBnet(inputs=inputs, batch_size=len(inputs))
        self.feature_map = self.DBnet.textdet_inferencer.model.feature
        self.polygons = self.DBnet.polygon
        return self.feature_map, self.polygons
