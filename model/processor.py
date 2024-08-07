import numpy as np
import torch
import torch.nn.functional as F
import cv2 as cv


def get_roi_feature(feature_map, polygon, scale=2, square_len=30, use_mask=True, use_dilate=True):
    """
    从整张图中提取ROI, feature
    :param feature_map:单个图像的feature_map 1*C*H*W
    :param polygon: 这里应当只有一个polygon
    :param scale:
    :param square_len:
    :param use_mask:是否采用硬掩码
    :param use_dilate:是否采用膨胀
    :return:

    """
    bb_h = cal_roi(polygon, scale*square_len, scale)

    # 检查是否出现越界
    for i in range(len(bb_h)):
        if bb_h[i] > feature_map.shape[2] * scale:
            bb_h[i] = feature_map.shape[2] * scale
        if bb_h[i] < 0:
            bb_h[i] = 0

    f_0 = feature_map[:, :, int(bb_h[1] / scale):int(bb_h[3] / scale), int(bb_h[0] / scale):int(bb_h[2] / scale)]
    # square_len = np.max([f_0.shape[2], f_0.shape[3]])

    if use_mask:
        mask = np.zeros((f_0.shape[2], f_0.shape[3]))
        pts = np.array(polygon, np.int32)
        pts = pts.reshape((-1, 2))
        for i in range(pts.shape[0]):
            pts[i, :] = (pts[i, :] - np.array(bb_h[:2])) / 2

        cv.fillPoly(mask, [pts], color=(1))

        # 对硬掩码区域进行膨胀操作,以解决边界不准确的问题
        if use_dilate:
            kernel = np.array([[1, 1, 1],
                               [1, 1, 1],
                               [1, 1, 1]], dtype=int)
            mask = cv.dilate(mask, kernel)

        # cv.namedWindow("test", 0)
        # cv.imshow("test", mask*255)
        # cv.waitKey(0)
        # print(square_len)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
        f_0 = f_0 * mask

    f_0 = F.pad(f_0, (0, square_len - f_0.shape[3], 0, square_len - f_0.shape[2]))  # 使得得到一个正方形的ROI_tensor
    # # print(f_0.shape)
    # f_90 = torch.rot90(f_0, k=1, dims=(2, 3))
    # f_180 = torch.rot90(f_0, k=2, dims=(2, 3))
    # f_270 = torch.rot90(f_0, k=3, dims=(2, 3))
    # feature = torch.cat((f_0, f_90, f_180, f_270), dim=1)
    return f_0


def cal_roi(polygon: list, square_len=60, scale=2):
    """
    计算roi区域应该是多少
    :param polygon:此处的polygon是一张图像中所有的polygon
    :param square_len:
    :param scale:
    :return:
    """
    poly_x = np.asarray(polygon)[::2]
    poly_y = np.asarray(polygon)[1::2]
    # square_len = np.max([np.max(poly_x) - np.min(poly_x), np.max(poly_y) - np.min(poly_y)])
    # square_len = 60
    square_center = [(np.max(poly_x) + np.min(poly_x)) / scale, (np.max(poly_y) + np.min(poly_y)) / scale]
    bbox_h = [int(square_center[0] - square_len / scale), int(square_center[1] - square_len / scale),
              int(square_center[0] + square_len / scale),
              int(square_center[1] + square_len / scale)]  # 产生bounding box水平方向
    return bbox_h
