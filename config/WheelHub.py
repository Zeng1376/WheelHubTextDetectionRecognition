WheelHub_textdet_data_root = 'dataset/WheelHub'

# 训练集配置
WheelHub_textdet_train = dict(
    type='OCRDataset',
    data_root=WheelHub_textdet_data_root,  # 数据根目录
    ann_file='textdet_train.json',  # 标注文件名称
    filter_cfg=dict(filter_empty_gt=True, min_size=32),  # 数据过滤
    pipeline=None)
# 测试集配置
WheelHub_textdet_test = dict(
    type='OCRDataset',
    data_root=WheelHub_textdet_data_root,
    ann_file='textdet_test.json',
    test_mode=True,
    pipeline=None)
