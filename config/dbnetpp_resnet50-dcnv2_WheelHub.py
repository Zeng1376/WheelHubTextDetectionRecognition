# 用dbnetpp在WheelHub进行微调的配置文件

_base_ = [
    '/home/h666/zengyue/gd/WHTextNet/mmocr/configs/textdet/dbnetpp/_base_dbnetpp_resnet50-dcnv2_fpnc.py',
    '../mmocr/configs/textdet/_base_/default_runtime.py',
    'WheelHub.py',
    '../mmocr/configs/textdet/_base_/schedules/schedule_sgd_1200e.py',
]

load_from = 'https://download.openmmlab.com/mmocr/textdet/dbnetpp/tmp_1.0_pretrain/dbnetpp_r50dcnv2_fpnc_100k_iter_synthtext-20220502-352fec8a.pth'  # noqa

# dataset settings
train_list = [_base_.WheelHub_textdet_train]
test_list = [_base_.WheelHub_textdet_test]

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=train_list,
        pipeline=_base_.train_pipeline))

val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=test_list,
        pipeline=_base_.test_pipeline))

test_dataloader = val_dataloader

auto_scale_lr = dict(base_batch_size=16)