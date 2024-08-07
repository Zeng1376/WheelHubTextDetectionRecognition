import json
from model.MONet import MON

with open("./config/WHTextNet.json", 'r') as f:
    cfg = json.load(f)
    print(cfg)
    pass

model = MON(cfg["network"])
model.demo('/home/h666/zengyue/gd/WHTextNet/dataset/WheelHub/textdet_imgs/train/000360.bmp')