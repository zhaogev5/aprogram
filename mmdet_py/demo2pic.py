from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import glob
matplotlib.use('TkAgg')
import cv2
from demo.toxml import to_xml
import os

config_file = './configs/faster_rcnn_r50_fpn_1x.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = './configs/checkpoint/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')
imgs = glob.glob('./pic/*.jpg')
print(imgs) #['./pic\\1.jpg', './pic\\2.jpg', './pic\\demo.jpg']
score_thr = 0.5#设置阈值
pic_path = './pic_path/'
for i, img in enumerate(imgs):
    print(img) #./pic\1.jpg
    result = inference_detector(model, img)
    file_name = imgs[i].split('/')[-1].split('\\')[1]
    print(file_name)
    out_file = os.path.join(pic_path, file_name)
    show_result(img, result, model.CLASSES, score_thr=score_thr, out_file=out_file)


