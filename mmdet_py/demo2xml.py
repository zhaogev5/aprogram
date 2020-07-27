from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import glob
matplotlib.use('TkAgg')
import cv2
from demo.toxml import to_xml

config_file = './configs/faster_rcnn_r50_fpn_1x.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = './configs/checkpoint/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')
imgs = glob.glob('./pic/*.jpg')
print(imgs)
score_thr = 0.5#设置阈值
xml_path = './xml_path/'
for i, img in enumerate(imgs):
  result = inference_detector(model, img)
  #可以删除
  
  bbox_result = result
  bboxes = np.vstack(bbox_result)
  labels = [
          np.full(bbox.shape[0], i, dtype=np.int32)
          for i, bbox in enumerate(bbox_result)
      ]
  labels = np.concatenate(labels)   #得到的到图片名字
  file_name = imgs[i].split('/')[-1]
  print(file_name)
  #print(bboxes)
  scores = bboxes[:, -1]
  inds = scores > score_thr
  bboxes = bboxes[inds, :] #对bbox进行筛选
  labels = labels[inds]
  to_xml(bboxes,xml_path,file_name)







# print(labels)
# bboxes = np.vstack(result) #这里转换成numpy
# print(bboxes.shape) #77,5
# labels = [
#     np.full(bbox.shape[0], i, dtype=np.int32)
#     for i, bbox in enumerate(result)
# ]
# print(labels)


  #表示暂停0表示一直 也可以输入暂停时间]
# plt.imshow(img)
# plt.show()
# result = inference_detector(model, img)


# show_result_pyplot(img, result, model.CLASSES)
