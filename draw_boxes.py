# -*- coding: utf-8 -*-

import cv2
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

full_labels = pd.read_csv('data/bm_labels.csv')

def draw_boxes(image_name):
  selected_value =  full_labels[full_labels.filename == image_name]
  img = cv2.imread('images/{}'.format(image_name))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  # 在图片绘制标注框
  for index, row in selected_value.iterrows():
    print(row)
    img = cv2.rectangle(img, (row['xmin'], row['ymin']), (row['xmax'], row['ymax']), (0, 255, 0), 3)
  return img

plt.figure()
plt.imshow(Image.fromarray(draw_boxes('bluemoon_zhizun_suyalanxiang_660g/74.jpg')))
plt.show()
# draw_boxes('9.JPG')