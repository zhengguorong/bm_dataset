"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=data/train.record

  # Create test data:
  python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=data/test.record
"""
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    # width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / row['width'])
        xmaxs.append(row['xmax'] / row['width'])
        ymins.append(row['ymin'] / row['height'])
        ymaxs.append(row['ymax'] / row['height'])
        classes_text.append(row['class'])
        if row['class'] == 'bluemoon_shouxi_fengqingbailai_500g':
            classes.append(1)
        elif row['class'] == 'bluemoon_zhizun_suyalanxiang_660g':
            classes.append(2)
        elif row['class'] == 'bluemoon_zhizun_suyalanxiang_box':
            classes.append(3)
        elif row['class'] == 'bluemoon_zhizun_suyalanxiang_bag_600g':
            classes.append(4)
        elif row['class'] == 'bluemoon_zhizun_qingyunmeixiang_box':
            classes.append(5)
        elif row['class'] == 'bluemoon_zhizun_qingyunmeixiang_bag_600g':
            classes.append(6)
        elif row['class'] == 'bluemoon_zhizun_qingyunmeixiang_660g':
            classes.append(7)
        elif row['class'] == 'bluemoon_shouxi_shuangyong_fengqingbailai_500g':
            classes.append(8)
        elif row['class'] == 'bluemoon_zhizun_new_suyalanxiang_660g':
            classes.append(9)
        elif row['class'] == 'bluemoon_zhizun_new_qingyunmeixiang_660g':
            classes.append(10)
        elif row['class'] == 'bluemoon_baobao_red_500g':
            classes.append(11)
        elif row['class'] == 'bluemoon_jiecibao_500g':
            classes.append(12)
        elif row['class'] == 'bluemoon_lanseyueguang_bai_600g':
            classes.append(13)
        elif row['class'] == 'bluemoon_shenceng_500g':
            classes.append(14)
        elif row['class'] == 'bluemoon_weinuo_500g':
            classes.append(15)
        elif row['class'] == 'bluemoon_yurong_500g':
            classes.append(16)
        elif row['class'] == 'bluemoon_ertongxishou_caomei_225g':
            classes.append(17)
        elif row['class'] == 'bluemoon_ertongxishou_qingpingguo_225g':
            classes.append(18)
        elif row['class'] == 'bluemoon_ertongxishou_tiancheng_225g':
            classes.append(19)
        elif row['class'] == 'bluemoon_guopaoduoduo_300ml':
            classes.append(20)
        elif row['class'] == 'bluemoon_xishouye_luhui_500g':
            classes.append(21)
        elif row['class'] == 'bluemoon_xishouye_weie_500g':
            classes.append(22)
        elif row['class'] == 'bluemoon_xishouye_yejuhua_500g':
            classes.append(23)
        elif row['class'] == 'bluemoon_84xiaoduye_1.2kg':
            classes.append(24)
        elif row['class'] == 'bluemoon_bolishui_500g':
            classes.append(25)
        elif row['class'] == 'bluemoon_chaqing_500g':
            classes.append(26)
        elif row['class'] == 'bluemoon_dibanqingjie_600g':
            classes.append(27)
        elif row['class'] == 'bluemoon_piaobaishui_600g':
            classes.append(28)
        elif row['class'] == 'bluemoon_quannengshui_500g':
            classes.append(29)
        elif row['class'] == 'bluemoon_roushunji_500g':
            classes.append(30)
        elif row['class'] == 'bluemoon_yilingjing_500g':
            classes.append(31)
        elif row['class'] == 'bluemoon_yiwuxiaoduye_1kg':
            classes.append(32)
        elif row['class'] == 'bluemoon_youwukexing_500g':
            classes.append(33)
        elif row['class'] == 'bluemoon_liangbai_xunyicao_2kg':
            classes.append(34)
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(row['height']),
        'image/width': dataset_util.int64_feature(row['width']),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def main(_):
  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
  path = os.path.join(os.getcwd(), 'images')
  examples = pd.read_csv(FLAGS.csv_input)
  grouped = split(examples, 'filename')
  for group in grouped:
    tf_example = create_tf_example(group, path)
    writer.write(tf_example.SerializeToString())
  writer.close()
  output_path = os.path.join(os.getcwd(), FLAGS.output_path)
  print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
  tf.app.run()