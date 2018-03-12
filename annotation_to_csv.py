# -*- coding: utf-8 -*-
import os
import glob
import pandas as pd
import json
import xml.etree.ElementTree as ET

# json数据转为csv格式


def json_to_csv(path):
    json_list = []
    for json_file_path in glob.glob(path + '/*.json'):
        with open(json_file_path) as json_file:
            data = json.load(json_file)
            # 如果一个图片存在多个标注对象，需要做对应循环
            for member in data['objects']:
                value = (data['filename'],  # 文件名
                         int(data['image_w_h'][0]),  # 图片宽度
                         int(data['image_w_h'][1]),  # 图片高度
                         member['label'].encode('utf-8'),  # 标注信息
                         int(member['x_y_w_h'][0]),  # 左上角X坐标
                         int(member['x_y_w_h'][1]),
                         int(member['x_y_w_h'][0] + \
                             member['x_y_w_h'][2]),  # 右下角X坐标
                         int(member['x_y_w_h'][1] + member['x_y_w_h'][3]),
                         data['path'])  # 右下角Y坐标
                json_list.append(value)
    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax', 'path']
    json_df = pd.DataFrame(json_list, columns=column_name)
    return json_df


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            path = root.find('path').text
            endIndex = path.rfind('\\')
            lastIndex = path.rfind('\\', 0 ,endIndex)
            dirName = path[lastIndex + 1: endIndex]
            fileName = path[endIndex + 1: len(path)]
            if dirName == 'white':
                dirName = 'pic'
            value = (dirName + '/' + fileName,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    image_path = os.path.join(os.getcwd(), 'annotations')
    json_df = xml_to_csv(image_path)
    json_df.to_csv('data/bm_labels.csv', index=None)
    print('Succesfully converted json to csv')


main()
