# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

np.random.seed(1)
full_labels = pd.read_csv('data/bm_labels.csv')
gb = full_labels.groupby('filename')
grouped_list = [gb.get_group(x) for x in gb.groups]

# 把76个数据分割为训练集和测试集，训练集为60个，测试集为14个
train_index = np.random.choice(len(grouped_list), size=1400, replace=False)
test_index = np.setdiff1d(list(range(len(grouped_list))), train_index)

# 根据随机索引，分割数据
train = pd.concat([grouped_list[i] for i in train_index])
test = pd.concat([grouped_list[i] for i in test_index])

train.to_csv('data/train_labels.csv', index=None)
test.to_csv('data/test_labels.csv', index=None)