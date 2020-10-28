# -*- coding: utf-8 -*-
# @Time : 2020/10/28 23:38 
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : train.py 
# @Software: PyCharm
import pandas as pd
from torch.utils.data import DataLoader
from engines.data import DataPrecessForSentence


def train(device, logger):
    batch_size = 128
    train_query_file = 'datasets/train/train.query.tsv'
    train_reply_file = 'datasets/train/train.reply.tsv'
    # 加载训练语料
    train_left = pd.read_csv(train_query_file, sep='\t', header=None)
    train_left.columns = ['id', 'query']
    train_right = pd.read_csv(train_reply_file, sep='\t', header=None)
    train_right.columns = ['id', 'id_sub', 'reply', 'label']
    train_data = train_left.merge(train_right, how='left')
    train_data['reply'] = train_data['reply'].fillna('好的')
    train_data = DataPrecessForSentence(train_data, logger)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)









