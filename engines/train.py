# -*- coding: utf-8 -*-
# @Time : 2020/10/28 23:38 
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : train.py 
# @Software: PyCharm
import pandas as pd
from torch.utils.data import DataLoader
from engines.data import DataPrecessForSentence
from engines.bert_esim_model import BertEsimModel
from transformers import BertModel
from transformers.optimization import AdamW
from tqdm import tqdm
import torch


def train(device, logger):
    batch_size = 128
    epoch = 30
    learning_rate = 2e-05
    train_query_file = 'datasets/train/train.query.1.tsv'
    train_reply_file = 'datasets/train/train.reply.1.tsv'
    # 加载训练语料
    train_left = pd.read_csv(train_query_file, sep='\t', header=None)
    train_left.columns = ['id', 'query']
    train_right = pd.read_csv(train_reply_file, sep='\t', header=None)
    train_right.columns = ['id', 'id_sub', 'reply', 'label']
    train_data = train_left.merge(train_right, how='left')
    train_data['reply'] = train_data['reply'].fillna('好的')
    train_data = DataPrecessForSentence(train_data, logger)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

    model = BertEsimModel(device).to(device)
    # optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

    for i in range(epoch):
        logger.info('epoch:{}/{}'.format(i + 1, epoch))
        for step, (batch_ids, batch_masks, batch_segments, batch_labels) in enumerate(tqdm(train_loader)):
            ids, masks, segments, labels = batch_ids.to(device), batch_masks.to(device), batch_segments.to(
                device), batch_labels.to(device)
            logits, probabilities = model(ids, masks, segments)





















