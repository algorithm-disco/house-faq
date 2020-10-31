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
from sklearn.model_selection import GroupKFold
from transformers.optimization import AdamW
from tqdm import tqdm
import torch


def train(device, logger):
    batch_size = 128
    epoch = 30
    learning_rate = 2e-05
    adam_epsilon = 1e-05
    patience = 3
    train_query_file = 'datasets/train/train.query.1.tsv'
    train_reply_file = 'datasets/train/train.reply.1.tsv'
    # 加载训练语料
    train_left = pd.read_csv(train_query_file, sep='\t', header=None)
    train_left.columns = ['id', 'query']
    train_right = pd.read_csv(train_reply_file, sep='\t', header=None)
    train_right.columns = ['id', 'id_sub', 'reply', 'label']
    train_data = train_left.merge(train_right, how='left')
    train_data['reply'] = train_data['reply'].fillna('好的')

    model = BertEsimModel(device).to(device)
    params = list(model.parameters())
    optimizer = AdamW(params, lr=learning_rate, eps=adam_epsilon)

    # N折交叉验证
    gkf = GroupKFold(n_splits=5).split(X=train_data.reply, groups=train_data.id)

    for fold, (train_idx, valid_idx) in enumerate(gkf):
        train_data_manger = DataPrecessForSentence(train_data.iloc[train_idx], logger)
        logger.info('train_data_length:{}\n'.format(len(train_data_manger)))
        train_loader = DataLoader(train_data_manger, shuffle=True, batch_size=batch_size)

        dev_data_manger = DataPrecessForSentence(train_data.iloc[valid_idx], logger)
        logger.info('dev_data_length:{}\n'.format(len(dev_data_manger)))
        dev_loader = DataLoader(dev_data_manger, shuffle=True, batch_size=batch_size)

    # for i in range(epoch):
    #     logger.info('epoch:{}/{}'.format(i + 1, epoch))
    #     for step, (batch_ids, batch_masks, batch_segments, batch_labels) in enumerate(tqdm(train_loader)):
    #         ids, masks, segments, labels = batch_ids.to(device), batch_masks.to(device), batch_segments.to(
    #             device), batch_labels.to(device)
    #         logits, probabilities = model(ids, masks, segments)





















