# -*- coding: utf-8 -*-
# @Time : 2020/10/31 19:25 
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : test.py 
# @Software: PyCharm
from tqdm import tqdm
import pandas as pd
from engines.data import DataPrecessForSentence
from torch.utils.data import DataLoader
import time
import torch


def test(logger, device, model):
    """
    运行测试集
    """
    batch_size = 128
    # 加载测试语料
    test_query_file = 'datasets/test/test.query.tsv'
    test_reply_file = 'datasets/test/test.reply.tsv'
    test_left = pd.read_csv(test_query_file, sep='\t', header=None, encoding='gbk')
    test_left.columns = ['id', 'query']
    test_right = pd.read_csv(test_reply_file, sep='\t', header=None, encoding='gbk')
    test_right.columns = ['id', 'id_sub', 'reply']
    test_data = test_left.merge(test_right, how='left')
    test_data['label'] = 666
    test_data_manger = DataPrecessForSentence(test_data, logger)
    logger.info('test_data_length:{}\n'.format(len(test_data_manger)))
    test_loader = DataLoader(test_data_manger, shuffle=False, batch_size=batch_size)

    start_time = time.time()
    model.eval()
    with torch.no_grad():
        for step, (batch_ids, batch_masks, batch_segments, batch_labels) in enumerate(tqdm(test_loader)):
            ids, masks, segments, labels = batch_ids.to(device), batch_masks.to(device), batch_segments.to(
                device), batch_labels.to(device)
            logits, probabilities = model(ids, masks, segments)
            predicts = torch.argmax(probabilities, dim=1)
    test_time = time.time() - start_time
    logger.info('time consumption of testing:%.2f(min)' % test_time)
