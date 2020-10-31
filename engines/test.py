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


def test(logger, device, model, test_loader):
    """
    运行测试集
    """
    label_results = []
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        for step, (batch_ids, batch_masks, batch_segments, batch_labels) in enumerate(tqdm(test_loader)):
            ids, masks, segments, labels = batch_ids.to(device), batch_masks.to(device), batch_segments.to(
                device), batch_labels.to(device)
            logits, probabilities = model(ids, masks, segments)
            predicts = torch.argmax(probabilities, dim=1)
            label_results.extend(predicts.cpu())
    test_time = time.time() - start_time
    logger.info('time consumption of testing:%.2f(min)' % test_time)
    return label_results
