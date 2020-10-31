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
from engines.utils.metrics import correct_predictions, cal_metrics
from sklearn.model_selection import GroupKFold
from transformers.optimization import AdamW
from tqdm import tqdm
import torch
import time


def evaluate(logger, device, model, criterion, dev_data_loader):
    """
    验证集评估函数，分别计算每个类别的f1、precision、recall
    """
    model.eval()
    start_time = time.time()
    loss_sum = 0.0
    correct_preds = 0
    all_predicts = []
    all_labels = []
    with torch.no_grad():
        for step, (batch_ids, batch_masks, batch_segments, batch_labels) in enumerate(tqdm(dev_data_loader)):
            ids, masks, segments, labels = batch_ids.to(device), batch_masks.to(device), batch_segments.to(
                device), batch_labels.to(device)
            logits, probabilities = model(ids, masks, segments)
            loss = criterion(logits, labels)
            loss_sum += loss.item()
            correct_preds += correct_predictions(probabilities, labels)
            predicts = torch.argmax(probabilities, dim=1)
            all_predicts.extend(predicts.cpu())
            all_labels.extend(batch_labels.cpu())
    val_time = time.time() - start_time
    val_loss = loss_sum / len(dev_data_loader)
    val_accuracy = correct_preds / len(dev_data_loader.dataset)
    val_measures = cal_metrics(all_predicts, all_labels)
    val_measures['accuracy'] = val_accuracy
    res_str = ''
    for k, v in val_measures.items():
        res_str += (k + ': %.3f ' % v)
    logger.info('training step: %5d, loss: %.5f, %s' % (step, val_loss, res_str))
    logger.info('time consumption of evaluating:%.2f(min)' % val_time)
    return val_measures


def train(device, logger):
    batch_size = 128
    epoch = 30
    learning_rate = 0.0004
    patience = 3
    print_per_batch = 40
    train_query_file = 'datasets/train/train.query.tsv'
    train_reply_file = 'datasets/train/train.reply.tsv'
    # 加载训练语料
    train_left = pd.read_csv(train_query_file, sep='\t', header=None)
    train_left.columns = ['id', 'query']
    train_right = pd.read_csv(train_reply_file, sep='\t', header=None)
    train_right.columns = ['id', 'id_sub', 'reply', 'label']
    train_data = train_left.merge(train_right, how='left')
    train_data['reply'] = train_data['reply'].fillna('好的')

    criterion = torch.nn.CrossEntropyLoss()

    # N折交叉验证
    gkf = GroupKFold(n_splits=5).split(X=train_data.reply, groups=train_data.id)

    for fold, (train_idx, valid_idx) in enumerate(gkf):
        train_data_manger = DataPrecessForSentence(train_data.iloc[train_idx], logger)
        logger.info('train_data_length:{}\n'.format(len(train_data_manger)))
        train_loader = DataLoader(train_data_manger, shuffle=True, batch_size=batch_size)

        val_data_manger = DataPrecessForSentence(train_data.iloc[valid_idx], logger)
        logger.info('dev_data_length:{}\n'.format(len(val_data_manger)))
        val_loader = DataLoader(val_data_manger, shuffle=True, batch_size=batch_size)

        model = BertEsimModel(device).to(device)
        params = list(model.parameters())
        optimizer = AdamW(params, lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=0)
        for i in range(epoch):
            train_start = time.time()
            logger.info('epoch:{}/{}'.format(i + 1, epoch))
            loss, loss_sum = 0.0, 0.0
            correct_preds = 0
            model.train()
            for step, (batch_ids, batch_masks, batch_segments, batch_labels) in enumerate(tqdm(train_loader)):
                ids, masks, segments, labels = batch_ids.to(device), batch_masks.to(device), batch_segments.to(
                    device), batch_labels.to(device)
                optimizer.zero_grad()
                logits, probabilities = model(ids, masks, segments)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                correct_preds += correct_predictions(probabilities, labels)
                # 打印下训练过程中的指标
                if step % print_per_batch == 0 and step != 0:
                    predicts = torch.argmax(probabilities, dim=1)
                    measures = cal_metrics(predicts.cpu(), labels.cpu())
                    res_str = ''
                    for k, v in measures.items():
                        res_str += (k + ': %.3f ' % v)
                    logger.info('training step: %5d, loss: %.5f, %s' % (step, loss, res_str))
            train_time = (time.time() - train_start) / 60
            train_accuracy = correct_preds / len(train_loader.dataset)
            scheduler.step(train_accuracy)
            logger.info('time consumption of training:%.2f(min)' % train_time)
            logger.info('start evaluate engines...')
            val_measures = evaluate(logger, device, model, criterion, val_loader)


























