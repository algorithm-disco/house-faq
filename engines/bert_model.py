# -*- coding: utf-8 -*-
# @Time : 2020/10/29 23:35 
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : bert_esim_model.py
# @Software: PyCharm
from abc import ABC
from torch import nn
from transformers import BertModel
import torch.nn.functional as F
import torch


class BertwwmModel(nn.Module, ABC):
    def __init__(self, device, num_classes=2):
        super().__init__()
        self.device = device
        self.dropout = 0.5
        self.embedding_dim = 768
        self.linear_size = 128
        self.num_classes = num_classes
        self.linear = nn.Linear(4 * self.embedding_dim, num_classes)
        self.dropout = nn.Dropout(self.dropout)
        self.bertwwm_model = BertModel.from_pretrained('hfl/chinese-bert-wwm-ext').to(device)
        for param in self.bertwwm_model.parameters():
            param.requires_grad = False

    @staticmethod
    def apply_multiple(x):
        p1 = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        p2 = F.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        # output: batch_size * (4 * hidden_size)
        return torch.cat([p1, p2], -1)

    def forward(self, ids, masks, segments):
        with torch.no_grad():
            bertwwm_hidden = self.bertwwm_model(ids, attention_mask=masks, token_type_ids=segments)[0].to(self.device)
        rep = self.apply_multiple(bertwwm_hidden)
        t = bertwwm_hidden[:, -1]
        e = bertwwm_hidden[:, 0]
        combined = torch.cat([rep, t, e], -1)
        dropout_results = self.dropout(combined)
        logits = self.linear(dropout_results)
        probabilities = F.softmax(logits, dim=-1)
        return logits, probabilities









