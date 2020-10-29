# -*- coding: utf-8 -*-
# @Time : 2020/10/29 23:35 
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : model.py 
# @Software: PyCharm
from abc import ABC
from torch import nn
from transformers import BertModel
import torch


class EsimModel(nn.Module, ABC):
    def __init__(self, device, num_classes=2):
        super().__init__()
        self.dropout = 0.5
        self.hidden_size = 768
        self.hidden_dim = 200
        self.num_classes = num_classes
        self.bilstm1 = nn.LSTM(self.hidden_dim, self.hidden_size, batch_first=True, bidirectional=True)

    def forward(self):
        pass





