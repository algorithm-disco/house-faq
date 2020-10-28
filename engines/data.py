# -*- coding: utf-8 -*-
# @Time : 2020/10/28 21:13 
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : data.py 
# @Software: PyCharm
from torch.utils.data import Dataset
from transformers import BertTokenizer
from tqdm import tqdm
import torch


class DataPrecessForSentence(Dataset):
    """
    文本处理
    """
    def __init__(self, df_data, logger):
        self.logger = logger
        self.tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm-ext')
        self.max_sequence_length = 103
        self.seqs, self.seq_masks, self.seq_segments, self.labels = self.prepare_data(df_data)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.seqs[idx], self.seq_masks[idx], self.seq_segments[idx], self.labels[idx]

    def prepare_data(self, df_data):
        input_ids_q, input_masks_q, input_segments_q = [], [], []
        for _, row in tqdm(df_data.iterrows()):
            query, reply = row.query, row.reply
            inputs = self.tokenizer.encode_plus(query, reply, add_special_tokens=True,
                                                max_length=self.max_sequence_length,
                                                truncation_strategy='longest_first')
            input_ids = inputs['input_ids']
            input_masks = [1] * len(input_ids)
            input_segments = inputs['token_type_ids']
            padding_length = self.max_sequence_length - len(input_ids)
            padding_id = self.tokenizer.pad_token_id
            input_ids = input_ids + ([padding_id] * padding_length)
            input_masks = input_masks + ([0] * padding_length)
            input_segments = input_segments + ([0] * padding_length)
            input_ids_q.append(input_ids)
            input_masks_q.append(input_masks)
            input_segments_q.append(input_segments)
        labels = df_data['label'].values
        return torch.LongTensor(input_ids_q), torch.LongTensor(input_masks_q), torch.LongTensor(input_segments_q), \
            torch.LongTensor(labels),









