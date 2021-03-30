#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/27 7:55 下午
# @Author  : 宋继贤
# @Description  : 
# @File    : models.py
# @Software: PyCharm


import torch
from torch import nn
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BaseModel(nn.Module):

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.args = args
        self.embed = nn.Sequential(
            nn.Embedding(args.vocab_size, args.embedding_size, padding_idx=0),
            nn.Dropout(args.dropout)
        )

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, sentence_ids, sentence_lengths):
        raise NotImplementedError

    def make_output_layer(self):
        raise NotImplementedError


class LSTMModel(BaseModel):

    def __init__(self, args):
        super(LSTMModel, self).__init__(args)
        self.lstm = nn.LSTM(input_size=args.embedding_size,
                            hidden_size=args.hidden_size,
                            num_layers=args.lstm_layer,
                            dropout=args.dropout,
                            batch_first=True,
                            bidirectional=args.bi)
        self.output_layer = self.make_output_layer()

    def forward(self, sentence_ids, sentence_lengths):
        sentence_embed = self.embed(sentence_ids)
        packed_sentence_embeds = pack_padded_sequence(sentence_embed, sentence_lengths.cpu(), batch_first=True,
                                                      enforce_sorted=False)
        packed_sentence_h, (hidden, cell) = self.lstm(packed_sentence_embeds)
        # sentence_h, _ = pad_packed_sequence(packed_sentence_h, batch_first=True)
        hidden = torch.cat([hidden[-2, :, :], hidden[-1, :, :]], dim=-1)
        pre = self.output_layer(hidden)
        return pre

    def make_output_layer(self):
        layers = []
        last_units = self.args.hidden_size * 2 if self.args.bi else self.args.hidden_size
        for i, units in enumerate(self.args.mlp_units):
            layers.append(nn.Linear(last_units, units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.args.dropout))
            last_units = units
        layers.append(nn.Linear(last_units, self.args.output_classes))
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)


class ConvModel(BaseModel):

    def __init__(self, args):
        super(ConvModel, self).__init__(args)
        self.convs = nn.ModuleList([nn.Conv1d(args.embedding_size, args.conv_hidden, i) for i in range(3, 6)])
        self.output_layer = self.make_output_layer()

    def forward(self, sentence_ids, sentence_lengths):
        sentence_embed = self.embed(sentence_ids).transpose(1, 2)
        conv_out = torch.cat([torch.relu(conv(sentence_embed)).max(dim=2)[0] for conv in self.convs], dim=1)
        pre = self.output_layer(conv_out)
        return pre

    def make_output_layer(self):
        layers = []
        last_units = self.args.conv_hidden * 3
        for i, units in enumerate(self.args.mlp_units):
            layers.append(nn.Linear(last_units, units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.args.dropout))
            last_units = units
        layers.append(nn.Linear(last_units, self.args.output_classes))
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)
