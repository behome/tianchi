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
from torch.nn.utils.rnn import pack_padded_sequence


class BaseModel(nn.Module):

    def __init__(self, args, embedding=None, co_occur=None):
        super(BaseModel, self).__init__()
        self.args = args
        if embedding is not None:
            self.embed = nn.Sequential(
                nn.Embedding.from_pretrained(embedding),
                nn.Dropout(args.dropout)
            )
        else:
            self.embed = nn.Sequential(
                nn.Embedding(args.vocab_size, args.embedding_size, padding_idx=0),
                nn.Dropout(args.dropout)
            )
        self.output_layer = self.make_output_layer(args.embedding_size)
        self.co_occur_layer = nn.Linear(17, args.output_classes, bias=False)
        if co_occur is not None:
            self.co_occur_layer.weight.data = co_occur
            if args.fix_occur:
                self.co_occur_layer.weight.requires_grad = False
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear) and m != self.co_occur_layer:
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, sentence_ids, sentence_lengths):
        sentence_embed = self.embed(sentence_ids)
        sentence_vec = sentence_embed.sum(1) / sentence_lengths.unsqueeze(1)
        multi_pre = self.output_layer(sentence_vec)
        pre = torch.sigmoid(self.co_occur_layer(multi_pre))
        multi_pro = torch.sigmoid(multi_pre)
        return multi_pro, pre

    def make_output_layer(self, last_units):
        layers = []
        for i, units in enumerate(self.args.mlp_units):
            layers.append(nn.Linear(last_units, units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.args.dropout))
            last_units = units
        layers.append(nn.Linear(last_units, 17))
        return nn.Sequential(*layers)


class LSTMModel(BaseModel):

    def __init__(self, args, embedding=None, co_occur=None):
        super(LSTMModel, self).__init__(args, embedding, co_occur)
        self.lstm = nn.LSTM(input_size=args.embedding_size,
                            hidden_size=args.hidden_size,
                            num_layers=args.lstm_layer,
                            dropout=args.dropout,
                            batch_first=True,
                            bidirectional=args.bi)
        last_units = self.args.hidden_size * 2 if self.args.bi else self.args.hidden_size
        self.output_layer = self.make_output_layer(last_units)
        self.init_weight()

    def forward(self, sentence_ids, sentence_lengths):
        sentence_embed = self.embed(sentence_ids)
        packed_sentence_embeds = pack_padded_sequence(sentence_embed, sentence_lengths.cpu(), batch_first=True,
                                                      enforce_sorted=False)
        packed_sentence_h, (hidden, cell) = self.lstm(packed_sentence_embeds)
        # sentence_h, _ = pad_packed_sequence(packed_sentence_h, batch_first=True)
        if self.args.bi:
            hidden = torch.cat([hidden[-2, :, :], hidden[-1, :, :]], dim=-1)
        else:
            hidden = hidden[-1, :, :]
        multi_pre = self.output_layer(hidden)
        pre = torch.sigmoid(self.co_occur_layer(multi_pre))
        multi_pro = torch.sigmoid(multi_pre)
        return multi_pro, pre


class ConvModel(BaseModel):

    def __init__(self, args, embedding=None, co_occur=None):
        super(ConvModel, self).__init__(args, embedding, co_occur)
        self.convs = nn.ModuleList([nn.Conv1d(args.embedding_size, args.conv_hidden, i) for i in range(3, 4)])
        last_units = self.args.conv_hidden
        self.output_layer = self.make_output_layer(last_units)
        self.init_weight()

    def forward(self, sentence_ids, sentence_lengths):
        sentence_embed = self.embed(sentence_ids).transpose(1, 2)
        conv_out = torch.cat([torch.relu(conv(sentence_embed)).max(dim=2)[0] for conv in self.convs], dim=1)
        multi_pre = self.output_layer(conv_out)
        pre = torch.sigmoid(self.co_occur_layer(multi_pre))
        multi_pro = torch.sigmoid(multi_pre)
        return multi_pro, pre


class CharCNNModel(BaseModel):

    def __init__(self, args, embedding=None, co_occur=None):
        super(CharCNNModel, self).__init__(args, embedding, co_occur)
        self.convs = nn.Sequential(
            nn.Conv1d(self.args.embedding_size, self.args.conv_hidden, 1),
            nn.BatchNorm1d(self.args.conv_hidden),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1),
            nn.Conv1d(self.args.conv_hidden, self.args.conv_hidden, 3, dilation=2),
            nn.BatchNorm1d(self.args.conv_hidden),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1),
            nn.Conv1d(self.args.conv_hidden, self.args.conv_hidden, 3, dilation=4),
            nn.BatchNorm1d(self.args.conv_hidden),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1),
        )
        last_units = self.args.conv_hidden
        self.output_layer = self.make_output_layer(last_units)
        self.init_weight()

    def forward(self, sentence_ids, sentence_lengths):
        sentence_embed = self.embed(sentence_ids).transpose(1, 2)
        conv_out = self.convs(sentence_embed).max(dim=2)[0]
        multi_pre = self.output_layer(conv_out)
        pre = torch.sigmoid(self.co_occur_layer(multi_pre))
        multi_pro = torch.sigmoid(multi_pre)
        return multi_pro, pre


