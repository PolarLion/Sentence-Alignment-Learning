#-*- coding:utf-8 –*-
import torch
import torch.nn as nn
from torch.autograd import Variable
# import pdb
# import math
import numpy as np
import torch.nn.functional as F
from misc.share_Linear import share_Linear


class _netW(nn.Module):
    def __init__(self, ntoken, ninp, dropout, name="", cuda=False):
        super(_netW, self).__init__()
        if cuda:
            self.word_embed = nn.Embedding(ntoken, ninp, padding_idx=0).cuda()
            self.Linear = share_Linear(self.word_embed.weight).cuda()
        else:
            self.word_embed = nn.Embedding(ntoken, ninp, padding_idx=0).cpu()
            self.Linear = share_Linear(self.word_embed.weight).cpu()

        self.init_weights()
        self.d = dropout
        self.name = name
        # print("_netW self.training", self.training)

    def init_weights(self):
        initrange = 0.1
        self.word_embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, format ='index'):
        if format == 'onehot':
            out = F.dropout(self.Linear(input), self.d, training=self.training)
        elif format == 'index':
            # print("_netW forward() self.training", self.training)
            # print("_netW forward() self.name", self.name)
            out = F.dropout(self.word_embed(input), self.d, training=self.training)

        return out


class _netE_att(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, nhid, vocab_size, dropout, is_target=False):
        super(_netE_att, self).__init__()

        self.d = dropout
        self.nhid = nhid
        self.is_target = is_target

        self.W1 = nn.Linear(self.nhid, self.nhid)
        self.W2 = nn.Linear(self.nhid, 1)
        self.GW1 = nn.Linear(self.nhid, self.nhid)
        self.GW2 = nn.Linear(self.nhid, self.nhid)
        self.fc = nn.Linear(self.nhid, self.nhid)
        self.ln = nn.LayerNorm(self.nhid)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, ques_emb, idx):
        # output = ques_emb
        mask = idx.data.eq(1)  # generate the mask

        if self.is_target == "tgt":
            mask[idx.data == 3] = 1 # also set the last token to be 1
        if isinstance(ques_emb, Variable):
            mask = Variable(mask)
        # print("encoder_att.py mask", mask, mask.size())
        # print("encoder_att.py idx", idx, idx.size())
        # exit()
        # Doing self attention here.
        # print("encoder_att_gate.py ques_emb", ques_emb.size())
        atten_input_feat = (self.GW1(ques_emb)).mul(F.tanh(self.GW2(ques_emb)))
        # print("encoder_att_gate.py GW1", self.GW1)
        # print("encoder_att_gate.py GW2", self.GW2)
        # print("encoder_att_gate.py ques_emb", ques_emb.size())
        # print("encoder_att_gate.py atten_input_feat", atten_input_feat.size())
        atten_input_feat = self.W1(atten_input_feat)
        atten_input_feat = F.tanh(atten_input_feat)
        atten = self.W2(atten_input_feat.view(-1, self.nhid)).view(idx.size())
        # print("encoder_att_gate.py atten", atten.size())

        atten = self.dropout1(atten)
        # atten_input_feat = self.W2(atten_input_feat)
        # print("encoder_sdp_att.py atten_input_feat", atten_input_feat.size())
        # atten = self.W2(F.dropout(atten_input_feat.view(-1, self.nhid), self.d, training=self.training)).view(idx.size())
        # atten = atten_input_feat.view(-1, self.nhid).view(idx.size())
        atten.masked_fill_(mask, -999999)
        weight = F.softmax(atten.t(), dim=1).view(-1,1,idx.size(0))
        # print("encoder_att.py weight", weight.size())

        # exit()

        feat = torch.bmm(weight, ques_emb.transpose(0,1)).view(-1,self.nhid)
        # feat = F.dropout(feat, self.d, training=self.training)
        feat = self.dropout2(feat)
        transform_output = self.fc(feat)
        # transform_output = feat
        # transform_output = F.dropout(transform_output, self.d, training=self.training)
        # transform_output = self.dropout2(transform_output)
        transform_output = self.ln(transform_output)
        # print("encoder_att_4.py transform_output", transform_output.size())
        # print("encoder_att_4.py weight", weight.size())
        # exit()

        return transform_output, weight
