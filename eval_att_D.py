from __future__ import print_function
import argparse
import os
import random
import sys
sys.path.append(os.getcwd())

import pdb
import time
import numpy as np
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

from misc.utils import repackage_hidden, clip_gradient, adjust_learning_rate, decode_txt
import data_loader as dl
import misc.model as model
# from misc.encoder_att_gate import _netE_att, _netW
from misc.encoder_att_gate_v2 import _netE_att, _netW
import datetime
import h5py
import opts

parser = argparse.ArgumentParser(
        description='eval_D.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# parser.add_argument('-input_valid_h5', default='./wmt/newstest2013.bpe.h5', help='path to label, now hdf5 file')
# parser.add_argument('-input_json', default='./wmt/newstest2013.bpe.json', help='path to label, now hdf5 file')

# parser.add_argument('-input_valid_h5', default='./nmpt/nmpt.bpe.train.h5', help='path to label, now hdf5 file')
# parser.add_argument('-input_json', default='./nmpt/nmpt.bpe.train.json', help='path to label, now hdf5 file')
# parser.add_argument('-input_valid_h5', default='./nmpt/nmpt.bpe.dev.h5', help='path to label, now hdf5 file')
# parser.add_argument('-input_json', default='./nmpt/nmpt.bpe.dev.json', help='path to label, now hdf5 file')
# parser.add_argument('-input_valid_h5', default='nmpt.bpe.test.h5', help='path to label, now hdf5 file')
#parser.add_argument('-input_json', default='nmpt.bpe.test.json', help='path to label, now hdf5 file')

# parser.add_argument('-input_valid_h5', default='./vi/tst2012.bpe.h5', help='path to label, now hdf5 file')
# parser.add_argument('-input_json', default='./vi/tst2012.bpe.json', help='path to label, now hdf5 file')
# parser.add_argument('-input_valid_h5', default='vi-train.bpe.h5', help='path to label, now hdf5 file')
# parser.add_argument('-input_json', default='train.bpe.clean.json', help='path to label, now hdf5 file')
parser.add_argument('-input_valid_h5', default='./nhfx/nist08.bpe.h5', help='path to label, now hdf5 file')
parser.add_argument('-input_json', default='./nhfx/nist08.bpe.json', help='path to label, now hdf5 file')
# parser.add_argument('-input_valid_h5', default='./nhfx/champollion-eval.bpe.h5', help='path to label, now hdf5 file')
# parser.add_argument('-input_json', default='./nhfx/champollion-eval.bpe.json', help='path to label, now hdf5 file')
# parser.add_argument('-input_valid_h5', default='./NIST/nist06.bpe.h5', help='path to label, now hdf5 file')
# parser.add_argument('-input_json', default='./NIST/nist06.bpe.json', help='path to label, now hdf5 file')
parser.add_argument('-cuda', action="store_true", help='')
parser.add_argument('-threshold', type=float, default=-1)
parser.add_argument('-options_num', default=100, type=int, help='Max length of sources')
# parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)

opts.add_md_help_argument(parser)
opts.model_opts(parser)
opts.translate_opts(parser)

opt = parser.parse_args()
outfilename = opt.output
# print (outfilename)
input_json = opt.input_json
srcfilename = opt.src
attn_debug = opt.attn_debug
options_num = opt.options_num
# print (srcfilename)
# print ("opt.num_val", opt.num_val)
# opt.manualSeed = 1317

# torch.cuda.manual_seed(opt.manualSeed)
# print("Random Seed: ", opt.manualSeed)
# random.seed(opt.manualSeed)
# torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

input_h5 = opt.input_valid_h5
model_path = opt.model

checkpoint = torch.load(opt.model)
opt = checkpoint['opt']
# print(opt)
# for k in checkpoint:
    # print(k)
# print ("checkpoint",checkpoint)



####################################################################################
# Data Loader
####################################################################################

dataset_val = dl.load_data(input_h5=input_h5,
                # input_vocab=opt.input_vocab, num_val=1000, split_type = 'valid')
                input_vocab=opt.input_vocab, num_val=-1, split_type = 'valid')

# print (len(dataset_val))
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1,shuffle=False, num_workers=int(opt.workers))
####################################################################################
# Build the Model
####################################################################################
print("opt.input_vocab", opt.input_vocab)
vocab = torch.load(opt.input_vocab)
vocab = dict(vocab)
src_stoi = vocab['src'].stoi
print("src_stoi", len(src_stoi))
tgt_stoi = vocab['tgt'].stoi
print("tgt_stoi", len(tgt_stoi))
src_itos = vocab['src'].itos
tgt_itos = vocab['tgt'].itos

src_length = dataset_val.src_length
tgt_length = dataset_val.tgt_length + 1
src_vocab_size = dataset_val.src_vocab_size
tgt_vocab_size = dataset_val.tgt_vocab_size

src_netE_att = _netE_att(opt.rnn_size, src_vocab_size, opt.dropout)
src_netW = _netW(src_vocab_size, opt.rnn_size, opt.dropout)
tgt_netW = _netW(tgt_vocab_size, opt.rnn_size, opt.dropout)
tgt_netE_att = _netE_att(opt.rnn_size, tgt_vocab_size, opt.dropout)

if model_path != '': # load the pre-trained model.
    src_netW.load_state_dict(checkpoint['src_netW'])
    tgt_netW.load_state_dict(checkpoint['tgt_netW'])
    src_netE_att.load_state_dict(checkpoint['src_netE_att'])
    tgt_netE_att.load_state_dict(checkpoint['tgt_netE_att'])
    print('Loading model Success!')


if opt.cuda:
    src_netW.cuda(), tgt_netW.cuda(), src_netE_att.cuda(), tgt_netE_att.cuda()

####################################################################################
# Some Functions
####################################################################################

rank_all = []

answer_index_list = []

score_gap = []

def eval():
    src_netW.eval()
    tgt_netW.eval()
    src_netE_att.eval()
    tgt_netE_att.eval()

    data_iter_val = iter(dataloader_val)
    i = 0
    display_count = 0
    rank_all_tmp = []
    result_all = []
    # img_atten = torch.FloatTensor(100 * 30, 10, 7, 7)
    print("len(dataloader_val)", len(dataloader_val))
    while i < len(dataloader_val):#len(1000):
    # while i < 120: #for test

        data = data_iter_val.next()
        # if i != 445: 
        #     i+= 1
        #     continue
        # image, history, question, answer, answerT, questionL, opt_answer, opt_answerT, answer_ids, answerLen, opt_answerLen, img_id  = data

        # question, answer, answerT, questionL, opt_answer, opt_answerT, answer_ids, answerLen, opt_answerLen  = data
        _, question, answer, answerLen, answer_ids, questionL, opt_answer, opt_answerT, opt_answerLen, opt_answerIdx = data
        batch_size = question.size(0)

        # for rnd in range(10):
        rnd=0
        # get the corresponding round QA and history.
        ques = question[:,:].t()
        ques_word = [src_itos[int(w)] for w in ques]
        ans_word = [tgt_itos[int(w)] for w in answer[0]]


        # his = history[:,:rnd+1,:].clone().view(-1, his_length).t()

        opt_ans = opt_answerT[:,:].clone().view(-1, tgt_length).t()
        # opt_ans = opt_answer[:,:].clone().view(-1, tgt_length).t()
        # print ("opt_answerIdx", opt_answerIdx.size(), opt_answerIdx)
        # exit()
        gt_id = answer_ids[:,0]

        # print("gt_id", gt_id.size(), gt_id)

        ques_input.data.resize_(ques.size()).copy_(ques)
        # his_input.data.resize_(his.size()).copy_(his)
        gt_index.data.resize_(gt_id.size()).copy_(gt_id)
        opt_ans_input.data.resize_(opt_ans.size()).copy_(opt_ans)

        ques_emb = src_netW(ques_input, format = 'index')

        featD, src_weight = src_netE_att(ques_emb, ques_input)

        opt_ans_emb = tgt_netW(opt_ans_input, format = 'index')
        opt_feat, tgt_weight = tgt_netE_att(opt_ans_emb, opt_ans_input)
        # print("eval_att_D.py eval() featD", featD.size())
        # opt_feat, tgt_weight = tgt_netE_att(featD, opt_ans_emb, opt_ans_input, opt_hidden, tgt_vocab_size)
        # exit()
        if attn_debug:
            print("ques", ques_word)
            print("src_weight", src_weight, src_weight.size())
            src_len = int(sum(src_weight.cpu().data.gt(0)[0][0]))
            print("src_length", src_len)
            src_weight = src_weight.cpu().data.numpy()[0][0][-src_len:]
            src_min = min(src_weight) - 0.001
            src_max_min = max(src_weight) - src_min + 0.001
            for i in range(len(src_weight)):
                src_weight[i] = (src_weight[i]-src_min) / src_max_min
            print("src_weight", list(src_weight))

            print("ans_word", ans_word)
            print("tgt_weight", tgt_weight[int(answer_ids)])
            tgt_len = int(sum(tgt_weight[int(answer_ids)].cpu().data.gt(0)[0]))
            print("tgt_length", tgt_len)
            tgt_weight = tgt_weight[int(answer_ids)].cpu().data.numpy()[0][:tgt_len]
            tgt_min = min(tgt_weight) - 0.001
            tgt_max_min = max(tgt_weight) - tgt_min + 0.001
            for i in range(len(tgt_weight)):
                tgt_weight[i] = (tgt_weight[i]-tgt_min) / tgt_max_min
            print("tgt_weight", list(tgt_weight))
            exit()

        opt_feat = opt_feat.view(batch_size, -1, opt.rnn_size)

        featD = featD.view(-1, opt.rnn_size, 1)
        score = torch.bmm(opt_feat, featD)
        # print(score)

        score = score.view(-1, options_num)
        # print(score.size())
        # score = score[:,:4]
        # print(score.size())
        # exit()

        for b in range(batch_size):
            gt_index.data[b] = gt_index.data[b] + b*options_num

        gt_score = score.view(-1).index_select(0, gt_index)
        sort_score, sort_idx = torch.sort(score, 1, descending=True)

        for inner_sort_idx in sort_idx:
            score_gap.append(float(sort_score[0,0]-sort_score[0,1]))

            if sort_score[0,0] > sort_score[0,1]:
                answer_index_list.append(int(opt_answerIdx[0, int(inner_sort_idx[0])]))
            else:
                answer_index_list.append(-1)
            # print(answer_index_list[-1])
        # print ("answer_index_list",answer_index_list)
        # if i > 5:
            # exit()

        count = sort_score.gt(gt_score.view(-1,1).expand_as(sort_score))
        # print ("count", count, count.size())
        # break

        rank = count.sum(1) + 1
        # print ("rank", rank)

        rank_all_tmp += list(rank.view(-1).data.cpu().numpy())
        # print (list(rank.view(-1).data.cpu().numpy()))
        # exit()

        i += 1

        if i % 10 == 0:
            sys.stdout.write("\r")
            sys.stdout.write("%.2f   %d" % (float(i)*100/len(dataloader_val), i))
            sys.stdout.flush()
            # R1 = np.sum(np.array(rank_all_tmp)==1) / float(len(rank_all_tmp))
            # R5 =  np.sum(np.array(rank_all_tmp)<=5) / float(len(rank_all_tmp))
            # R10 = np.sum(np.array(rank_all_tmp)<=10) / float(len(rank_all_tmp))
            # ave = np.sum(np.array(rank_all_tmp)) / float(len(rank_all_tmp))
            # mrr = np.sum(1/(np.array(rank_all_tmp, dtype='float'))) / float(len(rank_all_tmp))
            # print ('%d/%d: mrr: %f R1: %f R5 %f R10 %f Mean %f' %(i, len(dataloader_val), mrr, R1, R5, R10, ave))
        # if i > 6:break
    print("")
    return rank_all_tmp
    # return img_atten

####################################################################################
# Main
####################################################################################
# img_input = torch.FloatTensor(opt.batch_size)
ques_input = torch.LongTensor(src_length, opt.batch_size)
# his_input = torch.LongTensor(his_length, opt.batch_size)

# answer input
opt_ans_input = torch.LongTensor(tgt_length, opt.batch_size)
fake_ans_input = torch.FloatTensor(src_length, opt.batch_size, src_vocab_size)
sample_ans_input = torch.LongTensor(1, opt.batch_size)

# answer index location.
opt_index = torch.LongTensor(opt.batch_size)
fake_index = torch.LongTensor(opt.batch_size)

batch_sample_idx = torch.LongTensor(opt.batch_size)
# answer len
fake_len = torch.LongTensor(opt.batch_size)

# noise
noise_input = torch.FloatTensor(opt.batch_size)
gt_index = torch.LongTensor(opt.batch_size)

# ques_input, his_input, img_input = ques_input.cuda(), his_input.cuda(), img_input.cuda()
if opt.cuda:
    ques_input = ques_input.cuda()
    opt_ans_input = opt_ans_input.cuda()
    fake_ans_input, sample_ans_input = fake_ans_input.cuda(), sample_ans_input.cuda()
    opt_index, fake_index =  opt_index.cuda(), fake_index.cuda()

    fake_len = fake_len.cuda()
    noise_input = noise_input.cuda()
    batch_sample_idx = batch_sample_idx.cuda()
    gt_index = gt_index.cuda()


ques_input = Variable(ques_input)

opt_ans_input = Variable(opt_ans_input)
fake_ans_input = Variable(fake_ans_input)
sample_ans_input = Variable(sample_ans_input)

opt_index = Variable(opt_index)
fake_index = Variable(fake_index)

fake_len = Variable(fake_len)
noise_input = Variable(noise_input)
batch_sample_idx = Variable(batch_sample_idx)
gt_index = Variable(gt_index)

# atten = eval()
rank_all = eval()

print ("answer_index_list", len(answer_index_list))
# print ("answer_index_list", answer_index_list[0])

src_data = json.load(open(input_json,'r'))

targets = src_data['data']['targets']
# paras_list = src_data['data']['paras']
# print("paras_list", len(paras_list))
# print("paras_list", paras_list[0])
with open(outfilename, 'w') as outfile:
    for i, idx in enumerate(answer_index_list):
        # print (dialogs_list[i])
        # exit()
        # answer_idx = paras_list[i]['para']['ref_index'][idx]
        if idx < 0:
            outfile.write("\n")
        else:
            outfile.write(targets[idx].strip()+'\n')

with open(outfilename+'.score', 'w') as outfile:
    for score_diff in score_gap:
        outfile.write(str(score_diff)+'\n')

# print("rank_all", rank_all)
print("np.sum(np.array(rank_all)==1) ", np.sum(np.array(rank_all)==1) )
print("np.sum(np.array(rank_all)<=5) ", np.sum(np.array(rank_all)<=5) )
R1 = np.sum(np.array(rank_all)==1) / float(len(rank_all))
R5 =  np.sum(np.array(rank_all)<=5) / float(len(rank_all))
R10 = np.sum(np.array(rank_all)<=10) / float(len(rank_all))
ave = np.sum(np.array(rank_all)) / float(len(rank_all))
mrr = np.sum(1/(np.array(rank_all, dtype='float'))) / float(len(rank_all))
# print ("rank_all")
# print (rank_all)
print ('%d: mrr: %f R1: %f R5 %f R10 %f Mean %f' %(len(dataloader_val), mrr, R1, R5, R10, ave))
