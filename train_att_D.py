from __future__ import print_function

import argparse
import os
import random
import sys
sys.path.append(os.getcwd())

# import pdb
import time
import numpy as np
import json
# import progressbar
import sys
# sys.path.append('../')
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
# import torchvision.transforms as transforms
# import torchvision.utils as vutils
from torch.autograd import Variable
import opts
from misc.utils import repackage_hidden, adjust_learning_rate, sample_batch_neg
# from misc.utils import repackage_hidden, clip_gradient, adjust_learning_rate, decode_txt, sample_batch_neg, l2_norm
# import misc.dataLoader as dl
import data_loader as dl
import misc.model as model
# from misc.encoder_att_4 import _netE_att, _netW
# from misc.encoder_att5 import _netE_att, _netW
# from misc.encoder_att_gate import _netE_att, _netW
from misc.encoder_att_gate_v2 import _netE_att, _netW

# from misc.encoder_att import _netE_att, _netW
# from misc.encoder_att import _src_netE_att2 as _src_netE_att
# from misc.att_netD import 
import datetime


parser = argparse.ArgumentParser(
        description='train_D.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)


# parser.add_argument('-input_train_h5', default='./wmt/wmt14.bpe.shuffle.h5', help='path to label, now hdf5 file')
# parser.add_argument('-input_valid_h5', default='./wmt/newstest2014.bpe.h5', help='path to label, now hdf5 file')
# parser.add_argument('-input_vocab', default='./wmt/wmt_bpe.vocab.pt', help='path to dataset, now json file')


parser.add_argument('-input_train_h5', default='./nhfx/nhfx.bpe.h5', help='path to label, now hdf5 file')
parser.add_argument('-input_valid_h5', default='./nhfx/nist02.bpe.h5', help='path to label, now hdf5 file')
parser.add_argument('-input_vocab', default='./nhfx/nhfx_bpe.vocab.pt', help='path to dataset, now json file')


# parser.add_argument('-input_train_h5', default='nhf.bpe.train.h5', help='path to label, now hdf5 file')
# parser.add_argument('-input_valid_h5', default='nhf.bpe.dev.h5', help='path to label, now hdf5 file')
# parser.add_argument('-input_train_h5', default='vi-train.bpe.h5', help='path to label, now hdf5 file')
# parser.add_argument('-input_valid_h5', default='tst2012.bpe.h5', help='path to label, now hdf5 file')
# parser.add_argument('-input_vocab', default='/home/xwshi/easymoses_workspace2/reinforcement_learning/nmt/vi-en-baseline/corpus/vi_bpe.vocab.pt', help='path to dataset, now json file')

# parser.add_argument('-input_vocab', default='nhf_bpe.vocab.pt', help='path to dataset, now json file')
parser.add_argument('-num_val', default=1000, help='number of image split out as validation set.')
parser.add_argument('-negative_sample', type=int, default=20, help='folder to output images and model checkpoints')
parser.add_argument('-neg_batch_sample', type=int, default=20, help='folder to output images and model checkpoints')
parser.add_argument('-lr', type=float, default=0.0004, help='learning rate for, default=0.00005')
# parser.add_argument('-lr', type=float, default=0.0001, help='learning rate for, default=0.00005')
parser.add_argument('-beta1', type=float, default=0.8, help='beta1 for adam. default=0.5')
parser.add_argument('-beta2', type=float, default=0.999, help='beta1 for adam. default=0.5')
parser.add_argument('-clip', type=float, default=5, help='gradient clipping')
parser.add_argument('-margin', type=float, default=2, help='number of epochs to train for')
parser.add_argument('-log_interval', type=int, default=100, help='how many iterations show the log info')
parser.add_argument('-workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('-niter', type=int, default=30, help='number of epochs to train for')
parser.add_argument('-save_iter', type=int, default=1, help='number of epochs to train for')
parser.add_argument('-cuda', action="store_true", help='')

opts.add_md_help_argument(parser)
opts.model_opts(parser)
opts.train_opts(parser)
opt = parser.parse_args()

opt.manualSeed = 1317

print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
np.random.seed(opt.manualSeed)

if opt.cuda:
    torch.cuda.manual_seed(opt.manualSeed)

cudnn.benchmark = True


####################################################################################
# Data Loader
####################################################################################

dataset = dl.load_data(input_h5=opt.input_train_h5,
                input_vocab=opt.input_vocab, negative_sample = opt.negative_sample, num_val = -1, split_type = 'train')

dataset_val = dl.load_data(input_h5=opt.input_valid_h5,
                input_vocab=opt.input_vocab, negative_sample = opt.negative_sample, num_val = 1000, split_type = 'test')

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                         shuffle=True, num_workers=int(opt.workers))

dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1,
                                         shuffle=False, num_workers=int(opt.workers))

####################################################################################
# Build the Model
####################################################################################
n_neg = opt.negative_sample
src_vocab_size = dataset.src_vocab_size
tgt_vocab_size = dataset.tgt_vocab_size
src_length = dataset.src_length
tgt_length = dataset.tgt_length + 1

src_stoi = dataset.src_stoi
print("save_model", opt.save_model)
print("src_stoi[\'<unk>\']", src_stoi['<unk>'])
print("src_stoi[\'<blank>\']", src_stoi['<blank>'])
print("opt.dropout", opt.dropout)


def bulid_model(opt, src_vocab_size, tgt_vocab_size):
    last_epoch = 0
    if opt.train_from != '':
        print("=> loading checkpoint '{}'".format(opt.train_from))
        checkpoint = torch.load(opt.train_from)
        # model_path = opt.train_from
    else:
        # create new folder.
        # t = datetime.datetime.now()
        # cur_time = '%s-%s-%s' %(t.day, t.month, t.hour)
        # cur_time = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
        # save_path = os.path.join(opt.save_model, cur_time)
        save_path = opt.save_model
        print ("save_path", save_path)
        try:
            os.makedirs(save_path)
        except OSError:
            pass
    lr = opt.lr
    print("rnn size", opt.rnn_size)
    src_netE_att = _netE_att(opt.rnn_size, src_vocab_size, opt.dropout)
    tgt_netE_att = _netE_att(opt.rnn_size, tgt_vocab_size, opt.dropout, is_target=True)

    src_netW = _netW(src_vocab_size, opt.rnn_size, opt.dropout, name="src")
    tgt_netW = _netW(tgt_vocab_size, opt.rnn_size, opt.dropout, name="tgt")
    print("src_netW", src_vocab_size, src_netW.word_embed)
    print("tgt_netW", tgt_vocab_size, tgt_netW.word_embed)

    critD = model.nPairLoss(opt.rnn_size, opt.margin)
    # exit()

    if opt.train_from != '' : # load the pre-trained model.
        src_netW.load_state_dict(checkpoint['src_netW'])
        tgt_netW.load_state_dict(checkpoint['tgt_netW'])
        src_netE_att.load_state_dict(checkpoint['src_netE_att'])
        tgt_netE_att.load_state_dict(checkpoint['tgt_netE_att'])
        last_epoch = checkpoint['epoch']
        lr = checkpoint['lr']

    if opt.cuda:
        tgt_netW.cuda(), src_netW.cuda(), src_netE_att.cuda(), tgt_netE_att.cuda(), critD.cuda()

    return tgt_netW, src_netW, src_netE_att, tgt_netE_att, critD, lr, last_epoch

tgt_netW, src_netW, src_netE_att, tgt_netE_att, critD, lr, last_epoch = bulid_model(opt, src_vocab_size, tgt_vocab_size)


####################################################################################
# training model
####################################################################################
def train(epoch, lr):
    print ("in train()")
    # exit()
    src_netW.train()
    tgt_netW.train()
    src_netE_att.train()
    tgt_netE_att.train()

    lr = adjust_learning_rate(optimizer, epoch, lr)
    # print ("in train() 2")


    data_iter = iter(dataloader)
    average_loss = 0
    count = 0
    # i = 0
    print ("train_D train()", len(dataloader))
    data_size = len(dataloader) 
    t = time.time()
    for i in range(data_size):
        # print ("i", i)
        t1 = time.time()
        data = data_iter.next()
        _, question, answer, answerT, answerLen, answerIdx, questionL, _, opt_answerT, opt_answerLen, opt_answerIdx, answer_ids = data
        # print("answerLen", answerLen)
        # print("answerT", answerT)
        # print("opt_answerT", opt_answerT)
        # print("question", question.size())
        batch_size = question.size(0)
        # print ("batch_size", batch_size)

        # rnd = 0
        src_netW.zero_grad()
        tgt_netW.zero_grad()
        src_netE_att.zero_grad()
        tgt_netE_att.zero_grad()
        ques = question[:,:].t()

        # ans = answer[:,rnd,:].t()
        tans = answerT[:,:].t()

        # print("train_att_D.py opt_answerT", opt_answerT.size())
        wrong_ans = opt_answerT[:,:].clone().view(-1, tgt_length).t()
        # print("train_att_D.py wrong_ans", wrong_ans.size())

        # real_len = answerLen[:]
        wrong_len = opt_answerLen[:,:].clone().view(-1)

        ques_input.data.resize_(ques.size()).copy_(ques)

        # ans_input.data.resize_(ans.size()).copy_(ans)
        ans_target.data.resize_(tans.size()).copy_(tans)
        # print("ans_target", ans_target.size(), ans_target)
        # print("ans_target", ans_target)
        # exit()
        # print("train_att_D.py wrong_ans_input", wrong_ans_input.size())

        wrong_ans_input.data.resize_(wrong_ans.size()).copy_(wrong_ans)
        # print("train_att_D.py wrong_ans_input", wrong_ans_input.size())

        # sample in-batch negative index
        batch_sample_idx.data.resize_(batch_size, opt.neg_batch_sample).zero_()
        sample_batch_neg(answerIdx[:], opt_answerIdx[:,:], batch_sample_idx, opt.neg_batch_sample)

        ques_emb = src_netW(ques_input, format = 'index')
        # print("train_att_D.py ques_emb", ques_emb.size(), ques_emb.device)
        src_featD, _ = src_netE_att(ques_emb, ques_input)
        # print("train_att_D.py src_featD", src_featD.size(), src_featD.device)

        ans_real_emb = tgt_netW(ans_target, format='index')
        ans_wrong_emb = tgt_netW(wrong_ans_input, format='index')

        tgt_real_feat, _weight_ = tgt_netE_att(ans_real_emb, ans_target)
        tgt_wrong_feat, _weight_ = tgt_netE_att(ans_wrong_emb, wrong_ans_input)
        # print("train_att_D.py train() tgt_wrong_feat _weight_", _weight_, _weight_.size())

        # tgt_wrong_feat, _ = tgt_netE_att(src_featD, ans_wrong_emb, wrong_ans_input, wrong_hidden, tgt_vocab_size)
        # exit()
        # print("tgt_wrong_feat", tgt_wrong_feat.size())
        # print("batch_sample_idx", batch_sample_idx.size())
        # print("batch_sample_idx.view(-1)", batch_sample_idx.view(-1).size())
        batch_wrong_feat = tgt_wrong_feat.index_select(0, batch_sample_idx.view(-1))
        tgt_wrong_feat = tgt_wrong_feat.view(batch_size, -1, opt.rnn_size)
        batch_wrong_feat = batch_wrong_feat.view(batch_size, -1, opt.rnn_size)
        # print("src_featD", src_featD.size())
        # print("tgt_real_feat", tgt_real_feat.size())
        # print("tgt_wrong_feat", tgt_wrong_feat.size())
        # print("batch_wrong_feat", batch_wrong_feat.size())
        # exit()
        nPairLoss = critD(src_featD, tgt_real_feat, tgt_wrong_feat, batch_wrong_feat)

        average_loss += nPairLoss.data.item()
        nPairLoss.backward()
        optimizer.step()
        count += 1

        # i += 1
        if i % opt.log_interval == 0:
            average_loss /= count
            print("step {} / {} (epoch {}), g_loss {:.3f}, lr = {:.6f} Time: {:.3f}".format(i, data_size, epoch, average_loss, lr, time.time()-t))
            average_loss = 0
            count = 0
            t = time.time()

        # if i > 10: break

    print ("finish train()")
    return average_loss, lr
    # return average_loss

def val():
    print ("in val()")
    src_netE_att.eval()
    src_netW.eval()
    tgt_netW.eval()
    tgt_netE_att.eval()

    n_neg = 100
    data_iter_val = iter(dataloader_val)
    i = 0

    average_loss = 0
    rank_all_tmp = []

    print ("val() ", len(dataloader_val))
    while i < len(dataloader_val):
        # print("val i", i)
        data = data_iter_val.next()
        # return src, tgt, tgt_target, tgt_len, tgt_idx, src_ori, opt_tgt, opt_tgt_target, opt_tgt_len

        questionLen, question, answer, answerT, answer_ids, questionOri, opt_answer, opt_answerT, opt_answerLen, opt_answerIdx = data
        # print("answerLen", answerLen)
        # print("answerT", answerT)
        # print("opt_answerT", opt_answerT)
        # exit()
        # print (questionLen, questionLen.size())
        # if questionLen[0] > 40:continue
        # if i == 50:
            # print (answer_ids)
            # exit()
        # question, answer, answerT, answerLen, answerIdx, questionOri, opt_answerT, opt_answerLen, opt_answerIdx = data
        # print("i", i)
        # print("question", question.size())
        # print("answer", answer.size())
        # # print("answerT", answerT.size())
        # print("answerLen", answerLen.size())
        # print("answer_ids", answer_ids, answer_ids.size())
        # print("questionOri", questionOri.size())
        # print("opt_answer", opt_answer.size())
        # print("opt_answerT", opt_answerT.size(), opt_answerT)
        # print("opt_answerLen", opt_answerLen.size())

        batch_size = question.size(0)

        # rnd = 0
        # ques = question[:,:].t()
        ques = question.t()

        opt_ans = opt_answerT.clone().view(-1, tgt_length).t()
        # opt_ans = opt_answer.clone().view(-1, tgt_length).t()
        # print("train_att_D.py opt_ans", opt_ans.size())
        # exit()
        gt_id = answer_ids[:,0]
        # print("gt_id", gt_id)
        # continue

        # print(i , "ques", ques.size())
        ques_input.data.resize_(ques.size()).copy_(ques)
        # print(i, "ques_input", ques_input.size())

        opt_ans_input.data.resize_(opt_ans.size()).copy_(opt_ans)
        gt_index.data.resize_(gt_id.size()).copy_(gt_id)
        # opt_len = opt_answerLen.clone().view(-1)

        ques_emb = src_netW(ques_input, format = 'index')
        # ques_emb = src_netW(ques_input, format = 'index')

        # ques_hidden = repackage_hidden(ques_hidden, batch_size)

        # src_featD, src_featDA = src_netE_att(ques_emb, ques_hidden)
        src_featD, _ = src_netE_att(ques_emb, ques_input)
        # print("train_att_D.py val() src_featD", src_featD.size())

        # opt_ans_emb = src_netW(opt_ans_input, format = 'index')
        opt_ans_emb = tgt_netW(opt_ans_input, format = 'index')
        # print("train_att_D.py val() opt_ans_emb", opt_ans_emb.size())
        # opt_hidden = repackage_hidden(opt_hidden, opt_ans_input.size(1))
        # expand_src_featD = src_featD.expand(opt_ans_emb.size(1), -1, -1).reshape(opt_ans_emb.size(1), -1)
        # print("train_att_D.py val() expand_src_featD", expand_src_featD.size())
        # opt_feat, _ = tgt_netE_att(expand_src_featD, opt_ans_emb, opt_ans_input, opt_hidden, tgt_vocab_size)
        opt_feat, _ = tgt_netE_att(opt_ans_emb, opt_ans_input)
        opt_feat = opt_feat.view(batch_size, -1, opt.rnn_size)
        # print("train_att_D.py opt_feat", opt_feat.size())
        src_featD = src_featD.view(-1, opt.rnn_size, 1)
        # print("train_att_D.py src_featD", src_featD.size())
        score = torch.bmm(opt_feat, src_featD)
        # print("score", score.size())

        score = score.view(-1, 100)
        # print("score", score.size())
        # exit()
        # print (batch_size)
        for b in range(batch_size):
            gt_index.data[b] = gt_index.data[b] + b*100
            # print ("gt_index", gt_index.data[b], gt_index.data)
        # print ("gt_index",gt_index)
        # print ("gt_index",gt_index.size())

        gt_score = score.view(-1).index_select(0, gt_index)
        # print ("gt_score",gt_score.size())
        sort_score, sort_idx = torch.sort(score, 1, descending=True)
        # print ("sort_score, sort_idx", sort_score, sort_idx)
        # print ("sort_score, sort_idx", sort_score.size(), sort_idx.size())

        count = sort_score.gt(gt_score.view(-1,1).expand_as(sort_score))
        rank = count.sum(1) + 1
        rank_all_tmp += list(rank.view(-1).data.cpu().numpy())
            
        i += 1
        sys.stdout.write('Evaluating: {:d}/{:d}  \r' .format(i, len(dataloader_val)))
        sys.stdout.flush()

    return rank_all_tmp

####################################################################################
# Main
####################################################################################
ques_input = torch.LongTensor(src_length, opt.batch_size)
ans_input = torch.LongTensor(tgt_length, opt.batch_size)
ans_target = torch.LongTensor(tgt_length, opt.batch_size)
wrong_ans_input = torch.LongTensor(tgt_length, opt.batch_size)
sample_ans_input = torch.LongTensor(1, opt.batch_size)
opt_ans_input = torch.LongTensor(tgt_length, opt.batch_size)

batch_sample_idx = torch.LongTensor(opt.batch_size)
# fake_diff_mask = torch.ByteTensor(opt.batch_size)
fake_len = torch.LongTensor(opt.batch_size)
noise_input = torch.FloatTensor(opt.batch_size)
gt_index = torch.LongTensor(opt.batch_size)

if opt.cuda:
# ques_input, his_input = ques_input.cuda(), his_input.cuda()
    ques_input = ques_input.cuda()
    ans_input, ans_target = ans_input.cuda(), ans_target.cuda()
    wrong_ans_input = wrong_ans_input.cuda()
    sample_ans_input = sample_ans_input.cuda()

    fake_len = fake_len.cuda()
    noise_input = noise_input.cuda()
    batch_sample_idx = batch_sample_idx.cuda()
    # fake_diff_mask = fake_diff_mask.cuda()
    opt_ans_input = opt_ans_input.cuda()
    gt_index = gt_index.cuda()

ques_input = Variable(ques_input)
ans_input = Variable(ans_input)
ans_target = Variable(ans_target)
wrong_ans_input = Variable(wrong_ans_input)
sample_ans_input = Variable(sample_ans_input)

noise_input = Variable(noise_input)
batch_sample_idx = Variable(batch_sample_idx)
# fake_diff_mask = Variable(fake_diff_mask)
opt_ans_input = Variable(opt_ans_input)
gt_index = Variable(gt_index)

optimizer = optim.Adam([{'params': src_netW.parameters()},
                        {'params': tgt_netW.parameters()},
                        {'params': src_netE_att.parameters()},
                        {'params': tgt_netE_att.parameters()}], lr=opt.lr, betas=(opt.beta1, opt.beta2))

history = []

for epoch in range(last_epoch+1, opt.niter):
    t = time.time()
    train_loss, lr = train(epoch, lr)
    print ('Epoch: %d learningRate %4f train loss %4f Time: %3f' % (epoch, lr, train_loss, time.time()-t))
    train_his = {'loss': train_loss}
    print('Evaluating ... ')
    rank_all = val()
    print ("")
    # print(rank_all)
    # exit()
    R1 = np.sum(np.array(rank_all)==1) / float(len(rank_all))
    R5 =  np.sum(np.array(rank_all)<=5) / float(len(rank_all))
    R10 = np.sum(np.array(rank_all)<=10) / float(len(rank_all))
    ave = np.sum(np.array(rank_all)) / float(len(rank_all))
    mrr = np.sum(1/(np.array(rank_all, dtype='float'))) / float(len(rank_all))
    print ('%d/%d: mrr: %f R1: %f R5 %f R10 %f Mean %f' %(epoch, len(dataloader_val), mrr, R1, R5, R10, ave))
    val_his = {'R1': R1, 'R5':R5, 'R10': R10, 'Mean':ave, 'mrr':mrr}
    history.append({'epoch':epoch, 'train': train_his, 'val': val_his})

    # saving the model.
    if epoch % opt.save_iter == 0:
        print ("save!!")
        torch.save({'epoch': epoch,
                    'opt': opt,
                    'src_netW': src_netW.state_dict(),
                    'tgt_netW': tgt_netW.state_dict(),
                    'tgt_netE_att': tgt_netE_att.state_dict(),
                    'src_netE_att': src_netE_att.state_dict(),
                    'lr':lr},
                    '%s/epoch_%d.pth' % (opt.save_model, epoch))

        json.dump(history, open('%s/log.json' %(opt.save_model), 'w'))
