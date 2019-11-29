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
import progressbar

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
from misc.utils import repackage_hidden, clip_gradient, adjust_learning_rate, \
                    decode_txt, sample_batch_neg, l2_norm
from onmt.inputters.inputter import build_dataset_iter, lazily_load_dataset, \
    _load_fields, _collect_report_features
import data_loader as dl
import misc.model as model

from misc.encoder_att_gate import _netE_att, _netW
import datetime
import opts
from onmt.model_builder import *
import onmt.inputters as inputters
from onmt.utils.misc import tile


# from onmt.train_single import *

additional_device0 = "cuda:0"
additional_device1 = "cuda:1"


parser = argparse.ArgumentParser()




# parser.add_argument('-input_train_h5', default='vi-train.bpe.h5', help='path to label, now hdf5 file')
# parser.add_argument('-input_valid_h5', default='tst2012.bpe.h5', help='path to label, now hdf5 file')
# parser.add_argument('-input_vocab', default='/home/xwshi/easymoses_workspace2/reinforcement_learning/nmt/vi-en-baseline/corpus/vi_bpe.vocab.pt', help='path to dataset, now json file')
# parser.add_argument('-model_path_D', default='model/test/epoch_21.pth', help='folder to output images and model checkpoints')
# parser.add_argument('-model_path_G', default='/home/xwshi/easymoses_workspace2/reinforcement_learning/nmt/vi-en-baseline/train/t2t_baseline_vi_step_30000.pt', help='folder to output images and model checkpoints')
# parser.add_argument('-data', default='/home/xwshi/easymoses_workspace2/reinforcement_learning/nmt/vi-en-baseline/corpus/vi_bpe', help='path to dataset, now json file')


parser.add_argument('-input_train_h5', default='./wmt/wmt14.bpe.shuffle.h5', help='path to label, now hdf5 file')
parser.add_argument('-input_valid_h5', default='./wmt/newstest2014.bpe.h5', help='path to label, now hdf5 file')
parser.add_argument('-input_vocab', default='/home/xwshi/easymoses_workspace2/reinforcement_learning/nmt/wmt_v2/corpus/wmt_bpe.vocab.pt', help='path to dataset, now json file')
parser.add_argument('-model_path_D', default='./model/wmt_gate/epoch_11.pth', help='folder to output images and model checkpoints')
parser.add_argument('-model_path_G', default='/home/xwshi/easymoses_workspace2/reinforcement_learning/nmt/wmt_v2/train/t2t_baseline_wmt_step_348000.pt', help='folder to output images and model checkpoints')
parser.add_argument('-data', default='/home/xwshi/easymoses_workspace2/reinforcement_learning/nmt/wmt_v2/corpus/wmt_bpe', help='path to dataset, now json file')




# parser.add_argument('-input_train_h5', default='./nhfx/nhfx.bpe.h5', help='path to label, now hdf5 file')
# parser.add_argument('-input_valid_h5', default='./nhfx/nist02.bpe.h5', help='path to label, now hdf5 file')
# parser.add_argument('-input_vocab', default='/home/xwshi/easymoses_workspace2/reinforcement_learning/nmt/nhfx_v2/corpus/nhfx_bpe.vocab.pt', help='path to dataset, now json file')
# parser.add_argument('-model_path_D', default='./model/nhfx_gate/epoch_20.pth', help='folder to output images and model checkpoints')
# parser.add_argument('-model_path_G', default='/home/xwshi/easymoses_workspace2/reinforcement_learning/nmt/nhfx_v2/train/t2t_baseline_nhfx_step_110000.pt', help='folder to output images and model checkpoints')
# parser.add_argument('-data', default='/home/xwshi/easymoses_workspace2/reinforcement_learning/nmt/nhfx_v2/corpus/nhfx_bpe', help='path to dataset, now json file')

parser.add_argument('-num_val', default=1000, help='number of image split out as validation set.')
parser.add_argument('-update_LM', action='store_true', help='whether train use the GAN loss.')
parser.add_argument('-update_G', action='store_true', help='whether train use the GAN loss.')


parser.add_argument('-negative_sample', type=int, default=20, help='folder to output images and model checkpoints')
parser.add_argument('-neg_batch_sample', type=int, default=30, help='folder to output images and model checkpoints')

parser.add_argument('-niter', type=int, default=100, help='number of epochs to train for')
parser.add_argument('-start_epoch', type=int, default=0, help='start of epochs to train for')

parser.add_argument('-workers', type=int, help='number of data loading workers', default=1)

parser.add_argument('-adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('-D_lr', type=float, default=5e-5, help='learning rate for, default=0.00005')
parser.add_argument('-G_lr', type=float, default=5e-5, help='learning rate for, default=0.00005')
parser.add_argument('-LM_lr', type=float, default=5e-5, help='learning rate for, default=0.00005')
parser.add_argument('-beta1', type=float, default=0.5, help='beta1 for adam. default=0.8')

parser.add_argument('-cuda'  , action='store_true', help='enables cuda')
parser.add_argument('-debug'  , action='store_true', help='debug')
parser.add_argument('-verbose'  , action='store_true', help='show the sampled caption')

parser.add_argument('-clip', type=float, default=5, help='gradient clipping')
parser.add_argument('-margin', type=float, default=2, help='number of epochs to train for')
parser.add_argument('-gumble_weight', type=float, default=0.5, help='folder to output images and model checkpoints')

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

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
# create new folder.

num_train = -1
if opt.debug == True:
    print("debug")
    opt.batch_size = 1
    num_train = 2
    print("debug batch_size", opt.batch_size)

####################################################################################
# Data Loader
####################################################################################
dataset = dl.load_data(input_h5=opt.input_train_h5,
                input_vocab=opt.input_vocab, negative_sample = opt.negative_sample, num_val = num_train, split_type = 'train')

dataset_val = dl.load_data(input_h5=opt.input_valid_h5,
                input_vocab=opt.input_vocab, negative_sample = opt.negative_sample, num_val = 100, split_type = 'valid')

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
tgt_stoi = dataset.tgt_stoi
vocab = torch.load(opt.input_vocab)
vocab = dict(vocab)
src_itos = vocab['src'].itos
tgt_itos = vocab['tgt'].itos

start_token = tgt_stoi[inputters.BOS_WORD]

print('src_vocab_size', src_vocab_size)
print('tgt_vocab_size', tgt_vocab_size)
print("src_stoi[\'<unk>\']", src_stoi['<unk>'])
print("src_stoi[\'<blank>\']", src_stoi['<blank>'])
# exit()


def bulid_nmt_model(opt):
    # opt = training_opt_postprocessing(opt)
    # init_logger(opt.log_file)
    # Load checkpoint if we resume from a previous training.
    if opt.model_path_G:
        logger.info('Loading checkpoint from %s' % opt.model_path_G)
        checkpoint = torch.load(opt.model_path_G, map_location=lambda storage, loc: storage)
        model_opt = checkpoint['opt']
    else:
        raise(AssertionError("no nmt model"))

    # Peek the fisrt dataset to determine the data_type.
    # (All datasets have the same data_type).
    first_dataset = next(lazily_load_dataset("train", opt))
    data_type = first_dataset.data_type

    # Load fields generated from preprocess phase.
    fields = _load_fields(first_dataset, data_type, opt, checkpoint)

    # Report src/tgt features.

    src_features, tgt_features = _collect_report_features(fields)
    for j, feat in enumerate(src_features):
        logger.info(' * src feature %d size = %d'
                    % (j, len(fields[feat].vocab)))
    for j, feat in enumerate(tgt_features):
        logger.info(' * tgt feature %d size = %d'
                    % (j, len(fields[feat].vocab)))

    # Build model.
    model = build_model(model_opt, opt, fields, checkpoint, additional_device0)
    # model = build_model(model_opt, opt, fields, checkpoint, "cpu")

    return model


def bulid_model_D(opt, src_vocab_size, tgt_vocab_size):
    save_path = opt.save_model
    print ("save_path", save_path)
    try:
        os.makedirs(save_path)
    except OSError:
        pass

    # if opt.train_from != '':
        # print("=> loading checkpoint '{}'".format(opt.train_from))
        # checkpoint = torch.load(opt.train_from)
        # model_path = opt.train_from
    if opt.model_path_D != '' :
        print("=> loading checkpoint '{}'".format(opt.model_path_D))
        checkpoint_D = torch.load(opt.model_path_D)
    else:
        model_path_D = save_path


    # if opt.model_path_G != '':
    #     print("=> loading checkpoint '{}'".format(opt.model_path_G))
    #     checkpoint_G = torch.load(opt.model_path_G)

    lr = opt.LM_lr
    print("rnn size", opt.rnn_size)
    src_netE_att = _netE_att(opt.rnn_size, src_vocab_size, opt.dropout)
    src_netW = _netW(src_vocab_size, opt.rnn_size, opt.dropout, name="src", cuda=opt.cuda)
    print("src_netW", src_vocab_size, src_netW.word_embed)
    tgt_netW = _netW(tgt_vocab_size, opt.rnn_size, opt.dropout, name="tgt", cuda=opt.cuda)
    print("tgt_netW", tgt_vocab_size, tgt_netW.word_embed)
    tgt_netE_att = _netE_att(opt.rnn_size, tgt_vocab_size, opt.dropout, is_target=True)

    critD = model.nPairLoss(opt.rnn_size, opt.margin)

    if opt.model_path_D != '' : # load the pre-trained model.
        src_netW.load_state_dict(checkpoint_D['src_netW'])
        tgt_netW.load_state_dict(checkpoint_D['tgt_netW'])
        src_netE_att.load_state_dict(checkpoint_D['src_netE_att'])
        tgt_netE_att.load_state_dict(checkpoint_D['tgt_netE_att'])
        lr = checkpoint_D['lr']

    if opt.cuda:
        tgt_netW.cuda()
        src_netW.cuda()
        src_netE_att.cuda()
        tgt_netE_att.cuda()
        critD.cuda()

    print('init Generative model...')

    sampler = model.gumbel_sampler()
    critG = model.G_loss(opt.rnn_size)
    critLM = model.LMCriterion()
    BLEU_score = model.BLEU_score()

    if opt.cuda: # ship to cuda, if has GPU
        sampler.cuda()
        critG.cuda()
        critLM.cuda()
        BLEU_score.cuda()

    print("load netD successfully")
    return tgt_netW, src_netW, src_netE_att, tgt_netE_att, critD, lr, sampler, critG, critLM, BLEU_score


# torch.cuda.set_device(0)

####################################################################################
# training model
####################################################################################


netG = bulid_nmt_model(opt)
print("load nmt model successfully")
# if opt.cuda:
    # netG.cuda()
    # netG.to(additional_device0)
    # print("netG.device", netG.device)
# exit()

tgt_netW, src_netW, src_netE_att, tgt_netE_att, critD, lr, sampler, critG, critLM, BLEU_score = bulid_model_D(opt, src_vocab_size, tgt_vocab_size)


def train(epoch):
    tgt_netW.train(), src_netW.train(), src_netE_att.train(), tgt_netE_att.train(), netG.train()
    save_path = opt.save_model
    # alpha.train()

    if opt.cuda:
        tgt_netW.cuda()
        src_netW.cuda()
        src_netE_att.cuda()
        tgt_netE_att.cuda()
        netG.cuda()

    fake_len = torch.LongTensor(opt.batch_size)

    fake_len = fake_len.cuda()

    n_neg = opt.negative_sample

    data_iter = iter(dataloader)

    err_d = 0
    err_g = 0
    err_lm = 0
    average_loss = 0
    count = 0
    i = 0
    loss_store = []
    t = time.time()

    while i < len(dataloader)-1:
        t1 = time.time()
        data = data_iter.next()
        # image, history, question, answer, answerT, answerLen, answerIdx, questionL, opt_answerT, opt_answerLen, opt_answerIdx = data
        questionLen, question, answer, answerT, answerLen, answerIdx, questionL, _, opt_answerT, opt_answerLen, opt_answerIdx, answer_ids = data

        # print("question", question,question.size())
        # print("questionL", questionL,questionL.size())
        # print("questionLen", questionLen, questionLen.size())
        # exit()
        batch_size = question.size(0)

        # image = image.view(-1, 512)
        # img_input.data.resize_(image.size()).copy_(image)

        err_d_tmp = 0.
        err_g_tmp = 0.
        err_lm_tmp = 0.
        err_g_fake_tmp = 0.
        err_d_fake_tmp = 0.

        ques = question[:,:].t()
        ques_g = questionL[:,:].t()

        ans = answer[:,:].t()
        if opt.debug == True:
            ques_words = [src_itos[int(w)] for w in ques_g]
            answer_words = [tgt_itos[int(w)] for w in answer[0]]
            print("train_all.py train() ques_g.size()", ques_g.size())
            print("train_all.py train() answer.size()", answer.size())
            print("train_all.py train() ques_g", ques_g, len(ques_g))
            print("train_all.py train() answer", answer, len(answer))
            print("train_all.py train() ques_words", ques_words, len(ques_words))
            print("train_all.py train() answer_words", answer_words, len(answer_words))
            # print("train_all.py ans", ans.size())
        tans = answerT[:,:].t()
        wrong_ans = opt_answerT[:,:].clone().view(-1, tgt_length).t()

        real_len = answerLen[:].long()
        wrong_len = opt_answerLen[:,:].clone().view(-1)

        ques_input.data.resize_(ques.size()).copy_(ques)
        src_input_g.data.resize_(ques.size()).copy_(ques_g)
        # print("train_all.py ques_input",ques_input, ques_input.size(), ques_input.device)

        ans_input.data.resize_(ans.size()).copy_(ans)
        ans_target.data.resize_(tans.size()).copy_(tans)
        wrong_ans_input.data.resize_(wrong_ans.size()).copy_(wrong_ans)


        batch_sample_idx.data.resize_(batch_size, opt.neg_batch_sample).zero_()
        sample_batch_neg(answerIdx[:], opt_answerIdx[:,:], batch_sample_idx, opt.neg_batch_sample)


        src = src_input_g.view(-1, batch_size, 1).clone().to(additional_device0)
        tgt = ans_input.view(-1, batch_size, 1).clone().to(additional_device0)
        src_length = questionLen.view(batch_size)

        # if opt.update_G:
        fake_onehot = []
        fake_idx = []
        noise_input.data.resize_(tgt_length, batch_size, tgt_vocab_size)
        noise_input.data.uniform_(0,1)
        # print("train_all.py train() noise_input", noise_input.size())

        ans_sample = ans_input[0]
        enc_final, memory_bank = netG.encoder(src, src_length)

        # dec_states = netG.decoder.init_decoder_state(src, memory_bank, enc_states)
        dec_state = None
        dec_state =  netG.decoder.init_decoder_state(src, memory_bank, enc_final)
        memory_bank = tile(memory_bank, 1, dim=1)
        memory_lengths = tile(src_length, 1)
        
        alive_seq = torch.full(
            [opt.batch_size, 1],
            start_token,
            dtype=torch.long,
            device=additional_device0)

        for step in range(tgt_length):

            decoder_input = alive_seq[:, -1].view(1, -1, 1)

            dec_out, dec_state, _ = netG.decoder(decoder_input,
                memory_bank,
                dec_state,
                memory_lengths=memory_lengths,
                step=step)

            logprob = netG.generator.forward(dec_out.squeeze(0))

            one_hot, idx = sampler(logprob, noise_input[step].to(additional_device0), opt.gumble_weight)
            fake_onehot.append(one_hot.view(1, -1, tgt_vocab_size))
            fake_idx.append(idx)
            if step+1 < tgt_length:
                ans_sample = idx

            alive_seq = torch.cat(
                (alive_seq, idx.view(-1,1)), -1)

        fake_onehot = torch.cat(fake_onehot, 0)

        fake_idx = torch.cat(fake_idx,0)
        fake_len = fake_len.resize_(batch_size).fill_(tgt_length-1).clone()
        for di in range(tgt_length-1, 0, -1):
            fake_len.masked_fill_(fake_idx.data[di].eq(tgt_vocab_size), di)

        fake_mask.data.resize_(fake_idx.size()).fill_(1)
        for b in range(batch_size):
            fake_mask.data[:fake_len[b]+1, b] = 0

        fake_idx.masked_fill_(fake_mask.clone(), 0)

        fake_onehot = fake_onehot.view(-1, tgt_vocab_size)
        ques_emb_d = src_netW(ques_input, format = 'index')

        featD, _ = src_netE_att(ques_emb_d, ques_input)
        ans_real_emb = tgt_netW(ans_target, format='index')
        ans_fake_emb = tgt_netW(fake_onehot.to('cuda'), format='onehot')
        ans_fake_emb = ans_fake_emb.view(tgt_length, -1, opt.rnn_size)

        fake_feat, _ = tgt_netE_att(ans_fake_emb.cuda(), fake_idx.cuda())
        real_feat, _ = tgt_netE_att(ans_real_emb.cuda(), ans_target)


        ##############################   update_D   ###########################################
        src_netW.zero_grad()
        tgt_netW.zero_grad()
        src_netE_att.zero_grad()
        tgt_netE_att.zero_grad()
        ans_wrong_emb = tgt_netW(wrong_ans_input, format='index')

        tgt_wrong_feat, _weight_ = tgt_netE_att(ans_wrong_emb, wrong_ans_input)

        batch_wrong_feat = tgt_wrong_feat.index_select(0, batch_sample_idx.view(-1))
        tgt_wrong_feat = tgt_wrong_feat.view(batch_size, -1, opt.rnn_size)
        batch_wrong_feat = batch_wrong_feat.view(batch_size, -1, opt.rnn_size)
        # print("featD",featD.size())
        # print("fake_feat",fake_feat.size())
        # batch_wrong_feat = fake_feat.index_select(0, batch_sample_idx.view(-1))
        # fake_feat = fake_feat.view(batch_size, -1, opt.rnn_size)
        # batch_wrong_feat = batch_wrong_feat.view(batch_size, -1, opt.rnn_size)
        nPairLoss, d_fake = critD(featD, real_feat, tgt_wrong_feat, batch_wrong_feat, fake_feat)
        # print("train_gan.py d_fake", d_fake, float(d_fake))
        nPairLoss.backward(retain_graph=True)
        optimizerD.step()
        err_d += nPairLoss.data.item()
        err_d_tmp += nPairLoss.data.item()
        err_d_fake_tmp += d_fake.data.item()
        ##############################   update_D   ###########################################



        ##############################   update_G  #############################################
        d_g_loss, g_fake = critG(featD, real_feat, fake_feat)
        bleu_score = BLEU_score(ans_target, fake_idx)
        netG.zero_grad()
        d_g_loss.backward()
        optimizerG.step()
        err_g += d_g_loss.item()
        err_g_tmp += d_g_loss.item()
        err_g_fake_tmp += g_fake
        ##############################   update_G  #############################################


        ##############################   update_LM   ###########################################
        outputs, _, _ = netG(src, tgt, src_length, None)  
        logprob = netG.generator(outputs.view(-1, outputs.size(2)))
        lm_loss = critLM(logprob, tgt[1:, :, :].view(-1, 1))
        lm_loss = lm_loss / float(torch.sum(1-ans_target[1:, :].data.eq(1)))
        lm_loss = lm_loss*bleu_score
        netG.zero_grad()
        lm_loss.backward()
        optimizerLM.step()
        err_lm += lm_loss.item()
        err_lm_tmp += lm_loss.item()
        ##############################   update_LM   ###########################################

        count += 1

        i += 1
        loss_store.append({'iter':i, 'err_lm':err_lm_tmp/10, 'err_g':err_g_tmp/10, 'g_fake':err_g_fake_tmp/10, 'err_d':err_d_tmp})

        if i % opt.report_every == 0:
            print ('Epoch:%d %d/%d, err_lm %4f, err_g %4f, g_fake %4f, err_d %4f, d_fake %4f, Time: %.3f' % (epoch, i, len(dataloader), err_lm_tmp/10, err_g_tmp/10, err_g_fake_tmp/10, err_d_tmp/10, err_d_fake_tmp/10, time.time()-t))
            t = time.time()

        if i % opt.save_checkpoint_steps == 0:
            print("Saving ... ")
            torch.save({'epoch': epoch,
                    'opt': opt,
                    'src_netW': src_netW.state_dict(),
                    'tgt_netW': tgt_netW.state_dict(),
                    'tgt_netE_att': tgt_netE_att.state_dict(),
                    'src_netE_att': src_netE_att.state_dict(),
                    'netG': netG.state_dict(),
                    'optimizerD': optimizerD,
                    'optimizerG': optimizerG,
                    'optimizerLM': optimizerLM,
                    },
                    '%s/epoch_%d_%d.pth' % (save_path, epoch, i))

        # if i >= 20:break
    err_g = err_g / count
    err_lm = err_lm / count
    err_d = err_d / count
    return err_lm, err_g, err_d, loss_store


####################################################################################
# Main
####################################################################################
alpha = torch.tensor(0.5)
ques_input = torch.LongTensor(src_length, opt.batch_size)
src_input_g = torch.LongTensor(src_length, opt.batch_size)
ans_input = torch.LongTensor(tgt_length, opt.batch_size)
ans_target = torch.LongTensor(tgt_length, opt.batch_size)
wrong_ans_input = torch.LongTensor(tgt_length, opt.batch_size)
sample_ans_input = torch.LongTensor(1, opt.batch_size)

fake_len = torch.LongTensor(opt.batch_size)
fake_diff_mask = torch.ByteTensor(opt.batch_size)
fake_mask = torch.ByteTensor(opt.batch_size)
# answer len
batch_sample_idx = torch.LongTensor(opt.batch_size)
# noise
noise_input = torch.FloatTensor(opt.batch_size)

if opt.cuda:
    ques_input = ques_input.cuda()
    src_input_g = src_input_g.cuda()
    ans_input, ans_target = ans_input.cuda(), ans_target.cuda()
    wrong_ans_input = wrong_ans_input.cuda()
    sample_ans_input = sample_ans_input.cuda()
    fake_len = fake_len.cuda()
    noise_input = noise_input.cuda()
    batch_sample_idx = batch_sample_idx.cuda()
    fake_diff_mask = fake_diff_mask.cuda()
    fake_mask = fake_mask.cuda()

optimizerD = optim.Adam([{'params': src_netW.parameters()},
                        {'params': tgt_netW.parameters()},
                        {'params': src_netE_att.parameters()},
                        {'params': tgt_netE_att.parameters()}], lr=opt.D_lr, betas=(opt.beta1, 0.999))

optimizerG = optim.Adam([{'params': netG.parameters()}], lr=opt.G_lr, betas=(opt.beta1, 0.999))
optimizerLM = optim.Adam([{'params': netG.parameters()}], lr=opt.LM_lr, betas=(opt.beta1, 0.999))

history = []
train_his = {}
for epoch in range(opt.start_epoch+1, opt.niter):
# for epoch in range(opt.start_epoch+1, 2):
    t = time.time()
    train_loss_lm, train_loss_g, train_loss_d, loss_store = train(epoch)
    print ('Epoch: %d LM loss %4f, Generator loss %4f, Discriminator loss %4f, Time: %3f' % (epoch, train_loss_lm, train_loss_g, train_loss_d, time.time()-t))
    train_his = {'lossLM': train_loss_lm, 'loss_G':train_loss_g, 'loss_D':train_loss_d, 'loss_store':loss_store}
    
    history.append({'epoch':epoch, 'train': train_his})
    # saving the model.
    save_path = opt.save_model

    torch.save({'epoch': epoch,
                'opt': opt,
                'src_netW': src_netW.state_dict(),
                'tgt_netW': tgt_netW.state_dict(),
                'tgt_netE_att': tgt_netE_att.state_dict(),
                'src_netE_att': src_netE_att.state_dict(),
                'netG': netG.state_dict(),
                'optimizerD': optimizerD,
                'optimizerG': optimizerG,
                'optimizerLM': optimizerLM,
                },
                '%s/epoch_%d.pth' % (save_path, epoch))

    json.dump(history, open('%s/log.json' %(save_path), 'w'))
