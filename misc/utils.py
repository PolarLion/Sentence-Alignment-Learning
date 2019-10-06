import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from torch.autograd import 
import random
import pdb
"""
Some utility Functions.
"""

def repackage_hidden_volatile(h):
    if type(h) == Variable:
        return Variable(h.data, volatile=True)
    else:
        return tuple(repackage_hidden_volatile(v) for v in h)

def repackage_hidden(h, batch_size):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == torch.Tensor:
        # print ("in misc/utils repackage_hidden() %s true" % type(h))
        return Variable(h.data.resize_(h.size(0), batch_size, h.size(2)).zero_())
        # return h.data.resize_(h.size(0), batch_size, h.size(2)).zero_()
    else:
        # print ("in misc/utils repackage_hidden() false", type(h))
        # if type(h) != type(()):
            # print (h.size())
        # print ("in repackage_hidden type(h)", type(h))
        # exit()
        return tuple(repackage_hidden(v, batch_size) for v in h)

def clip_gradient(model):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        p.grad.data.clamp_(-5, 5)

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
    lr = lr * (0.5 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def decode_txt(itow, x):
    """Function to show decode the text."""
    out = []
    for b in range(x.size(1)):
        txt = ''
        for t in range(x.size(0)):
            idx = x[t,b]
            if idx == 0 or idx == len(itow)+1:
                break
            txt += itow[str(int(idx))]
            txt += ' '
        out.append(txt)

    return out

def l2_norm(input):
    """
    input: feature that need to normalize.
    output: normalziaed feature.
    """
    input_size = input.size()
    buffer = torch.pow(input, 2)

    normp = torch.sum(buffer, 1).add_(1e-10)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)

    return output


def sample_batch_neg(answerIdx, negAnswerIdx, sample_idx, num_sample):
    """
    input:
    answerIdx: batch_size
    negAnswerIdx: batch_size x opt.negative_sample

    output:
    sample_idx = batch_size x num_sample
    """

    batch_size = answerIdx.size(0)
    num_neg = negAnswerIdx.size(0) * negAnswerIdx.size(1)
    negAnswerIdx = negAnswerIdx.clone().view(-1)
    for b in range(batch_size):
        gt_idx = answerIdx[b]
        for n in range(num_sample):
            while True:
                rand = int(random.random() * num_neg)
                neg_idx = negAnswerIdx[rand]
                if gt_idx != neg_idx:
                    sample_idx.data[b, n] = rand
                    break
