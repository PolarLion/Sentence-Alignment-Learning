import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import numpy as np
import torch.nn.functional as F
import nltk
from collections import defaultdict
from math import exp
from functools import reduce
from operator import mul

class LMCriterion(nn.Module):

    def __init__(self):
        super(LMCriterion, self).__init__()

    # def forward(self, input, target):
    def forward(self, input, target):
        logprob_select = torch.gather(input, 1, target)
        # print("model.py LMCriterion() logprob_select", logprob_select, logprob_select.size())

        mask = target.data.gt(1)  # generate the mask
        if isinstance(input, Variable):
            # mask = Variable(mask, volatile=input.volatile)
            mask = Variable(mask)
        
        # print("model.py LMCriterion() mask", mask, mask.size())
        # print("model.py LMCriterion() target", target, target.size())
        out = torch.masked_select(logprob_select, mask)
        # print("model.py LMCriterion() out", out, out.size())
        
        # exit()

        loss = -torch.sum(out) # get the average loss.
        return loss


class mixture_of_softmaxes(torch.nn.Module):
    """
    Breaking the Softmax Bottleneck: A High-Rank RNN Language Model (ICLR 2018)    
    """
    def __init__(self, nhid, n_experts, ntoken):

        super(mixture_of_softmaxes, self).__init__()
        
        self.nhid=nhid
        self.ntoken=ntoken
        self.n_experts=n_experts
        
        self.prior = nn.Linear(nhid, n_experts, bias=False)
        self.latent = nn.Sequential(nn.Linear(nhid, n_experts*nhid), nn.Tanh())
        self.decoder = nn.Linear(nhid, ntoken)
   
    def forward(self, x):
        
        latent = self.latent(x)
        logit = self.decoder(latent.view(-1, self.nhid))

        prior_logit = self.prior(x).view(-1, self.n_experts)
        prior = nn.functional.softmax(prior_logit)

        prob = nn.functional.softmax(logit.view(-1, self.ntoken)).view(-1, self.n_experts, self.ntoken)
        prob = (prob * prior.unsqueeze(2).expand_as(prob)).sum(1)
        
        return prob


class nPairLoss(nn.Module):
    """
    Given the tgt_right, fake, tgt_wrong, wrong_sampled embedding, use the N Pair Loss
    objective (which is an extension to the triplet loss)

    Loss = log(1+exp(src_feat*tgt_wrong - src_feat*tgt_right + src_feat*fake - src_feat*tgt_right)) + L2 norm.

    Improved Deep Metric Learning with Multi-class N-pair Loss Objective (NIPS)
    """
    def __init__(self, ninp, margin):
        super(nPairLoss, self).__init__()
        self.ninp = ninp
        self.margin = np.log(margin)

    def forward(self, src_feat, tgt_right, tgt_wrong, batch_wrong, fake=None, fake_diff_mask=None):

        num_wrong = tgt_wrong.size(1)
        batch_size = src_feat.size(0)

        src_feat = src_feat.view(-1, self.ninp, 1)
        right_dis = torch.bmm(tgt_right.view(-1, 1, self.ninp), src_feat)
        wrong_dis = torch.bmm(tgt_wrong, src_feat)
        batch_wrong_dis = torch.bmm(batch_wrong, src_feat)

        wrong_score = torch.sum(torch.exp(wrong_dis - right_dis.expand_as(wrong_dis)),1) \
                + torch.sum(torch.exp(batch_wrong_dis - right_dis.expand_as(batch_wrong_dis)),1)
        # """???"""
        # wrong_score = torch.sum(torch.exp(wrong_dis - right_dis.expand_as(wrong_dis)),1)

        loss_dis = torch.sum(torch.log(wrong_score + 1))
        loss_norm = tgt_right.norm() + src_feat.norm() + tgt_wrong.norm() + batch_wrong.norm()

        if fake is not None:
            fake_dis = torch.bmm(fake.view(-1, 1, self.ninp), src_feat)
            # print("model.py nPairLoss fake", fake.size())
            # exit()
            # fake_score = torch.masked_select(torch.exp(fake_dis - right_dis), fake_diff_mask)
            fake_score = torch.exp(fake_dis - right_dis)

            margin_score = F.relu(torch.log(fake_score + 1) - self.margin)
            loss_fake = torch.sum(margin_score)
            # print("model.py loss_fake", loss_fake)
            loss_dis += loss_fake
            loss_norm += fake.norm()

        loss = (loss_dis + 0.1 * loss_norm) / batch_size
        if fake is not None:
            # print("model.py loss %f, loss_fake %f" %(loss, loss_fake.data / batch_size))
            return loss, loss_fake.data / batch_size
        else:
            return loss


class G_loss(nn.Module):
    """
    Generator loss:
    minimize tgt_right feature and fake feature L2 norm.
    maximinze the fake feature and tgt_wrong feature.
    """
    def __init__(self, ninp):
        # raise
        super(G_loss, self).__init__()
        self.ninp = ninp

    def forward(self, src_feat, tgt_right, fake):

        #num_wrong = tgt_wrong.size(1)
        batch_size = src_feat.size(0)

        src_feat = src_feat.view(-1, self.ninp, 1)
        #wrong_dis = torch.bmm(tgt_wrong, src_feat)
        #batch_wrong_dis = torch.bmm(batch_wrong, src_feat)
        fake_dis = torch.bmm(fake.view(-1, 1, self.ninp), src_feat)
        right_dis = torch.bmm(tgt_right.view(-1, 1, self.ninp), src_feat)

        fake_score = torch.exp(right_dis - fake_dis)
        loss_fake = torch.sum(torch.log(fake_score + 1))

        loss_norm = src_feat.norm() + fake.norm() + tgt_right.norm()
        loss = (loss_fake + 0.1 * loss_norm) / batch_size

        # return loss, loss_fake.data[0]/batch_size
        return loss, loss_fake.item()/batch_size



def get_ngrams(input_tokens, max_n=4):
    n_grams=[]
    for n in range(1, max_n+1):
        n_grams.append(defaultdict(int))
        for n_gram in zip(*[input_tokens[i:] for i in range(n)]):
            n_grams[n-1][n_gram] +=1
    return n_grams


def bleu_score(ref_tokens, hypothesis_tokens, max_n=4):

    def product(iterable):
        return reduce(mul, iterable, 1)

    def n_gram_precision(ref_ngrams, hyp_ngrams):
        precision=[]
        for n in range(1, max_n + 1):
            overlap = 0
            for ref_ngram, ref_ngram_count in ref_ngrams[n-1].items():
                if ref_ngram in hyp_ngrams[n-1]:
                    overlap += min(ref_ngram_count, hyp_ngrams[n-1][ref_ngram])
            hyp_length = max(0, len(hypothesis_tokens)-n+1)
            if n >=2:
                overlap += 1
                hyp_length += 1
            precision.append(overlap/hyp_length if hyp_length > 0 else 0.0)
        return precision

    def brevity_penalty(ref_length, hyp_length):
        return min(1.0, exp(1-(ref_length/hyp_length if hyp_length > 0 else 0.0)))

    hypothesis_length = len(hypothesis_tokens)
    ref_length = len(ref_tokens)
    hypothesis_ngrams = get_ngrams(hypothesis_tokens)
    ref_ngrams = get_ngrams(ref_tokens)

    np = n_gram_precision(ref_ngrams, hypothesis_ngrams)
    bp = brevity_penalty(ref_length, hypothesis_length)

    return product(np)**(1 / max_n) * bp


class BLEU_score(nn.Module):

    def __init__(self):
        super(BLEU_score, self).__init__()
        self.sf = nltk.translate.bleu_score.SmoothingFunction()

    def forward(self, ref, hyp):
        # print("model BLEU_score ref", ref.size())
        # print("model BLEU_score hyp", hyp.size())
        # exit()
        ref = ref.t().tolist()
        # print("model BLEU_score ref", ref)
        hyp = hyp.t().tolist()
        # print("model BLEU_score hyp", hyp)
        score = 0.0
        for i in range(len(ref)):
            score += nltk.translate.bleu_score.sentence_bleu([ref[i]], hyp[i], smoothing_function=self.sf.method1, weights = [1])
            # = bscore
            # print("%d bscore %f" % (i, bscore))
        return score


class gumbel_sampler(nn.Module):
    def __init__(self):
        # raise
        super(gumbel_sampler, self).__init__()

    def forward(self, input, noise, temperature=0.5):
        # print("misc/model.py gumbel_sampler input", input.device)
        # print("misc/model.py gumbel_sampler noise", noise.device)
        eps = 1e-20

        noise.data.add_(eps).log_().neg_()
        noise.data.add_(eps).log_().neg_()
        # print("misc/model.py input", input, input.size())
        # print("misc/model.py noise", noise, noise.size())
        # print("misc/model.py input.device", input.device)
        # print("misc/model.py noise.device", noise.device)
        # print(temperature.device)
        y = (input + noise) / temperature
        # y = (input) / temperature


        y = F.softmax(y, dim=-1)

        # print("misc/model.py y1", y, y.size())

        max_val, max_idx = torch.max(y, y.dim()-1)
        y_hard = y == max_val.view(-1,1).expand_as(y)
        # print("misc/model.py y2", y, y.size())

        y = (y_hard.float() - y).detach() + y
        # print("misc/model.py y3", y, y.size())
        # print("misc/model.py y3", y[0][max_idx], y.size())
        # print("misc/model.py max_idx", max_idx, max_idx.size())

        # log_prob = input.gather(1, max_idx.view(-1,1)) # gather the logprobs at sampled positions
        # exit()

        return y, max_idx.view(1, -1)#, log_prob


class AxB(nn.Module):
    def __init__(self, nhid):
        super(AxB, self).__init__()
        self.nhid = nhid

    def forward(self, nhA, nhB):
        mat = torch.bmm(nhB.view(-1, 100, self.nhid), nhA.view(-1,self.nhid,1))
        return mat.view(-1,100)
