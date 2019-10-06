#-*- coding:utf-8 –*-
import json
import h5py
import argparse
import numpy as np
from nltk.tokenize import word_tokenize
import nltk
from unidecode import unidecode
import os
import re
import pdb
import random 
import sys
import torch
from itertools import count


parser = argparse.ArgumentParser()

# Input files
parser.add_argument('-input_json', default='train.json', help='Input json file')
parser.add_argument('-output_h5', default='train.bpe.h5', help='Output hdf5 file')
# Options
# parser.add_argument('-vocab', default='/home/xwshi/easymoses_workspace2/reinforcement_learning/nmt/vi-en-baseline/corpus/vi_bpe.vocab.pt', help='path to dataset, now json file')


parser.add_argument('-vocab', default='/home/xwshi/easymoses_workspace2/reinforcement_learning/source_code/demo_data/nhfx/nhfx_bpe.vocab.pt', help='  ')
# parser.add_argument('-filename', default='/home/xwshi/easymoses_workspace2/reinforcement_learning/source_code/demo_data/nmpt.bpe', help='  ')
# parser.add_argument('-filename', default='nmpt.bpe', help='  ')


parser.add_argument('-max_src_len', default=50, type=int, help='Max length of sources')
parser.add_argument('-options_num', default=100, type=int, help='Max length of sources')
parser.add_argument('-max_tgt_len', default=50, type=int, help='Max length of target')
parser.add_argument('-test', action="store_true", help='')


args = parser.parse_args()


def load_from_vocab(vocabfile):
    """
    Load Field objects from `vocab.pt` file.
    """
    vocab = torch.load(vocabfile)
    vocab = dict(vocab)
    # print (len(vocab))
    print (len(vocab['src']), type(vocab['src']))
    print (len(vocab['tgt']), type(vocab['tgt']))
    return vocab


def tokenize_data(data):
    '''
    Tokenize captions, questions and answers
    Also maintain word count if required
    '''
    src_toks, tgt_toks = [], []

    # print 'Tokenizing questions...'
    for src in data['sources']:
        src_tok = src.strip().split()
        src_toks.append(src_tok)

    # print 'Tokenizing answers...'
    for tgt in data['targets']:
        tgt_tok = tgt.strip().split()
        tgt_toks.append(tgt_tok)

    return src_toks, tgt_toks


def encode_vocab(source, target, vocab):
    src_idx, tgt_idx = [], []
    src_stoi = vocab['src'].stoi
    # for k in src_stoi:
        # if k not in ['<unk>', '<blank>']:
            # src_stoi[k] += 30000
    tgt_stoi = vocab['tgt'].stoi
    # print("src_stoi['<unk>']", type(src_stoi['<unk>']), src_stoi['<unk>'])
    # exit()
    for txt in source:
        # print("txt", txt)
        ind = [src_stoi.get(w, src_stoi['<unk>']) for w in txt]
        # print("ind", ind)
        # exit()
        src_idx.append(ind)

    for txt in target:
        # print("txt", txt)
        ind = [tgt_stoi.get(w, tgt_stoi['<unk>']) for w in txt]
        # print("ind", ind)
        # exit()
        tgt_idx.append(ind)

    return src_idx, tgt_idx


def create_mats(data, vocab, src_idx, tgt_idx, params):
    # N = len(data['paras'])
    # data = data['data']
    options_number = params.options_num

    N = len(data['paras'])
    print ("data size", N)
    # exit()
    # num_round = 1
    max_src_len = params.max_src_len
    max_tgt_len = params.max_tgt_len

    sources = np.zeros([N, max_src_len], dtype='uint32')
    target = np.zeros([N, max_tgt_len], dtype='uint32')
    target_index = np.zeros([N],  dtype='uint32')

    source_len = np.zeros([N],  dtype='uint32')
    target_len = np.zeros([N], dtype='uint32')
    options = np.zeros([N, options_number], dtype='uint32')

    idx = 0
    for i, pa in enumerate(data['paras']):
        for j, para in enumerate(pa['para']):
            # print("idx", idx)
            # print(para)
            # print(para['source'])
            src_len = len(src_idx[para['source']][0:max_src_len])
            # exit()
            source_len[idx] = src_len
            sources[idx,:src_len] = src_idx[para['source']][0:src_len]
            if not params.test == "test":
                tgt_len = len(tgt_idx[para['target']][0:max_tgt_len])
                target_len[idx] = tgt_len
                target[idx,:tgt_len] = tgt_idx[para['target']][0:tgt_len]
                target_index[idx] = para['ref_index']
            options[idx][:len(para['target_options'])] = para['target_options']
            # print ("options", options[idx])
            # print ("target_index", target_index[idx])
            idx += 1

    print("data size", idx)

    options_list = np.zeros([len(tgt_idx), max_tgt_len], dtype='uint32')
    options_len = np.zeros(len(tgt_idx), dtype='uint32')

    for i, tgt in enumerate(tgt_idx):
        options_len[i] = len(tgt[0:max_tgt_len])
        options_list[i][0:options_len[i]] = tgt[0:max_tgt_len]

    return sources, source_len, target, target_len, options, options_list, options_len, target_index


if __name__ == "__main__":
    # print 'Reading json...'
    # data_train = json.load(open(args.input_json_train, 'r'))
    vocab = load_from_vocab(args.vocab)
    # exit()
    # src_lines = open(args.filename+'.'+args.src_id, 'r').readlines()
    # tgt_lines = open(args.filename+'.'+args.tgt_id, 'r').readlines()

    data = json.load(open(args.input_json, 'r'))
    split_type = data['split']
    print("split_type", split_type)
    # src_tok_train, tgt_tok_train = tokenize_data(data_train)
    src_tok, tgt_tok  = tokenize_data(data['data'])

    # print (src_tok[200])
    # print (tgt_tok[200])

    # print ("src_tok", len(src_tok))
    # print ("tgt_tok", len(tgt_tok))

    print ('Encoding based on vocabulary...')
    # src_idx_train, tgt_idx_train = encode_vocab(src_tok_train, tgt_tok_train, wtoi)
    src_idx, tgt_idx = encode_vocab(src_tok, tgt_tok, vocab)
    
    src, src_len, tgt, tgt_len, opt, opt_list, opt_len, tgt_index = create_mats(data['data'], vocab, src_idx, tgt_idx, args)
    # src, src_len, tgt, tgt_len, opt, opt_list, opt_len, tgt_index = create_mats(data, src_idx, tgt_idx, args)

    test_number = 50
    # print("src_tok", src_tok[test_number])
    # print("tgt_index", tgt_index[test_number])
    # print("opt", opt[test_number])
    # print("opt", opt[test_number][tgt_index[test_number]])
    # print(tgt_tok[opt[test_number][tgt_index[test_number]]])
    # print(tgt_idx[opt[test_number][tgt_index[test_number]]])
    # print(tgt_idx[test_number])
    # print(opt_list[test_number])
    print ('Saving hdf5...')
    f = h5py.File(args.output_h5, 'w')
    f.create_dataset('src', dtype='uint32', data=src)
    f.create_dataset('src_len', dtype='uint32', data=src_len)
    f.create_dataset('tgt', dtype='uint32', data=tgt)
    f.create_dataset('tgt_len', dtype='uint32', data=tgt_len)
    f.create_dataset('tgt_index', dtype='uint32', data=tgt_index)
    f.create_dataset('opt', dtype='uint32', data=opt)
    f.create_dataset('opt_len', dtype='uint32', data=opt_len)
    f.create_dataset('opt_list', dtype='uint32', data=opt_list)
    f.close()

    #data_toks, src_inds, tgt_inds = encode_vocab(data_toks, src_toks, tgt_toks, word2ind)
