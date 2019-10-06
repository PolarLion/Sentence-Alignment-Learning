import torch.utils.data as data
import torch
import numpy as np
import h5py
import json
import random

class load_data(data.Dataset): # torch wrapper
    def __init__(self, input_h5, input_vocab, num_val, split_type, negative_sample=20):

        print('DataLoader loading: %s' % split_type)
        vocab = torch.load(input_vocab)
        vocab = dict(vocab)
        if num_val <= 0:
            num_val = None
        # f = json.load(open(input_json, 'r'))
        self.src_stoi = vocab['src'].stoi
        self.tgt_stoi = vocab['tgt'].stoi
        self.src_itos = vocab['src'].itos
        self.tgt_itos = vocab['tgt'].itos

        print("self.src_itos[:2]", self.src_itos[:10])
        print("self.tgt_itos[:2]", self.tgt_itos[:10])

        print('Loading txt from %s' %input_h5)
        f = h5py.File(input_h5, 'r')


        self.src = f['src'][:num_val]
        print('%s number of data: %d' % (split_type, len(self.src)), self.src.shape)

        self.tgt = f['tgt'][:num_val]

        self.src_len = f['src_len'][:num_val]
        self.tgt_len = f['tgt_len'][:num_val]
        self.tgt_ids = f['tgt_index'][:num_val]
        # print("DataLoader", len(self.tgt_ids), self.tgt_ids[-1000:-1])
        self.opt_ids = f['opt'][:num_val]
        self.opt_list = f['opt_list'][:]
        self.opt_len = f['opt_len'][:]
        f.close()

        print(self.src.shape)
        self.src_length = self.src.shape[1]
        self.tgt_length = self.tgt.shape[1]
        self.src_vocab_size = len(self.src_itos)
        self.tgt_vocab_size = len(self.tgt_itos)

        # print('src vocab Size: %d' % self.src_vocab_size)
        # print('tgt vocab Size: %d' % self.tgt_vocab_size)
        self.split = split_type
        # self.rnd = 1
        self.negative_sample = negative_sample

        if not split_type == "train":
            self.negative_sample = 100
    
    def __getitem__(self, index):
        # print ("self.src_length", self.src_length)
        # print ("self.tgt_length", self.tgt_length)

        # src = np.zeros([self.src_length], dtype='int64')
        src = np.full([self.src_length], self.src_stoi['<blank>'], dtype='int64')
        # tgt = np.zeros([self.tgt_length+1], dtype='int64')
        tgt = np.full([self.tgt_length+1], self.tgt_stoi['<blank>'], dtype='int64')
        tgt_target = np.full([self.tgt_length+1], self.tgt_stoi['<blank>'], dtype='int64')
        # tgt_target = np.zeros((self.tgt_length+1), dtype='int64')
        # src_ori = np.zeros((self.src_length), dtype='int64')
        src_ori = np.full((self.src_length), self.src_stoi['<blank>'], dtype='int64')

        # if self.split == "train":
        # opt_tgt = np.zeros((self.negative_sample, self.tgt_length+1), dtype='int64')
        # else:
        opt_tgt = np.full((self.negative_sample, self.tgt_length+1), self.tgt_stoi['<blank>'], dtype='int64')
        # opt_tgt = np.full((self.negative_sample, self.tgt_length+1), self.tgt_stoi['<pad>'], dtype='int64')
        # opt_tgt = np.full((self.negative_sample, self.tgt_length+1), self.tgt_stoi['</s>'], dtype='int64')

        # opt_tgt_target = np.zeros((self.negative_sample, self.tgt_length+1), dtype='int64')
        opt_tgt_target = np.full((self.negative_sample, self.tgt_length+1), self.tgt_stoi['<blank>'], dtype='int64')


        tgt_len = np.zeros((1), dtype='int64')
        src_len = np.zeros((1), dtype='int64')
        opt_tgt_len = np.zeros((self.negative_sample), dtype='int64')

        tgt_idx = np.zeros((1),dtype='int64')
        tgt_ids = np.zeros((1),dtype='int64')
        opt_tgt_idx = np.zeros((self.negative_sample), dtype='int64')

        # for i in range(self.rnd):
        # get the index
        # print("index", index)
        s_len = self.src_len[index]
        # print("s_len", s_len)
        t_len = self.tgt_len[index]
        # print("t_len", t_len)
        # qt_len = s_len + t_len

        # if i+1 < self.rnd:
        #     print ("if i+1 < self.rnd:")
        
        """把内容放右端"""
        src[self.src_length-s_len:] = self.src[index, :s_len]
        """把内容放左端"""
        src_ori[:s_len] = self.src[index, :s_len]
        
        tgt[1:t_len+1] = self.tgt[index, :t_len]
        # print("DataLoader.py", self.tgt[index, :t_len])
        # exit()
        tgt[0] = self.tgt_stoi['<s>']

        if t_len < self.tgt_length:
            tgt[t_len+1] = self.tgt_stoi['</s>']
        # else:
        #     tgt[t_len] = self.tgt_stoi['</s>']


        tgt_target[:t_len] = self.tgt[index, :t_len]
        tgt_target[t_len] = self.tgt_stoi['</s>']

        tgt_len[0] = self.tgt_len[index]
        src_len[0] = self.src_len[index]

        opt_ids = self.opt_ids[index] # since python start from 0
        # print("opt_ids", opt_ids)
        # random select the negative samples.
        tgt_idx[0] = opt_ids[self.tgt_ids[index]]
        # print("load_data self.tgt_ids[index] index", index, self.tgt_ids[index])
        # print("load_data, tgt_idx", tgt_idx)
        tgt_ids[0] = self.tgt_ids[index]
        # print("tgt_ids", tgt_idx)
        # exclude the gt index.
        # opt_ids = np.delete(opt_ids, tgt_idx, 0)
        if self.split == "train":
            # print ("train")
            opt_ids = np.delete(opt_ids, self.tgt_ids[index], 0)
            # opt_ids = np.delete(opt_ids, tgt_idx[0], 0)

            # print("opt_ids", opt_ids)
            # print("self.negative_sample", self.negative_sample)
            random.shuffle(opt_ids)

            for j in range(self.negative_sample):
                ids = opt_ids[j]
                # print("j ids", ids)
                opt_tgt_idx[j] = ids

                opt_len = self.opt_len[ids]
                # print("DataLoader opt_len", opt_len)
                opt_tgt_len[j] = opt_len
                opt_tgt[j, 1:opt_len+1] = self.opt_list[ids,:opt_len]
                opt_tgt[j, 0] = self.tgt_stoi['<s>']
                if opt_len < self.tgt_length:
                    opt_tgt[j, opt_len+1] = self.tgt_stoi['</s>']
                # else:
                #     opt_tgt[j, opt_len] = self.tgt_stoi['</s>']

                opt_tgt_target[j, :opt_len] = self.opt_list[ids,:opt_len]
                opt_tgt_target[j, opt_len] = self.tgt_stoi['</s>']
        
        else:
            # print ("opt_ids", len(opt_ids))
            for j, ids in enumerate(opt_ids):
                opt_len = self.opt_len[ids]
                # print("DataLoader opt_len", opt_len)
                opt_tgt[j, 1:opt_len+1] = self.opt_list[ids,:opt_len]
                opt_tgt[j, 0] = self.tgt_stoi['<s>']
                if opt_len < self.tgt_length:
                    opt_tgt[j, opt_len+1] = self.tgt_stoi['</s>']
                # else:
                #     opt_tgt[j, opt_len] = self.tgt_stoi['</s>']
                if opt_len > 50:
                    print("DataLoader.py opt_len", opt_len)
                    exit()
                opt_tgt_target[j, :opt_len] = self.opt_list[ids,:opt_len]
                opt_tgt_target[j, opt_len] = self.tgt_stoi['</s>']

                opt_tgt_idx[j] = opt_ids[j]
                opt_tgt_len[j] = opt_len

        src = torch.from_numpy(src)
        tgt = torch.from_numpy(tgt)
        tgt_target = torch.from_numpy(tgt_target)
        src_ori = torch.from_numpy(src_ori)
        tgt_len = torch.from_numpy(tgt_len)
        src_len = torch.from_numpy(src_len)
        opt_tgt_len = torch.from_numpy(opt_tgt_len)
        opt_tgt = torch.from_numpy(opt_tgt)
        opt_tgt_target = torch.from_numpy(opt_tgt_target)
        tgt_idx = torch.from_numpy(tgt_idx)
        tgt_ids = torch.from_numpy(tgt_ids)
        opt_tgt_idx = torch.from_numpy(opt_tgt_idx)

        # print src
        # print tgt
        if not self.split == "train":
            # return src, tgt, tgt_target, tgt_len, tgt_idx, src_ori, opt_tgt, opt_tgt_target, opt_tgt_len
            return src_len, src, tgt, tgt_target, tgt_ids, src_ori, opt_tgt, opt_tgt_target, opt_tgt_len, opt_tgt_idx
            
        return src_len, src, tgt, tgt_target, tgt_len, tgt_idx, src_ori, opt_tgt, opt_tgt_target, opt_tgt_len, opt_tgt_idx, tgt_ids

    def __len__(self):
        return self.src.shape[0]
