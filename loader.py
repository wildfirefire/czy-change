import json
import pickle
import random
import torch
import numpy as np
from tree import Tree, head_to_tree, tree_to_adj

class DataLoader(object):
    def __init__(self, filename,graph, batch_size, args, dicts):
        self.batch_size = batch_size
        self.args = args
        self.dicts = dicts

        with open(filename) as infile:
            data = json.load(infile)

            # 传入情感图
        fin = open(graph, 'rb')
        idx2graph_sdat = pickle.load(fin)
        fin.close()
        # preprocess data
        data = self.preprocess(data, dicts, args,idx2graph_sdat)

        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        print("{} batches created for {}".format(len(data), filename))

    def preprocess(self, data, dicts, args,graph):
        
        processed = []
        graph_id = 0
        for d in data:
            for aspect in d['aspects']:
                # word token
                tok = list(d['token'])
                if args.lower == True:
                    tok = [t.lower() for t in tok]
                
                asp = list(aspect['term']) # aspect
                label = aspect['polarity'] # label 
                pos = list(d['pos'])       # pos
                head = list(d['head'])     # head
                dep = list(d['deprel'])    #deprel
                length = len(tok)          # real length
                fro = aspect['from']
                to = aspect['to']
                # position
                post = [i-aspect['from'] for i in range(aspect['from'])] \
                       + [0 for _ in range(aspect['from'], aspect['to'])] \
                       + [i-aspect['to']+1 for i in range(aspect['to'], length)]
                # mask of aspect
                if len(asp) == 0:
                    mask = [1 for _ in range(length)]    # for rest16
                else:
                    mask = [0 for _ in range(aspect['from'])] \
                       +[1 for _ in range(aspect['from'], aspect['to'])] \
                       +[0 for _ in range(aspect['to'], length)]

                # map to ids 
                tok = map_to_ids(tok, dicts['token'])
                asp = map_to_ids(asp, dicts['token'])
                label = dicts['polarity'][label]
                pos = map_to_ids(pos, dicts['pos'])
                dep = map_to_ids(dep, dicts['dep'])
                head = [int(x) for x in head]
                assert any([x == 0 for x in head])
                post = map_to_ids(post, dicts['post'])
                assert len(tok) == length \
                       and len(pos) == length \
                       and len(head) == length \
                       and len(post) == length \
                       and len(mask) == length

                # 矩阵填充
                sdat_graph = graph[graph_id]
                graph_id += 3

                processed += [(tok, asp, pos, head, dep, post, mask, length,sdat_graph,fro,to, label)]

        return processed

    def gold(self):
        return self.labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError

        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        
        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)

        # convert to tensors
        tok = get_long_tensor(batch[0], batch_size).cuda()
        asp = get_long_tensor(batch[1], batch_size).cuda()
        pos = get_long_tensor(batch[2], batch_size).cuda()
        head = get_long_tensor(batch[3], batch_size).cuda()
        dep = get_long_tensor(batch[4], batch_size).cuda()
        post = get_long_tensor(batch[5], batch_size).cuda()
        mask = get_float_tensor(batch[6], batch_size).cuda()
        length = torch.LongTensor(batch[7]).cuda()
        graph = batch[8]
        fro = torch.LongTensor(batch[9]).cuda()
        to = torch.LongTensor(batch[10]).cuda()
        label = torch.LongTensor(batch[11]).cuda()

        def inputs_to_tree_reps(maxlen, head, words, l):
            trees = [head_to_tree(head[i], words[i], l[i]) for i in range(l.size(0))]
            adj = [tree_to_adj(maxlen, tree, directed=self.args.direct, self_loop=self.args.loop).reshape(1, maxlen, maxlen) for tree in trees]
            adj = np.concatenate(adj, axis=0)
            return adj

        maxlen = max(length)
        adj = torch.tensor(inputs_to_tree_reps(maxlen, head, tok, length)).cuda()

        # 初始化填充后的图矩阵，默认用0填充
        padded_graphs = torch.zeros((batch_size, maxlen, maxlen), device='cuda')

        for i in range(batch_size):
            # 每个样本的原始图矩阵
            single_graph = graph[i]
            # 获取原始图的尺寸（假设为方阵）
            orig_size = single_graph.shape[0]
            # 只填充到maxlen，超过部分截断（如果有）
            fill_size = min(orig_size, maxlen)
            # 将原始图填充到padded_graphs中
            padded_graphs[i, :fill_size, :fill_size] = torch.tensor(single_graph[:fill_size, :fill_size],
                                                                    device='cuda')

        return (tok, asp, pos, head, dep, post, mask, length, adj,padded_graphs, fro,to, label)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else 1 for t in tokens] # the id of [UNK] is ``1''
    return ids

def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(0)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens

def get_float_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded FloatTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.FloatTensor(batch_size, token_len).fill_(0)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.FloatTensor(s)
    return tokens

def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]
