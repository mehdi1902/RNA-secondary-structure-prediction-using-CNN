from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Sampler
import torch
import numpy as np
from random import shuffle
from utils import *

def partition(sequences, structures, families, partition_size, stride):
    new_seqs = []
    new_strs = []
    new_fml = []
    for seq, st, fml in zip(sequences, structures, families):
        n = len(seq)
        if n<partition_size:
            new_seqs.append(seq)
            new_strs.append(st)
            new_fml.append(fml)
        else:
            for i in range(n//stride):
                a = min(i*stride, n-partition_size)
                b = min(i*stride+partition_size, n)
                new_seqs.append(seq[a:b])
                pairing = np.array(st[a:b])
                pairing -= a
                idx = np.where(pairing>=partition_size)
                pairing[idx] = idx
                idx = np.where(pairing<0)
                pairing[idx] = idx
                new_strs.append(pairing)
                new_fml.append(fml)
    return new_seqs, new_strs



def collate(list_of_samples, onehot=True):
    """
    In the output each sample is padded to the max size of the batch
    """
    # x = [sample[0] for sample in list_of_samples]
    # y = [sample[1] for sample in list_of_samples]
    # f = [sample[2] for sample in list_of_samples]
    x, y, f = list(zip(*list_of_samples))
    max_size = np.max([len(s) for s in x])
    x = [pad_matrix(create_matrix(s, onehot=onehot), max_size) for s in x]
    y = [pad_matrix(pairing_to_matrix(s), max_size) for s in y]
    return torch.stack(x), torch.stack(y), f


def collate_inverse(list_of_samples, onehot=True):
    """
    In the output each sample is padded to the max size of the batch
    """
    x, y, f = list(zip(*list_of_samples))
    max_size = np.max([len(s) for s in x])
    # x = [create_seq(s) for s in x]
    x = [pad_matrix(create_matrix(s, onehot=onehot), max_size) for s in x]
    y = [pad_matrix(pairing_to_matrix(s), max_size) for s in y]
    return torch.stack(y), torch.stack(x), f

    
class SequentialSampler(Sampler):
    r"""Samples elements sequentially, always in the same order.
    Arguments:
        data_source (Dataset): dataset to sample from
    """
    def __init__(self, data_source):
        self.data_source = data_source
        self.samples = {}
        self.lengths = []
        for i, sample in enumerate(self.data_source):
            l = len(sample[0]) // 1
            self.lengths.append(l)
            if l in self.samples:
                self.samples[l].append(i)
            else:
                self.samples[l] = [i]
        self.lengths = list(set(self.lengths))
#         self.lengths = {for i, sample in enumerate(self.data_source)}
#         self.lengths = [[i,len(sample[0])] for i, sample in enumerate(self.data_source)]

    def __iter__(self):
        lengths = [i for i in self.lengths]
        shuffle(lengths)
        samples = []
        for l in lengths:
            for idx in self.samples[l]:
                samples.append([idx, l])
#         lengths = [[i,(.05*np.random.randn()+1)*j] for (i, j) in self.lengths] # Noise
#         samples = sorted(samples, key=lambda sample: sample[1])
        return iter(samples)

    def __len__(self):
        return len(self.data_source)


class BatchSampler(Sampler):
    def __init__(self, sampler, max_size, max_batch_size, max_interval, max_length=2000):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
       
        self.sampler = sampler
        self.max_size = max_size
        self.max_batch_size = max_batch_size
        self.max_interval = max_interval
        self.max_length = max_length
        
    def __iter__(self):
        batch = []
        sum_len = 0
        last_len = 0
        for idx, l in self.sampler:
            if l > self.max_length:
                continue
            if not batch:
                batch.append(idx)
                last_len = l
            else:
                if last_len!=l or len(batch)>=self.max_batch_size or (len(batch)+1)*last_len*1>self.max_size:
                    yield batch
                    batch = [idx]
                    last_len = l
                else: # Same size
                    batch.append(idx)
        yield batch

    def __len__(self):
        print('Len is not correct at the moment')
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
    
    